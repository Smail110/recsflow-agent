"""Read-only OOD diagnostic for the trained experimental slot classifiers.

Uses a fixed AI-authored role/claim fixture. The fixture is neither training
data nor an independent final test. No thresholds, prompts or labels are tuned.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts import train_slot_evidence_base as base
from scripts import train_slot_evidence_tiny as common

GOLD_TO_TRAIN = {"entailed": "support", "contradicted": "contradiction", "unknown": "unknown"}
CLAIM_ROLES = ("target", "opposite", "competing")


def flatten_fixture(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("fixture must have a non-empty cases list")
    rows = []
    seen: set[str] = set()
    for case in payload["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not isinstance(case.get("utterance"), str):
            raise ValueError("fixture case needs id and utterance")
        if not case["utterance"].strip() or case["id"] in seen:
            raise ValueError("fixture case id must be unique and utterance non-empty")
        seen.add(case["id"])
        for role in CLAIM_ROLES:
            claim = case.get(role)
            if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str) or not claim["claim"].strip():
                raise ValueError(f"{case['id']} missing {role} claim")
            if claim.get("gold") not in GOLD_TO_TRAIN:
                raise ValueError(f"{case['id']} has unknown {role} gold")
            rows.append(
                {
                    "id": case["id"],
                    "family": str(case.get("family", "unknown")),
                    "role": role,
                    "premise": case["utterance"],
                    "hypothesis": claim["claim"],
                    "fixture_gold": claim["gold"],
                    "label": GOLD_TO_TRAIN[claim["gold"]],
                }
            )
    return rows


def summarize(rows: list[dict[str, str]], predictions: list[str]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("prediction count mismatch")
    metrics = common.classification_metrics([row["label"] for row in rows], predictions)
    false_support = [
        {"id": row["id"], "role": row["role"], "gold": row["label"], "claim": row["hypothesis"]}
        for row, guess in zip(rows, predictions, strict=True)
        if guess == "support" and row["label"] != "support"
    ]
    by_role = {
        role: common.classification_metrics(
            [row["label"] for row in rows if row["role"] == role],
            [guess for row, guess in zip(rows, predictions, strict=True) if row["role"] == role],
        )
        for role in CLAIM_ROLES
    }
    return {"metrics": metrics, "by_role": by_role, "false_support_count": len(false_support), "false_support": false_support}


def validate_training_report(kind: str, report: dict[str, Any], checkpoint: Path, snapshot: Path) -> dict[str, str]:
    if report.get("status") != "trained_experimental_not_default":
        raise ValueError("training report status is not completed experimental training")
    if report.get("checkpoint_sha256") != common.sha256_file(checkpoint):
        raise ValueError("trained checkpoint SHA-256 differs from training report")
    model = report.get("model", {})
    if kind == "tiny":
        if model.get("id") != common.MODEL_ID or model.get("revision") != common.MODEL_REVISION:
            raise ValueError("tiny training report model identity mismatch")
        hashes = common.validate_snapshot(snapshot)
    else:
        if model.get("id") != base.MODEL_ID or model.get("metadata_revision") != base.METADATA_REVISION:
            raise ValueError("base training report model identity mismatch")
        if not snapshot.is_dir() or snapshot.name != base.METADATA_REVISION:
            raise ValueError("base snapshot revision mismatch")
        files = ("config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
        hashes = {name: common.sha256_file(snapshot / name) for name in files}
        if hashes["config.json"] != base.CONFIG_SHA256:
            raise ValueError("base config SHA-256 mismatch")
    if any(report["model"]["files_sha256"].get(name) != value for name, value in hashes.items()):
        raise ValueError("local tokenizer/config differs from model used for training")
    return hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = args.fixture.resolve()
    trained_dir = args.trained_dir.resolve()
    checkpoint = trained_dir / "best.safetensors"
    training_report_path = trained_dir / "report.json"
    snapshot = args.snapshot.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    rows = flatten_fixture(fixture)
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    snapshot_hashes = validate_training_report(args.kind, training_report, checkpoint, snapshot)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    tokenizer = (
        AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        if args.kind == "tiny"
        else BertTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    )

    class PairClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(config.hidden_size, len(common.LABELS))

        def forward(self, batch: dict[str, Any]) -> Any:
            return self.classifier(self.dropout(self.encoder(**batch).last_hidden_state[:, 0]))

    model = PairClassifier()
    expected = set(model.state_dict())
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        actual = set(source.keys())
        if actual != expected:
            raise ValueError(f"trained state mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        state = {key: source.get_tensor(key) for key in expected}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    def predict_one(row: dict[str, str]) -> str:
        batch = tokenizer(
            row["premise"], row["hypothesis"], truncation=True, padding=True,
            max_length=training_report["run"]["max_length"], return_tensors="pt",
        )
        tensors = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            return common.LABELS[model(tensors).argmax(dim=-1).item()]

    predict_one(rows[0])  # warm up
    predictions = []
    timings = []
    for row in rows:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        guess = predict_one(row)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        predictions.append(guess)
        timings.append((time.perf_counter() - started) * 1000)
    summary = summarize(rows, predictions)
    result = {
        "status": "read_only_ood_diagnostic_not_final",
        "warning": "AI-authored fixture is not independent final evaluation; no model or threshold tuning was done.",
        "kind": args.kind,
        "model": training_report["model"]["id"],
        "fixture": str(fixture), "fixture_sha256": common.sha256_file(fixture),
        "training_report": str(training_report_path), "training_report_sha256": common.sha256_file(training_report_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": common.sha256_file(checkpoint),
        "snapshot": str(snapshot), "snapshot_files_sha256": snapshot_hashes,
        "script_sha256": common.sha256_file(Path(__file__)),
        "device": args.device, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "latency_single_pair_ms": {
            "warmup_excluded": True, "sample_count": len(timings),
            "p50": common.percentile(timings, 0.5), "p95": common.percentile(timings, 0.95),
        },
        **summary,
        "predictions": [
            {"id": row["id"], "family": row["family"], "role": row["role"],
             "utterance": row["premise"], "claim": row["hypothesis"],
             "gold": row["label"], "predicted": guess, "latency_ms": elapsed}
            for row, guess, elapsed in zip(rows, predictions, timings, strict=True)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("tiny", "base"))
    parser.add_argument("--trained-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"slot evidence probe failed: {exc}\n")
    print(json.dumps({key: result[key] for key in ("kind", "metrics", "false_support_count", "latency_single_pair_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
