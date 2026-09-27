"""Train a small, local sentence-pair classifier for slot evidence.

Input rows describe the user's utterance (premise) and one proposed condition
(hypothesis). This is a research harness, not a production resolver. It neither
loads evaluation holdouts nor changes the default workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("support", "contradiction", "unknown")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MODEL_ID = "cointegrated/rubert-tiny2"
MODEL_REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
MODEL_WEIGHTS_SHA256 = "26ebb6db2a68593c54c74902d7a74f332da66297693f965cc9f1b0af4abf3894"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        for key in ("id", "premise", "hypothesis", "label"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"missing non-empty {key} at {path}:{line_number}")
        if raw["label"] not in LABEL_TO_ID:
            raise ValueError(f"unknown label at {path}:{line_number}: {raw['label']}")
        if raw["id"] in ids:
            raise ValueError(f"duplicate id in {path}: {raw['id']}")
        ids.add(raw["id"])
        if "family" in raw and (not isinstance(raw["family"], str) or not raw["family"].strip()):
            raise ValueError(f"family must be a non-empty string at {path}:{line_number}")
        row = {key: raw[key].strip() for key in ("id", "premise", "hypothesis", "label")}
        if "family" in raw:
            row["family"] = raw["family"].strip()
        rows.append(row)
    if not rows:
        raise ValueError(f"empty split: {path}")
    return rows


def validate_splits(train: list[dict[str, str]], dev: list[dict[str, str]]) -> dict[str, Any]:
    missing = set(LABELS) - {row["label"] for row in train}
    if missing:
        raise ValueError(f"train split misses labels: {sorted(missing)}")
    shared_ids = {row["id"] for row in train} & {row["id"] for row in dev}
    if shared_ids:
        raise ValueError(f"train/dev id overlap: {sorted(shared_ids)[:5]}")
    shared_premises = {_norm(row["premise"]) for row in train} & {_norm(row["premise"]) for row in dev}
    if shared_premises:
        raise ValueError(f"train/dev premise overlap: {sorted(shared_premises)[:5]}")
    shared_pairs = {
        (_norm(row["premise"]), _norm(row["hypothesis"])) for row in train
    } & {(_norm(row["premise"]), _norm(row["hypothesis"])) for row in dev}
    if shared_pairs:
        raise ValueError("train/dev normalized pair overlap")

    has_family = all("family" in row for row in train + dev)
    if not has_family and any("family" in row for row in train + dev):
        raise ValueError("family must be present on every train/dev row or on none")
    if has_family:
        shared_families = {row["family"] for row in train} & {row["family"] for row in dev}
        if shared_families:
            raise ValueError(f"train/dev family overlap: {sorted(shared_families)[:5]}")
    return {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_labels": dict(sorted(Counter(row["label"] for row in train).items())),
        "dev_labels": dict(sorted(Counter(row["label"] for row in dev).items())),
        "exact_id_overlap": 0,
        "normalized_premise_overlap": 0,
        "normalized_pair_overlap": 0,
        "family_check": "passed" if has_family else "unavailable_no_family_on_every_row",
    }


def _div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    if not gold or len(gold) != len(predicted):
        raise ValueError("gold and predictions must have the same non-zero length")
    if (set(gold) | set(predicted)) - set(LABELS):
        raise ValueError("metrics received an unknown label")
    matrix = {actual: dict.fromkeys(LABELS, 0) for actual in LABELS}
    for actual, guess in zip(gold, predicted, strict=True):
        matrix[actual][guess] += 1
    by_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in LABELS)
        precision = _div(tp, predicted_count)
        recall = _div(tp, support)
        by_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": _div(2 * precision * recall, precision + recall),
        }
    return {
        "rows": len(gold),
        "accuracy": _div(sum(actual == guess for actual, guess in zip(gold, predicted, strict=True)), len(gold)),
        "macro_f1": statistics.mean(item["f1"] for item in by_class.values()),
        "per_class": by_class,
        "confusion_actual_by_predicted": matrix,
    }


def majority_baseline(train: list[dict[str, str]], dev: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in train)
    majority = max(LABELS, key=lambda label: (counts[label], -LABEL_TO_ID[label]))
    return {"label": majority, **classification_metrics([row["label"] for row in dev], [majority] * len(dev))}


def validate_snapshot(snapshot: Path) -> dict[str, str]:
    if not snapshot.is_dir() or snapshot.name != MODEL_REVISION:
        raise ValueError(f"--snapshot must point to the pinned {MODEL_ID} revision {MODEL_REVISION}")
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"incomplete local model snapshot: {missing}")
    hashes = {name: sha256_file(snapshot / name) for name in required}
    if hashes["model.safetensors"] != MODEL_WEIGHTS_SHA256:
        raise ValueError("pinned model weights SHA-256 mismatch")
    return hashes


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires values and fraction in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.train.resolve()
    dev_path = args.dev.resolve()
    if train_path == dev_path:
        raise ValueError("train and dev paths must differ")
    train_hash = sha256_file(train_path)
    dev_hash = sha256_file(dev_path)
    if args.expected_train_sha256 and train_hash != args.expected_train_sha256.lower():
        raise ValueError("train SHA-256 differs from frozen protocol")
    if args.expected_dev_sha256 and dev_hash != args.expected_dev_sha256.lower():
        raise ValueError("dev SHA-256 differs from frozen protocol")
    train = read_rows(train_path)
    dev = read_rows(dev_path)
    split_report = validate_splits(train, dev)
    snapshot = args.snapshot.resolve()
    model_hashes = validate_snapshot(snapshot)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.max_length < 8 or args.latency_samples < 1:
        raise ValueError("epochs, batch-size, learning-rate, max-length and latency-samples must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")

    # Set CUDA determinism before initializing a CUDA context.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected Python environment")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True
    )

    class PairClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(encoder.config.hidden_size, len(LABELS))

        def forward(self, batch: dict[str, Any]) -> Any:
            features = self.encoder(**batch).last_hidden_state[:, 0]
            return self.classifier(self.dropout(features))

    model = PairClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def token_batch(rows: list[dict[str, str]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["premise"] for row in rows],
            [row["hypothesis"] for row in rows],
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return {key: tensor.to(device) for key, tensor in encoded.items()}

    def predict(rows: list[dict[str, str]]) -> list[str]:
        model.eval()
        guesses: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                logits = model(token_batch(rows[start : start + args.batch_size]))
                guesses.extend(LABELS[index] for index in logits.argmax(dim=-1).tolist())
        return guesses

    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_macro_f1 = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch_rows = [train[index] for index in order[start : start + args.batch_size]]
            labels = torch.tensor([LABEL_TO_ID[row["label"]] for row in batch_rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_batch(batch_rows))
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_rows)
        dev_metrics = classification_metrics([row["label"] for row in dev], predict(dev))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "dev": dev_metrics})
        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, output / "best.safetensors")

    model.load_state_dict(load_file(output / "best.safetensors", device=str(device)))
    predictions = predict(dev)
    final_metrics = classification_metrics([row["label"] for row in dev], predictions)
    baseline = majority_baseline(train, dev)
    sample_rows = dev[: min(len(dev), args.latency_samples)]
    predict(sample_rows[:1])  # warm up before timing
    timings = []
    model.eval()
    with torch.inference_mode():
        for row in sample_rows:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(token_batch([row]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)

    report = {
        "status": "trained_experimental_not_default",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(snapshot), "files_sha256": model_hashes},
        "input": {
            "train": str(train_path), "train_sha256": train_hash,
            "dev": str(dev_path), "dev_sha256": dev_hash, "split_integrity": split_report,
        },
        "run": {
            "script_sha256": sha256_file(Path(__file__)), "seed": args.seed, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "max_length": args.max_length, "device": args.device, "best_epoch": best_epoch,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "majority_baseline": baseline,
        "dev_metrics": final_metrics,
        "dev_metrics_note": "Dev was used to choose the best epoch; these metrics are not an independent final test.",
        "history": history,
        "latency_single_pair_ms": {
            "warmup_excluded": True, "sample_count": len(timings),
            "p50": percentile(timings, 0.5), "p95": percentile(timings, 0.95),
        },
        "checkpoint_sha256": sha256_file(output / "best.safetensors"),
        "predictions": [{"id": row["id"], "gold": row["label"], "predicted": guess} for row, guess in zip(dev, predictions, strict=True)],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path, help="Local pinned rubert-tiny2 snapshot; no downloads")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-train-sha256", help="Refuse if the frozen train file changed")
    parser.add_argument("--expected-dev-sha256", help="Refuse if the frozen dev file changed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--latency-samples", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"slot evidence experiment failed: {exc}\n")
    print(json.dumps({key: report[key] for key in ("status", "dev_metrics", "majority_baseline", "latency_single_pair_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
