"""Replay fixed NLI contract pairs against one pinned checkpoint on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_name(value: object) -> str:
    return str(value).casefold().replace("_", " ").strip()


def label_mapping(config) -> dict[int, str]:
    mapping = {int(key): _label_name(value) for key, value in config.id2label.items()}
    allowed = {"entailment", "neutral", "contradiction"}
    if set(mapping.values()) != allowed:
        raise ValueError(f"Unverified NLI label mapping: {mapping!r}")
    return mapping


def judgement(positive: str, opposite: str, category: str) -> tuple[str, bool]:
    """Apply the fixed pre-inference gate rule, separate from raw labels."""

    if category == "supported":
        decision = "SUPPORTED" if positive == "entailment" and opposite != "entailment" else "BLOCKED"
        return decision, decision == "SUPPORTED"
    if category == "contradicted":
        decision = "CONTRADICTED" if positive == "contradiction" or opposite == "entailment" else "FALSE_ACCEPT"
        return decision, decision == "CONTRADICTED"
    if category in {"unresolved_value_relation", "scope_uncertainty"}:
        decision = "UNRESOLVED" if positive != "entailment" and opposite != "entailment" else "FALSE_ACCEPT"
        return decision, decision == "UNRESOLVED"
    return "SKIPPED_NOT_ELIGIBLE", True


def _rss_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


def run(args) -> dict[str, Any]:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if fixtures.get("origin") != "synthetic_ai_authored_nli_contract_pairs":
        raise ValueError("fixture provenance must remain explicit")
    started = time.perf_counter()
    snapshot = Path(
        snapshot_download(
            args.model,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
            local_files_only=True,
        )
    )
    rss_before = _rss_bytes()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=args.use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32)
    model.to("cpu").eval()
    labels = label_mapping(model.config)
    rss_after_load = _rss_bytes()
    rows = []
    for pair in fixtures["pairs"]:
        if not pair["eligible"]:
            rows.append({**pair, "decision": "SKIPPED_NOT_ELIGIBLE", "correct": True, "reason": "deterministic_gate"})
            continue
        inputs = tokenizer(
            [pair["premise"], pair["premise"]],
            [pair["hypothesis"], pair["opposite_hypothesis"]],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        length = int(inputs["input_ids"].shape[1])
        if length > int(model.config.max_position_embeddings):
            rows.append({**pair, "decision": "UNRESOLVED_CONTEXT_TOO_LONG", "correct": False, "tokens": length})
            continue
        pair_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits.detach().cpu().float().tolist()
        predicted = [labels[int(max(range(len(logit)), key=logit.__getitem__))] for logit in logits]
        decision, correct = judgement(predicted[0], predicted[1], pair["category"])
        rows.append(
            {
                **pair,
                "tokens": length,
                "positive_label": predicted[0],
                "opposite_label": predicted[1],
                "positive_logits": logits[0],
                "opposite_logits": logits[1],
                "decision": decision,
                "correct": correct,
                "latency_ms": round((time.perf_counter() - pair_started) * 1000, 3),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    report = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "snapshot": str(snapshot),
        "fixtures": {"path": str(args.fixtures), "sha256": sha256(args.fixtures), "origin": fixtures["origin"]},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "use_fast": args.use_fast,
            "label_mapping": labels,
            "rss_bytes_before_load": rss_before,
            "rss_bytes_after_load": rss_after_load,
        },
        "summary": {
            "pairs": len(rows),
            "eligible_pairs": len(eligible),
            "correct_eligible": sum(row["correct"] for row in eligible),
            "false_blocks": sum(row["category"] == "supported" and not row["correct"] for row in eligible),
            "false_accepts": sum(row["decision"] == "FALSE_ACCEPT" for row in eligible),
            "correct_unresolved": sum(
                row["category"] in {"unresolved_value_relation", "scope_uncertainty"} and row["correct"] for row in eligible
            ),
            "decisions": dict(Counter(row["decision"] for row in rows)),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "rows": rows,
    }
    report["report_sha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"model": args.model, "revision": args.revision, **report["summary"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
