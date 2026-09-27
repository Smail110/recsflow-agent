"""Evaluate pinned NLI adapters on a separate citation-control cohort.

The MASSIVE annotations identify slot spans, not NLI labels. The wrong-span
control is deterministic silver data and cannot establish product quality.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.train_context_nli import ID_TO_LABEL, MAX_LENGTH, sha256_file, validate_snapshot


def read_pairs(path: Path, expected_hash: str) -> list[dict[str, Any]]:
    if any("blind" in part.casefold() or "holdout" in part.casefold() for part in path.parts):
        raise ValueError("sealed data are prohibited")
    if sha256_file(path) != expected_hash.lower():
        raise ValueError("evaluation dataset SHA-256 differs from frozen protocol")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("label") not in {"support", "unknown"} or not isinstance(row.get("group_id"), str):
            raise ValueError("invalid evaluation row")
        groups[row["group_id"]].append(row)
    for name, pair in groups.items():
        if len(pair) != 2 or {row["label"] for row in pair} != {"support", "unknown"}:
            raise ValueError(f"invalid matched pair: {name}")
        if len({(row["message"], row["hypothesis"]) for row in pair}) != 1:
            raise ValueError(f"pair changes message or hypothesis: {name}")
        if pair[0]["premise"] == pair[1]["premise"]:
            raise ValueError(f"pair does not change selected citation: {name}")
    if not groups:
        raise ValueError("empty evaluation cohort")
    return rows


def paired_metrics(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if len(rows) != len(labels):
        raise ValueError("prediction count differs from rows")
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for row, label in zip(rows, labels, strict=True):
        groups[row["group_id"]][row["label"]] = label
    count = len(groups)
    primary_support = sum(pair["support"] == "support" for pair in groups.values())
    wrong_support = sum(pair["unknown"] == "support" for pair in groups.values())
    both_correct = sum(
        pair["support"] == "support" and pair["unknown"] == "unknown" for pair in groups.values()
    )
    changed = sum(pair["support"] != pair["unknown"] for pair in groups.values())
    return {
        "groups": count,
        "support_recall": primary_support / count,
        "wrong_citation_false_support_rate": wrong_support / count,
        "both_exact_rate": both_correct / count,
        "prediction_flip_rate": changed / count,
        "counts": {
            "primary_support": primary_support,
            "wrong_citation_support": wrong_support,
            "both_exact": both_correct,
            "changed": changed,
        },
    }


def cluster_intervals(rows: list[dict[str, Any]], labels: list[str], *, seed: int = 42, samples: int = 1000) -> dict[str, list[float]]:
    """Percentile intervals resample source utterances, not derived pairs."""
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[row["source_id"]].append(index)
    sources = sorted(by_source)
    rng = random.Random(seed)
    keys = ("support_recall", "wrong_citation_false_support_rate", "both_exact_rate", "prediction_flip_rate")
    values: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(samples):
        selected = [sources[rng.randrange(len(sources))] for _ in sources]
        sampled_rows = []
        sampled_labels = []
        for draw, source in enumerate(selected):
            for index in by_source[source]:
                sampled_rows.append({**rows[index], "group_id": f"{draw}:{rows[index]['group_id']}"})
                sampled_labels.append(labels[index])
        metrics = paired_metrics(sampled_rows, sampled_labels)
        for key in keys:
            values[key].append(metrics[key])
    return {
        key: [sorted(items)[int(0.025 * samples)], sorted(items)[int(0.975 * samples) - 1]]
        for key, items in values.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_snapshot(args.snapshot)
    rows = read_pairs(args.dataset, args.expected_dataset_sha256)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True, trust_remote_code=False)

    def infer(adapter: Path | None) -> tuple[list[str], float]:
        base = AutoModelForSequenceClassification.from_pretrained(
            str(args.snapshot), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32,
        ).to(device)
        model = base if adapter is None else PeftModel.from_pretrained(base, str(adapter), is_trainable=False).to(device)
        model.eval()
        output: list[str] = []
        start = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(rows), 16):
                batch = rows[offset : offset + 16]
                encoded = tokenizer(
                    [row["premise"] for row in batch], [row["hypothesis"] for row in batch],
                    truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
                ).to(device)
                output.extend(ID_TO_LABEL[index] for index in model(**encoded).logits.argmax(dim=-1).tolist())
        elapsed = time.perf_counter() - start
        del model, base
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return output, elapsed

    results = {}
    predictions = {}
    for name, adapter in (("zero_shot", None), ("previous_unweighted", args.adapter_before), ("paired_margin", args.adapter_after)):
        labels, elapsed = infer(adapter)
        predictions[name] = labels
        results[name] = {
            **paired_metrics(rows, labels),
            "cluster_bootstrap_95pct": cluster_intervals(rows, labels),
            "inference_seconds": elapsed,
        }
    report = {
        "status": "component_silver_only",
        "data": {"path": str(args.dataset), "sha256": sha256_file(args.dataset), "rows": len(rows)},
        "models": {
            "snapshot": str(args.snapshot),
            "previous_adapter": str(args.adapter_before),
            "previous_adapter_sha256": sha256_file(args.adapter_before / "adapter_model.safetensors"),
            "paired_adapter": str(args.adapter_after),
            "paired_adapter_sha256": sha256_file(args.adapter_after / "adapter_model.safetensors"),
        },
        "metrics": results,
        "predictions": [
            {"id": row["id"], "group_id": row["group_id"], "label": row["label"],
             **{name: labels[index] for name, labels in predictions.items()}}
            for index, row in enumerate(rows)
        ],
        "limitations": "MASSIVE slot spans are human-annotated; NLI support/unknown pairs are deterministic silver, not gold or product traffic.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter-before", type=Path, required=True)
    parser.add_argument("--adapter-after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"citation evaluation failed: {exc}\n")
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
