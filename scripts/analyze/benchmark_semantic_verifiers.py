"""Build the frozen semantic-verifier benchmark without loading ML models.

The artifact deliberately separates data readiness from model availability. Model
scores stay unavailable until an explicitly installed, compatible local runtime can
run the requested candidates offline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "llm-first-product"
ATTRIBUTIONS = (
    ("8b", ARTIFACTS / "adapter-attribution-before-8b.json"),
    ("14b", ARTIFACTS / "adapter-attribution-before-14b.json"),
)
MODELS = (
    ("multilingual_e5_large", "intfloat/multilingual-e5-large", "embedding"),
    ("bge_m3", "BAAI/bge-m3", "embedding"),
    ("bge_reranker_v2_m3", "BAAI/bge-reranker-v2-m3", "pairwise_reranker"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(
    sample_id: str,
    *,
    field: str,
    schema: str,
    source_text: str,
    proposed_value: object,
    expected: str,
    provenance: str,
    reason: str,
    category: str,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "field": field,
        "schema": schema,
        "source_text": source_text,
        "proposed_canonical_value": proposed_value,
        "expected": expected,
        "provenance": provenance,
        "reason": reason,
        "category": category,
    }


def pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["schema"],
        row["field"],
        str(row["source_text"]).casefold(),
        json.dumps(row["proposed_canonical_value"], ensure_ascii=False, sort_keys=True),
    )


def historic_samples(
    forbidden_pairs: set[tuple[str, str, str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    seen: set[tuple[object, ...]] = set()
    for label, path in ATTRIBUTIONS:
        report = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"label": label, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        for turn in report["turns"]:
            for row in turn["rows"]:
                update = row.get("update")
                if not update or row["verdict"] not in {"preserved_correctly", "veto_of_correct"}:
                    continue
                value, source_text = update.get("value"), update.get("source_text")
                if value is None or not source_text:
                    continue
                key = (row["field"], json.dumps(value, ensure_ascii=False, sort_keys=True), source_text.casefold(), row["verdict"])
                if key in seen:
                    continue
                seen.add(key)
                category = "historic_veto" if row["verdict"] == "veto_of_correct" else "historic_accept"
                item = sample(
                    f"historic-{label}-{len(rows):03d}",
                    field=row["field"],
                    schema="demo.Query",
                    source_text=source_text,
                    proposed_value=value,
                    expected="MATCH",
                    provenance=f"{path.relative_to(ROOT)}:{turn['dialogue']}:{turn['index']}:{row['field']}",
                    reason=row.get("note") or row["verdict"],
                    category=category,
                )
                if pair_key(item) in forbidden_pairs:
                    exclusions.append(
                        {"provenance": item["provenance"], "reason": "pairwise-only protocol conflicts with safety NO_MATCH/AMBIGUOUS"}
                    )
                    continue
                rows.append(item)
    return rows, sources, exclusions


def safety_samples() -> list[dict[str, Any]]:
    """Hard negatives from adapter safety tests, kept separate from model outputs."""
    return [
        sample(
            "safety-middle-advanced",
            field="level",
            schema="demo.Query",
            source_text="middle-разработчика",
            proposed_value="продвинутый",
            expected="NO_MATCH",
            category="middle",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="middle must remain unresolved",
        ),
        sample(
            "safety-middle-beginner",
            field="level",
            schema="demo.Query",
            source_text="middle",
            proposed_value="начальный",
            expected="NO_MATCH",
            category="middle",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="nearest enum coercion is forbidden",
        ),
        sample(
            "safety-negated-enum",
            field="tone",
            schema="demo.Query",
            source_text="не мрачный",
            proposed_value="мрачный",
            expected="NO_MATCH",
            category="negation",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="negated evidence cannot set the opposite enum",
        ),
        sample(
            "safety-decimal-fragment",
            field="max_minutes",
            schema="demo.Query",
            source_text="1.5",
            proposed_value=5,
            expected="NO_MATCH",
            category="numeric",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="decimal fragment is not integer evidence",
        ),
        sample(
            "safety-numeric-mismatch",
            field="max_minutes",
            schema="demo.Query",
            source_text="90",
            proposed_value=900,
            expected="NO_MATCH",
            category="numeric",
            provenance="synthetic fixture derived from tests/unit/test_adapter_resolution.py",
            reason="numeric citation must equal proposed value",
        ),
        sample(
            "safety-embedded-citation",
            field="format",
            schema="synthetic.FormatQuery",
            source_text="art",
            proposed_value="art",
            expected="NO_MATCH",
            category="embedded_citation",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="surface occurs only inside cartoon",
        ),
        sample(
            "safety-ambiguous-surface",
            field="material",
            schema="synthetic.AmbiguousQuery",
            source_text="материалом",
            proposed_value="small",
            expected="AMBIGUOUS",
            category="ambiguous",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="citation matches two declared values",
        ),
        sample(
            "safety-unsupported-vocabulary",
            field="kind",
            schema="demo.Query",
            source_text="программа обучения",
            proposed_value="course",
            expected="NO_MATCH",
            category="unsupported",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="unknown vocabulary must not be guessed",
        ),
        sample(
            "cross-domain-wood",
            field="material",
            schema="synthetic.FurnitureQuery",
            source_text="деревянный",
            proposed_value="wood",
            expected="MATCH",
            category="cross_domain",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="declared cross-domain surface",
        ),
        sample(
            "cross-domain-negation",
            field="style",
            schema="synthetic.StyleQuery",
            source_text="not warm",
            proposed_value="warm",
            expected="NO_MATCH",
            category="cross_domain",
            provenance="tests/unit/test_adapter_resolution.py",
            reason="profile-driven negation safety",
        ),
    ]


def model_preflight() -> dict[str, Any]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    packages = {
        name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "sentence_transformers", "FlagEmbedding")
    }
    candidates = []
    for key, model_id, kind in MODELS:
        cache_name = "models--" + model_id.replace("/", "--")
        cached = (hub / cache_name).exists()
        candidates.append(
            {
                "key": key,
                "model_id": model_id,
                "kind": kind,
                "cached": cached,
                "cache_path": str(hub / cache_name) if cached else None,
                "revision": None,
                "status": "ready_for_offline_runner" if cached and all(packages.values()) else "unavailable",
            }
        )
    return {"hub_path": str(hub), "packages": packages, "candidates": candidates}


def unavailable_metrics(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "metrics": None, "threshold_curve": None, "safety": None}


def build() -> dict[str, Any]:
    safety = safety_samples()
    historic, sources, exclusions = historic_samples({pair_key(row) for row in safety if row["expected"] != "MATCH"})
    rows = historic + safety
    counts = Counter((row["expected"], row["category"]) for row in rows)
    preflight = model_preflight()
    unavailable = "weights and compatible local ML runtime are not both available"
    return {
        "schema_version": 1,
        "analysis": "semantic_verifier_benchmark_preflight",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inference_calls": 0,
        "production_adapter_modified": False,
        "final_holdout_used": False,
        "sources": sources,
        "dataset": {
            "samples": rows,
            "count": len(rows),
            "excluded_contextual_conflicts": exclusions,
            "composition": {f"{label}:{category}": count for (label, category), count in sorted(counts.items())},
        },
        "current_resolver": {
            "status": "historical_baseline_only",
            "preservation": {"8b": "20/32=0.6250", "14b": "28/41=0.6829"},
            "shadow_ab": {"8b": "0.6250->0.9062", "14b": "0.6829->0.9024", "false_accepts": 0, "safety_probes": "12/12"},
            "note": "Current-commit replay requires the unavailable Python 3.11 project environment.",
        },
        "preflight": preflight,
        "results": {
            "multilingual_e5_large": unavailable_metrics(unavailable),
            "bge_m3": unavailable_metrics(unavailable),
            "bge_reranker_v2_m3": unavailable_metrics(unavailable),
            "hybrid": unavailable_metrics("requires at least one measured ML challenger"),
        },
        "decision": "KEEP_CURRENT_RESOLVER",
        "production_recommendation": "NEED_MORE_EVIDENCE",
        "limitations": [
            "No ML candidate was loaded, downloaded, or scored.",
            "The artifact is a frozen benchmark dataset and runtime preflight, not a quality comparison.",
            "Historical current-resolver measurements predate later safety hardening and must be replayed after Python repair.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_count": payload["dataset"]["count"],
                "composition": payload["dataset"]["composition"],
                "decision": payload["decision"],
                "production_recommendation": payload["production_recommendation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
