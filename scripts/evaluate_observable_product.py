"""Replay saved open-cohort reports; no model or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evals.observable_product import VERSION, compare_reports, safe_read_json, validate_cohort

from recagent.catalog import catalog_sha256, generate_catalog

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/observable-product-v1.json")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    config, config_hash = safe_read_json(args.config)
    if config["metric_version"] != VERSION:
        raise ValueError("Unsupported metric version")
    data, dataset_hash = safe_read_json(ROOT / config["dataset_path"])
    validate_cohort(data)
    if dataset_hash != config["dataset_sha256"]:
        raise ValueError("Configured cohort hash mismatch")
    seed = data["catalog"]["seed"]
    if catalog_sha256(seed) != config["catalog_sha256"]:
        raise ValueError("Public catalog hash mismatch")
    baseline, before_hash = safe_read_json(args.baseline)
    candidate, after_hash = safe_read_json(args.candidate)
    sources = ["evals/observable_product.py", "evals/oracle.py", "evals/explanations.py", "scripts/evaluate_observable_product.py"]
    protocol = {
        "metric_version": VERSION,
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
        "catalog_sha256": config["catalog_sha256"],
        "baseline_file_sha256": before_hash,
        "candidate_file_sha256": after_hash,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "rules": config["rules"],
        "limitations": config["limitations"],
        "inference": False,
    }
    # Freeze inputs and rules before computing any verdict. Existing mismatched
    # protocols are never silently overwritten.
    if args.protocol.exists():
        if safe_read_json(args.protocol)[0] != protocol:
            raise ValueError("Existing protocol differs; choose a new protocol path")
    else:
        write_json(args.protocol, protocol)
    result = compare_reports(
        data,
        baseline,
        candidate,
        cohort_sha256=dataset_hash,
        catalog={i.id: i for i in generate_catalog(seed)},
        require_matching_run_configuration=config["require_matching_run_configuration"],
    )
    result["protocol"] = protocol
    result["source_changes"] = {"baseline": baseline.get("source"), "candidate": candidate.get("source")}
    write_json(args.output, result)
    print(json.dumps({"complete": True, "before": result["before"]["lanes"], "after": result["after"]["lanes"]}))


if __name__ == "__main__":
    main()
