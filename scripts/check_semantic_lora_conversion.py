"""Gate the exported Ollama adapter on frozen DEV cases before opening sealed blind."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluate_semantic_lora import (
    EvaluationOllamaClient,
    evaluate_rows,
    model_identity,
    verify_baseline,
)
from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    load_config,
    read_jsonl,
    resolve,
    select_cases,
    sha256_bytes,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v1/evaluation/base.json"),
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v1/export/export-report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v1/export/conversion-gate.json"),
    )
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()
    config = load_config(args.config)
    config_hash = file_sha256(args.config)
    evaluator_hash = file_sha256(Path(__file__).with_name("evaluate_semantic_lora.py"))
    baseline = verify_baseline(
        args.baseline_report,
        config_hash,
        evaluator_hash,
        config["dataset"]["blind"]["sha256"],
    )
    export = json.loads(args.export_report.read_text(encoding="utf-8"))
    export_integrity = export.pop("report_sha256", None)
    if export_integrity != sha256_bytes(canonical(export).encode()):
        raise ValueError("export report integrity mismatch")
    if export["status"] != "COMPLETE" or export["config_sha256"] != config_hash:
        raise ValueError("conversion gate requires matching complete export")
    export["report_sha256"] = export_integrity
    evaluation = config["evaluation"]
    tag = evaluation["lora_ollama_tag"]
    identity = model_identity(evaluation["base_url"], tag, evaluation["timeout_seconds"])
    if identity["digest"] != export["ollama_model"]["digest"]:
        raise ValueError("exported Ollama model digest drift")
    dev = read_jsonl(resolve(config["dataset"]["dev"]["path"]))
    selected = select_cases(dev, args.cases, config["seeds"]["evaluation"], "conversion-gate")
    options = {
        "temperature": evaluation["temperature"],
        "seed": evaluation["seed"],
        "num_predict": evaluation["num_predict"],
        "num_ctx": evaluation["num_ctx"],
    }
    client = EvaluationOllamaClient(tag, evaluation["base_url"], evaluation["timeout_seconds"], options)
    evaluated = evaluate_rows(selected, client)
    lora_summary = summarize(evaluated, len(selected))
    selected_ids = {row["id"] for row in selected}
    baseline_cases = [item for item in baseline["suites"]["unseen_schemas_seen_domains"]["cases"] if item["id"] in selected_ids]
    baseline_summary = summarize(baseline_cases, len(selected))
    valid = lora_summary["valid_structured_output_rate"] == 1.0
    not_catastrophic = lora_summary["grounded_correct_rate"] >= baseline_summary["grounded_correct_rate"] - 0.25
    status = "PASS" if valid and not_catastrophic else "FAIL"
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "purpose": "GGUF/Ollama route validation on DEV before sealed blind",
        "config_sha256": config_hash,
        "baseline_report_sha256": baseline["report_sha256"],
        "export_report_sha256": export_integrity,
        "ollama_model": identity,
        "case_ids": sorted(selected_ids),
        "baseline": baseline_summary,
        "lora": lora_summary,
        "checks": {
            "all_outputs_valid": valid,
            "grounded_drop_not_greater_than_0_25": not_catastrophic,
        },
        "cases": evaluated,
        "limitations": [
            "This gate detects a broken converted route; it is not a quality acceptance test.",
            "Direct HF-vs-Ollama text equality is not used because Ollama applies JSON-schema constrained decoding.",
            "Only DEV is used; sealed blind is not opened by this script.",
        ],
    }
    report["report_sha256"] = sha256_bytes(canonical(report).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "status": status,
                "output": str(args.output),
                "report_sha256": report["report_sha256"],
                "baseline": baseline_summary,
                "lora": lora_summary,
            }
        )
    )
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
