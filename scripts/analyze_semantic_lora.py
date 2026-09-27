"""Compare frozen base and LoRA reports and apply the predeclared acceptance gates."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.semantic_lora_common import canonical, file_sha256, load_config, sha256_bytes


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    integrity = report.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(report).encode()):
        raise ValueError(f"report integrity mismatch: {path}")
    report["report_sha256"] = integrity
    return report


def _metric(report: dict, suite: str, name: str) -> float:
    return report["suites"][suite]["summary"][name]


def _delta(base: dict, lora: dict, suite: str, name: str) -> float:
    return _metric(lora, suite, name) - _metric(base, suite, name)


def _paired_bootstrap(
    base_cases: list[dict],
    lora_cases: list[dict],
    samples: int,
    seed: int,
    numerator: Callable[[dict], float],
    denominator: Callable[[dict], float],
) -> dict[str, float]:
    base_by_id = {item["id"]: item for item in base_cases}
    lora_by_id = {item["id"]: item for item in lora_cases}
    ids = sorted(set(base_by_id) & set(lora_by_id))
    if len(ids) != len(base_cases) or len(ids) != len(lora_cases):
        raise ValueError("paired reports have different case ids")

    def rate(cases: list[dict]) -> float:
        den = sum(denominator(item) for item in cases)
        return sum(numerator(item) for item in cases) / den if den else 0.0

    observed = rate([lora_by_id[key] for key in ids]) - rate([base_by_id[key] for key in ids])
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [ids[rng.randrange(len(ids))] for _ in ids]
        draws.append(rate([lora_by_id[key] for key in chosen]) - rate([base_by_id[key] for key in chosen]))
    draws.sort()
    low = draws[math.floor(0.025 * (len(draws) - 1))]
    high = draws[math.ceil(0.975 * (len(draws) - 1))]
    return {"delta": observed, "ci95_low": low, "ci95_high": high, "samples": samples}


def _issue_delta(base: dict, lora: dict, kind: str) -> float:
    suite = "unseen_schemas_unseen_domains"
    before = base["suites"][suite]["summary"]["issue_preservation"].get(kind, 0.0)
    after = lora["suites"][suite]["summary"]["issue_preservation"].get(kind, 0.0)
    return after - before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--lora-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config_hash = file_sha256(args.config)
    base = _load_report(args.baseline_report)
    lora = _load_report(args.lora_report)
    if base["arm"] != "base" or lora["arm"] != "lora":
        raise ValueError("expected base and lora reports")
    if base["protocol"]["config_sha256"] != config_hash or lora["protocol"]["config_sha256"] != config_hash:
        raise ValueError("report/config mismatch")
    for key in ("evaluator_sha256", "common_sha256", "blind_sha256", "structured_request_schema_sha256"):
        if base["protocol"][key] != lora["protocol"][key]:
            raise ValueError(f"protocol mismatch: {key}")

    blind = "unseen_schemas_unseen_domains"
    bootstrap = _paired_bootstrap(
        base["suites"][blind]["cases"],
        lora["suites"][blind]["cases"],
        config["evaluation"]["bootstrap_samples"],
        config["seeds"]["bootstrap"],
        lambda item: item.get("score", {}).get("grounded_correct", 0),
        lambda item: item.get("score", {}).get("expected_facts", 0),
    )
    gates = config["acceptance"]
    seen_gain = _delta(base, lora, "seen_schemas_seen_domains", "grounded_correct_rate")
    unseen_gain = _delta(base, lora, blind, "grounded_correct_rate")
    gain_ratio = unseen_gain / seen_gain if seen_gain > 0 else (float("inf") if unseen_gain > 0 else 0.0)
    renamed_drop = _metric(lora, "field_renaming_original", "grounded_correct_rate") - _metric(
        lora, "field_renaming", "grounded_correct_rate"
    )
    enum_drop = _metric(lora, "enum_randomization_original", "grounded_correct_rate") - _metric(
        lora, "enum_randomization", "grounded_correct_rate"
    )
    base_conditional = base["suites"][blind]["counterfactual"]["schema_conditional_correctness"]
    lora_conditional = lora["suites"][blind]["counterfactual"]["schema_conditional_correctness"]
    base_no_schema = base["suites"]["without_schema"]["without_schema_safety"]["unsafe_guess_rate"]
    lora_no_schema = lora["suites"]["without_schema"]["without_schema_safety"]["unsafe_guess_rate"]
    valid_rate = min(suite["summary"]["valid_structured_output_rate"] for suite in lora["suites"].values())
    checks = {
        "valid_structured_output": valid_rate >= gates["valid_structured_output_rate_min"],
        "blind_grounded_gain": unseen_gain >= gates["blind_grounded_correct_delta_min"],
        "blind_grounded_ci": bootstrap["ci95_low"] > gates["blind_grounded_correct_delta_ci_low_min"],
        "blind_fact_f1_gain": _delta(base, lora, blind, "fact_f1") >= gates["blind_fact_f1_delta_min"],
        "blind_omissions_decrease": _delta(base, lora, blind, "omission_rate") <= gates["blind_omission_rate_delta_max"],
        "blind_compound_gain": _delta(base, lora, blind, "compound_complete_rate") >= gates["blind_compound_complete_delta_min"],
        "invented_facts_do_not_increase": _delta(base, lora, blind, "invented_fact_rate") <= gates["blind_invented_rate_delta_max"],
        "ambiguity_gain": _issue_delta(base, lora, "ambiguity") >= gates["issue_preservation_delta_min_each"],
        "conflict_gain": _issue_delta(base, lora, "conflict") >= gates["issue_preservation_delta_min_each"],
        "unsupported_gain": _issue_delta(base, lora, "unsupported_constraint") >= gates["issue_preservation_delta_min_each"],
        "unseen_schema_gain": unseen_gain >= gates["unseen_schema_grounded_delta_min"],
        "unseen_to_seen_gain_ratio": gain_ratio >= gates["unseen_to_seen_gain_ratio_min"],
        "field_renaming_drop": renamed_drop <= gates["renamed_grounded_drop_max"],
        "field_renaming_gain": _delta(base, lora, "field_renaming", "grounded_correct_rate") >= gates["renamed_gain_min"],
        "enum_randomization_drop": enum_drop <= gates["enum_randomized_grounded_drop_max"],
        "enum_randomization_gain": _delta(base, lora, "enum_randomization", "grounded_correct_rate") >= gates["enum_randomized_gain_min"],
        "schema_conditional_absolute": lora_conditional >= gates["schema_conditional_correctness_min"],
        "schema_conditional_gain": lora_conditional - base_conditional >= gates["schema_conditional_delta_min"],
        "without_schema_absolute_safety": lora_no_schema <= gates["without_schema_unsafe_guess_rate_max"],
        "without_schema_no_new_guessing": lora_no_schema - base_no_schema <= gates["without_schema_unsafe_guess_delta_max"],
    }
    unsafe = (
        not checks["valid_structured_output"]
        or not checks["invented_facts_do_not_increase"]
        or not checks["without_schema_absolute_safety"]
        or not checks["without_schema_no_new_guessing"]
    )
    generalization_checks = (
        "unseen_schema_gain",
        "unseen_to_seen_gain_ratio",
        "field_renaming_drop",
        "field_renaming_gain",
        "enum_randomization_drop",
        "enum_randomization_gain",
        "schema_conditional_absolute",
        "schema_conditional_gain",
    )
    meaningful = checks["blind_grounded_gain"] and checks["blind_fact_f1_gain"]
    if all(checks.values()):
        verdict = "LORA_SCHEMA_GENERALIZATION_CONFIRMED"
    elif unsafe:
        verdict = "LORA_UNSAFE"
    elif meaningful and not all(checks[name] for name in generalization_checks):
        verdict = "LORA_IMPROVES_BUT_OVERFITS_SCHEMA"
    else:
        verdict = "LORA_NO_MEANINGFUL_GAIN"
    metrics = {
        "blind": {
            name: {
                "base": _metric(base, blind, name),
                "lora": _metric(lora, blind, name),
                "delta": _delta(base, lora, blind, name),
            }
            for name in (
                "fact_precision",
                "fact_recall",
                "fact_f1",
                "grounded_correct_rate",
                "omission_rate",
                "invented_fact_rate",
                "compound_complete_rate",
                "valid_structured_output_rate",
            )
        },
        "issues": {
            kind: {
                "base": base["suites"][blind]["summary"]["issue_preservation"].get(kind, 0.0),
                "lora": lora["suites"][blind]["summary"]["issue_preservation"].get(kind, 0.0),
                "delta": _issue_delta(base, lora, kind),
            }
            for kind in ("ambiguity", "conflict", "unsupported_constraint")
        },
        "seen_schema_grounded": {
            "base": _metric(base, "seen_schemas_seen_domains", "grounded_correct_rate"),
            "lora": _metric(lora, "seen_schemas_seen_domains", "grounded_correct_rate"),
            "delta": seen_gain,
        },
        "unseen_schema_grounded": {
            "base": _metric(base, blind, "grounded_correct_rate"),
            "lora": _metric(lora, blind, "grounded_correct_rate"),
            "delta": unseen_gain,
            "unseen_to_seen_gain_ratio": gain_ratio,
        },
        "field_renaming": {
            "base": _metric(base, "field_renaming", "grounded_correct_rate"),
            "lora": _metric(lora, "field_renaming", "grounded_correct_rate"),
            "gain": _delta(base, lora, "field_renaming", "grounded_correct_rate"),
            "lora_original": _metric(lora, "field_renaming_original", "grounded_correct_rate"),
            "lora_drop": renamed_drop,
        },
        "enum_randomization": {
            "base": _metric(base, "enum_randomization", "grounded_correct_rate"),
            "lora": _metric(lora, "enum_randomization", "grounded_correct_rate"),
            "gain": _delta(base, lora, "enum_randomization", "grounded_correct_rate"),
            "lora_original": _metric(lora, "enum_randomization_original", "grounded_correct_rate"),
            "lora_drop": enum_drop,
        },
        "schema_conditional_correctness": {
            "base": base_conditional,
            "lora": lora_conditional,
            "delta": lora_conditional - base_conditional,
        },
        "without_schema_unsafe_guess_rate": {
            "base": base_no_schema,
            "lora": lora_no_schema,
            "delta": lora_no_schema - base_no_schema,
        },
        "paired_blind_grounded_bootstrap": bootstrap,
    }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "config_sha256": config_hash,
        "base_report_sha256": base["report_sha256"],
        "lora_report_sha256": lora["report_sha256"],
        "acceptance_thresholds": gates,
        "checks": checks,
        "metrics": metrics,
        "case_level_blind_failures_inspected": False,
        "final_holdout_used": False,
        "limitations": [
            "Results are from deterministic schema-first synthetic data, not production traffic.",
            "wrong_kind is a declared proxy because StructuredRequest has no explicit kind/polarity annotation.",
            "The paired bootstrap resamples turns; synthetic template clustering can make its interval optimistic.",
        ],
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "verdict": verdict,
                "output": str(args.output),
                "report_sha256": result["report_sha256"],
                "checks": checks,
                "metrics": metrics,
            }
        )
    )


if __name__ == "__main__":
    main()
