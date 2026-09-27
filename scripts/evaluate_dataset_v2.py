"""Evaluate a frozen v2 manifest; consume external final data only explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from evals.dataset_v2 import ROOT, V2Case, V2Manifest, canonical_json, load_cases, load_manifest, sha256_value
from evals.metrics import paired_success_interval, summarize, wilson_interval
from evals.oracle import judge, satisfies_spoken

from recagent.catalog import catalog_sha256, generate_catalog
from scripts.evaluate_baselines import CONFIGURATIONS, run_case, source_revision


def summarize_v2(records: list[dict]) -> dict:
    result = summarize(records)
    result.update(
        unique_users=len({row["user_id"] for row in records}),
        absolute_final_success_ci95=wilson_interval(sum(row["final_success"] for row in records), len(records)),
        interval_unit="independent user; one case per user is enforced by the dataset loader",
    )
    return result


def _outcome_success(criteria, catalog_by_id, *, state: str, ids: list[str], clarifications: int = 0) -> bool:
    verdict = judge(
        criteria=criteria,
        catalog_by_id=catalog_by_id,
        state=state,
        shown_ids=ids,
        clarifications=clarifications,
    )
    complete = state != "recommend" or len(set(ids)) >= min(5, len(criteria.acceptable_ids))
    return verdict.success and complete


def mutation_controls(cases: list[V2Case], catalog: list) -> dict:
    """Bad-output controls exercise distinct evaluator failure modes."""
    by_id = {item.id: item for item in catalog}
    recommend = next(
        case for case in cases if case.scenario.final_expected.value == "recommend" and len(case.scenario.criteria.acceptable_ids) >= 5
    )
    no_results = next(case for case in cases if case.scenario.final_expected.value == "no_results")
    scenario = recommend.scenario
    criteria = scenario.criteria.to_criteria(scenario.theta, scenario.final_spoken, scenario.final_expected, scenario.user_id)
    good = list(criteria.ceiling_ids)
    violating = next(
        item.id for item in catalog if not satisfies_spoken(item, scenario.final_spoken)[0] and item.id not in criteria.acceptable_ids
    )
    empty_criteria = no_results.scenario.criteria.to_criteria(
        no_results.scenario.theta,
        no_results.scenario.final_spoken,
        no_results.scenario.final_expected,
        no_results.scenario.user_id,
    )
    checks = {
        "oracle_positive": _outcome_success(criteria, by_id, state="recommend", ids=good),
        "fabricated_id_rejected": not _outcome_success(criteria, by_id, state="recommend", ids=[*good[:-1], "fake-v2-id"]),
        "duplicate_slate_rejected": not _outcome_success(criteria, by_id, state="recommend", ids=[good[0]] * 5),
        "constraint_violation_rejected": not _outcome_success(criteria, by_id, state="recommend", ids=[*good[:-1], violating]),
        "false_no_result_rejected": not _outcome_success(criteria, by_id, state="no_results", ids=[]),
        "results_when_none_rejected": not _outcome_success(empty_criteria, by_id, state="recommend", ids=[next(iter(by_id))]),
        "overclarification_rejected": not _outcome_success(
            criteria,
            by_id,
            state="recommend",
            ids=good,
            clarifications=criteria.max_clarifications + 1,
        ),
        "short_slate_rejected": not _outcome_success(criteria, by_id, state="recommend", ids=good[:1]),
    }
    if not all(checks.values()):
        raise ValueError(f"evaluation mutation control failed: {checks}")
    return checks


def evaluate_manifest(
    manifest: V2Manifest,
    cases: list[V2Case],
    *,
    mode: str = "rules",
    configurations: tuple[str, ...] = CONFIGURATIONS,
    resamples: int = 2000,
    include_records: bool = False,
) -> dict:
    unknown = set(configurations) - set(CONFIGURATIONS)
    if unknown or not configurations:
        raise ValueError(f"unknown/empty configurations: {sorted(unknown)}")
    catalog = generate_catalog(manifest.catalog_seed)
    if catalog_sha256(manifest.catalog_seed) != manifest.catalog_sha256:
        raise ValueError("catalog hash differs from frozen manifest")
    scenarios = [case.scenario for case in cases]
    runs: dict[str, dict] = {}
    for configuration in configurations:
        records = [run_case(scenario, configuration, catalog, mode=mode) for scenario in scenarios]
        by_family: dict[str, list[dict]] = defaultdict(list)
        by_surface: dict[str, list[dict]] = defaultdict(list)
        case_by_id = {case.case_id: case for case in cases}
        for row in records:
            case = case_by_id[row["scenario_id"]]
            by_family[case.scenario_family].append(row)
            by_surface[case.surface_family_id].append(row)
        run = {
            "summary": summarize_v2(records),
            "by_family": {name: summarize_v2(rows) for name, rows in sorted(by_family.items())},
            "by_surface_family": {name: summarize_v2(rows) for name, rows in sorted(by_surface.items())},
        }
        if include_records:
            run["records"] = records
        runs[configuration] = run
        # Comparisons need records even when the report omits them.
        run["_records"] = records
    comparisons = {}
    if "current" in runs:
        right = [dict(row, success=row["final_success"]) for row in runs["current"]["_records"]]
        for baseline in configurations:
            if baseline == "current":
                continue
            left = [dict(row, success=row["final_success"]) for row in runs[baseline]["_records"]]
            comparisons[f"current_minus_{baseline}"] = paired_success_interval(left, right, resamples=resamples)
    for run in runs.values():
        run.pop("_records")
    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": {
            **source_revision(),
            "python_sources_sha256": sha256_value(
                {
                    path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for directory in ("src/recagent", "evals", "scripts")
                    for path in sorted((ROOT / directory).rglob("*.py"))
                }
            ),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "dataset": {
            "version": manifest.dataset_version,
            "split": manifest.split,
            "origin": manifest.origin,
            "blind_claim": manifest.blind_claim,
            "cases_sha256": manifest.cases_sha256,
            "manifest_sha256": manifest.manifest_sha256,
            "cases": len(cases),
            "unique_users": len({case.scenario.user_id for case in cases}),
            "by_family": dict(sorted(Counter(case.scenario_family for case in cases).items())),
            "surface_families": len({case.surface_family_id for case in cases}),
            "statistical_target": manifest.statistical_target,
        },
        "configuration": {
            "mode": mode,
            "configurations": list(configurations),
            "resamples": resamples,
            "history": "cold_start",
        },
        "mutation_controls": mutation_controls(cases, catalog),
        "runs": runs,
        "paired_final_success": comparisons,
        "limitations": [
            *manifest.limitations,
            "Strict all-items success can saturate simple baselines at zero; hit rate and utility are also reported.",
            "No real Recsflow, human preference labels, or adaptive user simulation is evaluated.",
        ],
    }


def _consume_final(manifest_path: Path, manifest: V2Manifest, supplied_hash: str | None) -> Path:
    if supplied_hash != manifest.cases_sha256:
        raise PermissionError("pass --consume-final-holdout with the exact cases SHA256")
    record_path = manifest_path.parent / f"CONSUMED-{manifest.cases_sha256[:16]}.json"
    if record_path.exists():
        raise PermissionError(f"final_holdout was already consumed: {record_path}")
    record = {
        "dataset_version": manifest.dataset_version,
        "cases_sha256": manifest.cases_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "consumed_at_utc": datetime.now(UTC).isoformat(),
        "source": source_revision(),
    }
    # Written before evaluation: a failed or interrupted look still consumes the data.
    try:
        with record_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise PermissionError(f"final_holdout was already consumed: {record_path}") from exc
    return record_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", choices=("rules", "ollama"), default="rules")
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval-v2-20260914/dev-results.json"))
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--allow-final-holdout", action="store_true")
    parser.add_argument("--consume-final-holdout", metavar="CASES_SHA256")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    manifest = load_manifest(args.manifest, allow_final_holdout=args.allow_final_holdout)
    consumption = None
    if manifest.split == "final_holdout":
        if args.validate_only:
            parser.error("final_holdout cannot be opened in validate-only mode")
        consumption = _consume_final(args.manifest, manifest, args.consume_final_holdout)
    cases = load_cases(args.manifest, manifest)
    if args.validate_only:
        print(canonical_json({"valid": True, "split": manifest.split, "cases": len(cases)}))
        return
    result = evaluate_manifest(
        manifest,
        cases,
        mode=args.mode,
        configurations=tuple(args.configurations),
        resamples=args.resamples,
        include_records=args.include_records,
    )
    result["command"] = ["python", "-m", "scripts.evaluate_dataset_v2", *sys.argv[1:]]
    result["consumption_record"] = str(consumption) if consumption else None
    result["report_sha256"] = sha256_value({key: value for key, value in result.items() if key != "report_sha256"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical_json({name: run["summary"] for name, run in result["runs"].items()}))


if __name__ == "__main__":
    main()
