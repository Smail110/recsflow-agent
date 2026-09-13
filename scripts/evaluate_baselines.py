"""Run frozen scripted dialogues through the independent preference oracle.

This is a cold-start baseline, not an adaptive user simulator. Only public
utterances enter the agent. Ground truth is used after each response is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from evals.metrics import paired_success_interval, summarize
from evals.oracle import judge, satisfies_spoken
from evals.scenarios import ANSWER_SIZE, SPLIT_SEEDS, Scenario, generate_scenarios

from recagent.agent import Agent
from recagent.catalog import catalog_sha256, generate_catalog
from recagent.grounding import validate_evidence
from recagent.models import ChatRequest, Item, Query
from recagent.providers import DemoProvider

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("random", "popularity", "platform_quality", "current", "oracle")


def fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_revision() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", stderr=subprocess.DEVNULL).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_case(scenario: Scenario, configuration: str, catalog: Sequence[Item], *, mode: str = "rules") -> dict:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration: {configuration}")
    by_id = {item.id: item for item in catalog}
    criteria = scenario.criteria.to_criteria(scenario.theta, scenario.final_spoken, scenario.final_expected, scenario.user_id)
    record = {
        "scenario_id": scenario.scenario_id,
        "user_id": scenario.user_id,
        "kind": scenario.kind,
        "configuration": configuration,
        "clarifications": 0,
        "latencies_ms": [],
        "llm_calls": 0,
        "llm_tokens": 0,
        "evidence_count": 0,
        "invalid_evidence": 0,
        "hard_constraint_violations": 0,
        "turns": [],
    }
    failures = []
    if configuration == "current":
        provider = DemoProvider(list(catalog))
        agent = Agent(provider=provider, mode=mode)
        sid, shown_before = None, set()
        for index, turn in enumerate(scenario.turns):
            response = agent.chat(ChatRequest(message=turn.utterance, user_id=scenario.user_id, session_id=sid))
            sid = response.session_id
            ids = [rec.item.id for rec in response.recommendations]
            violations = sum(not satisfies_spoken(by_id[item_id], turn.spoken)[0] for item_id in ids)
            record["hard_constraint_violations"] += violations
            if violations:
                failures.append("hard_constraint_violation")
            if index < len(scenario.turns) - 1 and response.state != turn.expected.value:
                failures.append("intermediate_state_mismatch")
            if scenario.kind == "more_variants" and index > 0 and shown_before.intersection(ids):
                failures.append("repeated_item")
            shown_before.update(ids)
            history = provider.lookup(provider.history(scenario.user_id))
            seed_item = provider.find_title(response.query.seed_title) if response.query.seed_title else None
            for recommendation in response.recommendations:
                for evidence in recommendation.evidence:
                    record["evidence_count"] += 1
                    record["invalid_evidence"] += not validate_evidence(evidence, by_id[recommendation.item.id], response.query, history, seed_item)
            record["latencies_ms"].append(response.latency_ms)
            record["llm_calls"] += response.llm_calls
            record["llm_tokens"] += response.llm_tokens
            record["turns"].append({
                "utterance": turn.utterance,
                "expected_state": turn.expected.value,
                "state": response.state,
                "message": response.message,
                "shown_ids": ids,
                "query": response.query.model_dump(mode="json"),
                "warnings": response.warnings,
            })
        state = response.state
        record["clarifications"] = response.clarification_count
    else:
        state = "recommend"
        if configuration == "oracle":
            state = scenario.final_expected.value
            ids = list(criteria.ceiling_ids) if state == "recommend" else []
        elif configuration == "random":
            rng = random.Random(fingerprint([scenario.seed, scenario.scenario_id]))
            ids = rng.sample(sorted(by_id), min(ANSWER_SIZE, len(by_id)))
        elif configuration == "popularity":
            ids = [item.id for item in sorted(catalog, key=lambda item: (-(item.popularity or 0), item.id))[:ANSWER_SIZE]]
        else:
            # B1 sees only user_id: Query is EMPTY, never populated from the scenario.
            ids = DemoProvider(list(catalog)).retrieve(scenario.user_id, Query(), limit=ANSWER_SIZE)
        record["hard_constraint_violations"] = sum(not satisfies_spoken(by_id[item_id], scenario.final_spoken)[0] for item_id in ids)

    verdict = judge(criteria=criteria, catalog_by_id=by_id, state=state, shown_ids=ids, clarifications=record["clarifications"])
    # A one-item answer must not beat a five-item answer merely by hiding misses.
    incomplete = state == "recommend" and len(ids) < min(ANSWER_SIZE, len(criteria.acceptable_ids))
    if incomplete:
        failures.append("incomplete_slate")
    if record["invalid_evidence"]:
        failures.append("invalid_evidence")
    record.update({
        "state": state,
        "shown_ids": ids,
        "success": verdict.success and not failures,
        "final_success": verdict.success and not incomplete,
        "hit_rate": verdict.hit_rate,
        "mean_utility": verdict.mean_utility,
        "failures": sorted(set(verdict.failures) | set(failures)),
    })
    return record


def evaluate(*, split: str = "dev", size: int | None = None, mode: str = "rules", resamples: int = 1000) -> dict:
    catalog = generate_catalog(SPLIT_SEEDS["dev"])
    scenarios, dataset_summary = generate_scenarios(split, size, catalog=catalog)
    runs = {}
    for configuration in CONFIGURATIONS:
        records = [run_case(scenario, configuration, catalog, mode=mode) for scenario in scenarios]
        groups = defaultdict(list)
        for record in records:
            groups[record["kind"]].append(record)
        runs[configuration] = {
            "summary": summarize(records),
            "by_kind": {kind: summarize(rows) for kind, rows in sorted(groups.items())},
            "records": records,
        }
    comparisons = {}
    for baseline in ("random", "popularity", "platform_quality"):
        # All methods share the same final-turn criterion; trajectory checks are
        # reported separately because list-only baselines do not have a dialogue.
        left = [dict(row, success=row["final_success"]) for row in runs[baseline]["records"]]
        right = [dict(row, success=row["final_success"]) for row in runs["current"]["records"]]
        comparisons[f"current_minus_{baseline}"] = paired_success_interval(left, right, resamples=resamples)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": {
            **source_revision(),
            "python_sources_sha256": fingerprint({
                path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in ("src/recagent", "evals", "scripts")
                for path in sorted((ROOT / directory).rglob("*.py"))
            }),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "configuration": {"mode": mode, "split": split, "size": len(scenarios), "slate_size": ANSWER_SIZE, "history": "cold_start"},
        "dataset": {
            **dataset_summary,
            "catalog_seed": SPLIT_SEEDS["dev"],
            "catalog_sha256": catalog_sha256(),
            "scenarios_sha256": fingerprint([scenario.model_dump(mode="json") for scenario in scenarios]),
        },
        "runs": runs,
        "paired_final_success": comparisons,
        "limitations": [
            "Synthetic cold-start dialogues: no real Recsflow integration or user-history experiment.",
            "Fixed utterance replay, without extra replies to final clarifications; not an adaptive simulator.",
            "platform_quality is DemoProvider without query filters, not measured Recsflow quality.",
            "Final-success comparisons use identical final criteria; full success also checks agent trajectory and slate size.",
            "Evidence validation reuses the product checker; it is not an independent audit of explanation text.",
            "Latency is sequential local execution, not HTTP load; zero calls in rules mode do not measure LLM speed.",
            "Holdout shares template families with dev; only profiles and scenario sampling are separated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--size", type=int)
    parser.add_argument("--mode", choices=("rules", "ollama"), default="rules")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--check", action="store_true", help="fail if the oracle ceiling or random negative control fails")
    parser.add_argument("--output", type=Path, default=Path("report/baselines-local.json"))
    parser.add_argument("--summary-output", type=Path, help="optional compact report without dialogue records")
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    result = evaluate(split=args.split, size=args.size, mode=args.mode, resamples=args.resamples)
    result["command"] = ["python", "-m", "scripts.evaluate_baselines", *sys.argv[1:]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.summary_output:
        compact = {**result, "runs": {
            name: {"summary": run["summary"], "by_kind": run["by_kind"]}
            for name, run in result["runs"].items()
        }}
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, run in result["runs"].items():
        metrics = run["summary"]
        print(f"{name:18} final={metrics['final_success_rate']:.3f} full={metrics['success_rate']:.3f} n={metrics['scenarios']}")
    print(f"Report: {args.output}")
    if args.check:
        check_controls(result)


def check_controls(result: dict) -> None:
    """Check the measurement instrument, without tuning the agent to its output."""
    oracle = result["runs"]["oracle"]["records"]
    random_rows = result["runs"]["random"]["records"]
    recommendation_ids = {row["scenario_id"] for row in oracle if row["state"] == "recommend"}
    negative_control = [row for row in random_rows if row["scenario_id"] in recommendation_ids]
    if not oracle or not all(row["success"] for row in oracle):
        raise ValueError("oracle ceiling failed: inspect evaluation criteria")
    if len(negative_control) < 20:
        raise ValueError("at least 20 recommendation scenarios are required for the negative control")
    if sum(row["final_success"] for row in negative_control) / len(negative_control) > 0.1:
        raise ValueError("random control succeeds too often: inspect dataset degeneracy")


if __name__ == "__main__":
    main()
