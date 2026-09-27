"""Offline stage attribution for saved LLM-first product reports; zero inference.

Reads a saved runner report, replays SchemaRequestAdapter on the captured raw
validated_output + payload state, and attributes every turn to one of:
  model_omission, model_wrong_value, adapter_veto_loss, policy_question,
  retrieval_or_rerank, evaluation_strictness, pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recagent.contracts import ConstraintState  # noqa: E402
from recagent.domains.demo import request_adapter  # noqa: E402
from recagent.interpretation import InterpretationIssue, StructuredRequest  # noqa: E402
from recagent.models import Query  # noqa: E402
from recagent.request_mapping import canonical_text  # noqa: E402
from recagent.state import reduce_state  # noqa: E402
from recagent.validation import normalize_proposal, validate_proposal  # noqa: E402


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage_context(adapter) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Mirror WorkflowAgent's adapter-derived coverage configuration."""
    surfaces: dict[str, set[str]] = {}
    for table_name in ("enums", "aliases", "scalar_aliases", "numeric_units"):
        for field, values in getattr(adapter, table_name, {}).items():
            surfaces.setdefault(field, set()).update(str(value) for value in values)
    dependencies: dict[str, set[str]] = {}
    for source, mappings in getattr(adapter, "implied_values", {}).items():
        for implied in mappings.values():
            for target in implied:
                dependencies.setdefault(target, set()).add(source)
    return surfaces, dependencies


def classify_turn(
    turn: dict,
    expected_query: dict,
    *,
    dialogue_id: str,
    state: ConstraintState,
    exempt_seed_title_coverage: bool,
) -> tuple[dict, ConstraintState]:
    # A successful recommended turn has a second structured call for the
    # grounded renderer.  It is deliberately a different schema, so choosing
    # the last captured call corrupts an otherwise inference-free replay.
    calls = [call for call in (turn.get("structured_calls") or []) if call.get("role") == "interpretation"]
    if not calls or not calls[-1].get("validated_output"):
        return {"status": "no_raw_output", "detail": "structured call missing or failed", "fields": {}}, state
    raw = calls[-1]["validated_output"]
    payload = calls[-1].get("payload") or {}
    final = turn.get("final_query") or {}
    raw_updates = {u["field"]: u for u in raw.get("updates", []) if isinstance(u, dict)}

    # Exact offline replay of the shipped adapter on the captured raw output.
    adapter = request_adapter()
    coverage_surfaces, coverage_dependencies = _coverage_context(adapter)
    previous = Query.model_validate(payload.get("previous") or {})
    unresolved = [InterpretationIssue.model_validate(i) for i in (payload.get("unresolved") or [])]
    request = StructuredRequest.model_validate(raw)
    replayed_query, replayed_issues = adapter.apply(request, previous, payload.get("message", turn.get("user", "")), unresolved)
    replayed = replayed_query.model_dump(mode="json")
    normalized = normalize_proposal(
        request,
        turn_id=f"{dialogue_id}:{turn.get('index', 0)}",
        message=payload.get("message", turn.get("user", "")),
        constraint_operators=getattr(adapter, "constraint_operators", {}),
    )
    validation = validate_proposal(
        normalized,
        message=payload.get("message", turn.get("user", "")),
        known_fields=set(Query.model_fields),
        coverage_surfaces=coverage_surfaces,
        coverage_dependencies=coverage_dependencies,
        state=state,
        coverage_exempt_spans=(
            update.source_text for update in request.updates if exempt_seed_title_coverage and update.field == "seed_title"
        ),
    )
    runtime_validation = validation
    if validation.status == "PASS" and replayed_issues:
        runtime_validation = validation.model_copy(update={"status": "UNCERTAIN"})
    next_state = reduce_state(state, runtime_validation)
    replay_matches = replayed == final or all(replayed.get(k) == final.get(k) for k in expected_query)

    fields = {}
    for field, want in expected_query.items():
        got = final.get(field)
        if got == want:
            fields[field] = "ok"
            continue
        update = raw_updates.get(field)
        if update is None:
            fields[field] = "model_omission"
        else:
            vetoed = any(issue.field == field for issue in replayed_issues)
            if vetoed or canonical_text(str(update.get("value"))) == canonical_text(str(want)):
                fields[field] = "adapter_veto_loss"
            else:
                fields[field] = "model_wrong_value"

    statuses = [value for value in fields.values() if value != "ok"]
    if not statuses:
        # Query fields are right; the failure (if any) is state or retrieval.
        status = (
            "pass"
            if turn["actual_state"] == turn["expected_state"] and not turn.get("failures")
            else ("policy_question" if turn["actual_state"] == "clarify" else "retrieval_or_rerank")
        )
    elif all(status == "adapter_veto_loss" for status in statuses):
        status = "adapter_veto_loss"
    elif any(status == "adapter_veto_loss" for status in statuses):
        status = "adapter_veto_loss+model"
    elif all(status == "model_omission" for status in statuses):
        status = "model_omission"
    else:
        status = "model_error"
    return {
        "status": status,
        "fields": fields,
        "replay_matches_saved_query": replay_matches,
        "replayed_issues": [{"kind": i.kind, "field": i.field, "value": i.value} for i in replayed_issues],
        "raw_issues": [{"kind": i.get("kind"), "field": i.get("field"), "value": i.get("value")} for i in raw.get("issues", [])],
        "raw_clarification_required": bool(raw.get("clarification_required")),
        "normalized_proposal": normalized.model_dump(mode="json"),
        "proposal_validation": validation.model_dump(mode="json"),
        "runtime_validation": runtime_validation.model_dump(mode="json"),
        "active_constraints_before": [constraint.model_dump(mode="json") for constraint in state.constraints],
        "active_constraints_after": [constraint.model_dump(mode="json") for constraint in next_state.constraints],
        "staged_proposal_before": state.pending.model_dump(mode="json") if state.pending else None,
        "staged_proposal_after": next_state.pending.model_dump(mode="json") if next_state.pending else None,
        "actual_state": turn["actual_state"],
        "expected_state": turn["expected_state"],
        "failures": turn.get("failures", []),
        "user": turn.get("user"),
    }, next_state


def analyze(path: Path, cohort_path: Path, *, exempt_seed_title_coverage: bool) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_id = {dialogue["id"]: dialogue for dialogue in cohort["dialogues"]}
    turns_out = []
    for dialogue in report["dialogues"]:
        expected_turns = cohort_by_id[dialogue["id"]]["turns"]
        state = ConstraintState()
        for index, turn in enumerate(dialogue["turns"]):
            expected = expected_turns[index]
            result, state = classify_turn(
                turn,
                expected.get("expected_query", {}),
                dialogue_id=dialogue["id"],
                state=state,
                exempt_seed_title_coverage=exempt_seed_title_coverage,
            )
            result.update(dialogue=dialogue["id"], tags=dialogue.get("tags", []), index=index)
            turns_out.append(result)

    status_counts = Counter(item["status"] for item in turns_out)
    field_counts = Counter(status for item in turns_out for status in item["fields"].values())
    veto_fields = Counter(field for item in turns_out for field, status in item["fields"].items() if status == "adapter_veto_loss")
    omission_fields = Counter(field for item in turns_out for field, status in item["fields"].items() if status == "model_omission")
    wrong_fields = Counter(field for item in turns_out for field, status in item["fields"].items() if status == "model_wrong_value")
    strictness = [
        item
        for item in turns_out
        if item["status"] in ("pass", "retrieval_or_rerank")
        and any(reason.startswith(("query:seed_title", "literal_title_not_exact")) for reason in item["failures"])
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analysis": "offline_stage_attribution",
        "inference_calls": 0,
        "report": {
            "path": str(path),
            "sha256": sha256_bytes(path),
            "label": report.get("label"),
            "model_identity": report.get("model_identity_after") or report.get("model_identity_before"),
            "cohort_sha256": report.get("cohort", {}).get("sha256"),
            "coverage_exempt_seed_title": exempt_seed_title_coverage,
        },
        "cohort": {"path": str(cohort_path), "sha256": sha256_bytes(cohort_path)},
        "summary": {
            "turns": len(turns_out),
            "status_counts": dict(sorted(status_counts.items())),
            "field_status_counts": dict(sorted(field_counts.items())),
            "adapter_veto_fields": dict(sorted(veto_fields.items())),
            "model_omission_fields": dict(sorted(omission_fields.items())),
            "model_wrong_value_fields": dict(sorted(wrong_fields.items())),
            "replay_matches_saved_final_query": sum(bool(item["replay_matches_saved_query"]) for item in turns_out),
            "evaluation_strictness_only_turns": len(strictness),
        },
        "turns": turns_out,
        "limitations": [
            "The historical seed-title coverage flag is selected explicitly per report. Other adapter or validator "
            "changes after a report make replay attribution approximate.",
            "Expected fields absent from the cohort's expected_query are not scored, so a turn can pass here and "
            "still fail the runner on state or recommendation contract.",
            "Attribution is diagnostic on a fixed AI-authored DEV cohort; it is not a production or traffic claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "product_llm_first_dev_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed-title-coverage-exempt-report",
        type=Path,
        action="append",
        default=[],
        help="Report path whose historical runtime exempted cited seed-title spans from coverage.",
    )
    args = parser.parse_args()
    exempt = {path.resolve() for path in args.seed_title_coverage_exempt_report}
    results = [analyze(path, args.cohort, exempt_seed_title_coverage=path.resolve() in exempt) for path in args.report]
    payload = results[0] if len(results) == 1 else {"schema_version": 1, "runs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for result in results:
        print(json.dumps({"report": result["report"]["path"], "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
