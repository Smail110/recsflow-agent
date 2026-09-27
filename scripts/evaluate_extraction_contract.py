"""Replay versioned extraction-contract fixtures through one interpretation call."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from recagent.domains.demo import domain_spec, request_adapter
from recagent.interpretation import LLMRequestInterpreter
from recagent.parsing import OllamaClient
from recagent.validation import normalize_proposal, validate_proposal
from recagent.workflow import _WorkflowPromptBackend

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actual_updates(structured) -> list[tuple[str, str, object]]:
    return [(update.field, update.operation, update.value) for update in structured.updates]


def score_case(case: dict[str, Any], structured, validation, adapter_issues: list[Any]) -> dict[str, Any]:
    """Score against pre-authored contract expectations, never agent state."""

    expected = [tuple(update) for update in case.get("expected_updates", [])]
    actual = _actual_updates(structured)
    required_fields = {field for field, _, _ in expected}
    missing = [update for update in expected if update not in actual]
    operation_errors = [
        update
        for update in actual
        if update[0] in required_fields
        and not any(expected_update[0] == update[0] and expected_update[1] == update[1] for expected_update in expected)
    ]
    value_errors = [
        update
        for update in actual
        if update[0] in required_fields
        and any(expected_update[0] == update[0] and expected_update[1] == update[1] for expected_update in expected)
        and update not in expected
    ]
    forbidden = set(case.get("forbidden_fields", []))
    extra = [update for update in actual if update[0] not in required_fields or update[0] in forbidden]
    expected_unresolved = set(case.get("expected_unresolved_fields", []))
    unresolved_fields = {issue.field for issue in structured.issues}
    unresolved_ok = expected_unresolved <= unresolved_fields
    expected_issue_kinds = set(case.get("expected_issue_kinds", []))
    actual_issue_kinds = {issue.kind for issue in structured.issues}
    issue_kinds_ok = expected_issue_kinds <= actual_issue_kinds
    missing_issue_evidence = [issue.field for issue in structured.issues if issue.field in expected_unresolved and not issue.source_text]
    fabricated = [update for update in actual if update[0] in forbidden]
    evidence_findings = [
        finding.code
        for finding in validation.findings
        if finding.code in {"missing_evidence", "evidence_not_in_turn", "issue_missing_evidence", "issue_evidence_not_in_turn"}
    ]
    semantic_issues = [issue.kind for issue in adapter_issues]
    evidence_ok = not evidence_findings and not any(kind == "ambiguity" for kind in semantic_issues)
    proposal_correct = (
        not missing
        and not operation_errors
        and not value_errors
        and not extra
        and unresolved_ok
        and issue_kinds_ok
        and not missing_issue_evidence
    )
    return {
        "schema_valid": True,
        "actual_updates": [list(update) for update in actual],
        "missing_updates": [list(update) for update in missing],
        "extra_updates": [list(update) for update in extra],
        "operation_errors": [list(update) for update in operation_errors],
        "value_errors": [list(update) for update in value_errors],
        "fabricated_updates": [list(update) for update in fabricated],
        "expected_unresolved_fields": sorted(expected_unresolved),
        "actual_unresolved_fields": sorted(unresolved_fields),
        "unresolved_ok": unresolved_ok,
        "expected_issue_kinds": sorted(expected_issue_kinds),
        "actual_issue_kinds": sorted(actual_issue_kinds),
        "issue_kinds_ok": issue_kinds_ok,
        "missing_issue_evidence": sorted(missing_issue_evidence),
        "evidence_ok": evidence_ok,
        "evidence_findings": evidence_findings,
        "adapter_issue_kinds": semantic_issues,
        "validation_status": validation.status,
        "proposal_correct": proposal_correct,
    }


def run_case(case: dict[str, Any], interpreter, adapter) -> dict[str, Any]:
    started = time.perf_counter()
    previous = case.get("previous", {})
    structured, tokens = interpreter.interpret(
        case["message"],
        previous,
        pending_question=case.get("pending_question"),
        pending_context=case.get("pending_context"),
        unresolved=case.get("unresolved"),
    )
    proposal = normalize_proposal(
        structured,
        turn_id=f"contract:{case['id']}",
        message=case["message"],
        constraint_operators=adapter.constraint_operators,
    )
    validation = validate_proposal(
        proposal,
        message=case["message"],
        known_fields=set(adapter.model.model_fields),
        coverage_surfaces={},
    )
    prior_query = adapter.model.model_validate(previous, strict=True)
    _, adapter_issues = adapter.apply(structured, prior_query, case["message"])
    score = score_case(case, structured, validation, adapter_issues)
    return {
        "id": case["id"],
        "family": case["family"],
        "origin": case["origin"],
        "message": case["message"],
        "previous": previous,
        "pending_question": case.get("pending_question"),
        "pending_context": case.get("pending_context"),
        "expectation": {key: case.get(key, []) for key in ("expected_updates", "forbidden_fields", "expected_unresolved_fields")},
        "raw_structured": structured.model_dump(mode="json"),
        "normalized_proposal": proposal.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "adapter_issues": [issue.model_dump(mode="json") for issue in adapter_issues],
        "tokens": tokens,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        **score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("flat", "domain-fields"), required=True)
    parser.add_argument(
        "--domain-value-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For domain-fields only: require a value for every non-clear proposal.",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    fixture_data = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if fixture_data.get("origin") != "synthetic_ai_authored_contract_fixtures":
        raise ValueError("fixture provenance must remain explicit")
    adapter = request_adapter()
    client = OllamaClient(model=args.model, base_url=args.base_url, timeout=args.timeout)
    interpreter = LLMRequestInterpreter(
        _WorkflowPromptBackend(client),
        adapter.descriptor,
        transport=args.transport,
        domain_spec=domain_spec(),
        domain_value_required=args.domain_value_required,
    )
    rows = [run_case(case, interpreter, adapter) for case in fixture_data["cases"]]
    summary = {
        "cases": len(rows),
        "schema_valid": sum(row["schema_valid"] for row in rows),
        "proposal_correct": sum(row["proposal_correct"] for row in rows),
        "missing_update_cases": sum(bool(row["missing_updates"]) for row in rows),
        "extra_update_cases": sum(bool(row["extra_updates"]) for row in rows),
        "operation_error_cases": sum(bool(row["operation_errors"]) for row in rows),
        "value_error_cases": sum(bool(row["value_errors"]) for row in rows),
        "evidence_error_cases": sum(not row["evidence_ok"] for row in rows),
        "unresolved_correct": sum(row["unresolved_ok"] for row in rows),
        "issue_kind_correct": sum(row["issue_kinds_ok"] for row in rows),
        "issue_evidence_missing_cases": sum(bool(row["missing_issue_evidence"]) for row in rows),
        "validation_statuses": dict(Counter(row["validation_status"] for row in rows)),
        "tokens": sum(row["tokens"] for row in rows),
        "latency_ms": {"p50": sorted(row["latency_ms"] for row in rows)[len(rows) // 2], "max": max(row["latency_ms"] for row in rows)},
    }
    output = {
        "schema_version": 1,
        "transport": args.transport,
        "domain_value_required": args.domain_value_required if args.transport == "domain-fields" else None,
        "model": args.model,
        "fixtures": {"path": str(args.fixtures), "sha256": digest(args.fixtures), "origin": fixture_data["origin"]},
        "domain_spec": domain_spec().model_dump(mode="json"),
        "descriptor_sha256": hashlib.sha256(canonical(adapter.descriptor).encode()).hexdigest(),
        "summary": summary,
        "rows": rows,
    }
    output["report_sha256"] = hashlib.sha256(canonical(output).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"transport": args.transport, **summary, "output": str(args.output)}))


if __name__ == "__main__":
    main()
