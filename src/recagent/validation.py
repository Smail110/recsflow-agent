"""Normalize proposed updates and check evidence, coverage and conflicts.

Validation returns findings without mutating session state or calling the LLM.
The state reducer decides whether to commit or stage the proposal.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from .contracts import Constraint, ConstraintState, NormalizedProposal, ProposedChange, SourceSpan, ValidationFinding, ValidationResult
from .interpretation import StructuredRequest
from .state import apply_changes


def _hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def normalize_proposal(
    request: StructuredRequest,
    *,
    turn_id: str,
    domain_version: str = "1",
    message: str = "",
    constraint_operators: dict[str, str] | None = None,
) -> NormalizedProposal:
    changes: list[ProposedChange] = []
    for index, update in enumerate(request.updates):
        # Offsets are provisional. validate_proposal checks the quoted text;
        # a nonnegative offset alone does not establish valid evidence.
        start = message.casefold().find(update.source_text.casefold()) if message else -1
        span = SourceSpan(turn_id=turn_id, start=max(start, 0), end=max(start, 0) + len(update.source_text), text=update.source_text)
        constraint = Constraint(
            id=f"{turn_id}:{index}",
            field=update.field,
            op="neq" if update.operation in ("exclude", "include") else (constraint_operators or {}).get(update.field, "eq"),
            value=update.value,
            turn_id=turn_id,
            domain_version=domain_version,
            source_spans=(span,),
        )
        operation = {"set": "add", "exclude": "add", "include": "remove", "clear": "clear"}[update.operation]
        changes.append(ProposedChange(id=f"{turn_id}:{index}", operation=operation, constraint=constraint, source_spans=(span,)))
    return NormalizedProposal(
        intent=request.intent,
        changes=tuple(changes),
        issues=tuple(json.dumps(issue.model_dump(), ensure_ascii=False, sort_keys=True) for issue in request.issues),
        raw_request_hash=_hash(request.model_dump(mode="json")),
    )


def _contains_surface(text: str, surface: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(surface.casefold()) + r"(?!\w)", text.casefold()))


def _coverage_obligations(
    message: str,
    coverage_surfaces: dict[str, Iterable[str]] | None,
    *,
    exempt_spans: Iterable[str] = (),
) -> list[tuple[str, str]]:
    if not message or not coverage_surfaces:
        return []
    # Words inside a cited title are not separate user constraints. Mask only
    # that occurrence and preserve positions for the remaining word matches.
    visible = message.casefold()
    for span in exempt_spans:
        start = visible.find(str(span).casefold())
        if start >= 0:
            visible = visible[:start] + (" " * len(str(span))) + visible[start + len(str(span)) :]
    obligations: list[tuple[str, str]] = []
    for field, surfaces in coverage_surfaces.items():
        for surface in surfaces:
            if surface and _contains_surface(visible, str(surface)):
                obligations.append((field, str(surface)))
    return obligations


def _proposed_constraints(proposal: NormalizedProposal, state: ConstraintState | None) -> list[Constraint]:
    return list(apply_changes(state.constraints if state else (), proposal.changes))


def _conflicts(constraints: list[Constraint]) -> list[tuple[str, str]]:
    """Return deterministic field/reason pairs for impossible scalar states."""
    findings: list[tuple[str, str]] = []
    for field in sorted({constraint.field for constraint in constraints}):
        values = [constraint for constraint in constraints if constraint.field == field]
        equals = {constraint.value for constraint in values if constraint.op == "eq"}
        excluded = {constraint.value for constraint in values if constraint.op == "neq"}
        if len(equals) > 1:
            findings.append((field, "несколько несовместимых равенств"))
        if equals & excluded:
            findings.append((field, "значение одновременно требуется и исключается"))
        numeric = [constraint for constraint in values if type(constraint.value) in (int, float)]
        lowers = [constraint for constraint in numeric if constraint.op in ("gt", "gte")]
        uppers = [constraint for constraint in numeric if constraint.op in ("lt", "lte")]
        if lowers and uppers:
            lower = max(lowers, key=lambda constraint: constraint.value)
            upper = min(uppers, key=lambda constraint: constraint.value)
            if lower.value > upper.value or (lower.value == upper.value and (lower.op == "gt" or upper.op == "lt")):
                findings.append((field, "пустой числовой интервал"))
    return findings


def validate_proposal(
    proposal: NormalizedProposal,
    *,
    message: str = "",
    known_fields: set[str] | None = None,
    coverage_surfaces: dict[str, Iterable[str]] | None = None,
    coverage_dependencies: dict[str, set[str]] | None = None,
    state: ConstraintState | None = None,
    coverage_exempt_spans: Iterable[str] = (),
    reference_fields: dict[str, str] | None = None,
) -> ValidationResult:
    findings: list[ValidationFinding] = []
    checks = ["shape", "evidence", "polarity", "coverage", "conflict"]
    effective_intent = proposal.intent or (state.intent if state is not None else None)
    reference_field = (reference_fields or {}).get(effective_intent)
    if reference_field:
        checks.append("intent_reference")
        if not any(
            c.field == reference_field and c.op == "eq" and isinstance(c.value, str) and c.value.strip()
            for c in _proposed_constraints(proposal, state)
        ):
            findings.append(
                ValidationFinding(
                    code="intent_reference_missing",
                    status="uncertain",
                    field=reference_field,
                    details="Выбранная операция требует названия объекта.",
                )
            )
    if not proposal.changes and not proposal.issues:
        findings.append(ValidationFinding(code="empty_patch", status="reject", details="No update or issue was extracted"))
    for change in proposal.changes:
        c = change.constraint
        if known_fields is not None and c is not None and c.field not in known_fields:
            findings.append(ValidationFinding(code="unknown_field", status="reject", change_ids=(change.id,), field=c.field))
        if not change.source_spans or not all(span.text.strip() for span in change.source_spans):
            findings.append(ValidationFinding(code="missing_evidence", status="reject", change_ids=(change.id,), field=c.field if c else None))
        elif message and not all(span.origin == "pending_source" or _contains_surface(message, span.text) for span in change.source_spans):
            findings.append(
                ValidationFinding(
                    code="evidence_not_in_turn",
                    status="reject",
                    change_ids=(change.id,),
                    spans=change.source_spans,
                    field=c.field if c else None,
                )
            )
        if c is not None and isinstance(c.value, (int, float)) and not isinstance(c.value, bool) and c.value < 0:
            findings.append(ValidationFinding(code="invalid_numeric", status="reject", change_ids=(change.id,), field=c.field))
        if change.operation == "clear":
            if c is None:
                if not change.target_constraint_ids:
                    findings.append(ValidationFinding(code="invalid_clear", status="reject", change_ids=(change.id,)))
                elif state is not None and not set(change.target_constraint_ids).issubset({active.id for active in state.constraints}):
                    findings.append(ValidationFinding(code="unknown_clear_target", status="reject", change_ids=(change.id,)))
            elif c.value is not None:
                findings.append(ValidationFinding(code="invalid_clear", status="reject", change_ids=(change.id,), field=c.field))
            elif state is not None:
                clearable_fields = {active.field for active in state.constraints}
                if state.pending is not None:
                    clearable_fields.update(
                        pending.constraint.field for pending in state.pending.proposal.changes if pending.constraint is not None
                    )
                    clearable_fields.update(finding.field for finding in state.pending.findings if finding.field is not None)
                if c.field not in clearable_fields:
                    findings.append(ValidationFinding(code="unknown_clear_target", status="reject", change_ids=(change.id,), field=c.field))
    changed_fields = {change.constraint.field for change in proposal.changes if change.constraint is not None}
    for field, surface in _coverage_obligations(message, coverage_surfaces, exempt_spans=coverage_exempt_spans):
        if field not in changed_fields and not (coverage_dependencies or {}).get(field, set()).intersection(changed_fields):
            findings.append(
                ValidationFinding(
                    code="coverage_gap", status="uncertain", field=field, details=f"Не покрыта объявленная поверхность: {surface}"
                )
            )
    for field, details in _conflicts(_proposed_constraints(proposal, state)):
        findings.append(ValidationFinding(code="constraint_conflict", status="reject", field=field, details=details))
    for raw_issue in proposal.issues:
        # Issues are stored as JSON in the proposal. They still need evidence
        # from this turn, just like accepted updates.
        try:
            issue = json.loads(raw_issue)
        except json.JSONDecodeError:
            issue = {}
        field = issue.get("field") if isinstance(issue, dict) else None
        source_text = issue.get("source_text") if isinstance(issue, dict) else None
        if not source_text:
            findings.append(
                ValidationFinding(
                    code="issue_missing_evidence",
                    status="uncertain",
                    field=field,
                    details="Неопределённое условие не содержит цитату из текущего сообщения.",
                )
            )
        elif message and not _contains_surface(message, str(source_text)):
            findings.append(
                ValidationFinding(
                    code="issue_evidence_not_in_turn",
                    status="reject",
                    field=field,
                    details="Цитата unresolved condition отсутствует в текущем сообщении.",
                )
            )
        findings.append(ValidationFinding(code="interpretation_issue", status="uncertain", field=field, details=raw_issue))
    if any(f.status == "reject" for f in findings):
        status = "REJECT"
    elif any(f.status == "uncertain" for f in findings):
        status = "UNCERTAIN"
    else:
        status = "PASS"
    return ValidationResult(status=status, proposal=proposal, findings=tuple(findings), checks_run=tuple(checks))
