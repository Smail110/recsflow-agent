import pytest

from recagent.contracts import Constraint, ConstraintState, NormalizedProposal, PendingState, ProposedChange, SourceSpan, ValidationFinding
from recagent.filtering import hard_filter
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.state import apply_changes, reduce_state
from recagent.validation import normalize_proposal, validate_proposal


def test_known_surfaces_omitted_by_interpreter_stage_a_coverage_clarification():
    message = "Нужен курс Python с практикой"
    proposal = normalize_proposal(
        StructuredRequest(updates=[ConstraintUpdate(field="kind", value="course", source_text="курс")]),
        turn_id="turn-1",
        message=message,
    )
    result = validate_proposal(
        proposal,
        message=message,
        known_fields={"kind", "genre", "practical"},
        coverage_surfaces={"kind": {"курс"}, "genre": {"Python"}, "practical": {"практикой"}},
    )
    assert result.status == "UNCERTAIN"
    assert {finding.field for finding in result.findings if finding.code == "coverage_gap"} == {"genre", "practical"}


def test_coverage_ignores_catalog_surfaces_inside_a_cited_seed_title_only():
    message = "Найди «Несуществующий курс по Python»; отдельно нужен курс."
    proposal = normalize_proposal(
        StructuredRequest(
            updates=[
                ConstraintUpdate(field="seed_title", value="Несуществующий курс по Python", source_text="«Несуществующий курс по Python»")
            ]
        ),
        turn_id="turn-title",
        message=message,
    )
    result = validate_proposal(
        proposal,
        message=message,
        known_fields={"seed_title", "kind", "genre"},
        coverage_surfaces={"kind": {"курс"}, "genre": {"Python"}},
        coverage_exempt_spans=("«Несуществующий курс по Python»",),
    )
    # The title does not demand `kind`/`genre`; the second, non-title “курс”
    # still does, which prevents a title exemption from hiding ordinary text.
    assert result.status == "UNCERTAIN"
    assert {finding.field for finding in result.findings if finding.code == "coverage_gap"} == {"kind"}


def test_validator_rejects_a_fabricated_evidence_quote():
    proposal = normalize_proposal(
        StructuredRequest(updates=[ConstraintUpdate(field="kind", value="course", source_text="курс")]),
        turn_id="turn-1",
        message="Нужен фильм",
    )
    result = validate_proposal(proposal, message="Нужен фильм", known_fields={"kind"})
    assert result.status == "REJECT"
    assert any(finding.code == "evidence_not_in_turn" for finding in result.findings)


def test_validator_rejects_eq_and_exclusion_of_the_same_value():
    span = SourceSpan(turn_id="turn-1", start=0, end=8, text="детектив")
    required = Constraint(id="eq", field="genre", op="eq", value="детектив", turn_id="turn-1", domain_version="1", source_spans=(span,))
    excluded = Constraint(id="neq", field="genre", op="neq", value="детектив", turn_id="turn-1", domain_version="1", source_spans=(span,))
    proposal = NormalizedProposal(
        changes=(
            ProposedChange(id="eq", operation="add", constraint=required, source_spans=(span,)),
            ProposedChange(id="neq", operation="add", constraint=excluded, source_spans=(span,)),
        ),
        raw_request_hash="proposal",
    )
    result = validate_proposal(proposal, message="детектив", known_fields={"genre"}, state=ConstraintState())
    assert result.status == "REJECT"
    assert any(finding.code == "constraint_conflict" for finding in result.findings)


def test_negative_constraint_fails_unknown_catalog_metadata():
    exclusion = Constraint(id="tone", field="tone", op="neq", value="мрачный", turn_id="turn-1", domain_version="1")
    kept, diagnostics = hard_filter(
        [
            {"id": "dark", "tone": "мрачный"},
            {"id": "light", "tone": "лёгкий"},
            {"id": "unknown", "tone": None},
        ],
        (exclusion,),
    )
    assert [item["id"] for item in kept] == ["light"]
    assert diagnostics == {"unknown": ["unknown"], "failed": ["dark"]}


def _constraint(identifier, field, operation, value, turn="current"):
    return Constraint(id=identifier, field=field, op=operation, value=value, turn_id=turn, domain_version="test/1")


def _proposal(*changes):
    span = SourceSpan(turn_id="current", start=0, end=6, text="update")
    return NormalizedProposal(
        changes=tuple(
            ProposedChange(id=constraint.id, operation=operation, constraint=constraint, source_spans=(span,))
            for operation, constraint in changes
        ),
        raw_request_hash="transition-test",
    )


@pytest.mark.parametrize("operation", ["add", "replace"])
def test_two_same_turn_equalities_remain_visible_as_an_atomic_conflict(operation):
    previous = _constraint("old-material", "material", "eq", "steel", turn="previous")
    state = ConstraintState(constraints=(previous,))
    proposal = _proposal(
        (operation, _constraint("first-material", "material", "eq", "wood")),
        (operation, _constraint("second-material", "material", "eq", "glass")),
    )

    preview = apply_changes(state.constraints, proposal.changes)
    result = validate_proposal(proposal, state=state)
    committed = reduce_state(state, result)

    assert {constraint.value for constraint in preview} == {"wood", "glass"}
    assert result.status == "REJECT"
    assert any(finding.code == "constraint_conflict" for finding in result.findings)
    assert committed.constraints == state.constraints
    assert committed.pending is not None


def test_upper_bound_correction_keeps_the_other_range_boundary():
    lower = _constraint("lower", "width", "gte", 10, turn="previous")
    upper = _constraint("upper", "width", "lte", 20, turn="previous")
    replacement = _constraint("new-upper", "width", "lte", 30)
    state = ConstraintState(constraints=(lower, upper))
    proposal = _proposal(("add", replacement))

    result = validate_proposal(proposal, state=state)
    committed = reduce_state(state, result)

    assert result.status == "PASS"
    assert committed.constraints == apply_changes(state.constraints, proposal.changes) == (lower, replacement)
    assert state.constraints == (lower, upper)


def test_clear_field_is_applied_before_conflict_validation_and_commit():
    material = _constraint("material", "material", "eq", "wood", turn="previous")
    width = _constraint("width", "width", "lte", 90, turn="previous")
    exclusion = _constraint("excluded-material", "material", "neq", "wood")
    state = ConstraintState(constraints=(material, width))
    proposal = _proposal(
        ("clear", _constraint("clear-material", "material", "eq", None)),
        ("add", exclusion),
    )

    result = validate_proposal(proposal, state=state)
    committed = reduce_state(state, result)

    assert result.status == "PASS"
    assert committed.constraints == apply_changes(state.constraints, proposal.changes) == (width, exclusion)


def test_clear_by_id_preserves_another_bound_of_the_same_field():
    lower = _constraint("lower", "width", "gte", 10, turn="previous")
    upper = _constraint("upper", "width", "lte", 20, turn="previous")
    state = ConstraintState(constraints=(lower, upper))
    span = SourceSpan(turn_id="current", start=0, end=22, text="remove the upper bound")
    proposal = NormalizedProposal(
        changes=(ProposedChange(id="clear-upper", operation="clear", target_constraint_ids=(upper.id,), source_spans=(span,)),),
        raw_request_hash="clear-by-id",
    )

    result = validate_proposal(proposal, state=state)

    assert result.status == "PASS"
    assert reduce_state(state, result).constraints == apply_changes(state.constraints, proposal.changes) == (lower,)


@pytest.mark.parametrize("source", ["proposal", "finding"])
def test_clear_can_cancel_a_named_pending_field_without_an_active_constraint(source):
    width = _constraint("width", "width", "lte", 90, turn="previous")
    staged = _proposal(("add", _constraint("material", "material", "eq", "uncertain")))
    pending = PendingState(
        proposal=staged if source == "proposal" else NormalizedProposal(raw_request_hash="pending-issue"),
        findings=(ValidationFinding(code="interpretation_issue", status="uncertain", field="material"),) if source == "finding" else (),
        question_id="material-question",
        base_version=0,
    )
    state = ConstraintState(constraints=(width,), pending=pending)
    proposal = _proposal(("clear", _constraint("clear-material", "material", "eq", None)))

    result = validate_proposal(proposal, known_fields={"material", "width", "color"}, state=state)
    committed = reduce_state(state, result)
    unrelated = validate_proposal(
        _proposal(("clear", _constraint("clear-color", "color", "eq", None))),
        known_fields={"material", "width", "color"},
        state=state,
    )

    assert result.status == "PASS"
    assert committed.constraints == (width,) and committed.pending is None
    assert unrelated.status == "REJECT"
    assert any(finding.code == "unknown_clear_target" and finding.field == "color" for finding in unrelated.findings)


def test_clear_by_unknown_id_remains_rejected():
    span = SourceSpan(turn_id="current", start=0, end=6, text="remove")
    proposal = NormalizedProposal(
        changes=(ProposedChange(id="clear-unknown", operation="clear", target_constraint_ids=("missing",), source_spans=(span,)),),
        raw_request_hash="clear-unknown-id",
    )

    result = validate_proposal(proposal, state=ConstraintState())

    assert result.status == "REJECT"
    assert any(finding.code == "unknown_clear_target" for finding in result.findings)
