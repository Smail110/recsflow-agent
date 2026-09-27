"""Invented-domain controls for a display-only, non-committing preview."""

from copy import deepcopy
from typing import Literal

import pytest
from pydantic import BaseModel

from recagent.contracts import ConstraintState, ValidationFinding
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.pending_preview import build_pending_preview
from recagent.request_mapping import SchemaRequestAdapter
from recagent.state import reduce_state
from recagent.validation import normalize_proposal, validate_proposal


class DeviceRequest(BaseModel):
    category: Literal["keyboard", "display"] | None = None
    max_price: int | None = None
    size: int | None = None


def fixture():
    adapter = SchemaRequestAdapter(DeviceRequest, domain_field="category", constraint_operators={"max_price": "lte"})
    request = StructuredRequest(
        updates=[
            ConstraintUpdate(field="category", value="keyboard", source_text="keyboard"),
            ConstraintUpdate(field="max_price", value=70, source_text="70"),
        ],
        issues=[{"kind": "ambiguity", "field": "size", "source_text": "large", "message": "Size?"}],
    )
    message = "keyboard below 70, large"
    accepted = adapter.normalize(request, message)
    proposal = normalize_proposal(accepted, turn_id="session:new", message=message, constraint_operators=adapter.constraint_operators)
    result = validate_proposal(proposal, message=message)
    return reduce_state(ConstraintState(), result), proposal


def preview(state, proposal):
    return build_pending_preview(
        state, proposal, turn_id="session:new", known_fields=set(DeviceRequest.model_fields), domain_field="category"
    )


def test_invented_domain_preview_is_a_verified_subset_without_committing():
    state, proposal = fixture()
    before = deepcopy(state)
    result = preview(state, proposal)
    assert result.status == "staged"
    assert [(c.field, c.operator, c.value) for c in result.changes] == [("category", "eq", "keyboard"), ("max_price", "lte", 70)]
    assert state == before and state.constraints == () and state.version == 0
    assert result.base_version == state.version


@pytest.mark.parametrize(
    "finding",
    [
        ValidationFinding(code="global", status="reject"),
        ValidationFinding(code="domain_conflict", status="reject", field="category"),
        ValidationFinding(code="global", status="uncertain", field="request"),
        ValidationFinding(code="unknown_scope", status="reject", field="unknown"),
        ValidationFinding(code="unknown_scope", status="reject", change_ids=("unrecognized",)),
        ValidationFinding(code="domain_conflict", status="reject", change_ids=("session:new:0",)),
    ],
)
def test_global_or_domain_blocker_hides_everything(finding):
    state, proposal = fixture()
    state = state.model_copy(update={"pending": state.pending.model_copy(update={"findings": (finding,)})})
    assert preview(state, proposal) is None


def test_field_conflict_hides_only_that_field_and_diagnostics_are_not_exposed():
    state, proposal = fixture()
    finding = ValidationFinding(code="constraint_conflict", status="reject", field="max_price", details="INTERNAL DIAGNOSTIC")
    state = state.model_copy(update={"pending": state.pending.model_copy(update={"findings": (finding,)})})
    result = preview(state, proposal)
    assert [c.field for c in result.changes] == ["category"]
    assert "INTERNAL" not in result.model_dump_json()


def test_absent_verification_stale_version_or_old_source_cannot_be_displayed():
    state, proposal = fixture()
    assert preview(state, proposal.model_copy(update={"changes": ()})) is None
    assert preview(state.model_copy(update={"version": 1}), proposal) is None
    assert build_pending_preview(state, proposal, turn_id="other-session:new", known_fields=set(DeviceRequest.model_fields)) is None
    assert preview(ConstraintState(), proposal) is None


def test_rejected_value_is_not_accepted_just_because_pending_has_it():
    state, proposal = fixture()
    change = proposal.changes[1]
    forged = change.model_copy(update={"constraint": change.constraint.model_copy(update={"value": 900})})
    pending = state.pending.model_copy(update={"proposal": proposal.model_copy(update={"changes": (forged,)})})
    assert preview(state.model_copy(update={"pending": pending}), proposal) is None


def test_workflow_preview_does_not_change_query_budget_and_clears_after_resolution():
    from recagent.models import ChatRequest
    from recagent.response_generation import EvidenceResponseGenerator
    from recagent.workflow import WorkflowAgent

    class Interpreter:
        def __init__(self):
            self.calls = 0

        def interpret(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return StructuredRequest(
                    intent="discovery",
                    updates=[
                        ConstraintUpdate(field="kind", value="course", source_text="курс"),
                        ConstraintUpdate(field="genre", value="python", source_text="Python"),
                    ],
                    issues=[{"kind": "unsupported_constraint", "field": "level", "source_text": "особый", "message": "Какой уровень?"}],
                ), 7
            return StructuredRequest(updates=[ConstraintUpdate(field="level", value="начальный", source_text="начальный")]), 5

    interpreter = Interpreter()
    agent = WorkflowAgent(mode="ollama", interpreter=interpreter, response_generator=EvidenceResponseGenerator())
    first = agent.chat(ChatRequest(user_id="preview-control", message="Нужен курс Python, особый уровень"))
    assert first.state == "clarify" and first.mode == "ollama" and not first.recommendations
    assert first.query.kind is None and first.preferences == {}
    assert first.llm_calls == 1 and first.llm_tokens == 7
    assert {c.field: c.value for c in first.pending_preview.changes} == {"kind": "course", "genre": "python"}
    before = agent.sessions[first.session_id].constraint_state
    assert before.version == 0 and before.constraints == ()
    second = agent.chat(ChatRequest(user_id="preview-control", session_id=first.session_id, message="Тогда начальный"))
    assert second.state == "recommend" and second.pending_preview is None
    assert second.query.kind == "course" and second.query.genre == "python" and second.query.level == "начальный"
    assert agent.sessions[second.session_id].constraint_state.version == 1
    assert second.llm_calls == 1 and interpreter.calls == 2


def test_response_preview_remains_optional_for_existing_clients():
    from recagent.models import ChatRequest, ChatResponse
    from recagent.workflow import WorkflowAgent

    response = WorkflowAgent(mode="rules").chat(ChatRequest(user_id="old-client", message="Нужен фильм комедия"))
    payload = response.model_dump()
    assert payload.pop("pending_preview") is None
    assert ChatResponse.model_validate(payload).query == response.query


@pytest.mark.parametrize("operation,operator,value", [("remove", "neq", "keyboard"), ("clear", "eq", None), ("replace", "eq", "display")])
def test_operations_are_displayed_as_proposals_not_rewritten_as_positive_constraints(operation, operator, value):
    state, proposal = fixture()
    old = proposal.changes[0]
    change = old.model_copy(
        update={"operation": operation, "constraint": old.constraint.model_copy(update={"op": operator, "value": value})}
    )
    proposal = proposal.model_copy(update={"changes": (change,)})
    state = state.model_copy(update={"pending": state.pending.model_copy(update={"proposal": proposal})})
    shown = preview(state, proposal).changes[0]
    assert (shown.operation, shown.operator, shown.value) == (operation, operator, value)


def test_implied_and_historical_changes_are_conservatively_omitted():
    state, proposal = fixture()
    old = proposal.changes[0]
    implied = old.model_copy(update={"id": old.id + ":implied:category"})
    historic = old.model_copy(update={"source_spans": tuple(s.model_copy(update={"origin": "pending_source"}) for s in old.source_spans)})
    pending = state.pending.model_copy(update={"proposal": proposal.model_copy(update={"changes": (implied, historic)})})
    assert preview(state.model_copy(update={"pending": pending}), proposal) is None
