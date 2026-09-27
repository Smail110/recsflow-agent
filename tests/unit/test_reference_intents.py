"""Reference-dependent intents cannot silently turn into ordinary retrieval."""

import pytest

from recagent.contracts import Constraint, ConstraintState
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest
from recagent.response_generation import EvidenceResponseGenerator
from recagent.validation import normalize_proposal, validate_proposal
from recagent.workflow import WorkflowAgent


def check(request, state=None):
    return validate_proposal(
        normalize_proposal(request, turn_id="new"), state=state, reference_fields={"navigation": "seed_title", "similar": "seed_title"}
    )


@pytest.mark.parametrize("intent", ["navigation", "similar"])
@pytest.mark.parametrize("value,operation", [(None, "set"), ("", "set"), ("   ", "set"), ("Named item", "exclude")])
def test_empty_or_negative_reference_is_not_a_resolvable_object(intent, value, operation):
    result = check(
        StructuredRequest(
            intent=intent, updates=[ConstraintUpdate(field="seed_title", operation=operation, value=value, source_text="Named item")]
        )
    )
    assert any(f.code == "intent_reference_missing" for f in result.findings)


def test_valid_reference_is_inherited_but_clear_cannot_leave_navigation_without_title():
    seed = Constraint(id="old", field="seed_title", value="Named item", op="eq", turn_id="old", domain_version="1")
    state = ConstraintState(intent="navigation", constraints=(seed,))
    patch = StructuredRequest(updates=[ConstraintUpdate(field="tone", value="лёгкий", source_text="лёгкий")])
    assert check(patch, state).status == "PASS"
    clear = StructuredRequest(updates=[ConstraintUpdate(field="seed_title", operation="clear", source_text="убрать")])
    assert any(f.code == "intent_reference_missing" for f in check(clear, state).findings)
    clear.intent = "discovery"
    assert not any(f.code == "intent_reference_missing" for f in check(clear, state).findings)


class BrokenRouting:
    def __init__(self, repair_intent):
        self.repair_intent = repair_intent
        self.repairs = []

    def result(self, intent):
        return StructuredRequest(
            intent=intent,
            updates=[
                ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                ConstraintUpdate(field="genre", value="фантастика", source_text="фантастика"),
            ],
        )

    def interpret(self, *args, **kwargs):
        return self.result("navigation"), 1

    def repair(self, *args, **kwargs):
        self.repairs.append(kwargs)
        return self.result(self.repair_intent), 1


@pytest.mark.parametrize("repair_intent,expected", [("navigation", "clarify"), ("discovery", "recommend")])
def test_one_budgeted_repair_may_fix_routing_but_unchanged_error_never_retrieves(repair_intent, expected):
    interpreter = BrokenRouting(repair_intent)
    agent = WorkflowAgent(mode="ollama", interpreter=interpreter, response_generator=EvidenceResponseGenerator())
    result = agent.chat(ChatRequest(user_id="reference-control", message="Нужен фильм, жанр фантастика"))
    assert result.state == expected and result.mode == "ollama"
    assert len(interpreter.repairs) == 1 and result.llm_calls == 2
    assert any(f["code"] == "intent_reference_missing" for f in interpreter.repairs[0]["feedback"])
    if expected == "clarify":
        assert not result.recommendations
        assert "Назовите объект" in result.message
    else:
        assert result.query.intent == "discovery"
        assert all(r.item.kind == "film" and r.item.genre == "фантастика" for r in result.recommendations)


def test_later_discovery_turn_removes_obsolete_reference_blocker_without_inventing_title():
    from types import SimpleNamespace

    outputs = iter([BrokenRouting("navigation").result("navigation"), BrokenRouting("discovery").result("discovery")])
    interpreter = SimpleNamespace(interpret=lambda *_args, **_kwargs: (next(outputs), 1))
    agent = WorkflowAgent(mode="ollama", interpreter=interpreter, response_generator=EvidenceResponseGenerator())
    first = agent.chat(ChatRequest(user_id="reference-recovery", message="Нужен фильм, жанр фантастика"))
    assert first.state == "clarify" and not first.recommendations
    second = agent.chat(ChatRequest(user_id="reference-recovery", session_id=first.session_id, message="Подберите фильм, фантастика"))
    assert second.state == "recommend" and second.query.intent == "discovery"
    assert second.query.seed_title is None
    assert agent.sessions[second.session_id].constraint_state.pending is None


def test_domain_switch_cannot_inherit_reference_from_previous_domain():
    agent = WorkflowAgent(mode="rules")
    previous = ConstraintState(
        intent="navigation",
        constraints=(
            Constraint(id="kind", field="kind", value="course", op="eq", turn_id="old", domain_version="1"),
            Constraint(id="seed", field="seed_title", value="Named course", op="eq", turn_id="old", domain_version="1"),
        ),
    )
    request = BrokenRouting("navigation").result("navigation")
    proposal = normalize_proposal(request, turn_id="new")
    base = agent._proposal_base(proposal, previous)
    assert base.constraints == () and base.intent == "discovery"
    assert any(f.code == "intent_reference_missing" for f in check(request, base).findings)
