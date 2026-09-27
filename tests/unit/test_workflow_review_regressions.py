"""Independent controls for review findings; no evaluation fixtures used."""

import pytest

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest, Item
from recagent.providers import DemoProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class Interpreter:
    def __init__(self, *responses):
        self.responses = iter(responses)

    def interpret(self, *args, **kwargs):
        return next(self.responses), 1


def update(field, value, text, operation="set"):
    return ConstraintUpdate(field=field, value=value, source_text=text, operation=operation)


def test_non_monotonic_retry_is_rejected_without_erasing_clean_updates():
    initial = StructuredRequest(updates=[update("kind", "film", "фильм")])
    repaired = StructuredRequest(
        updates=[update("kind", "film", "фильм")],
        issues=[{"kind": "ambiguity", "field": "genre", "value": "жанр", "message": "Уточните жанр."}],
        clarification_required=True,
    )

    assert WorkflowAgent._repair_regresses(initial, repaired)
    assert not WorkflowAgent._repair_regresses(initial, initial)


def test_more_pages_without_reinterpreting_an_empty_patch_but_does_not_hide_new_conditions():
    provider = DemoProvider(
        items=[Item(id=str(i), title=f"Item {i}", kind="series", genre="комедия", quality=0.5, description="Fixture") for i in range(12)]
    )
    interpreter = Interpreter(
        StructuredRequest(updates=[update("kind", "series", "сериал"), update("genre", "комедия", "комедия")]),
        StructuredRequest(),
    )
    agent = WorkflowAgent(mode="ollama", provider=provider, interpreter=interpreter, response_generator=EvidenceResponseGenerator())
    first = agent.chat(ChatRequest(user_id="paging", message="сериал, комедия"))
    second = agent.chat(ChatRequest(user_id="paging", session_id=first.session_id, message="Ещё!"))
    assert second.state == "recommend" and second.llm_calls == 0
    assert not {r.item.id for r in first.recommendations} & {r.item.id for r in second.recommendations}
    assert first.query == second.query
    third = agent.chat(ChatRequest(user_id="paging", session_id=first.session_id, message="Ещё, но не комедия"))
    assert third.state == "clarify" and not third.recommendations


@pytest.mark.parametrize("clear_message", ["Любой уровень", "Не ограничивай уровень"])
def test_clear_can_cancel_a_named_pending_condition(clear_message):
    agent = WorkflowAgent(
        mode="ollama",
        interpreter=Interpreter(
            StructuredRequest(
                updates=[
                    update("kind", "course", "курс"),
                    update("genre", "python", "Python"),
                    update("level", "продвинутый", "стажёрский"),
                ]
            ),
            StructuredRequest(updates=[update("level", None, clear_message, "clear")]),
        ),
        response_generator=EvidenceResponseGenerator(),
    )
    first = agent.chat(ChatRequest(user_id="clear-pending", message="курс Python, уровень стажёрский"))
    assert first.state == "clarify"
    second = agent.chat(ChatRequest(user_id="clear-pending", session_id=first.session_id, message=clear_message))
    assert second.state == "recommend" and second.query.level is None
    state = agent.sessions[first.session_id].constraint_state
    assert state.pending is None and {c.field for c in state.constraints} == {"kind", "genre"}


def test_generation_failure_reason_survives_graph_and_next_request_recovers():
    class BrokenGenerator:
        requires_llm = True

        def generate(self, **kwargs):
            raise TimeoutError("controlled outage")

    req = StructuredRequest(updates=[update("genre", "python", "Python")])
    agent = WorkflowAgent(mode="ollama", interpreter=Interpreter(req, req), response_generator=BrokenGenerator())
    first = agent.chat(ChatRequest(user_id="generator", message="Python"))
    assert first.recommendations and first.mode == "rules_fallback" and first.degradation == "NO_LLM"
    assert first.telemetry["fallback_reason"] == "response_generation_failure:TimeoutError"
    agent.response_generator = EvidenceResponseGenerator()
    second = agent.chat(ChatRequest(user_id="generator", session_id=first.session_id, message="Python"))
    assert second.mode == "ollama" and second.telemetry["fallback_reason"] is None


def test_interpretation_failure_is_observable_in_workflow_v2():
    class BrokenInterpreter:
        def interpret(self, *args, **kwargs):
            raise TimeoutError("controlled outage")

    agent = WorkflowAgent(mode="ollama", interpreter=BrokenInterpreter(), response_generator=EvidenceResponseGenerator())
    response = agent.chat(ChatRequest(user_id="parser", message="курс Python"))
    assert response.mode == "rules_fallback" and response.degradation == "NO_LLM"
    assert response.telemetry["fallback_reason"] == "llm_failure:TimeoutError"


def test_repair_keeps_supported_update_when_another_update_cites_the_wrong_field():
    class MixedRepair(Interpreter):
        def repair(self, *args, **kwargs):
            return (
                StructuredRequest(
                    updates=[
                        update("max_minutes", None, "Уберите лимит минут", "clear"),
                        update("genre", "комедия", "комедию оставьте"),
                        update("genre", None, "Уберите лимит минут", "clear"),
                    ]
                ),
                1,
            )

    agent = WorkflowAgent(
        mode="ollama",
        interpreter=MixedRepair(
            StructuredRequest(
                updates=[
                    update("kind", "film", "фильм"),
                    update("genre", "комедия", "комедию"),
                    update("max_minutes", 90, "90 минут"),
                ]
            ),
            StructuredRequest(updates=[update("max_minutes", None, "Уберите лимит минут", "clear")]),
        ),
        response_generator=EvidenceResponseGenerator(),
        question_policy="none",
    )
    first = agent.chat(ChatRequest(user_id="mixed-repair", message="фильм, комедию до 90 минут"))
    second = agent.chat(
        ChatRequest(user_id="mixed-repair", session_id=first.session_id, message="Уберите лимит минут, комедию оставьте")
    )
    assert first.state == "recommend"
    assert second.state == "recommend"
    assert second.query.genre == "комедия" and second.query.max_minutes is None
    assert second.llm_usage["repair_rejected_updates"] == 1.0


def test_repair_does_not_drop_a_conflicting_user_value():
    class ConflictingRepair(Interpreter):
        def repair(self, *args, **kwargs):
            return (
                StructuredRequest(
                    updates=[
                        update("kind", "film", "фильм"),
                        update("genre", "комедия", "комедию"),
                        update("genre", "драма", "комедию"),
                    ]
                ),
                1,
            )

    agent = WorkflowAgent(
        mode="ollama",
        interpreter=ConflictingRepair(StructuredRequest(updates=[update("kind", "film", "фильм")])),
        response_generator=EvidenceResponseGenerator(),
        question_policy="none",
    )
    response = agent.chat(ChatRequest(user_id="conflicting-repair", message="фильм, комедию"))
    assert response.state == "clarify" and not response.recommendations
    assert "repair_rejected_updates" not in response.llm_usage
