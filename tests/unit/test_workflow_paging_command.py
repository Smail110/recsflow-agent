"""Documented exact paging commands; controlled responses, no eval fixtures."""

import pytest

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest, Item
from recagent.providers import DemoProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class Interpreter:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages = []

    def interpret(self, message, *_args, **_kwargs):
        self.messages.append(message)
        return next(self.responses), 1


def update(field, value, text):
    return ConstraintUpdate(field=field, value=value, source_text=text)


def fixture_agent(*responses):
    provider = DemoProvider(
        items=[
            Item(
                id=str(i),
                title=f"Invented {i}",
                kind="series",
                genre="комедия",
                minutes=20 + i * 2,
                quality=0.5,
                description="Synthetic paging control",
            )
            for i in range(16)
        ]
    )
    interpreter = Interpreter(
        StructuredRequest(updates=[update("kind", "series", "сериал"), update("genre", "комедия", "комедия")]), *responses
    )
    service = WorkflowAgent(mode="ollama", provider=provider, interpreter=interpreter, response_generator=EvidenceResponseGenerator())
    first = service.chat(ChatRequest(user_id="paging-control", message="сериал, комедия"))
    assert first.recommendations
    return service, interpreter, first


@pytest.mark.parametrize(
    "message",
    [
        "Ещё варианты",
        "еще варианты.",
        "ЕЩЁ ВАРИАНТЫ!",
        "Другие варианты, без изменения условий",
        "Ещё; не меняя условий.",
        "Покажи ещё, сохраняя условия!",
        "Покажите ещё, сохранив условия.",
    ],
)
def test_documented_exact_command_pages_without_an_empty_llm_interpretation(message):
    service, interpreter, first = fixture_agent(StructuredRequest())
    before = list(interpreter.messages)
    second = service.chat(ChatRequest(user_id="paging-control", session_id=first.session_id, message=message))
    assert second.state == "recommend"
    assert "continue_recommendations" in second.trace
    assert second.query == first.query and second.llm_calls == second.llm_tokens == 0
    assert interpreter.messages == before
    assert second.recommendations
    assert not {r.item.id for r in first.recommendations} & {r.item.id for r in second.recommendations}


@pytest.mark.parametrize("message", ["Ещё варианты до 30 минут", "Ещё варианты, но до 30 минут!"])
def test_mixed_command_still_interprets_and_applies_new_limit(message):
    service, interpreter, first = fixture_agent(StructuredRequest(updates=[update("max_minutes", 30, "до 30 минут")]))
    second = service.chat(ChatRequest(user_id="paging-control", session_id=first.session_id, message=message))
    assert interpreter.messages[-1] == message and len(interpreter.messages) == 2
    assert "continue_recommendations" not in second.trace
    assert second.query.max_minutes == 30 and second.llm_calls == 1
    assert second.recommendations and all(r.item.minutes <= 30 for r in second.recommendations)


@pytest.mark.parametrize(
    "message",
    [
        "Не показывай ещё, сохранив условия",
        "Ещё без комедии, сохранив условия",
        "Покажи ещё, но до 40 минут",
        "Ещё, сохранив условия, но не комедию",
    ],
)
def test_preservation_clause_does_not_hide_new_or_negated_instructions(message):
    service, interpreter, first = fixture_agent(StructuredRequest())
    result = service.chat(ChatRequest(user_id="paging-control", session_id=first.session_id, message=message))
    assert interpreter.messages[-1] == message
    assert "continue_recommendations" not in result.trace


def test_exact_command_does_not_bypass_pending_semantic_issue():
    service, interpreter, first = fixture_agent(
        StructuredRequest(issues=[{"kind": "ambiguity", "field": "tone", "source_text": "особый тон", "message": "Какой тон?"}]),
        StructuredRequest(),
    )
    pending = service.chat(ChatRequest(user_id="paging-control", session_id=first.session_id, message="Особый тон"))
    assert pending.state == "clarify" and service.sessions[first.session_id].constraint_state.pending is not None
    third = service.chat(ChatRequest(user_id="paging-control", session_id=first.session_id, message="Ещё варианты"))
    assert len(interpreter.messages) == 3
    assert "continue_recommendations" not in third.trace
    assert third.state == "clarify" and not third.recommendations
    assert service.sessions[first.session_id].constraint_state.pending is not None


def test_exact_command_respects_spent_llm_budget_and_does_not_call_generator():
    service, interpreter, first = fixture_agent(StructuredRequest())
    service.max_calls = service.sessions[first.session_id].calls

    class NoGeneration:
        requires_llm = True
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            raise AssertionError("Exhausted budget must not invoke generation")

    generator = NoGeneration()
    service.response_generator = generator
    second = service.chat(ChatRequest(user_id="paging-control", session_id=first.session_id, message="Ещё варианты"))
    assert "continue_recommendations" in second.trace
    assert second.recommendations and second.query == first.query
    assert second.llm_calls == second.llm_tokens == generator.calls == 0
    assert len(interpreter.messages) == 1


def test_exact_command_without_previous_results_is_not_a_paging_shortcut():
    interpreter = Interpreter(StructuredRequest())
    service = WorkflowAgent(mode="ollama", interpreter=interpreter, response_generator=EvidenceResponseGenerator())
    first = service.chat(ChatRequest(user_id="paging-control", message="Ещё варианты"))
    assert interpreter.messages == ["Ещё варианты"]
    assert "continue_recommendations" not in first.trace
    assert first.state == "clarify" and not first.recommendations
