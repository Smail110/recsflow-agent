"""Styling changes presentation, never catalog facts or the generator contract."""

import pytest

from recagent.agent import Agent
from recagent.models import ChatRequest
from recagent.response_generation import (
    EvidenceResponseGenerator,
    GroundedOption,
    LLMGroundedResponseGenerator,
    with_response_style,
)
from recagent.workflow import WorkflowAgent

OPTIONS = [
    GroundedOption(id=str(index), title=f"Object {index}", claims=[f"Price: {index * 100}.", f"Weight: {index}."])
    for index in range(1, 4)
]


def generate(generator):
    return generator.generate(
        original_request="Need a recommendation",
        intent="discovery",
        accepted_constraints={},
        options=OPTIONS,
        unresolved=[],
    )


def test_default_rendering_preserved_and_short_style_uses_only_existing_claims():
    generator = EvidenceResponseGenerator()
    original, tokens = generate(generator)
    assert original == (
        "Подходит «Object 1». Price: 100. Weight: 1.\n"
        "Также можно рассмотреть «Object 2». Price: 200. Weight: 2.\n"
        "Также можно рассмотреть «Object 3». Price: 300. Weight: 3."
    )
    assert tokens == 0
    short = with_response_style(generator, tone="friendly", length="short")
    message, tokens = generate(short)
    assert message == "Предлагаю посмотреть «Object 1». Price: 100.\nЕщё один вариант — «Object 2». Price: 200."
    assert tokens == 0
    assert len(message) < len(original)
    # A request must not leak its style into another request using this instance.
    assert generate(generator)[0] == original


class RecordingBackend:
    def __init__(self):
        self.requests = []

    def structured(self, schema, system, payload):
        self.requests.append((schema, system, payload))
        return schema(items=[{"item_id": "3", "evidence_indexes": [1, 0]}, {"item_id": "1", "evidence_indexes": [0]}]), 13


def test_style_does_not_change_llm_prompt_selection_or_token_accounting():
    backend = RecordingBackend()
    generator = LLMGroundedResponseGenerator(backend)
    normal, normal_tokens = generate(generator)
    short, short_tokens = generate(with_response_style(generator, tone="friendly", length="short"))
    assert backend.requests[0] == backend.requests[1]
    assert normal_tokens == short_tokens == 13
    assert normal.startswith("Подходит «Object 3». Weight: 3. Price: 300.")
    assert short == "Предлагаю посмотреть «Object 3». Weight: 3.\nЕщё один вариант — «Object 1». Price: 100."


class BrokenGenerator:
    requires_llm = True

    def generate(self, *, original_request, intent, accepted_constraints, options, unresolved):
        raise RuntimeError("controlled failure")


@pytest.mark.parametrize("agent_class", [Agent, WorkflowAgent])
@pytest.mark.parametrize("broken", [False, True])
def test_chat_message_and_fallback_honor_style_without_changing_recommendations(agent_class, broken):
    def chat(tone, length):
        agent = agent_class(
            mode="rules", question_policy="none",
            response_generator=BrokenGenerator() if broken else EvidenceResponseGenerator(),
        )
        return agent.chat(ChatRequest(
            user_id="brand", message="Хочу фильм комедию", explanation_tone=tone, explanation_length=length,
        ))

    normal = chat("neutral", "normal")
    styled = chat("friendly", "short")
    assert normal.state == styled.state == "recommend"
    assert normal.message.startswith("Подходит")
    assert styled.message.startswith("Предлагаю посмотреть")
    assert len(styled.message) < len(normal.message)
    assert [rec.item.id for rec in styled.recommendations] == [rec.item.id for rec in normal.recommendations]
    assert styled.query == normal.query
    assert styled.llm_calls == normal.llm_calls
    assert styled.llm_tokens == normal.llm_tokens
    for line, rec in zip(styled.message.splitlines(), styled.recommendations, strict=False):
        assert line.endswith(rec.claim_texts[0])
    assert len(styled.message.splitlines()) <= 2
    assert any("grounded fallback" in warning for warning in styled.warnings) == broken


def test_third_party_generator_without_style_capability_keeps_its_interface():
    class CustomGenerator:
        def generate(self, *, original_request, intent, accepted_constraints, options, unresolved):
            return options[0].claims[0], 4

    generator = CustomGenerator()
    configured = with_response_style(generator, tone="friendly", length="short")
    assert configured is generator
    assert generate(configured) == ("Price: 100.", 4)

@pytest.mark.parametrize("agent_class", [Agent, WorkflowAgent])
def test_budget_fallback_keeps_response_style_without_making_a_call(agent_class):
    agent = agent_class(
        mode="rules", question_policy="none", response_generator=BrokenGenerator(), max_calls=0,
    )
    response = agent.chat(ChatRequest(
        user_id="budget", message="Хочу фильм комедию", explanation_tone="friendly", explanation_length="short",
    ))
    assert response.state == "recommend"
    assert response.message.startswith("Предлагаю посмотреть")
    assert len(response.message.splitlines()) <= 2
    assert response.llm_calls == 0
    assert any("Бюджет LLM исчерпан" in warning for warning in response.warnings)
