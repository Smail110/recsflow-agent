"""A skipped optional question belongs to its domain, including degraded paths."""

import pytest

from recagent.models import ChatRequest, Item
from recagent.providers import DemoProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class UnavailableBackend:
    def __init__(self):
        self.last_usage = {}

    def structured(self, *args, **kwargs):
        raise TimeoutError("controlled provider-independent LLM failure")


@pytest.mark.parametrize("mode", ["rules", "ollama"])
def test_domain_switch_after_optional_skip_reopens_useful_question_in_degraded_path(mode):
    catalog = [
        Item(
            id=f"{kind}-{index}",
            title=f"Fixture {kind} {index}",
            kind=kind,
            genre=genre,
            quality=0.5,
            description="Independent question-policy fixture",
        )
        for kind, genres in [("course", ["python", "машинное обучение"]), ("film", ["драма", "комедия"])]
        for index, genre in enumerate(genres)
    ]
    agent = WorkflowAgent(
        mode=mode,
        llm=UnavailableBackend(),
        provider=DemoProvider(catalog),
        question_policy="adaptive",
        response_generator=EvidenceResponseGenerator(),
    )
    first = agent.chat(ChatRequest(user_id="review-fallback", message="Нужен курс"))
    assert first.state == "clarify" and first.clarification_slot == "genre"
    skipped = agent.chat(ChatRequest(user_id="review-fallback", session_id=first.session_id, message="не важно"))
    assert skipped.state == "recommend" and skipped.query.kind == "course"
    assert {entry.item.kind for entry in skipped.recommendations} == {"course"}
    changed = agent.chat(ChatRequest(user_id="review-fallback", session_id=first.session_id, message="Теперь фильм"))
    assert changed.query.kind == "film"
    assert changed.state == "clarify" and changed.clarification_slot == "genre"
    assert changed.question_gain > 0
    if mode == "ollama":
        assert changed.mode == "rules_fallback"
        assert changed.telemetry["fallback_reason"] == "llm_failure:TimeoutError"
