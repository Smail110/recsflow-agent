"""Explicitly declining an optional preference must not block recommendations."""

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class Interpreter:
    def interpret(self, message, *_args, **_kwargs):
        if message == "Посоветуй сериал":
            return StructuredRequest(updates=[ConstraintUpdate(field="kind", value="series", source_text="сериал")]), 0
        return StructuredRequest(updates=[ConstraintUpdate(field="genre", operation="clear", source_text="Любой жанр")]), 0


def test_unset_optional_preference_is_a_verified_noop_in_a_dialogue():
    agent = WorkflowAgent(mode="ollama", interpreter=Interpreter(), response_generator=EvidenceResponseGenerator(), question_policy="none")
    first = agent.chat(ChatRequest(message="Посоветуй сериал", user_id="demo"))
    second = agent.chat(ChatRequest(message="Любой жанр", session_id=first.session_id, user_id="demo"))

    assert first.state == "recommend"
    assert second.state == "recommend"
    assert second.query.kind == "series" and second.query.genre is None
    assert second.recommendations
    assert "verified_no_preference" in second.trace
