"""A failed extraction should ask a useful question without trapping the session."""

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class Interpreter:
    def __init__(self, *answers):
        self.answers = iter(answers)

    def interpret(self, message, previous, **context):
        return next(self.answers), 1


def update(field, value, source):
    return ConstraintUpdate(field=field, value=value, source_text=source)


def agent(*answers):
    return WorkflowAgent(
        mode="ollama",
        interpreter=Interpreter(*answers),
        response_generator=EvidenceResponseGenerator(),
        question_policy="none",
    )


def test_empty_extraction_without_format_asks_for_format_and_accepts_short_answer():
    workflow = agent(StructuredRequest())
    first = workflow.chat(ChatRequest(user_id="empty-new", message="Посоветуй что-нибудь"))

    assert first.state == "clarify"
    assert "фильм" in first.message and "сериал" in first.message
    assert "условие запроса" not in first.message
    assert workflow.sessions[first.session_id].constraint_state.pending is None

    second = workflow.chat(ChatRequest(user_id="empty-new", session_id=first.session_id, message="фильм"))
    assert second.state == "recommend"
    assert second.query.kind == "film"


def test_empty_extraction_keeps_previous_request_and_can_accept_a_later_change():
    workflow = agent(
        StructuredRequest(updates=[update("kind", "film", "фильм")]),
        StructuredRequest(),
        StructuredRequest(updates=[update("genre", "комедия", "комедию")]),
    )
    first = workflow.chat(ChatRequest(user_id="empty-followup", message="Хочу фильм"))
    second = workflow.chat(ChatRequest(user_id="empty-followup", session_id=first.session_id, message="Поменяй подбор"))

    assert first.state == "recommend"
    assert second.state == "clarify"
    assert "что изменить" in second.message
    assert "условие запроса" not in second.message
    assert second.query.kind == "film"
    assert workflow.sessions[first.session_id].constraint_state.pending is None

    third = workflow.chat(ChatRequest(user_id="empty-followup", session_id=first.session_id, message="Комедию"))
    assert third.state == "recommend"
    assert third.query.kind == "film" and third.query.genre == "комедия"


def test_empty_extraction_with_cited_field_asks_about_that_field():
    workflow = agent(
        StructuredRequest(updates=[update("kind", "film", "фильм")]),
        StructuredRequest(),
    )
    first = workflow.chat(ChatRequest(user_id="empty-field", message="Хочу фильм"))
    second = workflow.chat(ChatRequest(user_id="empty-field", session_id=first.session_id, message="Комедию"))

    assert second.state == "clarify"
    assert "жанр" in second.message
    assert "условие запроса" not in second.message
    assert workflow.sessions[first.session_id].constraint_state.pending is None


def test_degraded_followup_does_not_expose_validator_details():
    class FailingFollowup(Interpreter):
        def interpret(self, message, previous, **context):
            if message == "Повторю пожелание":
                raise TimeoutError("controlled failure")
            return super().interpret(message, previous, **context)

    workflow = WorkflowAgent(
        mode="ollama",
        interpreter=FailingFollowup(
            StructuredRequest(
                updates=[
                    update("kind", "film", "фильм"),
                    update("genre", "комедия", "детектив"),
                ]
            )
        ),
        response_generator=EvidenceResponseGenerator(),
        question_policy="none",
    )
    first = workflow.chat(ChatRequest(user_id="validator-copy", message="Хочу фильм и комедию"))
    second = workflow.chat(ChatRequest(user_id="validator-copy", session_id=first.session_id, message="Повторю пожелание"))

    assert first.state == second.state == "clarify"
    assert second.mode == "rules_fallback"
    assert "жанр" in second.message
    assert "evidence" not in second.message
    assert "условие:" not in second.message


def test_new_request_after_recommendation_gets_its_own_question_budget():
    workflow = WorkflowAgent(
        mode="ollama",
        interpreter=Interpreter(
            StructuredRequest(),
            StructuredRequest(),
            StructuredRequest(updates=[update("genre", "комедия", "Комедию")]),
        ),
        response_generator=EvidenceResponseGenerator(),
        question_policy="adaptive",
        max_questions=2,
    )
    first = workflow.chat(ChatRequest(user_id="fresh-question", message="Посоветуй что-нибудь"))
    second = workflow.chat(ChatRequest(user_id="fresh-question", session_id=first.session_id, message="фильм"))
    third = workflow.chat(ChatRequest(user_id="fresh-question", session_id=first.session_id, message="Не знаю"))
    fourth = workflow.chat(ChatRequest(user_id="fresh-question", session_id=first.session_id, message="Поменяй подбор"))
    fifth = workflow.chat(ChatRequest(user_id="fresh-question", session_id=first.session_id, message="Комедию"))

    assert [turn.state for turn in (first, second, third, fourth, fifth)] == [
        "clarify", "clarify", "recommend", "clarify", "recommend"
    ]
    assert fourth.clarification_count == 3
    assert fifth.query.kind == "film" and fifth.query.genre == "комедия"
