"""A broad request should expose alternatives without inventing a genre."""

from recagent.domains.demo import request_adapter
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest
from recagent.providers import DemoProvider
from recagent.ranking import diversify_head
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class StaticInterpreter:
    def __init__(self, proposal):
        self.proposal = proposal

    def interpret(self, *args, **kwargs):
        return self.proposal, 0


def agent_for(*updates, question_policy="none"):
    return WorkflowAgent(
        mode="ollama",
        provider=DemoProvider(),
        interpreter=StaticInterpreter(StructuredRequest(updates=list(updates))),
        response_generator=EvidenceResponseGenerator(),
        question_policy=question_policy,
    )


def test_diverse_head_keeps_best_and_only_promotes_nearby_categories():
    items = [
        {"id": "a", "genre": "detective"},
        {"id": "b", "genre": "detective"},
        {"id": "c", "genre": "comedy"},
        {"id": "d", "genre": "drama"},
        {"id": "e", "genre": "adventure"},
    ]
    ranked = diversify_head(items, field="genre", size=3, window=4)
    assert [item["id"] for item in ranked] == ["a", "c", "d", "b", "e"]
    assert diversify_head(items, field="genre", size=3, window=2) == items


def test_broad_film_is_diverse_and_explains_profile_history():
    agent = agent_for(ConstraintUpdate(field="kind", value="film", source_text="фильм"))
    response = agent.chat(ChatRequest(user_id="demo", message="Хочу фильм"))
    assert response.state == "recommend"
    assert response.query.kind == "film" and response.query.genre is None
    assert len({rec.item.genre for rec in response.recommendations[:3]}) >= 2
    assert "история этого профиля" in response.message
    assert response.message.count("Жанр: детектив.") <= 1


def test_specific_genre_keeps_hard_filter_and_no_broad_note():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        ConstraintUpdate(field="genre", value="комедия", source_text="комедия"),
    )
    response = agent.chat(ChatRequest(user_id="demo", message="Хочу фильм комедия"))
    assert response.state == "recommend"
    assert response.query.genre == "комедия"
    assert {rec.item.genre for rec in response.recommendations} == {"комедия"}
    assert "Жанр не ограничен" not in response.message


def test_explicit_any_genre_is_not_described_as_missing():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        ConstraintUpdate(field="genre", operation="clear", source_text="любой жанр"),
    )
    response = agent.chat(ChatRequest(user_id="new-user", message="Хочу фильм, любой жанр"))
    assert response.state == "recommend" and response.query.genre is None
    assert response.message.startswith("Жанр не ограничен.")
    assert "не указали" not in response.message


def test_bare_film_asks_one_genre_question_and_uncertainty_continues_to_recommendations():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        question_policy="adaptive",
    )
    first = agent.chat(ChatRequest(user_id="demo", message="Хочу фильм посмотреть интересный"))
    assert first.state == "clarify" and first.clarification_slot == "genre"
    assert first.message == "Какой жанр вам сейчас хочется посмотреть?"

    second = agent.chat(ChatRequest(user_id="demo", session_id=first.session_id, message="Не знаю"))
    assert second.state == "recommend"
    assert second.query.genre is None
    assert len({rec.item.genre for rec in second.recommendations[:3]}) >= 2


def test_explicit_any_genre_does_not_trigger_adaptive_question():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        ConstraintUpdate(field="genre", operation="clear", source_text="жанр любой"),
        question_policy="adaptive",
    )
    response = agent.chat(ChatRequest(user_id="new-user", message="Хочу фильм, жанр любой"))
    assert response.state == "recommend"
    assert response.query.genre is None


def test_two_explicit_any_preferences_do_not_trigger_question():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        ConstraintUpdate(field="genre", operation="clear", source_text="жанр любой"),
        ConstraintUpdate(field="tone", operation="clear", source_text="настроение не важно"),
        question_policy="adaptive",
    )
    response = agent.chat(ChatRequest(user_id="new-user", message="Хочу фильм, жанр любой и настроение не важно"))
    assert response.state == "recommend"
    assert response.query.genre is None and response.query.tone is None


def test_omitted_any_preferences_are_not_reasked():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        question_policy="adaptive",
    )
    response = agent.chat(ChatRequest(user_id="new-user", message="Хочу фильм, жанр любой и настроение не важно"))
    assert response.state == "recommend"
    assert response.query.genre is None and response.query.tone is None


def test_reset_to_another_format_can_accept_any_genre():
    class SequenceInterpreter:
        def __init__(self):
            self.calls = 0

        def interpret(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return StructuredRequest(
                    updates=[
                        ConstraintUpdate(field="kind", value="series", source_text="сериал"),
                        ConstraintUpdate(field="genre", value="детектив", source_text="детектив"),
                    ]
                ), 0
            return StructuredRequest(
                reset_constraints=True,
                updates=[
                    ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                    ConstraintUpdate(field="genre", operation="clear", source_text="жанр любой"),
                ],
            ), 0

    agent = WorkflowAgent(
        mode="ollama",
        provider=DemoProvider(),
        interpreter=SequenceInterpreter(),
        response_generator=EvidenceResponseGenerator(),
        question_policy="adaptive",
    )
    first = agent.chat(ChatRequest(user_id="demo", message="Хочу сериал детектив"))
    assert first.state == "recommend" and first.query.kind == "series" and first.query.genre == "детектив"
    second = agent.chat(ChatRequest(user_id="demo", session_id=first.session_id, message="Теперь фильм, жанр любой"))
    assert second.state == "recommend"
    assert second.query.kind == "film" and second.query.genre is None


def test_reset_with_any_genre_does_not_serve_stale_results():
    class SequenceInterpreter:
        def __init__(self):
            self.calls = 0

        def interpret(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return StructuredRequest(
                    updates=[
                        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                        ConstraintUpdate(field="genre", value="комедия", source_text="комедию"),
                    ]
                ), 0
            return StructuredRequest(updates=[ConstraintUpdate(field="kind", value="film", source_text="Фильм")]), 0

    agent = WorkflowAgent(
        mode="ollama",
        provider=DemoProvider(),
        interpreter=SequenceInterpreter(),
        response_generator=EvidenceResponseGenerator(),
        question_policy="adaptive",
    )
    first = agent.chat(ChatRequest(user_id="demo", message="Хочу фильм комедию"))
    assert first.state == "recommend" and first.query.genre == "комедия"
    second = agent.chat(ChatRequest(user_id="demo", session_id=first.session_id, message="Начать заново, жанр любой"))
    # The explicit reset command is checked before LLM interpretation; its
    # declared opt-out is preserved while the old catalog filters clear.
    assert second.state == "clarify" and not second.recommendations
    assert second.query.kind is None and second.query.genre is None
    third = agent.chat(ChatRequest(user_id="demo", session_id=first.session_id, message="Фильм"))
    assert third.state == "recommend" and third.query.kind == "film" and third.query.genre is None, (
        third.state,
        third.message,
        third.trace,
        agent.sessions[third.session_id].skipped_slots,
    )


def test_negated_any_genre_does_not_suppress_question():
    agent = agent_for(
        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
        question_policy="adaptive",
    )
    response = agent.chat(ChatRequest(user_id="new-user", message="Хочу фильм, не любой жанр"))
    assert response.state == "clarify"
    assert response.clarification_slot == "genre"


def test_negated_any_cannot_clear_an_active_genre():
    for phrase in ("не любой жанр", "любой жанр не подходит", "жанр любой не подойдёт", "не совсем любой жанр"):
        assert not request_adapter()._explicit_no_preference("genre", phrase)
    assert request_adapter()._explicit_no_preference("genre", "жанр не важно")

    class SequenceInterpreter:
        def __init__(self):
            self.calls = 0

        def interpret(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return StructuredRequest(
                    updates=[
                        ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                        ConstraintUpdate(field="genre", value="комедия", source_text="комедию"),
                    ]
                ), 0
            return StructuredRequest(
                updates=[ConstraintUpdate(field="genre", operation="clear", source_text="не любой жанр")]
            ), 0

    agent = WorkflowAgent(
        mode="ollama",
        provider=DemoProvider(),
        interpreter=SequenceInterpreter(),
        response_generator=EvidenceResponseGenerator(),
        question_policy="adaptive",
    )
    first = agent.chat(ChatRequest(user_id="demo", message="Хочу фильм комедию"))
    assert first.state == "recommend" and first.query.genre == "комедия"
    second = agent.chat(ChatRequest(user_id="demo", session_id=first.session_id, message="не любой жанр"))
    assert second.query.genre == "комедия"
    assert second.state != "recommend" or {rec.item.genre for rec in second.recommendations} == {"комедия"}
