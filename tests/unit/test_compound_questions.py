from recagent.agent import Agent
from recagent.models import ChatRequest


def turn(agent, message, sid=None):
    return agent.chat(ChatRequest(user_id="experiment", message=message, session_id=sid))


def test_partial_answer_keeps_unanswered_preference_unknown():
    agent = Agent(mode="rules", question_policy="compound")
    first = turn(agent, "Хочу фильм")
    assert set(first.clarification_slots) == {"genre", "tone"}
    second = turn(agent, "Детектив", first.session_id)
    assert second.query.genre == "детектив"
    assert second.query.tone is None
    assert second.clarification_slots == ["tone"]
    assert "tone" not in second.preferences
    assert second.preferences["genre"]["source_message"] == "Детектив"


def test_complete_answer_resolves_both_preferences():
    agent = Agent(mode="rules", question_policy="compound")
    first = turn(agent, "Хочу фильм")
    second = turn(agent, "Лёгкий детектив", first.session_id)
    assert second.state == "recommend"
    assert second.clarification_count == 1
    assert second.recommendations
    assert all(rec.item.genre == "детектив" and rec.item.tone == "лёгкий" for rec in second.recommendations)
    assert "plan_action" in second.trace


def test_refusal_does_not_repeat_bundle():
    agent = Agent(mode="rules", question_policy="compound")
    first = turn(agent, "Хочу фильм")
    second = turn(agent, "Без разницы", first.session_id)
    assert second.state == "recommend"
    assert second.query.genre is None and second.query.tone is None


def test_domain_change_clears_preference_provenance():
    agent = Agent(mode="rules", question_policy="compound")
    first = turn(agent, "Лёгкий детективный сериал, один сезон")
    second = turn(agent, "Курс python для новичка с практикой", first.session_id)
    assert "max_seasons" not in second.preferences
    assert "tone" not in second.preferences
    assert second.preferences["kind"]["value"] == "course"


def test_no_optional_question_when_candidates_are_empty():
    agent = Agent(mode="rules", question_policy="compound")
    response = turn(agent, "Фильм не дольше 1 минуты")
    assert response.state == "no_results"
    assert response.clarification_count == 0
