from recagent.agent import Agent
from recagent.catalog import generate_catalog
from recagent.models import ChatRequest, Query
from recagent.questions import choose_question


def test_no_question_for_homogeneous_candidates():
    item = generate_catalog()[0]
    assert choose_question(Query(kind=item.kind), [item], policy="adaptive", skipped=set()) is None


def test_question_requires_gain_above_cost():
    items = [item for item in generate_catalog() if item.kind == "film"]
    assert choose_question(Query(kind="film"), items, policy="adaptive", skipped=set(), cost=2) is None
    assert choose_question(Query(kind="film"), items, policy="adaptive", skipped=set(), cost=0) is not None


def test_refusal_is_not_asked_again_and_questions_are_bounded():
    agent = Agent(mode="rules", question_policy="adaptive", max_questions=2)
    first = agent.chat(ChatRequest(user_id="new", message="Хочу фильм"))
    assert first.state == "clarify" and first.clarification_slot
    second = agent.chat(ChatRequest(user_id="new", session_id=first.session_id, message="Без разницы"))
    assert second.clarification_slot != first.clarification_slot
    third = agent.chat(ChatRequest(user_id="new", session_id=first.session_id, message="Без разницы"))
    assert third.state == "recommend"
    assert third.clarification_count <= 2


def test_named_item_bypasses_optional_questions():
    item = generate_catalog()[0]
    response = Agent(mode="rules", question_policy="adaptive").chat(ChatRequest(user_id="new", message=f'Найди «{item.title}»'))
    assert response.state == "recommend"
    assert response.timings_ms.keys() >= {"parse", "retrieve", "rank_explain"}
