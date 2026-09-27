import pytest

from recagent.agent import Agent
from recagent.catalog import generate_catalog
from recagent.models import ChatRequest, Query
from recagent.parsing import rule_parse
from recagent.providers import DemoProvider


@pytest.mark.parametrize(
    ("utterance", "genre", "field", "value"),
    [
        ("Нужен курс по теме «машинное обучение»; мой уровень — продвинутый.", "машинное обучение", "level", "продвинутый"),
        ("Мне нужен курс «python» по теме с практическими заданиями.", "python", "practical", True),
        ("Хочу изучать тему «машинное обучение» на курсе с практическими заданиями.", "машинное обучение", "practical", True),
        ("Помоги найти курс по теме «python» с практическими заданиями.", "python", "practical", True),
        ("Найди курс на тему «машинное обучение»; уровень продвинутый.", "машинное обучение", "level", "продвинутый"),
    ],
)
def test_quoted_course_topic_is_a_spoken_genre(utterance, genre, field, value):
    query, issue = rule_parse(utterance, Query())

    assert issue is None
    assert query.kind == "course"
    assert query.genre == genre
    assert getattr(query, field) == value


@pytest.mark.parametrize(
    ("utterance", "genre"),
    [
        ("Нужен курс по теме «машинное обучение»; мой уровень — продвинутый.", "машинное обучение"),
        ("Пожалуйста: курс «python» по теме с практическими заданиями.", "python"),
        ("Хочу изучать тему «машинное обучение» на курсе с практическими заданиями.", "машинное обучение"),
    ],
)
def test_adaptive_does_not_ask_for_spoken_quoted_course_topic(utterance, genre):
    agent = Agent(provider=DemoProvider(generate_catalog()), mode="rules", question_policy="adaptive")

    response = agent.chat(ChatRequest(user_id="course-topic-regression", message=utterance))

    assert response.query.genre == genre
    assert response.clarification_slot != "genre"


def test_navigation_word_inside_quoted_title_does_not_reset_previous_constraints():
    previous = Query(kind="film", genre="драма")

    query, issue = rule_parse("Хочу посмотреть «Найти приключения»", previous)

    assert issue is None
    assert query == previous


def test_exact_title_navigation_still_does_not_extract_title_as_genre():
    query, issue = rule_parse("Найди «Python: практический курс»", Query(kind="film", genre="драма"))
    assert issue is None
    assert query.intent == "navigation"
    assert query.seed_title == "Python: практический курс"
    assert query.genre is None
