import pytest

from recagent.agent import Agent
from recagent.catalog import generate_catalog
from recagent.models import ChatRequest, Query
from recagent.parsing import rule_parse


@pytest.mark.parametrize("text", [
    "Хочу детектив и одновременно не детектив",
    "Сериал в жанре комедия, но комедию не предлагать",
])
def test_contradiction_requires_a_question(text):
    previous = Query(kind="film", genre="драма")
    query, issue = rule_parse(text, previous)
    assert issue and "одновременно" in issue
    assert query == previous


def test_later_rejection_replaces_previous_genre():
    query, issue = rule_parse("Только без драмы", Query(kind="film", genre="драма"))
    assert issue is None
    assert query.genre is None
    assert query.excluded_genres == ["драма"]


@pytest.mark.parametrize("text,practical", [
    ("Курс python с заданиями", True),
    ("Курс python с практическими упражнениями", True),
    ("Курс python, не только теория", True),
    ("Курс python без практики", False),
])
def test_course_practice_is_extracted(text, practical):
    query, issue = rule_parse(text, Query())
    assert issue is None
    assert query.practical is practical


def test_not_a_beginner_is_not_beginner_level():
    query, _ = rule_parse("Курс python, я уже не новичок", Query())
    assert query.level == "продвинутый"


def test_named_item_does_not_need_a_format_question():
    item = generate_catalog()[0]
    agent = Agent(mode="rules")
    response = agent.chat(ChatRequest(user_id="new", message=f'Найди «{item.title}»'))
    assert response.state == "recommend"
    assert [rec.item.id for rec in response.recommendations] == [item.id]
    assert response.query.kind == item.kind


def test_words_inside_a_title_do_not_become_filters():
    query, issue = rule_parse('Найди «Мрачный курс детектива»', Query(kind="film", genre="комедия"))
    assert issue is None
    assert query.seed_title == "Мрачный курс детектива"
    assert query.genre is None and query.tone is None and query.kind is None
