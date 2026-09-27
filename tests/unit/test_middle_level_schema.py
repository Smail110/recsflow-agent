"""Средний уровень курса представлен в запросе и синтетическом каталоге."""

import pytest

from recagent.catalog import generate_catalog
from recagent.domains.demo import request_adapter
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import Query
from recagent.providers import DemoProvider
from recagent.questions import choose_question


@pytest.mark.parametrize(
    ("message", "source"),
    [
        ("Нужен курс среднего уровня.", "среднего уровня"),
        ("Нужен средний курс.", "средний"),
    ],
)
def test_explicit_medium_course_level_filters_catalog(message, source):
    request = StructuredRequest(updates=[ConstraintUpdate(field="level", value="средний", source_text=source)])
    query, issues = request_adapter().apply(request, Query(kind="course"), message)

    assert not issues
    assert query.level == "средний"

    items = generate_catalog(seed=42, size=600)
    provider = DemoProvider(items)
    retrieved = provider.retrieve("test", query)
    assert retrieved
    assert all(item.kind == "course" and item.level == "средний" for item in provider.lookup(retrieved))


def test_middle_is_not_an_unconditional_alias_for_medium():
    request = StructuredRequest(updates=[ConstraintUpdate(field="level", value="средний", source_text="middle-разработчика")])
    query, issues = request_adapter().apply(request, Query(kind="course"), "Нужен курс для middle-разработчика.")

    assert query.level is None
    assert any(issue.kind == "unsupported_constraint" and issue.field == "level" for issue in issues)


def test_level_questions_offer_all_catalog_levels():
    adapter = request_adapter()
    question = choose_question(
        Query(kind="course"),
        [item for item in generate_catalog(seed=42, size=600) if item.kind == "course"],
        policy="fixed",
        skipped={"genre", "practical"},
    )

    assert question is not None and question.slot == "level"
    for level in ("начальный", "средний", "продвинутый"):
        assert level in question.message
    for level in ("начального", "среднего", "продвинутого"):
        assert level in adapter.field_questions["level"]
