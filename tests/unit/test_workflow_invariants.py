"""State/evidence contracts authored independently of evaluation datasets.

The interpreter is a controlled dependency: these tests exercise what the
workflow may commit, not how a particular model parses benchmark utterances.
"""

import pytest

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest, Item, Query
from recagent.providers import DemoProvider
from recagent.request_mapping import SchemaRequestAdapter
from recagent.workflow import WorkflowAgent


class ScriptedInterpreter:
    def __init__(self, *requests):
        self.requests = iter(requests)
        self.contexts = []

    def interpret(self, message, previous, **context):
        del message, previous
        self.contexts.append(context)
        return next(self.requests), 1


def update(field, value, source, operation="set"):
    return ConstraintUpdate(field=field, value=value, source_text=source, operation=operation)


def make_agent(*requests, adapter=None):
    common = {"quality": 0.5, "description": "Independent contract fixture"}
    provider = DemoProvider(
        items=[
            Item(id="series-compact", title="A", kind="series", genre="комедия", minutes=16, seasons=2, tone="лёгкий", **common),
            Item(id="series-long", title="B", kind="series", genre="комедия", minutes=41, seasons=2, tone="лёгкий", **common),
            Item(id="film-comedy", title="C", kind="film", genre="комедия", minutes=70, tone="лёгкий", **common),
            Item(id="film-drama", title="D", kind="film", genre="драма", minutes=85, tone="мрачный", **common),
            Item(id="course-python", title="E", kind="course", genre="python", minutes=90, level="продвинутый", **common),
        ]
    )
    return WorkflowAgent(mode="ollama", provider=provider, interpreter=ScriptedInterpreter(*requests), request_adapter=adapter)


def chat(agent, message, session_id=None):
    response = agent.chat(ChatRequest(user_id="independent-contract", message=message, session_id=session_id))
    assert response.mode == "ollama", response.warnings
    return response


def canonical_state(agent, response):
    return agent.sessions[response.session_id].constraint_state


def test_fabricated_quote_on_an_active_field_is_not_reclassified_as_trusted_history():
    agent = make_agent(
        StructuredRequest(updates=[update("kind", "series", "сериал"), update("genre", "комедия", "комедия")]),
        StructuredRequest(updates=[update("genre", "драма", "драма")]),
    )
    first = chat(agent, "Нужен сериал, жанр комедия")
    before = canonical_state(agent, first)
    second = chat(agent, "Уточню позже", first.session_id)
    after = canonical_state(agent, second)

    assert second.state == "clarify"
    assert second.query.genre == "комедия"
    assert after.constraints == before.constraints
    assert after.pending is not None
    assert any(finding.code == "evidence_not_in_turn" for finding in after.pending.findings)
    fabricated = next(change for change in after.pending.proposal.changes if change.constraint.field == "genre")
    assert all(span.origin == "current" for span in fabricated.source_spans)


def test_scalar_correction_preserves_other_fields_and_current_turn_provenance():
    agent = make_agent(
        StructuredRequest(
            updates=[
                update("kind", "series", "сериал"),
                update("genre", "комедия", "комедия"),
                update("max_minutes", 17, "17 минут"),
            ]
        ),
        StructuredRequest(updates=[update("max_minutes", 43, "43 минуты")]),
    )
    first = chat(agent, "сериал, комедия, максимум 17 минут")
    before = canonical_state(agent, first)
    second = chat(agent, "Поменяйте предел на 43 минуты", first.session_id)
    after = canonical_state(agent, second)

    assert second.state == "recommend"
    assert (second.query.kind, second.query.genre, second.query.max_minutes) == ("series", "комедия", 43)
    untouched = tuple(constraint for constraint in before.constraints if constraint.field in {"kind", "genre"})
    assert tuple(constraint for constraint in after.constraints if constraint.field in {"kind", "genre"}) == untouched
    bounds = [constraint for constraint in after.constraints if constraint.field in {"max_minutes", "minutes"}]
    assert len(bounds) == 1
    assert bounds[0].value == 43 and bounds[0].op == "lte"
    assert bounds[0].source_spans
    assert all(span.text == "43 минуты" and span.origin == "current" for span in bounds[0].source_spans)
    assert bounds[0].turn_id not in {constraint.turn_id for constraint in untouched}
    assert {item.item.id for item in second.recommendations} == {"series-compact", "series-long"}


@pytest.mark.parametrize(
    ("field", "value", "negative", "positive"),
    [("genre", "комедия", "без комедии", "комедия"), ("tone", "мрачный", "без мрачного", "мрачный")],
)
def test_include_removes_only_exclusion_and_does_not_create_a_positive_requirement(field, value, negative, positive):
    agent = make_agent(
        StructuredRequest(updates=[update("kind", "film", "фильм"), update(field, value, negative, "exclude")]),
        StructuredRequest(updates=[update(field, value, positive, "include")]),
    )
    first = chat(agent, f"Нужен фильм {negative}")
    assert any(constraint.field == field and constraint.op == "neq" for constraint in canonical_state(agent, first).constraints)
    second = chat(agent, f"{positive} снова допускается", first.session_id)

    assert not any(constraint.field == field for constraint in canonical_state(agent, second).constraints)
    assert second.query.kind == "film"
    assert getattr(second.query, field) is None
    if field == "genre":
        assert second.query.excluded_genres == []


def test_domain_switch_clears_prior_domain_limits_in_query_and_canonical_state():
    agent = make_agent(
        StructuredRequest(
            updates=[
                update("kind", "series", "сериал"),
                update("genre", "комедия", "комедия"),
                update("max_seasons", 2, "2 сезона"),
                update("max_minutes", 17, "17 минут"),
            ]
        ),
        StructuredRequest(
            updates=[
                update("kind", "course", "курс"),
                update("genre", "python", "Python"),
                update("level", "продвинутый", "продвинутый"),
            ]
        ),
    )
    first = chat(agent, "сериал, комедия, до 2 сезона и до 17 минут")
    second = chat(agent, "Теперь нужен продвинутый курс Python", first.session_id)

    assert second.state == "recommend"
    assert (second.query.kind, second.query.genre, second.query.level) == ("course", "python", "продвинутый")
    assert second.query.max_seasons is None and second.query.max_minutes is None
    assert {constraint.field for constraint in canonical_state(agent, second).constraints} == {"kind", "genre", "level"}
    assert [recommendation.item.id for recommendation in second.recommendations] == ["course-python"]


@pytest.mark.parametrize("reverse", [False, True])
def test_two_different_scalar_sets_in_one_turn_cannot_choose_a_winner_by_order(reverse):
    conflicting = [update("max_minutes", 17, "17 минут"), update("max_minutes", 43, "43 минуты")]
    if reverse:
        conflicting.reverse()
    agent = make_agent(
        StructuredRequest(updates=[update("kind", "series", "сериал"), update("genre", "комедия", "комедия")]),
        StructuredRequest(updates=conflicting),
    )
    first = chat(agent, "сериал, комедия")
    before = canonical_state(agent, first)
    second = chat(agent, "Лимит одновременно 17 минут и 43 минуты", first.session_id)
    after = canonical_state(agent, second)

    assert second.state == "clarify"
    assert after.constraints == before.constraints
    assert second.query.max_minutes is None
    assert after.pending is not None
    assert any(finding.field == "max_minutes" and "conflict" in finding.code for finding in after.pending.findings)


@pytest.mark.parametrize("proposed_value", ["python", None])
def test_schema_implied_value_uses_the_validated_and_normalized_source(proposed_value):
    agent = make_agent(StructuredRequest(updates=[update("genre", proposed_value, "Python")]))
    response = chat(agent, "Python")
    constraints = canonical_state(agent, response).constraints

    assert response.state == "recommend"
    assert (response.query.kind, response.query.genre) == ("course", "python")
    assert {(constraint.field, constraint.op, constraint.value) for constraint in constraints} == {
        ("kind", "eq", "course"),
        ("genre", "eq", "python"),
    }
    assert all(
        recommendation.item.kind == "course" and recommendation.item.genre == "python" for recommendation in response.recommendations
    )


@pytest.mark.parametrize(
    ("proposed_value", "quote", "message"),
    [
        ("python", "Python", "Сюрприз"),
        (None, "Python", "Сюрприз"),
        ("python", "неизвестная тема", "неизвестная тема"),
        ("python", "драма", "драма"),
    ],
)
def test_schema_implication_cannot_activate_from_fabricated_or_unsupported_source(proposed_value, quote, message):
    agent = make_agent(StructuredRequest(updates=[update("genre", proposed_value, quote)]))
    response = chat(agent, message)

    assert response.state == "clarify"
    assert response.query.kind is None and response.query.genre is None
    assert canonical_state(agent, response).constraints == ()
    assert response.recommendations == []


@pytest.mark.parametrize("proposed_value", ["драма", None])
def test_implication_follows_injected_schema_instead_of_a_demo_topic_rule(proposed_value):
    adapter = SchemaRequestAdapter(Query, domain_field="kind", implied_values={"genre": {"драма": {"kind": "film"}}})
    agent = make_agent(StructuredRequest(updates=[update("genre", proposed_value, "драма")]), adapter=adapter)
    response = chat(agent, "драма")

    assert response.state == "recommend"
    assert response.query.kind == "film" and response.query.genre == "драма"
    assert {(constraint.field, constraint.value) for constraint in canonical_state(agent, response).constraints} == {
        ("genre", "драма"),
        ("kind", "film"),
    }
    assert [recommendation.item.id for recommendation in response.recommendations] == ["film-drama"]


def test_pending_domain_switch_drops_old_limits_after_a_valid_targeted_recovery():
    agent = make_agent(
        StructuredRequest(
            updates=[
                update("kind", "series", "сериал"),
                update("genre", "комедия", "комедия"),
                update("max_minutes", 17, "17 минут"),
            ]
        ),
        StructuredRequest(
            updates=[
                update("kind", "course", "курс"),
                update("genre", "python", "Python"),
                update("level", "продвинутый", "стажёрский"),
            ]
        ),
        StructuredRequest(updates=[update("level", "продвинутый", "продвинутый")]),
    )
    first = chat(agent, "сериал, комедия, до 17 минут")
    active_before = canonical_state(agent, first).constraints
    second = chat(agent, "Теперь курс Python, уровень стажёрский", first.session_id)
    assert second.state == "clarify"
    assert canonical_state(agent, second).constraints == active_before

    third = chat(agent, "Выбираю продвинутый", first.session_id)
    recovered = canonical_state(agent, third)
    assert third.state == "recommend"
    assert third.query.kind == "course" and third.query.genre == "python"
    assert third.query.level == "продвинутый" and third.query.max_minutes is None
    assert recovered.pending is None
    assert {constraint.field for constraint in recovered.constraints} == {"kind", "genre", "level"}
    assert [recommendation.item.id for recommendation in third.recommendations] == ["course-python"]


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_domain_cannot_contradict_a_schema_declared_implication(reverse):
    updates = [update("kind", "film", "фильм"), update("genre", "python", "Python")]
    if reverse:
        updates.reverse()
    agent = make_agent(StructuredRequest(updates=updates))
    response = chat(agent, "Нужен фильм про Python")
    state = canonical_state(agent, response)

    assert response.state == "clarify"
    assert response.recommendations == []
    assert state.constraints == ()
    assert state.pending is not None
    assert any("conflict" in finding.code for finding in state.pending.findings)


def test_correcting_one_of_two_pending_blockers_does_not_accept_the_other():
    agent = make_agent(
        StructuredRequest(
            updates=[
                update("kind", "course", "курс"),
                update("genre", "python", "Python"),
                update("level", "продвинутый", "стажёрский"),
                update("practical", True, "лабораторный формат"),
            ]
        ),
        StructuredRequest(updates=[update("level", "продвинутый", "продвинутый")]),
        StructuredRequest(updates=[update("practical", False, "только теория")]),
    )
    first = chat(agent, "курс Python: стажёрский уровень, лабораторный формат")
    assert first.state == "clarify"
    first_state = canonical_state(agent, first)
    assert first_state.pending is not None
    assert {finding.field for finding in first_state.pending.findings} >= {"level", "practical"}

    second = chat(agent, "Пусть уровень продвинутый", first.session_id)
    second_state = canonical_state(agent, second)
    assert second.state == "clarify" and second.recommendations == []
    assert second_state.constraints == ()
    assert second_state.pending is not None
    assert any(finding.field == "practical" for finding in second_state.pending.findings)
    assert not any(finding.field == "level" and finding.status != "pass" for finding in second_state.pending.findings)

    third = chat(agent, "Тогда только теория", first.session_id)
    assert third.state != "clarify"
    assert canonical_state(agent, third).pending is None
    assert (third.query.kind, third.query.genre, third.query.level, third.query.practical) == ("course", "python", "продвинутый", False)


@pytest.mark.parametrize(
    ("field", "value", "source"),
    [("unlisted_feature", "x", "вариант X"), ("level", "продвинутый", "стажёрский")],
)
def test_unknown_pending_update_is_not_exposed_as_trusted_staged_context(field, value, source):
    agent = make_agent(
        StructuredRequest(updates=[update("kind", "course", "курс"), update("genre", "python", "Python"), update(field, value, source)]),
        StructuredRequest(updates=[update("kind", "course", "курс")]),
    )
    first = chat(agent, f"курс Python, {source}")
    assert first.state == "clarify"
    second = chat(agent, "Подтверждаю курс", first.session_id)
    assert second.state == "clarify"
    context = agent.interpreter.contexts[1]["pending_context"]

    assert context is not None
    trusted = {(constraint["field"], constraint["value"]) for constraint in context["trusted_staged"]}
    assert trusted >= {("kind", "course"), ("genre", "python")}
    assert all(constraint["field"] != field for constraint in context["trusted_staged"])
    assert context["trusted_active"] == []
    assert any(blocker["field"] == field for blocker in context["blockers"])
