"""Invented dialogue controls for pending transitions and session isolation."""

from copy import deepcopy

import pytest

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.models import ChatRequest, Item
from recagent.providers import DemoProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


def update(field, value, source, operation="set"):
    return ConstraintUpdate(field=field, value=value, source_text=source, operation=operation)


class Interpreter:
    def __init__(self, requests):
        self.requests = iter(requests)
        self.calls = 0

    def interpret(self, *_args, **_kwargs):
        self.calls += 1
        return next(self.requests), 2


def agent(*requests):
    records = [
        Item(
            id=key,
            title=key,
            kind=kind,
            genre=genre,
            tone="нейтральный",
            level="начальный" if kind == "course" else None,
            minutes=minutes,
            quality=quality,
            description="Invented state fixture",
        )
        for key, kind, genre, minutes, quality in [
            ("copper", "film", "комедия", 42, 0.9),
            ("linen", "film", "драма", 72, 0.8),
            ("stone", "film", "фантастика", 32, 0.7),
            ("cedar", "course", "python", 105, 0.8),
        ]
    ]
    return WorkflowAgent(
        mode="ollama",
        provider=DemoProvider(records),
        interpreter=Interpreter(requests),
        response_generator=EvidenceResponseGenerator(),
        question_policy="adaptive",
    )


def chat(service, text, sid=None, owner="state-owner"):
    response = service.chat(ChatRequest(user_id=owner, session_id=sid, message=text))
    assert response.mode == "ollama", response.warnings
    return response


def test_pending_exclusions_accumulate_when_another_field_is_being_clarified():
    service = agent(
        StructuredRequest(
            updates=[update("kind", "film", "фильм"), update("genre", "комедия", "без комедии", "exclude")],
            issues=[{"kind": "ambiguity", "field": "tone", "source_text": "особый", "message": "Какой тон?"}],
        ),
        StructuredRequest(updates=[update("genre", "драма", "без драмы", "exclude"), update("tone", "нейтральный", "нейтральный")]),
    )
    first = chat(service, "Фильм без комедии, особый тон")
    assert first.state == "clarify"
    second = chat(service, "Также без драмы, тон нейтральный", first.session_id)
    assert second.state == "recommend"
    assert set(second.query.excluded_genres) == {"комедия", "драма"}
    assert [r.item.id for r in second.recommendations] == ["stone"]


@pytest.mark.parametrize("operation,expected", [("include", {"драма"}), ("clear", set())])
def test_pending_include_is_exact_while_clear_removes_the_whole_field(operation, expected):
    correction = (
        update("genre", "комедия", "комедию", "include") if operation == "include" else update("genre", None, "убрать жанр", "clear")
    )
    service = agent(
        StructuredRequest(
            updates=[
                update("kind", "film", "фильм"),
                update("genre", "комедия", "без комедии", "exclude"),
                update("genre", "драма", "без драмы", "exclude"),
            ],
            issues=[{"kind": "ambiguity", "field": "tone", "source_text": "особый", "message": "Какой тон?"}],
        ),
        StructuredRequest(updates=[correction, update("tone", "нейтральный", "нейтральный")]),
    )
    first = chat(service, "Фильм без комедии и без драмы, особый тон")
    second = chat(service, f"{'Вернуть комедию' if operation == 'include' else 'Убрать жанр'}, тон нейтральный", first.session_id)
    assert second.state == "recommend"
    assert set(second.query.excluded_genres) == expected and second.query.genre is None


def test_scalar_correction_replaces_conflicting_staged_bounds_without_losing_other_conditions():
    service = agent(
        StructuredRequest(
            updates=[
                update("kind", "film", "фильм"),
                update("tone", "нейтральный", "нейтральный"),
                update("max_minutes", 17, "17 минут"),
                update("max_minutes", 43, "43 минуты"),
            ]
        ),
        StructuredRequest(updates=[update("max_minutes", 50, "50 минут")]),
    )
    first = chat(service, "Фильм нейтральный, пределы одновременно 17 минут и 43 минуты")
    assert first.state == "clarify"
    second = chat(service, "Исправлю предел на 50 минут", first.session_id)
    assert second.state == "recommend" and second.query.max_minutes == 50
    state = service.sessions[second.session_id].constraint_state
    assert [c.value for c in state.constraints if c.field == "max_minutes"] == [50]
    assert second.query.tone == "нейтральный" and second.query.kind == "film"


def test_unrelated_field_does_not_erase_pending_adapter_scalar_conflict():
    service = agent(
        StructuredRequest(
            updates=[update("kind", "film", "фильм"), update("max_minutes", 17, "17 минут"), update("max_minutes", 43, "43 минуты")]
        ),
        StructuredRequest(updates=[update("tone", "нейтральный", "нейтральный")]),
        StructuredRequest(updates=[update("max_minutes", 50, "50 минут")]),
    )
    first = chat(service, "Фильм, предел одновременно 17 минут и 43 минуты")
    second = chat(service, "Тон нейтральный", first.session_id)
    state = service.sessions[first.session_id].constraint_state
    assert second.state == "clarify" and state.version == 0 and not state.constraints
    assert any(f.code == "adapter_conflict" for f in state.pending.findings)
    third = chat(service, "Предел 50 минут", first.session_id)
    assert third.state == "recommend" and third.query.max_minutes == 50 and third.query.tone == "нейтральный"


def test_new_exclusion_does_not_resolve_existing_positive_negative_conflict():
    service = agent(
        StructuredRequest(
            updates=[
                update("kind", "film", "фильм"),
                update("genre", "комедия", "комедия"),
                update("genre", "комедия", "без комедии", "exclude"),
            ]
        ),
        StructuredRequest(updates=[update("genre", "драма", "без драмы", "exclude")]),
        StructuredRequest(updates=[update("genre", "комедия", "комедию", "include")]),
    )
    first = chat(service, "Фильм, комедия, но без комедии")
    second = chat(service, "И без драмы", first.session_id)
    assert first.state == second.state == "clarify"
    assert service.sessions[first.session_id].constraint_state.version == 0
    third = chat(service, "Разрешить комедию", first.session_id)
    assert third.state == "recommend" and third.query.genre == "комедия"
    assert third.query.excluded_genres == ["драма"]


def test_unverified_pending_value_cannot_be_committed_by_adding_a_different_exclusion():
    service = agent(
        StructuredRequest(updates=[update("kind", "film", "фильм"), update("genre", "неизвестный жанр", "неизвестный жанр")]),
        StructuredRequest(updates=[update("genre", "комедия", "без комедии", "exclude")]),
    )
    first = chat(service, "Фильм, неизвестный жанр")
    second = chat(service, "И без комедии", first.session_id)
    state = service.sessions[first.session_id].constraint_state
    assert first.state == second.state == "clarify"
    assert state.version == 0 and state.constraints == () and state.pending is not None


def test_domain_switch_with_unresolved_is_atomic_then_drops_old_domain_limits():
    service = agent(
        StructuredRequest(
            updates=[update("kind", "film", "фильм"), update("genre", "комедия", "комедия"), update("max_minutes", 50, "50 минут")]
        ),
        StructuredRequest(
            updates=[update("kind", "course", "курс"), update("genre", "python", "Python")],
            issues=[{"kind": "ambiguity", "field": "level", "source_text": "особый", "message": "Какой уровень?"}],
        ),
        StructuredRequest(updates=[update("level", "начальный", "начальный")]),
    )
    first = chat(service, "Фильм комедия до 50 минут")
    before = deepcopy(service.sessions[first.session_id].constraint_state)
    second = chat(service, "Теперь курс Python, особый уровень", first.session_id)
    pending = service.sessions[first.session_id].constraint_state
    assert second.state == "clarify" and second.query == first.query
    assert pending.constraints == before.constraints and pending.version == before.version
    third = chat(service, "Тогда начальный", first.session_id)
    assert third.state == "recommend" and third.query.kind == "course" and third.query.genre == "python"
    assert third.query.max_minutes is None and third.query.excluded_genres == []
    assert service.sessions[first.session_id].constraint_state.version == before.version + 1


def test_new_exclusion_does_not_answer_an_explicit_unknown_preference_but_clear_does():
    service = agent(
        StructuredRequest(
            updates=[update("kind", "film", "фильм")],
            issues=[{"kind": "unsupported_constraint", "field": "genre", "source_text": "особый", "message": "Какой жанр?"}],
        ),
        StructuredRequest(updates=[update("genre", "комедия", "без комедии", "exclude")]),
        StructuredRequest(updates=[update("genre", None, "убрать жанр", "clear"), update("tone", "нейтральный", "нейтральный")]),
    )
    first = chat(service, "Фильм, особый жанр")
    second = chat(service, "Также без комедии", first.session_id)
    assert second.state == "clarify" and service.sessions[first.session_id].constraint_state.version == 0
    third = chat(service, "Убрать жанр, тон нейтральный", first.session_id)
    assert third.state == "recommend" and third.query.genre is None and third.query.excluded_genres == []


def test_pending_numeric_operators_remain_independent_until_explicit_clear():
    from recagent.contracts import Constraint, ProposedChange
    from recagent.state import apply_changes, merge_pending_changes

    def change(key, op, value, operation="add"):
        return ProposedChange(
            id=key,
            operation=operation,
            constraint=Constraint(id=key, field="price", op=op, value=value, turn_id=key, domain_version="device"),
        )

    lower, upper = change("lower", "gte", 10), change("upper", "lte", 100)
    correction = change("new-upper", "lte", 80)
    merged = merge_pending_changes((lower, upper), (correction,))
    assert {(c.op, c.value) for c in apply_changes((), merged)} == {("gte", 10), ("lte", 80)}
    clear = change("clear", "eq", None, "clear")
    assert apply_changes((), merge_pending_changes(merged, (clear,))) == ()


def test_foreign_session_and_feedback_cannot_mutate_owner_or_consume_interpreter():
    service = agent(StructuredRequest(updates=[update("kind", "film", "фильм"), update("genre", "комедия", "комедия")]))
    first = chat(service, "Фильм комедия")
    session = service.sessions[first.session_id]
    before = deepcopy(session.constraint_state)
    with pytest.raises(KeyError):
        chat(service, "Изменить", first.session_id, owner="different-owner")
    with pytest.raises(KeyError):
        service.feedback(first.session_id, "copper", "like", user_id="different-owner")
    with pytest.raises(ValueError):
        service.feedback(first.session_id, "not-shown", "like", user_id="state-owner")
    assert session.constraint_state == before and session.reactions == {} and service.interpreter.calls == 1
    service.feedback(first.session_id, "copper", "like", user_id="state-owner")
    assert session.reactions == {"copper": "like"}
