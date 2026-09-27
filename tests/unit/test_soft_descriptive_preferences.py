"""Descriptive wishes are soft ranking hints, not invented catalog facts."""

import json
from uuid import uuid4

import pytest

from recagent.domains.demo import request_adapter
from recagent.interpretation import ConstraintUpdate, InterpretationIssue, StructuredRequest
from recagent.models import ChatRequest, Item
from recagent.observability.catalog_gaps import CatalogGapRecorder
from recagent.providers import DemoProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent, _declines_optional_preference


class Interpreter:
    def __init__(self, request):
        self.request = request
        self.calls = 0

    def interpret(self, *args, **kwargs):
        self.calls += 1
        return self.request, 0


def update(field, value, source, operation="set"):
    return ConstraintUpdate(field=field, value=value, source_text=source, operation=operation)


def agent_for(request, *, policy="none", items=None, gap_recorder=None):
    default_items = [
        Item(id="drama", title="Drama", kind="film", genre="драма", quality=0.95, description="Family drama"),
        Item(id="space", title="Orbital film", kind="film", genre="фантастика", quality=0.4, description="Space journey"),
        *[
            Item(id=f"extra-{index}", title=f"Extra film {index}", kind="film", genre="драма", quality=0.2, description="Another story")
            for index in range(6)
        ],
    ]
    interpreter = Interpreter(request)
    agent = WorkflowAgent(
        mode="ollama",
        provider=DemoProvider(items=default_items if items is None else items),
        interpreter=interpreter,
        response_generator=EvidenceResponseGenerator(),
        question_policy=policy,
        catalog_gap_recorder=gap_recorder,
    )
    return agent, interpreter


def test_descriptive_topic_ranks_a_close_genre_without_claiming_a_hard_constraint():
    agent, _ = agent_for(
        StructuredRequest(
            updates=[
                update("kind", "film", "фильм"),
                update("genre", "фантастика", "космический"),
                update("tone", "мрачный", "необычный"),
            ]
        ),
        policy="adaptive",
    )
    response = agent.chat(ChatRequest(user_id="soft-topic", message="Хочу какой-нибудь такой фильм космический необычный"))
    assert response.state == "recommend" and response.recommendations[0].item.id == "space"
    assert response.query.kind == "film" and response.query.genre is None and response.query.tone is None
    assert {constraint.field for constraint in agent.sessions[response.session_id].constraint_state.constraints} == {"kind"}
    assert "космический" in response.message and "приблизительный" in response.message


@pytest.mark.parametrize("message, source", [
    ("Хочу фильм только космический", "космический"),
    ("Хочу фильм жанр космический", "космический"),
    ("Мне нужен обязательно какой-нибудь фильм такой очень космический", "космический"),
    ("Хочу фильм космический, он должен быть про космос", "космический"),
    ("Хочу фильм космический", "космический"),
    ("Хочу фильм боевик", "боевик"),
    ("Хочу фильм киберпанк", "киберпанк"),
    ("Хочу какой-нибудь интересный боевик", "интересный боевик"),
])
def test_explicit_unknown_requirement_is_not_downgraded_to_a_soft_hint(message, source):
    agent, _ = agent_for(
        StructuredRequest(updates=[update("kind", "film", "фильм"), update("genre", "фантастика", source)])
    )
    response = agent.chat(ChatRequest(user_id="hard-topic", message=message))
    assert response.state == "clarify" and not response.recommendations
    assert not agent.sessions[response.session_id].constraint_state.constraints


def test_unknown_topic_and_origin_get_a_specific_catalog_limitation():
    agent, interpreter = agent_for(StructuredRequest(updates=[
        update("genre", "корейский", "корейский фильм"),
        update("kind", "film", "фильм"),
        update("genre", "зомби", "про зомби"),
    ]))
    response = agent.chat(ChatRequest(
        user_id="unknown-topic-origin",
        message="Хочу корейский фильм который будет интересный про зомби",
    ))
    assert response.state == "no_results" and not response.recommendations
    assert "корейский фильм" in response.message and "про зомби" in response.message
    assert "Какой жанр" not in response.message
    assert response.query.genre is None
    assert agent.sessions[response.session_id].constraint_state.pending is None

    interpreter.request = StructuredRequest(updates=[update("genre", "комедия", "комедия")])
    follow_up = agent.chat(ChatRequest(user_id="unknown-topic-origin", session_id=response.session_id, message="комедия"))
    assert follow_up.state == "clarify" and not follow_up.recommendations


def test_catalog_gap_log_records_cited_wishes_once_but_not_replay(tmp_path):
    path = tmp_path / "gaps.jsonl"
    agent, _ = agent_for(StructuredRequest(updates=[
        update("genre", "корейский", "корейский фильм"),
        update("kind", "film", "фильм"),
        update("genre", "зомби", "про зомби"),
    ]), gap_recorder=CatalogGapRecorder(path))
    request = ChatRequest(
        user_id="gap-test", message_id=uuid4(),
        message="Хочу корейский фильм который будет интересный про зомби",
    )
    response = agent.chat(request)
    assert response.state == "no_results"
    agent.chat(request)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["response_state"] == "no_results"
    assert {entry["wish"] for entry in events[0]["wishes"]} == {"корейский фильм", "про зомби"}
    assert "gap-test" not in path.read_text(encoding="utf-8")
    assert request.message not in path.read_text(encoding="utf-8")


def test_unsupported_new_request_does_not_reuse_an_old_format():
    agent, interpreter = agent_for(StructuredRequest(updates=[update("kind", "film", "фильм")]))
    first = agent.chat(ChatRequest(user_id="old-format", message="Хочу фильм"))
    assert first.state == "recommend" and first.query.kind == "film"

    interpreter.request = StructuredRequest(updates=[
        update("genre", "корейский", "корейский фильм"),
        update("kind", "film", "фильм"),
        update("genre", "зомби", "про зомби"),
    ])
    blocked = agent.chat(ChatRequest(
        user_id="old-format", session_id=first.session_id,
        message="Хочу корейский фильм который будет интересный про зомби",
    ))
    assert blocked.state == "no_results" and blocked.query.kind == "film"
    assert {c.field for c in agent.sessions[first.session_id].constraint_state.constraints} == {"kind"}
    assert agent.sessions[first.session_id].restart_on_next_turn
    agent.feedback(first.session_id, first.recommendations[0].item.id, "like", user_id="old-format")

    interpreter.request = StructuredRequest(updates=[update("genre", "комедия", "комедия")])
    follow_up = agent.chat(ChatRequest(user_id="old-format", session_id=first.session_id, message="комедия"))
    assert follow_up.state == "clarify" and not follow_up.recommendations and follow_up.query.kind is None


def test_unsupported_wishes_in_different_fields_can_be_resolved_separately():
    agent, interpreter = agent_for(StructuredRequest(updates=[
        update("kind", "film", "фильм"),
        update("genre", "киберпанк", "киберпанк"),
        update("tone", "яркий", "яркий"),
    ]))
    first = agent.chat(ChatRequest(user_id="separate-fields", message="Хочу фильм киберпанк яркий"))
    assert first.state == "clarify" and not first.recommendations
    assert not agent.sessions[first.session_id].restart_on_next_turn

    interpreter.request = StructuredRequest(updates=[update("genre", "комедия", "комедия")])
    second = agent.chat(ChatRequest(user_id="separate-fields", session_id=first.session_id, message="комедия"))
    assert second.state == "clarify" and not second.recommendations


def test_unsupported_wishes_across_turns_do_not_disappear_together():
    agent, interpreter = agent_for(StructuredRequest(updates=[
        update("kind", "film", "фильм"), update("genre", "корейский", "корейский фильм"),
    ]))
    first = agent.chat(ChatRequest(user_id="cross-turn-unknown", message="Хочу корейский фильм"))
    assert first.state == "clarify"

    interpreter.request = StructuredRequest(updates=[update("genre", "зомби", "про зомби")])
    second = agent.chat(ChatRequest(user_id="cross-turn-unknown", session_id=first.session_id, message="про зомби"))
    assert second.state == "no_results" and not second.recommendations
    assert "корейский фильм" in second.message and "про зомби" in second.message

    interpreter.request = StructuredRequest(updates=[update("genre", "комедия", "комедия")])
    third = agent.chat(ChatRequest(user_id="cross-turn-unknown", session_id=first.session_id, message="комедия"))
    assert third.state == "clarify" and not third.recommendations


def test_model_issue_for_an_uncertain_description_does_not_force_genre_question():
    agent, _ = agent_for(StructuredRequest(
        updates=[update("kind", "film", "фильм")],
        issues=[InterpretationIssue(
            kind="unsupported_constraint", field="genre", value="космический",
            message="Неизвестная тема", source_text="космический",
        )],
        clarification_required=True,
    ), policy="adaptive")
    response = agent.chat(ChatRequest(user_id="issue-topic", message="Хочу какой-нибудь такой фильм космический"))
    assert response.state == "recommend" and response.query.genre is None
    assert "космический" in response.message and "приблизительный" in response.message


@pytest.mark.parametrize("message, source", [
    ("Хочу фильм боевик", "боевик"),
    ("Хочу какой-нибудь фильм киберпанк", "киберпанк"),
    ("Хочу какой-нибудь страшный ужастик", "страшный ужастик"),
])
def test_unknown_named_genre_issue_remains_blocking(message, source):
    agent, _ = agent_for(StructuredRequest(
        updates=[update("kind", "film", "фильм")],
        issues=[InterpretationIssue(
            kind="unsupported_constraint", field="genre", value=source,
            message="Неизвестный жанр", source_text=source,
        )],
        clarification_required=True,
    ))
    response = agent.chat(ChatRequest(user_id="named-genre", message=message))
    assert response.state == "clarify" and not response.recommendations


def test_explicit_genre_replaces_prior_soft_topic_for_more():
    agent, interpreter = agent_for(StructuredRequest(
        updates=[update("kind", "film", "фильм"), update("genre", "фантастика", "космический")]
    ))
    first = agent.chat(ChatRequest(user_id="soft-replace", message="Хочу какой-нибудь такой фильм космический"))
    assert agent.sessions[first.session_id].soft_preferences
    interpreter.request = StructuredRequest(updates=[update("genre", "драма", "драму")])
    agent.chat(ChatRequest(user_id="soft-replace", session_id=first.session_id, message="Лучше драму"))
    assert agent.sessions[first.session_id].soft_preferences == []


def test_any_genre_retracts_prior_soft_topic():
    agent, interpreter = agent_for(StructuredRequest(
        updates=[update("kind", "film", "фильм"), update("genre", "фантастика", "космический")]
    ))
    first = agent.chat(ChatRequest(user_id="soft-any", message="Хочу какой-нибудь такой фильм космический"))
    assert agent.sessions[first.session_id].soft_preferences
    interpreter.request = StructuredRequest(updates=[update("genre", None, "жанр любой", operation="clear")])
    agent.chat(ChatRequest(user_id="soft-any", session_id=first.session_id, message="Жанр любой"))
    assert agent.sessions[first.session_id].soft_preferences == []


def test_any_genre_noop_returns_no_results_without_leaking_internal_usage():
    agent, interpreter = agent_for(StructuredRequest(
        updates=[update("kind", "film", "фильм"), update("genre", "фантастика", "космический")]
    ), items=[])
    first = agent.chat(ChatRequest(user_id="soft-empty", message="Хочу какой-нибудь такой фильм космический"))
    assert first.state == "no_results"
    interpreter.request = StructuredRequest(updates=[update("genre", None, "жанр любой", operation="clear")])
    second = agent.chat(ChatRequest(user_id="soft-empty", session_id=first.session_id, message="Жанр любой"))
    assert second.state == "no_results"
    assert agent.sessions[first.session_id].soft_preferences == []
    assert "noop_preference_fields" not in second.llm_usage


def test_soft_description_survives_more_without_new_llm_extraction():
    agent, interpreter = agent_for(StructuredRequest(
        updates=[update("kind", "film", "фильм"), update("genre", "фантастика", "космический")]
    ))
    first = agent.chat(ChatRequest(user_id="soft-more", message="Хочу какой-нибудь такой фильм космический"))
    second = agent.chat(ChatRequest(user_id="soft-more", session_id=first.session_id, message="ещё"))
    assert first.state == second.state == "recommend"
    assert "космический" in second.message and "приблизительный" in second.message
    assert interpreter.calls == 1


def test_soft_only_followup_answers_an_optional_question():
    agent, interpreter = agent_for(StructuredRequest(updates=[update("kind", "film", "фильм")]), policy="adaptive")
    first = agent.chat(ChatRequest(user_id="soft-followup", message="Хочу фильм"))
    assert first.state == "clarify"
    interpreter.request = StructuredRequest(updates=[update("genre", "фантастика", "космическое")])
    second = agent.chat(ChatRequest(user_id="soft-followup", session_id=first.session_id, message="Что-нибудь космическое"))
    assert second.state == "recommend" and second.query.kind == "film" and second.query.genre is None
    assert "космическое" in second.message and "приблизительный" in second.message


def test_funny_description_asks_for_format_instead_of_repeating_mood_and_then_ranks_comedy():
    items = [
        Item(id="serious", title="Serious", kind="film", genre="драма", tone="лёгкий", quality=0.9, description="Drama"),
        Item(id="funny", title="Funny", kind="film", genre="комедия", tone="лёгкий", quality=0.3, description="Comedy"),
    ]
    agent, interpreter = agent_for(
        StructuredRequest(updates=[update("tone", "лёгкий", "интересное смешное")]), items=items
    )
    first = agent.chat(ChatRequest(user_id="funny-format", message="Хочу посмотреть что то такое интересное смешное"))
    assert first.state == "clarify" and first.clarification_slot == "kind"
    assert "формат" not in first.message and "настроение" not in first.message
    assert "фильм" in first.message and "сериал" in first.message
    assert "курс" not in first.message
    interpreter.request = StructuredRequest(intent="navigation")
    second = agent.chat(ChatRequest(user_id="funny-format", session_id=first.session_id, message="Фильм"))
    assert second.state == "recommend" and second.query.kind == "film" and second.query.tone is None
    assert second.recommendations[0].item.id == "funny"
    assert "приблизительный" in second.message
    assert interpreter.calls == 1


def test_explicit_format_missing_from_llm_patch_is_recovered_before_recommendation():
    items = [
        Item(id="serious", title="Serious", kind="film", genre="драма", tone="лёгкий", quality=0.9, description="Drama"),
        Item(id="funny", title="Funny", kind="film", genre="комедия", tone="лёгкий", quality=0.3, description="Comedy"),
    ]
    agent, _ = agent_for(StructuredRequest(updates=[update("tone", "лёгкий", "веселый")]), items=items)
    response = agent.chat(ChatRequest(user_id="explicit-format", message="Хочу веселый фильм"))
    assert response.state == "recommend" and response.query.kind == "film"
    assert response.query.tone is None
    assert response.recommendations[0].item.id == "funny"
    assert "приблизительный" in response.message


@pytest.mark.parametrize("message,expected", [
    ("Хочу веселый фильм", "film"),
    ("Хочу сериал для вечера", "series"),
    ("Нужен курс по Python", "course"),
    ("Хочу фильмы", "film"),
    ("Нужен кинозал", None),
    ("Не хочу фильм", None),
    ("Хочу фильм или сериал", None),
    ("Посоветуй сериал похожий на фильм", None),
])
def test_explicit_format_completion_requires_one_unnegated_declared_value(message, expected):
    recovered = request_adapter().recover_explicit_domain(StructuredRequest(intent="discovery"), message)
    assert (recovered.value if recovered else None) == expected


def test_explicit_format_completion_respects_model_update_and_issue():
    adapter = request_adapter()
    update_present = StructuredRequest(updates=[update("kind", "series", "сериал")])
    assert adapter.recover_explicit_domain(update_present, "Хочу фильм и сериал") is None
    unresolved = StructuredRequest(issues=[InterpretationIssue(
        kind="ambiguity", field="kind", value="фильм", message="Неясный формат", source_text="фильм",
    )])
    assert adapter.recover_explicit_domain(unresolved, "Хочу фильм") is None


def test_strict_unhedged_funny_request_is_not_softened():
    agent, _ = agent_for(StructuredRequest(updates=[
        update("kind", "film", "фильм"), update("tone", "лёгкий", "веселый"),
    ]))
    response = agent.chat(ChatRequest(user_id="strict-funny-unhedged", message="Хочу только веселый фильм"))
    assert response.state == "clarify" and not response.recommendations
    assert agent.sessions[response.session_id].soft_preferences == []


@pytest.mark.parametrize("message, source", [
    ("Подбери только то, над чем можно посмеяться", "над чем можно посмеяться"),
    ("Хочу фильм про космос", "про космос"),
    ("Подскажи что-нибудь про расследования", "про расследования"),
    ("Посоветуй что-то для детей", "для детей"),
    ("Хочу какой-нибудь интересный боевик", "интересный боевик"),
    ("Хочу какой-нибудь фильм с неожиданными поворотами", "неожиданными поворотами"),
])
def test_free_description_path_does_not_downgrade_hard_or_unverifiable_named_requirements(message, source):
    agent, _ = agent_for(StructuredRequest(updates=[update("genre", "комедия", source)]))
    response = agent.chat(ChatRequest(user_id="strict-expression", message=message))
    assert response.state == "clarify" and not response.recommendations
    assert agent.sessions[response.session_id].soft_preferences == []


def test_strict_funny_request_does_not_become_a_soft_genre_guess():
    agent, _ = agent_for(StructuredRequest(updates=[update("tone", "лёгкий", "смешное")]))
    response = agent.chat(ChatRequest(user_id="strict-funny", message="Хочу посмотреть только смешное"))
    assert response.state == "clarify" and not response.recommendations
    assert agent.sessions[response.session_id].soft_preferences == []


def test_clearing_original_mood_removes_derived_soft_genre():
    agent, interpreter = agent_for(StructuredRequest(updates=[
        update("kind", "film", "фильм"), update("tone", "лёгкий", "интересное смешное"),
    ]))
    first = agent.chat(ChatRequest(user_id="funny-clear", message="Хочу какой-нибудь фильм интересное смешное"))
    hints = agent.sessions[first.session_id].soft_preferences
    assert hints and hints[0]["field"] == "genre" and hints[0]["source_field"] == "tone"
    interpreter.request = StructuredRequest(updates=[update("tone", None, "настроение любое", operation="clear")])
    agent.chat(ChatRequest(user_id="funny-clear", session_id=first.session_id, message="Настроение любое"))
    assert agent.sessions[first.session_id].soft_preferences == []


def test_format_answer_with_a_new_condition_still_uses_full_interpretation():
    agent, interpreter = agent_for(StructuredRequest(updates=[update("tone", "лёгкий", "смешное")]))
    first = agent.chat(ChatRequest(user_id="format-condition", message="Хочу посмотреть что-то смешное"))
    assert first.state == "clarify" and first.clarification_slot == "kind"
    interpreter.request = StructuredRequest(updates=[
        update("kind", "film", "Фильм"), update("tone", "мрачный", "не мрачное", operation="exclude"),
    ])
    second = agent.chat(ChatRequest(
        user_id="format-condition", session_id=first.session_id, message="Фильм, но не мрачное"
    ))
    assert interpreter.calls == 2
    assert second.query.kind == "film"


def test_uncertain_answer_retracts_an_optional_pending_field_without_losing_verified_format():
    agent, interpreter = agent_for(
        StructuredRequest(updates=[update("kind", "film", "фильм"), update("genre", "киберпанк", "киберпанк")])
    )
    first = agent.chat(ChatRequest(user_id="unsure", message="Хочу фильм, жанр киберпанк"))
    assert first.state == "clarify"
    second = agent.chat(ChatRequest(user_id="unsure", session_id=first.session_id, message="Не знаю даже"))
    assert second.state == "recommend" and second.query.kind == "film" and second.query.genre is None
    assert agent.sessions[first.session_id].constraint_state.pending is None
    assert interpreter.calls == 1


def test_uncertain_answer_retracts_only_pending_optional_field_when_format_was_already_active():
    agent, interpreter = agent_for(StructuredRequest(updates=[update("kind", "film", "фильм")]))
    first = agent.chat(ChatRequest(user_id="unsure-active", message="Хочу фильм"))
    interpreter.request = StructuredRequest(updates=[update("genre", "киберпанк", "киберпанк")])
    second = agent.chat(ChatRequest(user_id="unsure-active", session_id=first.session_id, message="жанр киберпанк"))
    assert second.state == "clarify"
    third = agent.chat(ChatRequest(user_id="unsure-active", session_id=first.session_id, message="Не знаю даже"))
    assert third.state == "recommend" and third.query.kind == "film" and third.query.genre is None
    assert interpreter.calls == 2


@pytest.mark.parametrize("reply", ["не знаю даже", "даже не знаю", "ну мне всё равно", "любой"])
def test_short_uncertainty_is_an_optional_skip(reply):
    assert _declines_optional_preference(reply)
    assert not _declines_optional_preference(reply + ", но хочу фантастику")
