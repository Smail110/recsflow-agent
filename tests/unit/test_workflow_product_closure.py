"""Controlled product acceptance; fixtures are independent of frozen DEV."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from recagent.contracts import ValidationFinding
from recagent.interpretation import ConstraintUpdate, InterpretationIssue, StructuredRequest
from recagent.models import ChatRequest, Item
from recagent.providers import DemoProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class ScriptedInterpreter:
    def __init__(self, *requests):
        self.requests = iter(requests)
        self.contexts = []

    def interpret(self, message, previous, **context):
        self.contexts.append(deepcopy({"message": message, "previous": previous, **context}))
        return next(self.requests), 1


class Provider(DemoProvider):
    def __init__(self, items, history=()):
        super().__init__(items)
        self.history_ids = list(history)

    def history(self, user_id):
        return list(self.history_ids)


def update(field, value, source):
    return ConstraintUpdate(field=field, value=value, source_text=source)


def make_agent(items, *requests, history=(), policy="adaptive"):
    return WorkflowAgent(
        mode="ollama",
        provider=Provider(items, history),
        interpreter=ScriptedInterpreter(*requests),
        question_policy=policy,
        response_generator=EvidenceResponseGenerator(),
    )


def chat(agent, message, session_id=None):
    result = agent.chat(ChatRequest(user_id="product-component", message=message, session_id=session_id))
    assert result.mode == "ollama", result.warnings
    return result


def course(item_id, genre="python", level="начальный", practical=True):
    return Item(
        id=item_id,
        title=f"Предмет {item_id}",
        kind="course",
        genre=genre,
        level=level,
        practical=practical,
        quality=0.5,
        description="Component fixture",
    )


def test_exact_navigation_returns_seen_item_with_verified_metadata():
    seed = Item(id="known", title="Письмо с острова", kind="film", genre="драма", quality=0.7, description="Fixture")
    agent = make_agent(
        [seed], StructuredRequest(intent="navigation", updates=[update("seed_title", seed.title, seed.title)]), history=[seed.id]
    )
    response = chat(agent, f"Найди «{seed.title}»")
    assert response.state == "recommend"
    assert [entry.item.id for entry in response.recommendations] == [seed.id]
    assert response.recommendations[0].evidence


def test_coordinated_exclusions_and_later_revocation_keep_the_right_cards():
    items = [
        Item(id="adventure", title="Остров", kind="film", genre="приключения", tone="нейтральный", quality=0.8, description="Fixture"),
        Item(id="drama", title="Письмо", kind="film", genre="драма", tone="нейтральный", quality=0.9, description="Fixture"),
        Item(id="comedy", title="Город", kind="film", genre="комедия", tone="нейтральный", quality=0.9, description="Fixture"),
    ]
    agent = make_agent(
        items,
        StructuredRequest(intent="discovery", updates=[
            update("tone", "нейтральный", "тон нейтральный"),
            ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="исключить драму"),
            ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="исключить комедию"),
        ]),
        StructuredRequest(updates=[
            update("genre", "комедия", "Комедию снова разрешаю"),
            ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="запрет на драму остаётся"),
        ]),
    )
    first = chat(agent, "Ищу фильм: тон нейтральный, исключить драму и комедию")
    assert first.state == "recommend" and first.query.intent == "discovery"
    assert first.query.kind == "film" and first.query.excluded_genres == ["драма", "комедия"]
    assert [entry.item.id for entry in first.recommendations] == ["adventure"]
    second = chat(agent, "Комедию снова разрешаю; запрет на драму остаётся", first.session_id)
    assert second.state == "recommend"
    assert second.query.excluded_genres == ["драма"] and second.query.genre is None
    assert {entry.item.id for entry in second.recommendations} == {"adventure", "comedy"}


def test_contradictory_genre_gets_a_specific_question_without_cards():
    agent = make_agent(
        [Item(id="comedy", title="Город", kind="film", genre="комедия", quality=0.9, description="Fixture")],
        StructuredRequest(intent="discovery", updates=[
            update("kind", "film", "фильм"),
            update("genre", "комедия", "комедия"),
        ]),
    )
    response = chat(agent, "Нужен фильм-комедия, но без комедии")
    assert response.state == "clarify" and not response.recommendations
    assert "комедия" in response.message and "исключить" in response.message


def test_specific_conflict_question_can_be_answered_with_explicit_exclusion():
    agent = make_agent(
        [Item(id="drama", title="Письмо", kind="film", genre="драма", quality=0.9, description="Fixture")],
        StructuredRequest(intent="discovery", updates=[
            update("kind", "film", "фильм"),
            update("genre", "комедия", "комедия"),
        ]),
        StructuredRequest(updates=[
            ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="исключить комедию")
        ]),
    )
    first = chat(agent, "Нужен фильм-комедия, но без комедии")
    second = chat(agent, "Исключить комедию", first.session_id)
    assert second.state == "recommend", second.model_dump(include={"message", "query", "trace", "telemetry"})
    assert second.query.kind == "film" and second.query.excluded_genres == ["комедия"]
    assert [entry.item.id for entry in second.recommendations] == ["drama"]


def test_stale_model_conflict_on_explicit_answer_does_not_block_recommendation():
    stale = InterpretationIssue(
        kind="conflict", field="genre", value="комедию", source_text="Исключить комедию",
        message="Прежнее противоречие",
    )
    initial_issue = InterpretationIssue(
        kind="conflict", field="genre", value="", source_text="комедия",
        message="Неясно, оставить жанр или исключить",
    )
    agent = make_agent(
        [Item(id="drama", title="Письмо", kind="film", genre="драма", quality=0.9, description="Fixture")],
        StructuredRequest(intent="discovery", updates=[
            update("kind", "film", "фильм"), update("genre", "комедия", "комедия"),
        ], issues=[initial_issue], clarification_required=True),
        StructuredRequest(
            updates=[ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="Исключить комедию")],
            issues=[stale], clarification_required=True,
        ),
    )
    first = chat(agent, "Нужен фильм-комедия, но без комедии")
    second = chat(agent, "Исключить комедию", first.session_id)
    assert second.state == "recommend"
    assert second.query.kind == "film" and second.query.excluded_genres == ["комедия"]
    assert [entry.item.id for entry in second.recommendations] == ["drama"]


def test_model_set_does_not_reverse_an_explicit_exclusion_answer():
    agent = make_agent(
        [Item(id="drama", title="Письмо", kind="film", genre="драма", quality=0.9, description="Fixture")],
        StructuredRequest(intent="discovery", updates=[
            update("kind", "film", "фильм"), update("genre", "комедия", "комедия"),
        ]),
        StructuredRequest(updates=[
            ConstraintUpdate(field="genre", operation="set", value="комедия", source_text="Исключить комедию")
        ]),
    )
    first = chat(agent, "Нужен фильм-комедия, но без комедии")
    second = chat(agent, "Исключить комедию", first.session_id)
    assert second.state == "recommend"
    assert second.query.kind == "film" and second.query.genre is None
    assert second.query.excluded_genres == ["комедия"]


def test_negated_exclusion_revokes_ban_without_requiring_that_genre():
    agent = make_agent(
        [
            Item(id="drama", title="Письмо", kind="film", genre="драма", quality=0.9, description="Fixture"),
            Item(id="comedy", title="Город", kind="film", genre="комедия", quality=0.9, description="Fixture"),
        ],
        StructuredRequest(updates=[
            update("kind", "film", "фильм"),
            ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="без комедии"),
        ]),
        StructuredRequest(updates=[
            ConstraintUpdate(field="genre", operation="set", value="комедия", source_text="комедию")
        ]),
    )
    first = chat(agent, "Фильм без комедии")
    second = chat(agent, "Не исключайте комедию", first.session_id)
    assert second.state == "clarify"  # The adaptive policy may ask for a new preference.
    assert second.query.genre is None and second.query.excluded_genres == []
    third = chat(agent, "Не знаю", first.session_id)
    assert third.state == "recommend"
    assert {entry.item.id for entry in third.recommendations} == {"drama", "comedy"}


def test_stale_conflict_is_kept_when_answer_names_another_value():
    stale = InterpretationIssue(
        kind="conflict", field="genre", value="", source_text="Исключить драму",
        message="Прежнее противоречие",
    )
    agent = make_agent(
        [Item(id="drama", title="Письмо", kind="film", genre="драма", quality=0.9, description="Fixture")],
        StructuredRequest(intent="discovery", updates=[
            update("kind", "film", "фильм"), update("genre", "комедия", "комедия"),
        ]),
        StructuredRequest(
            updates=[ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="Исключить драму")],
            issues=[stale], clarification_required=True,
        ),
    )
    first = chat(agent, "Нужен фильм-комедия, но без комедии")
    second = chat(agent, "Исключить драму", first.session_id)
    assert second.state == "clarify" and not second.recommendations


def test_empty_model_conflict_value_does_not_crash_clarification():
    agent = make_agent([])
    finding = ValidationFinding(
        code="interpretation_issue",
        status="uncertain",
        field="genre",
        details=json.dumps({"kind": "conflict", "field": "genre", "value": "", "source_text": "комедия"}),
    )
    question = agent._clarification_message((finding,), "Нужен фильм-комедия, но без комедии")
    assert question == "Какой жанр или тему вы предпочитаете?"


def test_excluding_another_genre_does_not_resolve_the_pending_conflict():
    agent = make_agent(
        [Item(id="comedy", title="Город", kind="film", genre="комедия", quality=0.9, description="Fixture")],
        StructuredRequest(intent="discovery", updates=[
            update("kind", "film", "фильм"),
            update("genre", "комедия", "комедия"),
        ]),
        StructuredRequest(updates=[
            ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="исключить драму")
        ]),
    )
    first = chat(agent, "Нужен фильм-комедия, но без комедии")
    second = chat(agent, "Исключить драму", first.session_id)
    assert second.state == "clarify" and not second.recommendations


def test_titleless_navigation_without_descriptive_constraints_still_asks_for_title():
    agent = make_agent([], StructuredRequest(intent="navigation", updates=[update("kind", "film", "фильм")]))
    response = chat(agent, "Открой фильм")
    assert response.state == "clarify" and response.clarification_slot == "seed_title"


def test_unquoted_title_search_with_other_constraints_does_not_return_unrelated_cards():
    unrelated = Item(id="unrelated", title="Другая история", kind="film", genre="комедия", quality=0.7, description="Fixture")
    agent = make_agent(
        [unrelated],
        StructuredRequest(intent="navigation", updates=[
            update("kind", "film", "фильм"),
            ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="без драмы"),
        ]),
    )
    response = chat(agent, "Найдите фильм Лунный порт, без драмы")
    assert response.state == "clarify" and not response.recommendations
    assert response.clarification_slot == "seed_title"


def test_contradictory_genre_waits_for_correction_then_recommends():
    comedy = Item(id="comedy", title="Весёлый день", kind="film", genre="комедия", quality=0.8, description="Fixture")
    other = Item(id="other", title="Остров", kind="film", genre="приключения", quality=0.7, description="Fixture")
    agent = make_agent(
        [comedy, other],
        StructuredRequest(updates=[
            update("kind", "film", "фильм-комедия"),
            ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="комедия"),
        ]),
        StructuredRequest(updates=[
            ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="Оставьте комедии"),
            update("kind", "film", "фильм"),
        ]),
    )
    first = chat(agent, "Нужен фильм-комедия, но комедии не предлагайте")
    assert first.state == "clarify" and not first.recommendations
    second = chat(agent, "Оставьте комедии, нужен фильм", first.session_id)
    assert second.state == "recommend" and second.query.genre == "комедия"
    assert second.query.excluded_genres == []
    assert [entry.item.id for entry in second.recommendations] == ["comedy"]


def test_similarity_ranks_close_item_above_high_quality_unrelated_items():
    seed = Item(id="seed", title="Письмо с острова", kind="film", genre="драма", tone="мрачный", quality=0.6, description="Fixture")
    near = seed.model_copy(update={"id": "near", "title": "Тихая улица", "quality": 0.2})
    far = [
        seed.model_copy(update={"id": f"far-{index}", "title": f"Праздник {index}", "genre": "комедия", "tone": "лёгкий", "quality": 0.9})
        for index in range(6)
    ]
    agent = make_agent([seed, near, *far], StructuredRequest(intent="similar", updates=[update("seed_title", seed.title, seed.title)]))
    response = chat(agent, f"Что-нибудь похожее на «{seed.title}»")
    assert response.state == "recommend"
    assert response.recommendations[0].item.id == near.id
    assert seed.id not in {entry.item.id for entry in response.recommendations}


def test_kind_only_request_asks_useful_question_and_answer_filters_catalog():
    agent = make_agent(
        [course("syntax"), course("statistics", "машинное обучение")],
        StructuredRequest(updates=[update("kind", "course", "курс")]),
        StructuredRequest(updates=[update("genre", "python", "Python")]),
    )
    first = chat(agent, "Подбери курс")
    assert first.state == "clarify"
    assert first.clarification_slot == "genre"
    assert first.question_gain > 0
    second = chat(agent, "Python", first.session_id)
    assert second.state == "recommend"
    assert {entry.item.id for entry in second.recommendations} == {"syntax"}
    assert second.query.kind == "course"


def test_homogeneous_catalog_does_not_trigger_pointless_optional_question():
    agent = make_agent([course("one"), course("two")], StructuredRequest(updates=[update("kind", "course", "курс")]))
    response = chat(agent, "Подбери курс")
    assert response.state == "recommend"
    assert response.clarification_count == 0
    assert len(response.recommendations) == 2


def test_explicit_topic_produces_recommendations_without_optional_level_question():
    agent = make_agent(
        [course("intro"), course("advanced", level="продвинутый"), course("other", "машинное обучение")],
        StructuredRequest(updates=[update("kind", "course", "курс"), update("genre", "python", "Python")]),
    )
    response = chat(agent, "Подбери курс Python")
    assert response.state == "recommend"
    assert response.clarification_count == 0
    assert {entry.item.id for entry in response.recommendations} == {"intro", "advanced"}


@pytest.mark.parametrize("reply", ["не важно", "не знаю даже", "даже не знаю", "мне всё равно"])
def test_skipping_optional_question_keeps_constraints_and_returns_recommendations(reply):
    agent = make_agent(
        [course("syntax"), course("statistics", "машинное обучение")],
        StructuredRequest(updates=[update("kind", "course", "курс")]),
    )
    first = chat(agent, "Подбери курс")
    assert first.state == "clarify"
    before = agent.sessions[first.session_id].constraint_state.constraints
    second = chat(agent, reply, first.session_id)
    assert second.state == "recommend"
    assert second.query.kind == "course"
    assert agent.sessions[first.session_id].constraint_state.constraints == before
    assert {entry.item.id for entry in second.recommendations} == {"syntax", "statistics"}
    assert second.llm_calls == 0
    assert len(agent.interpreter.contexts) == 1


def test_pending_context_contains_trusted_state_without_recursive_diagnostics():
    diagnostic = "INTERNAL_DIAGNOSTIC_SENTINEL: clarification_required"
    agent = make_agent(
        [course("intro"), course("advanced", level="продвинутый")],
        StructuredRequest(updates=[update("kind", "course", "курс"), update("genre", "python", "Python")]),
        StructuredRequest(
            updates=[update("practical", True, "практикой")],
            issues=[InterpretationIssue(kind="ambiguity", field="level", value="серединный", source_text="серединный", message=diagnostic)],
        ),
        StructuredRequest(updates=[update("level", "начальный", "начальный")]),
    )
    first = chat(agent, "Подбери курс Python")
    second = chat(agent, "С практикой, серединный уровень", first.session_id)
    assert second.state == "clarify"
    third = chat(agent, "Пусть начальный уровень", first.session_id)
    context = agent.interpreter.contexts[-1]
    assert {(entry["field"], entry["value"]) for entry in context["previous"]["trusted_active"]} == {
        ("kind", "course"),
        ("genre", "python"),
    }
    pending = context["pending_context"]
    assert {(entry["field"], entry["value"]) for entry in pending["trusted_staged"]} == {("practical", True)}
    assert context["unresolved"] and pending["blockers"]
    assert all(set(entry) == {"field"} for entry in [*context["unresolved"], *pending["blockers"]])
    assert diagnostic not in str(context["unresolved"])
    assert diagnostic not in str(pending)
    assert third.state == "recommend"
    assert (third.query.kind, third.query.genre, third.query.level, third.query.practical) == ("course", "python", "начальный", True)


class RepairInterpreter:
    def __init__(self, backend, repaired, *, fail=False):
        self.backend, self.repaired, self.fail = backend, repaired, fail
        self.repair_contexts = []

    def interpret(self, *args, **kwargs):
        self.backend.last_usage = {"input_tokens": 7, "output_tokens": 4}
        return StructuredRequest(updates=[update("kind", "course", "курс")]), 11

    def repair(self, message, previous, **context):
        self.repair_contexts.append({"message": message, "previous": previous, **context})
        if self.fail:
            self.backend.last_usage = {"input_tokens": 5, "output_tokens": 0}
            raise TimeoutError("Controlled retry failure")
        self.backend.last_usage = {"input_tokens": 10, "output_tokens": 7}
        return self.repaired, 17


def repair_agent(repaired=None, *, fail=False, max_calls=8, max_tokens=100_000):
    backend = SimpleNamespace(last_usage={})
    repaired = repaired or StructuredRequest(updates=[update("kind", "course", "курс"), update("genre", "python", "Python")])
    interpreter = RepairInterpreter(backend, repaired, fail=fail)
    return WorkflowAgent(
        mode="ollama",
        llm=backend,
        provider=Provider([course("python-topic")]),
        interpreter=interpreter,
        response_generator=EvidenceResponseGenerator(),
        question_policy="adaptive",
        max_calls=max_calls,
        max_tokens=max_tokens,
    )


def test_bounded_recovery_covers_missing_field_and_accounts_both_attempts():
    agent = repair_agent()
    response = chat(agent, "Подбери курс Python")
    assert response.state == "recommend"
    assert (response.query.kind, response.query.genre) == ("course", "python")
    assert response.llm_calls == response.llm_calls_total == 2
    assert response.llm_tokens == response.llm_tokens_total == 28
    assert response.telemetry["llm_calls"] == 2 and response.telemetry["tokens"] == 28
    assert response.llm_usage["input_tokens"] == 17
    assert response.llm_usage["output_tokens"] == 11
    assert len(agent.interpreter.repair_contexts) == 1
    feedback = agent.interpreter.repair_contexts[0]["feedback"]
    assert any(entry["code"] == "coverage_gap" and entry["field"] == "genre" for entry in feedback)
    assert all({"code", "field"} <= set(entry) <= {"code", "field", "detail"} for entry in feedback)
    assert all(not ({"value", "expected", "expected_value", "oracle"} & set(entry)) for entry in feedback)
    # A coverage diagnostic may cite a surface already present in the input,
    # but cannot supply the desired semantic value from a scoring oracle.
    genre_feedback = next(entry for entry in feedback if entry["field"] == "genre")
    assert "python" in genre_feedback.get("detail", "").casefold()


def test_failed_repair_keeps_original_rejected_proposal_and_observed_usage():
    agent = repair_agent(fail=True)
    response = chat(agent, "Подбери курс Python")
    assert response.state == "clarify"
    session = agent.sessions[response.session_id]
    assert session.constraint_state.constraints == ()
    assert session.constraint_state.pending is not None
    assert any(finding.code == "coverage_gap" for finding in session.constraint_state.pending.findings)
    assert response.telemetry["fallback_reason"] is None
    assert response.llm_calls == 2
    assert response.llm_tokens == 16
    assert response.llm_usage == {"input_tokens": 12, "output_tokens": 4}


@pytest.mark.parametrize("limits", [{"max_calls": 1}, {"max_tokens": 11}])
def test_budget_prevents_recovery_without_committing_incomplete_extraction(limits):
    agent = repair_agent(**limits)
    response = chat(agent, "Подбери курс Python")
    assert response.state == "clarify"
    assert response.llm_calls == 1 and response.llm_tokens == 11
    assert not agent.interpreter.repair_contexts
    assert agent.sessions[response.session_id].constraint_state.constraints == ()


def test_invented_evidence_from_repair_is_still_rejected_without_more_retries():
    repaired = StructuredRequest(updates=[update("kind", "course", "курс"), update("genre", "машинное обучение", "машинное обучение")])
    agent = repair_agent(repaired)
    response = chat(agent, "Подбери курс Python")
    assert response.state == "clarify"
    session = agent.sessions[response.session_id]
    assert session.constraint_state.constraints == ()
    assert any(finding.code == "evidence_not_in_turn" for finding in session.constraint_state.pending.findings)
    assert response.llm_calls == 2 and response.llm_tokens == 28
    assert len(agent.interpreter.repair_contexts) == 1


def test_skipped_course_topic_does_not_disable_questions_after_domain_switch():
    films = [Item(id=genre, title=genre, kind="film", genre=genre, quality=0.5, description="Fixture") for genre in ("драма", "комедия")]
    agent = make_agent(
        [course("syntax"), course("statistics", "машинное обучение"), *films],
        StructuredRequest(updates=[update("kind", "course", "курс")]),
        StructuredRequest(updates=[update("kind", "film", "фильм")]),
    )
    first = chat(agent, "Подбери курс")
    assert first.state == "clarify"
    assert chat(agent, "не важно", first.session_id).state == "recommend"
    changed = chat(agent, "Теперь фильм", first.session_id)
    assert changed.query.kind == "film"
    assert changed.state == "clarify"
    assert changed.clarification_slot == "genre"


def test_generation_failure_preserves_observed_inference_usage_in_fallback():
    class BrokenGenerator:
        requires_llm = True

        def __init__(self):
            self.backend = SimpleNamespace(last_usage={})

        def generate(self, **kwargs):
            self.backend.last_usage = {"input_tokens": 9, "output_tokens": 4}
            raise ValueError("Controlled schema failure after measured inference")

    agent = make_agent(
        [course("syntax")], StructuredRequest(updates=[update("kind", "course", "курс"), update("genre", "python", "Python")])
    )
    agent.response_generator = BrokenGenerator()
    response = agent.chat(ChatRequest(user_id="product-component", message="курс Python"))
    assert response.state == "recommend"
    assert response.mode == "rules_fallback"
    assert response.telemetry["fallback_reason"] == "response_generation_failure:ValueError"
    assert response.llm_calls == 2
    assert response.llm_tokens == response.llm_tokens_total == 14
    assert response.llm_usage == {"input_tokens": 9, "output_tokens": 4}
    assert response.telemetry["tokens"] == 14


def test_declared_implied_kind_does_not_trigger_repair_for_valid_genre_update():
    class ImpliedInterpreter(RepairInterpreter):
        def interpret(self, *args, **kwargs):
            self.backend.last_usage = {"input_tokens": 7, "output_tokens": 4}
            return StructuredRequest(updates=[update("genre", "python", "Python")]), 11

    agent = repair_agent()
    agent.interpreter = ImpliedInterpreter(agent.llm, StructuredRequest())
    response = chat(agent, "Python")
    assert response.state == "recommend"
    assert (response.query.kind, response.query.genre) == ("course", "python")
    assert response.llm_calls == 1 and response.llm_tokens == 11
    assert not agent.interpreter.repair_contexts
