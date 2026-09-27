from typing import Literal

import pytest
from pydantic import BaseModel

from recagent.agent import Agent
from recagent.domains.demo import request_adapter
from recagent.interpretation import ConstraintUpdate, LLMRequestInterpreter, StructuredRequest
from recagent.models import ChatRequest, Query
from recagent.request_mapping import SchemaRequestAdapter
from recagent.resolution import LanguageProfile


def update(field, value, source, operation="set"):
    return ConstraintUpdate(field=field, value=value, source_text=source, operation=operation)


def test_unknown_enum_never_becomes_nearest_supported_value():
    request = StructuredRequest(updates=[update("level", "начальный", "middle")])
    query, issues = request_adapter().apply(request, Query(kind="course"), "Ищу курс уровня middle")
    assert query.level is None
    assert issues[0].kind == "unsupported_constraint"
    assert issues[0].value == "middle"


def test_negation_tokens_are_owned_by_the_injected_language_profile():
    class StyleRequest(BaseModel):
        style: Literal["warm", "cool"] | None = None

    adapter = SchemaRequestAdapter(
        StyleRequest,
        aliases={"style": {"warm": "warm", "cool": "cool"}},
        language_profile=LanguageProfile(negation_tokens=("not",)),
    )
    normalized = adapter.normalize(
        StructuredRequest(updates=[update("style", "warm", "не warm")]),
        "не warm",
    )
    assert normalized.updates and not normalized.issues


def test_fabricated_evidence_is_not_a_successful_normalization():
    request = StructuredRequest(updates=[update("level", "начальный", "начальный")])
    query, issues = request_adapter().apply(request, Query(kind="course"), "Ищу курс уровня middle")
    assert query.level is None and issues


def test_exclusion_requires_negative_polarity_inside_its_own_citation():
    request = StructuredRequest(updates=[update("genre", "драма", "драма", "exclude")])
    query, issues = request_adapter().apply(request, Query(kind="film"), "Драмы не предлагайте")
    assert query.excluded_genres == []
    assert any(issue.kind == "ambiguity" and issue.field == "genre" for issue in issues)


def test_positive_enum_is_not_rejected_by_negation_for_another_field():
    request = StructuredRequest(
        updates=[
            update("kind", "film", "Нужен фильм не мрачный"),
            update("tone", "мрачный", "Нужен фильм не мрачный", "exclude"),
        ]
    )
    query, issues = request_adapter().apply(request, Query(), "Нужен фильм не мрачный")
    assert not issues
    assert query.kind == "film"
    assert query.tone is None


@pytest.mark.parametrize(
    ("field", "value", "source"),
    [
        ("genre", "комедия", "без комедий"),
        ("tone", "мрачный", "не мрачные"),
        ("kind", "series", "без сериалов"),
        ("practical", True, "не практика"),
    ],
)
def test_negated_evidence_cannot_create_a_positive_constraint(field, value, source):
    query, issues = request_adapter().apply(
        StructuredRequest(updates=[update(field, value, source)]),
        Query(),
        source,
    )
    assert issues and getattr(query, field) in (None, [])


def test_only_explicit_updates_change_session_constraints():
    previous = Query(kind="course", genre="python", practical=True)
    request = StructuredRequest(updates=[update("level", "продвинутый", "продвинутый")])
    query, issues = request_adapter().apply(request, previous, "Прoшу продвинутый уровень")
    assert not issues
    assert query == Query(kind="course", genre="python", practical=True, level="продвинутый")


def test_domain_change_drops_old_fields_without_language_rules():
    request = StructuredRequest(intent="discovery", updates=[update("kind", "course", "курс"), update("genre", "python", "Python")])
    query, issues = request_adapter().apply(request, Query(kind="series", tone="лёгкий", max_seasons=1), "Теперь курс Python")
    assert not issues
    assert query == Query(kind="course", genre="python")


def test_conflicting_updates_cannot_silently_pick_a_winner():
    request = StructuredRequest(updates=[update("genre", "драма", "драма"), update("genre", "драма", "без драмы", "exclude")])
    _, issues = request_adapter().apply(request, Query(kind="film"), "Драма, но без драмы")
    assert any(issue.kind == "conflict" for issue in issues)


def test_unresolved_value_survives_an_unrelated_followup():
    adapter = request_adapter()
    previous, issues = adapter.apply(StructuredRequest(updates=[update("level", "middle", "middle")]), Query(kind="course"), "middle")
    query, pending = adapter.apply(StructuredRequest(updates=[update("genre", "python", "Python")]), previous, "Python", issues)
    assert query.genre == "python" and pending[0].value == "middle"
    query, pending = adapter.apply(
        StructuredRequest(updates=[update("level", "продвинутый", "продвинутый")]), query, "продвинутый", pending
    )
    assert not pending and query.level == "продвинутый"


def test_another_domain_and_backend_need_no_semantic_core_changes():
    class FurnitureQuery(BaseModel):
        intent: str = "discovery"
        material: Literal["wood", "steel"] | None = None
        max_width: int | None = None

    adapter = SchemaRequestAdapter(FurnitureQuery, aliases={"material": {"дерево": "wood"}})

    class AlternativeBackend:
        def structured(self, schema, system, payload):
            assert "material" in payload["domain"]["schema"]["properties"]
            assert "course" not in payload["domain"]["schema"]["properties"]
            return schema(updates=[update("material", "wood", "дерево"), update("max_width", 80, "80")]), 12

    interpreter = LLMRequestInterpreter(AlternativeBackend(), adapter.descriptor)
    structured, tokens = interpreter.interpret("дерево, ширина до 80", {})
    query, issues = adapter.apply(structured, FurnitureQuery(), "дерево, ширина до 80")
    assert not issues and tokens == 12
    assert query.material == "wood" and query.max_width == 80


def test_successful_llm_route_never_runs_semantic_rules(monkeypatch):
    def forbidden_rules(*args):
        raise AssertionError("semantic rules must not run")

    monkeypatch.setattr("recagent.agent.rule_parse", forbidden_rules)

    class Backend:
        def structured(self, schema, system, payload):
            return schema(
                intent="discovery",
                updates=[update("kind", "course", "курс").model_dump(), update("genre", "python", "Python").model_dump()],
            ), 10

    response = Agent(mode="ollama", llm=Backend()).chat(ChatRequest(user_id="routing", message="Помоги найти курс по теме Python"))
    assert response.mode == "ollama"
    assert response.query.intent == "discovery" and response.query.genre == "python"
    assert response.recommendations and response.llm_calls == 1


def test_llm_issue_reaches_policy_and_explicit_resolution_keeps_context():
    class Interpreter:
        def interpret(self, message, previous, **context):
            if "middle" in message:
                return StructuredRequest(
                    intent="discovery", updates=[update("kind", "course", "курс"), update("level", "начальный", "middle")]
                ), 7
            assert context["unresolved"] and context["pending_question"]
            return StructuredRequest(updates=[update("level", "продвинутый", "продвинутый"), update("genre", "python", "Python")]), 7

    agent = Agent(mode="ollama", interpreter=Interpreter())
    first = agent.chat(ChatRequest(user_id="routing", message="Ищу курс уровня middle"))
    assert first.state == "clarify" and first.query.level is None
    second = agent.chat(ChatRequest(user_id="routing", session_id=first.session_id, message="Тогда продвинутый Python"))
    assert second.state == "recommend" and second.query.level == "продвинутый"
    assert second.query.kind == "course" and second.llm_calls_total == 2


@pytest.mark.parametrize(
    ("field", "value", "evidence", "message"),
    [
        ("max_minutes", 900, "90", "До 90 минут"),
        ("max_minutes", 90, "90", "До 900 минут"),
        ("max_minutes", 60, "1 час", "До 21 часа"),
        ("max_minutes", 5, "5", "До 1.5 минут"),
        ("seed_title", "Выдуманный объект", "Настоящий объект", "Найди Настоящий объект"),
        ("practical", False, "практика", "Нужна практика"),
    ],
)
def test_scalar_value_must_match_cited_evidence(field, value, evidence, message):
    query, issues = request_adapter().apply(StructuredRequest(updates=[update(field, value, evidence)]), Query(), message)
    assert issues and getattr(query, field) is None


def test_reset_flag_alone_and_clear_domain_do_not_erase_constraints():
    before = Query(kind="film", genre="драма", max_minutes=90)
    unchanged, _ = request_adapter().apply(StructuredRequest(reset_constraints=True), before, "Ещё")
    assert unchanged == before
    cleared, issues = request_adapter().apply(
        StructuredRequest(updates=[update("kind", None, "Любой формат", "clear")]), before, "Любой формат"
    )
    assert not issues and cleared.kind is None
    assert cleared.genre == "драма" and cleared.max_minutes == 90


def test_schema_defined_category_inference_is_separate_from_literal_level():
    structured = StructuredRequest(
        updates=[update("kind", "course", "Python"), update("genre", "python", "Python"), update("level", "начальный", "новичка")]
    )
    query, issues = request_adapter().apply(structured, Query(), "Python для новичка")
    assert not issues and query == Query(kind="course", genre="python", level="начальный")


def test_budget_fallback_can_resolve_previous_unknown_level():
    class Interpreter:
        def interpret(self, message, previous, **context):
            return StructuredRequest(updates=[update("kind", "course", "курс"), update("level", "middle", "middle")]), 5

    agent = Agent(mode="ollama", interpreter=Interpreter(), max_calls=1)
    first = agent.chat(ChatRequest(user_id="fallback", message="Ищу курс уровня middle"))
    assert first.state == "clarify"
    second = agent.chat(ChatRequest(user_id="fallback", session_id=first.session_id, message="Продвинутый курс Python"))
    assert second.mode == "rules_fallback" and second.state == "recommend"
    assert not agent.sessions[first.session_id].unresolved
