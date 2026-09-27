"""Adapter semantic-resolution contract: what may and may not be accepted.

These tests pin the boundary between evidence checking and language
interpretation. The positive cases prove that an inflected citation no longer
destroys a canonical value; the safety cases prove that unknown vocabulary,
fabricated evidence, numeric boundaries, polarity and contradictions are still
refused. The cross-domain case proves the mechanism is schema-driven rather than
a hidden dictionary for courses and films.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from recagent.contracts import Constraint, ConstraintState, ValidationFinding
from recagent.domains.demo import request_adapter
from recagent.interpretation import StructuredRequest
from recagent.models import Query
from recagent.request_mapping import SchemaRequestAdapter
from recagent.resolution import (
    ExactSurfaceResolver,
    FieldSpec,
    InflectionalResolver,
    LanguageProfile,
    comparison_key,
    resolve_scalar,
    strip_enclosing_quotes,
)


def update(field, value, source, operation="set"):
    return {"field": field, "value": value, "source_text": source, "operation": operation}


def apply(updates, previous=None, message="", **kwargs):
    return request_adapter().apply(
        StructuredRequest(updates=[update(*u) for u in updates]), previous if previous is not None else Query(), message, **kwargs
    )


# --- Positive: an inflected citation must not destroy a canonical value -------


@pytest.mark.parametrize(
    ("field", "value", "source", "message"),
    [
        ("level", "начальный", "для начинающих", "Нужна программа обучения Python для начинающих"),
        ("level", "начальный", "начального уровня", "Покажите курс начального уровня"),
        ("level", "продвинутый", "для опытного специалиста", "Подберите обучение для опытного специалиста"),
        ("kind", "course", "практическом курсе", "Хочу освоить Python на практическом курсе"),
        ("genre", "комедия", "комедии", "Оставьте комедии, нужен фильм"),
        ("genre", "машинное обучение", "по машинному обучению", "Подберите обучение по машинному обучению"),
        ("practical", True, "практическими заданиями", "Нужен курс с практическими заданиями"),
    ],
)
def test_inflected_evidence_preserves_canonical_value(field, value, source, message):
    query, issues = apply([(field, value, source)], Query(kind="course") if field != "kind" else Query(), message)
    assert not issues
    assert getattr(query, field) == value


def test_unknown_synonym_phrase_stays_unresolved_rather_than_guessed():
    """Honest residual: «программа обучения» shares no stem with any declared
    course surface, so it must stay unsupported. Adding it to the alias table to
    pass this case would be development-cohort overfitting."""
    query, issues = apply([("kind", "course", "программа обучения")], Query(), "Нужна программа обучения Python")
    assert query.kind is None
    assert issues and issues[0].kind == "unsupported_constraint"


def test_null_value_is_resolved_when_evidence_denotes_one_declared_value():
    query, issues = apply([("genre", None, "python")], Query(kind="course"), "Найти курс по теме python.")
    assert not issues
    assert query.genre == "python"


def test_null_value_stays_unresolved_when_evidence_denotes_nothing():
    query, issues = apply([("genre", None, "что-нибудь интересное")], Query(kind="course"), "Найди что-нибудь интересное")
    assert query.genre is None
    assert issues[0].kind == "unsupported_constraint"


def test_numeric_evidence_with_unit_phrase_is_accepted():
    query, issues = apply([("max_minutes", 1, "не дольше 1 минуты")], Query(kind="film"), "Найди фильм-комедию не дольше 1 минуты")
    assert not issues and query.max_minutes == 1


def test_numeric_unit_cannot_be_ignored_when_it_changes_the_value():
    wrong, issues = apply([("max_minutes", 2, "2 часа")], Query(kind="film"), "Фильм до 2 часа")
    assert wrong.max_minutes is None and issues
    correct, no_issues = apply([("max_minutes", 120, "2 часа")], Query(kind="film"), "Фильм до 2 часа")
    assert not no_issues and correct.max_minutes == 120


def test_declared_numeric_word_surface_is_accepted_without_a_second_parser():
    query, issues = apply([("max_seasons", 1, "один сезон")], Query(kind="series"), "Нужен сериал в один сезон")
    assert not issues and query.max_seasons == 1


@pytest.mark.parametrize("source", ["одного сезона", "одной серии", "двух сезонов"])
def test_numeric_word_requires_a_declared_unit(source):
    value = 2 if source.startswith("двух") else 1
    query, issues = apply([("max_seasons", value, source)], Query(kind="series"), f"Сериал не длиннее {source}")
    if "серии" in source:
        assert query.max_seasons is None and issues
    else:
        assert not issues and query.max_seasons == value


@pytest.mark.parametrize(
    ("field", "source", "previous"),
    [
        ("genre", "жанр любой", Query(kind="series", genre="детектив")),
        ("max_seasons", "сезонов сколько угодно", Query(kind="series", max_seasons=1)),
        ("max_seasons", "количество сезонов не важно", Query(kind="series", max_seasons=1)),
    ],
)
def test_explicit_no_preference_clears_optional_constraint(field, source, previous):
    query, issues = apply([(field, "детектив" if field == "genre" else 100, source)], previous, f"Посоветуй сериал, {source}")
    assert not issues
    assert getattr(query, field) is None


def test_no_preference_marker_cannot_erase_a_concrete_value_in_same_citation():
    query, issues = apply([("genre", "детектив", "любой детектив")], Query(kind="series", genre="комедия"), "Посоветуй любой детектив")
    assert not issues and query.genre == "детектив"


def test_clear_must_cite_the_field_it_removes():
    previous = Query(kind="film", genre="приключения", max_minutes=110)
    message = "Снимите ограничение длительности; приключения оставьте"
    request = StructuredRequest(
        updates=[
            update("max_minutes", None, "Снимите ограничение длительности", "clear"),
            update("genre", None, "Снимите ограничение длительности", "clear"),
        ]
    )
    query, issues = request_adapter().apply(request, previous, message)
    assert query.max_minutes is None
    assert query.genre == "приключения"
    assert any(issue.field == "genre" for issue in issues)


def test_no_preference_for_one_field_cannot_clear_another():
    previous = Query(kind="film", genre="детектив", max_minutes=90)
    wrong_genre, genre_issues = apply(
        [("genre", None, "Без ограничений по длительности", "clear")], previous, "Без ограничений по длительности"
    )
    assert wrong_genre.genre == "детектив" and genre_issues
    wrong_duration, duration_issues = apply([("max_minutes", None, "Любой жанр", "clear")], previous, "Любой жанр")
    assert wrong_duration.max_minutes == 90 and duration_issues


def test_long_citation_with_a_concrete_genre_cannot_clear_it():
    previous = Query(kind="film", genre="приключения", max_minutes=110)
    message = "Снимите ограничение по длительности, жанр приключения оставьте"
    query, issues = apply([("genre", None, message, "clear")], previous, message)
    assert query.genre == "приключения" and query.max_minutes == 110
    assert issues


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("genre", "Снимите ограничение по длительности, жанр оставьте"),
        ("max_seasons", "Снимите длительность, два сезона оставьте"),
        ("genre", "Снимите ограничение на фильм, жанр оставьте"),
        ("genre", "Снимите ограничение на фильм жанр оставьте"),
        ("genre", "Жанр любой, жанр оставьте"),
    ],
)
def test_clear_citation_spanning_two_fields_stays_ambiguous(field, message):
    previous = Query(kind="series", genre="детектив", max_seasons=2, max_minutes=90)
    query, issues = apply([(field, None, message, "clear")], previous, message)
    assert query == previous and issues


@pytest.mark.parametrize(
    ("field", "source"),
    [("max_minutes", "Без ограничений по часам"), ("tone", "Без ограничений по тону")],
)
def test_inflected_clear_cue_still_accepts_explicit_opt_out(field, source):
    previous = Query(kind="film", max_minutes=90, tone="мрачный")
    query, issues = apply([(field, None, source, "clear")], previous, source)
    assert not issues and getattr(query, field) is None


def test_workflow_ignores_only_unset_opt_outs():
    from recagent.workflow import WorkflowAgent

    agent = WorkflowAgent(mode="rules")
    message = "Посоветуй сериал, жанр любой"
    request = StructuredRequest(updates=[update("genre", None, "жанр любой", "clear")])
    assert agent._without_unset_preferences(request, message, ConstraintState()).updates == []

    active = ConstraintState(constraints=(Constraint(id="g", field="genre", op="eq", value="детектив", turn_id="old", domain_version="1"),))
    assert agent._without_unset_preferences(request, message, active).updates == request.updates

    unrelated = StructuredRequest(updates=[update("genre", None, "жанр", "clear")])
    assert agent._without_unset_preferences(unrelated, message, ConstraintState()).updates == unrelated.updates


def test_clarification_uses_a_human_question_instead_of_schema_key():
    from recagent.workflow import WorkflowAgent

    finding = ValidationFinding(code="adapter_unsupported_constraint", status="uncertain", field="max_seasons")
    question = WorkflowAgent(mode="rules")._clarification_message([finding])
    assert "max_seasons" not in question
    assert "сезонов" in question


def test_quoted_title_is_normalized_without_losing_case():
    query, issues = apply([("seed_title", "«Декоратор в проде»", "«Декоратор в проде»")], Query(), "Найди «Декоратор в проде»")
    assert not issues
    assert query.seed_title == "Декоратор в проде"


# --- Safety: none of these may be silently accepted ---------------------------


@pytest.mark.parametrize(
    ("value", "source", "message"),
    [
        ("начальный", "middle", "Ищу курс уровня middle"),
        ("продвинутый", "middle-разработчика", "Нужна программа по Python для middle-разработчика"),
        ("продвинутый", "уровня middle", "Ищу курс уровня middle"),
        ("начальный", "начальный", "Ищу курс уровня middle"),  # fabricated citation
        ("course", "программа", "Нужна программа по Python"),  # unknown vocabulary
    ],
)
def test_unknown_or_fabricated_evidence_never_becomes_nearest_enum(value, source, message):
    query, issues = apply([("level" if value in ("начальный", "продвинутый") else "kind", value, source)], Query(kind="course"), message)
    assert query.level is None
    assert issues and issues[0].kind in {"unsupported_constraint", "ambiguity"}


def test_middle_stays_unresolved_and_survives_an_unrelated_followup():
    adapter = request_adapter()
    first, pending = adapter.apply(
        StructuredRequest(updates=[update("level", "продвинутый", "middle-разработчика")]),
        Query(kind="course", genre="python"),
        "Нужна программа по Python для middle-разработчика",
    )
    assert first.level is None and pending
    second, still_pending = adapter.apply(StructuredRequest(updates=[update("genre", "python", "Python")]), first, "Python", pending)
    assert second.level is None and any(issue.field == "level" for issue in still_pending)


def test_undetailed_clarification_flag_cannot_erase_a_complete_verified_patch():
    request = StructuredRequest(
        updates=[
            update("kind", "course", "курс"),
            update("genre", "python", "Python"),
            update("level", "начальный", "начальный"),
        ],
        clarification_required=True,
    )
    query, issues = request_adapter().apply(request, Query(), "Тогда начальный курс по Python")
    assert not issues
    assert (query.kind, query.genre, query.level) == ("course", "python", "начальный")


def test_undetailed_clarification_flag_without_updates_remains_a_clarification():
    query, issues = request_adapter().apply(StructuredRequest(clarification_required=True), Query(), "Подберите что-нибудь")
    assert query == Query()
    assert issues and issues[0].kind == "ambiguity"


@pytest.mark.parametrize(
    ("field", "value", "evidence", "message"),
    [
        ("max_minutes", 900, "90", "До 90 минут"),
        ("max_minutes", 90, "90", "До 900 минут"),
        ("max_minutes", 60, "1 час", "До 21 часа"),
        ("max_minutes", 5, "5", "До 1.5 минут"),
        ("seed_title", "Выдуманный объект", "Настоящий объект", "Найди Настоящий объект"),
        ("practical", False, "практика", "Нужна практика"),  # polarity inversion
        ("tone", "лёгкий", "не мрачный", "Что-нибудь не мрачное"),  # negation is not a value
    ],
)
def test_scalar_evidence_must_support_the_proposed_value(field, value, evidence, message):
    query, issues = apply([(field, value, evidence)], Query(kind="course"), message)
    assert issues and getattr(query, field) in (None, "")


def test_decimal_citation_cannot_supply_an_integer_fragment():
    query, issues = apply([("max_minutes", 5, "1.5")], Query(kind="film"), "До 1.5 минут")
    assert query.max_minutes is None
    assert any(issue.kind == "unsupported_constraint" for issue in issues)


def test_evidence_that_names_a_different_value_is_a_contradiction():
    query, issues = apply([("level", "начальный", "продвинутый")], Query(kind="course"), "Ищу курс продвинутый")
    assert query.level is None
    assert any(issue.kind == "ambiguity" for issue in issues)


def test_negated_enum_evidence_cannot_set_the_opposite_constraint():
    query, issues = apply([("tone", "мрачный", "не мрачный")], Query(kind="film"), "Что-нибудь не мрачное")
    assert query.tone is None
    assert any(issue.kind == "ambiguity" for issue in issues)


def test_citation_absent_from_the_message_is_refused():
    query, issues = apply([("genre", "комедия", "комедия")], Query(kind="film"), "Хочу драму")
    assert query.genre is None and issues


def test_evidence_matching_several_values_stays_ambiguous():
    class AmbiguousQuery(BaseModel):
        intent: str = "discovery"
        material: Literal["small", "large"] | None = None

    adapter = SchemaRequestAdapter(
        AmbiguousQuery,
        aliases={"material": {"материал": "small", "материала": "large"}},
        resolver=InflectionalResolver(LanguageProfile(min_stem=4, ending_tolerance=3)),
    )
    # «материалом» is an inflected form of BOTH declared aliases, so the citation
    # cannot prove which value was meant: ambiguity, not a coin flip.
    query, issues = adapter.apply(
        StructuredRequest(updates=[update("material", "small", "материалом")]), AmbiguousQuery(), "Нужен шкаф с материалом"
    )
    assert query.material is None
    assert any(issue.kind == "ambiguity" for issue in issues)


# --- Cross-domain: the mechanism is schema-driven, not a course/film dictionary


class FurnitureQuery(BaseModel):
    intent: str = "discovery"
    material: Literal["wood", "steel"] | None = None
    room: Literal["гостиная", "спальня"] | None = None
    max_width: int | None = None


def furniture_adapter(resolver=None):
    return SchemaRequestAdapter(
        FurnitureQuery,
        aliases={
            "material": {"дерево": "wood", "деревянный": "wood", "сталь": "steel", "металл": "steel"},
            "room": {"для гостиной": "гостиная", "в спальню": "спальня"},
        },
        numeric_units={"max_width": {"см": 1, "м": 100}},
        resolver=resolver,
    )


def test_cross_domain_inflected_evidence_resolves_without_core_changes():
    adapter = furniture_adapter()
    query, issues = adapter.apply(
        StructuredRequest(updates=[update("material", "wood", "деревянный"), update("max_width", 200, "2 м")]),
        FurnitureQuery(),
        "Нужен деревянный стол, 2 м",
    )
    assert not issues
    assert query.material == "wood" and query.max_width == 200


def test_cross_domain_unknown_vocabulary_stays_unresolved():
    adapter = furniture_adapter()
    query, issues = adapter.apply(
        StructuredRequest(updates=[update("material", "wood", "экологичный")]), FurnitureQuery(), "Нужен экологичный стол"
    )
    assert query.material is None and issues


def test_source_text_cannot_match_inside_a_larger_word():
    """A cited schema surface needs a token boundary in the actual message."""

    class FormatQuery(BaseModel):
        intent: str = "discovery"
        format: Literal["art", "science"] | None = None

    adapter = SchemaRequestAdapter(FormatQuery)
    query, issues = adapter.apply(StructuredRequest(updates=[update("format", "art", "art")]), FormatQuery(), "Find a cartoon")

    assert query.format is None
    assert any(issue.kind == "ambiguity" for issue in issues)


def test_cross_domain_language_profile_controls_negated_enum_evidence():
    class StyleQuery(BaseModel):
        intent: str = "discovery"
        style: Literal["warm", "cool"] | None = None

    adapter = SchemaRequestAdapter(StyleQuery, language_profile=LanguageProfile(negation_tokens=("not",)))
    query, issues = adapter.apply(StructuredRequest(updates=[update("style", "warm", "not warm")]), StyleQuery(), "Find something not warm")

    assert query.style is None
    assert any(issue.kind == "ambiguity" for issue in issues)


def test_injected_exact_resolver_restores_literal_only_behaviour():
    """The resolver is a swappable boundary: injecting the historical literal
    resolver must reject an inflected citation that the default accepts."""
    only_base = SchemaRequestAdapter(FurnitureQuery, aliases={"material": {"дерево": "wood"}}, resolver=ExactSurfaceResolver())
    rejected, issues = only_base.apply(
        StructuredRequest(updates=[update("material", "wood", "деревянный")]), FurnitureQuery(), "Нужен деревянный стол"
    )
    assert rejected.material is None and issues

    accepted, no_issues = furniture_adapter().apply(
        StructuredRequest(updates=[update("material", "wood", "деревянный")]), FurnitureQuery(), "Нужен деревянный стол"
    )
    assert not no_issues and accepted.material == "wood"


def test_resolver_receives_only_schema_and_runtime_inputs():
    """Anti-overfit invariant: resolution must not be able to see ground truth."""
    captured = {}

    class RecordingResolver:
        def resolve(self, spec, value, source_text):
            captured["spec_fields"] = sorted(spec.__dict__)
            captured["args"] = sorted([type(value).__name__, type(source_text).__name__])
            return InflectionalResolver().resolve(spec, value, source_text)

    adapter = SchemaRequestAdapter(FurnitureQuery, aliases={"material": {"дерево": "wood"}}, resolver=RecordingResolver())
    adapter.apply(StructuredRequest(updates=[update("material", "wood", "дерево")]), FurnitureQuery(), "Нужен дерево стол")
    assert captured["spec_fields"] == ["aliases", "enum", "name", "numeric_units", "scalar_aliases"]
    assert captured["args"] == ["str", "str"]


def test_language_profile_can_disable_morphological_tolerance():
    """A domain with no inflection gets exact surface matching only."""
    strict = LanguageProfile(min_stem=4, ending_tolerance=0)
    assert not strict.same_lexeme("комедии", "комедия")
    default = LanguageProfile()
    assert default.same_lexeme("комедии", "комедия")


def test_same_lexeme_never_links_unrelated_words():
    profile = LanguageProfile()
    for a, b in [("middle", "начальный"), ("middle", "продвинутый"), ("не", "мрачный"), ("программа", "курс"), ("курс", "фильм")]:
        assert not profile.same_lexeme(a, b), f"{a}~{b} must not look like one lexeme"


def test_inflectional_suffix_length_is_bounded():
    assert LanguageProfile().same_lexeme("course", "courses")
    assert not LanguageProfile().same_lexeme("course", "coursework")
    assert not LanguageProfile(ending_tolerance=0).same_lexeme("course", "courses")


def test_multiword_surface_requires_every_word_to_match():
    profile = LanguageProfile()
    assert profile.matches_surface("по машинному обучению", "машинное обучение")
    assert not profile.matches_surface("по машинному", "машинное обучение")


# --- Unit-level resolver and scalar contract ---------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("«Декоратор в проде»", "Декоратор в проде"),
        ('"Декоратор в проде"', "Декоратор в проде"),
        ("Декоратор в проде", "Декоратор в проде"),
        ("  «Титул»  ", "Титул"),
    ],
)
def test_strip_enclosing_quotes(text, expected):
    assert strip_enclosing_quotes(text) == expected


def test_comparison_key_unifies_normalization():
    assert comparison_key("«Комедия»") == comparison_key("комедия") == "комедия"
    assert comparison_key("Машинное обучение ") == "машинное обучение"
    assert comparison_key("Ёлка") == comparison_key("елка") == "елка"
    # Internal spacing is deliberately preserved: collapsing it would make a
    # negation indistinguishable from the word it negates.
    assert comparison_key("не мрачный") != comparison_key("мрачный")


def test_resolve_scalar_refuses_unsupported_type():
    spec = FieldSpec(name="x")
    assert resolve_scalar(spec, 1.5, "полтора").status == "unsupported"


def test_field_spec_surfaces_collects_enum_and_aliases():
    spec = FieldSpec(name="kind", enum={"course": "course"}, aliases={"курс": "course", "программа": "series"})
    assert spec.surfaces("course") == {"course", "курс"}
    assert spec.surfaces("series") == {"программа"}
