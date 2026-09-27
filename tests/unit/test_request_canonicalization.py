"""The canonical workflow and legacy projection share one evidence boundary."""

from typing import Literal

import pytest
from pydantic import BaseModel

from recagent.domains.demo import request_adapter
from recagent.interpretation import ConstraintUpdate, InterpretationIssue, StructuredRequest
from recagent.request_mapping import SchemaRequestAdapter
from recagent.resolution import LanguageProfile


class FurnitureQuery(BaseModel):
    intent: str = "discovery"
    material: Literal["wood", "steel"] | None = None
    max_width: int | None = None
    assembled: bool | None = None


def adapter():
    return SchemaRequestAdapter(
        FurnitureQuery,
        aliases={"material": {"timber": "wood"}},
        numeric_units={"max_width": {"m": 100}},
        scalar_aliases={"assembled": {"assembled": True}},
        canonical_exclusions={"material"},
        language_profile=LanguageProfile(negation_tokens=("not",), ending_tolerance=0),
    )


@pytest.mark.parametrize("proposed", ["timber", None])
def test_normalize_resolves_alias_and_null_before_either_state_projection(proposed):
    request = StructuredRequest(
        intent="discovery",
        reset_constraints=True,
        updates=[ConstraintUpdate(field="material", value=proposed, source_text="timber")],
    )
    boundary = adapter()
    canonical = boundary.normalize(request, "Find timber furniture")

    assert canonical.updates[0].value == "wood"
    assert canonical.updates[0].source_text == "timber"
    assert canonical.intent == request.intent and canonical.reset_constraints
    assert request.updates[0].value == proposed
    assert boundary.normalize(canonical, "Find timber furniture") == canonical
    projected, issues = boundary.apply(request, FurnitureQuery(), "Find timber furniture")
    assert not issues and projected.material == canonical.updates[0].value


@pytest.mark.parametrize(
    ("field", "value", "source", "operation", "message"),
    [
        ("material", "wood", "ecological", "set", "Find ecological furniture"),
        ("material", "wood", "timber", "set", "Find steel furniture"),
        ("material", "wood", "timber", "exclude", "Find not timber furniture"),
        ("max_width", 900, "90", "set", "Width at most 90"),
        ("assembled", False, "assembled", "set", "Find assembled furniture"),
    ],
)
def test_normalize_keeps_rejected_proposal_as_issue_not_canonical_update(field, value, source, operation, message):
    request = StructuredRequest(updates=[ConstraintUpdate(field=field, value=value, source_text=source, operation=operation)])
    canonical = adapter().normalize(request, message)

    assert not canonical.updates
    assert any(issue.field == field for issue in canonical.issues)
    assert request.updates


def test_normalize_keeps_valid_updates_and_explicit_blockers_together():
    issue = InterpretationIssue(kind="ambiguity", field="assembled", value="maybe", message="Should it be assembled?")
    request = StructuredRequest(
        updates=[
            ConstraintUpdate(field="material", value="timber", source_text="timber"),
            ConstraintUpdate(field="max_width", value=200, source_text="2 m"),
        ],
        issues=[issue],
        clarification_required=True,
    )
    canonical = adapter().normalize(request, "Find timber furniture, 2 m wide, maybe assembled")

    assert [(update.field, update.value) for update in canonical.updates] == [("material", "wood"), ("max_width", 200)]
    assert canonical.issues == [issue]
    assert canonical.clarification_required


def test_normalize_preserves_canonical_only_exclusion_and_clear_operations():
    boundary = adapter()
    excluded = boundary.normalize(
        StructuredRequest(updates=[ConstraintUpdate(field="material", operation="exclude", value=None, source_text="not timber")]),
        "Find not timber furniture",
    )
    cleared = boundary.normalize(
        StructuredRequest(updates=[ConstraintUpdate(field="max_width", operation="clear", source_text="any width")]),
        "Find furniture with any width",
    )

    assert not excluded.issues and excluded.updates[0].value == "wood"
    assert excluded.updates[0].operation == "exclude"
    assert not cleared.issues and cleared.updates[0].value is None
    assert cleared.updates[0].operation == "clear"


@pytest.mark.parametrize(
    ("message", "intent", "expected"),
    [
        ("Подберите сериал с короткими сериями", "discovery", "series"),
        ("Нужен фильм без мрачного тона", "discovery", "film"),
        ("Порекомендуйте другой фильм, похожий на «Тишина в ожидании поезда»", "similar", "film"),
        ("Не нужен фильм, лучше сериал", "discovery", "series"),
        ("Курс по анализу данных", "discovery", "course"),
    ],
)
def test_explicit_format_can_be_recovered_from_user_words(message, intent, expected):
    recovered = request_adapter().recover_explicit_domain(StructuredRequest(intent=intent), message)
    assert recovered is not None
    assert recovered.field == "kind" and recovered.value == expected
    assert recovered.source_text in message


@pytest.mark.parametrize(
    "message",
    [
        "Посоветуй что-нибудь вроде «Фильм о небе»",
        "Не фильм и не сериал",
        "Выбираю между фильмом и сериалом",
    ],
)
def test_explicit_format_recovery_does_not_guess_from_titles_negations_or_competing_formats(message):
    assert request_adapter().recover_explicit_domain(StructuredRequest(), message) is None


def test_coordinated_exclusions_keep_separate_literal_value_evidence():
    message = "Нужен фильм без драмы и комедии"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="без драмы"),
        ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="без комедии"),
    ])
    boundary = request_adapter()
    reconciled = boundary.reconcile_operation_evidence(request, message)
    checked = boundary.normalize(reconciled, message)
    assert not checked.issues
    assert [(u.operation, u.value) for u in checked.updates] == [("exclude", "драма"), ("exclude", "комедия")]
    assert all(u.source_text in message for u in checked.updates)


def test_revoked_exclusion_requires_existing_ban_and_local_inclusion_cue():
    message = "Комедию опять разрешаю; запрет на драму остаётся"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="комедия", source_text="Комедию опять разрешаю"),
        ConstraintUpdate(field="genre", operation="exclude", value="драма", source_text="запрет на драму остаётся"),
    ])
    boundary = request_adapter()
    with_ban = boundary.reconcile_operation_evidence(request, message, excluded={("genre", "комедия")})
    assert [(u.operation, u.value) for u in boundary.normalize(with_ban, message).updates] == [
        ("include", "комедия"), ("exclude", "драма")
    ]
    without_ban = boundary.reconcile_operation_evidence(request, message)
    assert without_ban.updates[0].operation == "set"


def test_exclusion_cue_in_another_clause_cannot_license_value():
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="комедию")
    ])
    checked = request_adapter().normalize(request, "Исключите драму; комедию я люблю")
    assert not checked.updates and checked.issues


def test_keep_existing_positive_value_is_not_revocation_of_an_exclusion():
    boundary = request_adapter()
    update = ConstraintUpdate(field="genre", operation="include", value="приключения", source_text="приключения оставьте")
    request = StructuredRequest(updates=[update])
    retained = boundary.reconcile_operation_evidence(
        request, "Снимите ограничение длительности; приключения оставьте", retained={("genre", "приключения")}
    )
    assert retained.updates[0].operation == "set"
    assert not boundary.normalize(retained, "Снимите ограничение длительности; приключения оставьте").issues


def test_same_declared_value_with_positive_and_negative_uses_blocks_the_turn():
    boundary = request_adapter()
    message = "Нужен фильм-комедия, но комедии не предлагайте"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="комедия")
    ])
    checked = boundary.detect_polarity_conflicts(boundary.reconcile_operation_evidence(request, message), message)
    assert checked.clarification_required
    assert [(issue.kind, issue.field, issue.value) for issue in checked.issues] == [("conflict", "genre", "комедия")]


@pytest.mark.parametrize("operation", ["set", "include", "exclude"])
def test_contradictory_value_is_blocked_independently_of_model_operation(operation):
    boundary = request_adapter()
    message = "Нужна комедия, но без комедии"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation=operation, value="комедия", source_text="комедия")
    ])
    checked = boundary.detect_polarity_conflicts(boundary.reconcile_operation_evidence(request, message), message)
    assert checked.clarification_required
    assert any(issue.kind == "conflict" and issue.field == "genre" for issue in checked.issues)


@pytest.mark.parametrize(
    "message",
    [
        "Нужен фильм без комедии",
        "Нужен фильм без драмы и комедии",
        "Нужен фильм, похожий на «Комедия на станции», без комедии",
    ],
)
def test_negative_only_or_title_mention_is_not_a_polarity_conflict(message):
    boundary = request_adapter()
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="комедии")
    ])
    checked = boundary.detect_polarity_conflicts(boundary.reconcile_operation_evidence(request, message), message)
    assert not checked.issues


def test_keep_word_can_correct_a_mislabeled_exclusion_to_positive_set():
    boundary = request_adapter()
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="exclude", value="комедия", source_text="Оставьте комедии")
    ])
    corrected = boundary.reconcile_operation_evidence(request, "Оставьте комедии, нужен фильм")
    assert corrected.updates[0].operation == "set"
    assert not boundary.normalize(corrected, "Оставьте комедии, нужен фильм").issues


@pytest.mark.parametrize(
    ("message", "value", "allowed"),
    [
        ("Нужен фильм без драмы и с комедией", "комедия", False),
        ("Исключите драму и на комедию сделайте упор", "комедия", False),
        ("Исключите драму и на комедию переключитесь", "комедия", False),
        ("Исключите драму и комедию люблю", "комедия", False),
        ("Без драмы и комедию люблю", "комедия", False),
        ("Исключите драму и комедию хочу", "комедия", False),
        ("Нужна комедия и не нужна драма", "комедия", False),
        ("Не исключайте драму", "драма", False),
        ("Не надо исключать драму", "драма", False),
        ("Не нужно исключать комедию", "комедия", False),
        ("Не собираюсь исключать драму", "драма", False),
        ("Не хочу совсем исключать комедию", "комедия", False),
        ("Не хочу не комедию", "комедия", False),
        ("Без комедии не хочу", "комедия", False),
        ("Без комедии не надо", "комедия", False),
        ("Не надо не драму", "драма", False),
        ("Нужен фильм «Не исключайте драму»", "драма", False),
        ("Исключите драму или комедию", "драма", False),
        ("Исключите драму или комедию", "комедия", False),
        ("Исключить драму и комедию", "комедия", True),
        ("Комедию тоже исключите", "комедия", True),
    ],
)
def test_exclusion_requires_operator_scoped_to_value_outside_title(message, value, allowed):
    boundary = request_adapter()
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="exclude", value=value, source_text=value)
    ])
    normalized = boundary.normalize(boundary.reconcile_operation_evidence(request, message), message)
    assert bool(normalized.updates) is allowed
    assert bool(normalized.issues) is not allowed


@pytest.mark.parametrize(
    "message",
    [
        "Не советуй мне фильм", "Не нужно никакое кино", "Не хочу сегодня фильм",
        "Фильм не нужен", "Фильм мне не нужен", "Кино не хочу",
        "Фильм мне сегодня точно не нужен", "Фильм мне не очень нужен",
        "Фильм я не буду смотреть", "Мне фильм не нравится",
    ],
)
def test_negated_format_does_not_become_positive_format(message):
    assert request_adapter().recover_explicit_domain(StructuredRequest(intent="discovery"), message) is None


def test_negated_keep_does_not_turn_include_into_positive_set():
    boundary = request_adapter()
    message = "Не оставьте комедию"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="include", value="комедия", source_text="Не оставьте комедию")
    ])
    reconciled = boundary.reconcile_operation_evidence(request, message, retained={("genre", "комедия")})
    checked = boundary.normalize(reconciled, message)
    assert reconciled.updates[0].operation == "include"
    assert not checked.updates and checked.issues


def test_negated_permission_does_not_restore_exclusion():
    boundary = request_adapter()
    message = "Не планирую разрешать комедию"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="include", value="комедия", source_text="комедию")
    ])
    checked = boundary.normalize(boundary.reconcile_operation_evidence(request, message), message)
    assert not checked.updates and checked.issues


def test_negated_exclusion_cannot_become_a_required_genre():
    boundary = request_adapter()
    message = "Не исключайте комедию"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="комедия", source_text="комедию")
    ])
    corrected = boundary.reconcile_operation_evidence(request, message)
    checked = boundary.normalize(corrected, message)
    assert [(item.operation, item.value) for item in checked.updates] == [("include", "комедия")]
    assert not checked.issues


def test_negated_exclusion_of_another_value_does_not_change_positive_set():
    boundary = request_adapter()
    message = "Не исключайте драму; комедию хочу"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="комедия", source_text="комедию")
    ])
    corrected = boundary.reconcile_operation_evidence(request, message)
    assert corrected.updates[0].operation == "set"


def test_negated_exclusion_covers_conjoined_declared_values_without_required_genre():
    boundary = request_adapter()
    message = "Не исключайте комедию и драму"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="драма", source_text="драму")
    ])
    corrected = boundary.reconcile_operation_evidence(request, message)
    checked = boundary.normalize(corrected, message)
    assert [(item.operation, item.value) for item in checked.updates] == [("include", "драма")]
    assert not checked.issues


def test_postposed_negated_exclusion_covers_conjoined_values():
    boundary = request_adapter()
    message = "Комедию и драму не исключайте"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="драма", source_text="драму")
    ])
    checked = boundary.normalize(boundary.reconcile_operation_evidence(request, message), message)
    assert [(item.operation, item.value) for item in checked.updates] == [("include", "драма")]
    assert not checked.issues


def test_postposed_negated_exclusion_disjunction_blocks_set():
    boundary = request_adapter()
    message = "Комедию или драму не исключайте"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="драма", source_text="драму")
    ])
    checked = boundary.normalize(boundary.reconcile_operation_evidence(request, message), message)
    assert not checked.updates and checked.issues


def test_negated_exclusion_disjunction_does_not_choose_a_value():
    boundary = request_adapter()
    message = "Не исключайте комедию или драму"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="драма", source_text="драму")
    ])
    corrected = boundary.reconcile_operation_evidence(request, message)
    assert corrected.updates[0].operation == "set"
    checked = boundary.normalize(corrected, message)
    assert not checked.updates and checked.issues


def test_exclusion_disjunction_does_not_choose_a_required_value():
    boundary = request_adapter()
    message = "Исключите комедию или драму"
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="genre", operation="set", value="драма", source_text="драму")
    ])
    checked = boundary.normalize(boundary.reconcile_operation_evidence(request, message), message)
    assert not checked.updates and checked.issues
