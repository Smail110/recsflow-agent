"""Tests for the degradation ladder.

These are the tests that make the task requirement verifiable:
"degradation to the classic Recsflow response on failure".
"""
from __future__ import annotations

import pytest

from recagent.resilience.degradation import (
    DegradationLevel,
    DegradationReason,
    full,
)


def test_levels_are_ordered_richest_to_poorest():
    assert DegradationLevel.FULL > DegradationLevel.NO_LLM > DegradationLevel.NO_EXPLAIN > DegradationLevel.UNAVAILABLE


@pytest.mark.parametrize(
    ("level", "expected_explanations", "expected_recommendations"),
    [
        (DegradationLevel.FULL, True, True),
        (DegradationLevel.NO_LLM, True, True),
        (DegradationLevel.NO_EXPLAIN, False, True),
        (DegradationLevel.UNAVAILABLE, False, False),
    ],
)
def test_capability_flags_match_the_level(level, expected_explanations, expected_recommendations):
    """NO_EXPLAIN must still return ids: that is the whole point of the ladder."""
    assert level.has_explanations is expected_explanations
    assert level.has_recommendations is expected_recommendations


def test_only_unavailable_is_terminal():
    for level in (DegradationLevel.FULL, DegradationLevel.NO_LLM, DegradationLevel.NO_EXPLAIN):
        assert not level.is_terminal
    assert DegradationLevel.UNAVAILABLE.is_terminal


def test_every_level_has_an_honest_user_message():
    messages = {level.user_message for level in DegradationLevel}
    assert len(messages) == len(DegradationLevel), "each level needs its own wording"
    for level in DegradationLevel:
        assert level.user_message.strip()


def test_no_explain_message_does_not_claim_explanations():
    """The wording must never promise what was not delivered."""
    message = DegradationLevel.NO_EXPLAIN.user_message
    assert "без объяснений" in message
    assert "проверены по данным каталога" not in message


def test_unavailable_message_does_not_pretend_to_serve():
    message = DegradationLevel.UNAVAILABLE.user_message
    assert "недоступен" in message
    assert "Подобрал" not in message


def test_full_decision_is_not_degraded():
    decision = full()
    assert decision.level is DegradationLevel.FULL
    assert not decision.is_degraded
    assert decision.reasons == ()


def test_downgrade_records_reason():
    decision = full().downgrade(DegradationLevel.NO_LLM, DegradationReason.LLM_UNREACHABLE)
    assert decision.level is DegradationLevel.NO_LLM
    assert decision.is_degraded
    assert decision.reasons == (DegradationReason.LLM_UNREACHABLE,)


def test_downgrade_accumulates_reasons_in_order():
    decision = (
        full()
        .downgrade(DegradationLevel.NO_LLM, DegradationReason.LLM_BUDGET_EXHAUSTED)
        .downgrade(DegradationLevel.NO_EXPLAIN, DegradationReason.METADATA_UNAVAILABLE)
    )
    assert decision.level is DegradationLevel.NO_EXPLAIN
    assert decision.reasons == (
        DegradationReason.LLM_BUDGET_EXHAUSTED,
        DegradationReason.METADATA_UNAVAILABLE,
    )


def test_upgrade_attempt_is_ignored():
    """Once degraded in a turn, a later richer level must not silently upgrade it.

    Otherwise a partially failed turn would be reported as complete.
    """
    degraded = full().downgrade(DegradationLevel.NO_EXPLAIN, DegradationReason.PROVIDER_UNREACHABLE)
    unchanged = degraded.downgrade(DegradationLevel.FULL, "should_not_apply")
    assert unchanged.level is DegradationLevel.NO_EXPLAIN
    assert unchanged.reasons == (DegradationReason.PROVIDER_UNREACHABLE,)


def test_same_level_downgrade_keeps_first_reason():
    decision = full().downgrade(DegradationLevel.NO_LLM, DegradationReason.LLM_UNREACHABLE)
    again = decision.downgrade(DegradationLevel.NO_LLM, DegradationReason.LLM_INVALID_STRUCTURE)
    assert again.level is DegradationLevel.NO_LLM
    assert again.reasons == (DegradationReason.LLM_UNREACHABLE,)


def test_decision_is_immutable():
    """downgrade returns a new object; the original must not be mutated."""
    original = full()
    derived = original.downgrade(DegradationLevel.NO_LLM, DegradationReason.LLM_UNREACHABLE)
    assert original.level is DegradationLevel.FULL
    assert original.reasons == ()
    assert derived is not original


def test_as_dict_is_json_friendly():
    payload = full().downgrade(DegradationLevel.NO_LLM, DegradationReason.LLM_UNREACHABLE).as_dict()
    assert payload == {"level": "NO_LLM", "reasons": ["llm_unreachable"]}
    import json

    json.dumps(payload)


def test_reasons_are_stable_metric_labels():
    """Reasons are used as metric labels, so they must be short snake_case identifiers."""
    for name in vars(DegradationReason):
        if name.startswith("_"):
            continue
        value = getattr(DegradationReason, name)
        assert isinstance(value, str)
        assert value == value.lower()
        assert " " not in value
        assert len(value) <= 40
