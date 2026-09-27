import pytest
from evals.metrics import paired_success_interval, percentile


def row(scenario, user, success):
    return {"scenario_id": scenario, "user_id": user, "success": success}


def test_paired_interval_keeps_profiles_together_and_is_reproducible():
    left = [row("a", "u1", False), row("b", "u1", False), row("c", "u2", False)]
    right = [row("a", "u1", True), row("b", "u1", True), row("c", "u2", True)]
    result = paired_success_interval(left, right)
    assert result["delta_success_rate"] == 1
    assert result["ci95"] == [1, 1]
    assert result["clusters"] == 2
    assert result == paired_success_interval(list(reversed(left)), right)


def test_interval_reports_uncertainty_when_profiles_disagree():
    left = [row("a", "u1", True), row("b", "u2", False)]
    right = [row("a", "u1", False), row("b", "u2", True)]
    result = paired_success_interval(left, right)
    assert result["delta_success_rate"] == 0
    assert result["ci95"] == [-1, 1]


@pytest.mark.parametrize("right", [[], [row("other", "u1", True)], [row("a", "u2", True)], [row("a", "u1", True)] * 2])
def test_unpaired_or_duplicate_results_are_rejected(right):
    with pytest.raises(ValueError):
        paired_success_interval([row("a", "u1", False)], right)


def test_percentile_has_explicit_empty_semantics():
    assert percentile([], 0.95) is None
    assert percentile([4, 1, 3, 2], 0.5) == 2
    with pytest.raises(ValueError):
        percentile([1], 2)
