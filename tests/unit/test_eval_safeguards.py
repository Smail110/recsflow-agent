import pytest
from evals.metrics import clustered_success_interval, paired_success_interval, summarize, wilson_interval
from evals.scenarios import generate_scenarios
from scripts.evaluate_baselines import run_case

from recagent.catalog import generate_catalog
from recagent.models import ChatResponse, Query, Recommendation
from recagent.providers import DemoProvider


@pytest.fixture(scope="module")
def cases():
    catalog = generate_catalog()
    scenarios, _ = generate_scenarios("dev", 30, catalog=catalog)
    return catalog, next(s for s in scenarios if len(s.criteria.ceiling_ids) == 5)


@pytest.mark.parametrize(
    "mutation,reason", [("duplicate", "duplicate_item_id"), ("fake", "unknown_item_id"), ("short", "incomplete_slate")]
)
def test_manipulated_provider_slates_fail_without_crashing(cases, monkeypatch, mutation, reason):
    catalog, scenario = cases
    ids = scenario.criteria.ceiling_ids
    malicious = {"duplicate": [ids[0]] * 5, "fake": ["missing-id"] * 5, "short": ids[:1]}[mutation]
    monkeypatch.setattr(DemoProvider, "retrieve", lambda *_args, **_kwargs: malicious)
    result = run_case(scenario, "platform_quality", catalog)
    assert not result["success"] and not result["final_success"]
    assert reason in result["failures"]


def test_wilson_at_boundaries_has_nonzero_uncertainty():
    assert wilson_interval(0, 100)[1] == pytest.approx(0.0369934982)
    assert wilson_interval(100, 100)[0] == pytest.approx(0.9630065018)
    assert wilson_interval(50, 100) == pytest.approx([0.40383153, 0.59616847])


@pytest.mark.parametrize("counts", [(1.5, 2), (1, 2.5), (True, 2), (1, True)])
def test_wilson_rejects_noninteger_counts(counts):
    with pytest.raises(ValueError, match="integers"):
        wilson_interval(*counts)


def test_quality_metrics_report_separate_denominators(cases, monkeypatch):
    catalog, scenario = cases
    monkeypatch.setattr(DemoProvider, "retrieve", lambda *_args, **_kwargs: ["missing-id"])
    row = run_case(scenario, "platform_quality", catalog)
    result = summarize([row])
    assert result["mean_acceptable_fraction_when_shown"] == 0
    assert result["mean_utility_when_shown"] is None
    assert result["slates_with_acceptable_fraction"] == 1
    assert result["slates_with_mean_utility"] == 0


def test_identical_systems_are_not_declared_equivalent():
    rows = [{"scenario_id": str(i), "user_id": str(i), "success": i % 2 == 0} for i in range(20)]
    result = paired_success_interval(rows, rows)
    assert result["discordant_scenarios"] == 0
    assert result["degenerate_bootstrap"]
    assert result["ci95"] == [0, 0]
    interval = clustered_success_interval(rows)
    assert interval["ci95"][0] < 0.5 < interval["ci95"][1]
    assert interval == clustered_success_interval(list(reversed(rows)))


@pytest.mark.parametrize("bad", ["false", "1", 0.5, 2, None, float("nan")])
def test_invalid_binary_outcomes_are_rejected(bad):
    row = {"scenario_id": "a", "user_id": "u", "success": bad}
    with pytest.raises(ValueError, match="success"):
        clustered_success_interval([row])
    with pytest.raises(ValueError, match="success"):
        paired_success_interval([row], [row])


def test_current_runner_rejects_unknown_recommendation_without_lookup_crash(cases, monkeypatch):
    catalog, scenario = cases
    fake = catalog[0].model_copy(update={"id": "missing-id"})

    class BrokenAgent:
        def __init__(self, **kwargs):
            pass

        def chat(self, request):
            return ChatResponse(
                session_id="test",
                state="recommend",
                message="Результат",
                query=Query(),
                recommendations=[Recommendation(item=fake, score=1, explanation="Подмена", evidence=[])],
                mode="rules",
                latency_ms=0,
                llm_calls=0,
                llm_calls_total=0,
                llm_tokens=0,
                clarification_count=0,
            )

    monkeypatch.setattr("scripts.evaluate_baselines.Agent", BrokenAgent)
    result = run_case(scenario, "current", catalog)
    assert not result["success"] and not result["final_success"]
    assert "unknown_item_id" in result["failures"]
