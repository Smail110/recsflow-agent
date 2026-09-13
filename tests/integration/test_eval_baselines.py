import json

import pytest
from evals.scenarios import generate_scenarios
from scripts import evaluate as legacy
from scripts import evaluate_baselines as runner

from recagent.catalog import generate_catalog
from recagent.models import ChatResponse, Query
from recagent.providers import DemoProvider


@pytest.fixture(scope="module")
def dataset():
    catalog = generate_catalog()
    scenarios, _ = generate_scenarios("dev", 30, catalog=catalog)
    return catalog, scenarios


def test_all_controls_use_frozen_criteria(dataset):
    catalog, scenarios = dataset
    before = [scenario.model_dump_json() for scenario in scenarios]
    for scenario in scenarios:
        result = runner.run_case(scenario, "oracle", catalog)
        assert result["success"], result
        assert runner.run_case(scenario, "random", catalog) == runner.run_case(scenario, "random", catalog)
    assert before == [scenario.model_dump_json() for scenario in scenarios]


def test_platform_baseline_never_receives_spoken_constraints(dataset, monkeypatch):
    catalog, scenarios = dataset
    calls = []
    original = DemoProvider.retrieve

    def retrieve(self, user_id, query, limit=100):
        calls.append(query.model_dump())
        return original(self, user_id, query, limit)

    monkeypatch.setattr(DemoProvider, "retrieve", retrieve)
    runner.run_case(scenarios[0], "platform_quality", catalog)
    assert calls == [Query().model_dump()]


def test_current_replays_each_utterance_once_and_never_passes_theta(dataset, monkeypatch):
    catalog, scenarios = dataset
    scenario = next(s for s in scenarios if len(s.turns) > 1)
    requests = []

    class RecordingAgent:
        def __init__(self, provider, mode):
            assert isinstance(provider, DemoProvider)
            assert mode == "rules"

        def chat(self, request):
            requests.append(request.model_dump())
            return ChatResponse(
                session_id="isolated", state="clarify", message="Уточните формат", query=Query(),
                mode="rules", latency_ms=1, llm_calls=0, llm_calls_total=0,
                llm_tokens=0, clarification_count=len(requests),
            )

    monkeypatch.setattr(runner, "Agent", RecordingAgent)
    result = runner.run_case(scenario, "current", catalog)
    assert [request["message"] for request in requests] == [turn.utterance for turn in scenario.turns]
    assert all(set(request) == {"message", "user_id", "session_id", "explanation_tone", "explanation_length"} for request in requests)
    assert requests[0]["session_id"] is None
    assert requests[1]["session_id"] == "isolated"
    assert not result["success"]
    assert result["state"] == "clarify"


def test_short_slates_do_not_artificially_improve_success(dataset, monkeypatch):
    catalog, scenarios = dataset
    scenario = next(s for s in scenarios if len(s.criteria.ceiling_ids) == 5)
    monkeypatch.setattr(DemoProvider, "retrieve", lambda *_args, **_kwargs: scenario.criteria.ceiling_ids[:1])
    result = runner.run_case(scenario, "platform_quality", catalog)
    assert not result["final_success"]
    assert "incomplete_slate" in result["failures"]


def test_end_to_end_report_has_provenance_and_serializes():
    result = runner.evaluate(size=8, resamples=30)
    assert result["dataset"]["size"] == 8
    assert len(result["dataset"]["scenarios_sha256"]) == 64
    assert len(result["source"]["python_sources_sha256"]) == 64
    assert result["runs"]["oracle"]["summary"]["success_rate"] == 1
    assert result["runs"]["current"]["summary"]["llm_calls"] == 0
    assert result["runs"]["random"]["summary"]["invalid_evidence_rate"] is None
    assert result["paired_final_success"]["current_minus_random"]["resampling_unit"] == "user_id"
    json.dumps(result, allow_nan=False)


def test_legacy_evaluation_validates_against_real_history(dataset, monkeypatch):
    catalog, _ = dataset
    history_id = catalog[0].id
    captured = []

    def validate(evidence, item, query, history, seed):
        captured.append([entry.id for entry in history])
        return True

    monkeypatch.setattr(DemoProvider, "history", lambda _self, _user_id: [history_id])
    monkeypatch.setattr(legacy, "validate_evidence", validate)
    legacy.evaluate(limit=1)
    assert captured
    assert all(ids == [history_id] for ids in captured)


def test_sanity_gate_rejects_broken_controls():
    oracle = [{"scenario_id": str(i), "state": "recommend", "success": True} for i in range(20)]
    random_rows = [{"scenario_id": str(i), "final_success": False} for i in range(20)]
    result = {"runs": {"oracle": {"records": oracle}, "random": {"records": random_rows}}}
    runner.check_controls(result)
    for row in random_rows[:3]:
        row["final_success"] = True
    with pytest.raises(ValueError, match="random control"):
        runner.check_controls(result)
    oracle[0]["success"] = False
    with pytest.raises(ValueError, match="oracle ceiling"):
        runner.check_controls(result)
