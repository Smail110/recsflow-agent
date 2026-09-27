from __future__ import annotations

import copy
from pathlib import Path

import pytest
from evals.observable_product import (
    assess_turn,
    audit_recommendation_text,
    canonical_hash,
    compare_reports,
    evaluate_report,
    safe_read_json,
    validate_cohort,
)

from recagent.catalog import generate_catalog
from recagent.models import Item


@pytest.fixture
def catalog():
    return {
        "a": Item(id="a", title="A", kind="film", genre="комедия", tone="нейтральный", minutes=90, quality=0.5, description="Invented"),
        "b": Item(id="b", title="B", kind="course", genre="python", level="начальный", minutes=30, quality=0.5, description="Invented"),
    }


def expected(**kwargs):
    return {"user": "Подберите фильм", "expected_state": "recommend", "recommendation_exists": True, "spoken": {"kind": "film"}, **kwargs}


def observed(**kwargs):
    return {"index": 0, "user": "Подберите фильм", "actual_state": "recommend", "shown_ids": ["a"], "final_query": {}, **kwargs}


def fixture_report(turn=None, result=None):
    data = {"split": "dev", "origin": "synthetic_code", "dialogues": [{"id": "arbitrary", "turns": [turn or expected()]}]}
    report = {
        "complete": True,
        "cohort": {"sha256": "frozen"},
        "configuration": {},
        "model_identity_before": {"digest": "same"},
        "model_identity_after": {"digest": "same"},
        "dialogues": [{"id": "arbitrary", "success": True, "turns": [result or observed()]}],
        "summary": {"full_dialog_successes": 1},
    }
    seal(report)
    return data, report


def seal(report):
    report["report_sha256"] = canonical_hash({k: v for k, v in report.items() if k != "report_sha256"})


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"shown_ids": ["missing"]}, "unknown_id"),
        ({"shown_ids": ["a", "a"]}, "duplicate_ids"),
        ({"shown_ids": []}, "recommendation_presence"),
        ({"shown_ids": ["b"]}, "spoken_constraint_violation"),
        ({"actual_state": "clarify", "shown_ids": []}, "state_mismatch"),
    ],
)
def test_negative_response_controls(catalog, changes, failure):
    verdict = assess_turn(expected(), observed(**changes), catalog)
    assert not verdict["pass"]
    assert failure in verdict["failures"]


@pytest.mark.parametrize("annotation", [{"exact_recommendation_ids": ["b"]}, {"forbidden_recommendation_ids": ["a"]}])
def test_exact_forbidden_ids_are_applied_uniformly(catalog, annotation):
    assert not assess_turn(expected(**annotation), observed(), catalog)["pass"]


def test_query_wrong_does_not_replace_visible_truth(catalog):
    result = assess_turn(expected(expected_query={"kind": "film"}), observed(final_query={"kind": "course"}), catalog)
    assert result["pass"]
    assert result["active_query_checks_separate"] == ["query:kind"]
    assert not assess_turn(expected(), observed(shown_ids=["b"], final_query={"kind": "film"}), catalog)["pass"]


def test_not_dark_does_not_force_light(catalog):
    assert assess_turn(expected(spoken={"excluded_tones": ["мрачный"]}), observed(), catalog)["pass"]


def test_explicit_allowed_state_and_committed_optional_query(catalog):
    turn = {
        "user": "Фильм",
        "allowed_states": ["clarify", "recommend"],
        "presence_by_state": {"clarify": False, "recommend": True},
        "commit_policy": "optional",
    }
    assert assess_turn(turn, observed(actual_state="clarify", shown_ids=[], final_query={"kind": "film"}), catalog)["pass"]


def test_disjoint_previous_outputs(catalog):
    data, report = fixture_report()
    data["dialogues"][0]["turns"].append(expected(disjoint_from_turns=[0]))
    report["dialogues"][0]["turns"].append(observed(index=1))
    seal(report)
    scored = evaluate_report(data, report, cohort_sha256="frozen", catalog=catalog)
    assert scored["lanes"]["observable_delivery"]["turns"] == 2
    assert scored["rows"][0]["turns"][1]["failures"] == ["repeated_previous_ids"]


@pytest.mark.parametrize("mutation", ["cohort", "text", "missing", "extra", "duplicate", "index", "legacy_summary", "tamper"])
def test_report_alignment_and_hash_fail_closed(catalog, mutation):
    data, report = fixture_report()
    if mutation == "cohort":
        report["cohort"]["sha256"] = "other"
    elif mutation == "text":
        report["dialogues"][0]["turns"][0]["user"] = "Changed"
    elif mutation == "missing":
        report["dialogues"][0]["turns"] = []
    elif mutation in {"extra", "duplicate"}:
        report["dialogues"].append(copy.deepcopy(report["dialogues"][0]))
    elif mutation == "index":
        report["dialogues"][0]["turns"][0]["index"] = False
    elif mutation == "legacy_summary":
        report["summary"]["full_dialog_successes"] = 0
    if mutation != "tamper":
        seal(report)
    else:
        report["complete"] = "true"
    with pytest.raises(ValueError):
        evaluate_report(data, report, cohort_sha256="frozen", catalog=catalog)


def test_comparison_rejects_unannounced_configuration_change(catalog):
    data, baseline = fixture_report()
    candidate = copy.deepcopy(baseline)
    candidate["configuration"] = {"new": True}
    seal(candidate)
    with pytest.raises(ValueError, match="configuration"):
        compare_reports(data, baseline, candidate, cohort_sha256="frozen", catalog=catalog)


@pytest.mark.parametrize("name", ["BLIND.json", "final_holdout/data.json", "final-holdout.json"])
def test_closed_paths_rejected_before_read(monkeypatch, name):
    def forbidden_read(*_args):
        pytest.fail("Closed file read attempted")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="Closed-data"):
        safe_read_json(Path(name))


def test_grounding_missing_coverage_is_unknown(catalog):
    assert audit_recommendation_text(observed(recommendation_evidence=[]), catalog)["claims"] is None


@pytest.mark.skipif(
    not Path("artifacts/evidence-repair-2026-09-23/after-ranking-e7.json").exists(), reason="Historical replay artifact is not bundled"
)
def test_frozen_open_dev_full_replay():
    data, digest = safe_read_json(Path("data/product_llm_first_dev.json"))
    report, _ = safe_read_json(Path("artifacts/evidence-repair-2026-09-23/after-ranking-e7.json"))
    scored = evaluate_report(data, report, cohort_sha256=digest, catalog={i.id: i for i in generate_catalog(42)})
    assert scored["lanes"]["legacy_recorded"]["dialogues_pass"] == 15
    assert scored["lanes"]["observable_delivery"] == {"dialogues_pass": 17, "dialogues": 20, "turns_pass": 22, "turns": 25}


def test_new_contract_validates_without_inference():
    data, _ = safe_read_json(Path("data/product_contract_ru_v1.json"))
    validate_cohort(data)


def test_rules_runner_does_not_pass_truth_to_agent():
    from scripts.evaluate_product_contract import run_cases

    from recagent.parsing import OllamaClient

    data = {
        "configuration": {"question_policy": "adaptive", "max_questions": 3},
        "dialogues": [{"id": "invented", "turns": [expected(spoken={"kind": "THIS IS NOT VALID TRUTH"})]}],
    }
    # Deliberately invalid labels: the execution boundary receives only user text,
    # and does not validate/pass these labels to any product component.
    rows = run_cases(data, generate_catalog(137), mode="rules", implementation="baseline-v1", client_factory=OllamaClient)
    assert len(rows[0]["turns"]) == 1
    assert rows[0]["turns"][0]["llm_calls"] == 0
    assert rows[0]["turns"][0]["structured_calls"] == []


@pytest.mark.parametrize("recorded", [None, "0" * 64])
def test_contract_catalog_hash_binding_fails_closed(catalog, recorded):
    data, report = fixture_report()
    data["split"] = "contract"
    report["catalog_sha256"] = recorded
    seal(report)
    with pytest.raises(ValueError, match="catalog hash"):
        evaluate_report(data, report, cohort_sha256="frozen", catalog=catalog)


def test_catalog_hash_binding_stays_required_in_cross_system_comparison(catalog):
    data, report = fixture_report()
    data["split"] = "contract"
    report["catalog_sha256"] = canonical_hash([i.model_dump(mode="json") for i in catalog.values()])
    seal(report)
    candidate = copy.deepcopy(report)
    candidate["catalog_sha256"] = "0" * 64
    seal(candidate)
    with pytest.raises(ValueError, match="catalog hash"):
        compare_reports(data, report, candidate, cohort_sha256="frozen", catalog=catalog, require_matching_run_configuration=False)


def test_historical_missing_catalog_hash_is_explicitly_unverified(catalog):
    data, report = fixture_report()
    result = evaluate_report(data, report, cohort_sha256="frozen", catalog=catalog)
    assert result["catalog_identity"]["recorded_sha256"] is None
    assert result["catalog_identity"]["verified"] is False


def test_transport_settings_match_actual_wire(respx_mock):
    import json

    import httpx
    from scripts.evaluate_product_contract import transport_configuration

    from recagent.models import Query
    from recagent.parsing import OllamaClient

    route = respx_mock.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(200, json={"message": {"content": "{}"}}))
    OllamaClient(base_url="http://localhost:11434").structured(Query, "generic", {"message": "invented"})
    body = json.loads(route.calls[0].request.content)
    settings = transport_configuration()
    assert settings == {key: body[key] for key in ("options", "think", "stream")}


def test_partial_run_preserves_completed_turns(monkeypatch):
    from scripts import evaluate_product_contract as runner

    from recagent.parsing import OllamaClient

    original = runner.make_agent

    def failing_agent(*args):
        agent = original(*args)
        chat = agent.chat
        calls = 0

        def fail_second(request):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("controlled failure")
            return chat(request)

        agent.chat = fail_second
        return agent

    monkeypatch.setattr(runner, "make_agent", failing_agent)
    rows = []
    data = {
        "configuration": {"question_policy": "adaptive", "max_questions": 3},
        "dialogues": [{"id": "invented", "turns": [expected(), expected()]}],
    }
    with pytest.raises(RuntimeError, match="controlled failure"):
        runner.run_cases(data, generate_catalog(137), mode="rules", implementation="baseline-v1", client_factory=OllamaClient, rows=rows)
    assert len(rows) == 1
    assert len(rows[0]["turns"]) == 1
