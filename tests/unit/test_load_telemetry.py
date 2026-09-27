from collections import deque

import httpx
import pytest
from scripts import load_test


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("failure", request=None, response=self)

    def json(self):
        return self.payload


class _Client:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, *_args, **_kwargs):
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_client(monkeypatch, outcomes):
    queue = deque(outcomes)
    monkeypatch.setattr(load_test.httpx, "Client", lambda **_kwargs: _Client(queue))


def _payload(*, mode="ollama", calls=1, tokens=10, usage=None, recommendations=None, degradation="FULL"):
    return {
        "recommendations": recommendations if recommendations is not None else [{"id": "x"}],
        "mode": mode,
        "llm_calls": calls,
        "llm_tokens": tokens,
        "llm_usage": usage,
        "degradation": degradation,
    }


def test_aggregate_usage_does_not_count_multi_call_request_as_successful_llm_calls(monkeypatch):
    _patch_client(
        monkeypatch,
        [_Response(_payload(calls=2, tokens=22, usage={"inference_seconds": 1.5}))],
    )

    report = load_test.run("http://test", count=1, workers=1)

    assert report["agent_llm_calls"] == 2
    assert report["requests_with_llm_usage"] == 1
    assert "successful_llm_calls" not in report
    assert report["inference_seconds"] == 1.5


def test_missing_usage_fields_are_unknown_not_zero(monkeypatch):
    _patch_client(monkeypatch, [_Response({"recommendations": [{"id": "x"}], "mode": "ollama"})])
    report = load_test.run("http://test", count=1, workers=1)
    assert report["results"][0]["llm_calls"] is None
    assert report["results"][0]["llm_tokens"] is None
    assert report["agent_tokens"] is None
    assert report["inference_seconds"] is None
    assert report["usage_coverage"]["requests_with_unknown_llm_calls"] == 1


def test_connection_error_and_mixed_modes_are_counted_without_sorting_failure(monkeypatch):
    _patch_client(
        monkeypatch,
        [
            _Response(_payload(mode=None, calls=0, tokens=0, usage=None, recommendations=[], degradation="NO_LLM")),
            httpx.ConnectError("offline"),
            _Response(_payload(mode="rules_fallback", calls=0, tokens=0, usage=None, degradation="NO_LLM")),
        ],
    )

    report = load_test.run("http://test", count=3, workers=1)

    assert report["http_successes"] == 2
    assert report["http_errors"] == 1
    assert report["successes"] == 1
    assert report["mode_counts"] == {"rules_fallback": 1, "unknown": 2}
    assert report["agent_llm_calls"] == 0
    assert report["inference_seconds"] is None


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 10}, {"inference_seconds": None}])
def test_missing_usage_keeps_inference_and_cost_unknown_when_calls_attempted(monkeypatch, usage):
    _patch_client(monkeypatch, [_Response(_payload(mode="rules_fallback", calls=2, tokens=0, usage=usage, degradation="NO_LLM"))])

    report = load_test.run("http://test", count=1, workers=1, gpu_hour_cost=30)

    assert report["agent_llm_calls"] == 2
    assert report["requests_with_llm_usage"] == int(bool(usage))
    assert report["fallback_requests"] == 1
    assert report["inference_seconds"] is None
    assert report["monetary_cost"] is None
    if usage:
        assert report["agent_tokens"] == 0
    else:
        assert report["agent_tokens"] is None
    assert report["observed_agent_tokens"] == 0
    assert report["observed_inference_seconds"] == 0
    assert report["usage_coverage"]["requests_missing_inference_seconds"] == 1


def test_measured_zero_is_preserved_and_clarify_is_http_success(monkeypatch):
    _patch_client(
        monkeypatch,
        [_Response(_payload(calls=1, tokens=0, usage={"inference_seconds": 0}, recommendations=[]))],
    )

    report = load_test.run("http://test", count=1, workers=1, gpu_hour_cost=30)

    assert report["http_successes"] == 1
    assert report["recommendation_successes"] == 0
    assert report["successes"] == 0
    assert report["inference_seconds"] == 0
    assert report["monetary_cost"] == 0
    assert report["usage_coverage"]["request_inference_complete"] is True


def test_rules_without_attempts_can_have_zero_cost_without_usage(monkeypatch):
    payload = _payload(mode="rules", calls=0, tokens=0)
    del payload["llm_usage"]
    _patch_client(monkeypatch, [_Response(payload)])

    report = load_test.run("http://test", count=1, workers=1, gpu_hour_cost=30)

    assert report["inference_seconds"] == 0
    assert report["monetary_cost"] == 0
    assert report["usage_coverage"]["requests_with_zero_llm_calls"] == 1
    assert report["usage_coverage"]["requests_with_inference_seconds"] == 0
    assert report["usage_coverage"]["request_inference_complete"] is True


def test_partial_usage_preserves_observed_total_without_claiming_complete_cost(monkeypatch):
    measured = _payload(calls=2, tokens=20, usage={"inference_seconds": 3.6})
    missing = _payload(mode="rules_fallback", calls=2, usage={}, degradation="NO_LLM")
    _patch_client(monkeypatch, [_Response(measured), _Response(missing)])

    report = load_test.run("http://test", count=2, workers=1, gpu_hour_cost=30)

    assert report["agent_llm_calls"] == 4
    assert report["requests_with_llm_usage"] == 1
    assert report["observed_inference_seconds"] == 3.6
    assert report["observed_agent_tokens"] == 20
    assert report["agent_tokens"] is None
    assert report["observed_monetary_cost"] == pytest.approx(0.03)
    assert report["inference_seconds"] is None
    assert report["monetary_cost"] is None
    assert report["usage_coverage"]["requests_with_inference_seconds"] == 1
    assert report["usage_coverage"]["requests_missing_inference_seconds"] == 1
    assert report["usage_coverage"]["request_inference_complete"] is False


def test_http_error_does_not_become_success_or_zero_usage(monkeypatch):
    _patch_client(monkeypatch, [_Response({}, status_code=503)])

    report = load_test.run("http://test", count=1, workers=1, gpu_hour_cost=30)

    assert report["http_successes"] == report["recommendation_successes"] == 0
    assert report["http_errors"] == 1
    assert report["results"][0]["error"] == "HTTPStatusError"
    assert report["inference_seconds"] is None
    assert report["monetary_cost"] is None
    assert report["usage_coverage"]["requests_with_unknown_llm_calls"] == 1
