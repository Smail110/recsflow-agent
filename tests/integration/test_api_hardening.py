"""Invented failure controls for public responses and request observations."""

import importlib
from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from recagent.api import create_app
from recagent.config.settings import Settings
from recagent.observability.logging import _redact_secrets
from recagent.providers import DemoProvider
from recagent.workflow import WorkflowAgent

SECRET = "invented-private-diagnostic-519"


@pytest.mark.parametrize("error", [RuntimeError, ValueError, KeyError])
def test_public_error_boundary_does_not_echo_arbitrary_service_details(error):
    agent = WorkflowAgent(mode="rules")

    def broken(_request):
        raise error(f"provider credential={SECRET}")

    agent.chat = broken
    client = TestClient(create_app(agent, settings=Settings(parse_mode="rules"), expose_diagnostics=False))
    response = client.post("/v1/chat", json={"message": "любой фильм", "user_id": "failure-control"})
    assert response.status_code >= 400
    assert response.headers["content-type"].startswith("application/problem+json")
    assert SECRET not in response.text
    assert response.headers["x-request-id"]


@pytest.mark.parametrize("fails", [False, True])
def test_public_readiness_preserves_status_without_exposing_provider_details(fails):
    class Provider(DemoProvider):
        async def readiness_probe(self):
            if fails:
                raise ConnectionError(f"endpoint credential={SECRET}")
            return f"internal endpoint credential={SECRET}"

    agent = WorkflowAgent(mode="rules", provider=Provider())
    app = create_app(agent, settings=Settings(parse_mode="rules"), expose_diagnostics=False)
    response = TestClient(app).get("/ready")
    assert response.status_code == (503 if fails else 200)
    assert response.json()["checks"]["recommendation_platform"]["status"] == ("unavailable" if fails else "ok")
    assert SECRET not in response.text


def test_nested_credential_redaction_preserves_counters_and_input():
    payload = {"provider": {"authorization": SECRET, "attempts": [{"access_token": SECRET, "input_tokens": 7}]}, "llm_tokens": 9}
    before = deepcopy(payload)
    output = _redact_secrets(None, "info", payload)
    assert output["provider"]["authorization"] == "[REDACTED]"
    assert output["provider"]["attempts"][0] == {"access_token": "[REDACTED]", "input_tokens": 7}
    assert output["llm_tokens"] == 9
    assert payload == before


@pytest.mark.parametrize("failure", ["provider", "unhandled", "validation"])
def test_every_failed_http_attempt_has_one_correlated_completion(failure, monkeypatch):
    middleware = importlib.import_module("recagent.api.middleware")
    events = []

    class Logger:
        def info(self, event, **values):
            events.append((event, values))

        def exception(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(middleware, "logger", Logger())

    class Provider(DemoProvider):
        def retrieve(self, *_args, **_kwargs):
            raise TimeoutError(SECRET)

    agent = WorkflowAgent(mode="rules", provider=Provider())
    if failure == "unhandled":

        def broken(_request):
            raise ArithmeticError(SECRET)

        agent.chat = broken
    client = TestClient(create_app(agent, settings=Settings(parse_mode="rules"), expose_diagnostics=False))
    payload = {"user_id": "failure-control", "message": "" if failure == "validation" else "Нужен фильм комедия"}
    response = client.post("/v1/chat", json=payload, headers={"X-Request-ID": "failure-rid"})
    assert response.status_code == {"provider": 503, "unhandled": 500, "validation": 422}[failure]
    assert SECRET not in response.text
    completions = [v for e, v in events if e == "http_request"]
    assert len(completions) == 1
    assert completions[0]["success"] is False
    assert completions[0]["status_code"] == response.status_code
    assert response.headers["x-request-id"] == "failure-rid"


def test_http_idempotent_replay_has_fresh_transport_telemetry_and_no_new_work():
    from recagent.interpretation import ConstraintUpdate, StructuredRequest
    from recagent.response_generation import EvidenceResponseGenerator

    class Interpreter:
        calls = 0

        def interpret(self, *_args, **_kwargs):
            self.calls += 1
            return StructuredRequest(
                updates=[
                    ConstraintUpdate(field="kind", value="film", source_text="фильм"),
                    ConstraintUpdate(field="genre", value="комедия", source_text="комедия"),
                ]
            ), 11

    class Provider(DemoProvider):
        calls = 0

        def retrieve(self, *args, **kwargs):
            self.calls += 1
            return super().retrieve(*args, **kwargs)

    interpreter, provider = Interpreter(), Provider()
    agent = WorkflowAgent(mode="ollama", interpreter=interpreter, provider=provider, response_generator=EvidenceResponseGenerator())
    client = TestClient(create_app(agent, settings=Settings(parse_mode="rules")))
    payload = {"message": "фильм комедия", "user_id": "replay-control", "message_id": str(uuid4())}
    first = client.post("/v1/chat", json=payload, headers={"X-Request-ID": "original-http"}).json()
    retry_response = client.post("/v1/chat", json=payload, headers={"X-Request-ID": "retry-http"})
    retry = retry_response.json()
    assert retry_response.status_code == 200
    assert retry["request_id"] == retry["telemetry"]["request_id"] == retry_response.headers["x-request-id"] == "retry-http"
    assert retry["telemetry"]["cache_hit"] is True
    assert retry["telemetry"]["original_request_id"] == "original-http"
    assert retry["llm_calls"] == retry["llm_tokens"] == 0
    assert retry["telemetry"]["llm_calls"] == retry["telemetry"]["tokens"] == 0
    assert retry["llm_usage"] == {} and retry["timings_ms"] == {}
    assert retry["latency_ms"] == retry["telemetry"]["latency_ms"]
    assert interpreter.calls == provider.calls == 1
    for field in ["recommendations", "query", "session_id", "mode", "degradation", "message", "llm_calls_total", "llm_tokens_total"]:
        assert retry[field] == first[field]
    session = agent.sessions[first["session_id"]]
    assert session.calls == 1 and session.tokens == 11
    assert first["llm_calls"] == 1 and first["llm_tokens"] == 11
    # A third retry still names the original execution, not the previous retry.
    third = client.post("/v1/chat", json=payload, headers={"X-Request-ID": "third-http"}).json()
    assert third["telemetry"]["original_request_id"] == "original-http"
    assert third["request_id"] == "third-http" and interpreter.calls == provider.calls == 1
