"""Integration tests for the HTTP surface.

These pin the contract a client integrates against: status codes, the RFC 7807
error envelope, correlation headers, and the rule that internal diagnostics never
reach a client-facing deployment.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from recagent.api import create_app
from recagent.config.settings import Settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(settings=Settings(), expose_diagnostics=True))


@pytest.fixture
def strict_client() -> TestClient:
    """Client-facing mode: diagnostics must be stripped."""
    return TestClient(create_app(settings=Settings(), expose_diagnostics=False))


CHAT = "/v1/chat"
COZY_DETECTIVE = "Хочу лёгкий детективный сериал, один сезон"


def _chat(client: TestClient, message: str, **kwargs):
    return client.post(CHAT, json={"message": message, "user_id": "demo", **kwargs})


def test_health_and_ready_are_ok(client: TestClient):
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert isinstance(ready.json(), dict)


def test_openapi_schema_is_published(client: TestClient):
    """The published schema is what a client codes against."""
    schema = client.get("/openapi.json").json()
    assert CHAT in schema["paths"]
    assert "/v1/feedback" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


def test_chat_returns_recommendations(client: TestClient):
    response = _chat(client, COZY_DETECTIVE)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "recommend"
    assert body["recommendations"]
    for rec in body["recommendations"]:
        assert rec["explanation"]
        assert rec["evidence"], "every recommendation must carry verifiable evidence"


def test_correlation_headers_are_present(client: TestClient):
    response = _chat(client, COZY_DETECTIVE)
    assert response.headers.get("x-request-id")
    assert response.headers.get("x-process-time-ms")
    float(response.headers["x-process-time-ms"])


def test_incoming_request_id_is_echoed(client: TestClient):
    response = client.post(CHAT, json={"message": COZY_DETECTIVE, "user_id": "demo"}, headers={"X-Request-ID": "trace-abc"})
    assert response.headers.get("x-request-id") == "trace-abc"


def test_oversized_incoming_request_id_is_truncated(client: TestClient):
    response = _chat(client, COZY_DETECTIVE, headers={"X-Request-ID": "z" * 500})
    assert len(response.headers.get("x-request-id", "")) <= 128


def test_session_continuity_accumulates_constraints(client: TestClient):
    first = _chat(client, "Лёгкий детективный сериал, один сезон").json()
    sid = first["session_id"]
    second = _chat(client, "Не дольше 30 минут", session_id=sid).json()
    # Earlier constraints survive the follow-up.
    assert second["query"]["genre"] == "детектив"
    assert second["query"]["max_seasons"] == 1
    assert second["query"]["max_minutes"] == 30


def test_session_belongs_to_its_user(client: TestClient):
    sid = _chat(client, COZY_DETECTIVE).json()["session_id"]
    response = client.post(CHAT, json={"message": "ещё", "user_id": "someone-else", "session_id": sid})
    assert response.status_code == 404
    assert response.json()["type"] == "urn:recagent:error:session-owner-mismatch"


def test_unknown_session_is_404(client: TestClient):
    response = _chat(client, "привет", session_id="does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "urn:recagent:error:session-not-found"
    assert body["status"] == 404


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"message": "", "user_id": "demo"}, 422),
        ({"message": "ok"}, 422),  # missing user_id is a client error, not a default
        ({"message": "x" * 5000, "user_id": "demo"}, 422),
        ({"message": "ok", "user_id": "demo", "unexpected": 1}, 422),
    ],
)
def test_validation_errors_are_problem_json(client: TestClient, payload: dict, expected_status: int):
    response = client.post(CHAT, json=payload)
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    for field in ("type", "title", "status"):
        assert field in body
    assert body["status"] == expected_status


def test_validation_errors_do_not_echo_user_content(client: TestClient):
    """Field paths are safe; input values may contain user text and must not leak."""
    secret = "мой приватный запрос 12345"
    response = client.post(CHAT, json={"message": secret, "user_id": "demo", "unexpected": 1})
    assert response.status_code == 422
    assert secret not in json.dumps(response.json(), ensure_ascii=False)


def test_feedback_requires_a_shown_item(client: TestClient):
    sid = _chat(client, COZY_DETECTIVE).json()["session_id"]
    response = client.post("/v1/feedback", json={"session_id": sid, "item_id": "demo-999", "reaction": "like"})
    assert response.status_code in (422, 404)
    assert response.headers["content-type"].startswith("application/problem+json")


def test_feedback_accepts_a_shown_item(client: TestClient):
    body = _chat(client, COZY_DETECTIVE).json()
    sid = body["session_id"]
    item_id = body["recommendations"][0]["item"]["id"]
    response = client.post("/v1/feedback", json={"session_id": sid, "item_id": item_id, "reaction": "dislike"})
    assert response.status_code == 200
    assert response.json() == {"status": "saved"}


def test_feedback_rejects_unknown_reaction(client: TestClient):
    body = _chat(client, COZY_DETECTIVE).json()
    response = client.post(
        "/v1/feedback",
        json={"session_id": body["session_id"], "item_id": body["recommendations"][0]["item"]["id"], "reaction": "meh"},
    )
    assert response.status_code == 422


def test_diagnostics_hidden_in_client_facing_mode(strict_client: TestClient):
    response = _chat(strict_client, COZY_DETECTIVE)
    assert response.status_code == 200
    body = response.json()
    assert "trace" not in body, "internal stage trace must not reach customers"
    assert "warnings" not in body, "internal warnings must not reach customers"
    # The actual product payload must survive.
    assert body["recommendations"]
    assert body["state"] == "recommend"


def test_diagnostics_visible_in_demo_mode(client: TestClient):
    body = _chat(client, COZY_DETECTIVE).json()
    assert body["trace"], "demo mode must expose the stage trace for inspection"


def test_response_is_valid_against_the_published_schema(client: TestClient):
    """The response must actually conform to ChatResponse, not just be JSON."""
    from recagent.models import ChatResponse

    body = _chat(client, COZY_DETECTIVE).json()
    parsed = ChatResponse.model_validate(body)
    assert parsed.latency_ms >= 0
    assert parsed.llm_calls >= 0


def test_llm_token_counter_is_reported_not_redacted(client: TestClient):
    """Cost accounting depends on this number reaching the response."""
    body = _chat(client, COZY_DETECTIVE).json()
    assert isinstance(body["llm_tokens"], int)
