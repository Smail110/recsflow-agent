"""Bound completed replay entries and in-flight capacity without model calls."""

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from recagent.agent import Agent, IdempotencyConflict
from recagent.api import create_app
from recagent.models import ChatRequest


def request(**kwargs):
    return ChatRequest(user_id="cache-control", message="фильм комедия", message_id=uuid4(), **kwargs)


def test_completed_cache_capacity_rejects_before_new_execution_but_keeps_replay():
    class Counting(Agent):
        parses = 0

        def _parse(self, state):
            self.parses += 1
            return super()._parse(state)

    agent = Counting(mode="rules", max_message_records=1)
    first_request = request()
    first = agent.chat(first_request)
    for _ in range(10):
        with pytest.raises(RuntimeError, match="лимит"):
            agent.chat(request(session_id=first.session_id))
    assert agent.parses == 1 and len(agent.message_records) == 1
    assert agent.message_locks == {} and not agent.message_record_reservations
    assert agent.chat(first_request).telemetry["cache_hit"]
    assert agent.parses == 1


def test_window_expiration_treats_key_as_new_and_reclaims_capacity(monkeypatch):
    import recagent.agent as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    agent = Agent(mode="rules", ttl=1000, max_message_records=1, message_record_ttl=10)
    shared = request()
    first = agent.chat(shared)
    clock[0] = 109
    assert agent.chat(shared).telemetry["cache_hit"]
    with pytest.raises(IdempotencyConflict):
        agent.chat(shared.model_copy(update={"message": "фильм драма"}))
    clock[0] = 110
    second = agent.chat(shared.model_copy(update={"message": "фильм драма"}))
    assert second.session_id != first.session_id
    assert second.query.genre == "драма"
    assert not second.telemetry.get("cache_hit")
    assert len(agent.message_records) == 1
    # Replay at t=109 did not extend the fixed original expiry.
    clock[0] = 120
    third = agent.chat(request())
    assert third.session_id not in {first.session_id, second.session_id}
    assert len(agent.message_records) == 1


def test_inflight_distinct_keys_reserve_capacity_before_graph_and_same_key_replays():
    entered, release = threading.Event(), threading.Event()

    class Pausing(Agent):
        parses = 0

        def _parse(self, state):
            self.parses += 1
            entered.set()
            assert release.wait(5)
            return super()._parse(state)

    agent = Pausing(mode="rules", max_message_records=1)
    shared = request()
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(agent.chat, shared)
        assert entered.wait(5)
        retry = pool.submit(agent.chat, shared)
        try:
            rejected = pool.submit(agent.chat, request())
            with pytest.raises(RuntimeError, match="лимит"):
                rejected.result(timeout=3)
            assert agent.parses == 1 and len(agent.message_record_reservations) == 1
        finally:
            release.set()
        a, b = first.result(timeout=5), retry.result(timeout=5)
    assert a.session_id == b.session_id and b.telemetry["cache_hit"]
    assert agent.parses == 1 and len(agent.message_records) == 1
    assert not agent.message_record_reservations and not agent.message_locks


def test_graph_and_session_failure_release_reserved_cache_slot():
    class Failing(Agent):
        fail = True

        def _parse(self, state):
            if self.fail:
                self.fail = False
                raise ArithmeticError("controlled graph failure")
            return super()._parse(state)

    agent = Failing(mode="rules", max_message_records=1)
    with pytest.raises(KeyError):
        agent.chat(request(session_id="missing"))
    assert not agent.message_record_reservations
    with pytest.raises(ArithmeticError):
        agent.chat(request())
    assert not agent.message_record_reservations and not agent.message_records
    assert agent.chat(request()).state == "recommend"
    assert len(agent.message_records) == 1 and not agent.message_record_reservations


def test_cache_capacity_is_an_http_503_without_new_session():
    agent = Agent(mode="rules", max_message_records=1)
    client = TestClient(create_app(agent, expose_diagnostics=False))
    payload = {"message": "фильм комедия", "user_id": "cache-control", "message_id": str(uuid4())}
    assert client.post("/v1/chat", json=payload).status_code == 200
    response = client.post("/v1/chat", json={**payload, "message_id": str(uuid4())})
    assert response.status_code == 503 and response.json()["type"] == "urn:recagent:error:capacity"
    assert len(agent.sessions) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_message_records": 0},
        {"max_message_records": -1},
        {"max_message_records": True},
        {"message_record_ttl": 0},
        {"message_record_ttl": float("inf")},
        {"message_record_ttl": float("nan")},
    ],
)
def test_invalid_cache_bounds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Agent(mode="rules", **kwargs)
