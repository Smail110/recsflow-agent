"""Bound transient reservations without splitting a key's concurrent lock."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from recagent.agent import Agent, IdempotencyConflict
from recagent.models import ChatRequest


def request(**values):
    return ChatRequest(user_id="reservation-control", message="фильм комедия", message_id=uuid4(), **values)


def test_capacity_failures_do_not_accumulate_message_reservations():
    agent = Agent(mode="rules", max_sessions=1)
    agent.chat(request())
    for _ in range(20):
        with pytest.raises(RuntimeError, match="лимит"):
            agent.chat(request())
    assert len(agent.sessions) == len(agent.message_records) == 1
    assert agent.message_locks == {}


def test_registered_same_key_waiters_share_one_execution_and_release_last_reservation():
    entered = threading.Event()
    release = threading.Event()

    class PausingAgent(Agent):
        parses = 0

        def _parse(self, state):
            self.parses += 1
            entered.set()
            assert release.wait(5)
            return super()._parse(state)

    agent = PausingAgent(mode="rules")
    shared = request()
    key = (shared.user_id, str(shared.message_id))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(agent.chat, shared)]
        assert entered.wait(5)
        futures.extend(pool.submit(agent.chat, shared) for _ in range(7))
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with agent.store_lock:
                    reservation = agent.message_locks[key]
                    count = getattr(reservation, "users", 0)
                if count == 8:
                    break
                threading.Event().wait(0.005)
            assert count == 8, "Every queued waiter must be registered before the holder may clean up"
        finally:
            release.set()
        results = [f.result(timeout=5) for f in futures]
    assert agent.parses == 1
    assert len({r.session_id for r in results}) == 1
    assert sum(bool(r.telemetry.get("cache_hit")) for r in results) == 7
    assert agent.message_locks == {}
    again = agent.chat(shared)
    assert again.telemetry["cache_hit"] and agent.parses == 1
    assert agent.message_locks == {}


def test_conflict_and_unknown_session_release_reservations_without_losing_record():
    agent = Agent(mode="rules")
    original = request()
    response = agent.chat(original)
    with pytest.raises(IdempotencyConflict):
        agent.chat(original.model_copy(update={"message": "фильм драма"}))
    with pytest.raises(KeyError):
        agent.chat(request(session_id="unknown-session"))
    assert agent.message_locks == {}
    assert agent.chat(original).session_id == response.session_id
    assert len(agent.message_records) == 1


def test_execution_exception_releases_reservation_and_same_key_can_retry():
    class FailingOnce(Agent):
        fail = True

        def _chat_once(self, *args):
            if self.fail:
                self.fail = False
                raise ArithmeticError("controlled execution failure")
            return super()._chat_once(*args)

    agent = FailingOnce(mode="rules")
    shared = request()
    with pytest.raises(ArithmeticError):
        agent.chat(shared)
    assert agent.message_locks == {} and agent.message_records == {}
    assert agent.chat(shared).state == "recommend"
    assert agent.message_locks == {}


def test_session_eviction_does_not_retain_idle_locks_or_change_saved_answer():
    agent = Agent(mode="rules", max_sessions=1, ttl=30)
    original = request()
    first = agent.chat(original)
    for _ in range(12):
        # Expire a completed idle session without wall-clock sleeps.
        for session in agent.sessions.values():
            session.touched = time.monotonic() - 31
        agent.chat(request())
        assert len(agent.sessions) == 1
        assert agent.message_locks == {}
    cached = agent.chat(original)
    assert cached.session_id == first.session_id
    assert cached.query == first.query and cached.recommendations == first.recommendations
    assert cached.telemetry["cache_hit"]
    assert len(agent.sessions) == 1 and agent.message_locks == {}
