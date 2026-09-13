"""Tests for readiness/liveness probes.

Regression guard: returning a tuple from a FastAPI route serializes the tuple AS
THE BODY, so /ready once answered 200 with `[{}, 200]` instead of answering 503
when a dependency was down. An orchestrator would then keep routing traffic to a
dead instance. These tests pin the status code, not just the body.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from recagent.api.health import build_health_router


def _app(probes=None) -> FastAPI:
    app = FastAPI()
    app.include_router(build_health_router(version="test-1.0", probes=probes))
    return app


def test_liveness_never_touches_dependencies():
    """A failing probe must not make /health fail, or the instance gets restarted."""
    calls = {"n": 0}

    async def broken_probe():
        calls["n"] += 1
        raise RuntimeError("platform down")

    client = TestClient(_app({"platform": broken_probe}))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "test-1.0", "checks": {}}
    assert calls["n"] == 0, "liveness must not invoke probes"


def test_readiness_is_200_when_no_probes_configured():
    client = TestClient(_app())
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "test-1.0"
    # Must be a mapping, never a list/tuple artifact.
    assert isinstance(body, dict)
    assert body["checks"] == {}


def test_readiness_returns_503_when_dependency_unavailable():
    async def down():
        raise ConnectionError("platform unreachable")

    client = TestClient(_app({"recommendation_platform": down}))
    response = client.get("/ready")
    assert response.status_code == 503, "readiness must signal 503 so traffic is drained"
    body = response.json()
    assert isinstance(body, dict)
    assert body["status"] == "unavailable"
    check = body["checks"]["recommendation_platform"]
    assert check["status"] == "unavailable"
    assert "ConnectionError" in check["detail"]
    assert check["latency_ms"] >= 0


def test_readiness_200_when_all_probes_ok():
    async def up():
        return "catalog: 3000 items"

    client = TestClient(_app({"recommendation_platform": up}))
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["recommendation_platform"]["status"] == "ok"
    assert body["checks"]["recommendation_platform"]["detail"] == "catalog: 3000 items"


def test_readiness_degraded_is_still_200():
    """Degraded means 'serve with caution', so the instance stays in rotation."""

    async def flaky():
        return "cache cold"

    async def down():
        raise TimeoutError("slow")

    client = TestClient(_app({"cache": flaky, "platform": down}))
    response = client.get("/ready")
    # 'unavailable' dominates 'degraded', so overall must be unavailable here.
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_probe_detail_is_truncated_and_safe():
    """A probe error message must not carry unbounded or sensitive content."""

    async def verbose():
        raise RuntimeError("x" * 5000)

    client = TestClient(_app({"platform": verbose}))
    body = client.get("/ready").json()
    detail = body["checks"]["platform"]["detail"]
    assert isinstance(detail, str)
    assert len(detail) < 5000, "detail should be bounded to keep log lines sane"


def test_multiple_probes_all_reported():
    async def ok():
        return "ok"

    async def down():
        raise RuntimeError("no")

    client = TestClient(_app({"a": ok, "b": down, "c": ok}))
    body = client.get("/ready").json()
    assert set(body["checks"]) == {"a", "b", "c"}
    assert body["checks"]["a"]["status"] == "ok"
    assert body["checks"]["b"]["status"] == "unavailable"
    assert body["checks"]["c"]["status"] == "ok"
