"""Liveness and readiness probes.

These mirror the platform contract (`GET /health`, `GET /ready`) so that the agent
and the recommendation platform behave the same way under an orchestrator. The
distinction matters operationally:

  /health (liveness)  - process is up and can serve. Never touches dependencies.
                        Failing here causes a restart, so it must not depend on
                        anything that can be temporarily unavailable.
  /ready  (readiness) - dependencies are usable. Failing here removes the instance
                        from the load balancer without restarting it, which is the
                        correct reaction to a platform outage.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

router = APIRouter(tags=["platform"])


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str  # ok | degraded | unavailable
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    checks: dict[str, CheckResult] = {}
# A probe detail lands in logs and readiness payloads; bound it so a chatty or
# accidentally large error cannot blow up a log line or a health response.
_MAX_DETAIL = 256


def _bounded(detail: str | None) -> str | None:
    if detail is None:
        return None
    text = str(detail)
    return text if len(text) <= _MAX_DETAIL else text[: _MAX_DETAIL - 1] + "…"

def _health_json(body: HealthResponse, status: int) -> JSONResponse:
    """Serialize a health body with an explicit status code."""
    return JSONResponse(status_code=status, content=body.model_dump())

async def _timed(name: str, probe) -> tuple[str, CheckResult]:
    """Run a probe, converting exceptions into an 'unavailable' check result.

    A probe must never raise into the HTTP layer: a broken dependency should
    produce a 503 readiness response, not a 500.
    """
    start = time.perf_counter()
    try:
        detail = await probe()
    except Exception as exc:  # probes are untrusted third-party calls
        elapsed = (time.perf_counter() - start) * 1000
        return name, CheckResult(status="unavailable", latency_ms=round(elapsed, 2), detail=_bounded(f"{type(exc).__name__}: {exc}"))
    elapsed = (time.perf_counter() - start) * 1000
    return name, CheckResult(status="ok", latency_ms=round(elapsed, 2), detail=_bounded(detail))


def build_health_router(*, version: str, probes: dict[str, Any] | None = None):
    """Create a router whose readiness depends on the supplied async probes.

    ``probes`` maps a dependency name to an async callable returning a short detail
    string. An empty mapping means readiness is trivially satisfied, which is the
    right behaviour for the in-memory demo provider.
    """
    local = APIRouter(tags=["platform"])
    configured_probes = probes or {}

    @local.get("/health", response_model=HealthResponse)
    async def liveness() -> HealthResponse:
        # Deliberately dependency-free.
        return HealthResponse(status="ok", version=version)

    @local.get("/ready")
    async def readiness() -> JSONResponse:
        # A tuple return would be serialized AS THE BODY, so the status code must be
        # set on an explicit Response: readiness must actually answer 503 when a
        # dependency is down, otherwise the orchestrator keeps routing traffic to it.
        if not configured_probes:
            return _health_json(HealthResponse(status="ok", version=version), 200)

        checks: dict[str, CheckResult] = {}
        for name, probe in configured_probes.items():
            key, result = await _timed(name, probe)
            checks[key] = result

        statuses = {check.status for check in checks.values()}
        if "unavailable" in statuses:
            overall = "unavailable"
            code = 503
        elif "degraded" in statuses:
            overall = "degraded"
            code = 200
        else:
            overall = "ok"
            code = 200
        return _health_json(HealthResponse(status=overall, version=version, checks=checks), code)

    return local
