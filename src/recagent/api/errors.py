"""RFC 7807 problem details.

A consistent error envelope matters here for a concrete reason: the platform side
of the integration contract returns application/problem+json too, and the agent
must distinguish "do not retry" (4xx) from "retry then degrade" (429/5xx). Using
one shape in both directions keeps that logic in a single place.
"""
from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from ..resilience.degradation import DegradationLevel


class ProblemDetail(BaseModel):
    """RFC 7807. `type` defaults to about:blank per the specification."""

    model_config = ConfigDict(extra="allow")

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


# Canonical problems, so clients can match on `type` rather than on human text.
SESSION_NOT_FOUND: Final = ProblemDetail(
    type="urn:recagent:error:session-not-found",
    title="Session not found or expired",
    status=404,
)
SESSION_OWNER_MISMATCH: Final = ProblemDetail(
    type="urn:recagent:error:session-owner-mismatch",
    title="Session does not belong to this user",
    status=404,
)
VALIDATION_ERROR: Final = ProblemDetail(
    type="urn:recagent:error:validation",
    title="Request validation failed",
    status=422,
)
CAPACITY_EXCEEDED: Final = ProblemDetail(
    type="urn:recagent:error:capacity",
    title="Server at capacity",
    status=503,
)
UPSTREAM_UNAVAILABLE: Final = ProblemDetail(
    type="urn:recagent:error:upstream-unavailable",
    title="Recommendation platform unavailable",
    status=503,
)
RATE_LIMITED: Final = ProblemDetail(
    type="urn:recagent:error:rate-limited",
    title="Too many requests",
    status=429,
)
INTERNAL_ERROR: Final = ProblemDetail(
    type="urn:recagent:error:internal",
    title="Internal error",
    status=500,
)


def problem(base: ProblemDetail, *, detail: str | None = None, instance: str | None = None, **extensions: Any) -> ProblemDetail:
    """Return a copy of a canonical problem with per-request specifics."""
    payload: dict[str, Any] = base.model_dump()
    if detail is not None:
        payload["detail"] = detail
    if instance is not None:
        payload["instance"] = instance
    payload.update(extensions)
    return ProblemDetail.model_validate(payload)


def degradation_extension(level: DegradationLevel, reason: str) -> dict[str, Any]:
    """Extension members telling a client that the answer is partial."""
    return {"degradation_level": level.name, "degradation_reason": reason}


def retry_after_headers(seconds: int) -> dict[str, str]:
    return {"Retry-After": str(max(1, seconds))}


def rate_limit_headers(limit: int, remaining: int, reset_s: int, prefix: str = "X-RateLimit") -> dict[str, str]:
    """Quota headers on every response, so clients can back off proactively."""
    return {
        f"{prefix}-Limit": str(max(0, limit)),
        f"{prefix}-Remaining": str(max(0, remaining)),
        f"{prefix}-Reset": str(max(0, reset_s)),
    }
