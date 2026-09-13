"""Request-scoped middleware: correlation id, timing, logging, error envelope.

One middleware owns the request lifecycle so that every response, including
failures, carries the same headers and the same log shape. This is what makes the
"bottleneck analysis" deliverable possible: each turn has a request_id that joins
HTTP logs, agent stage logs and platform adapter logs.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..observability.logging import get_logger, request_context
from .errors import INTERNAL_ERROR, ProblemDetail, problem

REQUEST_ID_HEADER = "X-Request-ID"
LATENCY_HEADER = "X-Process-Time-Ms"
MAX_INCOMING_REQUEST_ID = 128

logger = get_logger("recagent.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id, time the turn, log it, and convert errors to RFC 7807.

    An incoming X-Request-ID is honoured so the agent can be traced behind a
    gateway; it is length-checked and echoed back unchanged.
    """

    def __init__(self, app, *, expose_user_id: bool = False) -> None:
        super().__init__(app)
        # In prod the user id comes from auth middleware, not from the request body,
        # so it must not be logged from an unauthenticated payload.
        self.expose_user_id = expose_user_id

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming[:MAX_INCOMING_REQUEST_ID] if incoming else None

        start = time.perf_counter()
        with request_context(request_id=request_id) as rid:
            try:
                response = await call_next(request)
            except Exception:  # last-resort envelope; details stay in logs, never in the client response
                elapsed_ms = (time.perf_counter() - start) * 1000
                # Never leak internals to the client; the traceback stays in logs.
                logger.exception("unhandled_error", path=request.url.path, method=request.method, elapsed_ms=round(elapsed_ms, 2))
                body = problem(INTERNAL_ERROR, detail="Unexpected server error.", instance=request.url.path)
                response = _problem_response(body, request_id=rid, elapsed_ms=elapsed_ms)
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000
                response.headers[REQUEST_ID_HEADER] = rid
                response.headers[LATENCY_HEADER] = f"{elapsed_ms:.2f}"
                logger.info(
                    "http_request",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    elapsed_ms=round(elapsed_ms, 2),
                )
                return response
        return response


def _problem_response(body: ProblemDetail, *, request_id: str | None, elapsed_ms: float) -> JSONResponse:
    response = JSONResponse(
        status_code=body.status,
        content=body.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    response.headers[LATENCY_HEADER] = f"{elapsed_ms:.2f}"
    return response


def problem_response(body: ProblemDetail) -> JSONResponse:
    """Build an RFC 7807 response for use inside route handlers."""
    return JSONResponse(
        status_code=body.status,
        content=body.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Optional verbose body-size logging for load testing. Off by default."""

    def __init__(self, app, *, enabled: bool = False) -> None:
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled:
            return await call_next(request)
        length = request.headers.get("content-length")
        logger.debug("http_body", path=request.url.path, content_length=length)
        return await call_next(request)


async def noop_endpoint(_request: Request) -> Response:  # pragma: no cover - test helper
    return Response(status_code=204)


Callable_ = Callable[[Request], Awaitable[Response]]
