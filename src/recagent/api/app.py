"""FastAPI application factory.

Public surface: POST /v1/chat, POST /v1/feedback, GET /health, GET /ready, GET /docs.
The endpoint path and response shape are part of the integration contract, so the
client-facing fields stay stable while internals move.

Design notes:
- `create_app()` is a factory, not a module-level singleton, so tests get isolated
  instances. `app` is exposed for `uvicorn recagent.api:app`.
- Diagnostics (trace, warnings, degradation) are included because the demo must be
  inspectable. For a client-facing deployment they are removed by
  `expose_diagnostics=False`; see docs/integration.md.
- Errors are RFC 7807 `application/problem+json`, matching the platform contract so
  retry-vs-degrade logic reads the same on both sides.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..agent import Agent
from ..config.settings import Settings, get_settings
from ..models import ChatRequest, ChatResponse, FeedbackRequest
from ..observability.logging import configure_logging, get_logger
from .errors import (
    CAPACITY_EXCEEDED,
    INTERNAL_ERROR,
    SESSION_NOT_FOUND,
    SESSION_OWNER_MISMATCH,
    VALIDATION_ERROR,
    problem,
)
from .health import build_health_router
from .middleware import RequestContextMiddleware, RequestLoggingMiddleware

logger = get_logger("recagent.api")


def create_app(
    agent: Agent | None = None,
    *,
    settings: Settings | None = None,
    expose_diagnostics: bool = True,
    verbose_logging: bool = False,
) -> FastAPI:
    """Build a configured FastAPI app.

    ``expose_diagnostics`` controls whether trace/warnings/degradation reach the
    client. Keep it on for the demo and the internal UI; turn it off when the
    endpoint is published to a client's customers.
    """
    config = settings or get_settings()
    configure_logging(config)

    if agent is None:
        from ..factory import build_agent
        service = build_agent(config)
    else:
        service = agent
    version = config.observability.service_version

    @asynccontextmanager
    async def lifespan(_app):
        yield
        closer = getattr(service.provider, "close", None)
        if closer:
            await run_in_threadpool(closer)

    app = FastAPI(
        title="RecAgent — conversational recommendations",
        version=version,
        lifespan=lifespan,
        description=(
            "Диалоговый слой поверх платформы рекомендаций. Контракт RecAgent, а не API Recsflow. "
            "Демо-каталог синтетический. See docs/contract/openapi.yaml for the platform contract."
        ),
    )
    app.state.agent = service
    app.state.settings = config
    app.state.expose_diagnostics = expose_diagnostics

    app.add_middleware(RequestLoggingMiddleware, enabled=verbose_logging)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(build_health_router(version=version, probes=_readiness_probes(service)))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field paths are safe to expose; input values are not, they may hold user text.
        details = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        logger.warning("validation_error", path=request.url.path, detail=details)
        return _problem(problem(VALIDATION_ERROR, detail=details, instance=request.url.path))

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        """Session lookup failures. Agent raises KeyError with a user-facing message."""
        text = str(exc).strip("'\"")
        base = SESSION_OWNER_MISMATCH if "пользовател" in text.lower() else SESSION_NOT_FOUND
        return _problem(problem(base, detail=text, instance=request.url.path))

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        text = str(exc)
        base = CAPACITY_EXCEEDED if "лимит" in text.lower() or "capacity" in text.lower() else INTERNAL_ERROR
        logger.error("runtime_error", path=request.url.path, detail=text)
        return _problem(problem(base, detail=text, instance=request.url.path))

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, http_request: Request) -> Any:
        response = await run_in_threadpool(service.chat, request)
        if response.degradation == "UNAVAILABLE":
            return JSONResponse({"type": "about:blank", "title": "Источник рекомендаций недоступен", "status": 503,
                                 "detail": response.message, "degradation": "UNAVAILABLE"},
                                status_code=503, media_type="application/problem+json", headers={"Retry-After": "3"})
        logger.info(
            "chat_turn",
            state=response.state,
            mode=response.mode,
            recommendations=len(response.recommendations),
            llm_calls=response.llm_calls,
            llm_tokens=response.llm_tokens,
            latency_ms=response.latency_ms,
        )
        if not http_request.app.state.expose_diagnostics:
            payload = response.model_dump(exclude={"trace", "warnings"})
            return JSONResponse(payload)
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        text = str(exc)
        logger.warning("value_error", path=request.url.path, detail=text)
        return _problem(problem(VALIDATION_ERROR, detail=text, instance=request.url.path))

    @app.post("/v1/feedback")
    async def feedback(request: FeedbackRequest) -> dict[str, str]:
        await run_in_threadpool(service.feedback, request.session_id, request.item_id, request.reaction)
        logger.info("feedback", reaction=request.reaction)
        return {"status": "saved"}

    return app


def _problem(detail: Any) -> JSONResponse:
    return JSONResponse(status_code=detail.status, content=detail.model_dump(exclude_none=True), media_type="application/problem+json")


def _readiness_probes(service: Agent) -> dict[str, Any]:
    """Readiness probes for the configured provider.

    The memory provider has no external dependency, so readiness is trivially ok.
    A recsflow provider adds a real HTTP probe; see providers/recsflow.py.
    """
    probes: dict[str, Any] = {}
    health_probe = getattr(service.provider, "readiness_probe", None)
    if health_probe is not None:
        probes["recommendation_platform"] = health_probe
    return probes


app = create_app()
