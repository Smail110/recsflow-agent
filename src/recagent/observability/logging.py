"""Structured JSON logging with request-scoped context.

Every log line carries a request_id, so one dialogue turn can be reconstructed
end to end across the agent, the platform adapter and the LLM client. That is the
minimum requirement for the "analysis of bottlenecks" deliverable: without it,
latency numbers cannot be attributed to a stage.

Secrets are redacted by name, so an accidental log of a settings dump does not
leak provider credentials.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog

from ..config.settings import Settings

_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_SESSION_ID: ContextVar[str | None] = ContextVar("session_id", default=None)
_USER_ID: ContextVar[str | None] = ContextVar("user_id", default=None)

# Key substrings whose values must never reach logs or traces.
_REDACT_SUBSTRINGS = ("api_key", "apikey", "token", "secret", "password", "authorization")

_CONFIGURED = False


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def new_request_id() -> str:
    """Generate a request id. Short enough to fit response headers and log lines."""
    return uuid4().hex


@contextmanager
def request_context(
    request_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[str]:
    """Bind identifiers for the duration of a block.

    The user id is bound for correlation only. It is an opaque platform identifier,
    not user content, and is never logged together with free-text messages.
    """
    rid = request_id or new_request_id()
    tokens = (
        _REQUEST_ID.set(rid),
        _SESSION_ID.set(session_id),
        _USER_ID.set(user_id),
    )
    try:
        yield rid
    finally:
        _REQUEST_ID.reset(tokens[0])
        _SESSION_ID.reset(tokens[1])
        _USER_ID.reset(tokens[2])


def _add_context(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    rid = _REQUEST_ID.get()
    if rid:
        event_dict["request_id"] = rid
    sid = _SESSION_ID.get()
    if sid:
        event_dict["session_id"] = sid
    uid = _USER_ID.get()
    if uid:
        event_dict["user_id"] = uid
    return event_dict


def _redact_secrets(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        lowered = key.lower()
        if any(part in lowered for part in _REDACT_SUBSTRINGS):
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Idempotent setup of structlog on top of stdlib logging."""
    global _CONFIGURED

    level = getattr(logging, settings.observability.log_level.upper(), logging.INFO)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_context,
        _redact_secrets,
    ]

    if settings.observability.log_json:
        renderer: Any = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace handlers rather than appending: reconfiguring must not duplicate lines.
    root.handlers = [handler]
    root.setLevel(level)
    # Third-party loggers are noisy at debug; keep them at warning unless we are debugging.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING if level > logging.DEBUG else level)

    _CONFIGURED = True


def is_configured() -> bool:
    return _CONFIGURED


def get_logger(name: str = "recagent") -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Safe to call before configure_logging()."""
    if not _CONFIGURED:
        configure_logging(Settings())
    return structlog.get_logger(name)
