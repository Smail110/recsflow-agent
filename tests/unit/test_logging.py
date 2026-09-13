"""Tests for structured logging: request context, JSON output, secret redaction."""
from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest

from recagent.config.settings import Settings
from recagent.observability.logging import (
    configure_logging,
    get_logger,
    get_request_id,
    is_configured,
    new_request_id,
    request_context,
)


@pytest.fixture
def log_buffer() -> Iterator[io.StringIO]:
    """Capture log output from a JSON-configured logger."""
    buffer = io.StringIO()
    configure_logging(Settings.model_validate({"observability": {"log_json": True, "log_level": "info"}}))
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    assert len(root.handlers) == 1, "configure_logging must replace handlers, not append"
    root.handlers[0].stream = buffer
    yield buffer
    root.handlers = original_handlers
    root.setLevel(original_level)


def _records(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_request_id_is_generated_as_hex():
    rid = new_request_id()
    assert isinstance(rid, str) and len(rid) == 32
    int(rid, 16)  # must be valid hex


def test_context_binds_ids_to_log_lines(log_buffer: io.StringIO):
    rid = new_request_id()
    log = get_logger("unit")
    with request_context(request_id=rid, session_id="sess-42", user_id="u-7"):
        log.info("turn", stage="parse")
    record = _records(log_buffer)[-1]
    assert record["request_id"] == rid
    assert record["session_id"] == "sess-42"
    assert record["user_id"] == "u-7"
    assert record["stage"] == "parse"


def test_context_is_restored_after_block(log_buffer: io.StringIO):
    log = get_logger("unit")
    assert get_request_id() is None
    with request_context(request_id="abc"):
        assert get_request_id() == "abc"
    assert get_request_id() is None
    # A line outside any context must not carry a stale request_id.
    log.info("outside")
    assert "request_id" not in _records(log_buffer)[-1]


def test_nested_contexts_do_not_leak(log_buffer: io.StringIO):
    log = get_logger("unit")
    with request_context(request_id="outer", session_id="s1"):
        log.info("in-outer")
        with request_context(request_id="inner"):
            log.info("in-inner")
        log.info("back-outer")
    records = _records(log_buffer)
    assert records[0]["request_id"] == "outer"
    assert records[0]["session_id"] == "s1"
    assert records[1]["request_id"] == "inner"
    assert records[2]["request_id"] == "outer"
    assert records[2]["session_id"] == "s1"


@pytest.mark.parametrize(
    "field_name",
    ["api_key", "apiKey", "provider_api_key", "access_token", "TOKEN", "client_secret", "password", "Authorization"],
)
def test_secrets_are_redacted_by_key_name(log_buffer: io.StringIO, field_name: str):
    log = get_logger("unit")
    with request_context(request_id="r"):
        log.info("leaky", **{field_name: "s3cr3t-value"})
    record = _records(log_buffer)[-1]
    assert record[field_name] == "[REDACTED]"
    assert "s3cr3t-value" not in json.dumps(record, ensure_ascii=False)


def test_non_secret_values_are_preserved(log_buffer: io.StringIO):
    log = get_logger("unit")
    with request_context(request_id="r"):
        log.info("ok", latency_ms=12.5, candidates=100, mode="rules")
    record = _records(log_buffer)[-1]
    assert record["latency_ms"] == 12.5
    assert record["candidates"] == 100
    assert record["mode"] == "rules"


def test_russian_text_is_not_escaped(log_buffer: io.StringIO):
    """Explanations and catalog titles are Russian; logs must stay readable."""
    log = get_logger("unit")
    with request_context(request_id="r"):
        log.info("Объяснение проверено", genre="детектив")
    raw = log_buffer.getvalue()
    assert "Объяснение проверено" in raw
    assert "\\u" not in raw


def test_json_output_is_machine_readable(log_buffer: io.StringIO):
    log = get_logger("unit")
    with request_context(request_id="r"):
        log.warning("degraded", level_name="NO_LLM")
    record = _records(log_buffer)[-1]
    assert record["level"] == "warning"
    assert record["logger"] == "unit"
    assert "timestamp" in record


def test_console_mode_produces_readable_output():
    buffer = io.StringIO()
    configure_logging(Settings.model_validate({"observability": {"log_json": False, "log_level": "info"}}))
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers[0].stream = buffer
    try:
        get_logger("unit").info("readable line", stage="rank")
    finally:
        root.handlers = original_handlers
    output = buffer.getvalue()
    assert "readable line" in output
    assert "stage" in output
    assert not output.strip().startswith("{")


def _our_handlers() -> list[logging.Handler]:
    """Handlers installed by configure_logging, ignoring pytest capture handlers.

    pytest attaches its own LogCaptureHandler(s) to the root logger during a test,
    so an absolute count would depend on test internals rather than our code.
    """
    root = logging.getLogger()
    ours = []
    for handler in root.handlers:
        formatter = handler.formatter
        if type(handler) is logging.StreamHandler and formatter is not None and type(formatter).__name__ == "ProcessorFormatter":
            ours.append(handler)
    return ours


def test_configure_logging_is_idempotent():
    configure_logging(Settings())
    configure_logging(Settings())
    assert is_configured()
    # Reconfiguration must replace our handler, never append a second one,
    # otherwise every log line is emitted twice in production.
    assert len(_our_handlers()) == 1


def test_get_logger_before_configuration_does_not_crash():
    logger = get_logger("cold-start")
    logger.info("boot")
    assert len(_our_handlers()) == 1
