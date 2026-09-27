"""Local, append-only evidence of catalog wishes the agent could not check.

This is product feedback, not a training set or a claim about unique users.
Only short, cited wishes are stored; full messages and user/session IDs are not.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .logging import get_logger

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s()\-]*){10,}(?!\d)")


def _safe_wish(value: str) -> str:
    text = " ".join(value.split())
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[url]", text)
    return _PHONE.sub("[phone]", text)[:160]


class CatalogGapRecorder:
    """Write one JSONL event per completed turn with newly cited gaps."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(
        self,
        gaps: list[dict[str, str]],
        *,
        response_state: str,
        mode: str,
    ) -> None:
        wishes = [
            {
                "reason": gap["reason"],
                # The extraction field is only a hint, not a verified topic/category.
                "field_hint": gap["field_hint"],
                "wish": _safe_wish(gap["wish"]),
            }
            for gap in gaps
            if gap.get("wish", "").strip()
        ]
        if not wishes:
            return
        event = {
            "schema_version": 1,
            "event_id": uuid4().hex,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "response_state": response_state,
            "mode": mode,
            "wishes": wishes,
        }
        line = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    if os.write(fd, line) != len(line):
                        raise OSError("incomplete catalog gap write")
                finally:
                    os.close(fd)
        except OSError as exc:
            # A feedback sink must never turn a usable recommendation into 500.
            get_logger("recagent.catalog_gaps").warning("catalog_gap_write_failed", error_type=type(exc).__name__)
