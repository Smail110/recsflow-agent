"""Degradation ladder.

The task explicitly requires "degradation to the classic Recsflow response on
failure". Today a provider failure returns an error instead. This module makes
degradation a first-class, observable, and testable concept rather than an
exception handler.

Levels are ordered from richest to poorest. A level is chosen once per turn and
recorded in the log, the trace, the response, and a metric, so a client can tell
a complete answer from a partial one.

The key property: a LOWER level never pretends to be a higher one. If grounding
cannot be verified we do not invent an explanation; if the catalog is unreachable
we still return the platform's raw ids so the user gets something useful.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Final


class DegradationLevel(IntEnum):
    """Ordered so that ``min()`` yields the most capable level still available."""

    FULL = 4
    """LLM parsing + filtering + explanations + grounded evidence."""

    NO_LLM = 3
    """Rule-based parsing, explanations and evidence still available.
    Entered when the LLM is unreachable, returns an invalid structure, or the
    per-session LLM budget is exhausted."""

    NO_EXPLAIN = 2
    """Raw platform list without explanations. Entered when metadata lookup or
    grounding data is unavailable: we can still show ids and titles, but we must
    not fabricate facts about them."""

    UNAVAILABLE = 1
    """Nothing can be served. Client receives 503 with Retry-After."""

    @property
    def is_terminal(self) -> bool:
        return self is DegradationLevel.UNAVAILABLE

    @property
    def has_explanations(self) -> bool:
        return self >= DegradationLevel.NO_LLM

    @property
    def has_recommendations(self) -> bool:
        return self >= DegradationLevel.NO_EXPLAIN

    @property
    def user_message(self) -> str:
        """Honest wording shown to the user. Never claims more than was delivered."""
        return _USER_MESSAGES[self]


_USER_MESSAGES: Final[dict[DegradationLevel, str]] = {
    DegradationLevel.FULL: "Подобрал варианты. Объяснения проверены по данным каталога.",
    DegradationLevel.NO_LLM: "Подобрал варианты по упрощённому разбору запроса: языковая модель недоступна или её лимит исчерпан.",
    DegradationLevel.NO_EXPLAIN: "Показываю список платформы без объяснений: каталог с атрибутами сейчас недоступен.",
    DegradationLevel.UNAVAILABLE: "Сервис рекомендаций временно недоступен. Повторите запрос позже.",
}


# Reasons are stable strings, not free text: they are used as metric labels.
class DegradationReason:
    LLM_UNREACHABLE: Final = "llm_unreachable"
    LLM_INVALID_STRUCTURE: Final = "llm_invalid_structure"
    LLM_BUDGET_EXHAUSTED: Final = "llm_budget_exhausted"
    PROVIDER_UNREACHABLE: Final = "provider_unreachable"
    METADATA_UNAVAILABLE: Final = "metadata_unavailable"
    GROUNDING_UNVERIFIABLE: Final = "grounding_unverifiable"
    CIRCUIT_OPEN: Final = "circuit_open"


class DegradationDecision:
    """Immutable record of why a turn ran at a given level."""

    __slots__ = ("level", "reasons")

    def __init__(self, level: DegradationLevel, reasons: tuple[str, ...] = ()) -> None:
        self.level = level
        self.reasons = reasons

    def downgrade(self, level: DegradationLevel, reason: str) -> DegradationDecision:
        """Return a new decision at the poorer of the two levels, accumulating reasons."""
        if level >= self.level:
            return DegradationDecision(self.level, self.reasons)
        return DegradationDecision(level, (*self.reasons, reason))

    @property
    def is_degraded(self) -> bool:
        return self.level is not DegradationLevel.FULL

    def as_dict(self) -> dict[str, object]:
        return {"level": self.level.name, "reasons": list(self.reasons)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DegradationDecision({self.level.name}, {self.reasons!r})"


def full() -> DegradationDecision:
    return DegradationDecision(DegradationLevel.FULL)
