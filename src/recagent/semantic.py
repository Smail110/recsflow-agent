"""Optional semantic model ports.

Importing this module never downloads weights or initializes a runtime. Concrete
implementations can be supplied by applications through the small adapter
interfaces below; the built-in adapters report unavailable deterministically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SemanticStatus:
    status: str  # available | unavailable | error
    model: str
    revision: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NLIResult:
    label: str
    score: float
    status: SemanticStatus


class NLIAdapter(Protocol):
    model: str

    def predict(self, premise: str, hypothesis: str) -> NLIResult: ...


class EmbeddingAdapter(Protocol):
    model: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class RerankerAdapter(Protocol):
    model: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class UnavailableNLI:
    """NLI port used when optional transformers/weights are not installed."""

    def __init__(self, model: str, revision: str | None = None, reason: str = "weights_not_loaded"):
        self.model, self.revision = model, revision
        self.status = SemanticStatus("unavailable", model, revision, reason)

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        del premise, hypothesis
        return NLIResult("unavailable", 0.0, self.status)


class UnavailableEmbeddings:
    def __init__(self, model: str, revision: str | None = None, reason: str = "weights_not_loaded"):
        self.model, self.revision = model, revision
        self.status = SemanticStatus("unavailable", model, revision, reason)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        return []


class UnavailableReranker:
    def __init__(self, model: str, revision: str | None = None, reason: str = "weights_not_loaded"):
        self.model, self.revision = model, revision
        self.status = SemanticStatus("unavailable", model, revision, reason)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        del query, documents
        return []


def optional_nli(model: str, *, revision: str | None = None, enabled: bool = False) -> NLIAdapter:
    """Return a port without side effects; runtime loading is application-owned."""
    if not enabled:
        return UnavailableNLI(model, revision, "disabled_by_config")
    return UnavailableNLI(model, revision)


def optional_embeddings(model: str, *, revision: str | None = None, enabled: bool = False) -> EmbeddingAdapter:
    return UnavailableEmbeddings(model, revision, "disabled_by_config" if not enabled else "runtime_not_configured")


def optional_reranker(model: str, *, revision: str | None = None, enabled: bool = False) -> RerankerAdapter:
    return UnavailableReranker(model, revision, "disabled_by_config" if not enabled else "runtime_not_configured")


__all__ = [
    "EmbeddingAdapter",
    "NLIAdapter",
    "NLIResult",
    "RerankerAdapter",
    "SemanticStatus",
    "UnavailableEmbeddings",
    "UnavailableNLI",
    "UnavailableReranker",
    "optional_embeddings",
    "optional_nli",
    "optional_reranker",
]
