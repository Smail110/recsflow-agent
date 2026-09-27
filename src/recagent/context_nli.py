"""Experimental, context-aware evidence verification for structured updates.

This module does not load weights or change the default resolver. The caller
provides a pinned sequence-pair classifier. Its premise is the complete user
turn; its hypothesis states one schema field, value and operation. Structural
checks run before inference, while the classifier decides semantic support.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from .interpretation import ConstraintUpdate
from .resolution import FieldSpec, Resolution, comparison_key, resolve_scalar
from .semantic import NLIAdapter, NLIResult, SemanticStatus

_TRAINED_OPERATIONS = frozenset({"set", "exclude", "include"})
_LABELS = {"support", "contradiction", "unknown"}


def claim_for_update(field_label: str, alias: str, operation: str, *, is_preference: bool = False) -> str:
    """Build the schema-grounded claim used by the task-specific training data.

    The wording is a versioned experiment contract, not a language parser. A
    separate fixture tests paraphrased claims; runtime uses this train form.
    """
    if not field_label.strip() or not alias.strip():
        raise ValueError("field label and alias must be nonempty")
    if operation == "set":
        if is_preference:
            return f"Пользователь предпочёл бы для подбираемого варианта значение «{alias}» параметра «{field_label}»."
        return f"Пользователь выбрал для подбора значение «{alias}» параметра «{field_label}»."
    if operation == "exclude":
        return f"Пользователь исключил для подбора значение «{alias}» параметра «{field_label}»."
    if operation == "include":
        return f"Пользователь снова разрешил для подбора значение «{alias}» параметра «{field_label}»."
    raise ValueError(f"unsupported claim operation: {operation}")


def cited_claim_for_update(field_label: str, alias: str, operation: str, *, is_preference: bool = False) -> str:
    """V4 claim wording; kept separate from the existing unmarked V3 claim."""
    return "Выделенная цитата подтверждает: " + claim_for_update(
        field_label, alias, operation, is_preference=is_preference
    )


def tag_citation(
    message: str,
    source_text: str,
    *,
    source_start: int | None = None,
    source_end: int | None = None,
) -> str:
    """Mark one exact source span for V4 training/inference; fail on ambiguity.

    A caller with offsets can distinguish repeated identical citations. The
    current product transport has no offsets, so repeated text abstains there.
    Without offsets, only a unique whole-word occurrence is safe to mark. Raw
    marker strings in user text are refused, regardless of case, so they cannot
    impersonate the selected evidence. V3 input format remains unchanged.
    """
    lowered_message = message.casefold()
    if not message or not source_text or "<evidence>" in lowered_message or "</evidence>" in lowered_message:
        raise ValueError("message/source must be nonempty and unmarked")
    if (source_start is None) != (source_end is None):
        raise ValueError("both source offsets are required together")
    if source_start is None:
        matches = [
            match for match in re.finditer(re.escape(source_text), message, flags=re.IGNORECASE)
            if (match.start() == 0 or not (message[match.start() - 1].isalnum() or message[match.start() - 1] == "_"))
            and (match.end() == len(message) or not (message[match.end()].isalnum() or message[match.end()] == "_"))
        ]
        if len(matches) != 1:
            raise ValueError("citation does not have exactly one whole-word occurrence")
        source_start, source_end = matches[0].span()
    if not (0 <= source_start < source_end <= len(message)):
        raise ValueError("citation offsets outside message")
    if message[source_start:source_end].casefold() != source_text.casefold():
        raise ValueError("citation offsets do not match source text")
    return message[:source_start] + "<evidence>" + message[source_start:source_end] + "</evidence>" + message[source_end:]


@dataclass(frozen=True)
class EvidenceInput:
    message: str
    update: ConstraintUpdate
    spec: FieldSpec
    field_label: str
    allowed_operations: tuple[str, ...] = ("set",)
    is_preference: bool = False
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True)
class EvidenceVerdict:
    resolution: Resolution
    hypothesis: str | None = None
    model_label: str | None = None
    model_score: float | None = None
    cached: bool = False
    inference_ms: float = 0.0


@dataclass(frozen=True)
class EvidenceStats:
    attempts: int
    model_calls: int
    cache_hits: int
    inference_ms: float
    support: int
    contradiction: int
    unknown: int
    abstain: int
    structural: int


class ContextNLI:
    """Verify one LLM proposal with one full-context classifier prediction.

    ``citation_mode='unmarked'`` preserves the V3 hypothesis/input format.
    ``'tagged'`` uses the separate V4 citation-aware format and must only be
    paired with a classifier trained on that format. Tagged mode still requires
    a declared source alias by default: V4 train/dev do not cover undeclared
    synonyms. The opt-in for such citations is research-only. Neither mode is
    default in the application; both require explicit adapter injection.

    A bounded digest-keyed cache avoids repeated inference when workflow
    normalizes the same proposal several times. It stores no user text. Model
    errors, unknown labels and unsupported operation/value types abstain.
    Softmax scores are recorded for diagnostics, not treated as calibrated
    confidence in the proposal.
    """

    def __init__(
        self,
        model: NLIAdapter,
        *,
        cache_size: int = 256,
        citation_mode: Literal["unmarked", "tagged"] = "unmarked",
        allow_undeclared_citation_for_research: bool = False,
    ):
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        if citation_mode not in {"unmarked", "tagged"}:
            raise ValueError("unknown citation mode")
        self.model = model
        self.cache_size = cache_size
        self.citation_mode = citation_mode
        self.allow_undeclared_citation_for_research = allow_undeclared_citation_for_research
        self._cache: OrderedDict[str, NLIResult] = OrderedDict()
        self._lock = Lock()
        self._attempts = 0
        self._model_calls = 0
        self._cache_hits = 0
        self._inference_ms = 0.0
        self._outcomes = {"support": 0, "contradiction": 0, "unknown": 0, "abstain": 0, "structural": 0}

    @staticmethod
    def _cited(source_text: str, message: str) -> bool:
        source = comparison_key(source_text)
        if not source or not message.strip():
            return False
        return bool(re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", comparison_key(message)))

    @staticmethod
    def _canonical(spec: FieldSpec, value: object) -> object | None:
        if not isinstance(value, str):
            return None
        permitted = {**spec.enum, **spec.aliases}
        return permitted.get(comparison_key(value))

    @staticmethod
    def _surface(spec: FieldSpec, canonical: object) -> str | None:
        aliases = sorted(surface for surface, value in spec.aliases.items() if value == canonical)
        enums = sorted(surface for surface, value in spec.enum.items() if value == canonical)
        return next(iter(aliases or enums), None)

    @staticmethod
    def _key(message: str, hypothesis: str) -> str:
        content = json.dumps((message, hypothesis), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _record(self, outcome: Literal["support", "contradiction", "unknown", "abstain", "structural"]) -> None:
        with self._lock:
            self._outcomes[outcome] += 1

    def _abstain(self, source_text: str, reason: str) -> EvidenceVerdict:
        self._record("abstain")
        return EvidenceVerdict(Resolution("unsupported", evidence=source_text, reason=reason))

    def verify(self, evidence: EvidenceInput) -> EvidenceVerdict:
        with self._lock:
            self._attempts += 1
        update, spec = evidence.update, evidence.spec
        if update.field != spec.name or not evidence.field_label.strip():
            return self._abstain(update.source_text, "context_nli_invalid_field")
        if update.operation not in evidence.allowed_operations or update.operation not in _TRAINED_OPERATIONS:
            return self._abstain(update.source_text, "context_nli_unsupported_operation")
        if not self._cited(update.source_text, evidence.message):
            return self._abstain(update.source_text, "context_nli_uncited_source")
        if not spec.enum and not spec.aliases:
            # A classifier cannot replace literal number, unit, polarity or
            # catalog-title checks without separate training and validation.
            if update.operation != "set":
                return self._abstain(update.source_text, "context_nli_scalar_operation_unsupported")
            scalar = resolve_scalar(spec, update.value, update.source_text)
            self._record("structural")
            return EvidenceVerdict(scalar)
        canonical = self._canonical(spec, update.value)
        surface = self._surface(spec, canonical) if canonical is not None else None
        if canonical is None or surface is None:
            return self._abstain(update.source_text, "context_nli_value_not_declared")
        cited_value = {**spec.enum, **spec.aliases}.get(comparison_key(update.source_text))
        if self.citation_mode == "unmarked":
            # V3 sees no citation position. Require a declared surface for this
            # value; undeclared synonyms abstain instead of borrowing support.
            if cited_value != canonical:
                return self._abstain(update.source_text, "context_nli_citation_value_mismatch")
            premise = evidence.message
            hypothesis = claim_for_update(
                evidence.field_label, surface, update.operation, is_preference=evidence.is_preference
            )
        else:
            # A declared surface of a different enum value is an observable
            # schema conflict even when the cite-aware classifier is enabled.
            # The frozen V4 train/dev lack undeclared source synonyms, so they
            # abstain unless a research-only switch explicitly measures them.
            # Repeated spans without offsets stay unresolved.
            if cited_value is not None and cited_value != canonical:
                return self._abstain(update.source_text, "context_nli_citation_value_mismatch")
            if cited_value is None and not self.allow_undeclared_citation_for_research:
                return self._abstain(update.source_text, "context_nli_citation_not_declared")
            try:
                premise = tag_citation(
                    evidence.message, update.source_text,
                    source_start=evidence.source_start, source_end=evidence.source_end,
                )
            except ValueError:
                return self._abstain(update.source_text, "context_nli_ambiguous_or_invalid_citation")
            hypothesis = cited_claim_for_update(
                evidence.field_label, surface, update.operation, is_preference=evidence.is_preference
            )
        key = self._key(premise, hypothesis)
        cached = False
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
                self._cache_hits += 1
                cached = True
            else:
                self._model_calls += 1
        elapsed_ms = 0.0
        if result is None:
            started = time.perf_counter()
            try:
                result = self.model.predict(premise, hypothesis)
            except Exception as exc:
                return self._abstain(update.source_text, f"context_nli_model_error:{type(exc).__name__}")
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                with self._lock:
                    self._inference_ms += elapsed_ms
            if self._valid_result(result) and result.status.status == "available" and self.cache_size:
                with self._lock:
                    self._cache[key] = result
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        if not self._valid_result(result) or result.status.status != "available":
            return self._abstain(update.source_text, "context_nli_model_unavailable_or_invalid")
        self._record(result.label)
        if result.label == "support":
            resolution = Resolution("canonical", canonical, update.source_text, "context_nli_support")
        elif result.label == "contradiction":
            resolution = Resolution("ambiguous", evidence=update.source_text, reason="context_nli_contradiction")
        else:
            resolution = Resolution("unsupported", evidence=update.source_text, reason="context_nli_unknown")
        return EvidenceVerdict(resolution, hypothesis, result.label, result.score, cached, elapsed_ms)

    def stats(self) -> EvidenceStats:
        with self._lock:
            return EvidenceStats(
                self._attempts,
                self._model_calls,
                self._cache_hits,
                self._inference_ms,
                self._outcomes["support"],
                self._outcomes["contradiction"],
                self._outcomes["unknown"],
                self._outcomes["abstain"],
                self._outcomes["structural"],
            )

    @staticmethod
    def _valid_result(result: object) -> bool:
        return (
            isinstance(result, NLIResult)
            and isinstance(result.status, SemanticStatus)
            and result.label in _LABELS
            and isinstance(result.score, (int, float))
            and math.isfinite(result.score)
            and 0.0 <= result.score <= 1.0
        )


__all__ = [
    "ContextNLI",
    "EvidenceInput",
    "EvidenceStats",
    "EvidenceVerdict",
    "cited_claim_for_update",
    "claim_for_update",
    "tag_citation",
]
