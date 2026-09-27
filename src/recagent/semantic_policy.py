"""Conservative, gold-independent decisions over NLI hypothesis pairs.

Hypothesis wording belongs to the caller's domain descriptor. This module only
interprets model labels; it never treats a classifier score as a probability of
the proposed update being correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .semantic import NLIAdapter, NLIResult, SemanticStatus

Decision = Literal["ACCEPT", "REJECT", "ABSTAIN"]
_LABELS = {"entailment", "neutral", "contradiction"}


@dataclass(frozen=True)
class HypothesisResult:
    role: str
    hypothesis: str
    prediction: NLIResult


@dataclass(frozen=True)
class SemanticDecision:
    decision: Decision
    reason: str
    predictions: tuple[HypothesisResult, ...]


def decide_nli(
    adapter: NLIAdapter,
    premise: str,
    target_hypothesis: str,
    opposite_hypothesis: str,
    *,
    competing_role_hypotheses: Sequence[str] = (),
) -> SemanticDecision:
    """Verify one proposed update without access to evaluation labels.

    ACCEPT requires target entailment and opposite contradiction. Conflicting
    entailments, uncertain labels, and runtime failures abstain. REJECT requires
    a contradiction of the target or entailment of its opposite with no such
    conflict. The caller retains every raw model result for inspection.
    """
    hypotheses = (target_hypothesis, opposite_hypothesis, *competing_role_hypotheses)
    if not premise.strip() or any(not hypothesis.strip() for hypothesis in hypotheses):
        raise ValueError("premise and hypotheses must be nonempty")
    if target_hypothesis == opposite_hypothesis:
        raise ValueError("target and opposite hypotheses must differ")

    predictions: list[HypothesisResult] = []
    for index, hypothesis in enumerate(hypotheses):
        role = "target" if index == 0 else "opposite" if index == 1 else "competing_role"
        try:
            result = adapter.predict(premise, hypothesis)
        except Exception as exc:  # A failing optional backend must not accept an update.
            result = NLIResult(
                label="error",
                score=0.0,
                status=SemanticStatus("error", adapter.model, reason=type(exc).__name__),
            )
        predictions.append(HypothesisResult(role, hypothesis, result))
        if result.status.status != "available" or result.label not in _LABELS:
            return SemanticDecision("ABSTAIN", "model_unavailable_or_invalid", tuple(predictions))

    target, opposite, *competing = (item.prediction.label for item in predictions)
    if target == "entailment" and (opposite == "entailment" or "entailment" in competing):
        return SemanticDecision("ABSTAIN", "conflicting_entailment", tuple(predictions))
    if target == "entailment" and opposite == "contradiction":
        return SemanticDecision("ACCEPT", "target_entailed", tuple(predictions))
    if target == "contradiction" or opposite == "entailment":
        return SemanticDecision("REJECT", "target_contradicted", tuple(predictions))
    return SemanticDecision("ABSTAIN", "insufficient_evidence", tuple(predictions))


__all__ = ["Decision", "HypothesisResult", "SemanticDecision", "decide_nli"]
