"""Experimental NLI verification of cited enum updates.

The caller supplies a locally loaded, revision-pinned NLI adapter and
domain-specific hypothesis wording. Importing this module loads no weights.
Only enum resolution is replaced: SchemaRequestAdapter still checks citations,
operations, negation, and scalar values before or outside this resolver.

The resolver holds no per-request state. Concurrent callers must provide an
adapter and hypothesis factory that are themselves safe to share. For A/B
measurement, wrap ``adapter.predict`` with a counting adapter; every NLI call
passes through that method, while exact declared surfaces make zero calls.
Count NLI outcomes from the returned ``Resolution.reason`` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .resolution import FieldSpec, Resolution, comparison_key
from .semantic import NLIAdapter
from .semantic_policy import decide_nli


@dataclass(frozen=True)
class NLIHypotheses:
    """Domain wording for one canonical enum value and its opposite."""

    target: str
    opposite: str


class HypothesisFactory(Protocol):
    def __call__(self, spec: FieldSpec, canonical: object) -> NLIHypotheses: ...


class NLIResolver:
    """Resolve an enum proposal from a citation, abstaining on uncertain NLI.

    Literal enum/alias citations use exactly the InflectionalResolver contract.
    Other citations require target entailment, opposite contradiction, and no
    entailment of a competing declared value. Raw classifier scores never form
    a confidence threshold. Unknown ``value=None`` has no proposed hypothesis
    and stays unresolved unless the citation is a literal declared surface.
    """

    def __init__(self, adapter: NLIAdapter, hypotheses: HypothesisFactory):
        self.adapter = adapter
        self.hypotheses = hypotheses

    def resolve(self, spec: FieldSpec, value: object, source_text: str) -> Resolution:
        permitted = dict(spec.enum)
        permitted.update(spec.aliases)
        raw = comparison_key(source_text)

        if raw in permitted:
            declared = permitted[raw]
            if value is None:
                return Resolution("canonical", value=declared, evidence=source_text, reason="null_value_resolved")
            if isinstance(value, str) and permitted.get(comparison_key(value)) == declared:
                return Resolution("canonical", value=declared, evidence=source_text, reason="exact")
            return Resolution("ambiguous", evidence=source_text, reason="value_evidence_mismatch")

        if not raw:
            return Resolution("unsupported", evidence=source_text, reason="empty_citation")
        if not isinstance(value, str):
            return Resolution("unsupported", evidence=source_text, reason="no_declared_proposal")
        canonical = permitted.get(comparison_key(value))
        if canonical is None:
            return Resolution("unsupported", evidence=str(value), reason="value_not_declared")

        # A competing entailment makes the proposed enum member ambiguous even
        # when its own hypothesis is entailed. This list comes only from the
        # supplied field schema, never a catalog, Query, or evaluation labels.
        candidates: list[object] = []
        for declared in permitted.values():
            if declared != canonical and declared not in candidates:
                candidates.append(declared)
        try:
            target = self.hypotheses(spec, canonical)
            competing = tuple(self.hypotheses(spec, other).target for other in candidates)
            decision = decide_nli(
                self.adapter,
                source_text,
                target.target,
                target.opposite,
                competing_role_hypotheses=competing,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            return Resolution("unsupported", evidence=source_text, reason=f"hypothesis_error:{type(exc).__name__}")

        if decision.decision == "ACCEPT":
            return Resolution("canonical", value=canonical, evidence=source_text, reason="nli_entailed")
        if decision.decision == "REJECT" or decision.reason == "conflicting_entailment":
            return Resolution("ambiguous", evidence=source_text, reason=f"nli_{decision.reason}")
        return Resolution("unsupported", evidence=source_text, reason=f"nli_{decision.reason}")


__all__ = ["HypothesisFactory", "NLIHypotheses", "NLIResolver"]
