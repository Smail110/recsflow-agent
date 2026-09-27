"""Deterministic ranking and reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real
from typing import Any, TypeVar

from .domains.base import FieldSpec

RankedItem = TypeVar("RankedItem")


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Any]], *, k: int = 60, limit: int = 150, eligible_ids: set[str] | None = None
) -> list[str]:
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, raw in enumerate(ranking, 1):
            item_id = str(raw if isinstance(raw, str) else (raw.get("item_id", raw.get("id")) if isinstance(raw, dict) else raw.item_id))
            if item_id in seen:
                continue
            seen.add(item_id)
            if eligible_ids is not None and item_id not in eligible_ids:
                continue
            scores[item_id] += 1.0 / (k + rank)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def deterministic_rank(items: list, *, order: list[str] | None = None, limit: int = 5) -> list:
    positions = {item_id: index for index, item_id in enumerate(order or [])}
    return sorted(items, key=lambda item: (positions.get(item.id, 10**9), -float(getattr(item, "quality", 0)), item.id))[:limit]


def diversify_head(items: list[RankedItem], *, field: str, size: int = 5, window: int = 20) -> list[RankedItem]:
    """Expose nearby alternatives without changing eligibility or the top item.

    The input is already relevance-ranked. Only the display head is reshuffled,
    and only categories present in its small leading window can be promoted.
    """
    if not items or size < 2:
        return list(items)
    head = [items[0]]
    first_value = _attribute(items[0], field)
    seen = {first_value} if first_value is not None else set()
    for item in items[1:window]:
        value = _attribute(item, field)
        if value is None or value in seen:
            continue
        head.append(item)
        seen.add(value)
        if len(head) == size:
            break
    if len(head) == 1:
        return list(items)
    promoted = {_attribute(item, "id") for item in head}
    return [*head, *(item for item in items if _attribute(item, "id") not in promoted)]


def _attribute(item: Any, field: str) -> Any:
    return item.get(field) if isinstance(item, Mapping) else getattr(item, field, None)


def _similarity(item: Any, reference: Any, fields: tuple[FieldSpec, ...]) -> float:
    """Fraction of the reference's known categorical attributes that match.

    Unknown metadata supplies no evidence of similarity. Boolean False remains
    a known value. Multi-valued categorical attributes use Jaccard overlap.
    """
    total = 0.0
    known = 0
    for field in fields:
        name = field.item_field or field.name
        expected = _attribute(reference, name)
        if expected is None or expected == "" or expected == [] or expected == ():
            continue
        known += 1
        actual = _attribute(item, name)
        if actual is None:
            continue
        if field.cardinality == "multi":
            left = set(actual) if isinstance(actual, (list, tuple, set, frozenset)) else {actual}
            right = set(expected) if isinstance(expected, (list, tuple, set, frozenset)) else {expected}
            total += len(left & right) / len(left | right) if left | right else 0.0
        else:
            total += float(type(actual) is type(expected) and actual == expected)
    return total / known if known else 0.0


def soft_rank(
    items: Iterable[RankedItem],
    *,
    fields: Iterable[FieldSpec],
    intent: str,
    seed: Any = None,
    history: Iterable[Any] = (),
    liked_ids: Iterable[str] = (),
    soft_preferences: Iterable[tuple[str, Any]] = (),
    order: Iterable[str] | None = None,
    limit: int | None = None,
    quality_field: str | None = None,
) -> list[RankedItem]:
    """Rank eligible candidates without adding items or relaxing constraints.

    For similarity requests, seed agreement is the first soft preference.
    Explicit likes precede general history affinity, followed by retrieval
    relevance. An explicitly declared quality_field supplies a public quality
    prior after seed/like/history affinity and before retrieval order. The
    prior is enabled only when every candidate has a finite real numeric value
    (booleans are not scores); otherwise the whole pool retains retrieval order
    within affinity groups. Unknown quality is never imputed as zero. With no
    quality_field the existing relevance policy is unchanged. This is a
    transparent lexicographic policy, with no fitted weights. Exact navigation
    bypasses both personalization and the optional quality prior.

    Only schema-declared enum/boolean equality attributes contribute: numeric
    limits need a domain-specific distance/scale before they can express
    similarity, and free text requires more than exact equality. Callers must
    perform hard filtering and exclusions first and pass canonical metadata.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    candidates = list(items)
    quality_scores: dict[int, float] = {}
    if quality_field is not None and intent != "navigation":
        for item in candidates:
            value = _attribute(item, quality_field)
            if not isinstance(value, Real) or isinstance(value, bool):
                quality_scores.clear()
                break
            try:
                score = float(value)
            except OverflowError:
                quality_scores.clear()
                break
            if not isfinite(score):
                quality_scores.clear()
                break
            quality_scores[id(item)] = score
    positions: dict[str, int] = {}
    for position, item_id in enumerate(order if order is not None else (_attribute(item, "id") for item in candidates)):
        positions.setdefault(item_id, position)
    categorical = tuple(field for field in fields if field.value_type in {"enum", "boolean"} and "eq" in field.operators)
    references = {_attribute(item, "id"): item for item in history}
    liked = set(liked_ids)
    liked_references = [item for item_id, item in references.items() if item_id in liked]
    preferences = tuple(soft_preferences) if intent != "navigation" else ()

    def affinity(item: Any, profile: Iterable[Any]) -> float:
        scores = [_similarity(item, reference, categorical) for reference in profile]
        return sum(scores) / len(scores) if scores else 0.0

    def key(item: Any) -> tuple:
        personalization = (
            (
                _similarity(item, seed, categorical) if intent == "similar" and seed is not None else 0.0,
                affinity(item, liked_references),
                affinity(item, references.values()),
            )
            if intent != "navigation"
            else (0.0, 0.0, 0.0)
        )
        soft_match = sum(
            type(_attribute(item, field)) is type(value) and _attribute(item, field) == value
            for field, value in preferences
        )
        return (
            -personalization[0] if intent == "similar" else -soft_match,
            -soft_match if intent == "similar" else -personalization[0],
            -personalization[1],
            -personalization[2],
            -quality_scores.get(id(item), 0.0),
            positions.get(_attribute(item, "id"), len(positions)),
            -float(_attribute(item, "quality") or 0.0) if quality_field is None else 0.0,
            str(_attribute(item, "id")) if quality_field is None else "",
        )

    return sorted(candidates, key=key)[:limit]
