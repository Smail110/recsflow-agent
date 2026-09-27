"""Tri-state hard constraint evaluation against canonical catalog metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from .contracts import Constraint

Decision = Literal["PASS", "FAIL", "UNKNOWN"]


def evaluate_constraint(attributes: dict[str, Any], constraint: Constraint) -> Decision:
    if constraint.field not in attributes or attributes[constraint.field] is None:
        return "UNKNOWN"
    actual = attributes[constraint.field]
    expected = constraint.value
    try:
        return (
            "PASS"
            if {
                "eq": actual == expected,
                "neq": actual != expected,
                "lt": actual < expected,
                "lte": actual <= expected,
                "gt": actual > expected,
                "gte": actual >= expected,
            }[constraint.op]
            else "FAIL"
        )
    except (TypeError, ValueError):
        return "UNKNOWN"


def hard_filter(
    items: list[Any], constraints: tuple[Constraint, ...], attributes: Callable[[Any], dict[str, Any]] | None = None
) -> tuple[list[Any], dict[str, list[str]]]:
    kept, diagnostics = [], {"unknown": [], "failed": []}
    for item in items:
        attrs = attributes(item) if attributes else (item if isinstance(item, dict) else item.model_dump())
        decisions = [evaluate_constraint(attrs, c) for c in constraints if c.strength == "hard"]
        if "FAIL" in decisions or "UNKNOWN" in decisions:
            key = str(attrs.get("id", ""))
            diagnostics["failed" if "FAIL" in decisions else "unknown"].append(key)
            continue
        kept.append(item)
    return kept, diagnostics
