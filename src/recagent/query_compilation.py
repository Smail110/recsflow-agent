"""Compile validated generic constraints to the legacy provider boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Constraint, project_constraints_to_query
from .domains.base import ProviderCapabilities


@dataclass(frozen=True)
class CompiledRequest:
    provider_query: object
    residual_constraints: tuple[Constraint, ...]
    pushed_constraint_ids: tuple[str, ...]
    request_fingerprint: str


def compile_constraints(constraints: tuple[Constraint, ...] | list[Constraint], capabilities: ProviderCapabilities) -> CompiledRequest:
    query, _ = project_constraints_to_query(constraints)
    pushed, retained = [], []
    supported = set(capabilities.supported_fields)
    operation_map = dict(capabilities.supported_operators)
    for constraint in constraints:
        operations = operation_map.get(constraint.field, ("eq",))
        if constraint.field not in supported or constraint.op not in operations:
            retained.append(constraint)
        else:
            pushed.append(constraint)
    fingerprint = "|".join(f"{c.field}:{c.op}:{c.value!r}" for c in constraints)
    return CompiledRequest(query, tuple(retained), tuple(c.id for c in pushed), fingerprint)
