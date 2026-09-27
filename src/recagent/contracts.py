"""Frozen contracts shared by the post-interpretation workflow.

These types deliberately do not depend on a provider or on a customer domain.
The legacy :class:`Query` remains a compatibility projection, never the source
of truth for constraints.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


Scalar = str | int | float | bool | Decimal


class RequestContext(ContractModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    auth_mode: str = Field(min_length=1)


class TurnEnvelope(ContractModel):
    session_id: str | None = None
    message_id: UUID
    text: str = Field(min_length=1)
    received_at: datetime
    payload_hash: str = Field(min_length=1)


class SourceSpan(ContractModel):
    turn_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str = Field(min_length=1)
    origin: Literal["current", "pending_source"] = "current"


class Constraint(ContractModel):
    id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    op: Literal["eq", "neq", "lt", "lte", "gt", "gte"]
    value: Scalar | None = None
    unit: str | None = None
    strength: Literal["hard", "soft"] = "hard"
    source_spans: tuple[SourceSpan, ...] = ()
    turn_id: str = Field(min_length=1)
    domain_version: str = Field(min_length=1)


class ProposedChange(ContractModel):
    id: str = Field(min_length=1)
    operation: Literal["add", "replace", "remove", "clear"]
    constraint: Constraint | None = None
    target_constraint_ids: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()


class NormalizedProposal(ContractModel):
    intent: str | None = None
    changes: tuple[ProposedChange, ...] = ()
    issues: tuple[str, ...] = ()
    reset_evidence: tuple[SourceSpan, ...] = ()
    raw_request_hash: str = Field(min_length=1)


class ValidationFinding(ContractModel):
    code: str = Field(min_length=1)
    status: Literal["pass", "reject", "uncertain"]
    change_ids: tuple[str, ...] = ()
    spans: tuple[SourceSpan, ...] = ()
    field: str | None = None
    details: str = ""


class ValidationResult(ContractModel):
    status: Literal["PASS", "REJECT", "UNCERTAIN"]
    proposal: NormalizedProposal
    findings: tuple[ValidationFinding, ...] = ()
    checks_run: tuple[str, ...] = ()
    semantic_mode: str = "code"


class PendingState(ContractModel):
    """A non-active proposal plus the exact group currently being clarified."""

    proposal: NormalizedProposal
    findings: tuple[ValidationFinding, ...] = ()
    question_id: str = Field(min_length=1)
    target_change_ids: tuple[str, ...] = ()
    target_field: str | None = None
    base_version: int = Field(ge=0)


class ConstraintState(ContractModel):
    intent: str = "discovery"
    constraints: tuple[Constraint, ...] = ()
    pending: PendingState | None = None
    version: int = 0


def project_constraints_to_query(constraints: tuple[Constraint, ...] | list[Constraint]):
    """Return ``(legacy Query, residual constraints)`` without dropping data.

    Only the currently supported Query fields are projected. Unsupported or
    non-equality constraints stay in ``residual`` for the new filter boundary.
    """
    from .models import Query

    values: dict[str, object] = {}
    residual: list[Constraint] = []
    for constraint in constraints:
        if constraint.strength != "hard":
            residual.append(constraint)
            continue
        field = constraint.field
        if constraint.op == "eq" and field in Query.model_fields and field != "excluded_genres":
            values[field] = constraint.value
        elif constraint.op == "lte" and field == "minutes":
            values["max_minutes"] = constraint.value
        elif constraint.op == "lte" and field == "seasons":
            values["max_seasons"] = constraint.value
        elif constraint.op == "neq" and field in {"excluded_genre", "genre"}:
            values.setdefault("excluded_genres", []).append(constraint.value)
        else:
            residual.append(constraint)
    try:
        return Query.model_validate(values, strict=True), residual
    except Exception:
        # A projection failure must not erase a constraint or invent a value.
        return Query(), list(constraints)
