"""Domain and provider capability descriptors."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contracts import ContractModel


class FieldSpec(ContractModel):
    name: str = Field(min_length=1)
    value_type: Literal["string", "integer", "decimal", "boolean", "enum"]
    operators: tuple[Literal["eq", "neq", "lt", "lte", "gt", "gte"], ...] = ("eq",)
    default_operator: Literal["eq", "neq", "lt", "lte", "gt", "gte"] = "eq"
    cardinality: Literal["scalar", "multi"] = "scalar"
    unit: str | None = None
    item_field: str | None = None
    aliases: tuple[tuple[str, str], ...] = ()
    min_value: int | float | None = None
    max_value: int | float | None = None


class DomainSpec(ContractModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    fields: tuple[FieldSpec, ...] = ()
    request_schema: str = Field(min_length=1)
    field_labels: tuple[tuple[str, str], ...] = ()
    required_for_intent: tuple[tuple[str, tuple[str, ...]], ...] = ()
    transition_policy: str = "reset_on_domain_change"
    hypothesis_templates: tuple[tuple[str, str], ...] = ()
    question_templates: tuple[tuple[str, str], ...] = ()
    item_projection: tuple[tuple[str, str], ...] = ()
    search_document_template: str = "{title} {description}"


class ProviderCapabilities(ContractModel):
    provider_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    supports_retrieve: bool = False
    supports_lookup: bool = False
    supports_history: bool = False
    supports_find_title: bool = False
    supported_fields: tuple[str, ...] = ()
    supported_operators: tuple[tuple[str, tuple[Literal["eq", "neq", "lt", "lte", "gt", "gte"], ...]], ...] = ()
    supported_operations: tuple[str, ...] = ()
    max_candidates: int = Field(default=100, ge=1)
