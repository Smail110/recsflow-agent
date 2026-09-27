"""Small customer onboarding contract used by adapters and component checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import Field

from .contracts import ContractModel


class CustomerManifest(ContractModel):
    customer_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    domain_id: str = Field(min_length=1)
    request_schema: str = Field(min_length=1)
    operations: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    field_mapping: tuple[tuple[str, str], ...] = ()


def load_customer_manifest(path: str | Path) -> CustomerManifest:
    # YAML lists become JSON arrays at this boundary; strict scalar validation
    # stays enabled and does not coerce booleans/numbers into field names.
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CustomerManifest.model_validate_json(json.dumps(payload, ensure_ascii=False))


class UnsupportedOperation(ContractModel):
    ok: bool = False
    operation: str
    code: str = "unsupported_operation"
    message: str
    required_capability: str | None = None


def execute_operation(manifest: CustomerManifest, operation: str, handler: Any = None) -> Any:
    """Run an onboarding operation or return a typed graceful degradation."""
    if operation not in manifest.operations:
        return UnsupportedOperation(
            operation=operation,
            message=f"Операция «{operation}» не поддерживается клиентом {manifest.customer_id}.",
            required_capability=operation,
        )
    if handler is None:
        return UnsupportedOperation(operation=operation, message="Операция объявлена, но adapter handler не подключён.")
    return handler()


class UnavailableOperation(ContractModel):
    ok: bool = False
    operation: str
    code: str = "upstream_unavailable"
    message: str = "Компонент сервиса клиента временно недоступен. Повторите запрос позже."
    error_type: str


@dataclass
class OnboardedCustomer:
    """A manifest-checked component deployment, not a production API client."""

    manifest: CustomerManifest
    workflow: Any

    def execute(self, operation: str, **arguments):
        if operation == "recommend" and "lookup" not in self.manifest.operations:
            return execute_operation(self.manifest, "lookup")
        handlers = {"recommend": lambda: self.workflow.chat(**arguments), "lookup": lambda: self.workflow.provider.lookup(**arguments)}
        try:
            return execute_operation(self.manifest, operation, handlers.get(operation))
        except (TimeoutError, ConnectionError, httpx.HTTPError) as exc:
            return UnavailableOperation(operation=operation, error_type=type(exc).__name__)

    def chat(self, **arguments):
        # Retrieval needs both declared operations; do not silently use an
        # undeclared metadata capability even when a handler happens to exist.
        return self.execute("recommend", **arguments)


def onboard_customer(
    *, manifest: CustomerManifest, domain, provider, backend, adapter, supported_capabilities: set[str], item_projection=None
) -> OnboardedCustomer:
    """Bind declared canonical field mapping and verified adapter capabilities.

    A computed attribute is supplied explicitly by item_projection; the
    manifest maps to that canonical attribute, not to an unrelated raw value.
    """
    from .factory import build_domain_workflow

    if manifest.domain_id != domain.id or manifest.request_schema != adapter.model.__name__:
        raise ValueError("Manifest domain/schema does not match the connected adapter")
    mapping = dict(manifest.field_mapping)
    expected = {spec.name: spec.item_field or spec.name for spec in domain.fields}
    if len(mapping) != len(manifest.field_mapping) or mapping != expected:
        raise ValueError("Manifest must map every domain field to its canonical metadata field exactly once")
    if not set(manifest.capabilities).issubset(supported_capabilities):
        raise ValueError("Adapter does not support all manifest capabilities")
    for operation, method in (("recommend", "retrieve"), ("lookup", "lookup")):
        if operation in manifest.operations and not callable(getattr(provider, method, None)):
            raise ValueError(f"Declared operation {operation} requires callable provider.{method}")
    workflow = build_domain_workflow(domain=domain, provider=provider, backend=backend, adapter=adapter, item_projection=item_projection)
    return OnboardedCustomer(manifest=manifest, workflow=workflow)
