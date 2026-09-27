from typing import Literal

import pytest
from pydantic import BaseModel

from recagent.domains.base import DomainSpec, FieldSpec
from recagent.factory import build_domain_workflow
from recagent.interpretation import LLMRequestInterpreter
from recagent.onboarding import CustomerManifest, UnsupportedOperation, execute_operation
from recagent.request_mapping import SchemaRequestAdapter
from recagent.response_generation import GroundedOption, LLMGroundedResponseGenerator


class ElectronicsQuery(BaseModel):
    intent: str = "discovery"
    category: Literal["laptop", "tablet"] | None = None
    max_price: int | None = None
    portable: bool | None = None


class ElectronicsItem(BaseModel):
    sku: str
    name: str
    category: Literal["laptop", "tablet"]
    price_rub: int
    weight_kg: float
    score: float


class ElectronicsProvider:
    def __init__(self):
        self.items = {
            "e-1": ElectronicsItem(sku="e-1", name="Atlas Air 13", category="laptop", price_rub=89_000, weight_kg=1.2, score=0.91),
            "e-2": ElectronicsItem(sku="e-2", name="Atlas Pro 15", category="laptop", price_rub=129_000, weight_kg=1.8, score=0.95),
        }

    def retrieve(self, user_id: str, query: ElectronicsQuery, limit: int = 100) -> list[str]:
        del user_id
        rows = [
            item
            for item in self.items.values()
            if (query.category is None or item.category == query.category)
            and (query.max_price is None or item.price_rub <= query.max_price)
            and (query.portable is None or not query.portable or item.weight_kg <= 1.5)
        ]
        return [item.sku for item in sorted(rows, key=lambda item: -item.score)[:limit]]

    def lookup(self, item_ids: list[str]) -> list[ElectronicsItem]:
        return [self.items[item_id] for item_id in item_ids if item_id in self.items]


class ElectronicsInterpreterBackend:
    def structured(self, schema, system, payload):
        if "fields" in payload["domain"]:
            assert {field["name"] for field in payload["domain"]["fields"]} == {"category", "max_price", "portable"}
        else:
            assert "category" in payload["domain"]["schema"]["properties"]
        return schema(
            intent="discovery",
            updates=[
                {"field": "category", "value": "laptop", "source_text": "ноутбук"},
                {"field": "max_price", "value": 100_000, "source_text": "до 100000 рублей"},
                {"field": "portable", "value": True, "source_text": "лёгкий"},
            ],
        ), 21


class ElectronicsResponseBackend:
    def structured(self, schema, system, payload):
        assert payload["items"][0]["title"] == "Atlas Air 13"
        return schema(items=[{"item_id": "e-1", "evidence_indexes": [0, 1]}]), 8


def test_schema_provider_and_evidence_are_enough_for_a_distinct_customer():
    adapter = SchemaRequestAdapter(
        ElectronicsQuery,
        domain_field="category",
        aliases={"category": {"ноутбук": "laptop", "планшет": "tablet"}},
        scalar_aliases={"portable": {"лёгкий": True}},
    )
    interpreter = LLMRequestInterpreter(ElectronicsInterpreterBackend(), adapter.descriptor)
    user = "Нужен лёгкий ноутбук до 100000 рублей"
    structured, _ = interpreter.interpret(user, {})
    query, issues = adapter.apply(structured, ElectronicsQuery(), user)
    assert not issues
    assert query == ElectronicsQuery(category="laptop", max_price=100_000, portable=True)

    provider = ElectronicsProvider()
    items = provider.lookup(provider.retrieve("portable-user", query))
    assert [item.sku for item in items] == ["e-1"]
    item = items[0]
    options = [
        GroundedOption(
            id=item.sku,
            title=item.name,
            claims=[
                f"Цена: {item.price_rub} ₽.",
                f"Вес: {item.weight_kg} кг.",
            ],
        )
    ]
    message, _ = LLMGroundedResponseGenerator(ElectronicsResponseBackend()).generate(
        original_request=user, intent=query.intent, accepted_constraints=query.model_dump(), options=options, unresolved=[]
    )
    assert message == "Подходит «Atlas Air 13». Цена: 89000 ₽. Вес: 1.2 кг."


def test_second_customer_runs_a_full_schema_driven_workflow_through_factory():
    adapter = SchemaRequestAdapter(
        ElectronicsQuery,
        domain_field="category",
        aliases={"category": {"ноутбук": "laptop", "планшет": "tablet"}},
        scalar_aliases={"portable": {"лёгкий": True}},
    )
    domain = DomainSpec(
        id="electronics",
        version="1",
        request_schema="ElectronicsQuery",
        fields=(
            FieldSpec(name="category", value_type="enum"),
            FieldSpec(name="max_price", value_type="integer", item_field="price_rub", operators=("lte",), default_operator="lte"),
            FieldSpec(name="portable", value_type="boolean"),
        ),
    )
    service = build_domain_workflow(
        domain=domain,
        provider=ElectronicsProvider(),
        backend=ElectronicsInterpreterBackend(),
        adapter=adapter,
        item_projection=lambda item: {**item.model_dump(), "portable": item.weight_kg <= 1.5},
    )
    result = service.chat(user_id="portable-user", message="Нужен лёгкий ноутбук до 100000 рублей")
    assert result.state == "recommend"
    assert result.item_ids == ["e-1"]
    assert "hard_filter" in result.trace

    service.feedback(user_id="portable-user", session_id=result.session_id, item_id="e-1", reaction="dislike")
    with pytest.raises(KeyError, match="не найдена"):
        service.feedback(user_id="another-user", session_id=result.session_id, item_id="e-1", reaction="like")
    with pytest.raises(KeyError, match="не был показан"):
        service.feedback(user_id="portable-user", session_id=result.session_id, item_id="e-2", reaction="like")


def test_second_customer_manifest_mapping_capabilities_and_graceful_unsupported_operation():
    manifest = CustomerManifest(
        customer_id="electronics-mock",
        version="1",
        domain_id="electronics",
        request_schema="ElectronicsQuery",
        operations=("recommend", "lookup"),
        capabilities=("category_filter", "max_price_lte", "portable_filter"),
        field_mapping=(("category", "category"), ("max_price", "price_rub"), ("portable", "weight_kg")),
    )
    assert dict(manifest.field_mapping)["max_price"] == "price_rub"
    assert "portable_filter" in manifest.capabilities
    unsupported = execute_operation(manifest, "history")
    assert isinstance(unsupported, UnsupportedOperation)
    assert unsupported.ok is False
    assert unsupported.code == "unsupported_operation"
    assert "history" in unsupported.message


def test_second_customer_preserves_category_across_price_clarification():
    class ClarifyingBackend:
        def __init__(self):
            self.calls = 0

        def structured(self, schema, system, payload):
            del system, payload
            self.calls += 1
            if self.calls == 1:
                return schema(
                    updates=[{"field": "category", "value": "laptop", "source_text": "ноутбук"}],
                    issues=[{"kind": "ambiguity", "field": "max_price", "message": "Уточните цену", "source_text": "недорогой"}],
                ), 1
            return schema(updates=[{"field": "max_price", "value": 100000, "source_text": "100000 рублей"}]), 1

    class TwoItemsProvider(ElectronicsProvider):
        def retrieve(self, user_id, query, limit=100):
            return super().retrieve(user_id, query, limit)

    adapter = SchemaRequestAdapter(
        ElectronicsQuery,
        domain_field="category",
        aliases={"category": {"ноутбук": "laptop", "планшет": "tablet"}},
    )
    domain = DomainSpec(
        id="electronics",
        version="1",
        request_schema="ElectronicsQuery",
        fields=(
            FieldSpec(name="category", value_type="enum"),
            FieldSpec(name="max_price", value_type="integer", item_field="price_rub", operators=("lte",), default_operator="lte"),
        ),
    )
    service = build_domain_workflow(domain=domain, provider=TwoItemsProvider(), backend=ClarifyingBackend(), adapter=adapter)
    first = service.chat(user_id="clarify-user", message="Нужен недорогой ноутбук")
    assert first.state == "clarify"
    assert service.sessions[first.session_id].constraint_state.pending.__class__.__name__ == "PendingState"
    second = service.chat(user_id="clarify-user", session_id=first.session_id, message="До 100000 рублей")
    assert second.state == "recommend"
    assert second.query["category"] == "laptop"
    assert second.item_ids == ["e-1"]


def onboarded_electronics(provider=None, manifest_update=None, capabilities=None):
    from pathlib import Path

    from recagent.onboarding import load_customer_manifest, onboard_customer

    manifest = load_customer_manifest(Path(__file__).resolve().parents[2] / "configs/customers/electronics.yaml")
    if manifest_update:
        manifest = manifest.model_copy(update=manifest_update)

    class ConversationBackend:
        def structured(self, schema, system, payload):
            if "недорогой" in payload["message"]:
                return schema(
                    updates=[{"field": "category", "value": "laptop", "source_text": "ноутбук"}],
                    issues=[{"kind": "ambiguity", "field": "max_price", "message": "Какой бюджет?", "source_text": "недорогой"}],
                ), 1
            return schema(updates=[{"field": "max_price", "value": 100000, "source_text": "100000 рублей"}]), 1

    adapter = SchemaRequestAdapter(
        ElectronicsQuery, domain_field="category", aliases={"category": {"ноутбук": "laptop", "планшет": "tablet"}}
    )
    domain = DomainSpec(
        id="electronics",
        version="1",
        request_schema="ElectronicsQuery",
        fields=(
            FieldSpec(name="category", value_type="enum"),
            FieldSpec(name="max_price", value_type="integer", item_field="price_rub", operators=("lte",), default_operator="lte"),
            FieldSpec(name="portable", value_type="boolean"),
        ),
    )
    return onboard_customer(
        manifest=manifest,
        domain=domain,
        provider=provider or ElectronicsProvider(),
        backend=ConversationBackend(),
        adapter=adapter,
        supported_capabilities=set(manifest.capabilities) if capabilities is None else capabilities,
        item_projection=lambda item: {**item.model_dump(), "portable": item.weight_kg <= 1.5},
    )


def test_manifest_drives_real_second_customer_conversation_and_unsupported_operation():
    provider = ElectronicsProvider()
    provider.items["tablet"] = ElectronicsItem(sku="tablet", name="Tablet", category="tablet", price_rub=30000, weight_kg=0.5, score=1)
    customer = onboarded_electronics(provider)
    first = customer.chat(user_id="manifest-user", message="Нужен недорогой ноутбук")
    assert first.state == "clarify"
    second = customer.chat(user_id="manifest-user", session_id=first.session_id, message="До 100000 рублей")
    assert second.state == "recommend" and second.item_ids == ["e-1"]
    assert second.query["category"] == "laptop" and second.query["max_price"] == 100000
    assert customer.execute("history").code == "unsupported_operation"
    assert customer.execute("lookup", item_ids=second.item_ids)[0].price_rub <= 100000


def test_onboarding_fails_closed_on_mapping_capability_and_backend_failure():
    with pytest.raises(ValueError, match="canonical metadata"):
        onboarded_electronics(manifest_update={"field_mapping": (("max_price", "score"),)})
    with pytest.raises(ValueError, match="capabilities"):
        onboarded_electronics(capabilities=set())
    customer = onboarded_electronics(manifest_update={"operations": ("recommend",)})
    assert customer.chat(user_id="blocked", message="Нужен ноутбук").code == "unsupported_operation"

    class BrokenProvider(ElectronicsProvider):
        def retrieve(self, *args, **kwargs):
            raise TimeoutError("controlled backend outage")

    customer = onboarded_electronics(BrokenProvider())
    first = customer.chat(user_id="outage", message="Нужен недорогой ноутбук")
    response = customer.chat(user_id="outage", session_id=first.session_id, message="До 100000 рублей")
    assert response.ok is False and response.code == "upstream_unavailable"
    assert response.error_type == "TimeoutError"


@pytest.mark.parametrize("missing_method", ["lookup", "retrieve"])
def test_onboarding_rejects_declared_operation_without_callable_adapter(missing_method):
    from types import SimpleNamespace

    available_method = "retrieve" if missing_method == "lookup" else "lookup"
    provider = SimpleNamespace(**{available_method: lambda *_args, **_kwargs: []})
    with pytest.raises(ValueError, match=f"requires callable provider.{missing_method}"):
        onboarded_electronics(provider)


@pytest.mark.parametrize("error_name", ["ReadTimeout", "ConnectError"])
def test_onboarded_customer_degrades_real_http_transport_failure(error_name):
    import httpx

    from recagent.onboarding import UnavailableOperation

    class OfflineBackend:
        def structured(self, *args, **kwargs):
            raise getattr(httpx, error_name)("controlled offline transport failure")

    customer = onboarded_electronics()
    customer.workflow.interpreter.backend = OfflineBackend()
    response = customer.chat(user_id="transport-user", message="Нужен ноутбук")
    assert isinstance(response, UnavailableOperation)
    assert response.ok is False
    assert response.operation == "recommend"
    assert response.code == "upstream_unavailable"
    assert response.error_type == error_name
    assert "controlled offline" not in response.message
