from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from recagent.domains.base import DomainSpec, FieldSpec
from recagent.domains.demo import domain_spec, request_adapter
from recagent.factory import build_agent, build_domain_workflow
from recagent.interpretation import bridge_domain_transport, domain_contract_descriptor, domain_transport_schema
from recagent.models import ChatRequest
from recagent.request_mapping import SchemaRequestAdapter


def test_field_oriented_schema_preserves_multiple_operations_and_null_means_no_update():
    schema = domain_transport_schema(domain_spec())
    response = schema.model_validate(
        {
            "tone": [
                {
                    "operation": "exclude",
                    "value": "\u043c\u0440\u0430\u0447\u043d\u044b\u0439",
                    "source_text": "\u043d\u0435 \u043c\u0440\u0430\u0447\u043d\u044b\u0439",
                },
                {
                    "operation": "include",
                    "value": "\u043b\u0451\u0433\u043a\u0438\u0439",
                    "source_text": "\u043b\u0451\u0433\u043a\u0438\u0439 \u043c\u043e\u0436\u043d\u043e",
                },
            ],
            "kind": None,
        }
    )
    bridged = bridge_domain_transport(response, domain_spec())
    assert [(update.field, update.operation, update.value) for update in bridged.updates] == [
        ("tone", "exclude", "\u043c\u0440\u0430\u0447\u043d\u044b\u0439"),
        ("tone", "include", "\u043b\u0451\u0433\u043a\u0438\u0439"),
    ]
    assert not any(update.operation == "clear" for update in bridged.updates)


def test_field_oriented_schema_rejects_undeclared_operation_before_bridge():
    schema = domain_transport_schema(domain_spec())
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "level": [
                    {
                        "operation": "exclude",
                        "value": "\u043d\u0430\u0447\u0430\u043b\u044c\u043d\u044b\u0439",
                        "source_text": "\u043d\u0435 \u043d\u0430\u0447\u0430\u043b\u044c\u043d\u044b\u0439",
                    }
                ]
            }
        )


def test_field_oriented_schema_requires_typed_value_except_clear_and_descriptor_matches_spec():
    spec = domain_spec()
    schema = domain_transport_schema(spec)
    with pytest.raises(ValidationError):
        schema.model_validate({"max_minutes": [{"operation": "set", "value": "60", "source_text": "60 минут"}]})
    with pytest.raises(ValidationError):
        schema.model_validate({"tone": [{"operation": "exclude", "source_text": "не мрачный"}]})

    descriptor = domain_contract_descriptor(request_adapter().descriptor, spec)
    tone = next(field for field in descriptor["fields"] if field["name"] == "tone")
    minutes = next(field for field in descriptor["fields"] if field["name"] == "max_minutes")
    assert tone["canonical_operators"] == ["eq", "neq"]
    assert tone["wire_operations"] == ["set", "clear", "exclude", "include"]
    assert minutes["value_type"] == "integer" and minutes["unit"] == "minute"
    assert minutes["canonical_operators"] == ["eq", "lte"]


def test_b1_null_permissive_schema_is_explicit_and_b2_requires_a_value():
    permissive = domain_transport_schema(domain_spec(), require_value=False)
    required = domain_transport_schema(domain_spec(), require_value=True)
    assert permissive.model_validate({"tone": [{"operation": "set", "value": None, "source_text": "тон"}]}).tone
    with pytest.raises(ValidationError):
        required.model_validate({"tone": [{"operation": "set", "value": None, "source_text": "тон"}]})


def test_workflow_uses_domain_field_schema_and_existing_validation_path():
    class Backend:
        def structured(self, schema, system, payload):
            assert "kind" in schema.model_json_schema()["properties"]
            assert payload["domain"]["id"] == "demo-media-course"
            return schema(
                kind=[{"operation": "set", "value": "course", "source_text": "\u043a\u0443\u0440\u0441"}],
                genre=[{"operation": "set", "value": "python", "source_text": "Python"}],
            ), 17

    config = {
        "validation": {"semantic_mode": "code-only", "nli": False},
        "retrieval": {"provider_k": 100, "lexical_k": 100, "fused_k": 150, "dense": False},
        "ranking": {"method": "deterministic-rrf", "rrf_k": 60},
        "reranker": False,
        "interpretation": {"transport": "domain-fields", "experimental": True},
    }
    agent = build_agent(mode="ollama", llm=Backend(), workflow_config=config)
    response = agent.chat(ChatRequest(user_id="domain-wire", message="\u041d\u0443\u0436\u0435\u043d \u043a\u0443\u0440\u0441 Python"))
    assert agent.interpretation_transport == "domain-fields"
    assert agent.domain_value_required is True
    assert response.state == "recommend"
    assert response.query.kind == "course" and response.query.genre == "python"
    assert "proposal_validation" in response.trace


def test_second_customer_can_use_domain_transport_without_demo_fields():
    class ShopQuery(BaseModel):
        intent: str = "discovery"
        category: Literal["laptop"] | None = None

    class Item(BaseModel):
        sku: str
        category: str

    class Provider:
        def retrieve(self, user_id, query, limit=100):
            del user_id, query, limit
            return ["shop-1"]

        def lookup(self, ids):
            return [Item(sku="shop-1", category="laptop")] if "shop-1" in ids else []

    class Backend:
        def structured(self, schema, system, payload):
            assert "category" in schema.model_json_schema()["properties"]
            assert all(field["name"] != "kind" for field in payload["domain"]["fields"])
            return schema(category=[{"operation": "set", "value": "laptop", "source_text": "laptop"}]), 3

    domain = DomainSpec(
        id="shop",
        version="1",
        request_schema="ShopQuery",
        fields=(FieldSpec(name="category", value_type="enum"),),
    )
    adapter = SchemaRequestAdapter(ShopQuery, aliases={"category": {"laptop": "laptop"}})
    service = build_domain_workflow(
        domain=domain,
        provider=Provider(),
        backend=Backend(),
        adapter=adapter,
        interpretation_transport="domain-fields",
    )
    response = service.chat(user_id="shopper", message="laptop")
    assert response.state == "recommend"
