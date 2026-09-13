"""Одинаковые проверки для локального mock и согласованного внешнего API."""

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from recagent.mock_platform import create_mock
from recagent.platform_contract import validate_schema

pytestmark = pytest.mark.contract


@pytest.fixture
def client():
    base_url = os.getenv("RECAGENT_CONTRACT_URL")
    with (httpx.Client(base_url=base_url, timeout=10) if base_url else TestClient(create_mock())) as connection:
        yield connection


def test_platform_contract(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    response = client.post("/v1/recommendations", json={"user_id": "demo", "limit": 5})
    response.raise_for_status()
    data = response.json()
    validate_schema(data, "RecommendationResponse")
    assert len(data["items"]) <= 5
    ids = [item["item_id"] for item in data["items"]]
    assert ids, "Для проверки нужен непустой демонстрационный профиль"
    batch = client.post("/v1/items:batchGet", json={"item_ids": ids}).json()
    validate_schema(batch, "BatchResponse")
    assert {item["id"] for item in batch["items"]} | set(batch["missing_item_ids"]) == set(ids)
    item = client.get(f"/v1/items/{ids[0]}").json()
    validate_schema(item, "Item")
    search = client.get("/v1/items/search", params={"title": item["title"]}).json()
    validate_schema(search, "SearchResponse")
    assert ids[0] in {entry["id"] for entry in search["items"]}
    history = client.get("/v1/users/demo/history", params={"limit": 2}).json()
    validate_schema(history, "HistoryResponse")


def test_invalid_limit_is_rejected(client):
    response = client.post("/v1/recommendations", json={"user_id": "demo", "limit": 0})
    assert response.status_code in (400, 422)
