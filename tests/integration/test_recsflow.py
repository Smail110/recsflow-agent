import httpx
import pytest
from fastapi.testclient import TestClient

from recagent.agent import Agent
from recagent.api import create_app
from recagent.config.settings import ProviderSettings
from recagent.mock_platform import create_mock
from recagent.models import ChatRequest, Query
from recagent.providers import matches
from recagent.recsflow import RecsflowProvider


def test_chat_uses_http_contract_and_respects_constraints():
    with TestClient(create_mock()) as client:
        provider = RecsflowProvider(ProviderSettings(), client=client)
        response = Agent(provider=provider, mode="rules").chat(ChatRequest(user_id="demo", message="Лёгкий детективный сериал один сезон"))
        assert response.recommendations
        assert response.degradation == "FULL"
        assert all(matches(rec.item, response.query) for rec in response.recommendations)


def test_metadata_failure_does_not_fabricate_filtered_recommendations():
    mock = create_mock()
    mock.state.failures.add("/v1/items:batchGet")
    with TestClient(mock) as client:
        provider = RecsflowProvider(ProviderSettings(max_retries=0), client=client)
        response = Agent(provider=provider).chat(ChatRequest(user_id="new", message="Лёгкий детективный сериал"))
        assert response.degradation == "NO_EXPLAIN"
        assert response.platform_ids
        assert not response.recommendations


def test_retrieval_failure_returns_503_from_public_api():
    mock = create_mock()
    mock.state.failures.add("/v1/recommendations")
    with TestClient(mock) as client:
        provider = RecsflowProvider(ProviderSettings(max_retries=0), client=client)
        with TestClient(create_app(Agent(provider=provider))) as api:
            response = api.post("/v1/chat", json={"user_id": "new", "message": "Лёгкий детективный сериал"})
            assert response.status_code == 503
            assert response.json()["degradation"] == "UNAVAILABLE"


def test_null_metadata_does_not_satisfy_a_duration_limit():
    item = RecsflowProvider._item({"id": "unknown", "title": "Без данных", "kind": "film", "minutes": None})
    assert not matches(item, Query(kind="film", max_minutes=90))
    assert item.minutes is None


def test_circuit_breaker_stops_repeated_requests():
    calls = []

    def failure(request):
        calls.append(request)
        return httpx.Response(503)

    with httpx.Client(base_url="http://mock", transport=httpx.MockTransport(failure)) as client:
        provider = RecsflowProvider(ProviderSettings(max_retries=1, circuit_breaker_threshold=1), client=client)
        with pytest.raises(httpx.HTTPStatusError):
            provider.retrieve("u", Query())
        with pytest.raises(ConnectionError):
            provider.retrieve("u", Query())
    assert len(calls) == 2


def test_dislikes_are_not_positive_history():
    def history(request):
        return httpx.Response(200, json={"events": [{"item_id": "bad", "event_type": "disliked"}, {"item_id": "good", "event_type": "liked"}]})

    with httpx.Client(base_url="http://mock", transport=httpx.MockTransport(history)) as client:
        provider = RecsflowProvider(ProviderSettings(), client=client)
        positive, blocked = provider.history_snapshot("u")
        assert positive == ["good"] and blocked == {"bad"}
