"""The bundled HTTP mock declares exact-title support; unknown production does not."""

import httpx
import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError

from recagent.api import create_app
from recagent.config.settings import ProviderSettings, Settings, _env_overrides
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.mock_platform import create_mock
from recagent.models import Item
from recagent.recsflow import RecsflowProvider
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


def catalog_mock():
    app = create_mock()
    seed = Item(
        id="seed",
        title="Тёмный маршрут",
        kind="series",
        genre="детектив",
        tone="мрачный",
        minutes=43,
        seasons=1,
        quality=0.5,
        description="Synthetic contract fixture",
    )
    near = seed.model_copy(update={"id": "near", "title": "Последнее письмо", "quality": 0.3})
    far = seed.model_copy(update={"id": "far", "title": "Праздник у моря", "genre": "комедия", "tone": "лёгкий", "quality": 0.9})
    app.state.provider.items = {item.id: item for item in [seed, near, far]}
    return app, seed, near


class Interpreter:
    def __init__(self, intent, updates):
        self.result = StructuredRequest(intent=intent, updates=[ConstraintUpdate(**row) for row in updates])

    def interpret(self, *args, **kwargs):
        return self.result, 0


def agent(provider, intent, updates):
    return WorkflowAgent(
        provider=provider,
        mode="ollama",
        question_policy="adaptive",
        interpreter=Interpreter(intent, updates),
        response_generator=EvidenceResponseGenerator(),
    )


def test_unknown_provider_does_not_call_or_advertise_verified_title_lookup():
    def forbidden(request):
        raise AssertionError("Unsupported provider must never receive a guessed title endpoint")

    with httpx.Client(base_url="http://unconfigured", transport=httpx.MockTransport(forbidden)) as client:
        provider = RecsflowProvider(ProviderSettings(), client=client)
        assert provider.title_lookup_verified is False
        assert "find_title_verified" not in provider.capabilities.supported_operations
        with pytest.raises(NotImplementedError, match="not declared"):
            provider.find_title_verified("Anything")
        core = agent(provider, "navigation", [{"field": "seed_title", "value": "Anything", "source_text": "Anything"}])
        with TestClient(create_app(core)) as api:
            response = api.post("/v1/chat", json={"message": "Найди Anything", "user_id": "new"})
        assert response.status_code == 200
        assert response.json()["state"] == "clarify"
        assert not response.json()["recommendations"]


def test_exact_mock_lookup_normalizes_case_and_yo_but_never_fuzzy_matches():
    mock, seed, _ = catalog_mock()
    with TestClient(mock) as client:
        provider = RecsflowProvider(ProviderSettings(verified_title_lookup=True), client=client)
        assert provider.title_lookup_verified
        assert "find_title_verified" in provider.capabilities.supported_operations
        assert provider.find_title_verified("  ТЕМНЫЙ МАРШРУТ  ").id == seed.id
        assert provider.find_title_verified("Тёмный маршру") is None
        assert provider.find_title_verified("Нет такого названия") is None


def test_exhaustive_mock_duplicates_fail_closed_even_with_different_kinds():
    mock, seed, _ = catalog_mock()
    mock.state.provider.items["duplicate"] = seed.model_copy(update={"id": "duplicate", "kind": "film", "title": "Темный маршрут"})
    with TestClient(mock) as client:
        provider = RecsflowProvider(ProviderSettings(verified_title_lookup=True), client=client)
        response = client.get("/v1/items/resolve-title", params={"title": seed.title})
        assert response.json()["exhaustive"] is True and len(response.json()["items"]) == 2
        assert provider.find_title_verified(seed.title) is None
        core = agent(provider, "navigation", [{"field": "seed_title", "value": seed.title, "source_text": seed.title}])
        with TestClient(create_app(core)) as api:
            response = api.post("/v1/chat", json={"message": f"Найди {seed.title}", "user_id": "new"})
        assert response.json()["state"] == "clarify" and not response.json()["recommendations"]


@pytest.mark.parametrize(
    "invalid",
    [
        {"items": [], "exhaustive": False, "match_mode": "exact-normalized-v1"},
        {"items": [], "exhaustive": True, "match_mode": "fuzzy"},
        {"items": [{"id": "wrong", "title": "Other title", "kind": "film"}], "exhaustive": True, "match_mode": "exact-normalized-v1"},
    ],
)
def test_adapter_rejects_incomplete_or_nonexact_declared_response(invalid):
    with httpx.Client(base_url="http://mock", transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=invalid))) as client:
        provider = RecsflowProvider(ProviderSettings(verified_title_lookup=True), client=client)
        with pytest.raises((ValidationError, ValueError)):
            provider.find_title_verified("Required title")


@pytest.mark.parametrize(
    "intent,message,updates,expected_ids",
    [
        (
            "navigation",
            "Найди Тёмный маршрут",
            [{"field": "seed_title", "value": "Тёмный маршрут", "source_text": "Тёмный маршрут"}],
            ["seed"],
        ),
        (
            "similar",
            "Похожее на Тёмный маршрут",
            [{"field": "seed_title", "value": "Тёмный маршрут", "source_text": "Тёмный маршрут"}],
            ["near", "far"],
        ),
        (
            "discovery",
            "Хочу сериал детектив",
            [
                {"field": "kind", "value": "series", "source_text": "сериал"},
                {"field": "genre", "value": "детектив", "source_text": "детектив"},
            ],
            ["seed", "near"],
        ),
        (
            "mood",
            "Хочу лёгкий сериал",
            [{"field": "kind", "value": "series", "source_text": "сериал"}, {"field": "tone", "value": "лёгкий", "source_text": "лёгкий"}],
            ["far"],
        ),
    ],
)
def test_four_intents_through_http_adapter_and_public_chat_api(intent, message, updates, expected_ids):
    mock, _, _ = catalog_mock()
    with TestClient(mock) as client:
        provider = RecsflowProvider(ProviderSettings(verified_title_lookup=True), client=client)
        with TestClient(create_app(agent(provider, intent, updates))) as api:
            response = api.post("/v1/chat", json={"message": message, "user_id": "new"})
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "recommend" and data["query"]["intent"] == intent
        ids = [row["item"]["id"] for row in data["recommendations"]]
        assert set(ids) == set(expected_ids)
        if intent in {"navigation", "similar"}:
            assert ids[0] == expected_ids[0]


def test_verified_title_endpoint_failure_gracefully_degrades_public_http():
    mock, seed, _ = catalog_mock()
    mock.state.failures.add("/v1/items/resolve-title")
    with TestClient(mock) as client:
        provider = RecsflowProvider(ProviderSettings(verified_title_lookup=True, max_retries=0), client=client)
        core = agent(provider, "navigation", [{"field": "seed_title", "value": seed.title, "source_text": seed.title}])
        with TestClient(create_app(core)) as api:
            response = api.post("/v1/chat", json={"message": f"Найди {seed.title}", "user_id": "new"})
        assert response.status_code == 503
        assert response.json()["degradation"] == "UNAVAILABLE"
        assert not response.json().get("recommendations")


def test_verified_lookup_environment_requires_explicit_valid_boolean(monkeypatch):
    assert ProviderSettings().verified_title_lookup is False
    monkeypatch.setenv("RECAGENT_PROVIDER_VERIFIED_TITLE_LOOKUP", "true")
    assert Settings.model_validate(_env_overrides()).provider.verified_title_lookup is True
    monkeypatch.setenv("RECAGENT_PROVIDER_VERIFIED_TITLE_LOOKUP", "false")
    assert Settings.model_validate(_env_overrides()).provider.verified_title_lookup is False
    monkeypatch.setenv("RECAGENT_PROVIDER_VERIFIED_TITLE_LOOKUP", "maybe")
    with pytest.raises(ValueError):
        Settings.model_validate(_env_overrides())
