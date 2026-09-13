import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recagent.agent import Agent
from recagent.api import create_app
from recagent.catalog import CATALOG_SIZE, catalog_sha256, generate_catalog
from recagent.grounding import explain, validate_evidence
from recagent.models import ChatRequest, Evidence, Query
from recagent.providers import DemoProvider

_SRC = Path(__file__).resolve().parents[1] / "src"


def _seed_title(genre="детектив", kind="series"):
    """Реальное название из каталога. Хардкодить названия нельзя: каталог генерируется."""
    item = next(i for i in generate_catalog(42) if i.genre == genre and i.kind == kind)
    return item.title


def chat(agent, text, sid=None, user="new-user"):
    return agent.chat(ChatRequest(message=text, session_id=sid, user_id=user))


def test_catalog_reproducible():
    assert generate_catalog(42) == generate_catalog(42)
    assert generate_catalog(42) != generate_catalog(43)
    assert len(generate_catalog()) == CATALOG_SIZE == 3000


def test_catalog_deterministic_across_processes():
    """Отпечаток совпадает между процессами, а не только внутри одного.

    Сравнение объектов в одном процессе не поймало бы зависимость от порядка
    обхода словаря или от случайного hash-сида Python: PYTHONHASHSEED различается
    между запусками. Поэтому второй отпечаток считается в отдельном процессе.
    """
    code = "from recagent.catalog import catalog_sha256; print(catalog_sha256(42))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": str(_SRC)})
    assert out.stdout.strip() == catalog_sha256(42)


def test_negation_with_qualifier():
    result = chat(Agent(mode="rules"), "Детективный сериал, не слишком мрачный, один сезон")
    assert result.recommendations
    assert all(r.item.tone == "лёгкий" for r in result.recommendations)
    result = chat(Agent(mode="rules"), "Хочу не лёгкий фильм")
    assert result.state == "clarify"


def test_constraints_survive_followup_and_release():
    agent = Agent(mode="rules")
    first = chat(agent, "Хочу детективный сериал, не мрачный, не длиннее одного сезона")
    assert first.state == "recommend"
    assert all(r.item.seasons == 1 and r.item.tone == "лёгкий" and r.item.genre == "детектив" for r in first.recommendations)
    second = chat(agent, "Не дольше 30 минут", first.session_id)
    assert second.recommendations
    assert all(r.item.minutes <= 30 and r.item.seasons == 1 and r.item.genre == "детектив" for r in second.recommendations)
    third = chat(agent, "Без ограничений по длительности", first.session_id)
    assert third.query.max_minutes is None and third.query.max_seasons == 1


def test_clarification_and_new_domain():
    agent = Agent(mode="rules")
    first = chat(agent, "Посоветуй что-нибудь")
    assert first.state == "clarify"
    second = chat(agent, "Лёгкий детективный сериал", first.session_id)
    assert second.recommendations
    course = chat(agent, "Курс по машинному обучению для новичка, без воды", first.session_id)
    assert course.recommendations
    assert course.query.tone is None
    assert all(r.item.kind == "course" and r.item.level == "начальный" and r.item.practical for r in course.recommendations)


def test_empty_results_never_relax_constraints():
    result = chat(Agent(mode="rules"), "Лёгкий детективный сериал не дольше 1 минуты")
    assert result.state == "no_results" and not result.recommendations
    assert result.query.max_minutes == 1


def test_feedback_and_more():
    agent = Agent(mode="rules")
    first = chat(agent, "Комедийный фильм лёгкий")
    bad = first.recommendations[0].item.id
    agent.feedback(first.session_id, bad, "dislike")
    second = chat(agent, "Ещё", first.session_id)
    assert not ({r.item.id for r in first.recommendations} & {r.item.id for r in second.recommendations})
    with pytest.raises(ValueError):
        agent.feedback(first.session_id, "not-shown", "like")


def test_grounding_rejects_forgery_and_no_fake_history():
    agent = Agent(mode="rules")
    result = chat(agent, "Лёгкий детективный сериал")
    for rec in result.recommendations:
        assert all(validate_evidence(e, rec.item, result.query, []) for e in rec.evidence)
        assert not any(e.relation == "history" for e in rec.evidence)
        forged = Evidence(item_id=rec.item.id, field="seasons", value=99)
        assert not validate_evidence(forged, rec.item, result.query, [])
        fake_history = Evidence(item_id=rec.item.id, field="genre", value=rec.item.genre, relation="history", source_item_id="absent")
        assert not validate_evidence(fake_history, rec.item, result.query, [])


def test_history_exclusion_and_personalization():
    provider = DemoProvider()
    history_ids = set(provider.history("demo"))
    # Если история пуста, тест вырождается в проверку «ничего не исключено».
    assert history_ids, "демо-профиль обязан иметь историю, иначе персонализация не проверяется"

    result = chat(Agent(provider=provider, mode="rules"), "Лёгкий детективный сериал", user="demo")
    assert all(r.item.id not in history_ids for r in result.recommendations)
    assert any(e.relation == "history" for r in result.recommendations for e in r.evidence)
    # evidence про историю обязан ссылаться на объект, который реально в истории
    assert all(e.source_item_id in history_ids for r in result.recommendations for e in r.evidence if e.relation == "history")


def test_unknown_seasons_do_not_break_explanation():
    """Объект с неизвестным числом сезонов объясняется, а не роняет запрос.

    null = «нет данных»: про такой атрибут нельзя ни утверждать, ни падать.
    """
    item = next(i for i in generate_catalog(42) if i.kind == "series" and i.seasons is None)
    rec = explain(item, Query(kind="series"), [], 0.5)
    assert rec.explanation and "Сезонов" not in rec.explanation
    assert all(e.value is not None for e in rec.evidence)
    assert not any(e.field == "seasons" for e in rec.evidence)


def test_seed_and_missing_seed():
    title = _seed_title()
    agent = Agent(mode="rules")
    result = chat(agent, f"Сериал похожий на «{title}»")
    assert result.recommendations
    assert result.recommendations[0].item.genre == "детектив"
    assert all(r.item.title != title for r in result.recommendations)
    assert any(e.relation == "seed" for e in result.recommendations[0].evidence)

    result = chat(Agent(mode="rules"), "Сериал похожий на «Неизвестный сериал»")
    assert result.state == "clarify" and not result.recommendations

    # Опечатка в одном символе. Падежное склонение процедурных названий не
    # гарантировано, поэтому проверяем устойчивость к опечатке, а не к падежу.
    typo = title[:-2] + ("ж" if title[-1] != "ж" else "ш") + title[-1]
    result = chat(Agent(mode="rules"), f"Сериал похожий на «{typo}»")
    assert result.recommendations and result.recommendations[0].item.genre == "детектив"


def test_llm_domain_change_discards_stale_slots():
    class LeakyLLM:
        def parse(self, message, previous):
            if "Курс" in message:
                return Query(kind="course", genre="python", tone="лёгкий", level="начальный", practical=True), 10
            return Query(kind="series", genre="детектив", tone="лёгкий", max_seasons=1), 10
    agent = Agent(mode="ollama", llm=LeakyLLM())
    first = chat(agent, "Лёгкий детективный сериал один сезон")
    second = chat(agent, "Курс python для новичка с практикой", first.session_id)
    assert second.query.tone is None and second.recommendations


class BrokenLLM:
    def parse(self, *args):
        raise TimeoutError("test")


def test_fallback_budget_and_reset_do_not_bypass_budget():
    agent = Agent(mode="ollama", llm=BrokenLLM(), max_calls=1)
    first = chat(agent, "Лёгкий детективный сериал")
    assert first.recommendations and first.mode == "rules_fallback" and first.llm_calls == 1
    chat(agent, "сброс", first.session_id)
    second = chat(agent, "Комедийный фильм", first.session_id)
    assert second.llm_calls == 0 and second.llm_calls_total == 1 and second.warnings


class BrokenProvider(DemoProvider):
    def retrieve(self, *args, **kwargs):
        raise TimeoutError("test")


def test_provider_failure_does_not_fabricate_catalog():
    result = chat(Agent(provider=BrokenProvider(), mode="rules"), "Лёгкий сериал")
    assert result.state == "no_results" and not result.recommendations
    assert result.warnings


def test_api_validation_sessions_feedback_and_health():
    client = TestClient(create_app(Agent(mode="rules")))
    assert client.get("/health").status_code == 200
    assert client.post("/v1/chat", json={"message": "   "}).status_code == 422
    assert client.post("/v1/chat", json={"message": "x"*2001}).status_code == 422
    data = client.post("/v1/chat", json={"message": "Лёгкий сериал", "user_id": "alice"}).json()
    assert client.post("/v1/chat", json={"message": "ещё", "session_id": data["session_id"], "user_id": "bob"}).status_code == 404
    assert client.post("/v1/feedback", json={"session_id": data["session_id"], "item_id": data["recommendations"][0]["item"]["id"], "reaction": "like"}).status_code == 200
    assert client.post("/v1/chat", json={"message": "Привет", "session_id": "missing", "user_id": "alice"}).status_code == 404


def test_session_limit_expiry_and_parallel_isolation():
    agent = Agent(mode="rules", max_sessions=1, ttl=.01)
    first = chat(agent, "Лёгкий сериал")
    with pytest.raises(RuntimeError):
        chat(agent, "Лёгкий фильм")
    time.sleep(.02)
    with pytest.raises(KeyError):
        chat(agent, "Ещё", first.session_id)
    agent = Agent(mode="rules")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda x: chat(agent, x), ["Лёгкий сериал", "Мрачный фильм"]*5))
    assert len({r.session_id for r in results}) == 10
    assert all(r.query.kind == ("series" if index % 2 == 0 else "film") for index, r in enumerate(results))


def test_hard_filter_applied_even_if_provider_ignores_query():
    class LooseProvider(DemoProvider):
        def retrieve(self, user_id, query, limit=100):
            return list(self.items)
    result = chat(Agent(provider=LooseProvider(), mode="rules"), "Лёгкий детективный сериал, один сезон")
    assert result.recommendations
    assert all(r.item.genre == "детектив" and r.item.kind == "series" and r.item.seasons == 1 for r in result.recommendations)
