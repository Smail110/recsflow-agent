"""Тесты D2: скрытые предпочтения theta и типизированная история профилей.

Проверяют не «генератор работает», а свойства, от которых зависит честность
измерения (``docs/DATA-PLAN.md`` §5):

  * theta нормирована и покрывает все жанры — иначе масштаб полезности плавает;
  * история согласована с theta — иначе персонализацию агента нельзя отличить
    от шума, а бейзлайн B1 получил бы бессмысленный вход;
  * история ссылается только на реально существующие объекты каталога — иначе
    lookup молча вернул бы пустой список, и персонализация тихо отключилась бы;
  * полезность различает объекты — иначе приемлемое множество было бы всем;
  * набор воспроизводим между процессами — иначе «seed=42» ничего не значит.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from recagent.catalog import generate_catalog
from recagent.catalog.users import (
    ARCHETYPES,
    DISLIKED_POOL_FRACTION,
    GENRES,
    TONES,
    HISTORY_SIZE_RANGE,
    LIKED_POOL_FRACTION,
    LIKED_UTILITY_FLOOR,
    PROFILE_COUNT,
    Theta,
    UserProfile,
    course_fit,
    duration_fit,
    generate_profiles,
    profile_sha256,
    profile_stats,
    profiles_by_id,
    tone_fit,
    utility,
)
from recagent.models import Item

_SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def catalog() -> list[Item]:
    return generate_catalog(42)


@pytest.fixture(scope="module")
def profiles(catalog) -> list[UserProfile]:
    return generate_profiles(42, PROFILE_COUNT, catalog)


@pytest.fixture(scope="module")
def catalog_by_id(catalog) -> dict[str, Item]:
    return {item.id: item for item in catalog}


# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------


def test_profiles_reproducible(catalog):
    assert generate_profiles(42, 60, catalog) == generate_profiles(42, 60, catalog)
    assert generate_profiles(42, 60, catalog) != generate_profiles(43, 60, catalog)


def test_profiles_reproducible_across_processes():
    """Отпечаток совпадает между процессами, а не только внутри одного.

    Сравнение в одном процессе не поймало бы зависимость от порядка обхода
    словаря или от случайного hash-сида Python (PYTHONHASHSEED различается между
    запусками), поэтому второй отпечаток считается в отдельном процессе.
    """
    code = "from recagent.catalog.users import profile_sha256; print(profile_sha256(42, 50))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
    )
    assert out.stdout.strip() == profile_sha256(42, 50)


def test_rejects_invalid_arguments(catalog):
    with pytest.raises(ValueError):
        generate_profiles(-1, 10, catalog)
    with pytest.raises(ValueError):
        generate_profiles(42, 0, catalog)


def test_default_count_matches_plan(catalog):
    """D2: ~500 профилей (docs/DATA-PLAN.md §4). Меньше — недобор мощности."""
    assert PROFILE_COUNT == 500
    assert len(generate_profiles(42, catalog=catalog)) == PROFILE_COUNT


# ---------------------------------------------------------------------------
# theta
# ---------------------------------------------------------------------------


def test_genre_weights_are_normalized(profiles):
    for profile in profiles:
        assert math.isclose(math.fsum(profile.theta.genre_weights.values()), 1.0, abs_tol=1e-6)
        assert set(profile.theta.genre_weights) == set(GENRES)
        assert all(weight > 0 for weight in profile.theta.genre_weights.values())


def test_theta_rejects_unnormalized_weights():
    """Ненормированные веса обязаны падать при создании, а не молча менять масштаб.

    Профиль можно собрать и вручную (adversarial-сценарии, ручная валидация),
    поэтому проверка в модели, а не только в генераторе.
    """
    with pytest.raises(ValueError):
        Theta(genre_weights=dict.fromkeys(GENRES, 1.0), duration_tol=60)
    with pytest.raises(ValueError):
        Theta(genre_weights=dict.fromkeys(GENRES[:-1], 0.0) | {GENRES[-1]: 1.0}, duration_tol=60)
    missing = {genre: 1 / (len(GENRES) - 1) for genre in GENRES[:-1]}
    with pytest.raises(ValueError):
        Theta(genre_weights=missing, duration_tol=60)


def test_every_genre_has_nonzero_weight_for_every_profile(profiles):
    """Нет жанра, который профиль отвергает по построению.

    Если бы вес был нулевым, жанр выпал бы из приемлемого множества целиком, и
    сценарий «кино-зритель спросил про курс» проваливался бы независимо от
    качества диалога — то есть измерял бы не то, что мы хотим.
    """
    for profile in profiles:
        for genre in GENRES:
            assert profile.theta.genre_weights[genre] >= 1e-6, (profile.user_id, genre)


def test_archetypes_all_present(profiles):
    """Все архетипы представлены, включая «всеядного» и короткий курс.

    Равномерное распределение по архетипам (index % len) нарочно: иначе набор
    измерял бы качество только на одном типе зрителя.
    """
    stats = profile_stats(profiles)
    counts = stats["archetypes"]
    assert set(counts) == {archetype.name for archetype in ARCHETYPES}
    assert min(counts.values()) == max(counts.values())


def test_patience_and_verbosity_are_in_range(profiles):
    for profile in profiles:
        assert 2 <= profile.theta.patience <= 5
        assert 0.0 <= profile.theta.verbosity <= 1.0
        assert 15 <= profile.theta.duration_tol <= 1500


def test_indifferent_tone_branch_is_exercised(profiles):
    """Часть профилей не имеет предпочтения по тону.

    Иначе ветка ``indifferent`` в ``tone_fit`` не исполнялась бы ни разу, и её
    корректность осталась бы непроверенной — классическая мёртвая ветка, которая
    ломается в проде.
    """
    prefs = {profile.theta.tone_pref for profile in profiles}
    assert "indifferent" in prefs
    assert sum(1 for profile in profiles if profile.theta.tone_pref == "indifferent") > 50


# ---------------------------------------------------------------------------
# Полезность
# ---------------------------------------------------------------------------


def test_utility_is_bounded(catalog, profiles):
    """Полезность в 0..1 для любого сочетания объекта и профиля.

    Выход за границу сломал бы нормировку ``(ours - B2) / (B3 - B2)``: знаменатель
    перестал бы быть «максимально достижимым улучшением».
    """
    sample = catalog[:400]
    for profile in profiles[:20]:
        for item in sample:
            value = utility(item, profile.theta)
            assert 0.0 <= value <= 1.0, (profile.user_id, item.id, value)


def test_utility_discriminates_items(catalog):
    """Полезность различает объекты внутри одного запроса.

    Если бы у всех объектов она была равна, приемлемое множество было бы
    произвольным, и success_rate измерял бы удачу, а не качество.
    """
    theta = generate_profiles(42, 1, catalog)[0].theta
    values = [utility(item, theta) for item in catalog]
    assert max(values) - min(values) > 0.3
    assert len(set(values)) > len(values) * 0.5


def test_utility_respects_preferences(catalog):
    """Профиль получает высокую полезность именно за то, что любит.

    Проверка направления эффекта по каждому архетипу: лучший объект профиля
    обязан быть лучше худшего и обязан соответствовать заявленному предпочтению.
    """
    profiles = generate_profiles(42, 10, catalog)
    for profile in profiles:
        values = sorted(((utility(item, profile.theta), item) for item in catalog), key=lambda pair: -pair[0])
        best_value, best_item = values[0]
        worst_value, _ = values[-1]
        assert best_value > worst_value
        if profile.theta.tone_pref != "indifferent":
            # Лучший объект почти всегда совпадает по тону; жёсткое равенство
            # неверно (качество может перевесить), поэтому проверяем fit > 0.5.
            assert tone_fit(best_item, profile.theta) > 0.5


def test_tone_fit_table_is_consistent():
    """Таблица тона: совпадение — максимум, противоположный тон — минимум.

    Порог вроде «худший < 0.5» здесь был бы произвольным: у предпочтения
    «нейтральный» худший вариант — мрачный с 0.55, и это разумное значение,
    а не дефект. Проверяется упорядоченность, а не конкретная величина.
    """
    from recagent.catalog.users import TONE_FIT

    opposite = {"лёгкий": "мрачный", "мрачный": "лёгкий"}
    for preferred, row in TONE_FIT.items():
        assert set(row) == set(TONES), preferred
        assert row[preferred] == 1.0, preferred
        assert row[preferred] == max(row.values()), preferred
        assert all(0 < value < 1.0 for tone, value in row.items() if tone != preferred), preferred
        if preferred in opposite:
            # Крайнее несовпадение («хочу лёгкое — дали мрачное») штрафуется сильнее,
            # чем соседнее («хочу лёгкое — дали нейтральное»).
            assert min(row, key=row.get) == opposite[preferred], preferred


def test_duration_fit_is_soft_not_hard(catalog):
    """Превышение терпимости наказывает постепенно, а не обнуляет.

    Жёсткий порог превратил бы theta в копию hard-фильтров агента — ровно ту
    тавтологию, от которой уходим. Проверяем монотонность и отсутствие скачка.
    """
    theta = generate_profiles(42, 1, catalog)[0].theta
    theta = theta.model_copy(update={"duration_tol": 60})
    item = catalog[0].model_copy(update={"minutes": 60})
    assert duration_fit(item, theta) == 1.0
    previous = 1.0
    for minutes in range(61, 361, 7):
        item = catalog[0].model_copy(update={"minutes": minutes})
        current = duration_fit(item, theta)
        assert 0.0 < current < previous, minutes
        previous = current
    # Заметное, но не нулевое наказание на удвоенной длительности.
    doubled = catalog[0].model_copy(update={"minutes": 120})
    assert duration_fit(doubled, theta) == pytest.approx(math.exp(-1.0), rel=1e-6)


def test_course_fit_is_neutral_for_non_courses(catalog):
    """Фильм и сериал не наказываются за отсутствие уровня курса.

    Иначе кино-профиль получил бы нулевую полезность всех курсов даже при
    ненулевом весе жанра, и приемлемое множество схлопнулось бы по построению.
    """
    theta = generate_profiles(42, 1, catalog)[0].theta
    theta = theta.model_copy(update={"level_pref": "начальный", "practical_pref": True})
    for item in catalog:
        if item.kind != "course":
            assert course_fit(item, theta) == 1.0


def test_unknown_practical_is_middle_not_rejection(catalog):
    """«Нет данных» о практике — середина, а не отказ (контракт: null = нет данных)."""
    theta = generate_profiles(42, 1, catalog)[0].theta
    theta = theta.model_copy(update={"practical_pref": True, "level_pref": "any"})
    unknown = next(item for item in catalog if item.kind == "course" and item.practical is None)
    known_bad = next(item for item in catalog if item.kind == "course" and item.practical is False)
    assert course_fit(unknown, theta) > course_fit(known_bad, theta)
    assert course_fit(unknown, theta) < 1.0


# ---------------------------------------------------------------------------
# История
# ---------------------------------------------------------------------------


def test_history_items_exist_in_catalog(profiles, catalog_by_id):
    """Каждое событие истории ссылается на реальный объект каталога.

    Иначе lookup молча отфильтровал бы несуществующий id, история стала бы
    короче или пустой, и персонализация с исключением просмотренного тихо
    отключились бы — метрики остались бы красивыми, а измеряли бы ничего.
    """
    for profile in profiles:
        for event in profile.history:
            assert event.item_id in catalog_by_id, (profile.user_id, event.item_id)


def test_history_is_typed_and_has_both_poles(profiles):
    """История типизирована, а не плоский список: liked, viewed и disliked есть.

    Контракт требует типизированных событий (docs/contract/openapi.yaml), потому
    что агент трактует их по-разному: viewed/liked усиливают жанр, disliked
    исключает объект. Плоский список эту разницу уничтожает.
    """
    stats = profile_stats(profiles)
    assert stats["liked_total"] > 100
    assert stats["disliked_total"] > 100
    assert stats["history_events_total"] > 1000
    seen_types = {event.event_type for profile in profiles for event in profile.history}
    assert {"liked", "viewed", "disliked"} <= seen_types


def test_history_size_is_within_declared_range(profiles):
    for profile in profiles:
        assert HISTORY_SIZE_RANGE[0] <= len(profile.history) <= HISTORY_SIZE_RANGE[0] + HISTORY_SIZE_RANGE[1]
        assert len({event.item_id for event in profile.history}) == len(profile.history), profile.user_id


def test_liked_items_are_actually_preferred(profiles, catalog_by_id):
    """Лайк ставится тому, что профилю нравится, а не случайному объекту.

    Это главное свойство D2: история обязана быть согласована с theta. Иначе
    персонализацию агента нельзя отличить от шума, а бейзлайн B1 («сырой
    Recsflow по истории») получил бы бессмысленный вход.
    """
    mismatched = 0
    total = 0
    for profile in profiles[:100]:
        for event in profile.history:
            if event.event_type != "liked":
                continue
            total += 1
            if utility(catalog_by_id[event.item_id], profile.theta) < LIKED_UTILITY_FLOOR:
                mismatched += 1
    assert total > 100, "лайков слишком мало, тест ничего не проверяет"
    assert mismatched == 0


def test_disliked_items_are_not_preferred(profiles, catalog_by_id):
    """Дизлайк — из нижней части распределения полезности, не из верхней."""
    for profile in profiles[:100]:
        liked_values = [utility(catalog_by_id[e.item_id], profile.theta) for e in profile.history if e.event_type == "liked"]
        disliked_values = [utility(catalog_by_id[e.item_id], profile.theta) for e in profile.history if e.event_type == "disliked"]
        if liked_values and disliked_values:
            assert max(disliked_values) < max(liked_values)


def test_history_pool_fractions_are_consistent():
    """Пулы лайков и дизлайков не пересекаются по замыслу (верх против низа)."""
    assert LIKED_POOL_FRACTION + DISLIKED_POOL_FRACTION < 1.0


def test_event_types_match_contract():
    """Типы событий совпадают с EventType контракта.

    Если контракт и симулятор разъедутся, mock-платформа из P3 не сможет отдать
    оракулу ту же историю, которую видит агент, и оценка перестанет относиться к
    продукту.
    """
    from pathlib import Path as _Path

    import yaml

    contract_path = _Path(__file__).resolve().parents[2] / "docs" / "contract" / "openapi.yaml"
    spec = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    contract_types = set(spec["components"]["schemas"]["EventType"]["enum"])
    ours = {"viewed", "liked", "disliked", "rated"}
    assert ours == contract_types


# ---------------------------------------------------------------------------
# Индексация
# ---------------------------------------------------------------------------


def test_profiles_by_id_roundtrip(profiles):
    index = profiles_by_id(profiles)
    assert len(index) == len(profiles)
    assert all(index[profile.user_id] is profile for profile in profiles)
    assert len({profile.user_id for profile in profiles}) == len(profiles), "user_id не уникальны"
