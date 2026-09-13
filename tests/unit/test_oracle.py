"""Тесты независимого оракула.

Главный тест здесь — ``test_tautology_is_broken``. Он проверяет ровно то, ради
чего оракул переписан: соответствие сказанным ограничениям больше не достаточно
для успеха. Если этот тест однажды перестанет падать на «правильной» выдаче,
значит оракул снова проверяет то же, что и агент, и все метрики бессмысленны.

Остальные тесты — негативные контроли (``docs/DATA-PLAN.md`` §5.5): прибор обязан
ловить плохой ответ и обязан принимать хороший.
"""

from __future__ import annotations

import math
import random

import pytest
from evals.oracle import (
    ACCEPTABLE_QUANTILE,
    MAX_ACCEPTABLE_SHARE,
    MIN_FEASIBLE_FOR_ACCEPTABLE,
    ExpectedOutcome,
    SpokenConstraints,
    acceptable_set,
    build_criteria,
    feasible_set,
    is_degenerate,
    judge,
    satisfies_spoken,
    utility_threshold,
)

from recagent.catalog import generate_catalog
from recagent.catalog.users import Theta, UserProfile, generate_profiles, utility
from recagent.models import Item


@pytest.fixture(scope="module")
def catalog() -> list[Item]:
    return generate_catalog(42)


@pytest.fixture(scope="module")
def catalog_by_id(catalog) -> dict[str, Item]:
    return {item.id: item for item in catalog}


@pytest.fixture(scope="module")
def profiles(catalog) -> list[UserProfile]:
    # 40 профилей: по четыре на архетип. Достаточно, чтобы поймать зависимость
    # критерия от архетипа, и достаточно быстро, чтобы тест не стал медленным.
    return generate_profiles(42, 40, catalog)


@pytest.fixture
def theta(profiles) -> Theta:
    return profiles[0].theta


# ---------------------------------------------------------------------------
# Тавтология сломана
# ---------------------------------------------------------------------------


def test_tautology_is_broken(catalog, catalog_by_id, theta):
    """Выдача, формально удовлетворяющая ограничениям, обязана провалиться.

    Берём ВСЕ объекты, соответствующие сказанным ограничениям, и показываем их
    целиком. Прежний оракул принял бы такой ответ безоговорочно: он проверял
    только эти поля. Новый обязан отвергнуть, потому что большинство
    соответствующих объектов theta не подходит.
    """
    spoken = SpokenConstraints(kind="series", tone="лёгкий")
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=spoken)

    feasible = feasible_set(catalog, spoken)
    # Санитарная проверка самого теста: выполнимых объектов много, а приемлемых
    # существенно меньше. Иначе тест проходил бы из-за пустоты, а не из-за theta.
    assert len(feasible) > 100
    assert len(criteria.acceptable_ids) < len(feasible)

    verdict = judge(
        criteria=criteria,
        catalog_by_id=catalog_by_id,
        state="recommend",
        shown_ids=[item.id for item in feasible[:5]],
    )
    # По построению первые пять по порядку каталога почти наверняка не в top-25%.
    assert not verdict.success
    assert "unacceptable_item_shown" in verdict.failures
    # Но метрика не обязана быть нулевой: часть попаданий допустима.
    assert verdict.hit_rate is not None and verdict.hit_rate < 1.0


def test_acceptable_set_is_a_strict_subset_of_feasible(catalog, theta):
    """Приемлемое множество — подмножество выполнимого, и притом меньшее.

    Обратное означало бы, что theta либо не участвует в критерии (равенство),
    либо противоречит сказанным ограничениям (не подмножество).
    """
    spoken = SpokenConstraints(kind="film", genre="драма")
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=spoken)
    feasible_ids = {item.id for item in feasible_set(catalog, spoken)}
    assert criteria.acceptable_ids <= feasible_ids
    assert criteria.acceptable_ids < feasible_ids


# ---------------------------------------------------------------------------
# Потолок и пол: прибор обязан уметь сказать и «хорошо», и «плохо»
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(10))
def test_oracle_ceiling_always_passes(catalog, catalog_by_id, profiles, index):
    """B3 (oracle ceiling) проходит для любого архетипа.

    Если оракул отвергает и заведомо лучший ответ, прибор сломан в другую
    сторону: он не измеряет качество, а требует невозможного.
    """
    profile = profiles[index * 4]
    spoken = SpokenConstraints(kind="series", tone="лёгкий")
    criteria = build_criteria(catalog=catalog, theta=profile.theta, user_id=profile.user_id, spoken=spoken)
    verdict = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=criteria.ceiling_ids)
    assert verdict.success, verdict.failures
    assert verdict.hit_rate == 1.0


def test_random_answer_fails_everywhere(catalog, catalog_by_id, profiles):
    """B0 (sanity floor) обязан провалиться. Это тест самого прибора.

    Если случайная выдача проходит хотя бы у одного профиля из сорока, критерий
    слишком мягкий и success_rate можно получить, ничего не делая.
    """
    rng = random.Random(1337)
    series_ids = [item.id for item in catalog if item.kind == "series"]
    for profile in profiles:
        spoken = SpokenConstraints(kind="series", tone="лёгкий")
        criteria = build_criteria(catalog=catalog, theta=profile.theta, user_id=profile.user_id, spoken=spoken)
        shown = rng.sample(series_ids, 5)
        verdict = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=shown)
        assert not verdict.success, f"{profile.user_id}: случайная выдача прошла ({verdict.hits})"


def test_popularity_only_answer_fails_too(catalog, catalog_by_id, theta):
    """B0 в варианте «по популярности» тоже обязан провалиться.

    Популярность коррелирует с качеством каталога (это воспроизведено из
    MovieLens), поэтому такой бейзлайн опаснее случайного: если бы он проходил,
    любой продакшен-список платформы получал бы «успех» без диалога.
    """
    spoken = SpokenConstraints(kind="series")
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=spoken)
    by_popularity = sorted(
        (item for item in catalog if item.kind == "series" and item.popularity is not None),
        key=lambda item: (-item.popularity, item.id),
    )
    verdict = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=[item.id for item in by_popularity[:5]])
    assert not verdict.success


# ---------------------------------------------------------------------------
# Негативные контроли: подделка и вырождение
# ---------------------------------------------------------------------------


def test_fabricated_item_id_is_caught(catalog_by_id, theta):
    """Галлюцинация в строгом смысле: объект, которого нет в каталоге."""
    spoken = SpokenConstraints(kind="series")
    criteria = build_criteria(catalog=list(catalog_by_id.values()), theta=theta, user_id="u1", spoken=spoken)
    verdict = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=["it-does-not-exist"])
    assert not verdict.success
    assert "unknown_item_id" in verdict.failures
    assert verdict.mean_utility is None


def test_no_results_expected_but_items_shown(catalog, catalog_by_id, theta):
    """Невозможный запрос: выдача чего попало вместо честного «не найдено»."""
    # Одна минута — таких объектов в каталоге нет ни одного.
    spoken = SpokenConstraints(kind="series", max_minutes=1)
    assert not feasible_set(catalog, spoken)
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=spoken)
    assert criteria.expected is ExpectedOutcome.NO_RESULTS

    relaxed = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=criteria.ceiling_ids or ["it-00001"])
    assert not relaxed.success and "state_mismatch:recommend" in relaxed.failures

    honest = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="no_results", shown_ids=[])
    assert honest.success


def test_clarify_expected_but_answered(catalog, catalog_by_id, theta):
    """Уточнение ожидается — агент не имеет права отвечать вместо вопроса."""
    spoken = SpokenConstraints()
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=spoken, expected=ExpectedOutcome.CLARIFY)
    answered = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=list(criteria.acceptable_ids)[:3])
    assert not answered.success and "answered_instead_of_clarifying" in answered.failures
    assert judge(criteria=criteria, catalog_by_id=catalog_by_id, state="clarify", shown_ids=[]).success


def test_patience_is_enforced(catalog, catalog_by_id, theta):
    """Больше уточнений, чем выдержит профиль, — провал даже при верной выдаче."""
    spoken = SpokenConstraints(kind="series", tone="лёгкий")
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=spoken)
    within = judge(
        criteria=criteria,
        catalog_by_id=catalog_by_id,
        state="recommend",
        shown_ids=criteria.ceiling_ids,
        clarifications=criteria.max_clarifications,
    )
    assert within.success
    over = judge(
        criteria=criteria,
        catalog_by_id=catalog_by_id,
        state="recommend",
        shown_ids=criteria.ceiling_ids,
        clarifications=criteria.max_clarifications + 1,
    )
    assert not over.success and "patience_exceeded" in over.failures


def test_degenerate_scenarios_are_flagged(catalog, theta):
    """Вырожденный сценарий обязан быть помечен, а не молча принят.

    Три вырождения, которые тест обязан поймать: слишком мало выполнимых объектов,
    приемлемое множество на весь каталог и пустое приемлемое множество.
    """
    # Выполнимых объектов непусто, но меньше MIN_FEASIBLE_FOR_ACCEPTABLE: квантиль
    # на шести объектах решает всё, и порог приемлемости неустойчив. Запрос, при
    # котором таких объектов ровно шесть, существует в каталоге.
    tight_spoken = SpokenConstraints(kind="series", genre="фантастика", tone="мрачный", max_seasons=1)
    tight = build_criteria(catalog=catalog, theta=theta, user_id="tight", spoken=tight_spoken)
    assert 0 < tight.feasible_count < MIN_FEASIBLE_FOR_ACCEPTABLE, tight.feasible_count
    assert is_degenerate(tight)[0], (tight.feasible_count, tight.acceptable_share)
    assert any(problem.startswith("feasible_too_small") for problem in is_degenerate(tight)[1])

    # Пустое множество выполнимых: оракул сам выводит expected=NO_RESULTS, и такой
    # сценарий НЕ вырожден — он честный. Это отдельная ветка, её легко перепутать.
    impossible = build_criteria(catalog=catalog, theta=theta, user_id="none", spoken=SpokenConstraints(kind="series", max_minutes=1))
    assert impossible.feasible_count == 0
    assert impossible.expected is ExpectedOutcome.NO_RESULTS
    assert not is_degenerate(impossible)[0], is_degenerate(impossible)[1]

    # Порог, который не отбирает ничего: квантиль 0 принимает всё выполнимое.
    # Прямая проверка рабочей меры вырождения — доли от ВЫПОЛНИМОГО множества.
    # Доля от каталога такое вырождение не ловит в принципе: при квантили q она
    # ограничена сверху (1-q) * feasible/catalog, то есть 0.25 при q=0.75, и
    # проверка MAX_ACCEPTABLE_SHARE сработала бы только на более низкой квантили.
    flat = build_criteria(catalog=catalog, theta=theta, user_id="flat", spoken=SpokenConstraints(kind="series"), quantile=0.0)
    assert flat.acceptable_share_of_feasible == pytest.approx(1.0)
    assert flat.acceptable_ids == {item.id for item in feasible_set(catalog, SpokenConstraints(kind="series"))}
    assert any(problem.startswith("acceptable_covers_feasible") for problem in is_degenerate(flat)[1])

    # Широкий, но различающий сценарий НЕ вырожден: ограничений нет, приемлемое
    # множество — верхняя четверть каталога (750 из 3000). Случайная выдача из
    # пяти объектов проходит такой критерий с вероятностью 0.25^5, то есть B0
    # по-прежнему проваливается, и сценарий остаётся осмысленным.
    loose = build_criteria(catalog=catalog, theta=theta, user_id="loose", spoken=SpokenConstraints())
    assert loose.acceptable_share == pytest.approx(1 - ACCEPTABLE_QUANTILE, abs=0.01)
    assert loose.acceptable_share <= MAX_ACCEPTABLE_SHARE
    assert not is_degenerate(loose)[0], is_degenerate(loose)[1]

    # Нормальный сценарий не должен помечаться.
    normal = build_criteria(catalog=catalog, theta=theta, user_id="normal", spoken=SpokenConstraints(kind="series", tone="лёгкий"))
    assert not is_degenerate(normal)[0], is_degenerate(normal)[1]
    assert normal.acceptable_share <= MAX_ACCEPTABLE_SHARE


def test_criteria_are_immutable(catalog, theta):
    """Критерий нельзя поправить по ходу прогона — dataclass заморожен."""
    criteria = build_criteria(catalog=catalog, theta=theta, user_id="u1", spoken=SpokenConstraints(kind="series"))
    with pytest.raises(Exception):  # noqa: B017 - конкретный тип зависит от версии Python
        criteria.threshold = 0.0  # type: ignore[misc]
    with pytest.raises(Exception):  # noqa: B017
        criteria.acceptable_ids = frozenset()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Семантика null и расхождения с matches()
# ---------------------------------------------------------------------------


def test_unknown_seasons_do_not_satisfy_limit(catalog):
    """null = «нет данных», а не «совпало с ограничением» (контракт OpenAPI)."""
    unknown = next(item for item in catalog if item.kind == "series" and item.seasons is None)
    ok, reasons = satisfies_spoken(unknown, SpokenConstraints(kind="series", max_seasons=3))
    assert not ok and "seasons_unknown" in reasons

    known = next(item for item in catalog if item.kind == "series" and item.seasons is not None and item.seasons <= 3)
    assert satisfies_spoken(known, SpokenConstraints(kind="series", max_seasons=3))[0]


def test_season_limit_does_not_apply_to_films(catalog):
    """У фильма сезонов нет вообще — ограничение неприменимо, а не нарушено."""
    film = next(item for item in catalog if item.kind == "film")
    assert satisfies_spoken(film, SpokenConstraints(kind="film", max_seasons=1))[0]


def test_unknown_practical_does_not_satisfy_requirement(catalog):
    """«Нет данных» о практике не удовлетворяет явному требованию практики."""
    unknown = next(item for item in catalog if item.kind == "course" and item.practical is None)
    assert not satisfies_spoken(unknown, SpokenConstraints(kind="course", practical=True))[0]


def test_level_is_not_satisfied_by_films(catalog):
    """Сказанное вслух «начальный уровень» фильм удовлетворить не может."""
    film = next(item for item in catalog if item.kind == "film")
    ok, reasons = satisfies_spoken(film, SpokenConstraints(level="начальный"))
    assert not ok and "level" in reasons


def test_excluded_genre_is_rejected(catalog):
    spoken = SpokenConstraints(excluded_genres=["драма"])
    assert all(not satisfies_spoken(item, spoken)[0] for item in catalog if item.genre == "драма")
    assert all(satisfies_spoken(item, spoken)[0] for item in catalog if item.genre != "драма")


def test_reasons_are_diagnostic(catalog):
    """Причины провала перечислены все, а не только первая: провал диагностируется."""
    item = next(i for i in catalog if i.kind == "film" and i.genre == "драма" and i.tone == "мрачный")
    spoken = SpokenConstraints(kind="series", genre="комедия", tone="лёгкий", max_minutes=1)
    ok, reasons = satisfies_spoken(item, spoken)
    assert not ok
    assert set(reasons) == {"kind", "genre", "tone", "max_minutes"}


# ---------------------------------------------------------------------------
# Порог полезности
# ---------------------------------------------------------------------------


def test_threshold_is_a_quantile_of_the_feasible_subset(catalog, theta):
    """Порог считается по подмножеству, а не по всему каталогу.

    Это принципиально: глобальная константа сделала бы критерий недостижимым для
    профилей с узкими вкусами и тривиальным для всеядных.
    """
    spoken = SpokenConstraints(kind="course", genre="python")
    feasible = feasible_set(catalog, spoken)
    threshold = utility_threshold(feasible, theta)
    values = sorted(utility(item, theta) for item in feasible)
    expected_index = math.ceil(ACCEPTABLE_QUANTILE * len(values)) - 1
    assert threshold == values[expected_index]
    # Не меньше четверти выполнимых объектов попадают в приемлемое множество.
    acceptable = acceptable_set(catalog, theta, spoken)
    assert len(acceptable) / len(feasible) == pytest.approx(1 - ACCEPTABLE_QUANTILE, abs=0.02)


def test_threshold_of_empty_set_is_unreachable(theta):
    """Пустое множество даёт +inf: принять нечего, и это не «порог 0»."""
    assert utility_threshold([], theta) == math.inf
    assert acceptable_set([], theta, SpokenConstraints()) == frozenset()


def test_threshold_differs_between_profiles(catalog):
    """Порог зависит от профиля: один и тот же запрос разным людям подходит по-разному."""
    profiles = generate_profiles(42, 40, catalog)
    spoken = SpokenConstraints(kind="series", tone="лёгкий")
    feasible = feasible_set(catalog, spoken)
    thresholds = {round(utility_threshold(feasible, profile.theta), 4) for profile in profiles}
    assert len(thresholds) > 1


# ---------------------------------------------------------------------------
# Оракул не читает то, что извлёк агент
# ---------------------------------------------------------------------------


def test_oracle_never_receives_agent_query(catalog, theta):
    """``judge`` принимает только state и id: объекта ответа у него нет.

    Проверка на уровне сигнатуры. Если однажды сюда добавят параметр
    ``response`` или ``query``, циркулярность вернётся, и этот тест напомнит.
    """
    import inspect

    from evals.oracle import judge as judge_fn

    parameters = set(inspect.signature(judge_fn).parameters)
    assert parameters == {"criteria", "catalog_by_id", "state", "shown_ids", "clarifications", "require_all_acceptable"}
    assert "response" not in parameters and "query" not in parameters
