"""D3: генератор сценариев. Проверяем не «работает», а «не врёт».

Три класса проверок:

1. **Согласованность ground truth с репликой.** ``SpokenConstraints`` пишется
   генератором рядом с текстом, и расхождение между ними — самый опасный дефект
   всего набора: оракул тогда требует от агента то, чего пользователь не говорил.
   Такие ошибки не видно в метриках (все конфигурации проваливаются одинаково),
   поэтому их ловит отдельный тест на каждую реплику набора.

2. **Отрицательные контроли на РЕАЛЬНОМ наборе.** Потолок (``ceiling_ids``) обязан
   приниматься, случайная выдача — проваливаться. Это тест самого прибора
   (``docs/EVAL-PLAN.md`` §2, B0): если случайный ответ проходит, набор
   сценариев тривиален и лестница B0..L6 ничего не различает.

3. **Воспроизводимость и независимость сплитов.** Holdout обязан быть другим
   набором, а не перестановкой dev: иначе «не участвовал в подборе» — пустые
   слова.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
from evals.oracle import UNREACHABLE_THRESHOLD, SpokenConstraints, judge
from evals.scenarios import (
    ADVERSARIAL_KINDS,
    ANSWER_SIZE,
    BUILDERS,
    DEV_PROFILE_RANGE,
    HOLDOUT_PROFILE_RANGE,
    KIND_WEIGHTS,
    SCENARIO_KINDS,
    SPLIT_SEEDS,
    SPLIT_SIZE,
    Scenario,
    Turn,
    generate_scenarios,
    read_jsonl,
    scenarios_sha256,
    write_jsonl,
)

from recagent.catalog import CATALOG_SIZE, generate_catalog
from recagent.catalog.users import PROFILE_COUNT, Theta, generate_profiles
from recagent.models import Item

# ---------------------------------------------------------------------------
# Фикстуры. Каталог и профили генерируются один раз на модуль: полный набор
# стоит секунд, а тестов здесь много.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> list[Item]:
    return generate_catalog(SPLIT_SEEDS["dev"])


@pytest.fixture(scope="module")
def catalog_by_id(catalog: list[Item]) -> dict[str, Item]:
    return {item.id: item for item in catalog}


@pytest.fixture(scope="module")
def profiles(catalog: list[Item]) -> list:
    return generate_profiles(SPLIT_SEEDS["dev"], PROFILE_COUNT, catalog)


@pytest.fixture(scope="module")
def dev(catalog: list[Item], profiles: list) -> list[Scenario]:
    scenarios, _ = generate_scenarios("dev", catalog=catalog, profiles=profiles)
    return scenarios


@pytest.fixture(scope="module")
def holdout(catalog: list[Item], profiles: list) -> list[Scenario]:
    scenarios, _ = generate_scenarios("holdout", catalog=catalog, profiles=profiles)
    return scenarios


# ---------------------------------------------------------------------------
# Поверхностные формы: как значение слота должно выглядеть в реплике
# ---------------------------------------------------------------------------
# Таблицы намеренно написаны ЗДЕСЬ, а не взяты из scenarios.py: если бы тест
# читал генераторские словари, он проверял бы «шаблон подставлен в шаблон» и не
# поймал бы неверную форму в самом словаре.

#: Основа жанра: «драм» покрывает и «драма», и «драму», и «драмы».
GENRE_STEM: dict[str, str] = {
    "детектив": "детектив",
    "комедия": "комед",
    "драма": "драм",
    "фантастика": "фантаст",
    "приключения": "приключен",
    "машинное обучение": "обучен",
    "python": "python",
}
TONE_STEM: dict[str, str] = {"лёгкий": "лёгк", "нейтральный": "нейтральн", "мрачный": "мрачн"}
KIND_WORD: dict[str, str] = {"series": "сериал", "film": "фильм", "course": "курс"}
#: Формы уровня. «продвинутый» проверяется и на отсутствие новичковых формулировок:
#: иначе реплика «я уже не новичок» прошла бы как доказательство начального уровня.
LEVEL_SURFACE: dict[str, tuple[str, ...]] = {
    "начальный": ("новичк", "начальн", "с нуля", "начинающ"),
    "продвинутый": ("продвинут", "опытных", "не новичок"),
}
PRACTICAL_SURFACE: tuple[str, ...] = ("практик", "практич", "задани", "без воды", "не только теория")


def _lower(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _lower_forms(forms: tuple[str, ...]) -> tuple[str, ...]:
    """Те же основы, но в нормализации ``_lower``.

    Нормализовать ОБЕ стороны обязательно: ``_lower`` заменяет «ё» на «е», и без
    этого «тон лёгкий» в реплике не находился бы по основе «лёгк». Расхождение
    выглядело бы как «генератор не произнёсл тон», хотя реплика корректна.
    """
    return tuple(_lower(form) for form in forms)


def _assert_spoken_supported_by(spoken: SpokenConstraints, text: str, where: str) -> None:
    """Каждое ограничение ``spoken`` обязано быть произнесённым в ``text``.

    ``text`` — накопленный текст всех ходов до текущего включительно, потому что
    ``Turn.spoken`` накопительный: второй ход добавляет ограничение к первому, а не
    повторяет его. Проверять только реплику текущего хода означало бы требовать от
    пользователя произнести заново то, что он уже сказал.
    """
    if spoken.kind is not None:
        assert _lower(KIND_WORD[spoken.kind]) in text, f"{where}: формат не назван"
    if spoken.genre is not None:
        assert _lower(GENRE_STEM[spoken.genre]) in text, f"{where}: жанр не назван"
    for excluded in spoken.excluded_genres:
        assert _lower(GENRE_STEM[excluded]) in text, f"{where}: исключённый жанр не назван"
    if spoken.tone is not None:
        assert _lower(TONE_STEM[spoken.tone]) in text, f"{where}: тон не назван"
    for excluded_tone in spoken.excluded_tones:
        assert _lower(TONE_STEM[excluded_tone]) in text, f"{where}: отрицаемый тон не назван"
    if spoken.max_minutes is not None:
        assert "не дольше" in text or "не длиннее" in text, f"{where}: длительность не названа"
        assert str(spoken.max_minutes) in text or "час" in text, f"{where}: значение длительности не названо"
    if spoken.max_seasons is not None:
        assert "сезон" in text, f"{where}: сезоны не названы"
    if spoken.level is not None:
        assert any(form in text for form in _lower_forms(LEVEL_SURFACE[spoken.level])), f"{where}: уровень не назван"
        if spoken.level == "продвинутый":
            assert "для новичка" not in text and "для начинающих" not in text, f"{where}: противоречие уровня"
    if spoken.practical is not None:
        assert any(form in text for form in _lower_forms(PRACTICAL_SURFACE)), f"{where}: практика не названа"
    if spoken.named_title is not None:
        assert _lower(spoken.named_title) in text, f"{where}: название не названо"
    if spoken.seed_title is not None:
        assert _lower(spoken.seed_title) in text, f"{where}: опорное название не названо"


def _assert_utterance_is_well_formed(turn: Turn, scenario_id: str) -> None:
    """Реплика одного хода: подстановки выполнены, текст непустой.

    Значения слотов здесь НЕ проверяются — это делает
    ``_assert_spoken_supported_by`` по накопленному тексту, потому что ``spoken``
    накопительный и второй ход законно не повторяет первый.
    """
    where = f"{scenario_id} ход {turn.index}: {turn.utterance!r}"
    assert "{" not in turn.utterance and "}" not in turn.utterance, f"{where}: не подставлен шаблон"
    assert turn.utterance.strip(), f"{where}: пустая реплика"
    if turn.answer_if_clarified is not None:
        assert "{" not in turn.answer_if_clarified and turn.answer_if_clarified.strip(), f"{where}: ответ на уточнение испорчен"


# ---------------------------------------------------------------------------
# 1. Ground truth согласован с репликами
# ---------------------------------------------------------------------------


def test_every_spoken_constraint_is_actually_uttered(dev: list[Scenario], holdout: list[Scenario]) -> None:
    """Главный тест набора: оракул не требует того, чего пользователь не говорил.

    Проходит по ВСЕМ ходам обоих сплитов, а не по выборке: расхождение в одном
    шаблоне исказило бы метрики именно того типа сценария, где оно спрятано.
    """
    checked = 0
    for scenario in [*dev, *holdout]:
        said = ""
        for turn in scenario.turns:
            _assert_utterance_is_well_formed(turn, scenario.scenario_id)
            said += " " + _lower(turn.utterance)
            where = f"{scenario.scenario_id} ход {turn.index} (накопительно)"
            _assert_spoken_supported_by(turn.spoken, said, where)
            checked += 1
            # Ограничения, добавленные ОТВЕТОМ на уточнение, обязаны быть
            # произнесены в самом ответе, а не в исходной реплике.
            if turn.spoken_after_answer is not None and turn.answer_if_clarified is not None:
                _assert_spoken_supported_by(
                    turn.spoken_after_answer, said + " " + _lower(turn.answer_if_clarified), where + " после ответа"
                )
    assert checked == sum(len(scenario.turns) for scenario in [*dev, *holdout])
    assert checked > 250, f"проверено подозрительно мало ходов: {checked}"


def test_answer_if_clarified_is_a_real_reply(dev: list[Scenario]) -> None:
    """Ответ на уточнение — содержательная реплика, а не пустая строка.

    Симулятор произносит её дословно. Если бы генератор оставил её незаполненной,
    прогон после уточняющего вопроса агента отправил бы пустое сообщение и получил
    бы 422, а не провал по существу — то есть ошибка выглядела бы как сбой
    инфраструктуры, а не как сбой данных.
    """
    with_answer = [turn for scenario in dev for turn in scenario.turns if turn.answer_if_clarified is not None]
    assert len(with_answer) >= 20, f"слишком мало сценариев с ответом на уточнение: {len(with_answer)}"
    for turn in with_answer:
        assert turn.answer_if_clarified and turn.answer_if_clarified.strip()
        assert "{" not in turn.answer_if_clarified


def test_second_criteria_present_exactly_when_answer_adds_constraints(dev: list[Scenario]) -> None:
    """``criteria_after_clarify`` есть тогда и только тогда, когда задан ``spoken_after_answer``.

    Несоответствие означало бы, что прогон после уточнения мерит ответ по
    ограничениям, которых пользователь не называл, — в обе стороны это искажение.
    """
    for scenario in dev:
        final = scenario.turns[-1]
        has_after = scenario.criteria_after_clarify is not None
        needs_after = final.answer_if_clarified is not None and final.spoken_after_answer is not None
        assert has_after == needs_after, f"{scenario.scenario_id}: after={has_after}, нужно={needs_after}"
        if has_after:
            assert final.spoken_after_answer is not None
            after = scenario.criteria_after_clarify
            assert after is not None
            # Ответ снимает противоречие или добавляет слот. Оба направления
            # законны, поэтому сравнивать размеры множеств нельзя: у
            # ``contradiction`` базовое множество пусто (0), а после ответа —
            # сотни объектов. Проверяем содержательное: критерий построен по
            # расширенным ограничениям и НЕ пуст, иначе путь «агент уточнил»
            # непроходим и наказывал бы L4/L5 произвольно.
            assert after.acceptable_ids, f"{scenario.scenario_id}: критерий после уточнения пуст"
            assert after.feasible_count >= len(after.acceptable_ids)
            assert after.threshold <= 1.0, f"{scenario.scenario_id}: недостижимый порог после уточнения"


def test_no_scenario_claims_more_than_it_says_after_answer(dev: list[Scenario]) -> None:
    """``spoken_after_answer`` не теряет сказанное раньше.

    Ответ на уточнение ДОБАВЛЯЕТ информацию. Если бы он её заменял, накопленные
    ограничения исчезли бы, и агент получил бы незаслуженное снисхождение.
    """
    for scenario in dev:
        final = scenario.turns[-1]
        if final.spoken_after_answer is None:
            continue
        base, extended = final.spoken, final.spoken_after_answer
        for field in ("kind", "genre", "tone", "max_minutes", "max_seasons", "level", "practical"):
            value = getattr(base, field)
            if value is not None:
                assert getattr(extended, field) == value, f"{scenario.scenario_id}: потеряно {field}"


# ---------------------------------------------------------------------------
# 2. Структура набора
# ---------------------------------------------------------------------------


def test_kind_coverage_and_weights(dev: list[Scenario]) -> None:
    """Все 16 типов попали в набор, частоты близки к заявленным весам."""
    assert set(SCENARIO_KINDS) == set(KIND_WEIGHTS) == set(BUILDERS)
    counts = dict.fromkeys(SCENARIO_KINDS, 0)
    for scenario in dev:
        counts[scenario.kind] += 1
    missing = [kind for kind, count in counts.items() if count == 0]
    assert not missing, f"типы не представлены в наборе: {missing}"
    total = len(dev)
    for kind, weight in KIND_WEIGHTS.items():
        observed = counts[kind] / total
        # Отклонение до 5 п.п.: при n=200 и весе 0.02 это щедро, но не бессмысленно —
        # при большем разбросе набор перестал бы соответствовать зафиксированным весам.
        assert abs(observed - weight) < 0.05, f"{kind}: заявлено {weight}, фактически {observed:.3f}"


def test_split_sizes_match_plan(dev: list[Scenario], holdout: list[Scenario]) -> None:
    assert SPLIT_SIZE["dev"] == 200 and SPLIT_SIZE["holdout"] == 200
    assert len(dev) == SPLIT_SIZE["dev"]
    assert len(holdout) == SPLIT_SIZE["holdout"]
    assert SPLIT_SEEDS == {"dev": 42, "holdout": 1337}


def test_split_seeds_are_distinct() -> None:
    assert SPLIT_SEEDS["dev"] != SPLIT_SEEDS["holdout"], "сплиты обязаны различаться seed"


def test_default_holdout_uses_the_shared_catalog(catalog: list[Item], profiles: list) -> None:
    implicit, _ = generate_scenarios("holdout", 10)
    explicit, _ = generate_scenarios("holdout", 10, catalog=catalog, profiles=profiles)
    assert implicit == explicit


def test_dev_and_holdout_do_not_share_profiles(dev: list[Scenario], holdout: list[Scenario], profiles: list) -> None:
    """Holdout независим от dev: разные профили, а не только разные seed.

    Один и тот же theta в обоих наборах означал бы, что holdout — перестановка
    dev, и итоговые цифры на holdout были бы подогнаны подбором на dev.
    """
    dev_users = {scenario.user_id for scenario in dev}
    holdout_users = {scenario.user_id for scenario in holdout}
    assert not (dev_users & holdout_users), "сплиты делят пользователей"
    assert dev_users <= {profile.user_id for profile in profiles[DEV_PROFILE_RANGE[0] : DEV_PROFILE_RANGE[1]]}
    assert holdout_users <= {profile.user_id for profile in profiles[HOLDOUT_PROFILE_RANGE[0] : HOLDOUT_PROFILE_RANGE[1]]}
    assert DEV_PROFILE_RANGE[1] == HOLDOUT_PROFILE_RANGE[0], "диапазоны профилей обязаны быть смежными и непересекающимися"


def test_scenario_ids_are_unique(dev: list[Scenario], holdout: list[Scenario]) -> None:
    ids = [scenario.scenario_id for scenario in [*dev, *holdout]]
    assert len(ids) == len(set(ids)), "дубли scenario_id сломают paired-сравнение"


def test_turn_indices_are_sequential(dev: list[Scenario]) -> None:
    for scenario in dev:
        assert [turn.index for turn in scenario.turns] == list(range(len(scenario.turns)))


def test_multi_turn_share_is_substantial(dev: list[Scenario]) -> None:
    """Многоходовые сценарии — не украшение: без них не измерить накопление и снятие ограничений."""
    share = sum(1 for scenario in dev if len(scenario.turns) > 1) / len(dev)
    assert share >= 0.25, f"доля многоходовых слишком мала: {share:.2f}"


def test_adversarial_subset_is_present_and_documented(dev: list[Scenario]) -> None:
    """Adversarial-поднабор существует и каждый сценарий объясняет, что ловит."""
    adversarial = [scenario for scenario in dev if scenario.is_adversarial]
    assert len(adversarial) >= 10, f"adversarial-поднабор слишком мал: {len(adversarial)}"
    assert {scenario.kind for scenario in adversarial} <= ADVERSARIAL_KINDS
    for scenario in adversarial:
        assert scenario.adversarial_note, f"{scenario.kind}: нет пояснения, какой дефект ловим"
        assert len(scenario.adversarial_note) > 40


def test_expected_outcomes_cover_all_three_states(dev: list[Scenario]) -> None:
    """Набор обязан содержать и выдачу, и «не найдено», и уточнение.

    Без ``no_results`` невозможно измерить галлюцинации (агент выдаёт что попало
    при невыполнимом запросе), без ``clarify`` — политику уточнений.
    """
    states = {str(scenario.final_expected.value) for scenario in dev}
    assert states == {"recommend", "no_results", "clarify"}, f"не все исходы представлены: {states}"


def test_navigation_scenarios_exist(dev: list[Scenario]) -> None:
    """Отдельно проверяем навигацию: раньше её отбраковывало на 100%.

    У навигационного запроса ровно один выполнимый объект, и общие пороги
    вырожденности считали это дефектом. Тип исчезал из набора молча, а поиск по
    названию оставался неизмеренным.
    """
    navigation = [scenario for scenario in dev if scenario.kind == "navigation_title"]
    assert len(navigation) >= 5, f"навигационных сценариев слишком мало: {len(navigation)}"
    for scenario in navigation:
        assert scenario.feasible_count == 1
        assert scenario.acceptable_count == 1
        assert scenario.criteria.acceptable_ids


# ---------------------------------------------------------------------------
# 3. Критерии согласованы с каталогом
# ---------------------------------------------------------------------------


def test_criteria_counts_match_the_catalog(dev: list[Scenario], catalog: list[Item]) -> None:
    """Сохранённые счётчики — не декорация: их пересчёт по каталогу обязан совпасть."""
    assert len(catalog) == CATALOG_SIZE
    for scenario in dev[:50]:
        assert scenario.criteria.catalog_size == CATALOG_SIZE
        assert 0 <= scenario.acceptable_count <= scenario.feasible_count <= CATALOG_SIZE


def test_acceptable_set_never_covers_the_catalog(dev: list[Scenario], catalog: list[Item]) -> None:
    """Отрицательный контроль B0-вырождения: успех не может быть достижим чем угодно."""
    for scenario in dev:
        if scenario.final_expected.value != "recommend":
            continue
        share = scenario.acceptable_count / CATALOG_SIZE
        assert share < 0.5, f"{scenario.scenario_id}: приемлемо {share:.2f} каталога"


def test_ceiling_answer_always_passes(dev: list[Scenario], catalog_by_id: dict[str, Item]) -> None:
    """«Потолок» B3 обязан приниматься на каждом сценарии набора.

    Если оракул отвергает лучший по theta ответ, прибор сломан в другую сторону:
    он наказывает всё подряд, и лестница сравнений снова ничего не различает.
    """
    for scenario in dev:
        if scenario.final_expected.value != "recommend":
            continue
        criteria = scenario.criteria.to_criteria(
            theta=scenario.theta,
            spoken=scenario.final_spoken,
            expected=scenario.final_expected,
            user_id=scenario.user_id,
        )
        verdict = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=list(criteria.ceiling_ids))
        assert verdict.success, f"{scenario.scenario_id}: потолок отвергнут, failures={verdict.failures}"


def test_random_answer_fails_overwhelmingly(dev: list[Scenario], catalog_by_id: dict[str, Item]) -> None:
    """B0 sanity floor на реальном наборе: случайная выдача обязана проваливаться.

    Проверяется именно ДОЛЯ провалов, а не каждый сценарий: у случайной выдачи есть
    ненулевой шанс попасть в приемлемое множество, и требовать ста процентов
    означало бы тестировать удачу. Порог 0.9 взят с запасом относительно ожидаемых
    ~5% (``acceptable_share_median`` около 0.015 при выдаче из пяти объектов).
    """
    rng = random.Random(7)
    ids = list(catalog_by_id)
    failures = 0
    total = 0
    for scenario in dev:
        if scenario.final_expected.value != "recommend":
            continue
        criteria = scenario.criteria.to_criteria(
            theta=scenario.theta,
            spoken=scenario.final_spoken,
            expected=scenario.final_expected,
            user_id=scenario.user_id,
        )
        shown = rng.sample(ids, ANSWER_SIZE)
        verdict = judge(criteria=criteria, catalog_by_id=catalog_by_id, state="recommend", shown_ids=shown)
        failures += not verdict.success
        total += 1
    assert total > 100
    assert failures / total > 0.9, f"случайная выдача проходит в {1 - failures / total:.1%} случаев — набор тривиален"


def test_fabricated_ids_are_rejected_on_real_set(dev: list[Scenario], catalog_by_id: dict[str, Item]) -> None:
    """Галлюцинация (объект вне каталога) отвергается на каждом сценарии выдачи."""
    scenario = next(item for item in dev if item.final_expected.value == "recommend")
    criteria = scenario.criteria.to_criteria(
        theta=scenario.theta,
        spoken=scenario.final_spoken,
        expected=scenario.final_expected,
        user_id=scenario.user_id,
    )
    verdict = judge(
        criteria=criteria,
        catalog_by_id=catalog_by_id,
        state="recommend",
        shown_ids=[scenario.criteria.ceiling_ids[0], "it-99999999"],
    )
    assert not verdict.success
    assert "unknown_item_id" in verdict.failures


# ---------------------------------------------------------------------------
# 4. Воспроизводимость
# ---------------------------------------------------------------------------


def test_generation_is_deterministic(catalog: list[Item], profiles: list) -> None:
    first, _ = generate_scenarios("dev", 40, catalog=catalog, profiles=profiles)
    second, _ = generate_scenarios("dev", 40, catalog=catalog, profiles=profiles)
    assert [scenario.scenario_id for scenario in first] == [scenario.scenario_id for scenario in second]
    assert [scenario.model_dump(mode="json") for scenario in first] == [scenario.model_dump(mode="json") for scenario in second]


def test_sha256_is_stable_across_calls() -> None:
    """Отпечаток набора не зависит от процесса: это и есть проверяемая воспроизводимость."""
    assert scenarios_sha256("dev") == scenarios_sha256("dev")
    assert len(scenarios_sha256("dev")) == 64


def test_jsonl_roundtrip(dev: list[Scenario], tmp_path: Path) -> None:
    """Записанный набор читается обратно без потерь.

    Прогон читает сценарии из git, а не генерирует их заново, поэтому сериализация
    обязана сохранять и theta, и критерии, и ``criteria_after_clarify``.
    """
    # В выборку намеренно включён сценарий ``no_result``: у него выполнимое
    # множество пусто и порог недостижим. Именно на таком сценарии всплыл дефект
    # сериализации: ``math.inf`` pydantic записывает как ``null``, и чтение набора
    # из git падало с ValidationError. Порог обязан быть конечным числом.
    unreachable = next(scenario for scenario in dev if scenario.kind == "no_result")
    assert unreachable.criteria.threshold == UNREACHABLE_THRESHOLD
    sample = [*dev[:20], unreachable]
    path = write_jsonl(sample, tmp_path / "scenarios.jsonl")
    restored = read_jsonl(path)
    assert len(restored) == len(sample)
    assert [scenario.model_dump(mode="json") for scenario in restored] == [scenario.model_dump(mode="json") for scenario in sample]
    assert isinstance(restored[0].theta, Theta)
    assert isinstance(restored[0].turns[0].spoken, SpokenConstraints)


def test_all_thresholds_are_finite(dev: list[Scenario], holdout: list[Scenario]) -> None:
    """Ни один порог в наборе не ``inf``: иначе набор нельзя хранить в JSONL."""
    for scenario in [*dev, *holdout]:
        assert math.isfinite(scenario.criteria.threshold), f"{scenario.scenario_id}: нефинитный порог"
        if scenario.criteria_after_clarify is not None:
            assert math.isfinite(scenario.criteria_after_clarify.threshold)


def test_ceiling_is_inside_the_acceptable_set(dev: list[Scenario], catalog_by_id: dict[str, Item]) -> None:
    """Потолок согласован с критерием приёма ПО ПОСТРОЕНИЮ.

    ``ceiling_ids`` обязан быть подмножеством ``acceptable_ids``: потолок — это
    лучшие ПРИЕМЛЕМЫЕ ответы, а не топ-N по полезности. Различие возникает на ties
    у границы квантили, и тогда негативный контроль «B3 обязан проходить»
    проваливался на исправном оракуле.
    """
    for scenario in dev:
        ceiling = set(scenario.criteria.ceiling_ids)
        assert ceiling <= set(scenario.criteria.acceptable_ids), f"{scenario.scenario_id}: потолок вне приемлемого множества"
        assert all(item_id in catalog_by_id for item_id in ceiling)
        assert len(ceiling) == min(len(scenario.criteria.acceptable_ids), 5)


def test_rejection_is_reported_not_hidden(catalog: list[Item], profiles: list) -> None:
    """Отбраковка видна в сводке: сколько отброшено и почему.

    Скрывать её нельзя — иначе непонятно, на какой доле пространства сценариев
    измерены цифры, и рост отбраковки (признак сломавшейся калибровки) пройдёт
    незамеченным.
    """
    _, summary = generate_scenarios("dev", 40, catalog=catalog, profiles=profiles)
    for key in ("size", "attempted", "rejected_total", "rejected_by_kind", "rejected_by_reason", "by_kind", "multi_turn", "adversarial"):
        assert key in summary, f"в сводке нет {key}"
    assert summary["size"] == 40
    assert summary["attempted"] >= 40
    assert summary["rejected_total"] == summary["attempted"] - summary["size"]
    assert sum(summary["by_kind"].values()) == 40


def test_size_validation(catalog: list[Item], profiles: list) -> None:
    with pytest.raises(ValueError, match="не меньше 1"):
        generate_scenarios("dev", 0, catalog=catalog, profiles=profiles)
    with pytest.raises(ValueError, match="неизвестный сплит"):
        generate_scenarios("train", 5, catalog=catalog, profiles=profiles)


def test_small_generation_still_covers_kinds(catalog: list[Item], profiles: list) -> None:
    """Малый набор (используется тестами и CI-gate) обязан оставаться валидным."""
    scenarios, _ = generate_scenarios("dev", 30, catalog=catalog, profiles=profiles)
    assert len(scenarios) == 30
    for scenario in scenarios:
        assert scenario.turns and scenario.criteria.catalog_size == CATALOG_SIZE
        said = ""
        for turn in scenario.turns:
            _assert_utterance_is_well_formed(turn, scenario.scenario_id)
            said += " " + _lower(turn.utterance)
            _assert_spoken_supported_by(turn.spoken, said, scenario.scenario_id)
