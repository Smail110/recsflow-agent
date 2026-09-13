"""D3: генератор диалоговых сценариев.

Что такое сценарий
------------------
Сценарий — это профиль (theta), последовательность реплик и ground truth к ним:
какие ограничения пользователь ДЕЙСТВИТЕЛЬНО назвал к каждому ходу и какой исход
ожидается. Всё это строится ДО запуска агента и не меняется в ходе диалога.

Разделение ответственности (``docs/DATA-PLAN.md`` §4, D3) принципиальное::

    код:  выбирает theta, ограничения, приемлемое множество, ожидаемый исход,
          решает, что отвечать на уточнение (правдиво по theta)
    LLM:  только поверхностная реализация реплики (перефразирование)

LLM не принимает решений о содержании, поэтому не может «подыграть» агенту.
Каноническая реплика хранится в сценарии, и она же является эталоном: если LLM
недоступен, прогон идёт на канонических репликах и остаётся полностью
детерминированным.

Почему ограничения берёт код, а не theta целиком
------------------------------------------------
Пользователь вслух называет ПОДМНОЖЕСТВО своих предпочтений. Оракул знает theta
целиком, агент — только то, что сказано (и то, что он сумел извлечь). Именно этот
зазор и измеряется: чтобы дать приемлемый ответ, агент должен либо угадать
недосказанное по истории, либо задать правильный уточняющий вопрос.

Если бы сценарий всегда озвучивал theta целиком, задача свелась бы к применению
фильтров, и success_rate снова стал бы тавтологией.

Отбраковка сценариев
--------------------
Сгенерированный сценарий проверяется оракулом и отбраковывается, если он
вырожден (``is_degenerate``): приемлемое множество пусто, покрывает весь каталог
или выполнимых объектов слишком мало для устойчивой квантили. Непроходимый
сценарий занижал бы метрики всех конфигураций одинаково, а тривиальный — завышал,
и в обоих случаях лестница B0..L6 переставала бы что-либо различать.

Сколько отброшено — печатается в сводке и пишется в файл. Скрывать отбраковку
нельзя: иначе непонятно, на какой доле пространства сценариев измерены цифры.

dev / holdout
-------------
``dev`` (seed=42) используется для подбора промптов, порогов gain и весов fusion.
``holdout`` (seed=1337) — только для финальных цифр, в подборе не участвует
(``docs/EVAL-PLAN.md`` §5). Наборы генерируются из РАЗНЫХ seed и разных
диапазонов профилей, поэтому holdout не является перестановкой dev.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from evals.oracle import (
    ExpectedOutcome,
    OracleCriteria,
    SpokenConstraints,
    build_criteria,
    feasible_set,
    is_degenerate,
)
from recagent.catalog import generate_catalog
from recagent.catalog.users import GENRES, PROFILE_COUNT, TONES, Theta, UserProfile, generate_profiles, utility
from recagent.models import Item, StrictModel

Split = Literal["dev", "holdout"]

#: dev seed=42, holdout seed=1337 (docs/EVAL-PLAN.md §5).
SPLIT_SEEDS: Final[dict[Split, int]] = {"dev": 42, "holdout": 1337}

#: Размер первичного набора. Статистическая точность оценивается по интервалам,
#: а не гарантируется одним лишь числом сценариев.
SPLIT_SIZE: Final[dict[Split, int]] = {"dev": 200, "holdout": 200}

#: holdout берёт профили из второй половины набора, dev — из первой. Разделение по
#: профилям, а не только по seed: иначе один и тот же theta попал бы в оба набора
#: с разными ограничениями, и holdout перестал бы быть независимым.
DEV_PROFILE_RANGE: Final[tuple[int, int]] = (0, PROFILE_COUNT // 2)
HOLDOUT_PROFILE_RANGE: Final[tuple[int, int]] = (PROFILE_COUNT // 2, PROFILE_COUNT)

#: Размер выдачи агента. Совпадает с тем, что возвращает ``Agent._recommend``.
ANSWER_SIZE: Final[int] = 5


# ---------------------------------------------------------------------------
# Типы сценариев
# ---------------------------------------------------------------------------


SCENARIO_KINDS: Final[tuple[str, ...]] = (
    "discovery_vague",
    "discovery_genre",
    "discovery_multi",
    "mood_evening",
    "similar_seed",
    "navigation_title",
    "course_level",
    "course_practical",
    "incremental",
    "release_constraint",
    "domain_switch",
    "exclusion_genre",
    "tone_negation",
    "no_result",
    "contradiction",
    "more_variants",
)
ScenarioKind = Literal[
    "discovery_vague",
    "discovery_genre",
    "discovery_multi",
    "mood_evening",
    "similar_seed",
    "navigation_title",
    "course_level",
    "course_practical",
    "incremental",
    "release_constraint",
    "domain_switch",
    "exclusion_genre",
    "tone_negation",
    "no_result",
    "contradiction",
    "more_variants",
]
"""Тип сценария. ``Literal`` как псевдоним, а не класс: pydantic валидирует
значение по набору допустимых строк, а кортеж ``SCENARIO_KINDS`` даёт тот же набор
для итерации (например, чтобы проверить, что все типы покрыты генератором)."""


#: Какие типы считаются adversarial: они намеренно ловят конкретные дефекты, а не
#: измеряют среднее качество. В отчёте они идут отдельной строкой, иначе пара
#: сложных случаев утащит общую цифру, и будет непонятно, что именно сломано.
ADVERSARIAL_KINDS: Final[frozenset[str]] = frozenset({"tone_negation", "contradiction", "no_result"})


# ---------------------------------------------------------------------------
# Модель сценария
# ---------------------------------------------------------------------------


class Turn(StrictModel):
    """Один ход пользователя.

    ``spoken`` — накопленные ограничения ПОСЛЕ этого хода: второй ход не заменяет
    первый, а добавляет к нему (кроме смены формата, где доменные слоты сбрасываются
    явно — это фиксирует сам генератор).
    """

    index: int = Field(ge=0)
    utterance: str = Field(min_length=1, max_length=500)
    spoken: SpokenConstraints
    expected: ExpectedOutcome
    #: Текст, который пользователь выдаст в ответ на уточняющий вопрос агента.
    #: None — уточнять нечего, ход самодостаточен. Заполняется кодом правдиво по
    #: theta, а не LLM: решение о содержании ответа принимает генератор.
    answer_if_clarified: str | None = Field(default=None, max_length=500)
    #: Накопленные ограничения ПОСЛЕ этого ответа. Отдельное поле, потому что
    #: ответ на уточнение может ДОБАВИТЬ информацию («сериал» в ответ на «какой
    #: формат?»), и тогда критерий приёма обязан учитывать сказанное. Один
    #: критерий на оба пути был бы неверен в обе стороны: построенный по базовым
    #: ограничениям он разрешал бы выдать фильм после слова «сериал», а построенный
    #: по расширенным — наказывал бы агента, который не спросил и выбрал сам.
    #: Оба критерия фиксируются ДО диалога (``Scenario.criteria`` и
    #: ``Scenario.criteria_after_clarify``), поэтому выбрать удобный по ходу
    #: прогона нельзя.
    spoken_after_answer: SpokenConstraints | None = None
    #: Подсказка генератора: какой тип уточнения ожидается. Нужна для разбора
    #: результатов, на решение оракула не влияет.
    clarify_hint: str | None = None


class Scenario(StrictModel):
    """Полный сценарий диалога D3."""

    scenario_id: str
    kind: ScenarioKind
    split: Split
    seed: int
    user_id: str
    theta: Theta
    turns: list[Turn] = Field(min_length=1)
    #: Критерий финального хода. Хранится готовым, а не пересчитывается в прогоне:
    #: критерий, который можно построить заново во время измерения, можно и
    #: незаметно построить по-другому.
    criteria: OracleCriteriaSpec
    #: Критерий для пути, на котором агент уточнил ФИНАЛЬНЫЙ ход и получил
    #: ``answer_if_clarified``. None — если ответ не добавляет ограничений.
    #: Промежуточные ходы в этом не нуждаются: у них есть следующий ход, и прогон
    #: после ответа продолжает диалог по нему.
    criteria_after_clarify: OracleCriteriaSpec | None = None
    #: Что известно о проходимости: для отчёта и для контроля отбраковки.
    feasible_count: int = Field(ge=0)
    acceptable_count: int = Field(ge=0)
    #: Комментарий для adversarial: какой дефект ловим. Попадает в отчёт.
    adversarial_note: str | None = None

    @property
    def is_adversarial(self) -> bool:
        return self.kind in ADVERSARIAL_KINDS

    @property
    def final_expected(self) -> ExpectedOutcome:
        return self.turns[-1].expected

    @property
    def final_spoken(self) -> SpokenConstraints:
        return self.turns[-1].spoken


class OracleCriteriaSpec(StrictModel):
    """Сериализуемая часть критерия.

    ``OracleCriteria`` — dataclass с frozenset и вложенной theta; для записи в
    JSONL нужна pydantic-модель. Это представление, а не замена: прогон
    восстанавливает из него настоящий ``OracleCriteria``.
    """

    acceptable_ids: list[str]
    ceiling_ids: list[str]
    threshold: float
    catalog_size: int = Field(gt=0)
    feasible_count: int = Field(ge=0)
    max_clarifications: int = Field(ge=0)

    def to_criteria(self, theta: Theta, spoken: SpokenConstraints, expected: ExpectedOutcome, user_id: str) -> OracleCriteria:
        return OracleCriteria(
            user_id=user_id,
            theta=theta,
            spoken=spoken,
            expected=expected,
            acceptable_ids=frozenset(self.acceptable_ids),
            threshold=self.threshold,
            catalog_size=self.catalog_size,
            max_clarifications=self.max_clarifications,
            feasible_count=self.feasible_count,
            ceiling_ids=tuple(self.ceiling_ids),
        )


# ---------------------------------------------------------------------------
# Шаблоны реплик
# ---------------------------------------------------------------------------
# Каждый шаблон — это НЕ одна строка, а несколько формулировок одного смысла.
# Одна формулировка на тип сценария означала бы, что regex-парсер запоминает
# точную строку, и мы измеряли бы совпадение со строкой, а не извлечение слотов.
# Выбор формулировки детерминирован по rng сценария.

_KIND_WORD: Final[dict[str, str]] = {"series": "сериал", "film": "фильм", "course": "курс"}
#: Винительный падеж: «подбери (что?) комедию». Для «машинное обучение»
#: винительный совпадает с именительным, а дательный («по машинному обучению»)
#: живёт отдельно в ``_TOPIC``: курс «по машинному обучению», но фильм —
#: «фантастику», и одна таблица на оба падежа неизбежно дала бы «курс в жанре
#: машинному обучению».
_GENRE_ACCUSATIVE: Final[dict[str, str]] = {
    "детектив": "детектив",
    "комедия": "комедию",
    "драма": "драму",
    "фантастика": "фантастику",
    "приключения": "приключения",
    "машинное обучение": "машинное обучение",
    "python": "python",
}
_TONE_WORD: Final[dict[str, str]] = {"лёгкий": "что-то лёгкое", "нейтральный": "нейтральное", "мрачный": "мрачное"}


def _minutes_phrase(minutes: int) -> str:
    """Человеческая длительность: «не дольше 45 минут», «не дольше часа»."""
    if minutes % 60 == 0 and minutes // 60 in (1, 2):
        return {60: "не дольше часа", 120: "не дольше двух часов"}[minutes]
    return f"не дольше {minutes} минут"


def _seasons_phrase(seasons: int) -> str:
    words = {1: "один сезон", 2: "два сезона", 3: "три сезона"}
    return words.get(seasons, f"не больше {seasons} сезонов")


DISCOVERY_VAGUE: Final[tuple[str, ...]] = (
    "Посоветуй что-нибудь",
    "Что бы мне посмотреть сегодня?",
    "Не знаю, чего хочу. Подбери на свой вкус",
    "Есть что-нибудь интересное?",
)
DISCOVERY_GENRE: Final[tuple[str, ...]] = (
    "Хочу {genre_acc}",
    "Подбери {genre_acc}",
    "Посоветуй {genre_acc} на вечер",
    "Ищу {genre_acc}, есть что-то достойное?",
)
DISCOVERY_KIND_GENRE: Final[tuple[str, ...]] = (
    "{kind_cap} в жанре {genre}",
    "Хочу {genre_acc} — {kind}",
    "Подбери {kind}, жанр — {genre}",
    "Ищу {kind}, жанр {genre}",
)
MULTI_CONSTRAINT: Final[tuple[str, ...]] = (
    "{kind_cap} в жанре {genre}, тон {tone}, {minutes}",
    "Ищу {genre_acc}: {kind}, тон {tone}, {minutes}",
    "Подбери {kind}, жанр {genre}, желательно {tone}, {minutes}",
)
SERIES_SEASONS: Final[tuple[str, ...]] = (
    "{kind_cap} в жанре {genre}, {seasons}, тон {tone}",
    "Ищу {kind} в жанре {genre_acc}, {seasons}, {tone}",
    "Хочу {genre_acc} — {kind} на {seasons}, тон {tone}",
)
#: Ответ на уточнение «какой формат?». Отдельный набор, потому что ответ на
#: вопрос о формате не должен повторять жанр: повтор привёл бы к тому, что
#: симулятор дважды произносит одно и то же, а оракул не смог бы отличить
#: «добавил информацию» от «повторил реплику».
KIND_ONLY: Final[tuple[str, ...]] = (
    "Давай {kind}",
    "Хочу {kind}",
    "{kind_cap}, пожалуйста",
    "Пусть будет {kind}",
)
MOOD_EVENING: Final[tuple[str, ...]] = (
    "Хочу {tone_word} на вечер",
    "День был тяжёлый, нужно {tone_word}",
    "Подбери что-то {tone_short} для отдыха",
    "Устал после собраний, нужно {tone_word}",
)
SIMILAR_SEED: Final[tuple[str, ...]] = (
    "{kind_cap}, похожий на «{title}»",
    "Хочу {kind} вроде «{title}»",
    "Посоветуй {kind} в духе «{title}»",
    "Нравится «{title}», найди похожий {kind}",
)
NAVIGATION_TITLE: Final[tuple[str, ...]] = (
    "Найди в каталоге «{title}»",
    "Покажи «{title}»",
    "Есть ли в каталоге «{title}»?",
)
COURSE_LEVEL: Final[tuple[str, ...]] = (
    "Курс по {topic} для новичка",
    "Нужен курс по {topic}, начальный уровень",
    "Хочу освоить {topic} с нуля — подбери курс",
    "Курс по {topic} для начинающих",
)
COURSE_ADVANCED: Final[tuple[str, ...]] = (
    "Курс по {topic} для опытных",
    "Нужен продвинутый курс по {topic}",
    "Ищу курс по {topic}, я уже не новичок",
)
COURSE_PRACTICAL: Final[tuple[str, ...]] = (
    "Курс по {topic} с практикой, без воды",
    "Нужен практический курс по {topic}",
    "Курс по {topic} — чтобы были задания, а не только теория",
)
#: Открытие по теме курса. Отдельно от ``DISCOVERY_KIND_GENRE``, потому что
#: «курс в жанре машинное обучение» по-русски не говорят.
COURSE_DISCOVERY: Final[tuple[str, ...]] = (
    "Курс по {topic}",
    "Нужен курс по {topic}",
    "Ищу курс по {topic}, есть что-то достойное?",
)
INCREMENT_MINUTES: Final[tuple[str, ...]] = (
    "Только {minutes}",
    "И {minutes}, пожалуйста",
    "Добавь ограничение: {minutes}",
    "Пусть будет {minutes}",
)
RELEASE_MINUTES: Final[tuple[str, ...]] = (
    "Без ограничений по длительности",
    "Сними ограничение на время",
    "Длительность не важна",
)
RELEASE_SEASONS: Final[tuple[str, ...]] = (
    "Без ограничений по сезонам",
    "Количество сезонов не важно",
)
DOMAIN_SWITCH: Final[tuple[str, ...]] = (
    "Лучше {kind} по {topic}",
    "Нет, давай {kind} по {topic}",
    "Передумал, ищу {kind} по {topic}",
)
EXCLUSION_GENRE: Final[tuple[str, ...]] = (
    "{kind_cap} в жанре {genre}, только без {genre_gen}",
    "Подбери {genre_acc} ({kind}), исключая {excluded_acc}",
    "Хочу {genre_acc} — {kind}, но без {genre_gen}",
)
TONE_NEGATION: Final[tuple[str, ...]] = (
    "{kind_cap} в жанре {genre}, не слишком мрачный",
    "Хочу {genre_acc} — {kind}, но не мрачное",
    "Подбери {kind} {genre_acc} без мрачности",
    "{kind_cap} в жанре {genre}, не хочу мрачного",
)
CONTRADICTION: Final[tuple[str, ...]] = (
    "Хочу {kind} в жанре {genre} и одновременно не {genre_acc}",
    "{kind_cap} в жанре {genre}, но {genre_acc} не предлагать",
)
NO_RESULT: Final[tuple[str, ...]] = (
    "{kind_cap} в жанре {genre}, {minutes_hard}",
    "Ищу {genre_acc} — {kind}, {minutes_hard}",
)
MORE: Final[tuple[str, ...]] = ("Ещё варианты", "Покажи другие", "Что-нибудь ещё")
RESET: Final[tuple[str, ...]] = ("Сброс", "Начать заново")

_GENRE_GENITIVE: Final[dict[str, str]] = {
    "детектив": "детектива",
    "комедия": "комедии",
    "драма": "драмы",
    "фантастика": "фантастики",
    "приключения": "приключений",
    "машинное обучение": "машинного обучения",
    "python": "python",
}
#: Средний род — после «что-то». Слово «спокойное» здесь намеренно НЕ
#: используется как синоним нейтрального тона: лексика слотов — это словарь
#: предметной области, и подмена синонимом измеряла бы не диалог, а дыру в
#: словаре парсера. Для словаря есть D4 (docs/DATA-PLAN.md §4).
_TONE_SHORT: Final[dict[str, str]] = {"лёгкий": "лёгкое", "нейтральный": "нейтральное", "мрачный": "мрачное"}
#: Мужской род — согласуется с «фильм», «сериал», «курс».
_TONE_MASC: Final[dict[str, str]] = {"лёгкий": "лёгкий", "нейтральный": "нейтральный", "мрачный": "мрачный"}
_TOPIC: Final[dict[str, str]] = {"машинное обучение": "машинному обучению", "python": "python"}


# ---------------------------------------------------------------------------
# Вспомогательный выбор значений
# ---------------------------------------------------------------------------


#: Жанры кино и сериалов. Отдельно от ``GENRES``, потому что у курсов жанр — это
#: тема, и «курс в жанре python, тон нейтральный» по-русски не говорят.
FILM_GENRES: Final[tuple[str, ...]] = ("детектив", "комедия", "драма", "фантастика", "приключения")


def _film_genre(rng: random.Random, theta: Theta, *, top_bias: float = 0.75) -> str:
    """Кино-жанр по весам theta. Используется там, где формат — фильм или сериал."""
    weights = [theta.genre_weights[genre] for genre in FILM_GENRES]
    if rng.random() < top_bias:
        return max(zip(weights, FILM_GENRES, strict=True))[1]
    return rng.choices(list(FILM_GENRES), weights=weights, k=1)[0]


def _pick_genre(rng: random.Random, theta: Theta, *, top_bias: float = 0.75) -> str:
    """Жанр, который пользователь назовёт вслух.

    С вероятностью ``top_bias`` берётся любимый жанр профиля, иначе — случайный
    по его же весам. Второй случай важен: если бы пользователь всегда просил своё
    любимое, задача свелась бы к угадыванию очевидного, и уточняющие вопросы не
    имели бы смысла.
    """
    genres = list(GENRES)
    weights = [theta.genre_weights[genre] for genre in genres]
    if rng.random() < top_bias:
        return max(zip(weights, genres, strict=True))[1]
    return rng.choices(genres, weights=weights, k=1)[0]


def _kind_for(rng: random.Random, genre: str) -> str:
    """Формат, совместимый с жанром.

    У курсов свои жанры, и наоборот: предложить «курс в жанре детектив» означало
    бы построить заведомо непроходимый сценарий и потом отбросить его, потеряв
    бюджет генерации.
    """
    if genre in ("машинное обучение", "python"):
        return "course"
    return rng.choices(("series", "film"), weights=(0.55, 0.45), k=1)[0]


def _minutes_for(rng: random.Random, theta: Theta, kind: str) -> int:
    """Ограничение длительности, согласованное с терпимостью профиля.

    Берётся вокруг ``duration_tol``, а не произвольно: если озвучить «не дольше 20
    минут» профилю, которому комфортно 600, выполнимых объектов может не
    оказаться, и сценарий уйдёт в отбраковку.
    """
    factor = rng.choice((0.8, 1.0, 1.2, 1.5))
    value = theta.duration_tol * factor
    if kind == "series":
        return max(20, min(90, round(value)))
    if kind == "film":
        return max(70, min(200, round(value)))
    return max(60, min(1500, round(value / 30) * 30))


def _seasons_for(rng: random.Random) -> int:
    return rng.choices((1, 2, 3), weights=(0.55, 0.30, 0.15), k=1)[0]


def _title_for(rng: random.Random, catalog: Sequence[Item], kind: str, theta: Theta) -> Item:
    """Название для «похожего на X»: берём то, что профилю нравится.

    Объект выбирается из верхней части распределения полезности, иначе сценарий
    «похожее на X» просил бы похожее на то, что пользователь терпеть не может, и
    приемлемое множество оказалось бы пусто.
    """
    candidates = [item for item in catalog if item.kind == kind]
    scored = sorted(candidates, key=lambda item: -utility(item, theta))
    top = scored[: max(1, len(scored) // 4)]
    return rng.choice(top)


def _fill(template: str, **values: str | int) -> str:
    """Подстановка с автокапитализацией ``{kind_cap}``."""
    prepared = {key: str(value) for key, value in values.items()}
    if "kind" in prepared:
        prepared["kind_cap"] = prepared["kind"].capitalize()
    text = template.format(**prepared)
    return text[0].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# Конструкторы сценариев. Каждый возвращает список ходов, а критерий финального
# хода строит вызывающий: иначе критерий зависел бы от того, кто его построил,
# и независимость оракула перестала бы быть проверяемой.
# ---------------------------------------------------------------------------


def _kind_answer(rng: random.Random, kind: str) -> str:
    """Ответ на уточняющий вопрос о формате."""
    return _fill(rng.choice(KIND_ONLY), kind=_KIND_WORD[kind])


def _discovery_phrase(rng: random.Random, genre: str) -> str:
    """Запрос по жанру без формата.

    У курсов жанр — это тема, и «подбери python на вечер» звучит как запрос кино.
    Поэтому формулировка выбирается по домену жанра, а не по одному набору
    шаблонов: иначе половина реплик была бы по-русски некорректна, и мы
    измеряли бы устойчивость парсера к бессмыслице вместо извлечения слотов.
    """
    if genre in _TOPIC:
        return _fill(rng.choice(COURSE_DISCOVERY), topic=_TOPIC[genre])
    return _fill(rng.choice(DISCOVERY_GENRE), genre=genre, genre_acc=_GENRE_ACCUSATIVE[genre])


def _kind_genre_phrase(rng: random.Random, kind: str, genre: str) -> str:
    """Запрос с названными форматом И жанром. Тот же доменный выбор формулировки."""
    if kind == "course":
        return _fill(rng.choice(COURSE_DISCOVERY), topic=_TOPIC[genre])
    return _fill(rng.choice(DISCOVERY_KIND_GENRE), kind=_KIND_WORD[kind], genre=genre, genre_acc=_GENRE_ACCUSATIVE[genre])


def _tone_of(rng: random.Random, theta: Theta) -> str:
    """Тон, который пользователь озвучит.

    Берётся из предпочтения theta, а не случайно: если озвучить противоположный,
    приемлемое множество окажется пустым по построению, и сценарий уйдёт в
    отбраковку. Случайный тон допустим только там, где он намеренно создаёт
    конфликт (adversarial).
    """
    if theta.tone_pref == "indifferent":
        return rng.choice(TONES)
    return theta.tone_pref


def _build_discovery_vague(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Запрос без ограничений: агент обязан уточнить, а не выдать что попало.

    Первый ход намеренно пустой, второй даёт жанр по theta. Это базовая проверка
    того, что уточняющий вопрос вообще задаётся и что ответ на него правдив.
    """
    genre = _pick_genre(rng, theta)
    kind = _kind_for(rng, genre)
    tone = _tone_of(rng, theta)
    answer = _discovery_phrase(rng, genre)
    kind_answer = _kind_answer(rng, kind)
    return (
        [
            Turn(
                index=0,
                utterance=rng.choice(DISCOVERY_VAGUE),
                spoken=SpokenConstraints(),
                expected=ExpectedOutcome.CLARIFY,
                answer_if_clarified=answer,
                clarify_hint="kind_or_genre",
            ),
            Turn(
                index=1,
                utterance=answer,
                # В реплике назван ТОЛЬКО жанр: формат пользователь ещё не
                # произнёс. Записать сюда kind означало бы приписать сказанное,
                # чего не было, — и оракул начал бы требовать от агента угадать
                # формат по молчанию, что занижало бы метрики произвольно.
                spoken=SpokenConstraints(genre=genre),
                expected=ExpectedOutcome.RECOMMEND,
                answer_if_clarified=kind_answer,
                spoken_after_answer=SpokenConstraints(kind=kind, genre=genre),
                clarify_hint="kind",
            ),
        ],
        {"kind_hint": kind, "tone_hint": tone, "profile": profile},
    )


def _build_discovery_genre(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Один ход, жанр назван, формат не назван.

    Формат не озвучивается намеренно: если бы мы всегда называли и формат, и жанр,
    и тон, агенту не оставалось бы ничего угадывать, и адаптивная политика
    уточнений (L5) не имела бы пространства для выигрыша над фиксированной (L4).
    """
    genre = _pick_genre(rng, theta)
    kind = _kind_for(rng, genre)
    return (
        [
            Turn(
                index=0,
                utterance=_discovery_phrase(rng, genre),
                spoken=SpokenConstraints(genre=genre),
                expected=ExpectedOutcome.RECOMMEND,
                # Агент вправе спросить про формат: реплика его не называет. Ответ
                # задан кодом по kind_hint профиля, а не придумывается в прогоне.
                answer_if_clarified=_kind_answer(rng, kind),
                spoken_after_answer=SpokenConstraints(kind=kind, genre=genre),
                clarify_hint="kind",
            )
        ],
        {"kind_hint": kind, "profile": profile},
    )


def _build_discovery_multi(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Формат + жанр + тон + длительность в одном ходе.

    Жанр — только кино: в реплике произносится тон, а «курс, тон лёгкий»
    по-русски не говорят. Покрытие курсов с длительностью даёт ``incremental``.
    """
    genre = _film_genre(rng, theta)
    kind = rng.choice(("series", "film"))
    tone = _tone_of(rng, theta)
    minutes = _minutes_for(rng, theta, kind)
    utterance = _fill(
        rng.choice(MULTI_CONSTRAINT),
        kind=_KIND_WORD[kind],
        genre=genre,
        genre_acc=_GENRE_ACCUSATIVE[genre],
        tone=_TONE_MASC[tone],
        minutes=_minutes_phrase(minutes),
    )
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                spoken=SpokenConstraints(kind=kind, genre=genre, tone=tone, max_minutes=minutes),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {"profile": profile},
    )


def _build_seasons(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Сериал с ограничением на число сезонов."""
    genre = _pick_genre(rng, theta, top_bias=0.9)
    tone = _tone_of(rng, theta)
    seasons = _seasons_for(rng)
    utterance = _fill(
        rng.choice(SERIES_SEASONS),
        kind=_KIND_WORD["series"],
        genre=genre,
        genre_acc=_GENRE_ACCUSATIVE[genre],
        seasons=_seasons_phrase(seasons),
        tone=_TONE_MASC[tone],
    )
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                spoken=SpokenConstraints(kind="series", genre=genre, tone=tone, max_seasons=seasons),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {"profile": profile},
    )


def _build_mood(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Настроение без жанра: «что-то лёгкое на вечер».

    Проверяет, что агент понимает настроение как тон, а не как жанр. Прежний
    парсер имел на это отдельное правило («с юмором» в контексте детектива), и
    сценарий нужен, чтобы правило не деградировало молча.
    """
    tone = _tone_of(rng, theta)
    kind = rng.choice(("series", "film"))
    utterance = _fill(rng.choice(MOOD_EVENING), tone_word=_TONE_WORD[tone], tone_short=_TONE_SHORT[tone])
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                spoken=SpokenConstraints(tone=tone),
                expected=ExpectedOutcome.RECOMMEND,
                answer_if_clarified=_kind_answer(rng, kind),
                spoken_after_answer=SpokenConstraints(kind=kind, tone=tone),
                clarify_hint="kind",
            )
        ],
        {"profile": profile, "tone_hint": tone, "kind_hint": kind},
    )


def _build_similar(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """«Похожее на X». Опорный объект не является приемлемым ответом."""
    genre = _pick_genre(rng, theta, top_bias=0.9)
    kind = _kind_for(rng, genre)
    seed = _title_for(rng, catalog, kind, theta)
    utterance = _fill(rng.choice(SIMILAR_SEED), kind=_KIND_WORD[kind], title=seed.title)
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                spoken=SpokenConstraints(kind=seed.kind, seed_title=seed.title),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {"profile": profile, "seed_id": seed.id},
    )


def _build_navigation(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """«Найди в каталоге X». Приемлем ровно названный объект."""
    genre = _pick_genre(rng, theta, top_bias=0.9)
    kind = _kind_for(rng, genre)
    target = _title_for(rng, catalog, kind, theta)
    utterance = _fill(rng.choice(NAVIGATION_TITLE), title=target.title)
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                spoken=SpokenConstraints(named_title=target.title),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {"profile": profile, "target_id": target.id},
    )


def _build_course(
    rng: random.Random,
    theta: Theta,
    catalog: Sequence[Item],
    profile: UserProfile,
    *,
    force_practical: bool | None = None,
) -> tuple[list[Turn], dict]:
    """Курс с уровнем и практикой. Два подвида: начальный и продвинутый.

    ``force_practical`` задаётся типом сценария: ``course_practical`` обязан
    порождать реплику про практику, ``course_level`` — про уровень. Если выбор
    делать случайным внутри, метка типа перестанет соответствовать содержанию и
    разрез «какие сценарии провалены» станет бессмысленным.
    """
    topic_genre = rng.choice(("машинное обучение", "python"))
    advanced = rng.random() < 0.35
    level = "продвинутый" if advanced else "начальный"
    practical = force_practical if force_practical is not None else rng.random() < 0.6
    templates = COURSE_ADVANCED if advanced else COURSE_LEVEL
    utterance = _fill(rng.choice(templates), topic=_TOPIC[topic_genre])
    if practical:
        utterance = _fill(rng.choice(COURSE_PRACTICAL), topic=_TOPIC[topic_genre])
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                # Уровень записывается в spoken только если он ПРОИЗНЕСЁН: при
                # практическом шаблоне реплика про уровень молчит, и требовать его
                # от агента было бы приписыванием.
                spoken=SpokenConstraints(
                    kind="course", genre=topic_genre, level=None if practical else level, practical=True if practical else None
                ),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {"profile": profile, "level": level, "practical": practical},
    )


def _build_incremental(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Ограничение добавляется вторым ходом, первое сохраняется.

    Проверяет накопление состояния сессии. Если агент забудет первый ход, он
    выдаст объект, нарушающий озвученное ранее ограничение, и оракул это поймает
    именно как невыполненное spoken-ограничение.
    """
    genre = _pick_genre(rng, theta)
    kind = _kind_for(rng, genre)
    minutes = _minutes_for(rng, theta, kind)
    first = _kind_genre_phrase(rng, kind, genre)
    second = _fill(rng.choice(INCREMENT_MINUTES), minutes=_minutes_phrase(minutes))
    return (
        [
            Turn(index=0, utterance=first, spoken=SpokenConstraints(kind=kind, genre=genre), expected=ExpectedOutcome.RECOMMEND),
            Turn(
                index=1,
                utterance=second,
                spoken=SpokenConstraints(kind=kind, genre=genre, max_minutes=minutes),
                expected=ExpectedOutcome.RECOMMEND,
            ),
        ],
        {"profile": profile},
    )


def _build_release(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Ограничение снимается вторым ходом.

    Обратная сторона накопления: агент обязан уметь НЕ только сохранять, но и
    отпускать. Снятие длительности проверяется отдельно от сезонов, потому что в
    прежнем парсере на это были разные regex, и один мог деградировать без другого.
    """
    # Формат — только фильм или сериал: снятие ограничения на сезоны у курса
    # бессмысленно, а «курс в жанре X, тон Y» некорректно по-русски.
    genre = _film_genre(rng, theta)
    kind = rng.choice(("series", "film"))
    tone = _tone_of(rng, theta)
    if kind == "series":
        seasons = _seasons_for(rng)
        first = _fill(
            rng.choice(SERIES_SEASONS),
            kind=_KIND_WORD[kind],
            genre=genre,
            genre_acc=_GENRE_ACCUSATIVE[genre],
            seasons=_seasons_phrase(seasons),
            tone=_TONE_MASC[tone],
        )
        release = rng.choice(RELEASE_SEASONS)
        spoken_first = SpokenConstraints(kind=kind, genre=genre, tone=tone, max_seasons=seasons)
        spoken_final = SpokenConstraints(kind=kind, genre=genre, tone=tone)
    else:
        minutes = _minutes_for(rng, theta, kind)
        first = _fill(
            rng.choice(MULTI_CONSTRAINT),
            kind=_KIND_WORD[kind],
            genre=genre,
            genre_acc=_GENRE_ACCUSATIVE[genre],
            tone=_TONE_MASC[tone],
            minutes=_minutes_phrase(minutes),
        )
        release = rng.choice(RELEASE_MINUTES)
        spoken_first = SpokenConstraints(kind=kind, genre=genre, tone=tone, max_minutes=minutes)
        spoken_final = SpokenConstraints(kind=kind, genre=genre, tone=tone)
    return (
        [
            Turn(index=0, utterance=first, spoken=spoken_first, expected=ExpectedOutcome.RECOMMEND),
            Turn(index=1, utterance=release, spoken=spoken_final, expected=ExpectedOutcome.RECOMMEND),
        ],
        {"profile": profile},
    )


def _build_domain_switch(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Смена формата с переносом доменных слотов.

    Сложнейший случай для парсинга: «лёгкий детективный сериал» -> «курс python
    для новичка». Тон и жанр сериала обязаны быть сброшены, иначе выдача будет
    пустой или, что хуже, фильтрованной по бессмысленному остатку.
    """
    genre = _film_genre(rng, theta, top_bias=0.9)
    kind = rng.choice(("series", "film"))
    tone = _tone_of(rng, theta)
    topic_genre = rng.choice(("машинное обучение", "python"))
    minutes = _minutes_for(rng, theta, kind)
    first = _fill(
        rng.choice(MULTI_CONSTRAINT),
        kind=_KIND_WORD[kind],
        genre=genre,
        genre_acc=_GENRE_ACCUSATIVE[genre],
        tone=_TONE_MASC[tone],
        minutes=_minutes_phrase(minutes),
    )
    switch = _fill(rng.choice(DOMAIN_SWITCH), kind=_KIND_WORD["course"], topic=_TOPIC[topic_genre])
    answer = _fill(rng.choice(COURSE_LEVEL), topic=_TOPIC[topic_genre])
    return (
        [
            Turn(
                index=0,
                utterance=first,
                spoken=SpokenConstraints(kind=kind, genre=genre, tone=tone, max_minutes=minutes),
                expected=ExpectedOutcome.RECOMMEND,
            ),
            Turn(
                index=1,
                utterance=switch,
                spoken=SpokenConstraints(kind="course", genre=topic_genre),
                expected=ExpectedOutcome.RECOMMEND,
                answer_if_clarified=answer,
                # Ответ на уточнение добавляет уровень — значит критерий приёма
                # после уточнения строже, и это зафиксировано заранее.
                spoken_after_answer=SpokenConstraints(kind="course", genre=topic_genre, level="начальный"),
                clarify_hint="level",
            ),
        ],
        {"profile": profile},
    )


def _build_exclusion(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """Исключение жанра: «что-нибудь, только не драму».

    Исключаемый жанр выбирается НЕ самым любимым, иначе сценарий стал бы
    противоречием. Берём жанр с наименьшим весом среди кино-жанров.
    """
    excluded = min(FILM_GENRES, key=lambda genre: theta.genre_weights[genre])
    genre = rng.choice([candidate for candidate in FILM_GENRES if candidate != excluded])
    kind = rng.choice(("series", "film"))
    utterance = _fill(
        rng.choice(EXCLUSION_GENRE),
        kind=_KIND_WORD[kind],
        genre=genre,
        genre_acc=_GENRE_ACCUSATIVE[genre],
        excluded_acc=_GENRE_ACCUSATIVE[excluded],
        genre_gen=_GENRE_GENITIVE[excluded],
    )
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                # Положительный жанр ОБЯЗАН быть в spoken: он назван вслух. Если его
                # не записать, оракул станет мягче сказанного и пропустит выдачу,
                # которая игнорирует половину реплики.
                spoken=SpokenConstraints(kind=kind, genre=genre, excluded_genres=[excluded]),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {"profile": profile, "excluded": excluded},
    )


def _build_tone_negation(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """ADVERSARIAL: составное отрицание тона — «не слишком мрачный».

    Ловит конкретный дефект, зафиксированный в прежнем отчёте: «не слишком
    мрачный» парсился неверно. Земля здесь в том, что «не мрачное» допускает и
    нейтральное, а агент сводит его к «лёгкому» и теряет нейтральные объекты.
    Оракул это покажет как невыполнение spoken, а не как ошибку theta.
    """
    genre = _film_genre(rng, theta, top_bias=0.9)
    kind = rng.choice(("series", "film"))
    utterance = _fill(rng.choice(TONE_NEGATION), kind=_KIND_WORD[kind], genre=genre, genre_acc=_GENRE_ACCUSATIVE[genre])
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                spoken=SpokenConstraints(kind=kind, genre=genre, excluded_tones=["мрачный"]),
                expected=ExpectedOutcome.RECOMMEND,
            )
        ],
        {
            "profile": profile,
            "adversarial_note": (
                "Составное отрицание тона. «Не слишком мрачный» допускает нейтральное; "
                "сведение к «лёгкому» теряет recall, и оракул показывает это как "
                "невыполнение сказанного, а не как ошибку предпочтений."
            ),
        },
    )


def _build_contradiction(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """ADVERSARIAL: противоречивый запрос. Ожидается уточнение, а не выдача.

    Противоречие формулируется как «хочу X и не X одновременно». Правильное
    поведение — попросить уточнить, потому что выдача в любом случае нарушит
    одну из двух частей. Если агент всё же выдаст, оракул пометит провал по
    spoken-ограничению, и это видно в разборе причин.
    """
    genre = _film_genre(rng, theta, top_bias=0.9)
    kind = rng.choice(("series", "film"))
    utterance = _fill(rng.choice(CONTRADICTION), kind=_KIND_WORD[kind], genre=genre, genre_acc=_GENRE_ACCUSATIVE[genre])
    answer = _fill(rng.choice(DISCOVERY_GENRE), genre=genre, genre_acc=_GENRE_ACCUSATIVE[genre])
    return (
        [
            Turn(
                index=0,
                utterance=utterance,
                # Противоречие записывается как оба ограничения сразу: жанр выбран
                # И исключён. Такое множество невыполнимо, поэтому expected=CLARIFY.
                spoken=SpokenConstraints(kind=kind, genre=genre, excluded_genres=[genre]),
                expected=ExpectedOutcome.CLARIFY,
                answer_if_clarified=answer,
                # Пользователь снимает противоречие: жанр остаётся, исключение
                # уходит. Это ground truth ответа, а не догадка оракула.
                spoken_after_answer=SpokenConstraints(kind=kind, genre=genre),
                clarify_hint="contradiction",
            )
        ],
        {
            "profile": profile,
            "adversarial_note": (
                "Противоречивый запрос: жанр одновременно выбран и исключён. "
                "Правильное поведение — уточнить; выдача нарушит одну из частей."
            ),
        },
    )


def _build_no_result(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """ADVERSARIAL: заведомо невыполнимый запрос.

    Правильный ответ — честное «не найдено» с сохранёнными ограничениями.
    Ослабить ограничение и выдать что попало хуже, чем не выдать ничего: это
    ровно тот сбой, который в реальном продукте выглядит как галлюцинация.
    """
    genre = rng.choice(list(FILM_GENRES))
    kind = rng.choice(("series", "film"))
    # Одна минута: таких объектов нет ни одного, и это проверяется до возврата.
    utterance = _fill(
        rng.choice(NO_RESULT), kind=_KIND_WORD[kind], genre=genre, genre_acc=_GENRE_ACCUSATIVE[genre], minutes_hard="не дольше 1 минуты"
    )
    spoken = SpokenConstraints(kind=kind, genre=genre, max_minutes=1)
    assert not feasible_set(catalog, spoken), "сценарий no_result перестал быть невыполнимым"
    return (
        [Turn(index=0, utterance=utterance, spoken=spoken, expected=ExpectedOutcome.NO_RESULTS)],
        {
            "profile": profile,
            "adversarial_note": (
                "Невыполнимый запрос: объектов короче минуты в каталоге нет. "
                "Честный ответ — «не найдено» с сохранёнными ограничениями; "
                "ослабление ограничения и выдача чего попало считается провалом."
            ),
        },
    )


def _build_more(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
    """«Ещё»: исключение уже показанного.

    Финальный spoken тот же, что и на первом ходе: «ещё» не добавляет
    ограничений, оно просит другие объекты. Исключение показанного проверяется
    прогоном, а не критерием: оракул смотрит на id, и повтор сам по себе не
    нарушение spoken. Поэтому отдельное поле expected держит RECOMMEND, а
    проверка неповторения — задача прогона (evals/run.py).
    """
    genre = _pick_genre(rng, theta)
    kind = _kind_for(rng, genre)
    first = _kind_genre_phrase(rng, kind, genre)
    return (
        [
            Turn(index=0, utterance=first, spoken=SpokenConstraints(kind=kind, genre=genre), expected=ExpectedOutcome.RECOMMEND),
            Turn(index=1, utterance=rng.choice(MORE), spoken=SpokenConstraints(kind=kind, genre=genre), expected=ExpectedOutcome.RECOMMEND),
        ],
        {"profile": profile},
    )


def _course_builder(force_practical: bool):
    """Фабрика конструктора курса с фиксированным слотом.

    Нужна, чтобы все конструкторы в ``BUILDERS`` имели ОДИНАКОВУЮ сигнатуру
    ``(rng, theta, catalog, profile)``: вызов идёт по имени типа из таблицы, и
    различающиеся сигнатуры заставили бы ветвиться на месте вызова.
    """

    def build(rng: random.Random, theta: Theta, catalog: Sequence[Item], profile: UserProfile) -> tuple[list[Turn], dict]:
        return _build_course(rng, theta, catalog, profile, force_practical=force_practical)

    return build


BUILDERS: Final[dict[str, object]] = {
    "discovery_vague": _build_discovery_vague,
    "discovery_genre": _build_discovery_genre,
    "discovery_multi": _build_discovery_multi,
    "mood_evening": _build_mood,
    "similar_seed": _build_similar,
    "navigation_title": _build_navigation,
    "course_level": _course_builder(False),
    "course_practical": _course_builder(True),
    "incremental": _build_incremental,
    "release_constraint": _build_release,
    "domain_switch": _build_domain_switch,
    "exclusion_genre": _build_exclusion,
    "tone_negation": _build_tone_negation,
    "no_result": _build_no_result,
    "contradiction": _build_contradiction,
    "more_variants": _build_more,
}
"""Каждый тип сценария обязан иметь конструктор. Проверка покрытия — в тестах.

``similar_seed`` и ``navigation_title`` ведут к разным конструкторам, но оба
требуют названного объекта; ``course_level`` и ``course_practical`` разделяют один
конструктор, потому что различаются только озвученным слотом, а не структурой хода.
"""

#: Частоты типов в наборе. Не равномерные: основные сценарии должны преобладать,
#: иначе adversarial-поднабор утянет общую цифру и будет непонятно, что сломано.
#: Значения заданы ДО прогона и фиксируются коммитом (docs/EVAL-PLAN.md §5).
KIND_WEIGHTS: Final[dict[str, float]] = {
    "discovery_vague": 0.08,
    "discovery_genre": 0.09,
    "discovery_multi": 0.09,
    "mood_evening": 0.08,
    "similar_seed": 0.07,
    "navigation_title": 0.05,
    "course_level": 0.08,
    "course_practical": 0.05,
    "incremental": 0.09,
    "release_constraint": 0.07,
    "domain_switch": 0.06,
    "exclusion_genre": 0.06,
    "tone_negation": 0.04,
    "no_result": 0.02,
    "contradiction": 0.02,
    "more_variants": 0.05,
}


def _check_weights_sum_to_one() -> None:
    total = sum(KIND_WEIGHTS.values())
    if abs(total - 1.0) > 1e-6:
        raise AssertionError(f"KIND_WEIGHTS не нормированы: сумма={total!r}")
    if set(KIND_WEIGHTS) != set(SCENARIO_KINDS):
        raise AssertionError(f"веса и типы разошлись: {set(KIND_WEIGHTS) ^ set(SCENARIO_KINDS)}")


_check_weights_sum_to_one()


# ---------------------------------------------------------------------------
# Сборка сценария
# ---------------------------------------------------------------------------


def _spec_for(criteria: OracleCriteria) -> OracleCriteriaSpec:
    return OracleCriteriaSpec(
        acceptable_ids=sorted(criteria.acceptable_ids),
        ceiling_ids=list(criteria.ceiling_ids),
        threshold=criteria.threshold,
        catalog_size=criteria.catalog_size,
        feasible_count=criteria.feasible_count,
        max_clarifications=criteria.max_clarifications,
    )


def _build_second_criteria(
    *,
    catalog: Sequence[Item],
    profile: UserProfile,
    final: Turn,
) -> tuple[OracleCriteriaSpec | None, tuple[str, ...]]:
    """Критерий для пути «агент уточнил и получил ``answer_if_clarified``».

    Строится только если ответ пользователя ДОБАВЛЯЕТ ограничения: иначе критерий
    совпадает с основным, и дублировать его незачем. Второй критерий проверяется
    на вырожденность так же, как основной, — непроходимый сценарий после
    уточнения занижал бы метрики всем конфигурациям, а уточняющие вопросы (L4, L5)
    наказывал бы сильнее, чем молчаливую выдачу.
    """
    spoken_after = final.spoken_after_answer
    if final.answer_if_clarified is None or spoken_after is None:
        return None, ()
    criteria = build_criteria(
        catalog=catalog,
        theta=profile.theta,
        user_id=profile.user_id,
        spoken=spoken_after,
        expected=ExpectedOutcome.RECOMMEND,
        patience=profile.theta.patience,
    )
    if is_degenerate(criteria)[0]:
        # Вырожденность пути после уточнения отбраковывает сценарий ЦЕЛИКОМ, а не
        # оставляет его с базовым критерием. Иначе прогон незаметно мерил бы путь
        # «агент не спросил» там, где должен был мерить путь «агент спросил», и
        # адаптивная политика уточнений (L5) получала бы нечестное сравнение с L4.
        return None, ("clarify_path_degenerate",)
    return _spec_for(criteria), ()


def _assemble(
    *,
    scenario_id: str,
    kind: str,
    split: Split,
    seed: int,
    profile: UserProfile,
    turns: list[Turn],
    extra: dict,
    catalog: Sequence[Item],
) -> tuple[Scenario | None, tuple[str, ...]]:
    """Собрать сценарий или вернуть None, если он вырожден.

    Критерий строится по spoken ФИНАЛЬНОГО хода: именно последний ответ агента
    принимается или отвергается. Промежуточные ходы влияют на ожидание (CLARIFY на
    первом ходу), но критерий один — на финал.
    """
    final = turns[-1]
    criteria = build_criteria(
        catalog=catalog,
        theta=profile.theta,
        user_id=profile.user_id,
        spoken=final.spoken,
        expected=final.expected,
        patience=profile.theta.patience,
    )
    after_spec, after_problems = _build_second_criteria(catalog=catalog, profile=profile, final=final)
    degenerate, problems = is_degenerate(criteria)
    all_problems = tuple(problems) + after_problems
    if degenerate or all_problems:
        # Отбраковка — не молчаливая: причины возвращаются вызывающему в сводку.
        return None, all_problems

    feasible = feasible_set(catalog, final.spoken)
    return Scenario(
        scenario_id=scenario_id,
        kind=kind,  # type: ignore[arg-type]
        split=split,
        seed=seed,
        user_id=profile.user_id,
        theta=profile.theta,
        turns=turns,
        criteria=_spec_for(criteria),
        feasible_count=len(feasible),
        acceptable_count=len(criteria.acceptable_ids),
        adversarial_note=extra.get("adversarial_note"),
        criteria_after_clarify=after_spec,
    ), ()


def generate_scenarios(
    split: Split = "dev",
    size: int | None = None,
    seed: int | None = None,
    catalog: Sequence[Item] | None = None,
    profiles: Sequence[UserProfile] | None = None,
) -> tuple[list[Scenario], dict[str, object]]:
    """Сгенерировать набор сценариев для сплита.

    Возвращает пару ``(сценарии, сводка)``. Сводка содержит сколько кандидатов
    было отброшено и по каким причинам: скрывать отбраковку нельзя, иначе
    непонятно, на какой доле пространства сценариев измерены цифры.

    Цикл продолжается, пока не набрано ``size`` ПРИГОДНЫХ сценариев, поэтому
    размер набора стабилен и не зависит от того, сколько кандидатов оказалось
    вырожденными.
    """
    if split not in SPLIT_SEEDS:
        raise ValueError(f"неизвестный сплит {split!r}, допустимы {tuple(SPLIT_SEEDS)}")
    split_seed = seed if seed is not None else SPLIT_SEEDS[split]
    target = size if size is not None else SPLIT_SIZE[split]
    if target < 1:
        raise ValueError("size должен быть не меньше 1")

    # Both splits share a catalog and profile population; only dialogue sampling
    # and the disjoint profile ranges differ. Match the CLI's default dataset.
    items = list(catalog) if catalog is not None else generate_catalog(SPLIT_SEEDS["dev"])
    all_profiles = list(profiles) if profiles is not None else generate_profiles(SPLIT_SEEDS["dev"], PROFILE_COUNT, items)
    low, high = DEV_PROFILE_RANGE if split == "dev" else HOLDOUT_PROFILE_RANGE
    pool = all_profiles[low:high]
    if not pool:
        raise ValueError(f"диапазон профилей {low}..{high} пуст: profiles={len(all_profiles)}")

    rng = random.Random(split_seed)
    kinds = list(KIND_WEIGHTS)
    weights = [KIND_WEIGHTS[kind] for kind in kinds]

    scenarios: list[Scenario] = []
    rejected: collections.Counter[str] = collections.Counter()
    rejected_reasons: collections.Counter[str] = collections.Counter()
    attempted = 0
    # Предохранитель: если отбраковка станет тотальной, цикл обязан остановиться,
    # а не крутиться вечно. 40x — запас, при реальной отбраковке в единицы
    # процентов он никогда не достигается.
    max_attempts = target * 40
    index = 0

    while len(scenarios) < target and attempted < max_attempts:
        attempted += 1
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        profile = pool[(index + rng.randrange(len(pool))) % len(pool)]
        index += 1
        builder = BUILDERS[kind]
        # Курс-подвиды различаются только озвученным слотом, поэтому один
        # конструктор на два типа; выбор внутри делает сам конструктор по rng.
        turns, extra = builder(rng, profile.theta, items, profile)  # type: ignore[operator]
        scenario_id = f"{split}-{kind}-{len(scenarios) + 1:04d}"
        scenario, problems = _assemble(
            scenario_id=scenario_id,
            kind=kind,
            split=split,
            seed=split_seed,
            profile=profile,
            turns=turns,
            extra=extra,
            catalog=items,
        )
        if scenario is None:
            rejected[kind] += 1
            rejected_reasons.update(problems)
            continue
        scenarios.append(scenario)

    if len(scenarios) < target:
        raise RuntimeError(
            f"не удалось набрать {target} сценариев: собрано {len(scenarios)}, "
            f"отброшено {sum(rejected.values())} ({dict(rejected)}), причины {dict(rejected_reasons)}. "
            "Отбраковка тотальная — проверьте пороги оракула и калибровку каталога."
        )

    summary = {
        "split": split,
        "seed": split_seed,
        "size": len(scenarios),
        "attempted": attempted,
        "rejected_total": sum(rejected.values()),
        "rejected_by_kind": dict(rejected),
        "rejected_by_reason": dict(rejected_reasons),
        "profile_range": [low, high],
        "catalog_size": len(items),
        "adversarial": sum(1 for scenario in scenarios if scenario.is_adversarial),
        "by_kind": dict(collections.Counter(scenario.kind for scenario in scenarios)),
        "acceptable_share_median": _median([scenario.acceptable_count / scenario.criteria.catalog_size for scenario in scenarios]),
        "feasible_median": _median([scenario.feasible_count for scenario in scenarios]),
        "multi_turn": sum(1 for scenario in scenarios if len(scenario.turns) > 1),
    }
    return scenarios, summary


def _median(values: Sequence[float | int]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return float(ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2)


def scenarios_sha256(split: Split = "dev") -> str:
    """Отпечаток набора: воспроизводимость между процессами."""
    scenarios, _ = generate_scenarios(split)
    payload = [scenario.model_dump(mode="json") for scenario in scenarios]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_jsonl(scenarios: Sequence[Scenario], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(scenario.model_dump_json() + "\n")
    return path


def read_jsonl(path: Path) -> list[Scenario]:
    """Чтение набора. Сценарии хранятся в git, поэтому прогон не обязан их собирать заново."""
    return [Scenario.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать сценарии диалогов D3")
    parser.add_argument("--split", choices=tuple(SPLIT_SEEDS), default=None, help="по умолчанию оба")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/data"))
    args = parser.parse_args()

    catalog = generate_catalog(SPLIT_SEEDS["dev"])
    profiles = generate_profiles(SPLIT_SEEDS["dev"], PROFILE_COUNT, catalog)
    splits: tuple[Split, ...] = (args.split,) if args.split else ("dev", "holdout")

    for split in splits:
        scenarios, summary = generate_scenarios(split, args.size, catalog=catalog, profiles=profiles)
        path = write_jsonl(scenarios, args.output_dir / f"scenarios-{split}.jsonl")
        print(f"{split}: {summary['size']} сценариев -> {path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
