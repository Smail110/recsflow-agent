"""D2: скрытые предпочтения пользователей (theta) и их типизированная история.

Зачем этот модуль существует
-----------------------------
Метрика ``success_rate = 1.0`` у прежнего прототипа была тавтологичной: оракул
проверял те же поля, которые агент применял как жёсткие фильтры. Прибор, который
не может сказать «плохо», ничего не измеряет (``docs/EVAL-PLAN.md`` §1-2).

Лечится это единственным способом: ввести ground truth, которого агент не видит.
Таким ground truth и является theta — вектор скрытых предпочтений пользователя.
Профиль знает, что человеку на самом деле нравится; агент знает только то, что
человек сказал вслух, и то, что лежит в его истории.

Разделение ответственности (``docs/DATA-PLAN.md`` §4, D2/D3):

    theta + utility   знают оракул и симулятор; агент — никогда
    Query             извлекает агент; оракул — никогда его не читает

``utility`` живёт здесь, а не в оракуле: это часть модели пользователя (данные D2),
и она же нужна для генерации правдоподобной истории профиля. Оракул в
``evals/oracle.py`` строит на её основе критерии приёма ответа, но не наоборот.

Циркулярность запрещена тестом ``tests/unit/test_oracle_independence.py``: оракулу
нельзя импортировать код агента (``matches``, ``rule_parse``, ``explain``,
``Agent``), а коду агента — theta.

Что синтетическое и почему
--------------------------
Публичного набора с русским языком, нашей схемой слотов и скрытой полезностью не
существует (``docs/DATA-PLAN.md`` §2), поэтому профили генерируются по seed —
задача это прямо допускает. Архетипы не взяты с потолка: жанры и шкалы те же, что
в каталоге, откалиброванном по MovieLens-100k (``calibration.py``).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, field_validator

from ..models import Item, StrictModel

# ---------------------------------------------------------------------------
# Домены значений. Держим их рядом с генератором каталога, а не в оракуле:
# theta обязана говорить на том же языке атрибутов, что и Item.
# ---------------------------------------------------------------------------

GENRES: Final[tuple[str, ...]] = (
    "детектив",
    "комедия",
    "драма",
    "фантастика",
    "приключения",
    "машинное обучение",
    "python",
)
TONES: Final[tuple[str, ...]] = ("лёгкий", "нейтральный", "мрачный")
TonePref = Literal["лёгкий", "нейтральный", "мрачный", "indifferent"]
LevelPref = Literal["начальный", "средний", "продвинутый", "any"]
EventType = Literal["viewed", "liked", "disliked", "rated"]

# Тип события совпадает с EventType контракта (docs/contract/openapi.yaml). Это не
# случайность, а требование: симулятор ставит реакции и отвечает на уточнения так,
# как это делала бы платформа, поэтому история профиля обязана быть представима в
# схеме HistoryEvent. Иначе mock-платформа из P3 не смогла бы отдать оракулу ту же
# историю, которую видит агент.

#: Нижний предел веса жанра. При нулевом весе жанр выпадает из приемлемого
#: множества целиком, и сценарий «кино-зритель спросил про курс» проваливался бы
#: по построению, а не по измерению.
MIN_GENRE_WEIGHT: Final[float] = 0.015


class Theta(StrictModel):
    """Скрытый вектор предпочтений. Агент его не видит никогда."""

    genre_weights: dict[str, float] = Field(min_length=len(GENRES), max_length=len(GENRES))
    tone_pref: TonePref = "indifferent"
    duration_tol: int = Field(ge=10, le=1500, description="Мягкая терпимость к длительности, минуты")
    level_pref: LevelPref = "any"
    practical_pref: bool | None = None
    patience: int = Field(ge=1, le=6, description="Сколько уточняющих вопросов выдержит, прежде чем уйти")
    verbosity: float = Field(ge=0.0, le=1.0, description="Многословность: влияет на поверхностную реализацию реплик")

    @field_validator("genre_weights")
    @classmethod
    def normalized(cls, value: dict[str, float]) -> dict[str, float]:
        """Веса нормированы и покрывают все жанры.

        Проверка здесь, а не только в генераторе: профиль можно собрать и вручную
        (adversarial-сценарии, ручная валидация), и ненормированные веса молча
        изменили бы масштаб полезности — порог приёма перестал бы быть порогом.
        """
        if set(value) != set(GENRES):
            missing = sorted(set(GENRES) - set(value))
            raise ValueError(f"genre_weights обязан покрывать все жанры; нет: {missing}")
        if any(weight < 0 for weight in value.values()):
            raise ValueError("отрицательный вес жанра")
        total = math.fsum(value.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"genre_weights обязан быть нормирован, сумма={total!r}")
        return value


class HistoryEvent(StrictModel):
    """Событие истории. Схема совпадает с HistoryEvent контракта."""

    item_id: str = Field(min_length=1, max_length=64)
    event_type: EventType


class UserProfile(StrictModel):
    """Полный профиль D2: theta, архетип и типизированная история."""

    user_id: str = Field(min_length=1, max_length=64)
    archetype: str
    theta: Theta
    history: list[HistoryEvent] = Field(default_factory=list)
    seed: int = Field(ge=0)

    @property
    def liked_ids(self) -> tuple[str, ...]:
        return tuple(event.item_id for event in self.history if event.event_type == "liked")

    @property
    def disliked_ids(self) -> tuple[str, ...]:
        return tuple(event.item_id for event in self.history if event.event_type == "disliked")

    @property
    def seen_ids(self) -> tuple[str, ...]:
        """Всё, с чем пользователь уже взаимодействовал, независимо от типа события."""
        return tuple(event.item_id for event in self.history)


# ---------------------------------------------------------------------------
# Полезность: как theta превращается в число
# ---------------------------------------------------------------------------

#: Веса слагаемых полезности. Фиксированы ДО экспериментов — требование
#: ``docs/EVAL-PLAN.md`` §5: пороги регистрируются коммитом до прогона, иначе их
#: всегда можно подкрутить под результат.
W_GENRE: Final[float] = 0.40
W_TONE: Final[float] = 0.20
W_DURATION: Final[float] = 0.15
W_QUALITY: Final[float] = 0.20
W_FIT: Final[float] = 0.05

#: Насколько приемлем тон объекта при несовпадении с предпочтением. Таблица, а не
#: одно число: «хочу лёгкое, а дали нейтральное» и «хочу лёгкое, а дали мрачное» —
#: разные ситуации, и одна константа скрыла бы это различие.
TONE_FIT: Final[dict[str, dict[str, float]]] = {
    "лёгкий": {"лёгкий": 1.0, "нейтральный": 0.55, "мрачный": 0.15},
    "нейтральный": {"лёгкий": 0.70, "нейтральный": 1.0, "мрачный": 0.55},
    "мрачный": {"лёгкий": 0.15, "нейтральный": 0.60, "мрачный": 1.0},
}

#: Совпадение уровня и практики курса. Уровень штрафуется мягче тона: курс
#: «продвинутый» вместо «начального» чаще поправим (есть практика), чем мрачный
#: триллер вместо лёгкого на вечер.
LEVEL_FIT_MATCH: Final[float] = 1.0
LEVEL_FIT_MISMATCH: Final[float] = 0.20
PRACTICAL_FIT_MATCH: Final[float] = 1.0
PRACTICAL_FIT_MISMATCH: Final[float] = 0.35


def genre_fit(item: Item, theta: Theta) -> float:
    """Вклад жанра. Неизвестный жанр даёт ноль, а не средний вес."""
    return theta.genre_weights.get(item.genre, 0.0)


def tone_fit(item: Item, theta: Theta) -> float:
    """Вклад тона. ``indifferent`` — тон не участвует, но и не наказывает."""
    if theta.tone_pref == "indifferent":
        return 1.0
    return TONE_FIT[theta.tone_pref].get(item.tone, 0.3)


def duration_fit(item: Item, theta: Theta) -> float:
    """Вклад длительности — МЯГКОЕ предпочтение, не жёсткий фильтр.

    Экспоненциальное затухание сверх терпимости, а не порог. Это принципиально:
    жёсткие ограничения — то, что пользователь назвал вслух, а theta описывает,
    что ему в принципе комфортно. Человек с ``duration_tol=45`` не отвергнет
    пятиминутное превышение, но трёхчасовой фильм «на вечер» ему не зайдёт.
    ``exp(-1)`` на удвоенной длительности даёт как раз такую форму.
    """
    if item.minutes <= theta.duration_tol:
        return 1.0
    return math.exp(-(item.minutes - theta.duration_tol) / theta.duration_tol)


def course_fit(item: Item, theta: Theta) -> float:
    """Вклад уровня и практики. Для фильмов и сериалов нейтрален (1.0).

    Нейтральность важна: иначе кино-профиль получил бы нулевую полезность всех
    курсов даже при ненулевом весе жанра, и приемлемое множество схлопнулось бы
    по построению, а не по предпочтению.
    """
    if item.kind != "course":
        return 1.0
    level = 1.0 if theta.level_pref == "any" else (LEVEL_FIT_MATCH if item.level == theta.level_pref else LEVEL_FIT_MISMATCH)
    if theta.practical_pref is None:
        practical = 1.0
    elif item.practical is None:
        # «Нет данных» не удовлетворяет предпочтению и не нарушает его: середина.
        practical = (PRACTICAL_FIT_MATCH + PRACTICAL_FIT_MISMATCH) / 2
    else:
        practical = PRACTICAL_FIT_MATCH if item.practical == theta.practical_pref else PRACTICAL_FIT_MISMATCH
    return (level + practical) / 2


def utility(item: Item, theta: Theta) -> float:
    """Полезность объекта для профиля, 0..1.

    Аддитивная смесь с фиксированными весами. Аддитивность выбрана сознательно,
    хотя мультипликативная форма сильнее разделяла бы объекты: при произведении
    один несовпавший слот обнуляет всё, и приемлемое множество схлопывается к
    пересечению условий. Это ровно та тавтология, от которой мы уходим — оракул
    начал бы повторять поведение жёстких фильтров агента. Аддитивная смесь
    допускает компромиссы, как допускает их человек.
    """
    return round(
        W_GENRE * genre_fit(item, theta)
        + W_TONE * tone_fit(item, theta)
        + W_DURATION * duration_fit(item, theta)
        + W_QUALITY * item.quality
        + W_FIT * course_fit(item, theta),
        6,
    )


# ---------------------------------------------------------------------------
# Архетипы
# ---------------------------------------------------------------------------


class _Archetype(StrictModel):
    """Заготовка профиля: базовые веса жанров и характерные параметры."""

    model_config = StrictModel.model_config | {"frozen": True}

    name: str
    genre_weights: dict[str, float]
    tone_pref: TonePref = "indifferent"
    duration_tol: int = 100
    level_pref: LevelPref = "any"
    practical_pref: bool | None = None
    kind_hint: Literal["film_series", "course", "any"] = "any"


def _w(**genre_weights: float) -> dict[str, float]:
    """Нормированные веса с нижним пределом MIN_GENRE_WEIGHT для всех жанров.

    Предел добавляется ко всем жанрам, а не только к неназванным: профиль,
    который «любит детектив и ничего больше», сделал бы приемлемое множество
    одножанровым, и любой ответ вне него проваливался бы независимо от качества
    диалога.
    """
    raw = {genre: float(genre_weights.get(genre, 0.0)) for genre in GENRES}
    floored = {genre: weight + MIN_GENRE_WEIGHT for genre, weight in raw.items()}
    total = math.fsum(floored.values())
    return {genre: weight / total for genre, weight in floored.items()}


ARCHETYPES: Final[tuple[_Archetype, ...]] = (
    _Archetype(
        name="evening_detective",
        genre_weights=_w(детектив=0.55, драма=0.15, комедия=0.15, приключения=0.10, фантастика=0.05),
        tone_pref="лёгкий",
        duration_tol=45,
        kind_hint="film_series",
    ),
    _Archetype(
        name="comedy_relax",
        genre_weights=_w(комедия=0.50, приключения=0.20, детектив=0.15, драма=0.10, фантастика=0.05),
        tone_pref="лёгкий",
        duration_tol=100,
        kind_hint="film_series",
    ),
    _Archetype(
        name="dark_drama",
        genre_weights=_w(драма=0.55, детектив=0.20, фантастика=0.15, приключения=0.05, комедия=0.05),
        tone_pref="мрачный",
        duration_tol=135,
        kind_hint="film_series",
    ),
    _Archetype(
        name="scifi_fan",
        genre_weights=_w(фантастика=0.50, приключения=0.25, драма=0.15, детектив=0.10),
        tone_pref="нейтральный",
        duration_tol=125,
        kind_hint="film_series",
    ),
    _Archetype(
        name="adventure_family",
        genre_weights=_w(приключения=0.45, комедия=0.25, фантастика=0.20, детектив=0.10),
        tone_pref="лёгкий",
        duration_tol=105,
        kind_hint="film_series",
    ),
    _Archetype(
        name="binge_watcher",
        genre_weights=_w(детектив=0.40, драма=0.35, фантастика=0.15, комедия=0.10),
        tone_pref="нейтральный",
        duration_tol=40,
        kind_hint="film_series",
    ),
    _Archetype(
        name="ml_learner",
        genre_weights=_w(**{"машинное обучение": 0.60, "python": 0.40}),
        duration_tol=600,
        level_pref="начальный",
        practical_pref=True,
        kind_hint="course",
    ),
    _Archetype(
        name="python_practitioner",
        genre_weights=_w(**{"python": 0.65, "машинное обучение": 0.35}),
        duration_tol=300,
        level_pref="средний",
        practical_pref=True,
        kind_hint="course",
    ),
    _Archetype(
        name="omnivore",
        genre_weights=_w(
            детектив=0.16, комедия=0.16, драма=0.16, фантастика=0.16, приключения=0.16, **{"машинное обучение": 0.10, "python": 0.10}
        ),
        tone_pref="indifferent",
        duration_tol=150,
        kind_hint="any",
    ),
    # Профиль с терпимостью ниже медианной длительности курса: проверяет, что
    # полезность наказывает за длинное, а не только за «не то».
    _Archetype(
        name="short_course_learner",
        genre_weights=_w(**{"python": 0.55, "машинное обучение": 0.45}),
        duration_tol=90,
        level_pref="начальный",
        kind_hint="course",
    ),
)


# ---------------------------------------------------------------------------
# Генерация профилей
# ---------------------------------------------------------------------------

#: Число профилей D2 (docs/DATA-PLAN.md §4: ~500).
PROFILE_COUNT: Final[int] = 500

#: Число событий истории. Реальные истории длиннее; здесь достаточно, чтобы
#: персонализация и исключение просмотренного были проверяемы.
HISTORY_SIZE_RANGE: Final[tuple[int, int]] = (4, 24)
DISLIKE_COUNT_RANGE: Final[tuple[int, int]] = (0, 3)
LIKED_UTILITY_FLOOR: Final[float] = 0.68

#: Какую долю каталога берём для лайков и для дизлайков. Верхняя треть — то, что
#: человек мог посмотреть; нижняя пятая — то, что ему не зашло.
LIKED_POOL_FRACTION: Final[float] = 1 / 3
DISLIKED_POOL_FRACTION: Final[float] = 1 / 5


def _noisy_weights(rng: random.Random, base: Mapping[str, float]) -> dict[str, float]:
    """Логнормальный шум по архетипным весам с последующей нормировкой.

    Логнормаль, а не гаусс: веса обязаны быть положительными, а гауссов шум
    порядка 0.35 регулярно давал бы отрицательные значения для редких жанров.
    Обрезание нуля исказило бы форму распределения сильнее, чем лог-нормаль.
    """
    noisy = {genre: base[genre] * math.exp(rng.gauss(0.0, 0.35)) for genre in GENRES}
    total = math.fsum(noisy.values())
    return {genre: weight / total for genre, weight in noisy.items()}


def _sample_theta(rng: random.Random, archetype: _Archetype) -> Theta:
    tone_pref: TonePref = archetype.tone_pref
    # Часть профилей не имеет предпочтения по тону: если бы тон хотели все,
    # ветка ``indifferent`` в utility не исполнялась бы ни в одном тесте.
    if tone_pref != "indifferent" and rng.random() < 0.12:
        tone_pref = "indifferent"

    # Разброс терпимости к длительности логнормален: у большинства около
    # архетипной, у немногих сильно больше. Та же форма длинного хвоста, что в
    # каталоге, поэтому профили и объекты согласованы по шкале.
    duration_tol = min(1500, max(15, round(archetype.duration_tol * math.exp(rng.gauss(0.0, 0.22)))))

    level_pref = archetype.level_pref
    practical_pref = archetype.practical_pref
    if archetype.kind_hint == "course":
        # Часть учащихся не имеет явного предпочтения по уровню и практике: иначе
        # course_fit принимал бы только крайние значения, без промежуточных.
        if rng.random() < 0.25:
            level_pref = "any"
        if rng.random() < 0.2:
            practical_pref = None

    return Theta(
        genre_weights=_noisy_weights(rng, archetype.genre_weights),
        tone_pref=tone_pref,
        duration_tol=duration_tol,
        level_pref=level_pref,
        practical_pref=practical_pref,
        # Терпение распределено как в DATA-PLAN: большинство выдерживает три
        # уточнения, меньшинство — два или четыре-пять.
        patience=rng.choices((2, 3, 4, 5), weights=(0.25, 0.45, 0.20, 0.10), k=1)[0],
        verbosity=round(min(1.0, max(0.0, rng.betavariate(2.0, 2.6))), 4),
    )


def _sample_history(rng: random.Random, theta: Theta, catalog: Sequence[Item]) -> list[HistoryEvent]:
    """Типизированная история, согласованная с theta.

    История не случайна по составу: лайкается то, что профилю действительно
    нравится (utility выше порога), а дизлайкается то, что не нравится. Без этого
    персонализацию агента нельзя было бы отличить от шума, а бейзлайн B1
    («сырой Recsflow по истории») получал бы бессмысленный вход.

    Лайки берутся из верхней трети распределения полезности, но случайным
    выбором внутри неё, а не первыми N по убыванию: человек смотрит не только
    идеальное для себя, и детерминированный топ создал бы вырожденно
    предсказуемую историю.
    """
    scored = sorted(((utility(item, theta), item) for item in catalog), key=lambda pair: -pair[0])
    top = scored[: max(1, int(len(scored) * LIKED_POOL_FRACTION))]
    bottom = scored[-max(1, int(len(scored) * DISLIKED_POOL_FRACTION)) :]

    size = rng.randint(*HISTORY_SIZE_RANGE)
    picked = rng.sample(top, min(size, len(top)))

    events = [HistoryEvent(item_id=item.id, event_type="liked" if value >= LIKED_UTILITY_FLOOR else "viewed") for value, item in picked]

    dislike_count = rng.randint(*DISLIKE_COUNT_RANGE)
    if dislike_count:
        liked_or_viewed = {event.item_id for event in events}
        candidates = [item for _, item in bottom if item.id not in liked_or_viewed]
        for item in rng.sample(candidates, min(dislike_count, len(candidates))):
            events.append(HistoryEvent(item_id=item.id, event_type="disliked"))

    rng.shuffle(events)
    return events


def generate_profiles(seed: int = 42, count: int = PROFILE_COUNT, catalog: Sequence[Item] | None = None) -> list[UserProfile]:
    """Профили D2. Одинаковые ``(seed, count)`` всегда дают одинаковый результат.

    Каталог передаётся извне, а не генерируется внутри: история профиля обязана
    ссылаться на объекты того же каталога, который видит агент. Если бы профиль
    генерировал каталог сам, расхождение seed'ов дало бы историю из несуществующих
    id, и lookup молча вернул бы пустой список.
    """
    from .generator import generate_catalog

    if seed < 0:
        raise ValueError("seed не может быть отрицательным")
    if count < 1:
        raise ValueError("count должен быть не меньше 1")
    items = list(catalog) if catalog is not None else generate_catalog(seed)
    rng = random.Random(seed)

    profiles: list[UserProfile] = []
    for index in range(count):
        archetype = ARCHETYPES[index % len(ARCHETYPES)]
        theta = _sample_theta(rng, archetype)
        profiles.append(
            UserProfile(
                user_id=f"user-{index + 1:04d}",
                archetype=archetype.name,
                theta=theta,
                history=_sample_history(rng, theta, items),
                seed=seed,
            )
        )
    return profiles


def profiles_by_id(profiles: Iterable[UserProfile]) -> dict[str, UserProfile]:
    """Индекс для быстрого доступа симулятора и оракула."""
    return {profile.user_id: profile for profile in profiles}


def profile_sha256(seed: int = 42, count: int = PROFILE_COUNT) -> str:
    """Отпечаток набора профилей: проверка воспроизводимости между процессами."""
    payload = [profile.model_dump(mode="json") for profile in generate_profiles(seed, count)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def profile_stats(profiles: Sequence[UserProfile]) -> dict[str, object]:
    """Сводка по набору профилей: используется тестами и отчётом."""
    return {
        "count": len(profiles),
        "archetypes": dict(collections.Counter(profile.archetype for profile in profiles)),
        "tone_prefs": dict(collections.Counter(profile.theta.tone_pref for profile in profiles)),
        "duration_tol_median": statistics.median(profile.theta.duration_tol for profile in profiles),
        "patience_median": statistics.median(profile.theta.patience for profile in profiles),
        "history_size_mean": round(statistics.fmean(len(profile.history) for profile in profiles), 2),
        "history_events_total": sum(len(profile.history) for profile in profiles),
        "liked_total": sum(len(profile.liked_ids) for profile in profiles),
        "disliked_total": sum(len(profile.disliked_ids) for profile in profiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать профили пользователей D2 (theta + история)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=PROFILE_COUNT)
    parser.add_argument("--output", default="data/profiles.json")
    args = parser.parse_args()

    profiles = generate_profiles(args.seed, args.count)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([profile.model_dump(mode="json") for profile in profiles], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сгенерировано {len(profiles)} профилей: {path}")
    print(f"sha256={profile_sha256(args.seed, args.count)}")
    print(json.dumps(profile_stats(profiles), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
