"""Генератор каталога D1 (~3000 объектов, детерминированный по seed).

Названия процедурные и заведомо вымышленные: реальных фильмов и сериалов в каталоге
нет. Из открытых данных (см. calibration.py) берутся только распределения — частоты
жанров, длинный хвост популярности и форма распределения качества.

Главное отличие от прежнего генератора: ``quality`` больше не ``rng.uniform(.55, .98)``.
Это было фактически случайное ранжирование, которое невозможно отличить от шума. Теперь
качество — функция популярности с шумом выборки: у объекта с двумя оценками среднее
неустойчиво, у объекта с тремястами — устойчиво. Так у реранкера из P5 появляется сигнал,
который можно выучить, и так воспроизводится наблюдаемая в MovieLens связь
«популярное в среднем rated выше».

Названия собраны по строгим грамматическим шаблонам из words.py, поэтому читаются
по-русски, оставаясь вымышленными.

Детерминизм проверяется тестом по SHA-256 двух независимых прогонов (в отдельных
процессах), а не сравнением объектов в одном процессе: сравнение в одном процессе
не поймало бы зависимость от порядка обхода словаря или от хеш-сида Python.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from ..models import Item
from .calibration import (
    CATALOG_SIZE,
    COURSE_GENRE_SHARE,
    COURSE_MINUTES,
    EPISODES_PER_SEASON,
    FILM_MINUTES,
    GENRE_SHARE_MOVIELENS,
    KIND_SHARE,
    POPULARITY_QUANTILES,
    QUALITY_INTERCEPT,
    QUALITY_LOG2_SLOPE,
    QUALITY_TRUE_SD,
    RATING_SCALE_MAX,
    RATING_SCALE_MIN,
    RATING_SD_WITHIN_ITEM,
    SERIES_MINUTES,
    SERIES_SEASONS,
    TONE_WEIGHTS,
    UNKNOWN_RATE_EPISODES,
    UNKNOWN_RATE_LEVEL,
    UNKNOWN_RATE_PRACTICAL,
    UNKNOWN_RATE_SEASONS,
    UNKNOWN_RATE_YEAR,
    YEAR_RANGE_COURSE,
    YEAR_RANGE_FILM_SERIES,
)
from .words import ADJ_INDEX, ADJECTIVES, NOUNS, QUALIFIERS

LEVELS: tuple[str, ...] = ("начальный", "средний", "продвинутый")
LEVEL_WEIGHTS: tuple[float, ...] = (0.55, 0.28, 0.17)


def _pick(rng: random.Random, options: tuple[str, ...], weights: tuple[float, ...] | None = None) -> str:
    """Выбор одного варианта; weights нормируются внутри."""
    if weights is None:
        return rng.choice(options)
    return rng.choices(options, weights=weights, k=1)[0]


def _sample_popularity(rng: random.Random) -> int:
    """Число оценок объекта — интерполяция эмпирических квантилей в лог-шкале.

    Логарифмическая интерполяция нужна потому, что распределение длиннохвостое:
    линейная интерполяция между квантилями 27 и 378 сдвинула бы медиану вверх.
    """
    u = rng.random()
    q0, n0 = POPULARITY_QUANTILES[0]
    q1, n1 = POPULARITY_QUANTILES[-1]
    for (qa, na), (qb, nb) in pairwise(POPULARITY_QUANTILES):
        if u <= qb:
            q0, n0, q1, n1 = qa, na, qb, nb
            break
    if q1 == q0:
        return n1
    share = (u - q0) / (q1 - q0)
    value = math.exp(math.log(n0) + share * (math.log(n1) - math.log(n0)))
    return max(1, round(value))


def _sample_quality(rng: random.Random, popularity: int) -> float:
    """Истинное качество + шум выборки, зависящий от числа оценок.

    Шум делится на sqrt(n): у объекта с одной оценкой среднее может быть любым
    (пулированный sd отдельной оценки ~ 1.0 балла), у объекта с 583 — почти совпадает
    с истинным. Берётся именно sd внутри объекта, а не sd всех оценок набора: часть
    общей дисперсии — различие между объектами, и её добавляет QUALITY_TRUE_SD. Это ровно та
    картина, которую видно в u.data, и именно она делает длинный хвост не «плохим»,
    а «неизвестным».
    """
    true_rating = rng.gauss(QUALITY_INTERCEPT + QUALITY_LOG2_SLOPE * math.log2(max(1, popularity)), QUALITY_TRUE_SD)
    observed = rng.gauss(true_rating, RATING_SD_WITHIN_ITEM / math.sqrt(popularity))
    clipped = min(RATING_SCALE_MAX, max(RATING_SCALE_MIN, observed))
    return round((clipped - RATING_SCALE_MIN) / (RATING_SCALE_MAX - RATING_SCALE_MIN), 4)


def _minutes(rng: random.Random, kind: str) -> int:
    low, high = {"series": SERIES_MINUTES, "film": FILM_MINUTES, "course": COURSE_MINUTES}[kind]
    # Логнормаль даёт перекос вправо: большинство коротких, немного очень длинных.
    span = math.log(high) - math.log(low)
    value = math.exp(math.log(low) + span * rng.betavariate(2.2, 3.4))
    return max(low, min(high, round(value)))


def _genre_for(rng: random.Random, kind: str) -> str:
    if kind == "course":
        names = tuple(COURSE_GENRE_SHARE)
        return _pick(rng, names, tuple(COURSE_GENRE_SHARE[n] for n in names))
    names = tuple(GENRE_SHARE_MOVIELENS)
    return _pick(rng, names, tuple(GENRE_SHARE_MOVIELENS[n] for n in names))


def _title_candidates(genre: str) -> tuple[str, ...]:
    """Все допустимые названия жанра. Внутри жанра повторов нет.

    Две формы: «Тайна старого маяка» (существительное + модификатор в родительном)
    и «Долгая дорога» (прилагательное, согласованное с родом существительного).

    Повторы ВОЗМОЖНЫ между жанрами: одно и то же слово встречается в нескольких
    банках («станция» есть в драме и в фантастике, «маяк» — в детективе и в
    приключениях). Поэтому глобальную уникальность обеспечивает не этот пул,
    а _build(), который пропускает уже занятые названия.
    """
    nouns = NOUNS[genre]
    quals = QUALIFIERS[genre]
    titles = [f"{noun.capitalize()} {qual}" for noun, gender in nouns for qual in quals]
    titles += [f"{ADJECTIVES[k][ADJ_INDEX[gender]].capitalize()} {noun}" for noun, gender in nouns for k in range(len(ADJECTIVES))]
    if len(set(titles)) != len(titles):
        raise RuntimeError(f"в банке названий жанра {genre!r} есть дубликаты")
    return tuple(titles)


def title_pool_size(genre: str) -> int:
    """Запас названий жанра. Нужен тесту, который следит за исчерпанием пула."""
    return len(_title_candidates(genre))


@lru_cache(maxsize=8)
def _build(seed: int, size: int) -> tuple[Item, ...]:
    """Детерминированная сборка каталога. Кортеж, чтобы результат нельзя было изменить."""
    rng = random.Random(seed)
    kinds = tuple(KIND_SHARE)
    kind_weights = tuple(KIND_SHARE[k] for k in kinds)
    titles_left: dict[str, list[str]] = {}
    used_titles: set[str] = set()
    items: list[Item] = []

    for index in range(size):
        kind = _pick(rng, kinds, kind_weights)
        genre = _genre_for(rng, kind)
        # Одно и то же слово может быть в банке нескольких жанров, поэтому название
        # обязано быть уникальным во всём каталоге: find_title() ищет по названию,
        # а дубль сделал бы поиск неоднозначным.
        title = None
        while title is None:
            pool = titles_left.get(genre)
            if pool is None:
                pool = list(_title_candidates(genre))
                rng.shuffle(pool)
                titles_left[genre] = pool
            if not pool:
                raise RuntimeError(f"исчерпан запас названий для жанра {genre!r}: увеличьте банки слов")
            candidate = pool.pop()
            if candidate not in used_titles:
                title = candidate
                used_titles.add(title)

        popularity = _sample_popularity(rng)
        quality = _sample_quality(rng, popularity)
        tone = _pick(rng, ("лёгкий", "нейтральный", "мрачный"), TONE_WEIGHTS.get(genre, (0.33, 0.34, 0.33)))
        minutes = _minutes(rng, kind)
        year_low, year_high = YEAR_RANGE_COURSE if kind == "course" else YEAR_RANGE_FILM_SERIES
        year = None if rng.random() < UNKNOWN_RATE_YEAR else rng.randint(year_low, year_high)

        seasons = episodes = None
        if kind == "series":
            if rng.random() >= UNKNOWN_RATE_SEASONS:
                seasons = min(SERIES_SEASONS[1], max(SERIES_SEASONS[0], round(rng.paretovariate(1.6) + SERIES_SEASONS[0] - 1)))
            if rng.random() >= UNKNOWN_RATE_EPISODES:
                per_season = rng.randint(*EPISODES_PER_SEASON)
                episodes = per_season * (seasons if seasons is not None else rng.randint(*SERIES_SEASONS))

        level = practical = None
        if kind == "course":
            level = None if rng.random() < UNKNOWN_RATE_LEVEL else _pick(rng, LEVELS, LEVEL_WEIGHTS)
            practical = None if rng.random() < UNKNOWN_RATE_PRACTICAL else rng.random() < 0.62

        unit = {"series": "серия", "film": "фильм", "course": "весь курс"}[kind]
        items.append(
            Item(
                id=f"it-{index + 1:05d}",
                title=title,
                kind=kind,
                genre=genre,
                tone=tone,
                seasons=seasons,
                episodes=episodes,
                minutes=minutes,
                level=level,
                practical=practical,
                year=year,
                quality=quality,
                popularity=popularity,
                description=(
                    f"Вымышленный {'сериал' if kind == 'series' else 'фильм' if kind == 'film' else 'учебный курс'}. "
                    f"Жанр: {genre}; тон: {tone}; {unit}: {minutes} мин. "
                    f"Данные синтетические, совпадения с реальными названиями случайны."
                ),
            )
        )
    return tuple(items)


def generate_catalog(seed: int = 42, size: int = CATALOG_SIZE) -> list[Item]:
    """Каталог D1. Одинаковые (seed, size) всегда дают одинаковый результат.

    Возвращает новый список, но сами объекты разделяются между вызовами (кэш).
    Объекты каталога трактуются как неизменяемые: ни один модуль их не мутирует,
    а копирование 3000 pydantic-моделей на каждый вызов съело бы время тестов.
    """
    if seed < 0:
        raise ValueError("seed не может быть отрицательным")
    if size < 1:
        raise ValueError("size должен быть не меньше 1")
    return list(_build(seed, size))


def catalog_sha256(seed: int = 42, size: int = CATALOG_SIZE) -> str:
    """Отпечаток каталога для проверки воспроизводимости между процессами."""
    payload = [i.model_dump(mode="json") for i in _build(seed, size)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def catalog_stats(items: list[Item]) -> dict[str, object]:
    """Сводка по каталогу: используется тестами калибровки и отчётом."""
    kinds = collections.Counter(i.kind for i in items)
    genres = collections.Counter(i.genre for i in items)
    qualities = sorted(i.quality for i in items)
    pops = sorted(i.popularity for i in items)

    def quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return values[min(len(values) - 1, max(0, int(len(values) * q)))]

    return {
        "size": len(items),
        "kinds": dict(kinds),
        "genres": dict(genres),
        "quality_median": round(quantile(qualities, 0.5), 4),
        "quality_mean": round(statistics.fmean(qualities), 4),
        "quality_share_ge_075": round(sum(1 for q in qualities if q >= 0.75) / len(qualities), 4),
        "quality_share_lt_025": round(sum(1 for q in qualities if q < 0.25) / len(qualities), 4),
        "popularity_median": quantile(pops, 0.5),
        "popularity_p90": quantile(pops, 0.90),
        "popularity_max": pops[-1],
        "unknown_seasons": sum(1 for i in items if i.kind == "series" and i.seasons is None),
        "unknown_practical": sum(1 for i in items if i.kind == "course" and i.practical is None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать синтетический каталог RecAgent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=CATALOG_SIZE)
    parser.add_argument("--output", default="data/catalog.json")
    args = parser.parse_args()

    items = generate_catalog(args.seed, args.size)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([i.model_dump(mode="json") for i in items], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сгенерировано {len(items)} объектов: {path}")
    print(f"sha256={catalog_sha256(args.seed, args.size)}")


if __name__ == "__main__":
    main()
