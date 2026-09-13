"""Пересчёт констант калибровки каталога из сырых данных MovieLens-100k.

Зачем этот скрипт существует. ``recagent/catalog/calibration.py`` хранит числа,
от которых зависит форма синтетического каталога: доли жанров, распределение
оценок, квантили популярности и наклон зависимости «рейтинг от log2(популярности)».
Числа в файле были получены вручную, и без пересчёта они непроверяемы: любой
может справедливо спросить, откуда 0.4310 у драмы. Скрипт делает их
воспроизводимыми и добавляет тест, который сравнивает пересчитанное с
заявленным (``tests/unit/test_calibration.py``).

Данные: MovieLens-100k (apache-2.0), зеркало HuggingFace ``includeno/movielens-100k``.
Файлы кладутся вручную в ``data/raw/`` — из сети агента grouplens.org недоступен,
поэтому скрипт не качает ничего сам и падает с внятным сообщением, если файлов нет.
``data/raw/`` в .gitignore: реальные данные не коммитим, только производные константы.

В каталог не попадает ни одного реального названия: из MovieLens берутся
исключительно распределения.

Запуск:
    python -m scripts.calibrate_catalog                 # сравнение с calibration.py
    python -m scripts.calibrate_catalog --output data/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from recagent.catalog import calibration as C

RAW_DIR: Final = Path("data/raw")
U_ITEM: Final = "u.item"
U_DATA: Final = "u.data"

# Раскладка u.item: id|title|release|video_release|imdb_url|<19 флагов жанров>.
# Индексы флагов идут подряд с 5-го; здесь только те, что участвуют в калибровке.
GENRE_COLUMN: Final[dict[str, int]] = {
    "драма": 13,  # Drama
    "комедия": 10,  # Comedy
    "детектив": (18, 11),  # Mystery + Crime: нашему «детективу» соответствуют оба
    "приключения": 7,  # Adventure
    "фантастика": 20,  # Sci-Fi
}

# u.item в MovieLens-100k лежит в ISO-8859-1, не в UTF-8. Чтение в utf-8 упало бы
# на первом же акцентированном названии, а ошибки нам не нужны: названия мы не
# используем, но падать на кодировке было бы глупо.
ITEM_ENCODING: Final = "iso-8859-1"

QUANTILE_POINTS: Final[tuple[float, ...]] = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
RATING_VALUES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Calibration:
    """Всё, что генератор каталога берёт из реальных данных."""

    n_items: int
    n_ratings: int
    genre_shares: dict[str, float]
    rating_shares: tuple[float, ...]
    rating_sd_within_item: float
    popularity_quantiles: tuple[tuple[float, int], ...]
    quality_points: tuple[tuple[float, float], ...]
    quality_intercept: float
    quality_log2_slope: float
    quality_between_item_sd: float
    quality_observed_variance: float
    quality_sampling_variance: float


def require_raw_files(raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    items_path, ratings_path = raw_dir / U_ITEM, raw_dir / U_DATA
    missing = [str(p) for p in (items_path, ratings_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Не найдены исходники MovieLens-100k: " + ", ".join(missing) + ".\n"
            "Скачайте набор (apache-2.0, зеркало https://huggingface.co/datasets/includeno/movielens-100k)\n"
            f"и положите {U_ITEM} и {U_DATA} в {raw_dir}/. Скрипт ничего не качает сам:\n"
            "исходные данные не коммитятся, в репозитории живут только производные константы."
        )
    return items_path, ratings_path


def load_item_genres(path: Path) -> list[set[int]]:
    """Множества жанров по объектам. Порядок строк = порядок movie_id, но он не важен."""
    genres: list[set[int]] = []
    with path.open(encoding=ITEM_ENCODING) as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("|")
            flags = {index for index in range(5, len(fields)) if fields[index] == "1"}
            genres.append(flags)
    return genres


def load_ratings(path: Path) -> dict[int, list[int]]:
    """Оценки по объектам: movie_id -> [rating, ...]. Формат u.data: user id item rating ts."""
    by_item: dict[int, list[int]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            _, item_id, rating, _ = line.split("\t")[:4]
            by_item[int(item_id)].append(int(rating))
    return dict(by_item)


def genre_shares(genres: Sequence[set[int]]) -> dict[str, float]:
    """Доля объектов жанра. Жанров у объекта может быть несколько, поэтому суммы > 1."""
    total = len(genres)
    shares: dict[str, float] = {}
    for name, columns in GENRE_COLUMN.items():
        if isinstance(columns, int):
            hits = sum(1 for flags in genres if columns in flags)
        else:
            hits = sum(1 for flags in genres if any(column in flags for column in columns))
        shares[name] = round(hits / total, 4)
    return shares


def rating_shares(ratings: Sequence[int]) -> tuple[float, ...]:
    counts = Counter(ratings)
    total = len(ratings)
    return tuple(round(counts.get(value, 0) / total, 4) for value in RATING_VALUES)


def within_item_rating_sd(by_item: dict[int, list[int]]) -> float:
    """Пулированное стандартное отклонение отдельной оценки вокруг среднего своего объекта.

    Именно эта величина нужна генератору: наблюдаемый рейтинг объекта — это среднее
    n оценок, поэтому шум среднего равен sd / sqrt(n). Пулируем по всем объектам
    (делим на общее число оценок, а не усредняем sd объектов), чтобы объекты с двумя
    оценками не перевешивали объекты с тремястами.
    """
    total_sq = 0.0
    total_n = 0
    for values in by_item.values():
        if len(values) < 2:
            # Одна оценка не даёт отклонения; её вклад в сумму квадратов нулевой,
            # но в знаменатель она всё равно входит — иначе sd будет завышен.
            total_n += len(values)
            continue
        mean = statistics.fmean(values)
        total_sq += sum((value - mean) ** 2 for value in values)
        total_n += len(values)
    if not total_n:
        return 0.0
    return round(math.sqrt(total_sq / total_n), 4)


def _quantile(sorted_values: Sequence[int], q: float) -> int:
    """Квантиль ближайшего ранга. При q=1.0 даёт максимум, при q=0.0 — минимум."""
    if not sorted_values:
        return 0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    index = math.ceil(q * len(sorted_values)) - 1
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def popularity_quantiles(
    by_item: dict[int, list[int]], n_items: int, points: Sequence[float] = QUANTILE_POINTS
) -> tuple[tuple[float, int], ...]:
    """Квантили «сколько оценок у объекта», включая объекты без оценок (ноль не берём:
    генератор работает в лог-шкале, а log(0) не определён; минимум — одна оценка)."""
    counts = sorted(len(by_item.get(item_id, [])) for item_id in range(1, n_items + 1))
    counts = [max(1, count) for count in counts]
    return tuple((round(point, 2), _quantile(counts, point)) for point in points)


def _mean_rating_by_log2_bucket(by_item: dict[int, list[int]]) -> tuple[tuple[float, float], ...]:
    """Средний рейтинг объекта в бакете по округлённому log2(числа оценок)."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for values in by_item.values():
        if not values:
            continue
        buckets[round(math.log2(len(values)))].append(statistics.fmean(values))
    return tuple((float(key), round(statistics.fmean(buckets[key]), 4)) for key in sorted(buckets))


def _least_squares(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """(intercept, slope) обычного МНК. numpy не подключаем: точек меньше десяти."""
    n = len(points)
    if n < 2:
        return (points[0][1] if points else 0.0), 0.0
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return round(mean_y - slope * mean_x, 4), round(slope, 4)


def between_item_quality_sd(by_item: dict[int, list[int]], within_sd: float) -> tuple[float, float, float]:
    """Разброс ИСТИННОГО качества объекта, очищенный от шума выборки.

    Возвращает (sigma_between, V_observed, V_sampling).

    Почему это нужно считать, а не подбирать. Наблюдаемая дисперсия средних по
    объектам складывается из двух частей::

        V(mean_i) = sigma_between^2 + E[sw^2 / n_i]

    Первое — насколько объекты действительно разные по качеству (это то, что
    генератор должен воспроизвести через QUALITY_TRUE_SD), второе — шум конечной
    выборки (его генератор добавляет сам как sw / sqrt(n)). Если взять sigma из
    V(mean_i) целиком, дисперсия будет удвоена; если взять её «на глаз», хвосты
    распределения качества окажутся тоньше реальных, и оракул из P2 будет мерить
    систему на каталоге, где почти всё «средненькое».

    Метод — вычитание (method of moments), не оценка максимального правдоподобия:
    объектов 1682, оценка устойчива, а прозрачность здесь важнее эффективности.
    """
    means = [statistics.fmean(values) for values in by_item.values() if values]
    if len(means) < 2:
        return 0.0, 0.0, 0.0
    observed = statistics.pvariance(means)
    sampling = statistics.fmean(
        within_sd**2 / len(values) for values in by_item.values() if values
    )
    return round(math.sqrt(max(0.0, observed - sampling)), 4), round(observed, 4), round(sampling, 4)


def calibrate(raw_dir: Path = RAW_DIR) -> Calibration:
    items_path, ratings_path = require_raw_files(raw_dir)
    genres = load_item_genres(items_path)
    by_item = load_ratings(ratings_path)
    flat = [rating for values in by_item.values() for rating in values]
    points = _mean_rating_by_log2_bucket(by_item)
    intercept, slope = _least_squares(points)
    within_sd = within_item_rating_sd(by_item)
    sigma_between, observed, sampling = between_item_quality_sd(by_item, within_sd)
    return Calibration(
        n_items=len(genres),
        n_ratings=len(flat),
        genre_shares=genre_shares(genres),
        rating_shares=rating_shares(flat),
        rating_sd_within_item=within_sd,
        popularity_quantiles=popularity_quantiles(by_item, len(genres)),
        quality_points=points,
        quality_intercept=intercept,
        quality_log2_slope=slope,
        quality_between_item_sd=sigma_between,
        quality_observed_variance=observed,
        quality_sampling_variance=sampling,
    )


def to_json(result: Calibration) -> dict[str, object]:
    return {
        "source": "MovieLens-100k (apache-2.0), mirror huggingface.co/datasets/includeno/movielens-100k",
        "n_items": result.n_items,
        "n_ratings": result.n_ratings,
        "genre_shares": result.genre_shares,
        "rating_shares": list(result.rating_shares),
        "rating_sd_within_item": result.rating_sd_within_item,
        "popularity_quantiles": [{"q": q, "n": n} for q, n in result.popularity_quantiles],
        "quality_points": [{"log2_n": x, "mean_rating": y} for x, y in result.quality_points],
        "quality_intercept": result.quality_intercept,
        "quality_log2_slope": result.quality_log2_slope,
        "quality_observed_variance": result.quality_observed_variance,
        "quality_sampling_variance": result.quality_sampling_variance,
        "quality_between_item_sd": result.quality_between_item_sd,
    }


def diff_against_module(result: Calibration) -> list[dict[str, object]]:
    """Расхождения пересчитанного с тем, что заявлено в calibration.py.

    Возвращается список, а не исключение: расхождение — это информация для человека,
    а не ошибка выполнения. Допуски разные: доли жанров и оценок округлены до 4 знаков
    и обязаны совпасть почти точно, а интерцепт/наклон получены другой процедурой
    (МНК по бакетам против снятых вручную точек), поэтому допуск шире.
    """
    rows: list[dict[str, object]] = []

    def add(name: str, declared: object, computed: object, tolerance: float) -> None:
        delta = abs(float(declared) - float(computed))  # type: ignore[arg-type]
        rows.append(
            {
                "constant": name,
                "declared": declared,
                "computed": computed,
                "delta": round(delta, 6),
                "tolerance": tolerance,
                "ok": delta <= tolerance,
            }
        )

    for genre, declared in C.GENRE_SHARE_MOVIELENS.items():
        add(f"GENRE_SHARE_MOVIELENS[{genre!r}]", declared, result.genre_shares.get(genre, 0.0), 0.005)
    for index, declared in enumerate(C.RATING_SHARE):
        add(f"RATING_SHARE[{index}]", declared, result.rating_shares[index], 0.005)
    add("RATING_SD_WITHIN_ITEM", C.RATING_SD_WITHIN_ITEM, result.rating_sd_within_item, 0.05)
    for (dq, dn), (cq, cn) in zip(C.POPULARITY_QUANTILES, result.popularity_quantiles, strict=True):
        if dq != cq:
            raise AssertionError(f"сетка квантилей разошлась: {dq} != {cq}")
        add(f"POPULARITY_QUANTILES[{dq}]", dn, cn, max(1.0, 0.10 * dn))
    add("QUALITY_INTERCEPT", C.QUALITY_INTERCEPT, result.quality_intercept, 0.25)
    add("QUALITY_LOG2_SLOPE", C.QUALITY_LOG2_SLOPE, result.quality_log2_slope, 0.06)
    add("QUALITY_TRUE_SD", C.QUALITY_TRUE_SD, result.quality_between_item_sd, 0.05)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Пересчитать константы калибровки каталога из MovieLens-100k")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=None, help="куда сохранить JSON со всеми числами")
    args = parser.parse_args()

    result = calibrate(args.raw_dir)
    print(f"MovieLens-100k: {result.n_items} объектов, {result.n_ratings} оценок")
    print(f"Жанры: {json.dumps(result.genre_shares, ensure_ascii=False)}")
    print(f"Оценки 1..5: {list(result.rating_shares)}")
    print(f"SD оценки вокруг среднего объекта: {result.rating_sd_within_item}")
    print(f"Квантили популярности: {[(q, n) for q, n in result.popularity_quantiles]}")
    print(f"Рейтинг от log2(n): intercept={result.quality_intercept} slope={result.quality_log2_slope}")
    print(
        f"Дисперсия средних: всего {result.quality_observed_variance}, шум выборки {result.quality_sampling_variance}, "
        f"между объектами sd={result.quality_between_item_sd}"
    )

    rows = diff_against_module(result)
    width = max(len(str(row["constant"])) for row in rows)
    print("\nСверка с recagent/catalog/calibration.py:")
    for row in rows:
        mark = "OK " if row["ok"] else "РАЗ"
        print(
            f"  [{mark}] {row['constant']!s:<{width}}  заявлено={row['declared']:<8} пересчитано={row['computed']:<8} Δ={row['delta']} (допуск {row['tolerance']})"
        )
    bad = [row for row in rows if not row["ok"]]
    print(f"\nРасхождений сверх допуска: {len(bad)} из {len(rows)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(to_json(result), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
