"""Сверка констант калибровки с фактическими данными MovieLens-100k.

Зачем. ``recagent/catalog/calibration.py`` — единственное место, где заявлено, что
синтетический каталог откалиброван по реальным распределениям. Без проверки это
просто утверждение в комментарии. Тест делает его проверяемым: пересчитывает каждую
константу из ``data/raw/`` и сверяет с заявленной, а отдельно проверяет, что
генератор эти константы действительно применил (можно хранить правильные числа и
при этом их не использовать).

Тесты пропускаются, если исходников нет: ``data/raw/`` в .gitignore, потому что
реальные данные не коммитятся — в репозитории живут только производные константы.
В CI это обычно skip, локально и в приёмке — реальная проверка. Команда, которой
восстанавливаются исходники, написана в докстринге ``calibration.py`` и в
``docs/DATA-PLAN.md``.

Пороги в ``test_catalog_reproduces_calibrated_shape`` взяты не «на глаз», а из
тех же данных: доля объектов с качеством >= 0.75 в MovieLens равна 0.1046,
с качеством < 0.25 — 0.0767. Допуск 0.05 п.п. с запасом покрывает разницу между
1682 реальными объектами и 3000 синтетическими, в которые добавлены курсы и
сериалы (их в MovieLens нет).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from recagent.catalog import calibration as C

_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not (_RAW / "u.item").exists() or not (_RAW / "u.data").exists(),
    reason="нужны data/raw/u.item и data/raw/u.data (MovieLens-100k); см. docs/DATA-PLAN.md",
)

# Эталон из MovieLens-100k, посчитанный тем же скриптом калибровки.
ML_SHARE_QUALITY_GE_075 = 0.1046
ML_SHARE_QUALITY_LT_025 = 0.0767
ML_POPULARITY_P90_OVER_MEDIAN = 169 / 27


@pytest.fixture(scope="module")
def calibration():
    from scripts.calibrate_catalog import calibrate

    return calibrate()


def test_no_constant_drifts_from_source_data(calibration):
    """Ни одна константа не разошлась с данными сверх допуска.

    Один тест на всю сверку нарочно: ``diff_against_module`` возвращает все строки,
    поэтому падение показывает сразу каждое съехавшее число, а не первое попавшееся.
    """
    from scripts.calibrate_catalog import diff_against_module

    rows = diff_against_module(calibration)
    bad = [row for row in rows if not row["ok"]]
    detail = "\n".join(
        f"  {row['constant']}: заявлено={row['declared']} пересчитано={row['computed']} delta={row['delta']} (допуск {row['tolerance']})"
        for row in bad
    )
    assert not bad, f"константы калибровки разошлись с MovieLens-100k:\n{detail}"


def test_source_dataset_is_the_declared_one(calibration):
    """Сверяем именно тот набор, который назван в докстринге, а не похожий.

    Без этой проверки тест молча прошёл бы на ml-latest-small или ml-25m, где
    другие распределения, и «калибровка по MovieLens-100k» стала бы неправдой.
    """
    assert calibration.n_items == 1682
    assert calibration.n_ratings == 100_000


def test_variance_decomposition_is_consistent(calibration):
    """sigma_between^2 + шум выборки дают наблюдаемую дисперсию средних.

    Это проверка самой декомпозиции, из которой получен QUALITY_TRUE_SD. Без неё
    подстановка 0.6691 была бы просто числом из вывода скрипта.
    """
    assert calibration.quality_observed_variance == pytest.approx(
        calibration.quality_between_item_sd**2 + calibration.quality_sampling_variance, abs=1e-3
    )
    # Сравнение дисперсии с дисперсией, а не sd с дисперсией: sigma^2 = V_obs - V_noise,
    # поэтому sigma^2 < V_obs, но сама sigma может быть больше V_obs (единицы разные).
    assert 0 < calibration.quality_between_item_sd**2 < calibration.quality_observed_variance
    assert calibration.quality_sampling_variance > 0


def test_popularity_is_heavy_tailed(calibration):
    """Длинный хвост обязателен: p90 превышает медиану примерно в шесть раз.

    Если бы хвост исчез, популярность перестала бы нести информацию о качестве,
    а вместе с ней пропал бы сигнал, который реранкер из P5 должен выучить.
    """
    quantiles = dict(calibration.popularity_quantiles)
    ratio = quantiles[0.90] / quantiles[0.50]
    assert ratio > 5
    assert quantiles[1.00] > quantiles[0.99] > quantiles[0.75] > quantiles[0.50]


def test_quality_grows_with_popularity(calibration):
    """Популярное в среднем rated выше — зависимость обязана быть положительной."""
    assert calibration.quality_log2_slope > 0
    points = dict(calibration.quality_points)
    assert points[max(points)] > points[min(points)]


def test_catalog_reproduces_calibrated_shape():
    """Генератор применил константы: форма каталога совпадает с формой MovieLens.

    Проверка на другом конце цепочки. Хранить верные числа и не использовать их —
    обычная ошибка, и тесты выше её не ловят.
    """
    from recagent.catalog import catalog_stats, generate_catalog

    stats = catalog_stats(generate_catalog(42))
    assert stats["size"] == C.CATALOG_SIZE

    # Хвост популярности: отношение p90 к медиане то же, что в MovieLens (6.26).
    assert stats["popularity_p90"] / stats["popularity_median"] == pytest.approx(ML_POPULARITY_P90_OVER_MEDIAN, rel=0.15)

    # Хвосты качества: доли объектов на краях шкалы совпадают с MovieLens.
    assert stats["quality_share_ge_075"] == pytest.approx(ML_SHARE_QUALITY_GE_075, abs=0.05)
    assert stats["quality_share_lt_025"] == pytest.approx(ML_SHARE_QUALITY_LT_025, abs=0.05)
    assert stats["quality_mean"] == pytest.approx(0.519, abs=0.05)

    # Неполнота каталога воспроизводится: необязательные атрибуты бывают null.
    # Это нужно не для красоты, а чтобы null-семантика («нет данных», а не «совпало»)
    # реально отрабатывалась в ранжировании и объяснениях.
    assert stats["unknown_seasons"] > 0
    assert stats["unknown_practical"] > 0


def test_genres_are_present_in_generated_catalog():
    """Каждый жанр из MovieLens-калибровки представлен в каталоге.

    При редком жанре (фантастика 0.06) и 3000 объектов жанр обязан встречаться
    десятки раз; иначе диалог про фантастику был бы неотличим от пустой выдачи.
    """
    from recagent.catalog import generate_catalog

    counts: dict[str, int] = {}
    for item in generate_catalog(42):
        counts[item.genre] = counts.get(item.genre, 0) + 1
    # Знаменатель — фильмы и сериалы: у курсов свои жанры, их калибровка отдельная.
    film_series = sum(counts.get(g, 0) for g in C.GENRE_SHARE_MOVIELENS)
    for genre, share in C.GENRE_SHARE_MOVIELENS.items():
        assert counts.get(genre, 0) >= 50, f"жанр {genre!r} встречается {counts.get(genre, 0)} раз"
        # Допуск не произвольный, а четыре стандартных ошибки биномиальной доли:
        # sd = sqrt(p(1-p)/n). Для комедии это 0.0091, то есть допуск 0.036.
        # Расхождение сверх четырёх сигм означало бы, что генератор не применил
        # GENRE_SHARE_MOVIELENS, а не то, что конкретно этот seed дал такой хвост.
        tolerance = 4 * math.sqrt(share * (1 - share) / film_series)
        assert abs(counts[genre] / film_series - share) < tolerance, genre
