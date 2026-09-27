"""Аудит сохранённого dev baseline и расчёт размеров независимой выборки."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist

from evals.metrics import clustered_success_interval, wilson_interval


def sample_plan(delta=0.05, discordance=0.20, power=0.80, confidence=0.95, half_width=0.03):
    if not 0 < delta <= discordance <= 1 or not 0.5 < power < 1 or not 0 < confidence < 1 or not 0 < half_width < 0.5:
        raise ValueError("invalid sample planning assumptions")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    beta = NormalDist().inv_cdf(power)
    # Normal approximation to a paired binary difference. Discordance is a
    # planning assumption, not inferred from an identical-predictions pilot.
    paired = math.ceil((z * math.sqrt(discordance) + beta * math.sqrt(discordance - delta**2)) ** 2 / delta**2)
    # Worst-case Wilson half-width is attained at p=0.5.
    precision = math.ceil(z**2 / (4 * half_width**2) - z**2)
    return {
        "mde": delta,
        "discordance_assumption": discordance,
        "power": power,
        "confidence": confidence,
        "half_width": half_width,
        "paired_independent_units": paired,
        "wilson_independent_units": precision,
        "required_independent_units": max(paired, precision),
        "limitations": [
            "Нормальная аппроксимация мощности; окончательная проверка симуляцией до freeze.",
            "Нужны независимые пользователи. Повторы шаблонов не создают новые языковые семьи.",
            "При кластеризации число строк умножают на design effect; ICC пока неизвестна.",
        ],
    }


def audit(report):
    rows = report["runs"]["current"]["records"]
    texts = [tuple(turn["utterance"].casefold().replace("ё", "е").strip() for turn in row["turns"]) for row in rows]
    users = Counter(row["user_id"] for row in rows)
    return {
        "source": report["source"],
        "dataset": report["dataset"],
        "scenarios": len(rows),
        "unique_users": len(users),
        "max_scenarios_per_user": max(users.values()),
        "exact_duplicate_dialogues": len(texts) - len(set(texts)),
        "single_turn": sum(len(row["turns"]) == 1 for row in rows),
        "multi_turn": sum(len(row["turns"]) > 1 for row in rows),
        "history_scenarios": 0 if report["configuration"]["history"] == "cold_start" else None,
        "final_success_cluster_interval": clustered_success_interval(rows, field="final_success"),
        "full_success_cluster_interval": clustered_success_interval(rows),
        "coverage": [
            {
                "family": family,
                "count": stats["scenarios"],
                "share": stats["scenarios"] / len(rows),
                "success_rate": stats["final_success_rate"],
                "quality": "мало наблюдений" if stats["scenarios"] < 30 else "требуется языковой аудит",
                "target_independent_units_for_10pp_half_width": sample_plan(half_width=0.1)["wilson_independent_units"],
            }
            for family, stats in report["runs"]["current"]["by_kind"].items()
        ],
        "planning_sensitivity": [sample_plan(discordance=q) for q in (0.1, 0.2, 0.3, 0.5)],
        "stratum_precision_n": sample_plan(half_width=0.1)["wilson_independent_units"],
        "wilson_illustration_4_of_4": wilson_interval(4, 4),
        "limitations": [
            "Итоговые интервалы условны на синтетическом каталоге и существующих шаблонах.",
            "Страты могут пересекаться; их минимальные размеры нельзя механически складывать.",
            "Не запускать старый holdout как новую независимую оценку.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.baseline.read_text(encoding="utf-8"))
    if report["configuration"]["split"] != "dev":
        parser.error("аудит ошибок допускается только на dev")
    result = audit(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scenarios",
                    "unique_users",
                    "exact_duplicate_dialogues",
                    "final_success_cluster_interval",
                    "stratum_precision_n",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
