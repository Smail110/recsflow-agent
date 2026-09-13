"""Synthetic data only. No real titles, ratings or user histories are implied."""
import argparse
import json
import random
from pathlib import Path

from .models import Item


def generate_catalog(seed: int = 42) -> list[Item]:
    rng = random.Random(seed)
    items = []
    themes = {
        "детектив": ["Тайна старого маяка", "Дело о пропавшем чае", "Клуб тихих расследований", "Секреты набережной", "Улики в саду", "Последний конверт"],
        "комедия": ["Соседи по пятнице", "План на каникулы", "Почти отпуск", "Кофе для двоих", "Не тот чемодан", "Выходной в городе"],
        "драма": ["Письма домой", "Другая сторона реки", "После дождя", "Дальний берег", "Точка встречи", "Открытое окно"],
        "фантастика": ["Орбита тишины", "Время на двоих", "Архив будущего", "Планета за окном", "Сигнал с Венеры", "Вторая Земля"],
        "приключения": ["Карта ветров", "По следам лета", "Остров фонарей", "Путь к вершине", "Северный экспресс", "За горизонтом"],
    }
    for genre, titles in themes.items():
        for index, title in enumerate(titles):
            for kind in ("series", "film"):
                tone = "лёгкий" if index < 4 else ("мрачный" if index == 4 else "нейтральный")
                items.append(Item(
                    id=f"demo-{len(items)+1:03d}", title=title + (" · фильм" if kind == "film" else ""),
                    kind=kind, genre=genre, tone=tone,
                    seasons=(1 if index < 4 else 3) if kind == "series" else None,
                    episodes=rng.choice([6, 8, 10]) if kind == "series" else None,
                    minutes=rng.choice([25, 30, 40]) if kind == "series" else rng.choice([80, 90, 110]),
                    quality=round(rng.uniform(.55, .98), 3),
                    description=f"Вымышленный {'сериал' if kind == 'series' else 'фильм'} для проверки рекомендаций. Жанр: {genre}; тон: {tone}.",
                ))
    for genre in ("машинное обучение", "python"):
        for index in range(12):
            items.append(Item(
                id=f"demo-{len(items)+1:03d}", title=f"{genre.capitalize()}: лаборатория {index+1}",
                kind="course", genre=genre, tone="нейтральный", minutes=rng.choice([60, 90, 120, 180]),
                level="начальный" if index < 8 else "продвинутый", practical=index % 3 != 2,
                quality=round(rng.uniform(.55, .98), 3),
                description="Вымышленный учебный курс. Длительность — суммарное время занятий.",
            ))
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/catalog.json")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([i.model_dump() for i in generate_catalog(args.seed)], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(generate_catalog(args.seed))} synthetic items: {path}")

