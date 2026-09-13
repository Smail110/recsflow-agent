"""Create an auditable intent dataset. Demo mode is synthetic and is not a training claim."""

import argparse
import json
import random
from pathlib import Path

from recagent.catalog import generate_catalog

# Названия берутся из каталога, а не из строки: каталог перегенерируется, и
# зашитое название перестало бы существовать, оставив датасет с мёртвой ссылкой.
CATALOG_TITLES = tuple(i.title for i in generate_catalog(42))

TEMPLATES = {
    "discovery": ["Подбери {kind} {genre}", "Что посмотреть сегодня?", "Хочу {genre} без лишней мрачности"],
    "similar": ["Найди похожее на {title}", "Что-то в духе этого сериала"],
    "mood": ["Хочу лёгкое на вечер", "Подбери что-нибудь после тяжёлого дня"],
    "navigation": ["Покажи в каталоге {genre}", "Найди курс по {genre}"],
}


def build(seed=42, examples_per_label=40):
    rng = random.Random(seed)
    genres = ["детектив", "комедию", "фантастику", "машинному обучению", "python"]
    kinds = ["сериал", "фильм", "курс"]
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(examples_per_label):
            template = rng.choice(templates)
            text = template.format(kind=rng.choice(kinds), genre=rng.choice(genres), title=rng.choice(CATALOG_TITLES))
            rows.append({"text": text, "label": label, "synthetic": True, "seed": seed})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dialogues.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-label", type=int, default=40)
    args = parser.parse_args()
    if args.examples_per_label < 5:
        parser.error("Use at least 5 examples per label")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.seed, args.examples_per_label)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic rows to {path}. This is a dry-run dataset, not real user data.")
