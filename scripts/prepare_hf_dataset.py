"""Create a provenance-tracked intent dataset with a deterministic group split.

The generated data is a pipeline smoke test. It is not evidence of model quality
and synthetic rows must not be presented as real user conversations.
"""

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from recagent.catalog import generate_catalog

CATALOG_TITLES = tuple(item.title for item in generate_catalog(42))
TEMPLATE_VERSION = "intent-templates-v1"
VALID_SPLITS = frozenset({"train", "validation"})
MIN_ROWS_PER_LABEL = 5

# A family is one language pattern. Slot values create bounded variation inside
# a family; they do not create new independent families.
TEMPLATES = {
    "discovery": [
        ("discovery_kind_genre", "Подбери {kind} {genre}"),
        ("discovery_today", "Что посмотреть сегодня: {kind} {genre}?"),
        ("discovery_soft", "Хочу {genre} без лишней мрачности ({kind})"),
    ],
    "similar": [
        ("similar_title", "Найди похожее на {title}"),
        ("similar_kind_title", "Что-то в духе этого {kind}: {title}"),
    ],
    "mood": [
        ("mood_evening", "Хочу лёгкое на вечер: {genre}, например {title}"),
        ("mood_recovery", "Подбери что-нибудь после тяжёлого дня — {kind} про {genre}, например {title}"),
    ],
    "navigation": [
        ("navigation_genre", "Покажи {kind} в разделе {genre}, например {title}"),
        ("navigation_title", "Найди в каталоге {title}"),
    ],
}


def normalize_text(text: str) -> str:
    """Return the canonical form used for duplicate detection."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _migration_message(path: str | Path) -> str:
    return (
        f"Вход {path} устарел или не содержит provenance генерации. Пересоздайте его командой "
        "python -m scripts.prepare_hf_dataset --output data/dialogues.jsonl "
        "--seed 42 --examples-per-label 40"
    )


def _stable_group_order(group_ids: set[str], seed: int) -> list[str]:
    return sorted(group_ids, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())


def assign_group_splits(rows: list[dict], *, seed: int = 42, validation_fraction: float = 0.25) -> list[dict]:
    """Assign whole template families to train or validation deterministically."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    by_label: dict[str, set[str]] = {}
    for row in rows:
        label = row["label"]
        by_label.setdefault(label, set()).add(row["template_family_id"])
    assignment: dict[str, str] = {}
    for label, groups in sorted(by_label.items()):
        if len(groups) < 2:
            raise ValueError(f"label {label!r} has fewer than 2 independent template families")
        ordered = _stable_group_order(groups, seed)
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_count = min(validation_count, len(ordered) - 1)
        for group in ordered[:validation_count]:
            assignment[group] = "validation"
        for group in ordered[validation_count:]:
            assignment[group] = "train"
    return [{**row, "split": assignment[row["template_family_id"]]} for row in rows]


def validate_rows(rows: list[dict], *, source: str = "input") -> dict:
    """Validate provenance and return the exact indices used by training."""

    if not rows:
        raise ValueError("dataset is empty")
    required = {"text", "label", "template_family_id", "seed", "template_version", "normalized_text_sha256", "split"}
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    family_labels: dict[str, str] = {}
    labels: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{_migration_message(source)}; row {index} is not an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{_migration_message(source)}; row {index} lacks {', '.join(sorted(missing))}")
        split = row["split"]
        if split == "final_holdout":
            raise ValueError("final_holdout is forbidden in LoRA training input; provide train/validation data only")
        if split not in VALID_SPLITS:
            raise ValueError(f"row {index} has unsupported split {split!r}; expected train or validation")
        text = row["text"]
        normalized = normalize_text(text)
        if text in seen_exact:
            raise ValueError(f"exact duplicate text at row {index}")
        if normalized in seen_normalized:
            raise ValueError(f"normalized duplicate text at row {index}")
        seen_exact.add(text)
        seen_normalized.add(normalized)
        if row["template_version"] != TEMPLATE_VERSION:
            raise ValueError(f"row {index} uses unsupported template_version {row['template_version']!r}")
        expected_hash = normalized_text_sha256(text)
        if row["normalized_text_sha256"] != expected_hash:
            raise ValueError(f"normalized text hash mismatch at row {index}")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise ValueError(f"row {index} has invalid generation seed")
        family = row["template_family_id"]
        if not isinstance(family, str) or not family:
            raise ValueError(f"row {index} has empty template_family_id")
        if not isinstance(row["label"], str) or not row["label"]:
            raise ValueError(f"row {index} has empty label")
        if family in family_labels and family_labels[family] != row["label"]:
            raise ValueError(f"template family {family!r} is used by multiple labels")
        family_labels[family] = row["label"]
        labels.add(row["label"])

    if len(labels) < 2:
        raise ValueError("meaningful classifier split requires at least 2 labels")
    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    validation_indices = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    if not train_indices or not validation_indices:
        raise ValueError("meaningful group split requires non-empty train and validation sets")
    train_families = {rows[i]["template_family_id"] for i in train_indices}
    validation_families = {rows[i]["template_family_id"] for i in validation_indices}
    overlap = train_families & validation_families
    if overlap:
        raise ValueError(f"template-family overlap detected: {sorted(overlap)}")
    for label in sorted(labels):
        all_label_rows = [row for row in rows if row["label"] == label]
        label_families = {row["template_family_id"] for row in all_label_rows}
        if len(all_label_rows) < MIN_ROWS_PER_LABEL:
            raise ValueError(f"label {label!r} has fewer than {MIN_ROWS_PER_LABEL} rows")
        if len(label_families) < 2:
            raise ValueError(f"label {label!r} has fewer than 2 independent template families")
        if not any(rows[i]["label"] == label for i in train_indices):
            raise ValueError(f"label {label!r} is absent from train split")
        if not any(rows[i]["label"] == label for i in validation_indices):
            raise ValueError(f"label {label!r} is absent from validation split")
    return {
        "rows": rows,
        "labels": sorted(labels),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_groups": sorted(train_families),
        "validation_groups": sorted(validation_families),
        "counts": {
            "rows": len(rows),
            "train": len(train_indices),
            "validation": len(validation_indices),
            "groups": len(family_labels),
            "train_groups": len(train_families),
            "validation_groups": len(validation_families),
        },
    }


validate_dataset = validate_rows


def _render(spec: str, rng: random.Random) -> str:
    return spec.format(
        kind=rng.choice(("сериал", "фильм", "курс")),
        genre=rng.choice(("детектив", "комедию", "фантастику", "машинному обучению", "python")),
        title=rng.choice(CATALOG_TITLES),
    )


def build(seed: int = 42, examples_per_label: int = 40, validation_fraction: float = 0.25) -> list[dict]:
    """Generate unique, provenance-tracked rows and assign a group split."""

    if examples_per_label < MIN_ROWS_PER_LABEL:
        raise ValueError(f"Use at least {MIN_ROWS_PER_LABEL} examples per label")
    rng = random.Random(seed)
    rows: list[dict] = []
    candidate_seen: set[str] = set()
    for label, specs in TEMPLATES.items():
        family_rows: dict[str, list[dict]] = {family: [] for family, _ in specs}
        family_seen: dict[str, set[str]] = {family: set() for family, _ in specs}
        family_target = (examples_per_label + len(specs) - 1) // len(specs)
        for family, template in specs:
            attempts = 0
            while len(family_rows[family]) < family_target and attempts < examples_per_label * 100:
                attempts += 1
                text = _render(template, rng)
                normalized = normalize_text(text)
                if normalized in family_seen[family] or normalized in candidate_seen:
                    continue
                family_seen[family].add(normalized)
                candidate_seen.add(normalized)
                family_rows[family].append(
                    {
                        "text": text,
                        "label": label,
                        "synthetic": True,
                        "seed": seed,
                        "template_family_id": family,
                        "template_version": TEMPLATE_VERSION,
                        "normalized_text_sha256": normalized_text_sha256(text),
                    }
                )
            if not family_rows[family]:
                raise ValueError(f"template family {family!r} cannot produce unique rows")
        family_names = [family for family, _ in specs]
        for offset in range(examples_per_label):
            family = family_names[offset % len(family_names)]
            family_offset = offset // len(family_names)
            if family_offset >= len(family_rows[family]):
                raise ValueError(f"template family {family!r} produced {len(family_rows[family])} unique rows; need {family_target}")
            row = family_rows[family][family_offset]
            rows.append(row)
    rng.shuffle(rows)
    rows = assign_group_splits(rows, seed=seed, validation_fraction=validation_fraction)
    validate_rows(rows, source="generated dataset")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dialogues.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-label", type=int, default=40)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.examples_per_label < MIN_ROWS_PER_LABEL:
        parser.error(f"Use at least {MIN_ROWS_PER_LABEL} examples per label")
    try:
        rows = build(args.seed, args.examples_per_label, args.validation_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary = validate_rows(rows, source=path)
    print(
        json.dumps(
            {"status": "validated", "output": str(path), "counts": summary["counts"], "labels": summary["labels"]}, ensure_ascii=False
        )
    )
