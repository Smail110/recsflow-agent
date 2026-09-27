"""Data and measurement checks for the optional tiny evidence experiment."""

import json

import pytest
from scripts.train_slot_evidence_tiny import (
    LABELS,
    classification_metrics,
    majority_baseline,
    percentile,
    read_rows,
    validate_splits,
)


def row(identifier, premise, hypothesis, label, family):
    return {"id": identifier, "premise": premise, "hypothesis": hypothesis, "label": label, "family": family}


TRAIN = [
    row("t1", "Ищу лёгкий фильм", "Нужен лёгкий тон фильма", "support", "tone-positive"),
    row("t2", "Не хочу драму", "Пользователь требует драму", "contradiction", "genre-negation"),
    row("t3", "Порекомендуй кино", "Нужен курс Python", "unknown", "topic-neutral"),
]
DEV = [
    row("d1", "Фильм с юмором", "Нужна комедия", "support", "genre-positive"),
    row("d2", "Курс без практики", "Практика обязательна", "contradiction", "practice-negation"),
    row("d3", "Что посмотреть вечером", "Нужен сериал", "unknown", "kind-neutral"),
]


def test_jsonl_validates_labels_ids_and_family(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in TRAIN) + "\n", encoding="utf-8")
    assert read_rows(path) == TRAIN
    assert validate_splits(TRAIN, DEV)["family_check"] == "passed"

    path.write_text(json.dumps({**TRAIN[0], "label": "entailed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        read_rows(path)


def test_rejects_reused_premise_even_when_hypotheses_differ():
    duplicate = row("d4", "  ИЩУ лёгкий  фильм  ", "Не нужен лёгкий тон", "contradiction", "other-family")
    with pytest.raises(ValueError, match="premise overlap"):
        validate_splits(TRAIN, [*DEV, duplicate])


def test_rejects_reused_family_and_missing_train_class():
    with pytest.raises(ValueError, match="family overlap"):
        validate_splits(TRAIN, [*DEV, row("d4", "Нужно ещё что-то", "Нужен фильм", "unknown", "tone-positive")])
    with pytest.raises(ValueError, match="misses labels"):
        validate_splits(TRAIN[:2], DEV)


def test_per_class_confusion_and_majority_tie_are_deterministic():
    gold = list(LABELS)
    predicted = ["support", "support", "unknown"]
    metrics = classification_metrics(gold, predicted)
    assert metrics["confusion_actual_by_predicted"]["contradiction"]["support"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["contradiction"]["recall"] == 0
    assert majority_baseline(TRAIN, DEV)["label"] == "support"


def test_latency_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        percentile([], 0.5)
