"""Checks for synthetic slot-evidence conversion and split isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.build_slot_evidence_dataset import build, convert_row


def _row(split: str, *, row_id: str, family: str, operation: str = "set") -> dict:
    noun = "ноутбук" if split == "train" else "коврик"
    other = "планшет" if split == "train" else "тренажёр"
    return {
        "id": row_id,
        "split": split,
        "source": "deterministic_schema_first_synthetic",
        "generator_version": "unit-fixture",
        "domain_family_id": family,
        "schema_family_id": f"schema-{family}",
        "template_family_id": f"template-{family}",
        "counterfactual_group_id": None,
        "input": {
            "message": f"Подберите {noun}." if operation == "set" else f"Исключите {noun}.",
            "domain": {
                "schema": {
                    "properties": {
                        "kind": {"type": "string", "enum": ["first", "second"], "description": "Класс объекта; canonical value."},
                        "style": {"type": "string", "enum": ["calm"], "description": "Режим объекта; canonical value."},
                    }
                },
                "aliases": {"kind": {noun: "first", other: "second"}, "style": {"спокойный": "calm"}},
            },
        },
        "target": {
            "updates": [{"field": "kind", "operation": operation, "value": "first", "source_text": noun}],
            "issues": [],
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _permission_row(split: str, *, row_id: str, family: str) -> dict:
    row = _row(split, row_id=row_id, family=family)
    alias = "планшет" if split == "train" else "тренажёр"
    row["input"]["message"] = f"Не первый вариант; {alias} допустим."
    row["target"]["updates"] = [{"field": "kind", "operation": "set", "value": "second", "source_text": alias}]
    return row


def test_set_emits_supported_claim_opposite_operation_and_unknown() -> None:
    cases = convert_row(_row("train", row_id="train-1", family="a"))
    assert [case["label"] for case in cases] == ["support", "contradiction", "unknown"]
    assert [(case["field_id"], case["operation"]) for case in cases] == [
        ("kind", "set"), ("kind", "exclude"), ("style", "set")
    ]
    assert all(case["premise"] == "Подберите ноутбук." for case in cases)
    assert all(case["family"] == "a" for case in cases)
    assert "ноутбук" in cases[0]["hypothesis"]
    assert "Класс объекта" in cases[0]["hypothesis"]
    assert cases[2]["source_text"] is None


def test_exclusion_supports_exclude_and_contradicts_set() -> None:
    cases = convert_row(_row("train", row_id="train-1", family="a", operation="exclude"))
    assert [(case["label"], case["operation"]) for case in cases[:2]] == [
        ("support", "exclude"), ("contradiction", "set")
    ]


def test_build_is_reproducible_and_records_input_hashes(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    _write(train, [_row("train", row_id="train-1", family="a")])
    _write(dev, [_row("dev", row_id="dev-1", family="b")])
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build(train, dev, first)
    build(train, dev, second)
    assert manifest["splits"]["train"]["labels"] == {"support": 1, "contradiction": 1, "unknown": 1}
    assert manifest["split_isolation_overlap"]["normalized_message"] == 0
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_rejects_missing_source_and_overlapping_group(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    row = _row("train", row_id="train-1", family="a")
    row["target"]["updates"][0]["source_text"] = "не было"
    _write(train, [row])
    _write(dev, [_row("dev", row_id="dev-1", family="b")])
    with pytest.raises(ValueError, match="source_text not present"):
        build(train, dev, tmp_path / "output")

    _write(train, [_row("train", row_id="train-1", family="a")])
    _write(dev, [_row("dev", row_id="dev-1", family="a")])
    with pytest.raises(ValueError, match="train/dev overlap"):
        build(train, dev, tmp_path / "output")


def test_rejects_sealed_input_even_if_row_claims_train(tmp_path: Path) -> None:
    train = tmp_path / "blind.jsonl"
    dev = tmp_path / "dev.jsonl"
    _write(train, [_row("train", row_id="train-1", family="a")])
    _write(dev, [_row("dev", row_id="dev-1", family="b")])
    with pytest.raises(ValueError, match="sealed or holdout"):
        build(train, dev, tmp_path / "output")


def test_rejects_enum_source_homonym_across_fields() -> None:
    row = _row("train", row_id="train-1", family="a")
    row["input"]["domain"]["aliases"]["style"]["ноутбук"] = "calm"
    with pytest.raises(ValueError, match="ambiguous across fields"):
        convert_row(row)


def test_rejects_empty_support_class(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    row = _row("train", row_id="train-1", family="a")
    row["target"]["updates"] = []
    _write(train, [row])
    _write(dev, [_row("dev", row_id="dev-1", family="b")])
    with pytest.raises(ValueError, match="empty label class"):
        build(train, dev, tmp_path / "output")


def test_permission_filter_preserves_explicit_dev_exclusion(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    _write(train, [_row("train", row_id="train-valid", family="a"), _permission_row("train", row_id="train-permit", family="a")])
    dev_exclude = _row("dev", row_id="dev-exclude", family="b", operation="exclude")
    dev_exclude["input"]["message"] = "Исключите коврик; другие классы допустимы."
    _write(dev, [dev_exclude, _permission_row("dev", row_id="dev-permit", family="b")])
    manifest = build(train, dev, tmp_path / "output")
    for split in ("train", "dev"):
        assert manifest["splits"][split]["exclusions"]["by_reason"] == {"permission_is_not_selection": 1}
        assert manifest["splits"][split]["labels"] == {"support": 1, "contradiction": 1, "unknown": 2}
    dev_rows = [json.loads(line) for line in (tmp_path / "output" / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["operation"] == "exclude" and row["label"] == "support" for row in dev_rows)


def test_hedge_mood_conflict_and_preference_semantics() -> None:
    hedged = _row("train", row_id="train-hedge", family="a")
    hedged["input"]["message"] = "Эм, наверное ноутбук."
    assert [item["label"] for item in convert_row(hedged)] == ["unknown"]

    mood = _row("train", row_id="train-mood", family="a")
    mood["tags"] = ["intent", "mood"]
    assert convert_row(mood) == []

    conflict = _row("train", row_id="train-conflict", family="a")
    conflict["tags"] = ["conflict"]
    conflict["input"]["message"] = "Хочу одновременно ноутбук и планшет."
    conflict["target"]["updates"].append({"field": "kind", "operation": "set", "value": "second", "source_text": "планшет"})
    assert convert_row(conflict) == []

    preference = _row("train", row_id="train-preference", family="a")
    preference["input"]["message"] = "Если получится, предпочту спокойный."
    preference["input"]["domain"]["capabilities"] = {"preference_field": "style"}
    preference["target"]["updates"] = [{"field": "style", "operation": "set", "value": "calm", "source_text": "спокойный"}]
    assert "предпочёл бы" in convert_row(preference)[0]["hypothesis"]
