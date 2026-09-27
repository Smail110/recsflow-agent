"""Regression tests for provenance and leakage-safe intent training splits."""

import json

import pytest
from scripts.prepare_hf_dataset import (
    TEMPLATE_VERSION,
    build,
    normalized_text_sha256,
    validate_rows,
)
from scripts.train_lora import validation_report


def test_generated_rows_have_provenance_and_clean_group_split(tmp_path):
    rows = build(seed=42, examples_per_label=5)
    summary = validate_rows(rows)

    assert len(rows) == 20
    assert summary["counts"]["rows"] == 20
    assert summary["counts"]["train"] + summary["counts"]["validation"] == 20
    assert set(summary["train_groups"]).isdisjoint(summary["validation_groups"])
    assert {row["template_version"] for row in rows} == {TEMPLATE_VERSION}
    assert all(row["template_family_id"] and isinstance(row["seed"], int) for row in rows)
    assert all(row["normalized_text_sha256"] == normalized_text_sha256(row["text"]) for row in rows)

    path = tmp_path / "dialogues.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    roundtrip = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert validate_rows(roundtrip)["train_indices"] == summary["train_indices"]
    assert validate_rows(roundtrip)["validation_indices"] == summary["validation_indices"]


def test_same_seed_is_reproducible():
    assert build(seed=7, examples_per_label=5) == build(seed=7, examples_per_label=5)
    assert build(seed=7, examples_per_label=5) != build(seed=8, examples_per_label=5)


def test_template_family_leakage_is_rejected():
    rows = build(seed=42, examples_per_label=5)
    # Move one row so its family appears on both sides.
    rows[0]["split"] = "validation" if rows[0]["split"] == "train" else "train"
    with pytest.raises(ValueError, match="template-family overlap"):
        validate_rows(rows)


def test_normalized_duplicate_is_rejected():
    rows = build(seed=42, examples_per_label=5)
    duplicate = dict(rows[1])
    duplicate["text"] = f"  {rows[0]['text'].upper()}  "
    duplicate["normalized_text_sha256"] = normalized_text_sha256(duplicate["text"])
    rows.append(duplicate)
    with pytest.raises(ValueError, match="normalized duplicate"):
        validate_rows(rows)


def test_legacy_input_fails_with_migration_command():
    row = {"text": "Привет", "label": "discovery"}
    with pytest.raises(ValueError, match="prepare_hf_dataset"):
        validate_rows([row], source="legacy.jsonl")


def test_validation_report_exposes_checked_counts():
    rows = build(seed=42, examples_per_label=5)
    report = validation_report(validate_rows(rows), model="m", output="o")
    assert report["status"] == "validated"
    assert report["split_counts"]["train"] + report["split_counts"]["validation"] == 20
    assert set(report["label_counts"]["train"]) == set(report["labels"])
    assert set(report["label_counts"]["validation"]) == set(report["labels"])
