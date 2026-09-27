"""Checks for the role-aware synthetic NLI data builder."""

import json
from pathlib import Path

import pytest
from scripts.build_role_nli_dataset import build, convert_schema

from recagent.context_nli import claim_for_update


def _schema(family: str = "sf_a", domain: str = "devices") -> dict:
    aliases = {
        "kind": {"ноутбук": "laptop", "планшет": "tablet"},
        "style": {"складной": "folding", "устойчивый": "stable"},
        "context": {"для дома": "home", "для поездок": "travel"},
        "wish": {"тихий": "quiet", "лёгкий": "light"},
    }
    descriptions = {
        "kind": "Класс объекта", "style": "Режим объекта",
        "context": "Контекст использования", "wish": "Мягкое пожелание",
    }
    return {
        "schema_family_id": family,
        "domain_family_id": domain,
        "domain_id": domain,
        "aliases": aliases,
        "properties": {
            field: {"type": "string", "enum": sorted(set(mapping.values())),
                    "description": label + "; canonical value указан в aliases."}
            for field, (mapping, label) in (
                (name, (aliases[name], descriptions[name])) for name in aliases
            )
        },
        "preference_field": "wish",
    }


def _source_row(split: str, schema: dict) -> dict:
    return {
        "id": f"source-{split}-{schema['schema_family_id']}",
        "split": split,
        "source": "deterministic_schema_first_synthetic",
        "generator_version": "semantic-lora-schema-first-v2",
        "schema_family_id": schema["schema_family_id"],
        "domain_family_id": schema["domain_family_id"],
        "domain_id": schema["domain_id"],
        "input": {
            "message": "This text must never become a training template.",
            "domain": {
                "schema_family_id": schema["schema_family_id"],
                "aliases": schema["aliases"],
                "schema": {"properties": schema["properties"]},
                "capabilities": {"preference_field": schema["preference_field"]},
            },
        },
        "target": {"updates": [{"field": "kind", "value": "tablet"}]},
    }


def _write(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_matched_claims_and_implicit_role_cues() -> None:
    rows = convert_schema(_schema(), "train")
    assert {row["label"] for row in rows} == {"support", "contradiction", "unknown"}
    assert {row["operation"] for row in rows} == {"set", "exclude", "include"}
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(row["premise"], set()).add(row["label"])
    assert all(labels == {"support", "contradiction", "unknown"} for labels in grouped.values())
    implicit = [row for row in rows if "implicit_set" in row["template_family_id"] and row["label"] == "support"]
    assert {row["role"] for row in implicit} == {
        "Класс объекта", "Режим объекта", "Контекст использования", "Мягкое пожелание"
    }
    assert all(row["role"] not in row["premise"] for row in implicit)
    wish = next(row for row in implicit if row["role"] == "Мягкое пожелание")
    assert "предпочёл бы" in wish["hypothesis"]
    alternatives = [row for row in rows if row["reason"] == "unmentioned_alternative_same_role"]
    assert alternatives and all(row["label"] == "unknown" for row in alternatives)


def test_permission_and_non_target_mentions_are_unknown() -> None:
    rows = convert_schema(_schema(), "train")
    for family in ("permitted", "uncertain", "reported"):
        examples = [row for row in rows if row["template_family_id"].endswith(family)]
        assert examples
        assert all(row["label"] == "unknown" for row in examples if row["reason"] == f"{family}_is_not_selection")
        assert all(row["value_alias"] in row["premise"] for row in examples if row["reason"] == f"{family}_is_not_selection")
    include = [row for row in rows if row["template_family_id"].endswith("include")]
    assert any(row["operation"] == "set" and row["label"] == "unknown" for row in include)


def test_train_claim_matches_runtime_verifier_contract() -> None:
    for row in convert_schema(_schema(), "train"):
        assert row["hypothesis"] == claim_for_update(
            row["role"], row["value_alias"], row["operation"],
            is_preference=row["field_id"] == "wish",
        )


def test_build_reproducible_and_ignores_original_message_target(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train_row = _source_row("train", _schema())
    dev_row = _source_row("dev", _schema("sf_b", "travel"))
    _write(train, train_row)
    _write(dev, dev_row)
    first = build(train, dev, tmp_path / "first")
    train_row["input"]["message"] = "A completely different old message."
    train_row["target"]["updates"] = []
    _write(train, train_row)
    second = build(train, dev, tmp_path / "second")
    assert (tmp_path / "first" / "train.jsonl").read_bytes() == (tmp_path / "second" / "train.jsonl").read_bytes()
    assert first["splits"]["train"]["source_sha256"] != second["splits"]["train"]["source_sha256"]
    assert first["split_isolation_overlap"]["normalized_pair"] == 0
    assert first["splits"]["train"]["labels"]["unknown"] > 0


def test_rejects_overlapping_families_and_sealed_input(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    _write(train, _source_row("train", _schema()))
    _write(dev, _source_row("dev", _schema()))
    with pytest.raises(ValueError, match="overlap"):
        build(train, dev, tmp_path / "out")
    blind = tmp_path / "blind.jsonl"
    _write(blind, _source_row("train", _schema()))
    with pytest.raises(ValueError, match="sealed or holdout"):
        build(blind, dev, tmp_path / "out")
