"""Structural and semantic checks for citation-aware synthetic pairs."""

import json
from collections import Counter
from pathlib import Path

import pytest
from scripts.build_cited_nli_dataset import OPERATIONS, _tag, build, convert_schema

from recagent.context_nli import cited_claim_for_update, tag_citation


def _schema(family: str = "sf_a", domain: str = "devices") -> dict:
    aliases = {
        "kind": {"ноутбук": "laptop", "планшет": "tablet"},
        "style": {"складной": "folding", "устойчивый": "stable"},
        "context": {"для дома": "home", "для поездок": "travel"},
        "wish": {"тихий": "quiet", "лёгкий": "light"},
    }
    labels = {
        "kind": "Класс объекта",
        "style": "Режим объекта",
        "context": "Контекст использования",
        "wish": "Мягкое пожелание",
    }
    return {
        "schema_family_id": family,
        "domain_family_id": domain,
        "domain_id": domain,
        "aliases": aliases,
        "properties": {
            field: {
                "type": "string",
                "enum": sorted(set(mapping.values())),
                "description": labels[field] + "; canonical value указан в aliases.",
            }
            for field, mapping in aliases.items()
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
            "message": "Old generator text must never appear in v4.",
            "domain": {
                "schema_family_id": schema["schema_family_id"],
                "aliases": schema["aliases"],
                "schema": {"properties": schema["properties"]},
                "capabilities": {"preference_field": schema["preference_field"]},
            },
        },
        "target": {"updates": []},
    }


def _write(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_every_operation_has_all_labels_and_citation_counterfactual() -> None:
    rows = convert_schema(_schema(), "train")
    counts = Counter((row["operation"], row["label"]) for row in rows)
    assert all(counts[(operation, label)] for operation in OPERATIONS for label in ("support", "contradiction", "unknown"))
    pairs: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        assert row["message"][row["source_start"] : row["source_end"]] == row["source_text"]
        assert row["premise"] == _tag(row["message"], row["source_text"], row["source_start"], row["source_end"])
        assert row["premise"] == tag_citation(
            row["message"], row["source_text"], source_start=row["source_start"], source_end=row["source_end"]
        )
        assert row["hypothesis"] == cited_claim_for_update(
            row["role"], row["value_alias"], row["operation"], is_preference=row["field_id"] == "wish"
        )
        assert row["premise"].count("<evidence>") == 1
        assert row["hypothesis"].startswith("Выделенная цитата подтверждает: ")
        pairs.setdefault((row["message"], row["hypothesis"], row["citation_span_family"]), {})[row["citation_kind"]] = row["label"]
    assert all(set(pair) == {"primary", "counterfactual"} for pair in pairs.values())
    assert any(pair == {"primary": "support", "counterfactual": "unknown"} for pair in pairs.values())
    assert any(pair == {"primary": "contradiction", "counterfactual": "unknown"} for pair in pairs.values())
    assert {row["citation_span_family"] for row in rows} == {"clause", "value"}
    for kind in ("primary", "counterfactual"):
        assert {row["marked_clause_position"] for row in rows if row["citation_kind"] == kind} == {"first", "second"}
    same_alias = [row for row in rows if row["distractor_family"] == "same_alias_quote" and row["citation_span_family"] == "value"]
    assert any(row["citation_kind"] == "counterfactual" and row["value_alias"] == row["source_text"] for row in same_alias)
    assert all("именно" not in row["message"] for row in rows)


def test_include_is_permission_not_selection() -> None:
    rows = convert_schema(_schema(), "train")
    include_source = [row for row in rows if row["actual_operation"] == "include" and row["citation_kind"] == "primary"]
    assert all(row["label"] == "unknown" for row in include_source if row["operation"] == "set")
    assert all(row["label"] == "contradiction" for row in include_source if row["operation"] == "exclude")
    set_source = [row for row in rows if row["actual_operation"] == "set" and row["citation_kind"] == "primary"]
    assert all(row["label"] == "unknown" for row in set_source if row["operation"] == "include")


def test_reproducibility_source_isolation_and_manifest(tmp_path: Path) -> None:
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train_row = _source_row("train", _schema())
    _write(train, train_row)
    _write(dev, _source_row("dev", _schema("sf_b", "travel")))
    first = build(train, dev, tmp_path / "first")
    second = build(train, dev, tmp_path / "second")
    assert (tmp_path / "first" / "train.jsonl").read_bytes() == (tmp_path / "second" / "train.jsonl").read_bytes()
    assert (tmp_path / "first" / "manifest.json").read_bytes() == (tmp_path / "second" / "manifest.json").read_bytes()
    assert first == second
    assert all(value == 0 for value in first["split_isolation_overlap"].values())
    assert first["splits"]["train"]["citation_pair_audit"]["support_to_unknown"] > 0
    assert all(
        first["splits"]["train"]["marked_position_by_citation_kind"][kind][position] > 0
        for kind in ("primary", "counterfactual")
        for position in ("first", "second")
    )
    train_row["input"]["message"] = "Changed original generator text."
    train_row["target"]["updates"] = [{"field": "kind", "value": "tablet"}]
    _write(train, train_row)
    third = build(train, dev, tmp_path / "third")
    assert (tmp_path / "first" / "train.jsonl").read_bytes() == (tmp_path / "third" / "train.jsonl").read_bytes()
    assert first["splits"]["train"]["source_sha256"] != third["splits"]["train"]["source_sha256"]


def test_rejects_broken_citation_and_sealed_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="citation offsets"):
        _tag("один второй", "второй", 0, 6)
    train = tmp_path / "blind.jsonl"
    dev = tmp_path / "dev.jsonl"
    _write(train, _source_row("train", _schema()))
    _write(dev, _source_row("dev", _schema("sf_b", "travel")))
    with pytest.raises(ValueError, match="sealed or holdout"):
        build(train, dev, tmp_path / "out")
