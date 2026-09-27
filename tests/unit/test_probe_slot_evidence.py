"""Fixture mapping and safety accounting for read-only slot probes."""

import json

import pytest
from scripts.probe_slot_evidence import flatten_fixture, summarize


def fixture_case():
    return {
        "id": "R01",
        "family": "role-scope",
        "utterance": "Ищу вводный курс",
        "target": {"claim": "Нужен начальный курс", "gold": "entailed"},
        "opposite": {"claim": "Пользователь новичок", "gold": "unknown"},
        "competing": {"claim": "Нужен продвинутый курс", "gold": "contradicted"},
    }


def test_fixture_flattens_three_role_claims(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"cases": [fixture_case()]}, ensure_ascii=False), encoding="utf-8")
    rows = flatten_fixture(path)
    assert len(rows) == 3
    assert [row["label"] for row in rows] == ["support", "unknown", "contradiction"]
    assert {row["premise"] for row in rows} == {"Ищу вводный курс"}


def test_false_support_counts_unknown_and_contradiction(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"cases": [fixture_case()]}, ensure_ascii=False), encoding="utf-8")
    rows = flatten_fixture(path)
    report = summarize(rows, ["support", "support", "support"])
    assert report["false_support_count"] == 2
    assert {item["role"] for item in report["false_support"]} == {"opposite", "competing"}
    assert report["by_role"]["target"]["accuracy"] == 1.0


def test_unknown_gold_or_duplicate_case_is_rejected(tmp_path):
    path = tmp_path / "fixture.json"
    bad = fixture_case()
    bad["target"]["gold"] = "maybe"
    path.write_text(json.dumps({"cases": [bad]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown target gold"):
        flatten_fixture(path)
    path.write_text(json.dumps({"cases": [fixture_case(), fixture_case()]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        flatten_fixture(path)
