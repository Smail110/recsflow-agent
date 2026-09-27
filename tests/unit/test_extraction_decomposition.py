import json
from pathlib import Path

from scripts.evaluate_extraction_decomposition import (
    AtomicFact,
    FactInventory,
    FactMapping,
    compose,
    load_inputs,
    score,
)

ROOT = Path(__file__).resolve().parents[2]


def test_fact_fixture_is_complete_and_taxonomy_is_one_primary_per_failure():
    fixture = json.loads((ROOT / "data/model_extraction_dev_facts.json").read_text(encoding="utf-8"))
    assert len(fixture["cases"]) == 25
    failures = [case for case in fixture["cases"] if case["primary_category"]]
    assert len(failures) == 14
    assert len({case["dialogue_id"] for case in failures}) == 14
    assert {category: sum(case["primary_category"] == category for case in failures) for category in "ABCDEF"} == {
        "A": 3,
        "B": 0,
        "C": 0,
        "D": 11,
        "E": 0,
        "F": 0,
    }


def test_saved_baseline_is_self_contained_and_reproducible(monkeypatch, tmp_path):
    import scripts.evaluate_extraction_decomposition as evaluator

    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    fixture, contexts = load_inputs(
        ROOT / "data/model_extraction_dev_facts.json",
        ROOT / "this-context-report-must-not-exist.json",
    )
    rows = []
    from recagent.interpretation import StructuredRequest

    for case in fixture["cases"]:
        raw = contexts[(case["dialogue_id"], case["turn"])]["structured"]
        rows.append(score(case, StructuredRequest.model_validate(raw)))
    assert sum(row["expected_fact_count"] for row in rows) == 56
    assert sum(row["correct_fact_count"] for row in rows) == 30
    assert sum(len(row["missing_facts"]) for row in rows) == 18
    assert sum(len(row["wrong_facts"]) for row in rows) == 8
    assert sum(len(row["invented_facts"]) for row in rows) == 0


def test_composer_is_schema_relative_and_preserves_unmapped_fact_as_issue():
    inventory = FactInventory(
        intent="discovery",
        facts=[
            AtomicFact(source_text="ноутбук", meaning="категория ноутбук"),
            AtomicFact(source_text="лёгкий", meaning="портативный"),
        ],
    )
    request, diagnostics = compose(
        inventory,
        [
            FactMapping(fact_index=0, field="category", value="laptop"),
        ],
    )
    assert request.updates[0].model_dump() == {
        "field": "category",
        "operation": "set",
        "value": "laptop",
        "source_text": "ноутбук",
    }
    assert request.issues[0].kind == "ambiguity"
    assert diagnostics["composition_issues"] == ["unmapped:1"]


def test_electronics_fixture_uses_a_distinct_customer_shape():
    fixture, contexts = load_inputs(
        ROOT / "data/model_extraction_electronics_smoke.json",
        ROOT / "this-context-report-must-not-exist.json",
    )
    payload = contexts[("electronics-portable-budget", 0)]["payload"]
    assert set(payload["previous"]) == {"intent", "category", "max_price", "portable"}
    assert {fact["field"] for fact in fixture["cases"][0]["expected_facts"]} == {
        "category",
        "max_price",
        "portable",
    }


def test_schema_guard_does_not_apply_unknown_field_as_a_constraint():
    inventory = FactInventory(
        intent="discovery",
        facts=[
            AtomicFact(source_text="Найди", meaning="команда поиска"),
            AtomicFact(source_text="ноутбук", meaning="категория ноутбук"),
        ],
    )
    domain = {
        "schema": {
            "properties": {
                "intent": {"enum": ["discovery"], "type": "string"},
                "category": {"anyOf": [{"enum": ["laptop", "tablet"], "type": "string"}, {"type": "null"}]},
            }
        },
        "exclusion_fields": {},
    }
    request, _ = compose(
        inventory,
        [
            FactMapping(fact_index=0, field="intent", value="discovery"),
            FactMapping(fact_index=1, field="category", value="laptop"),
        ],
        domain,
    )
    assert [(update.field, update.value) for update in request.updates] == [("category", "laptop")]
    assert [(issue.kind, issue.field) for issue in request.issues] == [("unsupported_constraint", "intent")]


def test_schema_guard_does_not_apply_invalid_enum_type_or_exclusion():
    inventory = FactInventory(
        intent="discovery",
        facts=[
            AtomicFact(source_text="телефон", meaning="категория телефон"),
            AtomicFact(source_text="недорого", meaning="бюджет недорого"),
            AtomicFact(source_text="не ноутбук", meaning="исключить ноутбук"),
        ],
    )
    domain = {
        "schema": {
            "properties": {
                "category": {"anyOf": [{"enum": ["laptop", "tablet"], "type": "string"}, {"type": "null"}]},
                "max_price": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            }
        },
        "exclusion_fields": {},
    }
    request, _ = compose(
        inventory,
        [
            FactMapping(fact_index=0, field="category", value="phone"),
            FactMapping(fact_index=1, field="max_price", value="cheap"),
            FactMapping(fact_index=2, field="category", operation="exclude", value="laptop"),
        ],
        domain,
    )
    assert request.updates == []
    assert [(issue.kind, issue.field) for issue in request.issues] == [
        ("unsupported_constraint", "category"),
        ("unsupported_constraint", "max_price"),
        ("unsupported_constraint", "category"),
    ]
