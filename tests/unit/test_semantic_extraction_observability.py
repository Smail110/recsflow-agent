from pydantic import BaseModel
from scripts.evaluate_semantic_extraction_observability import (
    ControlledOllamaClient,
    ObservedFact,
    ShadowDiagnostics,
    evidence_matches,
    issue_matches,
    materialize_observability,
    needs_second_pass,
    normalize_mapping_result,
    score,
    second_pass,
    summarize,
)

from recagent.interpretation import ConstraintUpdate, InterpretationIssue, StructuredRequest
from recagent.parsing import OllamaClient


def diagnostics(*facts, compound=False, uncertain=False):
    return ShadowDiagnostics(
        intent="discovery",
        detected_facts=list(facts),
        compound=compound,
        uncertain=uncertain,
    )


def observed(**changes):
    values = {
        "source_text": "лёгкий",
        "kind": "attribute",
        "polarity": "positive",
        "intent": "discovery",
        "target_field": "portable",
        "operation": "set",
        "value": True,
        "mapping_result": "mapped",
        "rejection_reason": None,
    }
    return ObservedFact(**(values | changes))


def case():
    return {
        "user": "Нужен лёгкий ноутбук, но не тяжёлый.",
        "expected_intent": "discovery",
        "expected_reset": False,
        "tags": ["compound", "negation"],
        "expected_facts": [
            {
                "kind": "attribute",
                "polarity": "positive",
                "field": "portable",
                "operation": "set",
                "value": True,
                "source_any_of": ["лёгкий"],
                "mapping_result": "mapped",
            },
            {
                "kind": "attribute",
                "polarity": "negative",
                "field": "weight_class",
                "operation": "exclude",
                "value": "heavy",
                "source_any_of": ["не тяжёлый", "тяжёлый"],
                "mapping_result": "mapped",
            },
        ],
        "expected_issues": [],
    }


def test_grounded_scoring_separates_omission_and_invention():
    request = StructuredRequest(
        intent="discovery",
        updates=[
            ConstraintUpdate(field="portable", value=True, source_text="лёгкий"),
            ConstraintUpdate(field="color", value="red", source_text="ноутбук"),
        ],
    )
    result = score(case(), request, diagnostics(observed(), compound=True))
    assert result["grounded_correct_count"] == 1
    assert [item["field"] for item in result["omitted_facts"]] == ["weight_class"]
    assert [item["field"] for item in result["invented_facts"]] == ["color"]
    assert result["compound_complete"] is False


def test_scoring_reserves_exact_match_before_classifying_related_fact_as_wrong():
    expected = case()
    expected["user"] = "Нужен лёгкий ноутбук, но не тяжёлый."
    request = StructuredRequest(
        intent="discovery",
        updates=[
            ConstraintUpdate(
                field="weight_class",
                operation="exclude",
                value="heavy",
                source_text="лёгкий ноутбук, но не тяжёлый",
            ),
        ],
    )
    result = score(expected, request, diagnostics(compound=True))
    assert result["semantic_match_count"] == 1
    assert result["grounded_correct_count"] == 1
    assert [item["field"] for item in result["omitted_facts"]] == ["portable"]
    assert result["wrong_facts"] == []
    assert result["invented_facts"] == []


def test_stage_mapping_normalizes_rejected_prefix_and_requires_intent():
    assert normalize_mapping_result("rejected_unsupported") == "unsupported"
    assert normalize_mapping_result("unsupported") == "unsupported"
    reduced = case() | {
        "expected_facts": [
            case()["expected_facts"][0]
            | {
                "mapping_result": "rejected_unsupported",
            }
        ],
        "tags": [],
    }
    fact = observed(mapping_result="unsupported")
    request = StructuredRequest(intent="discovery")
    result = score(reduced, request, diagnostics(fact))
    assert [failure["stage"] for failure in result["stage_failures"]] == ["C"]

    fact_without_intent = observed(mapping_result="unsupported", intent=None)
    result = score(reduced, request, diagnostics(fact_without_intent))
    assert [failure["stage"] for failure in result["stage_failures"]] == ["B"]


def test_invalid_evidence_is_stage_d_not_grounded_correct():
    request = StructuredRequest(
        intent="discovery",
        updates=[
            ConstraintUpdate(field="portable", value=True, source_text="ноутбук"),
        ],
    )
    reduced = case() | {"expected_facts": [case()["expected_facts"][0]], "tags": []}
    result = score(reduced, request, diagnostics(observed()))
    assert result["semantic_match_count"] == 1
    assert result["grounded_correct_count"] == 0
    assert [failure["stage"] for failure in result["stage_failures"]] == ["D"]


def test_missing_expected_issue_is_stage_e():
    reduced = case() | {
        "expected_facts": [case()["expected_facts"][0]],
        "expected_issues": [{"kind": "unsupported_constraint", "field": "portable", "value": "maybe"}],
        "tags": [],
    }
    request = StructuredRequest(
        intent="discovery",
        updates=[
            ConstraintUpdate(field="portable", value=True, source_text="лёгкий"),
        ],
    )
    result = score(reduced, request, diagnostics(observed()))
    assert result["preserved_issue_count"] == 0
    assert [failure["stage"] for failure in result["stage_failures"]] == ["E"]


def test_issue_match_and_selective_gate_are_generic():
    request = StructuredRequest(
        issues=[InterpretationIssue(kind="ambiguity", field="size", value="средний", message="Уточните размер.")],
        clarification_required=True,
    )
    assert needs_second_pass(request, diagnostics())
    assert needs_second_pass(StructuredRequest(), diagnostics(observed(), compound=True))
    assert not needs_second_pass(StructuredRequest(), diagnostics())


def test_observability_materializes_mapping_result_without_mutating_request():
    request = StructuredRequest(
        intent="discovery",
        updates=[
            ConstraintUpdate(field="portable", value=True, source_text="лёгкий"),
        ],
    )
    before = request.model_dump(mode="json")
    payload = {
        "message": "Нужен лёгкий ноутбук",
        "domain": {
            "schema": {
                "properties": {
                    "portable": {"type": ["boolean", "null"]},
                }
            },
            "exclusion_fields": {},
        },
    }
    records = materialize_observability(diagnostics(observed()), request, payload)
    assert records[0]["mapping_result"] == "mapped"
    assert records[0]["rejection_reason"] is None
    assert request.model_dump(mode="json") == before


def test_evidence_requires_current_message_and_expected_span():
    expected = {"source_any_of": ["до 100000 рублей"]}
    assert evidence_matches("100000 рублей", expected, "Нужен ноутбук до 100000 рублей")
    assert not evidence_matches("ноутбук", expected, "Нужен ноутбук до 100000 рублей")


def test_think_client_request_differs_from_production_only_by_flag(monkeypatch):
    class Answer(BaseModel):
        value: str

    captured = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": '{"value":"ok"}'},
                "prompt_eval_count": 2,
                "eval_count": 1,
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            captured.append({"url": url, "body": json})
            return Response()

    import scripts.evaluate_semantic_extraction_observability as evaluator

    monkeypatch.setattr(evaluator.httpx, "Client", Client)
    OllamaClient("model", "http://local", 3).structured(Answer, "system", {"x": 1})
    ControlledOllamaClient("model", "http://local", 3, think=True).structured(Answer, "system", {"x": 1})
    current, candidate = captured
    assert current["url"] == candidate["url"] == "http://local/api/chat"
    assert current["body"] | {"think": True} == candidate["body"]


def test_second_pass_is_whole_replacement_and_receives_no_shadow_facts():
    class Backend:
        def __init__(self):
            self.last_usage = {}
            self.last_meta = {}

        def structured(self, schema, system, payload):
            assert "shadow_diagnostics" not in payload
            assert payload["first_request"]["updates"][0]["field"] == "old"
            return schema(
                updates=[
                    {
                        "field": "new",
                        "value": "kept",
                        "source_text": "новое",
                    }
                ]
            ), 3

    first = StructuredRequest(
        updates=[
            ConstraintUpdate(field="old", value="discarded", source_text="старое"),
        ]
    )
    final, _ = second_pass(Backend(), {"message": "новое", "domain": {}}, first)
    assert [(update.field, update.value) for update in final.updates] == [("new", "kept")]


def test_issue_matching_does_not_treat_false_or_zero_as_missing_value():
    assert not issue_matches(
        {"kind": "conflict", "field": "portable", "value": False},
        {"kind": "conflict", "field": "portable", "value": True},
    )
    assert not issue_matches(
        {"kind": "conflict", "field": "max_price", "value": 0},
        {"kind": "conflict", "field": "max_price", "value": 1},
    )


def test_summary_uses_actual_facts_as_invention_denominator():
    reduced = case() | {"expected_facts": [case()["expected_facts"][0]], "tags": []}
    request = StructuredRequest(
        intent="discovery",
        updates=[
            ConstraintUpdate(field="portable", value=True, source_text="лёгкий"),
            ConstraintUpdate(field="color", value="red", source_text="ноутбук"),
        ],
    )
    row = {
        "id": "one",
        "domain_id": "electronics",
        "tags": [],
        "score": score(reduced, request, diagnostics(observed())),
        "calls": [],
    }
    summary = summarize([row])
    assert summary["actual_facts"] == 2
    assert summary["invented_facts"] == 1
    assert summary["invented_fact_rate"] == 0.5
    assert summary["semantic_precision"] == 0.5
    assert summary["semantic_recall"] == 1


def test_invalid_prediction_stays_in_coverage_and_metric_denominators():
    reduced = case() | {"expected_facts": [case()["expected_facts"][0]], "tags": []}
    valid = {
        "id": "valid",
        "domain_id": "electronics",
        "tags": [],
        "score": score(
            reduced,
            StructuredRequest(
                intent="discovery",
                updates=[
                    ConstraintUpdate(field="portable", value=True, source_text="лёгкий"),
                ],
            ),
            diagnostics(observed()),
        ),
        "calls": [],
    }
    invalid = {
        "id": "invalid",
        "domain_id": "electronics",
        "tags": ["compound"],
        "error": "ValidationError: invalid StructuredRequest",
        "error_type": "ValidationError",
        "expected_fact_count": 2,
        "actual_fact_count": 0,
        "expected_issue_count": 1,
        "compound": True,
    }
    summary = summarize([valid, invalid])
    assert summary["turns"] == 2
    assert summary["completed_turns"] == 1
    assert summary["missing_prediction_turns"] == 1
    assert summary["coverage_rate"] == 0.5
    assert summary["expected_facts"] == 3
    assert summary["grounded_correct_rate"] == 1 / 3
    assert summary["omissions"] == 2
    assert summary["compound_turns"] == 1
    assert summary["compound_complete_rate"] == 0
    assert summary["expected_issues"] == 1
    assert summary["issue_preservation_rate"] == 0
    assert summary["structured_exact_match_rate"] == 0.5
