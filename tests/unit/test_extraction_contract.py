import pytest
from pydantic import ValidationError
from scripts.evaluate_extraction_contract import score_case
from scripts.evaluate_llm_first_product import install_capture

from recagent.domains.demo import domain_spec, request_adapter
from recagent.interpretation import ConstraintUpdate, InterpretationIssue, LLMRequestInterpreter, StructuredRequest, flat_transport_schema
from recagent.validation import normalize_proposal, validate_proposal


def _score(case, request):
    proposal = normalize_proposal(request, turn_id="test", message=case["message"])
    validation = validate_proposal(proposal, message=case["message"], known_fields={"kind", "tone"})
    return score_case(case, request, validation, [])


def test_contract_scorer_requires_every_declared_update():
    case = {
        "message": "Нужен лёгкий сериал.",
        "expected_updates": [["kind", "set", "series"], ["tone", "set", "лёгкий"]],
    }
    request = StructuredRequest(updates=[ConstraintUpdate(field="kind", value="series", source_text="сериал")])
    scored = _score(case, request)
    assert scored["proposal_correct"] is False
    assert scored["missing_updates"] == [["tone", "set", "лёгкий"]]


def test_contract_scorer_detects_fabricated_and_wrong_negation_operation():
    case = {
        "message": "Не хочу мрачный сериал.",
        "expected_updates": [["kind", "set", "series"], ["tone", "exclude", "мрачный"]],
        "forbidden_fields": ["practical"],
    }
    request = StructuredRequest(
        updates=[
            ConstraintUpdate(field="kind", value="series", source_text="сериал"),
            ConstraintUpdate(field="tone", operation="set", value="мрачный", source_text="мрачный"),
            ConstraintUpdate(field="practical", value=True, source_text="мрачный"),
        ]
    )
    scored = _score(case, request)
    assert scored["proposal_correct"] is False
    assert scored["operation_errors"] == [["tone", "set", "мрачный"]]
    assert scored["fabricated_updates"] == [["practical", "set", True]]


def test_contract_scorer_requires_evidence_for_unresolved_expression():
    case = {
        "message": "Нужен курс для middle-разработчика.",
        "expected_updates": [["kind", "set", "course"]],
        "expected_unresolved_fields": ["level"],
        "expected_issue_kinds": ["unsupported_constraint"],
    }
    request = StructuredRequest(
        updates=[ConstraintUpdate(field="kind", value="course", source_text="курс")],
        issues=[
            InterpretationIssue(
                kind="unsupported_constraint",
                field="level",
                value="middle",
                message="Уровень не представлен.",
            )
        ],
    )
    scored = _score(case, request)
    assert scored["proposal_correct"] is False
    assert scored["missing_issue_evidence"] == ["level"]


def test_validator_rejects_unresolved_issue_with_evidence_from_another_turn():
    message = "Нужен курс Python."
    request = StructuredRequest(
        updates=[ConstraintUpdate(field="kind", value="course", source_text="курс")],
        issues=[
            InterpretationIssue(
                kind="unsupported_constraint",
                field="level",
                value="middle",
                message="Уровень не представлен.",
                source_text="middle-разработчик",
            )
        ],
    )
    proposal = normalize_proposal(request, turn_id="test", message=message)
    validation = validate_proposal(proposal, message=message, known_fields={"kind", "level"})
    assert validation.status == "REJECT"
    assert any(finding.code == "issue_evidence_not_in_turn" for finding in validation.findings)


def test_canonical_flat_schema_excludes_legacy_query_members_without_mapping_them():
    schema = flat_transport_schema(domain_spec())
    with pytest.raises(ValidationError):
        schema.model_validate({"updates": [{"field": "excluded_genres", "operation": "set", "value": "комедия", "source_text": "комедии"}]})

    class Backend:
        def structured(self, backend_schema, system, payload):
            assert "excluded_genres" not in str(backend_schema.model_json_schema())
            assert "excluded_genres" not in str(payload["domain"])
            assert "excluded_genres" not in str(payload["previous"])
            return backend_schema(updates=[{"field": "genre", "operation": "exclude", "value": "комедия", "source_text": "без комедий"}]), 1

    request, _ = LLMRequestInterpreter(Backend(), request_adapter().descriptor, domain_spec=domain_spec()).interpret(
        "без комедий", {"intent": "discovery", "trusted_active": []}
    )
    assert [(update.field, update.operation, update.value) for update in request.updates] == [("genre", "exclude", "комедия")]


def test_focused_capture_classifies_dynamic_flat_schema_as_interpretation():
    class Client:
        def structured(self, schema, system, payload):
            del system, payload
            return schema(), 0

        def parse(self, message, previous):
            raise AssertionError("not used")

    client = Client()
    calls, _ = install_capture(client)
    client.structured(flat_transport_schema(domain_spec()), "system", {})
    assert calls[0]["role"] == "interpretation"
