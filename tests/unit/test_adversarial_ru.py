import json
from dataclasses import replace
from pathlib import Path

import pytest
from evals.adversarial_ru import (
    CASES,
    CORPUS_ORIGIN,
    EVALUATED_FIELDS,
    SPLIT,
    corpus_audit,
    corpus_sha256,
    near_duplicate_pairs,
    normalize_utterance,
    run_challenge,
    validate_corpus,
    write_report,
)

from recagent.models import Query

EXPECTED_CATEGORIES = {
    "mixed_script_typos",
    "navigation",
    "negation",
    "numeric_units",
    "preference_correction",
    "prompt_injection",
    "similarity",
    "uncertainty",
    "unsupported",
}


def test_corpus_has_declared_scope_provenance_and_manual_expectations():
    validate_corpus()
    assert len(CASES) == 72
    assert {item.category for item in CASES} == EXPECTED_CATEGORIES
    assert all(sum(item.category == category for item in CASES) == 8 for category in EXPECTED_CATEGORIES)
    assert CORPUS_ORIGIN == "ai_authored_synthetic"
    assert SPLIT == "diagnostic_dev"
    assert set(EVALUATED_FIELDS) == set(Query.model_fields)
    assert all(item.allowed and item.rationale for item in CASES)
    assert any(len(item.allowed) > 1 for item in CASES)


def test_language_families_are_explicit_groups_not_claimed_independent_units():
    families = [item.language_family_id for item in CASES]
    assert len(set(families)) < len(families)
    assert all("." in family for family in families)

    audit = corpus_audit()
    assert audit["case_count"] == 72
    assert audit["category_count"] == 9
    assert audit["category_counts"] == dict.fromkeys(sorted(EXPECTED_CATEGORIES), 8)
    assert audit["language_family_count"] == len(set(families))
    assert audit["singleton_language_family_count"] + audit["multi_case_language_family_count"] == audit["language_family_count"]
    assert set(audit["category_language_family_counts"]) == EXPECTED_CATEGORIES
    assert audit["exact_normalized_duplicate_count"] == 0


def test_near_duplicate_audit_flags_but_does_not_reject_semantic_minimal_pair():
    minimal_pair = [next(item for item in CASES if item.case_id == case_id) for case_id in ("neg-05", "neg-08")]
    validate_corpus(minimal_pair)
    pairs = near_duplicate_pairs(minimal_pair)
    assert [(pair["left_case_id"], pair["right_case_id"]) for pair in pairs] == [("neg-05", "neg-08")]
    assert pairs[0]["same_category"] is True
    assert pairs[0]["same_language_family"] is False


def test_normalized_duplicate_is_rejected():
    duplicate = replace(CASES[1], case_id="duplicate", utterance=CASES[0].utterance.upper() + "!!!")
    with pytest.raises(ValueError, match="normalized utterance duplicates"):
        validate_corpus([CASES[0], duplicate])
    assert normalize_utterance("Ёлка?!") == normalize_utterance("елка")


def test_invalid_ground_truth_fails_closed():
    invalid_policy = replace(CASES[0].allowed[0], issue="sometimes")
    with pytest.raises(ValueError, match="invalid issue policy"):
        validate_corpus([replace(CASES[0], allowed=(invalid_policy,))])

    invalid_value = replace(CASES[0].allowed[0], fields=(("max_minutes", 0),))
    with pytest.raises(ValueError, match="invalid allowed outcome"):
        validate_corpus([replace(CASES[0], allowed=(invalid_value,))])

    with pytest.raises(ValueError, match="invalid previous Query"):
        validate_corpus([replace(CASES[0], previous=(("max_minutes", 0),))])

    with pytest.raises(ValueError, match="empty rationale"):
        validate_corpus([replace(CASES[0], rationale="  ")])


def _first_allowed_parser(message: str, previous: Query):
    item = next(item for item in CASES if item.utterance == message)
    outcome = item.allowed[0]
    values = previous.model_dump()
    values.update(dict(outcome.fields))
    query = Query.model_validate(values)
    issue = "Нужно уточнение" if outcome.issue == "required" else None
    return query, issue


def test_runner_accepts_an_explicit_allowed_outcome_and_is_deterministic():
    first = run_challenge(_first_allowed_parser, parser_name="test.first_allowed")
    second = run_challenge(_first_allowed_parser, parser_name="test.first_allowed")
    assert first == second
    assert first["config"]["corpus_sha256"] == corpus_sha256()
    assert first["corpus_audit"] == corpus_audit()
    assert first["summary"] == {
        "passed": 72,
        "failed": 0,
        "pass_rate": 1.0,
        "by_category": {category: {"passed": 8, "failed": 0, "total": 8} for category in sorted(EXPECTED_CATEGORIES)},
    }


def test_report_is_json_serializable(tmp_path: Path):
    report = run_challenge(_first_allowed_parser, parser_name="test.first_allowed", cases=CASES[:2])
    output = tmp_path / "adversarial.json"
    write_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_negative_control_with_wrong_outputs_fails_and_serializes_reasons():
    def wrong_parser(message: str, previous: Query):
        return Query(intent="navigation", kind="course", seed_title="INJECTED"), None

    report = run_challenge(wrong_parser, parser_name="negative_control.always_wrong")
    assert report["summary"]["failed"] == len(CASES)
    failures = [item for item in report["cases"] if not item["passed"]]
    assert failures
    assert all(item["reason"] for item in failures)
    assert all(
        item.keys()
        == {"case_id", "category", "language_family_id", "checked_fields", "passed", "matched_outcome", "reason", "actual", "checks"}
        for item in failures
    )


def test_unmentioned_slots_cannot_be_invented_even_when_expected_fields_match():
    def contaminated(message, previous):
        query, issue = _first_allowed_parser(message, previous)
        return query.model_copy(update={"intent": "navigation", "seed_title": "INJECTED"}), issue

    report = run_challenge(contaminated)
    assert report["summary"]["failed"] == len(CASES)
    assert all(row["checks"] for row in report["cases"])


def test_whitespace_is_not_a_required_issue():
    item = next(item for item in CASES if item.case_id == "unsup-06")

    def empty_issue(message, previous):
        query, _ = _first_allowed_parser(message, previous)
        return query, " \t\n"

    assert run_challenge(empty_issue, cases=[item])["summary"]["failed"] == 1


def test_parser_exception_is_a_per_case_failure_not_a_runner_crash():
    def broken_parser(message: str, previous: Query):
        raise RuntimeError("boom")

    report = run_challenge(broken_parser, parser_name="negative_control.raises")
    assert report["summary"]["failed"] == len(CASES)
    assert report["cases"][0]["reason"] == "parser exception: RuntimeError: boom"
    assert report["cases"][0]["actual"] is None


def test_mutated_previous_and_constructed_invalid_query_cannot_bypass_runner():
    preserve_case = next(item for item in CASES if item.case_id == "corr-08")

    def mutating_parser(message: str, previous: Query):
        previous.kind = "course"
        return previous, None

    mutated = run_challenge(mutating_parser, parser_name="negative_control.mutates_previous", cases=[preserve_case])
    assert mutated["summary"]["failed"] == 1
    assert "query changed although previous must be preserved" in mutated["cases"][0]["reason"]

    def invalid_constructed_query(message: str, previous: Query):
        return Query.model_construct(max_minutes=0), None

    invalid = run_challenge(invalid_constructed_query, parser_name="negative_control.model_construct", cases=[CASES[0]])
    assert invalid["summary"]["failed"] == 1
    assert "parser exception: ValidationError" in invalid["cases"][0]["reason"]
