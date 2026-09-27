from __future__ import annotations

import copy

import pytest
from evals.redial_annotation import Annotation, annotation_payload, build_prompt, consensus, validate_annotation


def prefix(text="I want a surrealist comedy."):
    return {
        "id": "redial-train-123",
        "source_language": "en",
        "mentioned_entities": [],
        "messages": [
            {"role": "assistant", "source_message_id": 8, "text": "I recommend horror."},
            {"role": "user", "source_message_id": 9, "text": text},
        ],
    }


def annotation(**fact_changes):
    return {
        "status": "explicit",
        "facts": [
            {
                "predicate": "inclusion",
                "attribute": "genre",
                "target": "surrealist comedy",
                "message_id": 9,
                "quote": "I want a surrealist comedy.",
                **fact_changes,
            }
        ],
    }


def test_exact_unknown_domain_surface_and_offsets_are_preserved():
    result = validate_annotation(prefix(), annotation())
    assert result["valid"]
    fact = result["anchored_facts"][0]
    assert fact["target"] == "surrealist comedy"
    assert fact["target_span"] == {"start": 9, "end": 26}
    assert fact["quote_span"] == {"start": 0, "end": 27}


@pytest.mark.parametrize(
    "changes",
    [
        {"message_id": 8, "quote": "I recommend horror.", "target": "horror"},
        {"message_id": 99},
        {"quote": "I would like surreal comedy"},
        {"target": "comedy", "quote": "I want a surrealist comedy!"},
        {"target": "romance"},
        {"message_id": "9"},
        {"message_id": True},
        {"quote_span": {"start": 0, "end": 1}},
        {"attribute": "demo_genre"},
        {"predicate": "maybe_like"},
        {"target": " "},
    ],
)
def test_invalid_source_or_schema_never_accepted(changes):
    raw = annotation(**changes)
    assert not validate_annotation(prefix(), raw)["valid"]
    assert consensus(prefix(), raw, raw)["status"] == "REVIEW"


def test_repeated_quote_is_ambiguous_not_first_occurrence():
    raw = annotation(quote="comedy", target="comedy")
    assert not validate_annotation(prefix("comedy or comedy"), raw)["valid"]


def test_repeated_target_inside_unique_quote_requires_review():
    raw = annotation(quote="comedy or comedy", target="comedy")
    assert not validate_annotation(prefix("comedy or comedy"), raw)["valid"]


def test_offsets_are_unicode_codepoints_not_utf16():
    raw = annotation(quote="🙂 comedy", target="comedy")
    assert validate_annotation(prefix("🙂 comedy"), raw)["anchored_facts"][0]["target_span"] == {"start": 2, "end": 8}


def test_duplicate_assertions_rejected():
    raw = annotation()
    raw["facts"].append(copy.deepcopy(raw["facts"][0]))
    assert not validate_annotation(prefix(), raw)["valid"]


def test_seen_not_seen_and_affinity_remain_distinct():
    source = prefix("I have not seen @42 and I like @7.")
    raw = {
        "status": "explicit",
        "facts": [
            {"predicate": "not_seen", "attribute": "reference", "target": "@42", "message_id": 9, "quote": "I have not seen @42"},
            {"predicate": "likes", "attribute": "reference", "target": "@7", "message_id": 9, "quote": "I like @7."},
        ],
    }
    result = consensus(source, raw, copy.deepcopy(raw))
    assert result["status"] == "ACCEPT"
    assert result["eligible_for_scoring"]
    assert [f["predicate"] for f in result["annotation"]["facts"]] == ["not_seen", "likes"]


@pytest.mark.parametrize("status", ["no_explicit_preferences", "unclear"])
def test_empty_agreement_is_not_scoring_eligible(status):
    raw = {"status": status, "facts": []}
    result = consensus(prefix("Hello there"), raw, raw)
    assert result["status"] == "ACCEPT"
    assert result["unknown"]
    assert not result["eligible_for_scoring"]
    assert result["annotation"]["status"] == status


def test_explicit_empty_and_no_preference_nonempty_are_invalid():
    assert not validate_annotation(prefix(), {"status": "explicit", "facts": []})["valid"]
    raw = annotation()
    raw["status"] = "no_explicit_preferences"
    assert not validate_annotation(prefix(), raw)["valid"]


def test_unknown_explicit_assertion_keeps_surface_but_not_scoring_eligibility():
    raw = annotation(predicate="unknown")
    result = consensus(prefix(), raw, raw)
    assert result["status"] == "ACCEPT"
    assert result["annotation"]["facts"][0]["target"] == "surrealist comedy"
    assert not result["eligible_for_scoring"]


def test_disagreement_does_not_get_adjudicated_by_guessing():
    primary = annotation()
    verifier = annotation(predicate="preference")
    result = consensus(prefix(), primary, verifier)
    assert result["status"] == "REVIEW"
    assert result["annotation"] is None
    assert result["validations"]["primary"]["annotation"] == primary
    assert result["validations"]["verifier"]["annotation"] == verifier


def test_fact_order_is_not_a_semantic_disagreement():
    source = prefix("I like comedy and dislike horror.")
    raw = {
        "status": "explicit",
        "facts": [
            {"predicate": "likes", "attribute": "genre", "target": "comedy", "message_id": 9, "quote": "I like comedy"},
            {"predicate": "dislikes", "attribute": "genre", "target": "horror", "message_id": 9, "quote": "dislike horror."},
        ],
    }
    other = copy.deepcopy(raw)
    other["facts"].reverse()
    assert consensus(source, raw, other)["status"] == "ACCEPT"


def test_different_valid_quote_windows_same_assertion_anchor_agree():
    primary = annotation()
    verifier = annotation(quote="want a surrealist comedy")
    result = consensus(prefix(), primary, verifier)
    assert result["status"] == "ACCEPT"
    assert result["eligible_for_scoring"]
    left = result["validations"]["primary"]["anchored_facts"][0]
    right = result["validations"]["verifier"]["anchored_facts"][0]
    assert left["target_span"] == right["target_span"]
    assert left["quote_span"] != right["quote_span"]
    assert result["quote_window_variants_allowed"]
    assert "quote" not in result["agreement_fields"]


def test_different_target_occurrences_remain_disagreement():
    source = prefix("I want comedy, and comedy is nice.")
    primary = annotation(quote="I want comedy", target="comedy")
    verifier = annotation(quote="comedy is nice", target="comedy")
    result = consensus(source, primary, verifier)
    assert all(v["valid"] for v in result["validations"].values())
    assert result["status"] == "REVIEW"


@pytest.mark.parametrize("change", ["test", "future_key", "duplicate_id", "assistant_last", "metadata"])
def test_prefix_is_strict_train_only_without_future_or_guessed_metadata(change):
    source = prefix()
    if change == "test":
        source["id"] = "redial-test-123"
    elif change == "future_key":
        source["observed_annotations"] = {"future": "answer"}
    elif change == "duplicate_id":
        source["messages"][0]["source_message_id"] = 9
    elif change == "assistant_last":
        source["messages"][-1]["role"] = "assistant"
    else:
        source["mentioned_entities"] = [
            {"source_id": "42", "title": "Movie", "attributes": {"genre": "comedy"}, "catalog_mapping_id": None}
        ]
    with pytest.raises(ValueError):
        annotation_payload(source)
    assert not validate_annotation(source, annotation())["valid"]


def test_payload_keeps_prior_context_but_null_metadata_only():
    source = prefix("I like @42")
    source["mentioned_entities"] = [
        {"source_id": "42", "title": "A movie (1992)", "attributes": {"genre": None, "year": None}, "catalog_mapping_id": None}
    ]
    assert annotation_payload(source) == source


def test_open_split_metadata_is_valid_but_not_sent_to_annotator():
    source = prefix()
    source["split"] = "calibration"
    assert "split" not in annotation_payload(source)
    assert validate_annotation(source, annotation())["valid"]
    source["split"] = "final_holdout"
    with pytest.raises(ValueError):
        annotation_payload(source)


def test_wire_schema_is_compact_and_forbids_offsets():
    schema = Annotation.model_json_schema()
    assert set(schema["$defs"]["Fact"]["properties"]) == {"predicate", "attribute", "target", "message_id", "quote"}
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Fact"]["additionalProperties"] is False
    assert "not_seen" in schema["$defs"]["Fact"]["properties"]["predicate"]["enum"]


def test_both_prompts_independent_and_do_not_contain_other_annotation():
    primary, verifier = build_prompt("primary"), build_prompt("verifier")
    assert primary != verifier
    assert "from scratch" in verifier
    assert "not human gold" in primary
    assert "Do not invent later replies" in verifier
    assert ".\nReturn only" in primary
