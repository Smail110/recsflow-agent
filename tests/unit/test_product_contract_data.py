"""Static provenance/feasibility checks, no runtime interpretation or inference."""

import hashlib
import json
from pathlib import Path

from evals.oracle import SpokenConstraints, satisfies_spoken
from scripts.build_product_contract import build_cohort, encoded

from recagent.catalog import generate_catalog

ROOT = Path(__file__).resolve().parents[2]


def test_contract_is_reproducible_versioned_and_not_human_authored():
    path = ROOT / "data/product_contract_ru_v1.json"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    payload = path.read_bytes()
    assert payload == encoded(build_cohort()) == encoded(build_cohort())
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    data = json.loads(payload)
    assert data["origin"] == "ai_authored_synthetic" and data["split"] == "contract"
    assert manifest["signature"]["cryptographic_author_signature"] is None
    assert len(data["dialogues"]) == 12


def test_contract_labels_have_literal_sources_and_executable_response_contracts():
    data = build_cohort()
    catalog = generate_catalog(data["catalog"]["seed"])
    by_id = {item.id: item for item in catalog}
    intents = set()
    for dialogue in data["dialogues"]:
        for index, turn in enumerate(dialogue["turns"]):
            assert ("expected_state" in turn) != ("allowed_states" in turn)
            assert ("recommendation_exists" in turn) != ("presence_by_state" in turn)
            for evidence in turn["source_annotations"]["evidence"]:
                assert evidence["turn"] <= index
                assert evidence["quote"] in dialogue["turns"][evidence["turn"]]["user"]
            spoken = SpokenConstraints.model_validate(turn["spoken"])
            feasible = [item for item in catalog if satisfies_spoken(item, spoken)[0]]
            if turn.get("recommendation_exists") or turn.get("presence_by_state", {}).get("recommend"):
                assert feasible, (dialogue["id"], index)
            for item_id in turn.get("exact_recommendation_ids", []):
                assert item_id in by_id and satisfies_spoken(by_id[item_id], spoken)[0]
            for prior in turn.get("disjoint_from_turns", []):
                assert 0 <= prior < index
                assert len(feasible) >= 10  # Two top-five slates are feasible.
            intents.add(turn["expected_query"]["intent"])
    assert intents == {"discovery", "navigation", "similar", "mood"}


def test_contract_annotations_do_not_convert_negative_tone_to_a_positive_guess():
    case = next(d for d in build_cohort()["dialogues"] if d["id"] == "negative-tone-more")
    assert all(turn["spoken"]["excluded_tones"] == ["мрачный"] for turn in case["turns"])
    assert all("tone" not in turn["spoken"] for turn in case["turns"])
