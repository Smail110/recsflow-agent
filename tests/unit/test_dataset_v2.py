import json
from pathlib import Path

import pytest
from evals.dataset_v2 import (
    DATASET_VERSION,
    V2Manifest,
    generate_public_dataset,
    load_cases,
    load_dataset,
    load_manifest,
    near_duplicate_pairs,
    normalize_text,
    sha256_value,
    validate_cases,
    validate_split_isolation,
    write_dataset,
)
from scripts.evaluate_dataset_v2 import _consume_final, mutation_controls

from recagent.catalog import generate_catalog


@pytest.fixture(scope="module")
def public_splits():
    dev, dev_summary = generate_public_dataset("dev", per_family=3)
    validation, validation_summary = generate_public_dataset("validation", per_family=3)
    return dev, validation, dev_summary, validation_summary


def test_public_splits_have_one_profile_per_case_and_real_family_isolation(public_splits):
    dev, validation, dev_summary, validation_summary = public_splits
    assert len(dev) == len(validation) == 3 * 16
    assert {case.scenario.split for case in dev} == {"dev"}
    assert {case.scenario.split for case in validation} == {"validation"}
    assert dev_summary["capacity"] == validation_summary["capacity"] == 48
    assert len({case.scenario.user_id for case in dev}) == len(dev)
    assert len({case.scenario.user_id for case in validation}) == len(validation)
    assert {case.scenario_family for case in dev} == {case.scenario_family for case in validation}
    assert not ({case.surface_family_id for case in dev} & {case.surface_family_id for case in validation})
    assert not ({normalize_text(case.surface_pattern) for case in dev} & {normalize_text(case.surface_pattern) for case in validation})
    result = validate_split_isolation(dev, validation)
    assert all(result[key] == 0 for key in ("users", "surface_families", "patterns", "dialogues", "near_patterns"))


def test_roundtrip_manifest_and_tamper_detection(public_splits, tmp_path: Path):
    dev, _, _, _ = public_splits
    cases_path, manifest_path = write_dataset(dev, tmp_path, split="dev")
    manifest, restored = load_dataset(manifest_path)
    assert manifest.dataset_version == DATASET_VERSION
    assert manifest.case_count == len(restored) == len(dev)
    assert manifest.unique_users == len(dev)
    assert [case.model_dump(mode="json") for case in restored] == [case.model_dump(mode="json") for case in dev]

    cases_path.write_bytes(cases_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="cases hash mismatch"):
        load_dataset(manifest_path)


def test_manifest_hash_tamper_is_rejected(public_splits, tmp_path: Path):
    dev, _, _, _ = public_splits
    _, manifest_path = write_dataset(dev, tmp_path, split="dev")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["profile_seed"] += 1
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_dataset(manifest_path)


def test_duplicate_user_and_normalized_dialogue_are_rejected(public_splits):
    dev, _, _, _ = public_splits
    duplicate_user = dev[1].model_copy(update={"scenario": dev[1].scenario.model_copy(update={"user_id": dev[0].scenario.user_id})})
    with pytest.raises(ValueError, match="one case per independent user"):
        validate_cases([dev[0], duplicate_user])

    first_turn = dev[1].scenario.turns[0].model_copy(update={"utterance": dev[0].scenario.turns[0].utterance.upper() + "!!!"})
    duplicate_text = dev[1].model_copy(
        update={"scenario": dev[1].scenario.model_copy(update={"turns": [first_turn, *dev[0].scenario.turns[1:]]})}
    )
    # Make the complete dialogue identical after normalization, even for multi-turn cases.
    duplicate_text = duplicate_text.model_copy(
        update={
            "scenario": duplicate_text.scenario.model_copy(
                update={
                    "turns": [
                        turn.model_copy(update={"utterance": original.utterance.upper() + "!!!"})
                        for turn, original in zip(duplicate_text.scenario.turns, dev[0].scenario.turns, strict=False)
                    ]
                }
            )
        }
    )
    if len(duplicate_text.scenario.turns) == len(dev[0].scenario.turns):
        with pytest.raises(ValueError, match="exact normalized dialogue duplicates"):
            validate_cases([dev[0], duplicate_text])


def test_surface_semantics_fail_closed_when_a_spoken_slot_disappears(public_splits):
    dev, _, _, _ = public_splits
    original = next(case for case in dev if case.scenario.final_spoken.genre is not None)
    turns = [turn.model_copy(update={"utterance": "Посоветуйте что-нибудь."}) for turn in original.scenario.turns]
    broken = original.model_copy(update={"scenario": original.scenario.model_copy(update={"turns": turns})})
    with pytest.raises(ValueError, match="surface text loses ground truth"):
        validate_cases([broken])


def test_near_duplicate_patterns_and_split_leakage_fail_closed(public_splits):
    dev, validation, _, _ = public_splits
    copied = validation[0].model_copy(
        update={
            "surface_pattern": dev[0].surface_pattern,
            "surface_family_id": dev[0].surface_family_id,
            "scenario": validation[0].scenario.model_copy(update={"user_id": dev[0].scenario.user_id}),
        }
    )
    mutated = [copied, *validation[1:]]
    assert near_duplicate_pairs([dev[0]], [copied], threshold=0.88)
    with pytest.raises(ValueError, match="split leakage detected"):
        validate_split_isolation(dev, mutated)


def test_final_holdout_is_external_and_requires_explicit_guard(public_splits, tmp_path: Path):
    _, validation, _, _ = public_splits
    _, manifest_path = write_dataset(validation, tmp_path, split="validation")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["split"] = "final_holdout"
    payload["origin"] = "independent_external"
    cases_path = manifest_path.parent / payload["cases_file"]
    external_cases = [
        case.model_copy(update={"scenario": case.scenario.model_copy(update={"split": "final_holdout"})}) for case in validation
    ]
    case_bytes = "".join(case.model_dump_json() + "\n" for case in external_cases).encode("utf-8")
    cases_path.write_bytes(case_bytes)
    from evals.dataset_v2 import sha256_bytes

    payload["cases_sha256"] = sha256_bytes(case_bytes)
    payload["manifest_sha256"] = sha256_value({key: value for key, value in payload.items() if key != "manifest_sha256"})
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(PermissionError, match="explicit consumption guard"):
        load_manifest(manifest_path)
    manifest = load_manifest(manifest_path, allow_final_holdout=True)
    with pytest.raises(PermissionError, match="case bytes are loader-only"):
        load_dataset(manifest_path, allow_final_holdout=True)
    assert manifest.split == "final_holdout"
    with pytest.raises(PermissionError, match="exact cases SHA256"):
        _consume_final(manifest_path, manifest, "wrong")
    record = _consume_final(manifest_path, manifest, manifest.cases_sha256)
    assert record.exists()
    assert len(load_cases(manifest_path, manifest)) == len(validation)
    with pytest.raises(PermissionError, match="already consumed"):
        _consume_final(manifest_path, manifest, manifest.cases_sha256)


def test_mutation_controls_cover_bad_outputs(public_splits):
    dev, _, _, _ = public_splits
    result = mutation_controls(dev, generate_catalog(4202))
    assert result["oracle_positive"]
    assert all(result.values())


def test_final_holdout_cannot_be_generated():
    with pytest.raises(ValueError):
        generate_public_dataset("final_holdout")  # type: ignore[arg-type]


def test_manifest_rejects_unversioned_hashes():
    with pytest.raises(ValueError):
        V2Manifest.model_validate(
            {
                "split": "dev",
                "origin": "synthetic_code",
                "cases_file": "cases.jsonl",
                "qa_sample_file": "qa.md",
                "cases_sha256": "bad",
                "manifest_sha256": "bad",
                "case_count": 1,
                "unique_users": 1,
                "catalog_seed": 1,
                "catalog_sha256": "bad",
                "profile_seed": 2,
                "scenario_seed": 3,
                "by_family": {"x": 1},
                "surface_families": ["x"],
                "normalized_dialogues_sha256": "bad",
                "statistical_target": {},
                "limitations": [],
            }
        )
