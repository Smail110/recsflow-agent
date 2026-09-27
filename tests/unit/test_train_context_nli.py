"""Input and checkpoint safeguards for the experimental context NLI trainer."""

import argparse
import json

import pytest
from scripts import train_context_nli as trainer


def _rows(split: str, family: str) -> list[dict[str, str]]:
    return [
        {
            "id": f"{split}-{index}",
            "split": split,
            "premise": f"{family}: сообщение {index}",
            "hypothesis": f"{family}: claim {index}",
            "label": label,
            "domain_family_id": family,
            "template_family_id": f"template-{family}",
            "schema_family_id": f"schema-{family}",
        }
        for index, label in enumerate(trainer.LABEL_TO_ID)
    ]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_pretrained_head_label_mapping_is_preserved():
    assert trainer.LABEL_TO_ID == {"support": 0, "unknown": 1, "contradiction": 2}
    assert trainer.ID_TO_LABEL == {0: "support", 1: "unknown", 2: "contradiction"}


def test_snapshot_pins_every_tokenizer_input(tmp_path, monkeypatch):
    snapshot = tmp_path / trainer.REVISION
    snapshot.mkdir()
    config = {"model_type": "deberta-v2", "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
    for name in trainer.PINNED_HASHES:
        (snapshot / name).write_bytes(json.dumps(config).encode() if name == "config.json" else name.encode())
    monkeypatch.setattr(
        trainer, "PINNED_HASHES",
        {name: trainer.sha256_file(snapshot / name) for name in trainer.PINNED_HASHES},
    )
    assert set(trainer.validate_snapshot(snapshot)) == {
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "spm.model",
    }
    (snapshot / "spm.model").write_bytes(b"changed tokenizer model")
    with pytest.raises(ValueError, match="SHA-256"):
        trainer.validate_snapshot(snapshot)


def test_split_validation_rejects_shared_semantic_family():
    with pytest.raises(ValueError, match="split overlap"):
        trainer.validate_splits(_rows("train", "shared"), _rows("dev", "shared"))
    report = trainer.validate_splits(_rows("train", "one"), _rows("dev", "two"))
    assert report["overlap"]["premise"] == 0
    assert report["overlap"]["domain_family_id"] == 0


def test_blind_input_is_rejected_before_hashing(tmp_path, monkeypatch):
    def fail_if_hashed(_path):
        raise AssertionError("sealed path was read")

    monkeypatch.setattr(trainer, "sha256_file", fail_if_hashed)
    args = argparse.Namespace(
        train=tmp_path / "blind.jsonl", dev=tmp_path / "dev.jsonl",
        expected_train_sha256="0" * 64, expected_dev_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="sealed"):
        trainer.validate_inputs(args)


def test_hash_gate_and_three_class_input(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    _write_jsonl(train, _rows("train", "one"))
    _write_jsonl(dev, _rows("dev", "two"))
    args = argparse.Namespace(
        train=train, dev=dev, snapshot=tmp_path / "snapshot", output_dir=tmp_path / "out",
        expected_train_sha256="0" * 64, expected_dev_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        trainer.validate_inputs(args)
    args.expected_train_sha256 = trainer.sha256_file(train)
    args.expected_dev_sha256 = trainer.sha256_file(dev)
    monkeypatch.setattr(trainer, "validate_snapshot", lambda _path: {"model.safetensors": "pinned"})
    train_rows, dev_rows, receipt = trainer.validate_inputs(args)
    assert len(train_rows) == len(dev_rows) == 3
    assert receipt["train_sha256"] == args.expected_train_sha256
    assert receipt["split"]["overlap"]["pair"] == 0


def test_sqrt_inverse_loss_weights_use_train_counts_only():
    train = ([{"label": "support"}] * 4 + [{"label": "unknown"}] * 16 + [{"label": "contradiction"}] * 9)
    weights = trainer.loss_weights(train, "sqrt-inverse")
    assert sum(weights.values()) / 3 == pytest.approx(1.0)
    assert weights["support"] > weights["contradiction"] > weights["unknown"]
    assert weights["support"] / weights["unknown"] == pytest.approx(2.0)
    assert trainer.loss_weights(train, "none") == dict.fromkeys(trainer.LABEL_TO_ID, 1.0)
    with pytest.raises(ValueError, match="unknown loss-weighting"):
        trainer.loss_weights(train, "inverse")
