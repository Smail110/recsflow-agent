from __future__ import annotations

import copy
import json
import zipfile

import httpx
import pytest
from scripts.prepare_redial_pilot import (
    build,
    encoded,
    fetch,
    load_training,
    prepare,
    sha256,
    validate_dialogue,
)


def dialogue(index=1):
    first, second = index * 2, index * 2 + 1
    return {
        "conversationId": str(index),
        "initiatorWorkerId": first,
        "respondentWorkerId": second,
        "movieMentions": {"10": "Known movie", "20": "Future movie"},
        "initiatorQuestions": {"20": {"liked": 1, "seen": 0, "suggested": 1}},
        "respondentQuestions": [],
        "messages": [
            {"messageId": 1, "text": "I liked @10. Something funny?", "senderWorkerId": first, "timeOffset": 0},
            {"messageId": 2, "text": "Try @20", "senderWorkerId": second, "timeOffset": 1},
            {"messageId": 3, "text": "I loved that recommendation!", "senderWorkerId": first, "timeOffset": 2},
        ],
    }


def config():
    return {
        "seed": 42,
        "sample_size": 2,
        "schema_version": "test",
        "source": {},
        "selection": "test",
        "prefix": "test",
        "labels": "test",
        "split": "test",
    }


def archive_fixture(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "redial_dataset.zip"
    train = b"".join(encoded(dialogue(i)).replace(b"\n", b" ") + b"\n" for i in range(1, 5))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train_data.jsonl", train)
        archive.writestr("test_data.jsonl", "deliberately invalid; must not be read")
        archive.writestr("../../escape.txt", "must not extract")
    cfg = config()
    cfg["source"] = {
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256(path.read_bytes()),
        "train_member": "train_data.jsonl",
        "train_bytes": len(train),
        "train_sha256": sha256(train),
        "train_rows": 4,
        "max_train_bytes": 100000,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_bytes(encoded(cfg))
    return cache, cfg, cfg_path


def test_inputs_do_not_contain_future_messages_labels_or_entities():
    files, summary = prepare(config(), [dialogue(1), dialogue(2)])
    inputs = [json.loads(line) for line in files["inputs.en.jsonl"].splitlines()]
    for row in inputs:
        assert set(row) == {"id", "source_language", "messages", "mentioned_entities"}
        assert len(row["messages"]) == 1
        assert [entity["source_id"] for entity in row["mentioned_entities"]] == ["10"]
        assert set(row["mentioned_entities"][0]["attributes"].values()) == {None}
    labels = json.loads(files["observed-annotations.DO-NOT-PROMPT.jsonl"].splitlines()[0])
    assert labels["initiator_questions_raw"]["20"]["liked"] == 1
    assert labels["prefix_ground_truth"] is False
    assert labels["permitted_as_agent_input"] is False
    assert summary["human_annotations"] == 0
    template = json.loads(files["annotation-template.jsonl"].splitlines()[0])
    assert template["gold_labels"] is template["translation_ru"] is None


def test_prefix_is_invariant_to_later_text_and_questionnaire_changes():
    rows = [dialogue(1), dialogue(2)]
    before, _ = prepare(config(), rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["messages"][-1]["text"] = "Different future preference"
        row["initiatorQuestions"]["20"]["liked"] = 0
        row["movieMentions"]["20"] = "Different future title"
    after, _ = prepare(config(), changed)
    assert before["inputs.en.jsonl"] == after["inputs.en.jsonl"]
    assert before["observed-annotations.DO-NOT-PROMPT.jsonl"] != after["observed-annotations.DO-NOT-PROMPT.jsonl"]


def test_seed_selection_order_independent_and_worker_disjoint():
    rows = [dialogue(i) for i in range(1, 20)]
    overlapping = copy.deepcopy(rows[0])
    overlapping["conversationId"] = "100"
    rows.append(overlapping)
    first_files, first = prepare(config(), rows)
    second_files, second = prepare(config(), list(reversed(rows)))
    assert first_files == second_files
    assert first == second
    assert len(first["selected_worker_ids"]) == 2 * first["selected_conversations"]
    selected_workers = set(first["selected_worker_ids"])
    assert set(first["future_eval_excluded_training_conversation_ids"]) == {
        row["conversationId"] for row in rows if selected_workers.intersection({row["initiatorWorkerId"], row["respondentWorkerId"]})
    }


def test_insufficient_disjoint_groups_fail_not_reuse_workers():
    row = dialogue()
    other = copy.deepcopy(row)
    other["conversationId"] = "2"
    with pytest.raises(ValueError, match="worker-disjoint"):
        prepare(config(), [row, other])


def test_official_empty_questionnaires_and_unknown_title_preserved():
    row = dialogue()
    row["movieMentions"]["10"] = None
    validate_dialogue(row)
    cfg = config()
    cfg["sample_size"] = 1
    files, _ = prepare(cfg, [row])
    assert json.loads(files["inputs.en.jsonl"])["mentioned_entities"][0]["title"] is None


@pytest.mark.parametrize("mutation", ["label", "sender", "duplicate_message"])
def test_invalid_schema_rejected(mutation):
    row = dialogue()
    if mutation == "label":
        row["initiatorQuestions"]["20"]["liked"] = 3
    elif mutation == "sender":
        row["messages"][0]["senderWorkerId"] = 999
    else:
        row["messages"][1]["messageId"] = 1
    with pytest.raises(ValueError):
        validate_dialogue(row)


def test_builds_byte_identical_and_only_training_member_read(tmp_path, monkeypatch):
    cache, cfg, cfg_path = archive_fixture(tmp_path)
    original = zipfile.ZipFile.read
    opened = []

    def guarded_read(self, name, *args, **kwargs):
        opened.append(name)
        assert name == "train_data.jsonl"
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    first = build(cfg_path, cache, tmp_path / "first")
    second = build(cfg_path, cache, tmp_path / "second")
    assert first == second
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
    assert opened == ["train_data.jsonl", "train_data.jsonl"]
    with pytest.raises(FileExistsError):
        build(cfg_path, cache, tmp_path / "first")
    cfg["source"]["train_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training size/SHA"):
        load_training(cache / "redial_dataset.zip", cfg["source"])


def test_archive_tampering_rejected(tmp_path):
    cache, cfg, _ = archive_fixture(tmp_path)
    path = cache / "redial_dataset.zip"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="archive SHA"):
        load_training(path, cfg["source"])


def test_fetch_checks_digest_and_never_publishes_bad_bytes(tmp_path, monkeypatch):
    payload = b"fake pinned bytes"
    cfg = config()
    cfg["source"] = {
        "url": "https://example.test/source.zip",
        "archive_sha256": sha256(payload),
        "archive_bytes": len(payload),
        "max_download_bytes": 100,
    }
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        result = fetch(cfg, tmp_path / "good")
        assert result["cache_reused"] is False
        assert fetch(cfg, tmp_path / "good")["cache_reused"] is True
        cfg["source"]["archive_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="download size/SHA"):
            fetch(cfg, tmp_path / "bad")
        assert not (tmp_path / "bad/redial_dataset.zip").exists()


def test_prefix_unknown_entity_fails_closed():
    row = dialogue()
    row["messages"][0]["text"] = "I like @999"
    cfg = config()
    cfg["sample_size"] = 1
    with pytest.raises(ValueError, match="missing from source"):
        prepare(cfg, [row])
