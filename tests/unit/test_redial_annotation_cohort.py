import copy
import json

import pytest
from scripts.build_redial_annotation_cohort import prepare


def dialogue(index):
    user, assistant = 2 * index, 2 * index + 1
    return {
        "conversationId": str(index),
        "initiatorWorkerId": user,
        "respondentWorkerId": assistant,
        "movieMentions": {"10": "Mentioned title", "20": "Future title"},
        "initiatorQuestions": {"20": {"liked": 1}},
        "respondentQuestions": [],
        "messages": [
            {"messageId": index * 10, "text": "I liked @10; anything funny?", "senderWorkerId": user},
            {"messageId": index * 10 + 1, "text": "Try @20", "senderWorkerId": assistant},
            {"messageId": index * 10 + 2, "text": "Wonderful!", "senderWorkerId": user},
        ],
    }


def config():
    return {"seed": 20260923, "splits": {"calibration": 2, "dev": 3, "validation": 2}}


def test_splits_are_repeatable_and_worker_disjoint():
    rows = [dialogue(i) for i in range(1, 20)]
    overlap = copy.deepcopy(rows[0])
    overlap["conversationId"] = "other"
    rows.append(overlap)
    first, summary = prepare(config(), rows)
    second, other_summary = prepare(config(), list(reversed(rows)))
    assert first == second and summary == other_summary
    by_split = {}
    for line in first["source-provenance.jsonl"].splitlines():
        row = json.loads(line)
        by_split.setdefault(row["split"], set()).update(row["worker_ids"])
    assert {k: len(v) for k, v in by_split.items()} == {"calibration": 4, "dev": 6, "validation": 4}
    assert len(set.union(*by_split.values())) == 14
    assert summary["final_holdout"] is False


def test_future_labels_responses_and_unmentioned_titles_do_not_change_inputs():
    rows = [dialogue(i) for i in range(1, 10)]
    before, _ = prepare(config(), rows)
    altered = copy.deepcopy(rows)
    for row in altered:
        row["initiatorQuestions"] = {"20": {"liked": 0}}
        row["respondentQuestions"] = {"20": {"seen": 1}}
        row["messages"][-1]["text"] = "Actually I hated it"
        row["movieMentions"]["20"] = "Different future title"
    after, _ = prepare(config(), altered)
    assert before == after
    assert set(before) == {"inputs.en.jsonl", "source-provenance.jsonl"}
    for line in before["inputs.en.jsonl"].splitlines():
        item = json.loads(line)
        assert len(item["messages"]) == 1
        assert [e["source_id"] for e in item["mentioned_entities"]] == ["10"]
        assert set(item["mentioned_entities"][0]["attributes"].values()) == {None}


def test_insufficient_unique_workers_is_an_error():
    rows = [dialogue(1)] * 10
    with pytest.raises(ValueError, match="worker-disjoint"):
        prepare(config(), rows)


@pytest.mark.parametrize(
    "splits", [{"dev": 7}, {"calibration": 2, "dev": 0, "validation": 2}, {"calibration": True, "dev": 3, "validation": 2}]
)
def test_invalid_split_configuration_fails(splits):
    with pytest.raises(ValueError, match="positive"):
        prepare({**config(), "splits": splits}, [])
