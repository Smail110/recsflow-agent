"""Checks for deterministic MASSIVE citation-pair construction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import build_massive_citation_eval as massive
from scripts.build_massive_citation_eval import (
    build_pairs,
    parse_annotated_utterance,
    slot_quality,
    tag_citation,
)


def _record(**changes: object) -> dict:
    row = {
        "id": "42",
        "locale": "ru-RU",
        "partition": "dev",
        "utt": "поставь песню завтра вечером",
        "annot_utt": "поставь [song_name : песню] [date : завтра] [time : вечером]",
        "judgments": {"slots_score": [1, 1, 0]},
    }
    row.update(changes)
    return row


def test_parser_recovers_exact_unicode_offsets_and_rejects_alignment_errors() -> None:
    row = _record()
    spans = parse_annotated_utterance(row["utt"], row["annot_utt"])
    assert [(span.role, span.value, row["utt"][span.start : span.end]) for span in spans] == [
        ("song_name", "песню", "песню"),
        ("date", "завтра", "завтра"),
        ("time", "вечером", "вечером"),
    ]
    assert tag_citation(row["utt"], spans[1]) == "поставь песню <evidence>завтра</evidence> вечером"
    with pytest.raises(ValueError, match="does not reproduce"):
        parse_annotated_utterance(row["utt"], row["annot_utt"].replace("завтра", "сегодня"))
    with pytest.raises(ValueError, match="unclosed"):
        parse_annotated_utterance("текст", "[date : текст")


def test_each_pair_keeps_same_claim_and_changes_only_citation() -> None:
    rows, coverage = build_pairs([_record()])
    assert coverage["matched_groups"] == 3
    assert coverage["derived_rows"] == 6
    assert Counter(row["label"] for row in rows) == {"support": 3, "unknown": 3}
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert first["label"] == "support"
        assert second["label"] == "unknown"
        assert first["group_id"] == second["group_id"]
        assert first["message"] == second["message"]
        assert first["hypothesis"] == second["hypothesis"]
        assert first["premise"] != second["premise"]
        for row in (first, second):
            assert row["message"][row["source_start"] : row["source_end"]] == row["source_text"]


def test_coverage_counts_exclusions_and_never_uses_missing_slots_as_negative() -> None:
    rows, coverage = build_pairs(
        [
            _record(),
            _record(id="43", utt="поставь песню", annot_utt="поставь [song_name : песню]"),
            _record(id="44", judgments={"slots_score": [1, 0, 0]}),
            _record(id="45", annot_utt="сломано"),
            _record(id="46", utt="ничего", annot_utt="ничего"),
        ]
    )
    assert len(rows) == 6
    assert coverage["source_utterances"] == 5
    assert coverage["eligible_utterances"] == 1
    assert coverage["excluded_utterances_by_reason"] == {
        "fewer_than_two_slot_spans": 2,
        "invalid_markup_or_alignment": 1,
        "slot_quality_not_majority_valid": 1,
    }
    assert not slot_quality({"slots_score": [1, 0, 0]})
    assert slot_quality({"slots_score": [1, 1, 0]})


def test_duplicate_ids_and_wrong_partition_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_pairs([_record(), _record()])
    with pytest.raises(ValueError, match="pinned Russian validation"):
        build_pairs([_record(partition="test")])
    with pytest.raises(ValueError, match="pinned Russian validation"):
        build_pairs([_record(locale="en-US")])


def test_repeated_surface_can_still_use_offsets_but_same_role_value_is_not_wrong_citation() -> None:
    row = _record(
        utt="завтра и завтра",
        annot_utt="[date : завтра] и [date : завтра]",
    )
    rows, coverage = build_pairs([row])
    assert rows == []
    assert coverage["excluded_utterances_by_reason"] == {"no_distinct_role_value_counterfactual": 1}


def test_pinned_bytes_manifest_replay_and_existing_output_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([_record()]), source_path)
    source_bytes = source_path.read_bytes()
    monkeypatch.setattr(massive, "SOURCE_SHA256", massive.sha256(source_bytes))
    monkeypatch.setattr(massive, "SOURCE_BYTES", len(source_bytes))
    monkeypatch.setattr(massive, "SOURCE_ROWS", 1)

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = massive.build(first, source_path=source_path)
    massive.build(second, source_path=source_path)
    assert (first / "pairs.jsonl").read_bytes() == (second / "pairs.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert manifest["files"]["pairs.jsonl"]["sha256"] == massive.sha256((first / "pairs.jsonl").read_bytes())
    assert manifest["coverage"]["matched_groups"] == 3
    with pytest.raises(FileExistsError):
        massive.build(first, source_path=source_path)

    source_path.write_bytes(source_bytes + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        massive.build(tmp_path / "corrupt", source_path=source_path)
    assert not (tmp_path / "corrupt").exists()
