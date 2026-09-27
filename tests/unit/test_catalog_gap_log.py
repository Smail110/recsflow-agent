"""Product demand log is local, bounded and separate from user identity."""

import json

from scripts.summarize_catalog_gaps import summarize

from recagent.observability.catalog_gaps import CatalogGapRecorder


def test_recorder_redacts_obvious_contact_details_and_counts_turns(tmp_path):
    path = tmp_path / "gaps.jsonl"
    recorder = CatalogGapRecorder(path)
    recorder.record(
        [
            {"reason": "unsupported_catalog_constraint", "field_hint": "genre", "wish": "про зомби"},
            {"reason": "unsupported_catalog_constraint", "field_hint": "genre", "wish": "про зомби"},
            {"reason": "unsupported_catalog_constraint", "field_hint": "genre", "wish": "test@example.org"},
        ],
        response_state="clarify", mode="ollama",
    )
    event = json.loads(path.read_text(encoding="utf-8"))
    assert "request_id" not in event
    assert event["wishes"][-1]["wish"] == "[email]"
    assert "test@example.org" not in path.read_text(encoding="utf-8")
    summary = summarize([path])
    assert summary["turns_with_catalog_gaps"] == 1
    assert summary["wish_turn_counts"]["про зомби"] == 1


def test_unwritable_sink_does_not_fail_chat_path(tmp_path):
    recorder = CatalogGapRecorder(tmp_path)
    recorder.record(
        [{"reason": "unsupported_catalog_constraint", "field_hint": "genre", "wish": "зомби"}],
        response_state="clarify", mode="ollama",
    )
    assert tmp_path.is_dir()
