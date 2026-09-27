import hashlib
import json
from pathlib import Path

import pytest
from scripts.reproduce_legacy_dialogues import (
    LEGACY_CANONICAL_SHA256,
    LegacyReproductionError,
    reproduce,
    verify_legacy_reference,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data" / "dialogues.jsonl"


def test_historical_generator_exactly_reproduces_tracked_snapshot():
    report = verify_legacy_reference(REFERENCE)

    assert report["canonical_sha256"] == LEGACY_CANONICAL_SHA256
    assert report["rows"] == 160
    assert report["labels"] == {"discovery": 40, "mood": 40, "navigation": 40, "similar": 40}
    assert report["duplicate_rows_by_text"] > 0


def test_reproduction_writes_canonical_lf_without_changing_legacy_schema(tmp_path: Path):
    output = tmp_path / "legacy-dialogues.jsonl"
    reproduce(output, REFERENCE)

    payload = output.read_bytes()
    assert b"\r\n" not in payload
    assert hashlib.sha256(payload).hexdigest() == LEGACY_CANONICAL_SHA256
    rows = [json.loads(line) for line in payload.splitlines()]
    assert all(set(row) == {"text", "label", "synthetic", "seed"} for row in rows)
    assert all(row["synthetic"] is True and row["seed"] == 42 for row in rows)


def test_reference_drift_fails_before_output_is_written(tmp_path: Path):
    changed_reference = tmp_path / "changed.jsonl"
    changed_reference.write_bytes(REFERENCE.read_bytes().replace("лёгкое".encode(), "яркое".encode(), 1))
    output = tmp_path / "output.jsonl"

    with pytest.raises(LegacyReproductionError, match="reference mismatch"):
        reproduce(output, changed_reference)
    assert not output.exists()
