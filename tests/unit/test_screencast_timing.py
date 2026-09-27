"""Reading pauses in the submitted video must survive encoding preparation."""

import pytest
from scripts.assemble_screencast import frame_durations


def test_reading_pause_is_not_shortened_to_two_seconds():
    frames = [{"time": 0, "duration_seconds": 10}, {"time": 1000, "duration_seconds": 8}]
    assert frame_durations(frames, 12) == [10, 8]


def test_capture_gaps_use_configured_limit():
    assert frame_durations([{"time": 0}, {"time": 9000}, {"time": 29000}], 12) == [9, 12, 8]


@pytest.mark.parametrize("value", [0, -1, float("nan"), 31])
def test_invalid_duration_is_rejected(value):
    with pytest.raises(ValueError):
        frame_durations([{"time": 0, "duration_seconds": value}], 12)
