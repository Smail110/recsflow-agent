"""The submission report must show the selected run, without stale conclusions."""

from scripts.build_final_report import findings


def test_submission_snapshot_uses_selected_scores_and_check_counts():
    result = findings({
        "evaluation-final-20260927": {
            "kind": "submission_snapshot",
            "date": "2026-09-27",
            "quality": {"dev": {"passed": 14, "total": 20}, "contract": {"passed": 11, "total": 12}},
            "engineering": {"tests_passed": 1175, "subtests_passed": 7, "ruff": "PASS", "diff_check": "PASS"},
        }
    })
    assert "14/20" in result and "11/12" in result
    assert "1175 passed; 7 subtests" in result
    assert "исправление команды" not in result
    assert "вероятность успеха" in result
