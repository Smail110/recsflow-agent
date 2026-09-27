"""Matched citation-control scoring has one unit per utterance/span pair."""

from scripts.eval_massive_citation_nli import paired_metrics


def test_paired_metrics_count_exact_support_and_wrong_citation():
    rows = [
        {"group_id": "one", "label": "support"},
        {"group_id": "one", "label": "unknown"},
        {"group_id": "two", "label": "support"},
        {"group_id": "two", "label": "unknown"},
    ]
    result = paired_metrics(rows, ["support", "unknown", "support", "support"])
    assert result["groups"] == 2
    assert result["support_recall"] == 1
    assert result["wrong_citation_false_support_rate"] == 0.5
    assert result["both_exact_rate"] == 0.5
    assert result["prediction_flip_rate"] == 0.5
