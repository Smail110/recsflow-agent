"""Pair construction checks for the experimental citation-margin trainer."""

import pytest
from scripts.train_context_nli_pair import citation_pairs


def _row(kind: str, label: str, *, message: str = "A B") -> dict[str, str]:
    return {
        "id": f"case-{kind}",
        "citation_kind": kind,
        "label": label,
        "message": message,
        "premise": f"{message} <evidence>{kind}</evidence>",
        "hypothesis": "claim",
        "citation_span_family": "value",
        "actual_operation": "set",
        "operation": "set",
        "split": "train",
    }


def test_citation_pairs_accepts_matched_pair_in_fixed_order():
    wrong = _row("counterfactual", "unknown")
    right = _row("primary", "support")
    assert citation_pairs([wrong, right]) == [(right, wrong)]


@pytest.mark.parametrize(
    "members",
    [
        [_row("primary", "support")],
        [_row("primary", "support"), _row("counterfactual", "support")],
        [_row("primary", "support"), _row("counterfactual", "unknown", message="other")],
    ],
)
def test_citation_pairs_rejects_unpaired_or_corrupt_examples(members):
    with pytest.raises(ValueError):
        citation_pairs(members)
