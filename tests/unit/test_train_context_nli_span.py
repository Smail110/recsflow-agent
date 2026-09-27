"""Boundary checks for the research citation pooling experiment."""

import pytest
from scripts.train_context_nli_span import citation_bounds, span_masks


def test_masks_use_sequence_and_exact_selected_occurrence():
    premise = "x <evidence>x</evidence>"
    start, end = citation_bounds(premise)
    citation, hypothesis = span_masks(
        premise, "x", [(0, 0), (0, 1), (start, end), (0, 1), (0, 0)],
        [None, 0, 0, 1, None],
    )
    assert citation == [False, False, True, False, False]
    assert hypothesis == [False, False, False, True, False]


@pytest.mark.parametrize("premise", ["x", "<evidence></evidence>", "<evidence>x</evidence><evidence>y</evidence>"])
def test_missing_empty_or_ambiguous_citation_fails(premise):
    with pytest.raises(ValueError):
        citation_bounds(premise)


@pytest.mark.parametrize("offset", [(9, 10), (10, 11)])
def test_crossing_marker_or_truncated_span_fails(offset):
    with pytest.raises(ValueError):
        span_masks("<evidence>xx</evidence>", "y", [offset, (0, 1)], [0, 1])


def test_truncated_hypothesis_fails():
    with pytest.raises(ValueError):
        span_masks("<evidence>x</evidence>", "yz", [(10, 11), (0, 1)], [0, 1])


def test_padding_and_whitespace_are_not_pooled():
    citation, hypothesis = span_masks(
        "<evidence>x y</evidence>", "z", [(10, 11), (12, 13), (0, 1), (0, 0)], [0, 0, 1, None],
    )
    assert citation == [True, True, False, False]
    assert hypothesis == [False, False, True, False]


def test_punctuation_merged_with_closing_marker_is_excluded():
    citation, hypothesis = span_masks(
        "<evidence>x.</evidence>", "z", [(10, 11), (11, 14), (0, 1)], [0, 0, 1],
    )
    assert citation == [True, False, False]
    assert hypothesis == [False, False, True]
