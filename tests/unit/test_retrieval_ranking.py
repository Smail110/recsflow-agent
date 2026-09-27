from recagent.ranking import reciprocal_rank_fusion
from recagent.retrieval import BM25Index, tokenize


def test_tokenizer_is_casefolded_and_normalizes_yo():
    assert tokenize("Ёлка ABC-12") == ("елка", "abc", "12")


def test_bm25_is_deterministic_and_uses_id_tie_break():
    hits = BM25Index({"b": "cat", "a": "cat", "x": "dog"}).search("cat")
    assert [h.item_id for h in hits] == ["a", "b"]
    assert hits[0].score == hits[1].score


def test_rrf_preserves_original_rank_after_filter_and_deduplicates():
    result = reciprocal_rank_fusion(
        [[{"item_id": "x"}, {"item_id": "drop"}, {"item_id": "a"}], [{"item_id": "a"}, {"item_id": "x"}]], eligible_ids={"x", "a"}, k=60
    )
    assert result == ["x", "a"]
