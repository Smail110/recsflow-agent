from recagent.semantic import optional_embeddings, optional_nli, optional_reranker


def test_optional_adapters_have_no_runtime_side_effects():
    nli = optional_nli("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    result = nli.predict("premise", "hypothesis")
    assert result.label == "unavailable"
    assert result.status.status == "unavailable"
    assert optional_embeddings("intfloat/multilingual-e5-small").encode(["x"]) == []
    assert optional_reranker("BAAI/bge-reranker-v2-m3").score("q", ["d"]) == []


def test_enabled_without_runtime_is_explicitly_unavailable():
    adapter = optional_nli("model", enabled=True)
    assert adapter.predict("p", "h").status.reason == "weights_not_loaded"
