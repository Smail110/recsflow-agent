import pytest
from scripts.evaluate_nli_contract import judgement, label_mapping


class Config:
    def __init__(self, mapping):
        self.id2label = mapping


def test_nli_mapping_reads_checkpoint_order_instead_of_assuming_it():
    assert label_mapping(Config({0: "contradiction", 1: "neutral", 2: "entailment"})) == {
        0: "contradiction",
        1: "neutral",
        2: "entailment",
    }
    with pytest.raises(ValueError, match="Unverified"):
        label_mapping(Config({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))


def test_nli_gate_rejects_false_accept_and_false_block_without_overriding_deterministic_categories():
    assert judgement("entailment", "contradiction", "supported") == ("SUPPORTED", True)
    assert judgement("entailment", "entailment", "supported") == ("BLOCKED", False)
    assert judgement("entailment", "contradiction", "unresolved_value_relation") == ("FALSE_ACCEPT", False)
    assert judgement("contradiction", "entailment", "contradicted") == ("CONTRADICTED", True)
    assert judgement("entailment", "contradiction", "unsupported") == ("SKIPPED_NOT_ELIGIBLE", True)
