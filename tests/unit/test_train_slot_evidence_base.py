"""Safe local checkpoint loading for the experimental ruBERT-base classifier."""

import pytest
from scripts.train_slot_evidence_base import IGNORED_BUFFER, encoder_state_keys


def test_only_encoder_tensors_are_selected():
    checkpoint = {"bert.embeddings.word_embeddings.weight", IGNORED_BUFFER, "cls.predictions.bias"}
    selected, ignored = encoder_state_keys(checkpoint, {"embeddings.word_embeddings.weight"})
    assert selected == {"embeddings.word_embeddings.weight"}
    assert ignored == {IGNORED_BUFFER, "cls.predictions.bias"}


def test_missing_encoder_tensor_fails_closed():
    with pytest.raises(ValueError, match="encoder tensor mismatch"):
        encoder_state_keys({"bert.embeddings.word_embeddings.weight"}, {"embeddings.word_embeddings.weight", "encoder.layer.0.weight"})


def test_unknown_checkpoint_tensor_fails_closed():
    with pytest.raises(ValueError, match="unrecognized checkpoint tensors"):
        encoder_state_keys({"bert.embeddings.word_embeddings.weight", "unexpected.weight"}, {"embeddings.word_embeddings.weight"})
