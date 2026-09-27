from recagent.semantic import NLIResult, SemanticStatus
from recagent.semantic_policy import decide_nli


class StubNLI:
    model = "stub"

    def __init__(self, labels):
        self.labels = labels
        self.queries = []

    def predict(self, premise, hypothesis):
        self.queries.append((premise, hypothesis))
        label = self.labels[hypothesis]
        return NLIResult(label, 0.999, SemanticStatus("available", self.model))


def test_supported_update_is_accepted_with_raw_predictions():
    model = StubNLI({"target": "entailment", "opposite": "contradiction"})
    verdict = decide_nli(model, "utterance", "target", "opposite")
    assert verdict.decision == "ACCEPT"
    assert [item.prediction.score for item in verdict.predictions] == [0.999, 0.999]
    assert model.queries == [("utterance", "target"), ("utterance", "opposite")]


def test_contradicted_update_is_rejected():
    model = StubNLI({"target": "contradiction", "opposite": "entailment"})
    assert decide_nli(model, "utterance", "target", "opposite").decision == "REJECT"


def test_neutral_update_abstains():
    model = StubNLI({"target": "neutral", "opposite": "neutral"})
    assert decide_nli(model, "utterance", "target", "opposite").decision == "ABSTAIN"


def test_both_entailment_abstains():
    model = StubNLI({"target": "entailment", "opposite": "entailment"})
    assert decide_nli(model, "utterance", "target", "opposite").reason == "conflicting_entailment"


def test_competing_role_entailment_abstains():
    model = StubNLI({"target": "entailment", "opposite": "contradiction", "other_role": "entailment"})
    verdict = decide_nli(model, "utterance", "target", "opposite", competing_role_hypotheses=("other_role",))
    assert verdict.decision == "ABSTAIN"
    assert [item.role for item in verdict.predictions] == ["target", "opposite", "competing_role"]


def test_unavailable_model_abstains():
    class UnavailableNLI(StubNLI):
        def predict(self, premise, hypothesis):
            return NLIResult("unavailable", 0.0, SemanticStatus("unavailable", self.model))

    assert decide_nli(UnavailableNLI({}), "utterance", "target", "opposite").decision == "ABSTAIN"


def test_backend_exception_abstains_with_observable_error():
    class BrokenNLI(StubNLI):
        def predict(self, premise, hypothesis):
            raise RuntimeError("failed")

    verdict = decide_nli(BrokenNLI({}), "utterance", "target", "opposite")
    assert verdict.decision == "ABSTAIN"
    assert verdict.predictions[0].prediction.status.status == "error"
    assert verdict.predictions[0].prediction.status.reason == "RuntimeError"
