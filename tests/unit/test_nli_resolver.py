"""Stub-only tests for the optional enum resolver; no model runtime is loaded."""

from typing import Literal

from pydantic import BaseModel

from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.nli_resolver import NLIHypotheses, NLIResolver
from recagent.request_mapping import SchemaRequestAdapter
from recagent.resolution import FieldSpec
from recagent.semantic import NLIResult, SemanticStatus

SPEC = FieldSpec(
    name="level",
    enum={"beginner": "beginner", "advanced": "advanced"},
    aliases={"starter": "beginner"},
)


def hypotheses(spec: FieldSpec, canonical: object) -> NLIHypotheses:
    return NLIHypotheses(f"{spec.name}={canonical}", f"{spec.name}!={canonical}")


class StubNLI:
    model = "stub"

    def __init__(self, labels: dict[str, str], *, status: str = "available"):
        self.labels = labels
        self.status = status
        self.calls: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        self.calls.append((premise, hypothesis))
        return NLIResult(self.labels.get(hypothesis, "neutral"), 0.999, SemanticStatus(self.status, self.model))


def resolver(labels: dict[str, str], *, status: str = "available") -> tuple[NLIResolver, StubNLI]:
    adapter = StubNLI(labels, status=status)
    return NLIResolver(adapter, hypotheses), adapter


def test_exact_declared_surfaces_share_the_baseline_contract_without_nli_calls():
    semantic, model = resolver({})
    assert semantic.resolve(SPEC, "beginner", "starter").value == "beginner"
    assert semantic.resolve(SPEC, None, "starter").value == "beginner"
    mismatch = semantic.resolve(SPEC, "advanced", "starter")
    assert (mismatch.status, mismatch.reason) == ("ambiguous", "value_evidence_mismatch")
    assert model.calls == []


def test_unknown_citation_requires_entailment_opposite_contradiction_and_no_competitor():
    semantic, model = resolver({
        "level=beginner": "entailment",
        "level!=beginner": "contradiction",
        "level=advanced": "neutral",
    })
    result = semantic.resolve(SPEC, "starter", "novice")
    assert (result.status, result.value, result.reason) == ("canonical", "beginner", "nli_entailed")
    assert model.calls == [
        ("novice", "level=beginner"),
        ("novice", "level!=beginner"),
        ("novice", "level=advanced"),
    ]


def test_neutral_and_unavailable_never_accept_despite_high_scores():
    semantic, model = resolver({"level=beginner": "neutral"})
    result = semantic.resolve(SPEC, "beginner", "novice")
    assert (result.status, result.reason) == ("unsupported", "nli_insufficient_evidence")
    assert len(model.calls) == 3

    semantic, model = resolver({"level=beginner": "entailment"}, status="unavailable")
    result = semantic.resolve(SPEC, "beginner", "novice")
    assert (result.status, result.reason) == ("unsupported", "nli_model_unavailable_or_invalid")
    assert len(model.calls) == 1


def test_conflicting_entailments_and_contradictions_block_wrong_proposals():
    semantic, model = resolver({
        "level=beginner": "entailment",
        "level!=beginner": "contradiction",
        "level=advanced": "entailment",
    })
    result = semantic.resolve(SPEC, "beginner", "mixed level")
    assert (result.status, result.reason) == ("ambiguous", "nli_conflicting_entailment")
    assert len(model.calls) == 3

    semantic, _ = resolver({"level=beginner": "contradiction", "level!=beginner": "entailment"})
    assert semantic.resolve(SPEC, "beginner", "expert").status == "ambiguous"


def test_no_proposed_declared_value_cannot_be_inferred_from_unknown_text():
    semantic, model = resolver({})
    assert semantic.resolve(SPEC, None, "novice").status == "unsupported"
    assert semantic.resolve(SPEC, "intermediate", "middle").status == "unsupported"
    assert model.calls == []


def test_backend_error_abstains_without_accepting():
    class FailingNLI(StubNLI):
        def predict(self, premise: str, hypothesis: str) -> NLIResult:
            self.calls.append((premise, hypothesis))
            raise RuntimeError("backend failed")

    model = FailingNLI({})
    result = NLIResolver(model, hypotheses).resolve(SPEC, "beginner", "novice")
    assert (result.status, result.reason) == ("unsupported", "nli_model_unavailable_or_invalid")
    assert len(model.calls) == 1


class Query(BaseModel):
    level: Literal["beginner", "advanced"] | None = None
    max_minutes: int | None = None


def test_schema_adapter_keeps_scalar_and_operation_checks_outside_nli():
    semantic, model = resolver({
        "level=beginner": "entailment",
        "level!=beginner": "contradiction",
        "level=advanced": "neutral",
    })
    adapter = SchemaRequestAdapter(Query, aliases={"level": {"starter": "beginner"}}, resolver=semantic)
    request = StructuredRequest(updates=[
        ConstraintUpdate(field="max_minutes", value=7, source_text="8"),
        ConstraintUpdate(field="level", operation="exclude", value="beginner", source_text="novice"),
    ])
    query, issues = adapter.apply(request, Query(), "8 novice")
    assert query == Query()
    assert {issue.field for issue in issues} == {"max_minutes", "level"}
    assert model.calls == []


def test_schema_adapter_uses_nli_only_for_nonexact_enum_citation():
    semantic, model = resolver({
        "level=beginner": "entailment",
        "level!=beginner": "contradiction",
        "level=advanced": "neutral",
    })
    adapter = SchemaRequestAdapter(Query, resolver=semantic)
    request = StructuredRequest(updates=[ConstraintUpdate(field="level", value="beginner", source_text="novice")])
    query, issues = adapter.apply(request, Query(), "novice")
    assert query.level == "beginner" and not issues
    assert len(model.calls) == 3
