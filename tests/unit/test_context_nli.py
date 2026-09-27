"""Contract tests for the optional full-context verifier; no weights loaded."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from recagent.context_nli import ContextNLI, EvidenceInput, cited_claim_for_update, claim_for_update, tag_citation
from recagent.interpretation import ConstraintUpdate, StructuredRequest
from recagent.request_mapping import SchemaRequestAdapter
from recagent.resolution import FieldSpec, LanguageProfile
from recagent.semantic import NLIResult, SemanticStatus


class StubModel:
    model = "stub"

    def __init__(self, label: str = "support", status: str = "available"):
        self.label = label
        self.status = status
        self.calls: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        self.calls.append((premise, hypothesis))
        return NLIResult(self.label, 0.98, SemanticStatus(self.status, self.model))


SPEC = FieldSpec(
    name="genre",
    enum={"комедия": "комедия", "драма": "драма"},
    aliases={"смешное": "комедия", "грустное": "драма", "драмы": "драма"},
)


def evidence(
    *,
    message: str = "Хочу что-то смешное, но драмы не надо",
    citation: str = "смешное",
    value: object = "комедия",
    operation: str = "set",
    spec: FieldSpec = SPEC,
    allowed: tuple[str, ...] = ("set", "exclude", "include"),
    is_preference: bool = False,
) -> EvidenceInput:
    return EvidenceInput(
        message=message,
        update=ConstraintUpdate(field=spec.name, value=value, operation=operation, source_text=citation),
        spec=spec,
        field_label="Жанр",
        allowed_operations=allowed,
        is_preference=is_preference,
    )


def test_verifier_passes_full_message_and_role_value_operation_claim() -> None:
    model = StubModel()
    verifier = ContextNLI(model)
    item = evidence()
    verdict = verifier.verify(item)
    assert verdict.resolution.status == "canonical"
    assert verdict.resolution.value == "комедия"
    assert model.calls == [(item.message, claim_for_update("Жанр", "смешное", "set"))]
    assert "драмы не надо" in model.calls[0][0]
    assert verdict.model_score == 0.98


@pytest.mark.parametrize(
    ("label", "status", "expected"),
    [
        ("support", "canonical", "support"),
        ("contradiction", "ambiguous", "contradiction"),
        ("unknown", "unsupported", "unknown"),
        ("unexpected", "unsupported", "abstain"),
    ],
)
def test_model_labels_have_distinct_safe_outcomes(label: str, status: str, expected: str) -> None:
    verifier = ContextNLI(StubModel(label))
    result = verifier.verify(evidence()).resolution
    assert result.status == status
    assert getattr(verifier.stats(), expected) == 1


def test_unavailable_or_exception_abstains_without_caching() -> None:
    unavailable = StubModel(status="unavailable")
    verifier = ContextNLI(unavailable)
    assert verifier.verify(evidence()).resolution.status == "unsupported"
    assert verifier.verify(evidence()).resolution.status == "unsupported"
    assert verifier.stats().model_calls == 2
    assert verifier.stats().cache_hits == 0

    class BrokenModel(StubModel):
        def predict(self, premise: str, hypothesis: str) -> NLIResult:
            raise RuntimeError("offline")

    broken = ContextNLI(BrokenModel())
    result = broken.verify(evidence()).resolution
    assert result.status == "unsupported"
    assert result.reason == "context_nli_model_error:RuntimeError"

    class MalformedModel(StubModel):
        def predict(self, premise: str, hypothesis: str) -> None:
            return None

    malformed = ContextNLI(MalformedModel())
    assert malformed.verify(evidence()).resolution.reason == "context_nli_model_unavailable_or_invalid"


def test_uncited_value_and_untrained_operation_never_call_model() -> None:
    model = StubModel()
    verifier = ContextNLI(model)
    assert verifier.verify(evidence(citation="смешное выдуманное")).resolution.reason == "context_nli_uncited_source"
    assert verifier.verify(evidence(value="необъявленное")).resolution.reason == "context_nli_value_not_declared"
    assert verifier.verify(evidence(operation="clear", value=None)).resolution.reason == "context_nli_unsupported_operation"
    assert verifier.verify(evidence(operation="exclude", allowed=("set",))).resolution.reason == "context_nli_unsupported_operation"
    assert model.calls == []


def test_wrong_citation_cannot_borrow_full_message_support() -> None:
    model = StubModel("support")
    verifier = ContextNLI(model)
    item = evidence(message="Хочу комедию, без драмы", citation="драмы", value="комедия")
    verdict = verifier.verify(item)
    assert verdict.resolution.status == "unsupported"
    assert verdict.resolution.reason == "context_nli_citation_value_mismatch"
    assert model.calls == []

    unrelated = evidence(message="Хочу комедию прямо сейчас", citation="сейчас", value="комедия")
    assert verifier.verify(unrelated).resolution.status == "unsupported"
    assert model.calls == []


def test_v4_citation_representation_marks_exact_offset_without_changing_v3_claim() -> None:
    message = "Комедию исключить, комедию снова разрешаю"
    second_start = message.rfind("комедию")
    marked = tag_citation(message, "комедию", source_start=second_start, source_end=second_start + len("комедию"))
    assert marked == "Комедию исключить, <evidence>комедию</evidence> снова разрешаю"
    assert cited_claim_for_update("Жанр", "комедия", "include") == (
        "Выделенная цитата подтверждает: Пользователь снова разрешил для подбора значение «комедия» параметра «Жанр»."
    )
    assert claim_for_update("Жанр", "комедия", "include") == (
        "Пользователь снова разрешил для подбора значение «комедия» параметра «Жанр»."
    )
    with pytest.raises(ValueError, match="exactly one"):
        tag_citation(message, "комедию")
    with pytest.raises(ValueError, match="offsets do not match"):
        tag_citation(message, "комедию", source_start=0, source_end=8)
    with pytest.raises(ValueError, match="unmarked"):
        tag_citation("<evidence>комедия</evidence>", "комедия")
    with pytest.raises(ValueError, match="unmarked"):
        tag_citation("<EVIDENCE>комедия</EVIDENCE>", "комедия")


def test_tagged_mode_marks_full_premise_and_abstains_on_unknown_schema_synonym_by_default() -> None:
    model = StubModel("support")
    verifier = ContextNLI(model, citation_mode="tagged")
    item = evidence(message="Хочу что-то смешное прямо сейчас", citation="смешное")
    verdict = verifier.verify(item)
    assert verdict.resolution.status == "canonical"
    assert model.calls == [(
        "Хочу что-то <evidence>смешное</evidence> прямо сейчас",
        cited_claim_for_update("Жанр", "смешное", "set"),
    )]

    # V4 has no training coverage for undeclared source synonyms. Only an
    # explicit research switch may send one to the classifier.
    unknown = evidence(message="Хочу уморительное кино", citation="уморительное")
    assert verifier.verify(unknown).resolution.reason == "context_nli_citation_not_declared"
    assert len(model.calls) == 1
    research = ContextNLI(StubModel(), citation_mode="tagged", allow_undeclared_citation_for_research=True)
    assert research.verify(unknown).resolution.status == "canonical"
    assert ContextNLI(StubModel(), citation_mode="unmarked").verify(unknown).resolution.status == "unsupported"


def test_tagged_mode_fails_closed_on_wrong_declared_value_and_repeat_without_offset() -> None:
    model = StubModel("support")
    verifier = ContextNLI(model, citation_mode="tagged")
    wrong = evidence(message="Хочу комедию, без драмы", citation="драмы", value="комедия")
    assert verifier.verify(wrong).resolution.reason == "context_nli_citation_value_mismatch"
    assert model.calls == []

    repeated = evidence(message="Комедия нравилась, комедия теперь не нужна", citation="комедия")
    assert verifier.verify(repeated).resolution.reason == "context_nli_ambiguous_or_invalid_citation"
    assert model.calls == []

    second = repeated.message.rfind("комедия")
    with_offset = EvidenceInput(
        repeated.message, repeated.update, repeated.spec, repeated.field_label,
        allowed_operations=repeated.allowed_operations,
        source_start=second, source_end=second + len("комедия"),
    )
    assert verifier.verify(with_offset).resolution.status == "canonical"
    assert model.calls[0][0] == "Комедия нравилась, <evidence>комедия</evidence> теперь не нужна"

    distractor_model = StubModel("unknown")
    distractor_verifier = ContextNLI(
        distractor_model, citation_mode="tagged", allow_undeclared_citation_for_research=True
    )
    distractor = evidence(message="Хочу комедию прямо сейчас", citation="сейчас")
    assert distractor_verifier.verify(distractor).resolution.status == "unsupported"
    assert distractor_model.calls[0][0] == "Хочу комедию прямо <evidence>сейчас</evidence>"


def test_tagged_mode_unavailable_abstains_and_caches_only_available_result() -> None:
    unavailable = StubModel(status="unavailable")
    verifier = ContextNLI(unavailable, citation_mode="tagged")
    assert verifier.verify(evidence()).resolution.status == "unsupported"
    assert verifier.verify(evidence()).resolution.status == "unsupported"
    assert verifier.stats().model_calls == 2 and verifier.stats().cache_hits == 0

    model = StubModel("unknown")
    verifier = ContextNLI(model, citation_mode="tagged")
    assert verifier.verify(evidence()).resolution.status == "unsupported"
    assert verifier.verify(evidence()).cached
    assert verifier.stats().model_calls == 1 and verifier.stats().cache_hits == 1

def test_operation_and_preference_change_claim_not_premise() -> None:
    model = StubModel()
    verifier = ContextNLI(model, cache_size=0)
    set_claim = verifier.verify(evidence(is_preference=True)).hypothesis
    exclude_claim = verifier.verify(evidence(operation="exclude", is_preference=True)).hypothesis
    include_claim = verifier.verify(evidence(operation="include", is_preference=True)).hypothesis
    assert set_claim == "Пользователь предпочёл бы для подбираемого варианта значение «смешное» параметра «Жанр»."
    assert exclude_claim == "Пользователь исключил для подбора значение «смешное» параметра «Жанр»."
    assert include_claim == "Пользователь снова разрешил для подбора значение «смешное» параметра «Жанр»."
    assert [premise for premise, _ in model.calls] == [evidence().message] * 3


def test_repeated_normalization_uses_bounded_digest_cache() -> None:
    model = StubModel()
    verifier = ContextNLI(model, cache_size=1)
    first = verifier.verify(evidence())
    second = verifier.verify(evidence())
    assert not first.cached and second.cached
    assert verifier.stats().attempts == 2
    assert verifier.stats().model_calls == 1
    assert verifier.stats().cache_hits == 1
    assert len(verifier._cache) == 1
    assert evidence().message not in next(iter(verifier._cache))

    verifier.verify(evidence(message="Дайте смешное", citation="смешное"))
    assert verifier.stats().model_calls == 2
    assert len(verifier._cache) == 1


def test_scalar_literal_remains_structural_and_exclusion_abstains() -> None:
    model = StubModel()
    verifier = ContextNLI(model)
    spec = FieldSpec(name="max_minutes")
    assert verifier.verify(evidence(message="до 55 минут", citation="55", value=55, spec=spec)).resolution.value == 55
    assert verifier.verify(evidence(message="до 55 минут", citation="55", value=35, spec=spec)).resolution.status == "unsupported"
    assert verifier.verify(evidence(message="до 55 минут", citation="55", value=55, spec=spec, operation="exclude")).resolution.status == "unsupported"
    assert model.calls == []
    assert verifier.stats().structural == 2


def test_wrong_field_cannot_be_laundered_by_model() -> None:
    model = StubModel()
    verifier = ContextNLI(model)
    item = evidence()
    item = EvidenceInput(item.message, item.update.model_copy(update={"field": "tone"}), item.spec, item.field_label)
    assert verifier.verify(item).resolution.reason == "context_nli_invalid_field"
    assert model.calls == []


class Query(BaseModel):
    genre: Literal["комедия", "драма"] | None = None
    max_minutes: int | None = None
    excluded_genres: list[str] = []


def adapter(model: StubModel | None = None, *, preference_fields: set[str] | None = None) -> SchemaRequestAdapter:
    verifier = ContextNLI(model) if model is not None else None
    return SchemaRequestAdapter(
        Query,
        aliases={"genre": {"смешное": "комедия"}},
        exclusions={"genre": "excluded_genres"},
        field_labels={"genre": "Жанр"},
        no_preference_markers={"genre": ("любой",)},
        language_profile=LanguageProfile(negation_tokens=("не",)),
        context_verifier=verifier,
        preference_fields=preference_fields,
    )


def test_optional_adapter_routes_enum_through_full_context_without_lexical_operation_gate() -> None:
    model = StubModel("support")
    experimental = adapter(model)
    request = StructuredRequest(updates=[ConstraintUpdate(
        field="genre", operation="exclude", value="комедия", source_text="комедия",
    )])
    normalized = experimental.normalize(request, "Избегаю комедия")
    assert not normalized.issues
    assert normalized.updates[0].operation == "exclude"
    assert model.calls[0][0] == "Избегаю комедия"
    assert "исключил" in model.calls[0][1]

    baseline = adapter().normalize(request, "Избегаю комедия")
    assert not baseline.updates and baseline.issues


def test_preference_claim_requires_explicit_schema_role_configuration() -> None:
    ordinary_model, preference_model = StubModel(), StubModel()
    request = StructuredRequest(updates=[ConstraintUpdate(field="genre", value="комедия", source_text="смешное")])
    adapter(ordinary_model).normalize(request, "Хочу смешное")
    adapter(preference_model, preference_fields={"genre"}).normalize(request, "Хочу смешное")
    assert "выбрал" in ordinary_model.calls[0][1]
    assert "предпочёл бы" in preference_model.calls[0][1]


def test_adapter_abstain_does_not_commit_and_scalar_clear_use_old_path() -> None:
    model = StubModel(status="unavailable")
    experimental = adapter(model)
    disputed = StructuredRequest(updates=[ConstraintUpdate(field="genre", value="комедия", source_text="смешное")])
    updated, issues = experimental.apply(disputed, Query(), "Хочу смешное")
    assert updated.genre is None and issues
    assert model.calls

    scalar = StructuredRequest(updates=[ConstraintUpdate(field="max_minutes", value=55, source_text="55")])
    assert experimental.normalize(scalar, "до 55 минут").updates[0].value == 55
    assert len(model.calls) == 1

    clear = StructuredRequest(updates=[ConstraintUpdate(field="genre", operation="clear", value=None, source_text="жанр любой")])
    assert experimental.normalize(clear, "жанр любой").updates[0].operation == "clear"
    assert len(model.calls) == 1


def test_adapter_rejects_wrong_citation_despite_full_turn_support() -> None:
    model = StubModel("support")
    experimental = adapter(model)
    request = StructuredRequest(updates=[ConstraintUpdate(field="genre", value="комедия", source_text="драмы")])
    normalized = experimental.normalize(request, "Хочу комедию, без драмы")
    assert not normalized.updates and normalized.issues
    assert model.calls == []

def test_workflow_extraction_skips_lexical_rewrite_only_when_context_verifier_enabled(monkeypatch) -> None:
    from recagent.domains.demo import request_adapter
    from recagent.models import ChatRequest
    from recagent.workflow import WorkflowAgent

    class Interpreter:
        def interpret(self, message, previous, **context):
            return StructuredRequest(updates=[ConstraintUpdate(field="genre", value="комедия", source_text="комедия")]), 1

    model = StubModel()
    configured = request_adapter()
    configured.context_verifier = ContextNLI(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("lexical operation rewrite ran in context NLI mode")

    monkeypatch.setattr(configured, "reconcile_operation_evidence", forbidden)
    monkeypatch.setattr(configured, "detect_polarity_conflicts", forbidden)
    agent = WorkflowAgent(mode="ollama", interpreter=Interpreter(), request_adapter=configured)
    result = agent.chat(ChatRequest(user_id="context-nli-test", message="Хочу комедия"))
    assert result.mode == "ollama"
    assert model.calls and model.calls[0][0] == "Хочу комедия"


def test_workflow_repair_also_skips_lexical_rewrite_in_context_nli_mode(monkeypatch) -> None:
    from recagent.domains.demo import request_adapter
    from recagent.models import ChatRequest
    from recagent.workflow import WorkflowAgent

    class RepairingInterpreter:
        repaired = False

        def interpret(self, message, previous, **context):
            return StructuredRequest(), 1

        def repair(self, message, previous, **context):
            self.repaired = True
            return StructuredRequest(updates=[ConstraintUpdate(field="genre", value="комедия", source_text="комедия")]), 1

    interpreter = RepairingInterpreter()
    model = StubModel()
    configured = request_adapter()
    configured.context_verifier = ContextNLI(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("lexical operation rewrite ran during NLI repair")

    monkeypatch.setattr(configured, "reconcile_operation_evidence", forbidden)
    monkeypatch.setattr(configured, "detect_polarity_conflicts", forbidden)
    agent = WorkflowAgent(mode="ollama", interpreter=interpreter, request_adapter=configured)
    result = agent.chat(ChatRequest(user_id="context-nli-repair-test", message="Хочу комедия"))
    assert interpreter.repaired
    assert result.mode == "ollama"
    assert model.calls and model.calls[0][0] == "Хочу комедия"
