"""Missing-field completion preserves clean proposals and all validation gates."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from recagent.domains.base import DomainSpec, FieldSpec
from recagent.domains.demo import domain_spec, request_adapter
from recagent.interpretation import LLMRequestInterpreter, StructuredRequest
from recagent.models import ChatRequest
from recagent.response_generation import EvidenceResponseGenerator
from recagent.workflow import WorkflowAgent


class Backend:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_usage = {}

    def structured(self, schema, system, payload):
        self.calls.append((schema, system, deepcopy(payload)))
        self.last_usage = {"input_tokens": 6, "output_tokens": 4}
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response), 10


def update(field, value, text, operation="set"):
    return {"field": field, "value": value, "source_text": text, "operation": operation}


def interpreter(backend):
    adapter = request_adapter()
    return LLMRequestInterpreter(backend, adapter.descriptor, domain_spec=domain_spec())


def test_completion_only_appends_and_preserves_intent_reset_and_source():
    backend = Backend({"updates": [update("kind", "series", "сериал")], "issues": []})
    attempt = StructuredRequest(intent="mood", reset_constraints=True, updates=[update("genre", "драма", "драма")])
    original = attempt.model_dump()
    result, tokens = interpreter(backend).complete_missing(
        "Новый сериал, драма", {}, feedback=[{"code": "coverage_gap", "field": "kind", "detail": "do not expose"}], attempt=attempt
    )
    assert attempt.model_dump() == original
    assert result.intent == "mood" and result.reset_constraints
    assert result.updates[0] == attempt.updates[0] and result.updates[1].value == "series"
    assert tokens == 10
    schema, _, payload = backend.calls[0]
    assert set(schema.model_fields) == {"updates", "issues"}
    assert [f["name"] for f in payload["domain"]["fields"]] == ["kind"]
    assert "do not expose" not in str(payload)


def test_completion_preserves_unknown_issue_without_guessing():
    backend = Backend(
        {
            "updates": [],
            "issues": [
                {
                    "kind": "unsupported_constraint",
                    "field": "kind",
                    "value": "радиопьеса",
                    "message": "Неизвестный формат",
                    "source_text": "радиопьеса",
                }
            ],
        }
    )
    result, _ = interpreter(backend).complete_missing(
        "Радиопьеса, драма",
        {},
        feedback=[{"code": "coverage_gap", "field": "kind"}],
        attempt=StructuredRequest(updates=[update("genre", "драма", "драма")]),
    )
    assert len(result.updates) == 1 and result.clarification_required
    assert result.issues[0].value == "радиопьеса"


@pytest.mark.parametrize(
    "result",
    [
        {"updates": [update("genre", "драма", "драма")], "issues": []},
        {"updates": [], "issues": [{"kind": "ambiguity", "field": "genre", "message": "bad"}]},
    ],
)
def test_out_of_target_completion_is_rejected(result):
    with pytest.raises((ValueError, ValidationError)):
        interpreter(Backend(result)).complete_missing(
            "сериал, драма",
            {},
            feedback=[{"code": "coverage_gap", "field": "kind"}],
            attempt=StructuredRequest(updates=[update("genre", "драма", "драма")]),
        )


def test_schema_is_from_customer_and_false_is_not_dropped():
    spec = DomainSpec(
        id="appliance",
        version="1",
        request_schema="Fixture",
        fields=(FieldSpec(name="portable", value_type="boolean"), FieldSpec(name="max_mass", value_type="integer")),
    )
    backend = Backend({"updates": [update("portable", False, "стационарный")], "issues": []})
    result, _ = LLMRequestInterpreter(backend, {}, domain_spec=spec).complete_missing(
        "стационарный до 7",
        {},
        feedback=[{"code": "coverage_gap", "field": "portable"}],
        attempt=StructuredRequest(updates=[update("max_mass", 7, "7")]),
    )
    assert result.updates[1].value is False
    assert [f["name"] for f in backend.calls[0][2]["domain"]["fields"]] == ["portable"]


def make_workflow(backend, **kwargs):
    return WorkflowAgent(
        mode="ollama", llm=backend, interpreter=interpreter(backend), response_generator=EvidenceResponseGenerator(), **kwargs
    )


def test_workflow_validates_completion_evidence_and_keeps_pending_atomic():
    backend = Backend(
        {"updates": [update("kind", "course", "курс")]},
        {"updates": [update("level", "начальный", "выдуманная цитата")], "issues": []},
    )
    agent = make_workflow(backend)
    response = agent.chat(ChatRequest(user_id="fixture", message="курс, начальный"))
    assert response.state == "clarify" and not response.recommendations
    assert not agent.sessions[response.session_id].constraint_state.constraints
    assert len(backend.calls) == 2


def test_adapter_failure_remains_blocked_after_format_recovery():
    backend = Backend({"updates": [update("genre", "комедия", "не комедия")]}, {"updates": []})
    agent = make_workflow(backend)
    response = agent.chat(ChatRequest(user_id="fixture", message="Мне нужен фильм, не комедия"))
    assert len(backend.calls) == 1
    assert response.state == "clarify"


def test_existing_call_budget_uses_verified_explicit_format_without_completion():
    backend = Backend({"updates": [update("genre", "драма", "драма")]})
    agent = make_workflow(backend, max_calls=1)
    response = agent.chat(ChatRequest(user_id="fixture", message="фильм, драма"))
    assert len(backend.calls) == 1 and response.state == "recommend"
    assert response.query.kind == "film" and response.query.genre == "драма"
    assert response.mode == "ollama" and response.llm_calls == 1


def test_verified_explicit_format_avoids_unnecessary_completion_and_tokens():
    backend = Backend({"updates": [update("genre", "драма", "драма")]}, TimeoutError("controlled"))
    agent = make_workflow(backend)
    response = agent.chat(ChatRequest(user_id="fixture", message="фильм, драма"))
    assert response.state == "recommend" and response.mode == "ollama"
    assert response.query.kind == "film" and response.query.genre == "драма"
    assert response.llm_calls == 1 and response.llm_tokens == 10


def test_merged_limit_is_enforced():
    backend = Backend({"updates": [update("kind", "film", "фильм"), update("kind", "series", "сериал")], "issues": []})
    attempt = StructuredRequest(updates=[update("genre", "драма", "драма") for _ in range(29)])
    with pytest.raises(ValidationError):
        interpreter(backend).complete_missing(
            "фильм или сериал драма", {}, feedback=[{"code": "coverage_gap", "field": "kind"}], attempt=attempt
        )
