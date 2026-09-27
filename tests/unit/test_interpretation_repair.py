"""Repair diagnostics must remain separate from user/session evidence."""

from copy import deepcopy

from recagent.interpretation import LLMRequestInterpreter, StructuredRequest


class CaptureBackend:
    def structured(self, schema, system, payload):
        self.system, self.payload = system, payload
        return schema(), 1


def test_repair_keeps_pending_context_and_does_not_turn_diagnostics_into_user_issues():
    backend = CaptureBackend()
    interpreter = LLMRequestInterpreter(backend, {})
    pending = {"target_fields": ["max_price"], "staged": [{"field": "category", "value": "laptop"}]}
    original = deepcopy(pending)
    attempt = StructuredRequest(updates=[{"field": "category", "value": "laptop", "source_text": "ноутбук"}])
    interpreter.repair(
        "ноутбук до 40000 рублей",
        {},
        feedback=[{"code": "coverage_gap", "field": "max_price", "detail": "untrusted diagnostic text"}],
        attempt=attempt,
        pending_context=pending,
    )
    assert pending == original == backend.payload["pending_context"]
    assert backend.payload["extraction_review"]["findings"] == [{"code": "coverage_gap", "field": "max_price"}]
    assert backend.payload["extraction_review"]["rejected_attempt"] == attempt.model_dump()
    assert "untrusted diagnostic text" not in str(backend.payload)
    assert "extraction_feedback" not in backend.payload["pending_context"]
    assert "не превращай" in backend.system.casefold()


def test_normal_interpretation_has_no_repair_context_or_instructions():
    backend = CaptureBackend()
    interpreter = LLMRequestInterpreter(backend, {})
    interpreter.interpret("Нужен ноутбук", {})
    assert "extraction_review" not in backend.payload
    assert "повторная проверка extraction" not in backend.system.casefold()
