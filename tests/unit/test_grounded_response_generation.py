from recagent.agent import Agent
from recagent.models import ChatRequest
from recagent.response_generation import GroundedOption, LLMGroundedResponseGenerator


class PlannerBackend:
    def structured(self, schema, system, payload):
        assert "продуктовые утверждения" in system
        assert payload["original_user_request"]
        return schema(
            items=[
                {"item_id": "missing", "evidence_indexes": [0]},
                {"item_id": "item-1", "evidence_indexes": [1, 99]},
            ]
        ), 17


def test_invalid_llm_plan_falls_back_to_catalog_backed_claims_only():
    options = [GroundedOption(id="item-1", title="Точный вариант", claims=["Цена: 90 000 ₽.", "Вес: 1,2 кг."])]
    message, tokens = LLMGroundedResponseGenerator(PlannerBackend()).generate(
        original_request="Нужен лёгкий вариант до 100 тысяч",
        intent="discovery",
        accepted_constraints={"max_price": 100_000},
        options=options,
        unresolved=[],
    )
    assert tokens == 17
    assert message == "Подходит «Точный вариант». Цена: 90 000 ₽. Вес: 1,2 кг."
    assert "missing" not in message and "99" not in message


class SelectiveBackend:
    def structured(self, schema, system, payload):
        return schema(items=[{"item_id": "item-2", "evidence_indexes": [0]}]), 9


def test_llm_selects_relevant_evidence_but_cannot_write_free_form_claims():
    options = [
        GroundedOption(id="item-1", title="Первый", claims=["Рейтинг: 4,6."]),
        GroundedOption(id="item-2", title="Второй", claims=["Цена: 75 000 ₽.", "Вес: 1,5 кг."]),
    ]
    message, _ = LLMGroundedResponseGenerator(SelectiveBackend()).generate(
        original_request="До 80 тысяч", intent="discovery", accepted_constraints={"max_price": 80_000}, options=options, unresolved=[]
    )
    assert message == "Подходит «Второй». Цена: 75 000 ₽."


class MalformedBackend:
    def __init__(self):
        self.last_usage = {"input_tokens": 7, "output_tokens": 5, "inference_seconds": 0.2}

    def structured(self, schema, system, payload):
        raise ValueError("malformed response")


def test_malformed_response_plan_is_grounded_fallback_and_counts_usage():
    response = Agent(
        mode="rules",
        question_policy="none",
        response_generator=LLMGroundedResponseGenerator(MalformedBackend()),
    ).chat(ChatRequest(user_id="usage", message="Найди «Декоратор в проде»"))

    assert response.state == "recommend"
    assert response.message.startswith("Подходит «Декоратор в проде».")
    assert response.llm_calls == 1
    assert response.llm_tokens == 12
    assert response.llm_usage == {"input_tokens": 7.0, "output_tokens": 5.0, "inference_seconds": 0.2}
    assert any("grounded fallback" in warning for warning in response.warnings)
