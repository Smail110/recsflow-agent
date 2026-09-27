"""Domain-neutral response planning over already validated evidence."""

from copy import copy
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GroundedOption(ResponseModel):
    id: str
    title: str
    claims: list[str] = Field(min_length=1, max_length=20)


class PlannedOption(ResponseModel):
    item_id: str
    evidence_indexes: list[int] = Field(min_length=1, max_length=3)


class GroundedResponsePlan(ResponseModel):
    items: list[PlannedOption] = Field(min_length=1, max_length=3)


class ResponseBackend(Protocol):
    def structured(self, schema: type[BaseModel], system: str, payload: dict) -> tuple[BaseModel, int]: ...


class EvidenceResponseGenerator:
    """Safe fallback: one generic renderer, with no domain-specific branches."""

    requires_llm = False

    def __init__(self, *, tone: Literal["neutral", "friendly"] = "neutral", length: Literal["normal", "short"] = "normal"):
        self.tone = tone
        self.length = length

    def with_style(self, *, tone: str, length: str):
        """Configure one request without mutating a shared generator or backend."""
        configured = copy(self)
        configured.tone = tone
        configured.length = length
        return configured

    @staticmethod
    def _fallback_plan(options: list[GroundedOption]) -> GroundedResponsePlan:
        return GroundedResponsePlan(
            items=[PlannedOption(item_id=option.id, evidence_indexes=list(range(min(2, len(option.claims))))) for option in options[:3]]
        )

    @staticmethod
    def _validated_plan(plan: GroundedResponsePlan, options: list[GroundedOption]) -> GroundedResponsePlan:
        by_id = {option.id: option for option in options}
        seen = set()
        valid = []
        for item in plan.items:
            option = by_id.get(item.item_id)
            indexes = list(dict.fromkeys(item.evidence_indexes))
            if option is None or item.item_id in seen or any(index < 0 or index >= len(option.claims) for index in indexes):
                continue
            seen.add(item.item_id)
            valid.append(PlannedOption(item_id=item.item_id, evidence_indexes=indexes))
        return GroundedResponsePlan(items=valid) if valid else EvidenceResponseGenerator._fallback_plan(options)

    def _render(self, plan: GroundedResponsePlan, options: list[GroundedOption]) -> str:
        by_id = {option.id: option for option in options}
        parts = []
        selected_items = plan.items[:2] if self.length == "short" else plan.items
        for index, selected in enumerate(selected_items):
            option = by_id[selected.item_id]
            indexes = selected.evidence_indexes[:1] if self.length == "short" else selected.evidence_indexes
            evidence = " ".join(option.claims[i] for i in indexes)
            lead = "Подходит" if index == 0 else "Также можно рассмотреть"
            if self.tone == "friendly":
                lead = "Предлагаю посмотреть" if index == 0 else "Ещё один вариант —"
            parts.append(f"{lead} «{option.title}». {evidence}".strip())
        return "\n".join(parts)

    def generate(
        self, *, original_request: str, intent: str, accepted_constraints: dict, options: list[GroundedOption], unresolved: list[dict]
    ) -> tuple[str, int]:
        del original_request, intent, accepted_constraints, unresolved
        plan = self._fallback_plan(options)
        return self._render(plan, options), 0


class LLMGroundedResponseGenerator(EvidenceResponseGenerator):
    """Let the LLM select useful evidence; render only validated claim atoms."""

    requires_llm = True

    def __init__(self, backend: ResponseBackend):
        super().__init__()
        self.backend = backend

    def generate(
        self, *, original_request: str, intent: str, accepted_constraints: dict, options: list[GroundedOption], unresolved: list[dict]
    ) -> tuple[str, int]:
        result, tokens = self.backend.structured(
            GroundedResponsePlan,
            (
                "Составь план ответа рекомендательной системы. Выбери до трёх реально переданных items и "
                "для каждого до трёх evidence_indexes, которые лучше всего объясняют соответствие исходной цели. Нельзя "
                "создавать новые item_id, индексы или продуктовые утверждения. Каждое утверждение финального ответа будет "
                "взято только из выбранного evidence. Не показывай внутренние schema field names. Верни только JSON."
            ),
            {
                "original_user_request": original_request,
                "interpreted_intent": intent,
                "accepted_constraints": accepted_constraints,
                "items": [option.model_dump() for option in options],
                "unresolved_constraints": unresolved,
            },
        )
        plan = self._validated_plan(GroundedResponsePlan.model_validate(result.model_dump()), options)
        return self._render(plan, options), tokens


def with_response_style(generator, *, tone: str, length: str):
    """Use optional styling while preserving the existing generate interface."""
    configure = getattr(generator, "with_style", None)
    return configure(tone=tone, length=length) if callable(configure) else generator
