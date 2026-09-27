"""Модели запросов, объектов каталога и ответов демонстрационного API."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import Scalar


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(StrictModel):
    intent: Literal["discovery", "similar", "mood", "navigation"] = "discovery"
    kind: Literal["series", "film", "course"] | None = None
    genre: Literal["детектив", "комедия", "драма", "фантастика", "приключения", "машинное обучение", "python"] | None = None
    tone: Literal["лёгкий", "мрачный", "нейтральный"] | None = None
    max_seasons: int | None = Field(default=None, ge=1, le=100)
    max_minutes: int | None = Field(default=None, ge=1, le=10000)
    level: Literal["начальный", "средний", "продвинутый"] | None = None
    practical: bool | None = None
    seed_title: str | None = Field(default=None, max_length=200)
    excluded_genres: list[str] = Field(default_factory=list, max_length=10)


class Item(StrictModel):
    id: str
    title: str
    kind: Literal["series", "film", "course"]
    genre: str | None = None
    tone: str | None = None
    seasons: int | None = None
    episodes: int | None = None
    minutes: int | None = None
    level: str | None = None
    practical: bool | None = None
    year: int | None = Field(default=None, ge=1888, le=2100)
    quality: float = Field(ge=0, le=1)
    # Число взаимодействий с объектом. Нужно b0-популярностному бейзлайну и фичам
    # реранкера. Платформа может его не отдавать — тогда None, и это «нет данных».
    popularity: int | None = Field(default=None, ge=0)
    description: str
    synthetic: bool = True


class Evidence(StrictModel):
    item_id: str
    field: str
    value: str | int | bool
    relation: Literal["catalog", "request", "history", "seed"] = "catalog"
    source_item_id: str | None = None


class Recommendation(StrictModel):
    item: Item
    score: float
    explanation: str
    evidence: list[Evidence]
    claim_texts: list[str] = Field(default_factory=list)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=80)
    # Обязательное поле: без него нельзя выбрать историю пользователя.
    # В production его должен задавать слой авторизации, а не клиент.
    user_id: str = Field(min_length=1, max_length=80)
    message_id: UUID | None = Field(default=None, exclude=True)
    explanation_tone: Literal["neutral", "friendly"] = "neutral"
    explanation_length: Literal["short", "normal"] = "normal"

    @field_validator("message")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Сообщение не должно быть пустым")
        return value.strip()


class PendingChangePreview(StrictModel):
    field: str
    label: str
    operation: Literal["add", "replace", "remove", "clear"]
    operator: Literal["eq", "neq", "lt", "lte", "gt", "gte"]
    value: Scalar | None = None
    source_text: list[str]


class PendingPreview(StrictModel):
    """Display-only verified proposals; never active retrieval constraints."""

    status: Literal["staged"] = "staged"
    question_id: str
    base_version: int
    changes: list[PendingChangePreview]


class ChatResponse(StrictModel):
    request_id: str | None = None
    provider: str | None = None
    session_id: str
    state: Literal["clarify", "recommend", "no_results"]
    message: str
    query: Query
    recommendations: list[Recommendation] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    mode: str
    latency_ms: float
    llm_calls: int
    llm_calls_total: int
    llm_tokens: int
    clarification_count: int
    clarification_slot: str | None = None
    question_gain: float | None = None
    degradation: str = "FULL"
    platform_ids: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_tokens_total: int = 0
    llm_usage: dict[str, float] = Field(default_factory=dict)
    clarification_slots: list[str] = Field(default_factory=list)
    action_reason: str | None = None
    preferences: dict = Field(default_factory=dict)
    telemetry: dict[str, object] = Field(default_factory=dict)
    pending_preview: PendingPreview | None = None


class FeedbackRequest(StrictModel):
    session_id: str
    item_id: str
    reaction: Literal["like", "dislike", "seen"]
    user_id: str = Field(min_length=1, max_length=80)
