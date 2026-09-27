"""Планировщик экспериментальной политики: ответить или уточнить до двух слотов."""

from dataclasses import dataclass

from .models import Item, Query
from .questions import choose_question


@dataclass(frozen=True)
class ActionPlan:
    action: str
    reason: str
    slots: tuple[str, ...] = ()
    message: str = ""
    gain: float = 0.0


def plan_action(query: Query, candidates: list[Item], skipped: set[str], remaining: int, cost: float) -> ActionPlan:
    if not candidates:
        return ActionPlan("recommend", "Нет кандидатов: дополнительные предпочтения не исправят пустую выдачу.")
    if remaining <= 0 or query.intent == "navigation":
        return ActionPlan("recommend", "Бюджет вопросов исчерпан либо запрошен конкретный объект.")
    first = choose_question(query, candidates, policy="adaptive", skipped=skipped, cost=cost)
    if first is None:
        return ActionPlan("recommend", "Не найдено информативных неизвестных атрибутов.")
    second = choose_question(query, candidates, policy="adaptive", skipped=skipped | {first.slot}, cost=cost + 0.15)
    slots = (first.slot, second.slot) if second else (first.slot,)
    prompts = {
        "genre": "какой жанр или тема интересует",
        "tone": "какого настроения хочется",
        "level": "с какого уровня начинаете",
        "practical": "нужны ли практические задания",
    }
    message = "Подскажите, " + " и ".join(prompts[slot] for slot in slots) + ". Можно ответить одной фразой или только на часть вопроса."
    return ActionPlan(
        "clarify",
        "Выбраны неизвестные атрибуты с разнообразием среди кандидатов; максимум два, чтобы ограничить усилие ответа.",
        slots,
        message,
        first.gain + (second.gain - 0.15 if second else 0),
    )
