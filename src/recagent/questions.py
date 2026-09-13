"""Выбор уточнения по неоднородности доступных кандидатов."""

import math
from collections import Counter
from dataclasses import dataclass

from .models import Item, Query


@dataclass(frozen=True)
class Question:
    slot: str
    message: str
    gain: float


MESSAGES = {
    "genre": "Какой жанр вам ближе? Для курса можно назвать тему: Python или машинное обучение.",
    "tone": "Какой тон предпочитаете: лёгкий, нейтральный или мрачный?",
    "level": "Какой уровень курса нужен: начальный или продвинутый?",
    "practical": "Нужны практические задания или достаточно теории?",
}
WEIGHTS = {"genre": 1.0, "tone": 0.7, "level": 0.8, "practical": 0.5}


def choose_question(query: Query, candidates: list[Item], *, policy: str, skipped: set[str], cost: float = 0.25) -> Question | None:
    """Энтропия — приближение пользы вопроса, а не доступ к скрытой полезности."""
    slots = ("genre", "level", "practical") if query.kind == "course" else ("genre", "tone")
    choices = []
    for slot in slots:
        if getattr(query, slot) is not None or slot in skipped:
            continue
        counts = Counter(getattr(item, slot) for item in candidates if getattr(item, slot) is not None)
        if len(counts) < 2:
            continue
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values()) / math.log2(len(counts))
        choices.append(Question(slot, MESSAGES[slot], round(entropy * WEIGHTS[slot], 6)))
    if not choices:
        return None
    if policy == "fixed":
        return choices[0]
    best = max(choices, key=lambda question: question.gain)
    return best if best.gain > cost else None
