"""Независимая проверка текста шаблонных объяснений по каталогу и истории."""

import re


def audit_text(text, item, history=(), catalog=()):
    text = text.removeprefix("Вот что о нём известно: ")
    claims = [part.strip() for part in re.findall(r"[^.!?]+[.!?]?", text) if part.strip()]
    valid = []
    fields = {"Жанр": "genre", "Тон": "tone", "Сезонов": "seasons", "Серий всего": "episodes", "Уровень": "level"}
    for claim in claims:
        accepted = False
        for label, field in fields.items():
            if claim == f"{label}: {getattr(item, field)}." and getattr(item, field) is not None:
                accepted = True
        unit = {"series": "Серия", "film": "Фильм", "course": "Курс целиком"}[item.kind]
        if item.minutes is not None and claim == f"{unit}: {item.minutes} мин.":
            accepted = True
        if claim == "Есть практические задания." and item.practical is True:
            accepted = True
        if claim == "Этот жанр есть в вашей истории." and item.genre is not None:
            accepted = any(previous.genre == item.genre for previous in history)
        seed = re.fullmatch(r"Тот же жанр, что у «(.+)»\.", claim)
        if seed and item.genre is not None:
            accepted = any(other.title == seed[1] and other.genre == item.genre for other in catalog)
        valid.append(accepted)
    return {"claims": len(claims), "invalid": [claim for claim, accepted in zip(claims, valid, strict=True) if not accepted]}
