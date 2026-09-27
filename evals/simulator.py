"""Ответы пользователя из скрытых предпочтений, без импортов агента."""

from .scenarios import Scenario


def answer(scenario: Scenario, slot: str | None, kind: str | None) -> tuple[str, dict]:
    theta = scenario.theta
    if slot == "kind":
        extension = scenario.turns[-1].spoken_after_answer
        value = scenario.final_spoken.kind or (extension.kind if extension else None)
        value = value or ("course" if scenario.final_spoken.genre in ("python", "машинное обучение") else "series")
        return {"series": "Сериал", "film": "Фильм", "course": "Курс"}[value], {"kind": value}
    if slot == "genre":
        genres = ("python", "машинное обучение") if kind == "course" else ("детектив", "комедия", "драма", "фантастика", "приключения")
        available = [genre for genre in genres if genre not in scenario.final_spoken.excluded_genres]
        value = max(available, key=lambda genre: theta.genre_weights[genre])
        return value, {"genre": value}
    if slot == "tone" and theta.tone_pref != "indifferent":
        return theta.tone_pref, {"tone": theta.tone_pref}
    if slot == "level" and theta.level_pref in ("начальный", "продвинутый"):
        return f"Уровень {theta.level_pref}", {"level": theta.level_pref}
    if slot == "practical" and theta.practical_pref is not None:
        return ("С практическими заданиями" if theta.practical_pref else "Только теория, без практики"), {"practical": theta.practical_pref}
    return "Без разницы", {}
