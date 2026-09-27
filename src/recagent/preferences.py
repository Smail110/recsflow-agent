"""Память предпочтений текущего диалога с происхождением каждого значения."""

from dataclasses import dataclass, field

from .models import Query


@dataclass
class PreferenceModel:
    entries: dict = field(default_factory=dict)

    def update(self, query: Query, previous: Query, message: str, mode: str):
        if previous.kind != query.kind and previous.kind is not None:
            self.entries.clear()
        for name, value in query.model_dump().items():
            if name == "intent":
                continue
            if value is None or value == []:
                self.entries.pop(name, None)
            elif value != getattr(previous, name) or name not in self.entries:
                self.entries[name] = {
                    "value": value,
                    "source_message": message,
                    "parser": mode,
                    "scope": "dialogue",
                    "status": "interpreted",
                }
        # Значения извлечены парсером, а не подтверждены отдельным действием.
        # Мы не придумываем численную уверенность и постоянные вкусы пользователя.
