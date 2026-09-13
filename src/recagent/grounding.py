from .models import Evidence, Item, Query, Recommendation


def validate_evidence(evidence: Evidence, item: Item, query: Query, history: list[Item], seed: Item | None = None) -> bool:
    if evidence.item_id != item.id or evidence.field not in Item.model_fields:
        return False
    value = getattr(item, evidence.field)
    if type(value) is not type(evidence.value) or value != evidence.value:
        return False
    if evidence.relation == "history":
        return any(h.id == evidence.source_item_id and h.genre == item.genre for h in history) and evidence.field == "genre"
    if evidence.relation == "seed":
        return bool(seed and seed.id == evidence.source_item_id and evidence.field == "genre" and seed.genre == item.genre)
    if evidence.relation == "request":
        if evidence.field == "seasons":
            return query.max_seasons is not None and value <= query.max_seasons
        if evidence.field == "minutes":
            return query.max_minutes is not None and value <= query.max_minutes
        return getattr(query, evidence.field, None) == value
    return True


def explain(item: Item, query: Query, history: list[Item], score: float, seed: Item | None = None) -> Recommendation:
    claims = []
    def add(field, sentence, relation="catalog", source=None):
        evidence = Evidence(item_id=item.id, field=field, value=getattr(item, field), relation=relation, source_item_id=source)
        if validate_evidence(evidence, item, query, history, seed):
            claims.append((sentence, evidence))
    add("genre", f"Жанр: {item.genre}.", "request" if query.genre else "catalog")
    add("tone", f"Тон: {item.tone}.", "request" if query.tone else "catalog")
    if item.kind == "series":
        add("seasons", f"Сезонов: {item.seasons}.", "request" if query.max_seasons else "catalog")
        add("episodes", f"Серий всего: {item.episodes}.")
    unit = "Серия" if item.kind == "series" else "Фильм" if item.kind == "film" else "Курс целиком"
    add("minutes", f"{unit}: {item.minutes} мин.", "request" if query.max_minutes else "catalog")
    if item.level:
        add("level", f"Уровень: {item.level}.", "request" if query.level else "catalog")
    if item.practical:
        add("practical", "Есть практические задания.", "request" if query.practical else "catalog")
    prior = next((h for h in history if h.genre == item.genre), None)
    if prior:
        add("genre", "Этот жанр есть в вашей истории.", "history", prior.id)
    if seed and seed.genre == item.genre:
        add("genre", f"Тот же жанр, что у «{seed.title}».", "seed", seed.id)
    return Recommendation(item=item, score=round(score, 4), explanation=" ".join(c[0] for c in claims), evidence=[c[1] for c in claims])

