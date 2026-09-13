"""The protocol is OUR integration contract, not an alleged Recsflow API."""
from typing import Protocol
from difflib import SequenceMatcher
from .catalog import generate_catalog
from .models import Item, Query


class RecommendationProvider(Protocol):
    def retrieve(self, user_id: str, query: Query, limit: int = 100) -> list[str]: ...
    def lookup(self, item_ids: list[str]) -> list[Item]: ...
    def history(self, user_id: str) -> list[str]: ...
    def find_title(self, title: str) -> Item | None: ...


def matches(item: Item, query: Query) -> bool:
    if query.kind and item.kind != query.kind:
        return False
    if query.genre and item.genre != query.genre:
        return False
    if item.genre in query.excluded_genres:
        return False
    if query.tone and item.tone != query.tone:
        return False
    if query.max_seasons is not None and (item.seasons is None or item.seasons > query.max_seasons):
        return False
    if query.max_minutes is not None and item.minutes > query.max_minutes:
        return False
    if query.level and item.level != query.level:
        return False
    if query.practical is not None and item.practical != query.practical:
        return False
    return True


class DemoProvider:
    def __init__(self, items: list[Item] | None = None):
        self.items = {i.id: i for i in (items if items is not None else generate_catalog())}

    def retrieve(self, user_id: str, query: Query, limit: int = 100) -> list[str]:
        # A real adapter maps its platform's retrieval here; hard filters are checked again by the agent.
        candidates = [i for i in self.items.values() if matches(i, query)]
        return [i.id for i in sorted(candidates, key=lambda i: (-i.quality, i.id))[:limit]]

    def lookup(self, item_ids: list[str]) -> list[Item]:
        return [self.items[i] for i in item_ids if i in self.items]

    def history(self, user_id: str) -> list[str]:
        # Only the explicitly labelled demo profile has synthetic history.
        return ["demo-009", "demo-013"] if user_id == "demo" else []

    def find_title(self, title: str) -> Item | None:
        normalized = title.strip().casefold().replace("ё", "е")
        exact = next((i for i in self.items.values() if i.title.casefold().replace("ё", "е") == normalized), None)
        if exact:
            return exact
        # Conservative typo/inflection tolerance; only a unique near-exact match is accepted.
        ranked = sorted(((SequenceMatcher(None, normalized, i.title.casefold().replace("ё", "е")).ratio(), i) for i in self.items.values()), key=lambda pair: -pair[0])
        if ranked and ranked[0][0] >= .92 and (len(ranked) == 1 or ranked[0][0]-ranked[1][0] >= .04):
            return ranked[0][1]
        return None
