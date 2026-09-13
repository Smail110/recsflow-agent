"""HTTP-адаптер: таймауты, ограниченные повторы и временное отключение при сбоях."""

import asyncio
import threading
import time
import uuid
from contextlib import suppress
from urllib.parse import quote

import httpx

from .config.settings import ProviderSettings
from .models import Item
from .observability.logging import get_request_id
from .platform_contract import validate_schema


class RecsflowProvider:
    def __init__(self, settings: ProviderSettings, client=None):
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self.client = client or httpx.Client(base_url=settings.base_url, timeout=settings.timeout_s, headers=headers, trust_env=False)
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def close(self):
        self.client.close()

    def _request(self, method, path, **kwargs):
        with self._lock:
            if time.monotonic() < self._open_until:
                raise ConnectionError("Источник временно отключён после повторных сбоев")
        request_id = get_request_id() or str(uuid.uuid4())
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self.client.request(method, path, headers={"X-Request-ID": request_id}, **kwargs)
                response.raise_for_status()
                with self._lock:
                    self._failures = 0
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code == 429 or exc.response.status_code >= 500
                if not retryable:
                    raise
                if attempt == self.settings.max_retries:
                    with self._lock:
                        self._failures += 1
                        if self._failures >= self.settings.circuit_breaker_threshold:
                            self._open_until = time.monotonic() + self.settings.circuit_breaker_cooldown_s
                    raise
                delay = min(0.1 * 2 ** attempt, 1.0)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    with suppress(ValueError):
                        delay = max(delay, float(exc.response.headers.get("Retry-After", "0")))
                    # Длинное ожидание должен планировать вызывающий сервис.
                    if delay > self.settings.timeout_s:
                        raise
                time.sleep(delay)
        raise RuntimeError("Недостижимая ветка повторов")

    def retrieve(self, user_id, query, limit=100):
        filters = {field: getattr(query, field) for field in ("tone", "max_minutes", "max_seasons", "level", "practical") if getattr(query, field) is not None}
        if query.kind:
            filters["kinds"] = [query.kind]
        if query.genre:
            filters["genres_include"] = [query.genre]
        if query.excluded_genres:
            filters["genres_exclude"] = query.excluded_genres
        data = self._request("POST", "/v1/recommendations", json={"user_id": user_id, "limit": min(limit, 1000), "filters": filters})
        validate_schema(data, "RecommendationResponse")
        return [item["item_id"] for item in data["items"]]

    @staticmethod
    def _item(payload):
        validate_schema(payload, "Item")
        data = {key: value for key, value in payload.items() if key in Item.model_fields}
        data["quality"] = data.get("quality") or 0
        data["description"] = data.get("description") or ""
        data.setdefault("synthetic", False)
        return Item.model_validate(data)

    def lookup(self, item_ids):
        if not item_ids:
            return []
        items = []
        for offset in range(0, len(item_ids), 500):
            payload = self._request("POST", "/v1/items:batchGet", json={"item_ids": item_ids[offset:offset + 500]})
            validate_schema(payload, "BatchResponse")
            items.extend(self._item(item) for item in payload["items"])
        return items

    def find_title(self, title):
        data = self._request("GET", "/v1/items/search", params={"title": title})
        validate_schema(data, "SearchResponse")
        return self._item(data["items"][0]) if len(data["items"]) == 1 else None

    def history(self, user_id):
        return self.history_snapshot(user_id)[0]

    def history_snapshot(self, user_id):
        ids, blocked, cursor, visited = [], [], None, set()
        for _ in range(20):
            params = {"limit": 100, "types": "viewed,liked,disliked"}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", f"/v1/users/{quote(user_id, safe='')}/history", params=params)
            validate_schema(data, "HistoryResponse")
            ids.extend(event["item_id"] for event in data["events"] if event["event_type"] in ("viewed", "liked"))
            blocked.extend(event["item_id"] for event in data["events"] if event["event_type"] == "disliked")
            cursor = data.get("next_cursor")
            if not cursor:
                return list(dict.fromkeys(ids)), set(blocked)
            if cursor in visited:
                raise ValueError("Источник повторяет курсор истории")
            visited.add(cursor)
        raise ValueError("Превышен лимит страниц истории")

    async def readiness_probe(self):
        await asyncio.to_thread(self._request, "GET", "/ready")
        return "Платформа доступна"
