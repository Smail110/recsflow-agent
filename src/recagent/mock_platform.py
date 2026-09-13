"""Локальная реализация предлагаемого контракта, без доступа к настоящему Recsflow."""

from datetime import UTC, datetime

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from jsonschema import ValidationError

from .platform_contract import validate_schema
from .providers import DemoProvider


def create_mock():
    app = FastAPI(title="Учебная платформа рекомендаций", version="1.0.0")
    provider = DemoProvider()
    app.state.provider = provider
    app.state.failures = set()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        if request.url.path in app.state.failures:
            return JSONResponse({"type": "about:blank", "title": "Источник недоступен", "status": 503}, status_code=503, media_type="application/problem+json")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "mock")
        return response

    @app.exception_handler(ValidationError)
    async def invalid_body(_request, _exc):
        return JSONResponse({"type": "about:blank", "title": "Неверный запрос", "status": 422}, status_code=422, media_type="application/problem+json")

    @app.get("/health")
    @app.get("/ready")
    def health():
        return {"status": "ok", "version": "1.0.0", "checks": {}}

    def metadata(item):
        return item.model_dump(exclude={"popularity"})

    @app.post("/v1/recommendations")
    def recommendations(body: dict = Body(...)):
        validate_schema(body, "RecommendationRequest")
        filters = body.get("filters", {})
        candidates = list(provider.items.values())
        for plural, field in (("kinds", "kind"), ("genres_include", "genre")):
            if filters.get(plural):
                candidates = [item for item in candidates if getattr(item, field) in filters[plural]]
        candidates = [item for item in candidates if item.genre not in filters.get("genres_exclude", [])]
        for field in ("tone", "level", "practical"):
            if field in filters:
                candidates = [item for item in candidates if getattr(item, field) == filters[field]]
        for field, attribute in (("max_minutes", "minutes"), ("max_seasons", "seasons")):
            if field in filters:
                candidates = [item for item in candidates if getattr(item, attribute) is not None and getattr(item, attribute) <= filters[field]]
        candidates.sort(key=lambda item: (-item.quality, item.id))
        return {"items": [{"item_id": item.id, "score": item.quality, "rank": index + 1} for index, item in enumerate(candidates[:body["limit"]])],
                "model_version": "synthetic-quality-v1", "generated_at": datetime.now(UTC).isoformat(),
                "pool_size_before_limit": len(candidates), "filters_applied": filters}

    @app.get("/v1/items/search")
    def search(title: str = Query(min_length=1, max_length=300)):
        exact = [item for item in provider.items.values() if item.title.casefold() == title.casefold()]
        near = provider.find_title(title) if not exact else None
        return {"items": [metadata(item) for item in exact or ([near] if near else [])]}

    @app.get("/v1/items/{item_id}")
    def item(item_id: str):
        found = provider.lookup([item_id])
        if not found:
            return JSONResponse({"type": "about:blank", "title": "Объект не найден", "status": 404}, status_code=404, media_type="application/problem+json")
        return metadata(found[0])

    @app.post("/v1/items:batchGet")
    def batch(body: dict = Body(...)):
        validate_schema(body, "BatchRequest")
        ids = body["item_ids"]
        return {"items": [metadata(item) for item in provider.lookup(ids)], "missing_item_ids": [item_id for item_id in ids if item_id not in provider.items]}

    @app.get("/v1/users/{user_id}/history")
    def history(user_id: str, limit: int = Query(default=100, ge=1, le=1000), cursor: int = Query(default=0, ge=0), types: str = ""):
        events = [{"item_id": item_id, "event_type": "viewed"} for item_id in provider.history(user_id)]
        if types:
            events = [event for event in events if event["event_type"] in types.split(",")]
        page = events[cursor:cursor + limit]
        next_cursor = cursor + limit
        return {"events": page, "next_cursor": str(next_cursor) if next_cursor < len(events) else None}

    return app


app = create_mock()
