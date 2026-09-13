from fastapi import FastAPI, HTTPException

from .agent import Agent
from .models import ChatRequest, ChatResponse, FeedbackRequest


def create_app(agent: Agent | None = None) -> FastAPI:
    service = agent if agent is not None else Agent()
    app = FastAPI(title="RecAgent prototype", version="0.1.0", description="Демонстрационный каталог. Контракт RecAgent, не API Recsflow. Сессии живут в памяти одного процесса.")
    app.state.agent = service

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": service.mode, "provider": type(service.provider).__name__}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        try:
            return service.chat(request)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/v1/feedback")
    def feedback(request: FeedbackRequest):
        try:
            service.feedback(request.session_id, request.item_id, request.reaction)
            return {"status": "saved"}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()

