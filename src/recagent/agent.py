import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .grounding import explain
from .models import ChatRequest, ChatResponse, Query
from .parsing import OllamaClient, normalize, rule_parse
from .providers import DemoProvider, RecommendationProvider, matches
from .questions import choose_question


@dataclass
class Session:
    user_id: str
    query: Query = field(default_factory=Query)
    calls: int = 0
    clarifications: int = 0
    reactions: dict[str, str] = field(default_factory=dict)
    shown: set[str] = field(default_factory=set)
    last_ids: list[str] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_slot: str | None = None
    skipped_slots: set[str] = field(default_factory=set)
    tokens: int = 0


class State(TypedDict, total=False):
    request: ChatRequest
    session: Session
    session_id: str
    query: Query
    issue: str | None
    warnings: list[str]
    trace: list[str]
    mode: str
    tokens: int
    candidates: list
    history: list
    seed: object
    response: ChatResponse
    clarification_slot: str | None
    question_gain: float
    degradation: str
    platform_ids: list[str]
    timings_ms: dict[str, float]


class Agent:
    def __init__(self, provider: RecommendationProvider | None = None, mode: str | None = None, llm=None, max_calls=8, max_sessions=500, ttl=3600,
                 question_policy="legacy", max_questions=3, question_cost=0.25, max_tokens=100_000):
        self.provider = provider if provider is not None else DemoProvider()
        self.mode = mode or os.getenv("RECAGENT_MODE", "rules")
        if self.mode not in ("rules", "ollama"):
            raise ValueError("RECAGENT_MODE must be rules or ollama")
        self.llm = llm if llm is not None else OllamaClient(os.getenv("OLLAMA_MODEL", "qwen3:8b"), os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
        self.max_calls, self.max_sessions, self.ttl = max_calls, max_sessions, ttl
        if question_policy not in ("legacy", "none", "fixed", "adaptive"):
            raise ValueError("Неизвестная политика уточнений")
        self.question_policy = question_policy
        self.max_questions, self.question_cost, self.max_tokens = max_questions, question_cost, max_tokens
        self.sessions = {}
        self.store_lock = threading.Lock()
        graph = StateGraph(State)
        for name, node in [("parse", self._parse), ("retrieve", self._retrieve), ("clarify", self._clarify), ("rank_explain", self._recommend)]:
            graph.add_node(name, self._timed(name, node))
        graph.add_edge(START, "parse")
        graph.add_conditional_edges("parse", lambda s: "clarify" if s.get("issue") else "retrieve")
        graph.add_conditional_edges("retrieve", lambda s: "clarify" if s.get("issue") else "rank_explain")
        graph.add_edge("clarify", END)
        graph.add_edge("rank_explain", END)
        self.graph = graph.compile()

    @staticmethod
    def _timed(name, node):
        def measured(state):
            started = time.perf_counter()
            result = node(state)
            result["timings_ms"] = state.get("timings_ms", {}) | {name: round((time.perf_counter() - started) * 1000, 3)}
            return result
        return measured

    def _get_session(self, request):
        with self.store_lock:
            now = time.monotonic()
            for key in list(self.sessions):
                session = self.sessions[key]
                if now-session.touched > self.ttl and not session.lock.locked():
                    del self.sessions[key]
            if request.session_id:
                if request.session_id not in self.sessions:
                    raise KeyError("Сессия не найдена или истекла. Начните новый диалог.")
                session = self.sessions[request.session_id]
                if session.user_id != request.user_id:
                    raise KeyError("Сессия не найдена для этого пользователя.")
                session.touched = now
                return request.session_id, session
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError("Достигнут лимит активных сессий. Повторите позже.")
            sid = str(uuid.uuid4())
            session = Session(user_id=request.user_id)
            self.sessions[sid] = session
            return sid, session

    def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        sid, session = self._get_session(request)
        with session.lock:
            calls_before = session.calls
            state = self.graph.invoke({"request": request, "session": session, "session_id": sid, "warnings": [], "trace": [], "tokens": 0, "mode": self.mode})
            response = state["response"]
            response.latency_ms = round((time.perf_counter()-start)*1000, 2)
            response.llm_calls = session.calls-calls_before
            response.llm_calls_total = session.calls
            session.tokens += response.llm_tokens
            response.llm_tokens_total = session.tokens
            response.timings_ms = state.get("timings_ms", {})
            session.touched = time.monotonic()
            return response

    def _parse(self, state):
        request, session = state["request"], state["session"]
        text = normalize(request.message)
        previous = session.query
        if session.pending_slot and any(phrase in text for phrase in ("без разницы", "не знаю", "не важно", "любой")):
            session.skipped_slots.add(session.pending_slot)
        session.pending_slot = None
        if text in ("сброс", "заново", "начать заново"):
            session.query = Query()
            session.clarifications = 0
            session.shown.clear()
            session.last_ids.clear()
            session.skipped_slots.clear()
            # Budget and feedback survive reset: resetting preferences must not bypass the call cap.
            return {"query": session.query, "issue": "Что подбираем: фильм, сериал или курс?", "trace": ["parse", "reset"]}
        query, issue = rule_parse(request.message, previous)
        domain_changed = bool(previous.kind and query.kind and previous.kind != query.kind)
        warnings, tokens, mode = [], 0, self.mode
        if self.mode == "ollama" and not issue:
            if session.calls >= self.max_calls or session.tokens >= self.max_tokens:
                warnings.append("Бюджет LLM исчерпан: включён разбор по правилам.")
                mode = "rules_fallback"
            else:
                session.calls += 1
                try:
                    extracted, tokens = self.llm.parse(request.message, Query() if domain_changed else previous)
                    if domain_changed:
                        # A model may still leak old domain slots; never retain those without fresh evidence.
                        data = extracted.model_dump()
                        for name in ("genre", "tone", "max_seasons", "max_minutes", "level", "practical", "seed_title"):
                            if getattr(query, name) is None and data[name] == getattr(previous, name):
                                data[name] = None
                        extracted = Query.model_validate(data)
                    # Explicitly recognized constraints take precedence over LLM guesses.
                    query, issue = rule_parse(request.message, extracted)
                except Exception as exc:
                    warnings.append(f"LLM недоступна или вернула неверную структуру ({type(exc).__name__}): разбор по правилам.")
                    mode = "rules_fallback"
        session.query = query
        if not issue and not query.kind and not query.seed_title:
            issue = "Что подбираем: фильм, сериал или курс?"
        elif self.question_policy == "legacy" and not issue and not (query.genre or query.tone or query.seed_title or query.level) and session.clarifications < 2:
            issue = "Какой жанр или настроение вам ближе? Для курса можно назвать тему или уровень."
        if query.kind != "course" and (query.level or query.practical is not None):
            issue = "Уровень и практика относятся к курсам. Уточните формат или напишите «сброс»."
        slot = "kind" if issue and "Что подбираем" in issue else None
        return {"query": query, "issue": issue, "warnings": warnings, "tokens": tokens, "mode": mode, "trace": ["parse"],
                "clarification_slot": slot, "degradation": "NO_LLM" if mode == "rules_fallback" else "FULL"}

    def _retrieve(self, state):
        query, session = state["query"], state["session"]
        seed = None
        try:
            if query.seed_title:
                seed = self.provider.find_title(query.seed_title)
                if seed is None:
                    return {"issue": f"В демонстрационном каталоге нет «{query.seed_title}». Укажите название из каталога или напишите «сброс».", "trace": state["trace"]+["metadata_lookup"]}
                if not query.kind:
                    query = query.model_copy(update={"kind": seed.kind})
                    session.query = query
            ids = self.provider.retrieve(session.user_id, query, limit=100)
            candidates = self.provider.lookup(ids)
            history_ids = self.provider.history(session.user_id)
            history_ids += [i for i, reaction in session.reactions.items() if reaction in ("like", "seen")]
            history = self.provider.lookup(list(set(history_ids)))
        except Exception as exc:
            # Never silently replace a real provider's catalog with made-up demo items.
            return {"candidates": [], "history": [], "seed": None, "warnings": state["warnings"]+[f"Источник рекомендаций недоступен ({type(exc).__name__}). Повторите запрос позже."], "trace": state["trace"]+["retrieval_failed"]}
        blocked = {i.id for i in history} | {i for i, reaction in session.reactions.items() if reaction == "dislike"}
        if seed and query.intent == "similar":
            blocked.add(seed.id)
        if query.intent == "navigation" and seed:
            candidates = [seed]
            blocked = set()
        if re_more(state["request"].message):
            blocked |= session.shown
        candidates = list({i.id: i for i in candidates if matches(i, query) and i.id not in blocked}.values())
        result = {"query": query, "candidates": candidates, "history": history, "seed": seed,
                  "trace": state["trace"]+["retrieval", "metadata_lookup", "hard_filters"]}
        if self.question_policy in ("fixed", "adaptive") and query.intent != "navigation" and session.clarifications < self.max_questions:
            question = choose_question(query, candidates, policy=self.question_policy, skipped=session.skipped_slots, cost=self.question_cost)
            if question:
                result.update(issue=question.message, clarification_slot=question.slot, question_gain=question.gain)
        return result

    def _response(self, state, status, message, recommendations=None):
        return ChatResponse(session_id=state["session_id"], state=status, message=message, query=state["query"], recommendations=recommendations or [], trace=state["trace"]+[status], warnings=state["warnings"], mode=state["mode"], latency_ms=0, llm_calls=0, llm_calls_total=state["session"].calls, llm_tokens=state["tokens"], clarification_count=state["session"].clarifications,
                            clarification_slot=state.get("clarification_slot"), question_gain=state.get("question_gain"),
                            degradation=state.get("degradation", "FULL"), platform_ids=state.get("platform_ids", []))

    def _clarify(self, state):
        if state["session"].clarifications >= self.max_questions:
            return {"response": self._response(state, "no_results", "Достигнут лимит уточнений. Сформулируйте запрос целиком или начните заново.")}
        state["session"].clarifications += 1
        state["session"].pending_slot = state.get("clarification_slot")
        return {"response": self._response(state, "clarify", state["issue"])}

    def _recommend(self, state):
        session, seed = state["session"], state.get("seed")
        history = state["history"]
        def score(item):
            affinity = .15 if any(h.genre == item.genre for h in history) else 0
            similarity = .5 if seed and seed.genre == item.genre else 0
            similarity += .1 if seed and seed.tone == item.tone else 0
            return item.quality + affinity + similarity
        ranked = sorted(state["candidates"], key=lambda i: (-score(i), i.id))[:5]
        recommendations = [explain(i, state["query"], history, score(i), seed) for i in ranked]
        session.last_ids = [i.id for i in ranked]
        session.shown.update(session.last_ids)
        state["trace"] += ["rerank", "grounding_check"]
        if not ranked:
            message = "Подходящих вариантов не найдено. Можно изменить жанр или снять ограничение: «без ограничений по длительности». Ограничения сохранены."
            if any("Источник рекомендаций недоступен" in w for w in state["warnings"]):
                message = "Не удалось получить данные каталога. Повторите запрос позже."
            return {"response": self._response(state, "no_results", message)}
        return {"response": self._response(state, "recommend", f"Подобрал вариантов: {len(ranked)}. Объяснения проверены по данным каталога.", recommendations)}

    def feedback(self, sid: str, item_id: str, reaction: str):
        if reaction not in ("like", "dislike", "seen"):
            raise ValueError("Неизвестная реакция")
        with self.store_lock:
            session = self.sessions.get(sid)
            if not session or time.monotonic()-session.touched > self.ttl:
                raise KeyError("Сессия не найдена или истекла")
        with session.lock:
            if item_id not in session.shown:
                raise ValueError("Можно оценить только показанный в сессии объект")
            session.reactions[item_id] = reaction
            session.touched = time.monotonic()


def re_more(message):
    return any(word in normalize(message) for word in ("еще", "другие", "другой вариант"))
