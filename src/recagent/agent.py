"""Shared LangGraph execution, session storage and legacy rules/LLM agent.

WorkflowAgent overrides interpretation and retrieval for the default workflow.
This base handles request locking, budgets, feedback and response telemetry.
"""

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from math import isfinite
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .grounding import explain
from .interpretation import InterpretationIssue, RequestInterpreter
from .models import ChatRequest, ChatResponse, Query
from .observability.logging import get_request_id
from .parsing import OllamaClient, normalize, rule_parse
from .planner import plan_action
from .preferences import PreferenceModel
from .providers import DemoProvider, RecommendationProvider, matches
from .questions import choose_question
from .response_generation import EvidenceResponseGenerator, GroundedOption, LLMGroundedResponseGenerator, with_response_style


class IdempotencyConflict(ValueError):
    """A client reused one message ID for a different logical turn."""


@dataclass
class _MessageReservation:
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Protected by store_lock; includes the holder and every registered waiter.
    users: int = 0


@dataclass
class Session:
    user_id: str
    query: Query = field(default_factory=Query)
    calls: int = 0
    clarifications: int = 0
    question_streak: int = 0
    reactions: dict[str, str] = field(default_factory=dict)
    shown: set[str] = field(default_factory=set)
    last_ids: list[str] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_slot: str | None = None
    skipped_slots: set[str] = field(default_factory=set)
    tokens: int = 0
    unresolved: list[InterpretationIssue] = field(default_factory=list)
    pending_question: str | None = None
    pending_slots: list[str] = field(default_factory=list)
    preferences: PreferenceModel = field(default_factory=PreferenceModel)
    soft_preferences: list[dict[str, object]] = field(default_factory=list)
    restart_on_next_turn: bool = False


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
    llm_usage: dict[str, float]
    structured_request: dict
    clarification_slots: list[str]
    action_reason: str
    validation_status: str
    fallback_reason: str | None
    generation_fallback_reason: str | None
    retry_rejection_reason: str | None
    soft_preferences: list[dict[str, object]]
    unsupported_catalog_request: bool
    catalog_gaps: list[dict[str, str]]
    broad_discovery: bool


class Agent:
    def __init__(
        self,
        provider: RecommendationProvider | None = None,
        mode: str | None = None,
        llm=None,
        max_calls=8,
        max_sessions=500,
        ttl=3600,
        question_policy="legacy",
        max_questions=3,
        question_cost=0.25,
        max_tokens=100_000,
        interpreter: RequestInterpreter | None = None,
        request_adapter=None,
        response_generator=None,
        session_id_factory=None,
        max_message_records=5000,
        message_record_ttl=None,
        catalog_gap_recorder=None,
    ):
        record_ttl = ttl if message_record_ttl is None else message_record_ttl
        if not isinstance(max_message_records, int) or isinstance(max_message_records, bool) or max_message_records < 1:
            raise ValueError("max_message_records must be a positive integer")
        if not isinstance(record_ttl, (int, float)) or isinstance(record_ttl, bool) or not isfinite(record_ttl) or record_ttl <= 0:
            raise ValueError("message_record_ttl must be finite and positive")
        self.max_message_records = max_message_records
        self.message_record_ttl = record_ttl
        self.provider = provider if provider is not None else DemoProvider()
        self.catalog_gap_recorder = catalog_gap_recorder
        self.mode = mode or os.getenv("RECAGENT_MODE", "ollama")
        if self.mode not in ("rules", "ollama"):
            raise ValueError("RECAGENT_MODE must be rules or ollama")
        self.llm = (
            llm
            if llm is not None
            else OllamaClient(os.getenv("OLLAMA_MODEL", "qwen3:8b"), os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
        )
        uses_default_interpreter = interpreter is None
        self.response_generator = (
            response_generator
            if response_generator is not None
            else (
                LLMGroundedResponseGenerator(self.llm)
                if self.mode == "ollama" and isinstance(self.llm, OllamaClient) and uses_default_interpreter
                else EvidenceResponseGenerator()
            )
        )
        from .factory import build_request_components

        default_interpreter, default_adapter = build_request_components(self.llm)
        self.interpreter = interpreter if interpreter is not None else default_interpreter
        self.request_adapter = request_adapter if request_adapter is not None else default_adapter
        self.max_calls, self.max_sessions, self.ttl = max_calls, max_sessions, ttl
        if question_policy not in ("legacy", "none", "fixed", "adaptive", "compound"):
            raise ValueError("Неизвестная политика уточнений")
        self.question_policy = question_policy
        self.max_questions, self.question_cost, self.max_tokens = max_questions, question_cost, max_tokens
        self.session_id_factory = session_id_factory if session_id_factory is not None else uuid.uuid4
        self.sessions = {}
        self.message_records = {}
        self.message_record_reservations = set()
        self.message_locks = {}
        self.store_lock = threading.Lock()
        graph = StateGraph(State)
        for name, node in [
            ("parse", self._parse),
            ("retrieve", self._retrieve),
            ("plan_action", self._plan),
            ("clarify", self._clarify),
            ("rank_explain", self._recommend),
        ]:
            graph.add_node(name, self._timed(name, node))
        graph.add_edge(START, "parse")
        graph.add_conditional_edges("parse", lambda s: "clarify" if s.get("issue") else "retrieve")
        graph.add_conditional_edges(
            "retrieve", lambda s: "clarify" if s.get("issue") else ("plan_action" if self.question_policy == "compound" else "rank_explain")
        )
        graph.add_conditional_edges("plan_action", lambda s: "clarify" if s.get("issue") else "rank_explain")
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

    @staticmethod
    def _merged_usage(current: dict, added: dict) -> dict[str, float]:
        keys = set(current) | set(added)
        return {
            key: float(current.get(key, 0)) + float(added.get(key, 0))
            for key in keys
            if isinstance(current.get(key, 0), (int, float)) and isinstance(added.get(key, 0), (int, float))
        }

    def _get_session(self, request):
        with self.store_lock:
            now = time.monotonic()
            for key in list(self.sessions):
                session = self.sessions[key]
                if now - session.touched > self.ttl and not session.lock.locked():
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
            sid = str(self.session_id_factory())
            if sid in self.sessions:
                raise RuntimeError("Session ID factory returned an existing ID")
            session = Session(user_id=request.user_id)
            self.sessions[sid] = session
            return sid, session

    def chat(self, request: ChatRequest) -> ChatResponse:
        message_key = None
        payload_hash = None
        if request.message_id is not None:
            message_key = (request.user_id, str(request.message_id))
            # The session is part of the logical request.  Omitting it here
            # lets a replay key leak a cached response across conversations.
            payload_hash = hashlib.sha256(request.model_dump_json(exclude_none=True).encode()).hexdigest()
            with self.store_lock:
                reservation = self.message_locks.setdefault(message_key, _MessageReservation())
                reservation.users += 1
            # This lock is intentionally acquired before a session lock. A retry
            # of a first turn therefore waits for the same reservation instead
            # of allocating another session or calling the interpreter again.
            try:
                with reservation.lock:
                    return self._chat_once(request, message_key, payload_hash)
            finally:
                with self.store_lock:
                    reservation.users -= 1
                    # A plain pop after releasing the lock splits queued waiters
                    # from new arrivals. Retain one shared reservation until the
                    # last participant leaves, including error/capacity paths.
                    if reservation.users == 0 and self.message_locks.get(message_key) is reservation:
                        del self.message_locks[message_key]
        return self._chat_once(request, None, None)

    def _chat_once(self, request: ChatRequest, message_key, payload_hash) -> ChatResponse:
        start = time.perf_counter()
        if message_key is not None:
            with self.store_lock:
                now = time.monotonic()
                for key in list(self.message_records):
                    if self.message_records[key][2] <= now:
                        del self.message_records[key]
                existing = self.message_records.get(message_key)
                if existing is None:
                    # Count in-flight distinct keys as well as completed answers.
                    # Reject before session allocation or any graph/LLM work.
                    if len(self.message_records) + len(self.message_record_reservations) >= self.max_message_records:
                        raise RuntimeError("Достигнут лимит записей повторных запросов. Повторите позже.")
                    self.message_record_reservations.add(message_key)
            if existing is not None:
                if existing[0] != payload_hash:
                    raise IdempotencyConflict("message_id уже использован с другим содержимым")
                response = existing[1].model_copy(deep=True)
                original_request_id = response.request_id
                response.request_id = get_request_id()
                response.latency_ms = round((time.perf_counter() - start) * 1000, 2)
                # Replay preserves the saved business answer and its cumulative
                # session totals at original execution. This HTTP attempt did
                # no LLM/provider work and must not charge the original usage.
                response.llm_calls = 0
                response.llm_tokens = 0
                response.llm_usage = {}
                response.timings_ms = {}
                response.telemetry = {
                    **response.telemetry,
                    "request_id": response.request_id,
                    "original_request_id": original_request_id,
                    "cache_hit": True,
                    "llm_calls": 0,
                    "tokens": 0,
                    "latency_ms": response.latency_ms,
                }
                return response
        try:
            return self._execute_turn(request, message_key, payload_hash, start)
        finally:
            if message_key is not None:
                with self.store_lock:
                    self.message_record_reservations.discard(message_key)

    def _execute_turn(self, request, message_key, payload_hash, start) -> ChatResponse:
        sid, session = self._get_session(request)
        with session.lock:
            calls_before = session.calls
            state = self.graph.invoke(
                {"request": request, "session": session, "session_id": sid, "warnings": [], "trace": [], "tokens": 0, "mode": self.mode}
            )
            response = state["response"]
            if response.state == "recommend":
                session.question_streak = 0
            response.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            response.llm_calls = session.calls - calls_before
            response.llm_calls_total = session.calls
            session.tokens += response.llm_tokens
            response.llm_tokens_total = session.tokens
            response.timings_ms = state.get("timings_ms", {})
            response.request_id = get_request_id()
            response.provider = getattr(getattr(self.provider, "capabilities", None), "provider_id", type(self.provider).__name__)
            fallback_reason = state.get("fallback_reason") or state.get("generation_fallback_reason")
            response.telemetry = {
                "request_id": response.request_id,
                "mode": response.mode,
                "provider": response.provider,
                "success": response.state in {"recommend", "clarify"},
                "llm_calls": response.llm_calls,
                "tokens": response.llm_tokens,
                "fallback_reason": fallback_reason,
                "retry_rejection_reason": state.get("retry_rejection_reason"),
                "latency_ms": response.latency_ms,
            }
            if self.catalog_gap_recorder is not None and state.get("catalog_gaps"):
                self.catalog_gap_recorder.record(
                    state["catalog_gaps"],
                    response_state=response.state,
                    mode=response.mode,
                )
            session.touched = time.monotonic()
            if message_key is not None:
                with self.store_lock:
                    # Fixed window starts after a completed execution; replay
                    # does not refresh it. Expired keys are ordinary new requests.
                    self.message_records[message_key] = (
                        payload_hash,
                        response.model_copy(deep=True),
                        time.monotonic() + self.message_record_ttl,
                    )
                    self.message_record_reservations.discard(message_key)
            return response

    def _parse(self, state):
        request, session = state["request"], state["session"]
        text = normalize(request.message)
        previous = session.query
        if session.pending_slots and any(phrase in text for phrase in ("без разницы", "не знаю", "не важно", "любой")):
            session.skipped_slots.update(session.pending_slots)
        session.pending_slots.clear()
        if session.pending_slot and any(phrase in text for phrase in ("без разницы", "не знаю", "не важно", "любой")):
            session.skipped_slots.add(session.pending_slot)
        session.pending_slot = None
        if text in ("сброс", "заново", "начать заново"):
            session.query = Query()
            session.clarifications = 0
            session.question_streak = 0
            session.shown.clear()
            session.last_ids.clear()
            session.skipped_slots.clear()
            session.unresolved.clear()
            session.pending_question = None
            session.pending_slots.clear()
            session.preferences = PreferenceModel()
            # Budget and feedback survive reset: resetting preferences must not bypass the call cap.
            return {"query": session.query, "issue": "Что подбираем: фильм, сериал или курс?", "trace": ["parse", "reset"]}
        query, issue = previous, None
        warnings, tokens, mode, usage = [], 0, self.mode, {}
        fallback_reason = None
        structured = None
        if self.mode == "ollama":
            if session.calls >= self.max_calls or session.tokens >= self.max_tokens:
                warnings.append("Бюджет LLM исчерпан: включён разбор по правилам.")
                mode = "rules_fallback"
                fallback_reason = "llm_budget_exhausted"
            else:
                session.calls += 1
                try:
                    structured, tokens = self.interpreter.interpret(
                        request.message,
                        previous.model_dump(),
                        pending_question=session.pending_question,
                        unresolved=[entry.model_dump() for entry in session.unresolved],
                    )
                    query, session.unresolved = self.request_adapter.apply(structured, previous, request.message, session.unresolved)
                    if session.unresolved:
                        issue = session.unresolved[0].message
                except Exception as exc:
                    warnings.append(f"LLM недоступна или вернула неверную структуру ({type(exc).__name__}): разбор по правилам.")
                    mode = "rules_fallback"
                    fallback_reason = f"llm_failure:{type(exc).__name__}"
                usage = getattr(self.llm, "last_usage", {})
                tokens = max(tokens, int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
        if self.mode == "rules" or mode == "rules_fallback":
            query, issue = rule_parse(request.message, previous)
            if not issue:
                session.unresolved = [
                    entry
                    for entry in session.unresolved
                    if not hasattr(query, entry.field) or getattr(query, entry.field) == getattr(previous, entry.field)
                ]
            if session.unresolved:
                issue = session.unresolved[0].message
        session.pending_question = None
        session.query = query
        session.preferences.update(query, previous, request.message, mode)
        if not issue and not query.kind and not query.seed_title:
            issue = "Что подбираем: фильм, сериал или курс?"
        elif (
            self.question_policy == "legacy"
            and not issue
            and not (query.genre or query.tone or query.seed_title or query.level)
            and session.question_streak < 2
        ):
            issue = "Какой жанр или настроение вам ближе? Для курса можно назвать тему или уровень."
        if query.kind != "course" and (query.level or query.practical is not None):
            issue = "Уровень и практика относятся к курсам. Уточните формат или напишите «сброс»."
        slot = session.unresolved[0].field if session.unresolved else ("kind" if issue and "Что подбираем" in issue else None)
        return {
            "query": query,
            "issue": issue,
            "warnings": warnings,
            "tokens": tokens,
            "mode": mode,
            "trace": ["parse"],
            "clarification_slot": slot,
            "degradation": "NO_LLM" if mode == "rules_fallback" else "FULL",
            "llm_usage": usage,
            "structured_request": structured.model_dump() if structured is not None else None,
            "fallback_reason": fallback_reason,
        }

    def _retrieve(self, state):
        query, session = state["query"], state["session"]
        seed = None
        ids, blocked_history = [], set()
        try:
            if query.seed_title:
                seed = self._find_seed(query.seed_title)
                if seed is None:
                    return {
                        "issue": f"В демонстрационном каталоге нет «{query.seed_title}». Укажите название из каталога или напишите «сброс».",
                        "trace": state["trace"] + ["metadata_lookup"],
                    }
                if not query.kind:
                    query = query.model_copy(update={"kind": seed.kind})
                    session.query = query
            ids = self.provider.retrieve(session.user_id, query, limit=self._provider_limit())
            candidates = self.provider.lookup(ids)
            snapshot = getattr(self.provider, "history_snapshot", None)
            history_ids, blocked_history = snapshot(session.user_id) if snapshot else (self.provider.history(session.user_id), set())
            history_ids += [i for i, reaction in session.reactions.items() if reaction in ("like", "seen")]
            history = self.provider.lookup(list(set(history_ids)))
        except Exception as exc:
            # Never silently replace a real provider's catalog with made-up demo items.
            return {
                "candidates": [],
                "history": [],
                "seed": None,
                "warnings": state["warnings"] + [f"Источник рекомендаций недоступен ({type(exc).__name__}). Повторите запрос позже."],
                "trace": state["trace"] + ["retrieval_failed"],
                "degradation": "NO_EXPLAIN" if ids else "UNAVAILABLE",
                "platform_ids": ids[:5],
            }
        blocked = {i.id for i in history} | blocked_history | {i for i, reaction in session.reactions.items() if reaction == "dislike"}
        if seed and query.intent == "similar":
            blocked.add(seed.id)
        if query.intent == "navigation" and seed:
            candidates = [seed]
            blocked = set()
        if re_more(state["request"].message):
            blocked |= session.shown
        candidates = list({i.id: i for i in candidates if matches(i, query) and i.id not in blocked}.values())
        result = {
            "query": query,
            "candidates": candidates,
            "history": history,
            "seed": seed,
            "trace": state["trace"] + ["retrieval", "metadata_lookup", "hard_filters"],
        }
        if self.question_policy in ("fixed", "adaptive") and query.intent != "navigation" and session.question_streak < self.max_questions:
            question = choose_question(
                query, candidates, policy=self.question_policy, skipped=session.skipped_slots, cost=self.question_cost
            )
            if question:
                result.update(issue=question.message, clarification_slot=question.slot, question_gain=question.gain)
        return result

    def _find_seed(self, title: str):
        """Resolve a seed title for the legacy route.

        Workflow implementations may require a stricter provider operation
        without changing baseline-v1's explicitly supported typo tolerance.
        """
        return self.provider.find_title(title)

    def _provider_limit(self) -> int:
        """Return the provider retrieval window for this implementation."""
        return 100

    def _plan(self, state):
        session = state["session"]
        plan = plan_action(
            state["query"],
            state["candidates"],
            session.skipped_slots,
            self.max_questions - session.question_streak,
            self.question_cost,
        )
        result = {"trace": state["trace"] + ["plan_action"], "action_reason": plan.reason}
        if plan.action == "clarify":
            result.update(
                issue=plan.message,
                clarification_slot=plan.slots[0] if plan.slots else None,
                clarification_slots=list(plan.slots),
                question_gain=plan.gain,
            )
        return result

    def _response(self, state, status, message, recommendations=None):
        return ChatResponse(
            session_id=state["session_id"],
            state=status,
            message=message,
            query=state["query"],
            recommendations=recommendations or [],
            trace=state["trace"] + [status],
            warnings=state["warnings"],
            mode=state["mode"],
            latency_ms=0,
            llm_calls=0,
            llm_calls_total=state["session"].calls,
            llm_tokens=state["tokens"],
            clarification_count=state["session"].clarifications,
            clarification_slot=state.get("clarification_slot"),
            question_gain=state.get("question_gain"),
            degradation=state.get("degradation", "FULL"),
            platform_ids=state.get("platform_ids", []),
            llm_usage=state.get("llm_usage", {}),
            clarification_slots=state.get("clarification_slots", []),
            action_reason=state.get("action_reason"),
            preferences=state["session"].preferences.entries,
            telemetry={},
        )

    def _clarify(self, state):
        if state["session"].question_streak >= self.max_questions:
            return {
                "response": self._response(
                    state, "no_results", "Пока не удалось точно понять пожелание. Напишите запрос целиком одним сообщением: формат и важные детали."
                )
            }
        state["session"].clarifications += 1
        state["session"].question_streak += 1
        state["session"].pending_question = state["issue"]
        state["session"].pending_slot = state.get("clarification_slot")
        state["session"].pending_slots = list(state.get("clarification_slots", []))
        return {"response": self._response(state, "clarify", state["issue"])}

    def _recommend(self, state):
        session, seed = state["session"], state.get("seed")
        history = state["history"]

        def score(item):
            affinity = 0.15 if any(h.genre == item.genre for h in history) else 0
            similarity = 0.5 if seed and seed.genre == item.genre else 0
            similarity += 0.1 if seed and seed.tone == item.tone else 0
            return item.quality + affinity + similarity

        ranked = sorted(state["candidates"], key=lambda i: (-score(i), i.id))[:5]
        recommendations = [
            explain(
                i,
                state["query"],
                history,
                score(i),
                seed,
                tone=state["request"].explanation_tone,
                length=state["request"].explanation_length,
            )
            for i in ranked
        ]
        session.last_ids = [i.id for i in ranked]
        session.shown.update(session.last_ids)
        state["trace"] += ["rerank", "grounding_check"]
        if not ranked:
            message = "Подходящих вариантов не найдено. Можно изменить жанр или снять ограничение: «без ограничений по длительности». Ограничения сохранены."
            if any("Источник рекомендаций недоступен" in w for w in state["warnings"]):
                message = "Не удалось получить данные каталога. Повторите запрос позже."
            return {"response": self._response(state, "no_results", message)}
        options = [
            GroundedOption(id=rec.item.id, title=rec.item.title, claims=rec.claim_texts or [rec.explanation]) for rec in recommendations
        ]
        style = {"tone": state["request"].explanation_tone, "length": state["request"].explanation_length}
        fallback_generator = EvidenceResponseGenerator(**style)
        generator = with_response_style(self.response_generator, **style)
        if getattr(generator, "requires_llm", False) and (
            session.calls >= self.max_calls or session.tokens + state.get("tokens", 0) >= self.max_tokens
        ):
            generator = fallback_generator
            state["warnings"].append("Бюджет LLM исчерпан: ответ собран напрямую из проверенного evidence.")
        try:
            if getattr(generator, "requires_llm", False):
                session.calls += 1
            message, response_tokens = generator.generate(
                original_request=state["request"].message,
                intent=state["query"].intent,
                accepted_constraints=state["query"].model_dump(mode="json"),
                options=options,
                unresolved=[issue.model_dump() for issue in session.unresolved],
            )
            response_usage = getattr(getattr(generator, "backend", None), "last_usage", {})
            response_tokens = max(
                response_tokens,
                int(response_usage.get("input_tokens", 0) + response_usage.get("output_tokens", 0)),
            )
            state["tokens"] = state.get("tokens", 0) + response_tokens
            state["llm_usage"] = self._merged_usage(state.get("llm_usage", {}), response_usage)
        except Exception as exc:
            state["generation_fallback_reason"] = f"response_generation_failure:{type(exc).__name__}"
            response_usage = getattr(getattr(generator, "backend", None), "last_usage", {})
            state["tokens"] = state.get("tokens", 0) + int(response_usage.get("input_tokens", 0) + response_usage.get("output_tokens", 0))
            state["llm_usage"] = self._merged_usage(state.get("llm_usage", {}), response_usage)
            state["warnings"].append(f"Генератор ответа недоступен ({type(exc).__name__}): использован grounded fallback.")
            message, _ = fallback_generator.generate(
                original_request=state["request"].message,
                intent=state["query"].intent,
                accepted_constraints=state["query"].model_dump(mode="json"),
                options=options,
                unresolved=[issue.model_dump() for issue in session.unresolved],
            )
        return {"response": self._response(state, "recommend", message, recommendations)}

    def feedback(self, sid: str, item_id: str, reaction: str, *, user_id: str | None = None):
        if reaction not in ("like", "dislike", "seen"):
            raise ValueError("Неизвестная реакция")
        with self.store_lock:
            session = self.sessions.get(sid)
            if not session or time.monotonic() - session.touched > self.ttl:
                raise KeyError("Сессия не найдена или истекла")
            if user_id is not None and session.user_id != user_id:
                raise KeyError("Сессия не найдена для этого пользователя.")
        with session.lock:
            if item_id not in session.shown:
                raise ValueError("Можно оценить только показанный в сессии объект")
            session.reactions[item_id] = reaction
            session.touched = time.monotonic()


def re_more(message):
    return any(word in normalize(message) for word in ("еще", "другие", "другой вариант"))


def is_exact_more_command(message):
    """Recognize standalone paging commands without parsing mixed requests."""
    text = normalize(message).strip(" .!?;:—–-\t")
    # Only explicit preservation clauses may accompany this system command.
    # Full matching below keeps new preferences and negated commands on the
    # ordinary LLM path instead of silently discarding them.
    for suffix in ("сохранив условия", "сохраняя условия", "без изменения условий", "не меняя условий"):
        if text.endswith(" " + suffix):
            text = text[: -(len(suffix) + 1)].rstrip(" ,;:")
            break
    return text in {
        "еще",
        "еще варианты",
        "другие",
        "другие варианты",
        "другой вариант",
        "покажи еще",
        "покажите еще",
        "показывай еще",
        "покажите еще варианты",
    }
