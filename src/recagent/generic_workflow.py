"""Schema-driven workflow service for a customer domain outside the demo DTO.

It intentionally has a small surface: an injected schema adapter interprets a
turn, deterministic validation runs before state is committed, the provider
returns canonical records, and the generic residual filter verifies each hard
constraint.  It is the portability path used by onboarding tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Constraint, ConstraintState
from .domains.base import DomainSpec
from .filtering import hard_filter
from .state import reduce_state
from .validation import normalize_proposal, validate_proposal


class GenericWorkflowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    session_id: str
    message: str
    query: dict[str, Any]
    item_ids: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


@dataclass
class GenericWorkflowSession:
    """The same ownership and feedback boundary used by the demo façade."""

    user_id: str
    query: BaseModel
    constraint_state: ConstraintState = field(default_factory=ConstraintState)
    unresolved: list[Any] = field(default_factory=list)
    shown: set[str] = field(default_factory=set)
    reactions: dict[str, str] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class GenericWorkflowService:
    """A reusable domain workflow, independent of the media demo model."""

    domain: DomainSpec
    provider: Any
    interpreter: Any
    adapter: Any
    item_projection: Any = None
    sessions: dict[str, GenericWorkflowSession] = field(default_factory=dict)
    store_lock: threading.Lock = field(default_factory=threading.Lock)

    def _fields(self) -> dict[str, Any]:
        return {field.name: field for field in self.domain.fields}

    def _coverage_surfaces(self) -> dict[str, set[str]]:
        """Use only DomainSpec-declared vocabulary for omission detection."""
        surfaces: dict[str, set[str]] = {}
        for spec in self.domain.fields:
            values = {alias for alias, _ in spec.aliases}
            if values:
                surfaces[spec.name] = values
        for table_name in ("aliases", "scalar_aliases", "numeric_units"):
            for name, values in getattr(self.adapter, table_name, {}).items():
                surfaces.setdefault(name, set()).update(str(value) for value in values)
        return surfaces

    def _constraints(self, query: BaseModel, turn_id: str) -> tuple[Constraint, ...]:
        specs = self._fields()
        values = query.model_dump(exclude_none=True)
        output = []
        for name, value in values.items():
            if name == "intent" or name not in specs:
                continue
            spec = specs[name]
            output.append(
                Constraint(
                    id=f"{turn_id}:{name}",
                    field=spec.item_field or name,
                    op=spec.default_operator,
                    value=value,
                    turn_id=turn_id,
                    domain_version=self.domain.version,
                )
            )
        return tuple(output)

    def _attributes(self, item: Any) -> dict[str, Any]:
        if self.item_projection is not None:
            return dict(self.item_projection(item))
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if isinstance(item, dict):
            return dict(item)
        return vars(item)

    def _pending_constraints(self, pending) -> tuple[Constraint, ...]:
        if pending is None:
            return ()
        return tuple(change.constraint for change in pending.proposal.changes if change.constraint is not None)

    def _session(self, user_id: str, session_id: str | None) -> tuple[str, GenericWorkflowSession]:
        with self.store_lock:
            if session_id is None:
                sid = f"{user_id}:{len(self.sessions) + 1}"
                session = GenericWorkflowSession(user_id=user_id, query=self.adapter.model())
                self.sessions[sid] = session
                return sid, session
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError("Сессия не найдена или истекла.")
            if session.user_id != user_id:
                raise KeyError("Сессия не найдена для этого пользователя.")
            return session_id, session

    def chat(self, *, user_id: str, message: str, session_id: str | None = None) -> GenericWorkflowResponse:
        sid, session = self._session(user_id, session_id)
        with session.lock:
            return self._chat_locked(sid=sid, session=session, user_id=user_id, message=message)

    def _chat_locked(self, *, sid: str, session: GenericWorkflowSession, user_id: str, message: str) -> GenericWorkflowResponse:
        previous, state, unresolved = session.query, session.constraint_state, session.unresolved
        structured, _ = self.interpreter.interpret(message, previous.model_dump(), unresolved=[issue.model_dump() for issue in unresolved])
        proposal = normalize_proposal(
            structured,
            turn_id=sid,
            domain_version=self.domain.version,
            message=message,
            constraint_operators={field.name: field.default_operator for field in self.domain.fields},
        )
        validation = validate_proposal(
            proposal, message=message, known_fields=set(self._fields()), coverage_surfaces=self._coverage_surfaces(), state=state
        )
        trace = ["structured_request", "shape_validation", "proposal_validation"]
        if validation.status == "REJECT":
            session.constraint_state = state.model_copy(update={"pending": self._pending_state(proposal, validation, state)})
            return GenericWorkflowResponse(
                state="clarify",
                session_id=sid,
                message="Уточните условие запроса.",
                query=previous.model_dump(),
                trace=trace,
                findings=[finding.code for finding in validation.findings],
            )
        # A proposal can be valid in isolation while the adapter still reports
        # an explicit ambiguity carried by the structured response. Keep the
        # already-cited fields staged; they become active when the ambiguity is
        # answered on a later turn.
        if structured.issues:
            session.constraint_state = state.model_copy(update={"pending": self._pending_state(proposal, validation, state)})
            session.unresolved = list(structured.issues)
            return GenericWorkflowResponse(
                state="clarify",
                session_id=sid,
                message=structured.issues[0].message,
                query=previous.model_dump(),
                trace=[*trace, "semantic_validation"],
                findings=[issue.kind for issue in structured.issues],
            )
        # Apply the new turn on top of the last accepted DTO. Pending proposals
        # are not active yet, but constraints already committed before the
        # clarification must remain visible to the adapter and provider.
        adapter_previous = previous
        staged_constraints = self._pending_constraints(state.pending)
        if state.constraints or staged_constraints:
            values = previous.model_dump()
            for constraint in (*state.constraints, *staged_constraints):
                field_name = next(
                    (name for name, spec in self._fields().items() if (spec.item_field or name) == constraint.field),
                    constraint.field,
                )
                if field_name in values:
                    values[field_name] = constraint.value
            adapter_previous = self.adapter.model.model_validate(values, strict=True)
        # A clarification answer resolves only the fields it actually updates.
        # Carrying the full unresolved list into the adapter would re-attach an
        # old ambiguity (for example ``недорогой``) to a later numeric answer,
        # causing the accepted category to be discarded with the whole turn.
        updated_fields = {update.field for update in structured.updates}
        remaining_unresolved = [issue for issue in unresolved if issue.field not in updated_fields]
        query, issues = self.adapter.apply(structured, adapter_previous, message, remaining_unresolved)
        if issues:
            session.constraint_state = state.model_copy(update={"pending": self._pending_state(proposal, validation, state)})
            session.unresolved = issues
            return GenericWorkflowResponse(
                state="clarify",
                session_id=sid,
                message=issues[0].message,
                query=previous.model_dump(),
                trace=[*trace, "semantic_validation"],
                findings=[issue.kind for issue in issues],
            )
        accepted = self._constraints(query, sid)
        # A clarification turn may contain only the answer to one pending
        # field. Preserve prior accepted fields until explicitly replaced.
        if state.constraints:
            current_fields = {constraint.field for constraint in accepted}
            retained = [constraint for constraint in state.constraints if constraint.field not in current_fields]
            accepted = tuple(retained) + tuple(accepted)
            query_values = query.model_dump()
            for constraint in retained:
                field_name = next(
                    (name for name, spec in self._fields().items() if (spec.item_field or name) == constraint.field),
                    constraint.field,
                )
                if field_name in query_values and query_values[field_name] is None:
                    query_values[field_name] = constraint.value
            query = self.adapter.model.model_validate(query_values, strict=True)
        accepted_result = validation.model_copy(update={"status": "PASS"})
        new_state = reduce_state(state, accepted_result)
        # The generic reducer gets canonical item-field constraints after schema
        # mapping; its state is never reconstructed from a legacy Query.
        new_state = new_state.model_copy(update={"constraints": accepted, "pending": None, "version": state.version + 1})
        ids = self.provider.retrieve(user_id, query, limit=100)
        records = self.provider.lookup(ids)
        eligible, diagnostics = hard_filter(records, accepted, attributes=self._attributes)
        eligible = [
            item
            for item in eligible
            if str(self._attributes(item).get("id") or self._attributes(item).get("sku"))
            not in {item_id for item_id, reaction in session.reactions.items() if reaction == "dislike"}
        ]
        session.query, session.constraint_state, session.unresolved = query, new_state, []
        item_ids = [str(self._attributes(item).get("id") or self._attributes(item).get("sku")) for item in eligible[:5]]
        session.shown.update(item_ids)
        if not item_ids:
            return GenericWorkflowResponse(
                state="no_results",
                session_id=sid,
                message="Подходящих объектов не найдено.",
                query=query.model_dump(),
                trace=[*trace, "canonical_lookup", "hard_filter"],
                findings=[*diagnostics["failed"], *diagnostics["unknown"]],
            )
        return GenericWorkflowResponse(
            state="recommend",
            session_id=sid,
            message="Подобраны объекты по подтверждённым условиям.",
            query=query.model_dump(),
            item_ids=item_ids,
            trace=[*trace, "canonical_lookup", "hard_filter", "render"],
        )

    def feedback(self, *, user_id: str, session_id: str, item_id: str, reaction: str) -> None:
        if reaction not in {"like", "dislike", "seen"}:
            raise ValueError("Недопустимая реакция")
        _, session = self._session(user_id, session_id)
        with session.lock:
            if item_id not in session.shown:
                raise KeyError("Объект не был показан в этой сессии.")
            session.reactions[item_id] = reaction

    @staticmethod
    def _pending_state(proposal, validation, state):
        """Store a typed pending contract until the clarification is answered."""
        from .contracts import PendingState

        blocker = next((finding for finding in validation.findings if finding.status != "pass"), None)
        return PendingState(
            proposal=proposal,
            findings=validation.findings,
            question_id=f"{proposal.raw_request_hash}:{blocker.field if blocker else 'proposal'}",
            target_change_ids=blocker.change_ids if blocker else (),
            target_field=blocker.field if blocker else None,
            base_version=state.version,
        )
