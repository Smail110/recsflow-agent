"""Main recommendation workflow used by the API, UI and evaluation runners.

LLM updates are validated and committed atomically before retrieval. Candidates
then pass canonical filtering, ranking and grounded response generation.
Session management and graph execution are inherited from Agent.
"""

from __future__ import annotations

import json
import re

from .agent import Agent, is_exact_more_command, re_more
from .contracts import Constraint, ConstraintState, NormalizedProposal, ValidationFinding, project_constraints_to_query
from .filtering import hard_filter
from .grounding import explain
from .interpretation import ConstraintUpdate, InterpretationIssue, LLMRequestInterpreter, StructuredRequest
from .models import Query
from .parsing import normalize, rule_parse
from .pending_preview import build_pending_preview
from .preferences import PreferenceModel
from .query_compilation import compile_constraints
from .questions import choose_question
from .ranking import diversify_head, reciprocal_rank_fusion, soft_rank
from .response_generation import EvidenceResponseGenerator, GroundedOption
from .retrieval import BM25Index
from .state import merge_pending_changes, reduce_state
from .validation import normalize_proposal, validate_proposal


def _declines_optional_preference(message: str) -> bool:
    """Recognize short uncertainty answers, without swallowing new conditions."""
    text = " ".join(normalize(message).strip(" .!?,").split())
    if text.startswith("ну "):
        text = text[3:]
    return text in {
        "не знаю",
        "не знаю даже",
        "даже не знаю",
        "без разницы",
        "не важно",
        "неважно",
        "все равно",
        "мне все равно",
        "любой",
        "любая",
        "любое",
        "как угодно",
        "не принципиально",
    }


class _WorkflowPromptBackend:
    """Adds the v2 extraction checklist without changing the frozen LoRA prompt."""

    suffix = (
        " Перед ответом сверяй каждую явную поверхность текущего сообщения с domain.aliases, "
        "domain.scalar_aliases и domain.numeric_units: один фрагмент может подтверждать несколько полей, "
        "и тогда нужны отдельные updates для каждого поля. Не оставляй такую поверхность только в одном update. "
        "Например, «лёгкий детективный сериал» требует отдельные updates tone=лёгкий, genre=детектив и "
        "kind=series с короткими цитатами; «курс Python с практикой» — kind=course, genre=python и "
        "practical=true. Для булевого отрицания «без практики» используй set practical=false; exclude "
        "предназначен для явно поддерживаемого поля исключений. Никогда не цитируй в source_text слово, "
        "которого нет в текущем message."
    )

    def __init__(self, backend):
        self.backend = backend

    def structured(self, schema, system, payload):
        return self.backend.structured(schema, system + self.suffix, payload)


class WorkflowAgent(Agent):
    implementation = "workflow-v2"
    migration_stage = "core"

    def _clarify(self, state):
        if state.get("unsupported_catalog_request"):
            return {"response": self._response(state, "no_results", state["issue"])}
        return super()._clarify(state)

    def _response(self, state, status, message, recommendations=None):
        response = super()._response(state, status, message, recommendations)
        if (
            status != "clarify"
            or state.get("validation_status") not in {"UNCERTAIN", "REJECT"}
            or not state.get("structured_request")
            or state.get("mode") != "ollama"
        ):
            return response
        # This is only a view: do not modify session/query or pass staged values
        # to retrieval. Reuse the ordinary adapter, never infer extra values.
        try:
            session = state["session"]
            turn_id = f"{state['session_id']}:{session.calls}"
            canonical = self.request_adapter.normalize(
                StructuredRequest.model_validate(state["structured_request"]),
                state["request"].message,
            )
            verified = normalize_proposal(
                canonical, turn_id=turn_id, message=state["request"].message, constraint_operators=self.request_adapter.constraint_operators
            )
            preview = build_pending_preview(
                session.constraint_state,
                verified,
                turn_id=turn_id,
                known_fields=set(self.request_adapter.model.model_fields),
                domain_field=self.request_adapter.domain_field,
                labels=self.request_adapter.display_labels,
            )
        except Exception:
            # An unavailable optional display must not break the actual response
            # or turn uncertain values into active constraints.
            return response
        return response.model_copy(update={"pending_preview": preview})

    def __init__(self, *args, interpreter=None, **kwargs):
        super().__init__(*args, interpreter=interpreter, **kwargs)
        if interpreter is None:
            self.set_interpretation_transport("flat")

    def set_interpretation_transport(self, transport: str, *, domain_value_required: bool = True) -> None:
        """Select an explicit experimental wire format without changing v2 defaults."""
        if transport not in {"flat", "domain-fields"}:
            raise ValueError(f"Неизвестный interpretation transport: {transport}")
        from .domains.demo import domain_spec

        self.interpreter = LLMRequestInterpreter(
            _WorkflowPromptBackend(self.llm),
            self.request_adapter.descriptor,
            transport=transport,
            domain_spec=domain_spec(),
            domain_value_required=domain_value_required,
        )
        self.interpretation_transport = transport
        self.domain_value_required = domain_value_required

    def _runtime_options(self) -> dict[str, int]:
        config = getattr(self, "workflow_config", {})
        validation = config.get("validation", {})
        retrieval = config.get("retrieval", {})
        ranking = config.get("ranking", {})
        if validation.get("semantic_mode", "code-only") != "code-only" or validation.get("nli", False):
            raise RuntimeError("Текущий workflow-v2 поддерживает только проверенный code-only validator.")
        if retrieval.get("dense", False) or config.get("reranker", False):
            raise RuntimeError("Непроверенный dense/reranker нельзя включить в workflow-v2.")
        if ranking.get("method", "deterministic-rrf") != "deterministic-rrf":
            raise RuntimeError("workflow-v2 требует deterministic-rrf ranking.")
        options = {
            "provider_k": int(retrieval.get("provider_k", 100)),
            "lexical_k": int(retrieval.get("lexical_k", 100)),
            "fused_k": int(retrieval.get("fused_k", 150)),
            "rrf_k": int(ranking.get("rrf_k", retrieval.get("rrf_k", 60))),
        }
        invalid = [name for name, value in options.items() if value <= 0]
        if invalid:
            raise RuntimeError(f"workflow-v2 требует положительные retrieval limits: {', '.join(invalid)}")
        return options

    def _coverage_surfaces(self) -> dict[str, set[str]]:
        """Declared adapter vocabulary used only to detect omitted LLM updates."""
        adapter = self.request_adapter
        surfaces: dict[str, set[str]] = {}
        for table_name in ("enums", "aliases", "scalar_aliases"):
            table = getattr(adapter, table_name, {})
            for field, values in table.items():
                # Detect omitted fields from declared vocabulary; this check
                # does not extract values or create constraints.
                declared = {str(key) for key in values}
                # Scalar values such as True or 1 are not text aliases.
                if table_name != "scalar_aliases":
                    declared.update(str(value) for value in values.values())
                surfaces.setdefault(field, set()).update(declared)
        return surfaces

    def _coverage_dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {}
        for source, mappings in getattr(self.request_adapter, "implied_values", {}).items():
            for implied in mappings.values():
                for target in implied:
                    dependencies.setdefault(target, set()).add(source)
        return dependencies

    def _project_query(self, constraints: tuple[Constraint, ...]) -> Query:
        fields = getattr(self.request_adapter, "constraint_item_fields", {})
        mapped = tuple(c.model_copy(update={"field": fields.get(c.field, c.field)}) for c in constraints)
        return project_constraints_to_query(mapped)[0]

    def _declared_relations(self, proposal: NormalizedProposal) -> NormalizedProposal:
        """Derive only schema-declared values, retaining the source evidence."""
        changes = list(proposal.changes)
        explicit = {
            (change.constraint.field, change.constraint.op, change.constraint.value)
            for change in changes
            if change.constraint is not None and change.operation in {"add", "replace"}
        }
        for change in proposal.changes:
            source = change.constraint
            if source is None or source.op != "eq" or change.operation not in {"add", "replace"}:
                continue
            targets = self.request_adapter.implied_values.get(source.field, {}).get(source.value, {})
            for field, value in targets.items():
                if (field, "eq", value) in explicit:
                    continue
                constraint = source.model_copy(update={"id": f"{source.id}:implied:{field}", "field": field, "value": value})
                changes.append(change.model_copy(update={"id": constraint.id, "constraint": constraint}))
        return proposal.model_copy(update={"changes": tuple(changes)})

    def _domain_changed(self, proposal: NormalizedProposal, state: ConstraintState) -> bool:
        field = self.request_adapter.domain_field
        previous = next((c.value for c in state.constraints if c.field == field and c.op == "eq"), None)
        values = {
            change.constraint.value
            for change in proposal.changes
            if change.constraint is not None
            and change.constraint.field == field
            and change.constraint.op == "eq"
            and change.operation in {"add", "replace"}
        }
        return previous is not None and len(values) == 1 and previous not in values

    def _unsupported_wishes(self, findings, message: str, *, prior_findings=()) -> list[tuple[str, str]]:
        """Return verified unsupported descriptions when no other blocker exists."""
        unsupported = []
        unsupported_fields = set()
        invalid_prior_fields = {
            item.field for item in prior_findings if item.code in {"issue_missing_evidence", "issue_evidence_not_in_turn"}
        }
        trusted_prior = set()
        for item in prior_findings:
            if item.code != "interpretation_issue" or item.field in invalid_prior_fields or not item.details:
                continue
            try:
                issue = json.loads(item.details)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(issue, dict) and isinstance(issue.get("source_text"), str):
                trusted_prior.add((item.field, issue["source_text"]))
        for item in findings:
            if item.code != "interpretation_issue" or not item.details:
                continue
            try:
                issue = json.loads(item.details)
            except (TypeError, json.JSONDecodeError):
                continue
            source = issue.get("source_text") if isinstance(issue, dict) else None
            if (
                isinstance(issue, dict)
                and issue.get("kind") == "unsupported_constraint"
                and item.field in self.request_adapter.soft_description_fields
                and isinstance(source, str)
                and (self.request_adapter._source_is_cited(source, message) or (item.field, source) in trusted_prior)
            ):
                if (item.field, source) not in unsupported:
                    unsupported.append((item.field, source))
                unsupported_fields.add(item.field)
        if unsupported and all(
            item.status == "pass"
            or (item.code in {"interpretation_issue", "adapter_unsupported_constraint"} and item.field in unsupported_fields)
            or (item.code == "constraint_conflict" and item.field in unsupported_fields)
            for item in findings
        ):
            return unsupported
        return []

    @staticmethod
    def _collapsed_unsupported(wishes: list[tuple[str, str]]) -> bool:
        fields = [field for field, _ in wishes]
        return len(fields) != len(set(fields))

    def _clarification_message(self, findings, message: str = "", *, prior_findings=(), has_domain: bool = True) -> str:
        for candidate in (*findings, *prior_findings):
            if candidate.code != "interpretation_issue":
                continue
            try:
                issue = json.loads(candidate.details)
            except (TypeError, ValueError):
                continue
            if not isinstance(issue, dict):
                continue
            field, value = issue.get("field"), issue.get("value")
            if (
                issue.get("kind") != "conflict"
                or field not in self.request_adapter.enums
                or not isinstance(value, str)
                or not value.strip()
            ):
                continue
            probe = StructuredRequest(updates=[ConstraintUpdate(field=field, value=value, source_text=value)])
            if self.request_adapter.detect_polarity_conflicts(probe, message).issues:
                label = "жанром" if field == "genre" else "условием"
                return f"С {label} «{value}» есть противоречие: оставить в подборе или исключить?"
        finding = next((item for item in findings if item.status != "pass" and item.field is not None), None)
        if finding is None:
            finding = next((item for item in prior_findings if item.status != "pass" and item.field is not None), None)
        if finding is None:
            finding = next((item for item in findings if item.status != "pass"), None)
        field = finding.field if finding else None
        if finding and finding.code == "intent_reference_missing":
            return "Назовите объект, который нужно найти или для которого подобрать похожие варианты."
        # An unknown, cited topic is not an absent preference. Do not ask for
        # the same genre again or pretend that a nearby enum value was requested.
        unsupported = self._unsupported_wishes(findings, message, prior_findings=prior_findings)
        if unsupported:
            quoted = [f"«{source}»" for _, source in unsupported]
            wishes = " и ".join([", ".join(quoted[:-1]), quoted[-1]]) if len(quoted) > 1 else quoted[0]
            noun = "пожелания" if len(quoted) > 1 else "пожелание"
            if self._collapsed_unsupported(unsupported):
                return (
                    f"Не могу проверить по каталогу {noun} {wishes}, поэтому не стану выдавать случайные совпадения. "
                    "Этот запрос не применён. Следующее сообщение начнёт новый подбор: укажите формат и пожелания заново."
                )
            if len(quoted) > 1:
                return (
                    f"Не могу проверить по каталогу {noun} {wishes}, поэтому не стану выдавать случайные совпадения. "
                    "Напишите, какие условия можно изменить, и я продолжу подбор."
                )
            return (
                f"Не могу проверить по каталогу {noun} {wishes}, поэтому не стану выдавать случайные совпадения. "
                "Напишите, можно ли снять это условие, и я продолжу подбор."
            )
        question = self.request_adapter.field_questions.get(field)
        if question:
            return question
        if field is None or field not in self.request_adapter.display_labels:
            if not has_domain:
                return self.request_adapter.domain_question(message)
            return "Не понял, что изменить в подборе. Напишите пожелание другими словами: тему, настроение или важное ограничение."
        if field == self.request_adapter.domain_field and has_domain:
            return "Что именно изменить в подборе: формат, жанр или другое пожелание?"
        label = self.request_adapter.display_labels.get(field, "условие запроса")
        aliases = self.request_adapter.aliases.get(field, {})
        options = ", ".join(
            next((alias for alias, canonical in aliases.items() if canonical == value), value) if field == "kind" else value
            for value in self.request_adapter.enums.get(field, {}).values()
        )
        return f"Уточните «{label}»" + (f": {options}." if options else ". Назовите желаемое значение.")

    @staticmethod
    def _constraint_context(constraints: tuple[Constraint, ...]) -> list[dict[str, object]]:
        """Serialize trusted v2 constraints without leaking legacy Query keys."""

        return [
            {
                "field": constraint.field,
                "operator": constraint.op,
                "value": constraint.value,
                "source_turn": constraint.turn_id,
            }
            for constraint in constraints
        ]

    def _interpretation_context(self, state: ConstraintState) -> dict[str, object]:
        """The prior context shown to extraction; Query is an output-only projection."""

        return {"intent": state.intent, "trusted_active": self._constraint_context(state.constraints)}

    def _pending_context(self, state: ConstraintState) -> dict[str, object] | None:
        pending = state.pending
        if pending is None:
            return None
        return {
            "question_id": pending.question_id,
            "target_change_ids": list(pending.target_change_ids),
            "target_field": pending.target_field,
            "base_version": pending.base_version,
            "trusted_active": self._constraint_context(state.constraints),
            "trusted_staged": self._constraint_context(
                tuple(
                    change.constraint
                    for change in pending.proposal.changes
                    if change.constraint is not None
                    and not any(
                        finding.status != "pass" and (change.id in finding.change_ids or finding.field == change.constraint.field)
                        for finding in pending.findings
                    )
                )
            ),
            # Diagnostics describe the extractor, not new user preferences.
            # In particular do not feed recursively serialized model issues
            # back as instructions to reproduce an earlier ambiguity.
            "blockers": [
                {"field": field}
                for field in sorted({finding.field for finding in pending.findings if finding.status != "pass" and finding.field})
            ],
        }

    def _discard_answered_conflict(self, request: StructuredRequest, message: str, state: ConstraintState) -> StructuredRequest:
        """Ignore a model's stale conflict only after a cited answer to that conflict.

        The previous diagnostic is context, not evidence that the new turn is
        contradictory. A new contradiction or an answer about another value
        remains subject to ordinary validation.
        """
        pending = state.pending
        if pending is None or pending.base_version != state.version:
            return request
        targets = set()
        for finding in pending.findings:
            if finding.code != "interpretation_issue":
                continue
            try:
                prior = json.loads(finding.details)
            except (TypeError, ValueError):
                continue
            if prior.get("kind") == "conflict" and prior.get("field") and prior.get("value"):
                targets.add((prior["field"], str(prior["value"])))
        if len(targets) != 1:
            return request
        field, value = next(iter(targets))
        answers = [
            update for update in request.updates
            if update.field == field and str(update.value) == value and update.operation in {"exclude", "include"}
        ]
        if len(answers) != 1 or len([u for u in request.updates if u.field == field]) != 1:
            return request
        answer = answers[0]
        if self.request_adapter._operation_evidence(answer, message, answer.operation) is None:
            return request
        current = self.request_adapter.detect_polarity_conflicts(StructuredRequest(updates=[answer]), message)
        if current.issues:
            return request
        def answered_issue(issue: InterpretationIssue) -> bool:
            if issue.kind != "conflict" or issue.field != field:
                return False
            if issue.value in {"", value}:
                return True
            if field not in self.request_adapter.enums:
                return False
            resolved = self.request_adapter.resolver.resolve(
                self.request_adapter._field_spec(field), issue.value, issue.source_text or ""
            )
            return resolved.status == "canonical" and str(resolved.value) == value

        remaining = [issue for issue in request.issues if not answered_issue(issue)]
        if len(remaining) == len(request.issues):
            return request
        return request.model_copy(update={"issues": remaining, "clarification_required": bool(remaining)})

    def _filter_constraints(self, constraints: tuple[Constraint, ...]) -> tuple[Constraint, ...]:
        """Map schema fields to canonical metadata without mutating dialogue state."""
        fields = getattr(self.request_adapter, "constraint_item_fields", {})
        # A seed title controls metadata lookup and similarity/navigation
        # routing; it is not a catalog attribute to send through hard filtering.
        return tuple(
            constraint.model_copy(update={"field": fields.get(constraint.field, constraint.field)})
            for constraint in constraints
            if constraint.field != "seed_title"
        )

    def _find_seed(self, title: str):
        """Use v2's exact, uniqueness-backed title operation only."""
        finder = getattr(self.provider, "find_title_verified", None)
        if not callable(finder):
            return None
        return finder(title)

    @staticmethod
    def _pending_sources(proposal: NormalizedProposal) -> NormalizedProposal:
        """Mark carried evidence so validation never recites it as new text."""
        changes = []
        for change in proposal.changes:
            spans = tuple(span.model_copy(update={"origin": "pending_source"}) for span in change.source_spans)
            constraint = change.constraint
            if constraint is not None:
                constraint = constraint.model_copy(update={"source_spans": spans})
            changes.append(change.model_copy(update={"constraint": constraint, "source_spans": spans}))
        return proposal.model_copy(update={"changes": tuple(changes)})

    def _merge_pending(self, proposal: NormalizedProposal, state: ConstraintState, *, reset: bool) -> NormalizedProposal:
        """Replace only answered target fields; retain other staged evidence."""
        pending = state.pending
        # ``reset_constraints`` has no source span in StructuredRequest. It
        # therefore cannot authorize throwing away a staged user request. The
        # explicit reset commands are handled before interpretation in _parse;
        # an ungrounded model flag is retained as an ordinary correction turn.
        del reset
        if pending is None or pending.base_version != state.version:
            return proposal
        # New updates are evidence from this turn.  The deterministic validator
        # will re-evaluate every retained and new change as one atomic proposal.
        return proposal.model_copy(
            update={
                "changes": merge_pending_changes(self._pending_sources(pending.proposal).changes, proposal.changes),
                "intent": proposal.intent or pending.proposal.intent,
            }
        )

    @staticmethod
    def _findings_as_issues(findings: tuple[ValidationFinding, ...]) -> list[InterpretationIssue]:
        issues: list[InterpretationIssue] = []
        for finding in findings:
            if finding.status == "pass":
                continue
            field = finding.field or "request"
            kind = "conflict" if finding.code == "constraint_conflict" else "ambiguity"
            issues.append(
                InterpretationIssue(
                    kind=kind,
                    field=field,
                    value=finding.code,
                    message=finding.details or f"Нужно уточнить условие: {field}.",
                )
            )
        return issues

    def _provider_limit(self) -> int:
        return self._runtime_options()["provider_k"]

    def _without_unset_preferences(self, request: StructuredRequest, message: str, state: ConstraintState) -> StructuredRequest:
        """An explicit opt-out is a no-op when there is nothing to clear.

        Keep clears for active or pending constraints, so a later correction still
        removes the previous preference. A reset makes old fields inactive.
        Other clear operations remain subject to the ordinary validator.
        """
        active_fields = set() if request.reset_constraints else {constraint.field for constraint in state.constraints}
        if state.pending is not None and not request.reset_constraints:
            active_fields.update(change.constraint.field for change in state.pending.proposal.changes if change.constraint is not None)
            active_fields.update(finding.field for finding in state.pending.findings if finding.field is not None)
        updates = []
        for update in request.updates:
            no_preference = self.request_adapter._explicit_no_preference(update.field, update.source_text)
            if not (
                update.field not in active_fields
                and update.operation in {"set", "clear"}
                and no_preference
                and self.request_adapter._source_is_cited(update.source_text, message)
            ):
                updates.append(update)
        return request.model_copy(update={"updates": updates})

    def _with_soft_descriptions(self, request: StructuredRequest, message: str):
        """Keep unsupported descriptions as non-binding ranking hypotheses."""
        updates = []
        soft = []
        for update in request.updates:
            preference = self.request_adapter.soft_preference(update, message)
            if preference is None:
                updates.append(update)
            else:
                soft.append(preference)
        issues = []
        for issue in request.issues:
            preference = self.request_adapter.soft_issue(issue, message)
            if preference is None:
                issues.append(issue)
            else:
                soft.append(preference)
        return request.model_copy(update={
            "updates": updates,
            "issues": issues,
            "clarification_required": request.clarification_required and bool(issues),
        }), soft

    def _decline_pending_preference(self, session):
        """Accept independently verified staged conditions when a user opts out.

        An uncertain optional facet must not make a previously understood
        format disappear. Revalidate all surviving staged changes before any
        commit; global errors and required fields stay blocking.
        """
        pending = session.constraint_state.pending
        if pending is None or pending.base_version != session.constraint_state.version:
            return None
        blockers = [finding for finding in pending.findings if finding.status != "pass"]
        optional = set(self.request_adapter.no_preference_markers) - {self.request_adapter.domain_field, "seed_title"}
        allowed = {"adapter_unsupported_constraint", "adapter_ambiguity", "interpretation_issue", "issue_missing_evidence"}
        if not blockers or any(finding.field not in optional or finding.code not in allowed for finding in blockers):
            return None
        blocked_fields = {finding.field for finding in blockers}
        safe_changes = tuple(
            change
            for change in pending.proposal.changes
            if change.constraint is not None
            and change.constraint.field not in blocked_fields
            and ":implied:" not in change.id
            and change.source_spans
        )
        if not safe_changes:
            if session.query.kind is None and not session.query.seed_title:
                return None
            session.constraint_state = session.constraint_state.model_copy(update={"pending": None})
            session.skipped_slots.update(blocked_fields)
            session.soft_preferences = [
                hint for hint in session.soft_preferences
                if hint["field"] not in blocked_fields and hint.get("source_field") not in blocked_fields
            ]
            session.unresolved.clear()
            return session.query
        evidence = " ".join(span.text for change in safe_changes for span in change.source_spans)
        for change in safe_changes:
            constraint = change.constraint
            operation = "exclude" if constraint.op == "neq" else "set"
            checked = self.request_adapter.normalize(
                StructuredRequest(
                    updates=[
                        ConstraintUpdate(
                            field=constraint.field,
                            operation=operation,
                            value=constraint.value,
                            source_text=change.source_spans[0].text,
                        )
                    ]
                ),
                evidence,
            )
            if not checked.updates or checked.issues:
                return None
        proposal = pending.proposal.model_copy(update={"changes": safe_changes, "issues": ()})
        base = session.constraint_state.model_copy(update={"pending": None})
        validation = validate_proposal(
            proposal,
            message=evidence,
            known_fields=set(self.request_adapter.model.model_fields),
            state=base,
            reference_fields={"navigation": "seed_title", "similar": "seed_title"},
        )
        if validation.status != "PASS":
            return None
        session.constraint_state = reduce_state(base, validation)
        session.skipped_slots.update(blocked_fields)
        session.soft_preferences = [
            hint for hint in session.soft_preferences
            if hint["field"] not in blocked_fields and hint.get("source_field") not in blocked_fields
        ]
        session.unresolved.clear()
        return self._project_query(session.constraint_state.constraints).model_copy(update={"intent": session.constraint_state.intent})

    def _accept_exact_format_answer(self, session, session_id: str, message: str, pending_question: str | None):
        """Validate an exact option selected from the format question."""
        if session.constraint_state.pending is not None or session.unresolved:
            return None
        update = self.request_adapter.exact_domain_answer(message, pending_question)
        if update is None:
            return None
        normalized = self.request_adapter.normalize(StructuredRequest(updates=[update]), message)
        if normalized.issues or len(normalized.updates) != 1:
            return None
        proposal = normalize_proposal(
            normalized,
            turn_id=f"{session_id}:format-choice:{session.constraint_state.version + 1}",
            message=message,
            constraint_operators=self.request_adapter.constraint_operators,
        )
        validation = validate_proposal(
            proposal,
            message=message,
            known_fields=set(self.request_adapter.model.model_fields),
            coverage_surfaces=self._coverage_surfaces(),
            state=session.constraint_state,
            reference_fields={"navigation": "seed_title", "similar": "seed_title"},
        )
        if validation.status != "PASS":
            return None
        session.constraint_state = reduce_state(session.constraint_state, validation)
        session.unresolved.clear()
        if session.query.kind is not None:
            session.skipped_slots.clear()
        return self._project_query(session.constraint_state.constraints).model_copy(update={"intent": session.constraint_state.intent})

    def _extract(self, message, session):
        previous = self._interpretation_context(session.constraint_state)
        context = {
            "pending_question": session.pending_question,
            "unresolved": [{"field": field} for field in sorted({entry.field for entry in session.unresolved})],
            "pending_context": self._pending_context(session.constraint_state),
        }
        interpreted, tokens = self.interpreter.interpret(message, previous, **context)
        opt_out_fields = self._explicit_opt_out_fields(interpreted, message)
        structured = self._without_unset_preferences(interpreted, message, session.constraint_state)
        # Verify a literal, declared format before deciding whether the model
        # omitted a field. Otherwise the completion call can invent extra
        # format updates and block an otherwise valid request atomically.
        explicit_domain = self.request_adapter.recover_explicit_domain(structured, message)
        if explicit_domain is not None:
            structured = structured.model_copy(update={"updates": [explicit_domain, *structured.updates]})
        excluded = {(constraint.field, constraint.value) for constraint in session.constraint_state.constraints if constraint.op == "neq"}
        retained = {(constraint.field, constraint.value) for constraint in session.constraint_state.constraints if constraint.op == "eq"}
        if self.request_adapter.context_verifier is None:
            structured = self.request_adapter.reconcile_operation_evidence(structured, message, excluded=excluded, retained=retained)
            structured = self.request_adapter.detect_polarity_conflicts(structured, message)
            structured = self._discard_answered_conflict(structured, message, session.constraint_state)
        skipped_unset_preference = len(structured.updates) < len(interpreted.updates)
        skipped_fields = {update.field for update in interpreted.updates if update not in structured.updates}
        usage = dict(getattr(self.llm, "last_usage", {}))
        tokens = max(tokens, int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
        repair = getattr(self.interpreter, "repair", None)
        # The retry is driven by a failed transport/coverage check, never by
        # evaluation labels. Both attempts pass the same full validation below.
        proposal = normalize_proposal(
            structured, turn_id="extraction-check", message=message, constraint_operators=self.request_adapter.constraint_operators
        )
        accepted = self.request_adapter.normalize(structured, message)
        implied = self._declared_relations(
            normalize_proposal(
                accepted,
                turn_id="extraction-check",
                message=message,
                constraint_operators=self.request_adapter.constraint_operators,
            )
        )
        proposal = proposal.model_copy(update={"changes": (*proposal.changes, *(c for c in implied.changes if ":implied:" in c.id))})
        check = validate_proposal(
            proposal,
            message=message,
            coverage_surfaces=self._coverage_surfaces(),
            coverage_exempt_spans=(u.source_text for u in structured.updates if u.field == "seed_title"),
        )
        if (
            skipped_unset_preference
            and not structured.updates
            and not structured.issues
            and not structured.clarification_required
            and not structured.reset_constraints
            and structured.intent in {None, session.query.intent}
            and session.query.kind is not None
            and session.constraint_state.pending is None
            and not session.unresolved
            and all(f.code == "empty_patch" or (f.code == "coverage_gap" and f.field in skipped_fields) for f in check.findings)
        ):
            usage["noop_preference"] = 1.0
            usage["noop_preference_fields"] = sorted(skipped_fields)
            return structured, tokens, usage, opt_out_fields
        if not callable(repair) or session.calls >= self.max_calls or session.tokens + tokens >= self.max_tokens:
            return structured, tokens, usage, opt_out_fields
        retry_codes = {
            "coverage_gap",
            "evidence_not_in_turn",
            "missing_evidence",
            "empty_patch",
            "issue_missing_evidence",
            "issue_evidence_not_in_turn",
        }
        reference_base = self._proposal_base(implied, session.constraint_state)
        reference_check = validate_proposal(
            self._merge_pending(implied, reference_base, reset=False),
            state=reference_base,
            reference_fields={"navigation": "seed_title", "similar": "seed_title"},
        )
        feedback = [
            {"code": f.code, "field": f.field, "detail": f.details}
            for f in check.findings
            if f.code in retry_codes and not (f.code == "coverage_gap" and f.field in skipped_fields)
        ]
        feedback.extend({"code": f.code, "field": f.field} for f in reference_check.findings if f.code == "intent_reference_missing")
        if not feedback:
            return structured, tokens, usage, opt_out_fields
        completion = getattr(self.interpreter, "complete_missing", None)
        focused_completion = False
        if (
            callable(completion)
            and not structured.issues
            and not structured.clarification_required
            and not accepted.issues
            and not accepted.clarification_required
            and all(f.code == "coverage_gap" for f in (*check.findings, *reference_check.findings))
        ):
            repair = completion
            focused_completion = True
        session.calls += 1
        try:
            repaired, repair_tokens = repair(message, previous, feedback=feedback, attempt=structured, **context)
        except Exception:
            # Preserve the rejected original for honest clarification; a failed
            # repair must not turn it into a guessed rules interpretation.
            repair_usage = dict(getattr(self.llm, "last_usage", {}))
            return (
                structured,
                tokens + int(repair_usage.get("input_tokens", 0) + repair_usage.get("output_tokens", 0)),
                self._merged_usage(usage, repair_usage),
                opt_out_fields,
            )
        repair_usage = dict(getattr(self.llm, "last_usage", {}))
        repair_tokens = max(repair_tokens, int(repair_usage.get("input_tokens", 0) + repair_usage.get("output_tokens", 0)))
        repaired = self._without_unset_preferences(repaired, message, session.constraint_state)
        explicit_domain = self.request_adapter.recover_explicit_domain(repaired, message)
        if explicit_domain is not None:
            repaired = repaired.model_copy(update={"updates": [explicit_domain, *repaired.updates]})
        if self.request_adapter.context_verifier is None:
            repaired = self.request_adapter.reconcile_operation_evidence(repaired, message, excluded=excluded, retained=retained)
            repaired = self.request_adapter.detect_polarity_conflicts(repaired, message)
            repaired = self._discard_answered_conflict(repaired, message, session.constraint_state)
        # A completion may mix a valid missing field with an unrelated update.
        # Retain only individually supported additions; the complete proposal
        # still passes the ordinary atomic validation before any state change.
        supported_updates = []
        rejected_updates = []
        for update in repaired.updates:
            checked = self.request_adapter.normalize(StructuredRequest(updates=[update]), message)
            if checked.updates and not checked.issues:
                supported_updates.append(update)
            else:
                rejected_updates.append(update)
        if focused_completion and rejected_updates:
            # A focused completion may add only fields absent from a clean
            # first attempt. Its unverifiable additions are not user evidence.
            # The final coverage/atomic checks still reject a genuinely
            # missing field; never apply this rule to a general repair.
            repaired = repaired.model_copy(update={"updates": supported_updates})
            repair_usage["completion_rejected_updates"] = float(len(rejected_updates))
            rejected_updates = []
        # Discard only a fabricated clear that cites another declared field,
        # when the same field also has a supported update. Other rejected
        # values retain their diagnostic and block the complete proposal.
        supported_fields = {update.field for update in supported_updates}
        if rejected_updates and all(
            update.operation == "clear"
            and update.field in supported_fields
            and (cues := self.request_adapter._clear_fields(update.source_text))
            and update.field not in cues
            for update in rejected_updates
        ):
            repaired = repaired.model_copy(update={"updates": supported_updates})
            repair_usage["repair_rejected_updates"] = float(len(rejected_updates))
        # Repair may add missing evidence, but must not make a clean extraction
        # worse or silently remove an already accepted update.
        if self._repair_regresses(structured, repaired):
            merged = self._merged_usage(usage, repair_usage)
            # llm_usage is an aggregate numeric map kept backwards compatible;
            # the human-readable reason is exposed separately in telemetry.
            merged["retry_rejected"] = 1.0
            return structured, tokens + repair_tokens, merged, opt_out_fields
        return repaired, tokens + repair_tokens, self._merged_usage(usage, repair_usage), opt_out_fields

    def _explicit_opt_out_fields(self, request: StructuredRequest, message: str) -> set[str]:
        """Remember cited opt-outs even when their no-op updates are dropped."""
        return {
            update.field
            for update in request.updates
            if update.field in self.request_adapter.no_preference_markers
            and update.field != self.request_adapter.domain_field
            and update.operation in {"set", "clear"}
            and self.request_adapter._source_is_cited(update.source_text, message)
            and self.request_adapter._explicit_no_preference(update.field, update.source_text)
        }

    def _reset_command_opt_outs(self, message: str) -> set[str] | None:
        """Accept an explicit reset command with only declared opt-outs."""
        text = normalize(message).strip(" .!?")
        for command in ("сброс", "заново", "начать заново"):
            if text == command:
                return set()
            if not text.startswith(command) or text[len(command) : len(command) + 1] not in {",", ";"}:
                continue
            rest = text[len(command) + 1 :].strip()
            clauses = [clause.strip() for clause in re.split(r"[,;]|\s+и\s+", rest) if clause.strip()]
            if not clauses:
                return None
            fields = set()
            for clause in clauses:
                matches = [
                    field
                    for field in self.request_adapter.no_preference_markers
                    if field != self.request_adapter.domain_field
                    and self.request_adapter._explicit_no_preference(field, clause)
                ]
                if len(matches) != 1:
                    return None
                fields.add(matches[0])
            return fields
        return None

    @staticmethod
    def _repair_regresses(initial, repaired) -> bool:
        if initial.issues == [] and not initial.clarification_required and (repaired.issues or repaired.clarification_required):
            return True
        # Evidence wording may be improved by the repair; preserve semantic
        # updates by field/operation/value and let the normalizer re-check spans.
        initial_updates = {(u.field, u.operation, str(u.value)) for u in initial.updates}
        repaired_updates = {(u.field, u.operation, str(u.value)) for u in repaired.updates}
        if not initial_updates.issubset(repaired_updates):
            return True
        return (
            initial.intent in {"discovery", "mood"}
            and repaired.intent in {"navigation", "similar"}
            and not any(u.field == "seed_title" for u in repaired.updates)
        )

    def _proposal_base(self, accepted, state):
        """Use the same domain reset for reference precheck and final commit."""
        if self._domain_changed(accepted, state):
            return ConstraintState(version=state.version)
        if state.pending is not None:
            domain_blocked = any(f.field == self.request_adapter.domain_field for f in state.pending.findings)
            staged = self._merge_pending(accepted, state, reset=False)
            if not domain_blocked and self._domain_changed(staged, state):
                return state.model_copy(update={"constraints": (), "intent": "discovery"})
        return state

    @staticmethod
    def _canonical_constraints(query: Query, turn_id: str) -> tuple[Constraint, ...]:
        fields = (
            ("kind", "eq", query.kind),
            ("genre", "eq", query.genre),
            ("tone", "eq", query.tone),
            ("max_seasons", "lte", query.max_seasons),
            ("max_minutes", "lte", query.max_minutes),
            ("level", "eq", query.level),
            ("practical", "eq", query.practical),
        )
        constraints = [
            Constraint(id=f"{turn_id}:{field}:{op}", field=field, op=op, value=value, turn_id=turn_id, domain_version="demo-v1")
            for field, op, value in fields
            if value is not None
        ]
        constraints.extend(
            Constraint(id=f"{turn_id}:genre:neq:{genre}", field="genre", op="neq", value=genre, turn_id=turn_id, domain_version="demo-v1")
            for genre in query.excluded_genres
        )
        return tuple(constraints)

    def _retrieve(self, state):
        provider = self.provider
        options = self._runtime_options()
        if state["query"].intent in {"navigation", "similar"} and (
            not getattr(provider, "title_lookup_verified", False) or not callable(getattr(provider, "find_title_verified", None))
        ):
            return {
                "issue": "Источник не подтверждает уникальность названия. Уточните запрос или используйте проверенный каталог.",
                "trace": [*state["trace"], "title_lookup_unverified"],
            }
        # Refresh the provider's Query from canonical state so corrections
        # reach retrieval with their current values.
        canonical_query = self._project_query(state["session"].constraint_state.constraints)
        # Keep the current compatibility projection when the canonical state
        # is empty (for example in a rules-baseline navigation test).
        if not state["session"].constraint_state.constraints:
            canonical_query = state["query"]
        canonical_query = canonical_query.model_copy(update={"intent": state["query"].intent})
        retrieve_state = state | {"query": canonical_query}
        result = super()._retrieve(retrieve_state)
        canonical_query = result.get("query", canonical_query)
        result["query"] = canonical_query
        items = getattr(provider, "items", None)
        if result.get("degradation") == "UNAVAILABLE":
            return result
        if result.get("issue") and not result.get("candidates"):
            return result
        # The legacy adaptive question chooser runs inside Agent._retrieve.
        # Workflow-v2 accepts a complete validated proposal first; an optional
        # question must never skip compiler, hard filter or a grounded answer.
        result.pop("issue", None)
        result.pop("clarification_slot", None)
        result.pop("question_gain", None)
        # A production adapter has no local catalog snapshot.  Its canonical
        # lookup records still pass through the same compiler/filter path; BM25
        # then operates on the retrieved window only rather than bypassing it.
        source_items = items or {item.id: item for item in result.get("candidates", [])}
        documents = {
            item_id: " ".join(str(value or "") for value in (item.title, item.genre, item.tone, item.description))
            for item_id, item in source_items.items()
        }
        session = state["session"]
        constraints = session.constraint_state.constraints
        if not constraints:
            # Rules are an explicit degraded fallback.  This bridge is the only
            # place where the legacy DTO supplies canonical constraints.
            constraints = self._canonical_constraints(result["query"], state["session_id"])
            session.constraint_state = session.constraint_state.model_copy(update={"constraints": constraints})
        constraints = self._filter_constraints(constraints)
        capabilities = getattr(provider, "capabilities", None)
        if capabilities is None:
            return {**result, "issue": "Адаптер источника не объявил поддерживаемые фильтры."}
        compiled = compile_constraints(constraints, capabilities)
        active_ids = {constraint.id for constraint in constraints if constraint.strength == "hard"}
        covered_ids = set(compiled.pushed_constraint_ids) | {constraint.id for constraint in compiled.residual_constraints}
        if active_ids != covered_ids:
            return {**result, "issue": "Не удалось скомпилировать все обязательные условия запроса."}
        lexical = BM25Index(documents).search(state["request"].message, limit=options["lexical_k"])
        provider_ids = [item.id for item in result.get("candidates", [])][: options["provider_k"]]
        fused_ids = reciprocal_rank_fusion(
            [[*provider_ids], [hit.item_id for hit in lexical]], k=options["rrf_k"], limit=options["fused_k"]
        )
        if result["query"].intent == "navigation" and result.get("seed") is not None:
            fused_ids = [result["seed"].id]
        lookup = {item.id: item for item in provider.lookup(fused_ids)} if items else source_items
        canonical, diagnostics = hard_filter([lookup[item_id] for item_id in fused_ids if item_id in lookup], constraints)
        blocked = {item.id for item in result.get("history", [])}
        blocked |= {item_id for item_id, reaction in session.reactions.items() if reaction == "dislike"}
        seed = result.get("seed")
        if seed is not None and result["query"].intent == "similar":
            blocked.add(seed.id)
        if re_more(state["request"].message):
            blocked |= session.shown
        if seed is not None and result["query"].intent == "navigation":
            blocked.clear()
        # Use the canonical filter result, which also enforces constraints
        # absent from the compatibility Query, such as tone exclusions.
        candidates = [item for item in canonical if item.id not in blocked]
        from .domains.demo import domain_spec

        candidates = soft_rank(
            candidates,
            fields=domain_spec().fields,
            intent=result["query"].intent,
            seed=seed,
            history=result.get("history", []),
            liked_ids={item_id for item_id, reaction in session.reactions.items() if reaction == "like"},
            soft_preferences=[
                (hint["field"], hint["value"])
                for hint in state.get("soft_preferences", [])
                if hint.get("value") is not None
            ],
            quality_field="quality",
        )
        # A bare format request expresses no genre choice. Keep the best item
        # first, then show nearby genres instead of letting one profile/quality
        # cluster fill the whole visible slate.
        broad_discovery = (
            state.get("mode") == "ollama"
            and not session.unresolved
            and result["query"].intent in {"discovery", "mood"}
            and result["query"].kind in {"film", "series"}
            and seed is None
            and not re_more(state["request"].message)
            and not any(c.field != self.request_adapter.domain_field for c in constraints)
            and not state.get("soft_preferences")
            and not any(reaction == "like" for reaction in session.reactions.values())
        )
        if broad_discovery:
            candidates = diversify_head(candidates, field="genre")
        result["candidates"] = candidates
        result["broad_discovery"] = broad_discovery
        result["filter_diagnostics"] = diagnostics
        result["trace"] = [*result.get("trace", []), "query_compiler", "canonical_lookup", "bm25", "rrf", "tri_state_hard_filter"]
        # The model may omit a stated opt-out such as "жанр любой". For an
        # unset optional field only, use the adapter's declared markers to
        # avoid asking a question the user has already answered. This never
        # clears an active constraint or bypasses proposal validation.
        if not session.unresolved and session.constraint_state.pending is None:
            active_fields = {constraint.field for constraint in session.constraint_state.constraints}
            clauses = re.split(r"[,;.!?]|\s+и\s+", state["request"].message, flags=re.IGNORECASE)
            session.skipped_slots.update(
                field
                for field in self.request_adapter.no_preference_markers
                if field not in active_fields
                and field != self.request_adapter.domain_field
                and any(self.request_adapter._explicit_no_preference(field, clause) for clause in clauses)
            )
        # Ask only when the user has supplied no distinguishing preference.
        # Specific requests can be answered immediately; missing optional
        # attributes are not extraction failures. Entropy is computed AFTER
        # canonical filtering, excluding fields the user already constrained.
        has_preference = any(c.field != self.request_adapter.domain_field for c in constraints) or bool(state.get("soft_preferences"))
        if (
            self.question_policy in {"adaptive", "fixed"}
            and not has_preference
            and seed is None
            and result["query"].intent != "navigation"
            and not re_more(state["request"].message)
            and not session.skipped_slots
            and session.question_streak < self.max_questions
        ):
            question = choose_question(
                result["query"], candidates, policy=self.question_policy, skipped=session.skipped_slots, cost=self.question_cost
            )
            if question:
                result.update(issue=question.message, clarification_slot=question.slot, question_gain=question.gain)
        return result

    def _parse(self, state):
        """Interpret the turn and commit only a fully validated proposal."""
        request, session = state["request"], state["session"]
        if not hasattr(session, "constraint_state"):
            session.constraint_state = ConstraintState()
        if session.restart_on_next_turn:
            session.query = Query()
            session.constraint_state = ConstraintState()
            session.clarifications = 0
            session.question_streak = 0
            session.last_ids.clear()
            session.skipped_slots.clear()
            session.unresolved.clear()
            session.pending_question = None
            session.pending_slot = None
            session.pending_slots.clear()
            session.preferences = PreferenceModel()
            session.soft_preferences.clear()
            session.restart_on_next_turn = False
        text, previous = normalize(request.message), session.query
        pending_slot = session.pending_slot
        pending_question = session.pending_question
        declines_preference = _declines_optional_preference(text)
        if declines_preference and session.constraint_state.pending is not None:
            accepted = self._decline_pending_preference(session)
            if accepted is not None:
                session.query = accepted
                session.pending_question = None
                session.pending_slot = None
                session.pending_slots.clear()
                return {
                    "query": accepted,
                    "issue": None,
                    "trace": ["parse", "decline_pending_preference"],
                    "mode": self.mode,
                    "validation_status": "PASS",
                    "llm_usage": {},
                    "soft_preferences": session.soft_preferences,
                }
        skip_optional = (
            declines_preference
            and (session.pending_slot or session.pending_slots)
            and session.constraint_state.pending is None
            and not session.unresolved
        )
        if session.pending_slots and declines_preference:
            session.skipped_slots.update(session.pending_slots)
        session.pending_slots.clear()
        if session.pending_slot and declines_preference:
            session.skipped_slots.add(session.pending_slot)
        session.pending_slot = None
        if skip_optional:
            session.pending_question = None
            return {
                "query": previous,
                "issue": None,
                "trace": ["parse", "skip_optional_question"],
                "mode": self.mode,
                "validation_status": "PASS",
                "llm_usage": {},
                "soft_preferences": session.soft_preferences,
            }
        if pending_slot == self.request_adapter.domain_field:
            accepted = self._accept_exact_format_answer(session, state["session_id"], request.message, pending_question)
            if accepted is not None:
                session.query = accepted
                session.pending_question = None
                return {
                    "query": accepted,
                    "issue": None,
                    "trace": ["parse", "verified_format_choice"],
                    "mode": self.mode,
                    "validation_status": "PASS",
                    "llm_usage": {},
                    "soft_preferences": session.soft_preferences,
                }
        reset_opt_outs = self._reset_command_opt_outs(request.message)
        if reset_opt_outs is not None:
            session.query, session.constraint_state = Query(), ConstraintState()
            session.clarifications = 0
            session.question_streak = 0
            session.shown.clear()
            session.last_ids.clear()
            session.skipped_slots.clear()
            session.skipped_slots.update(reset_opt_outs)
            session.unresolved.clear()
            session.pending_question = None
            session.preferences = PreferenceModel()
            session.soft_preferences.clear()
            return {
                "query": session.query,
                "issue": "Что подбираем: фильм, сериал или курс?",
                "clarification_slot": self.request_adapter.domain_field,
                "trace": ["parse", "shape_validation", "state_reducer", "reset"],
                "mode": self.mode,
            }

        query, issue, structured = previous, None, None
        soft_preferences = []
        warnings, tokens, mode, usage = [], 0, self.mode, {}
        fallback_reason = None
        validation_status = "PASS"
        validation_slot = None
        unsupported_catalog_request = False
        catalog_gaps = []
        opt_out_fields: set[str] = set()
        # Paging is an explicit system action, not an empty semantic update.
        # A message containing new conditions still follows interpretation.
        if (
            is_exact_more_command(text)
            and session.last_ids
            and session.constraint_state.constraints
            and session.constraint_state.pending is None
            and not session.unresolved
            and previous.intent in {"discovery", "mood", "similar"}
        ):
            return {
                "query": previous,
                "issue": None,
                "trace": ["parse", "continue_recommendations"],
                "mode": mode,
                "validation_status": "PASS",
                "llm_usage": {},
                "soft_preferences": session.soft_preferences,
            }
        if self.mode == "ollama":
            if session.calls >= self.max_calls or session.tokens >= self.max_tokens:
                warnings.append("Бюджет LLM исчерпан: включён разбор по правилам.")
                mode = "rules_fallback"
                fallback_reason = "llm_budget_exhausted"
            else:
                session.calls += 1
                try:
                    structured, tokens, usage, opt_out_fields = self._extract(request.message, session)
                    explicit_domain = self.request_adapter.recover_explicit_domain(structured, request.message)
                    if explicit_domain is not None:
                        structured = structured.model_copy(update={"updates": [explicit_domain, *structured.updates]})
                    structured, soft_preferences = self._with_soft_descriptions(structured, request.message)
                    if (
                        soft_preferences
                        and not structured.updates
                        and not structured.issues
                        and not structured.clarification_required
                        and not structured.reset_constraints
                        and structured.intent in {None, previous.intent}
                        and previous.kind is None
                        and session.constraint_state.pending is None
                        and not session.unresolved
                    ):
                        session.soft_preferences = soft_preferences
                        question = self.request_adapter.domain_question(request.message)
                        return {
                            "query": previous,
                            "issue": f"Пожелание учёл. {question}",
                            "clarification_slot": self.request_adapter.domain_field,
                            "tokens": tokens,
                            "mode": mode,
                            "trace": ["parse", "soft_description", "ask_format"],
                            "validation_status": "PASS",
                            "llm_usage": usage,
                            "soft_preferences": session.soft_preferences,
                        }
                    if (
                        soft_preferences
                        and not structured.updates
                        and not structured.issues
                        and not structured.clarification_required
                        and not structured.reset_constraints
                        and structured.intent in {None, previous.intent}
                        and previous.kind is not None
                        and session.constraint_state.pending is None
                        and not session.unresolved
                    ):
                        fields = {field for hint in soft_preferences for field in (hint["field"], hint.get("source_field"))}
                        session.soft_preferences = [
                            hint for hint in session.soft_preferences
                            if hint["field"] not in fields and hint.get("source_field") not in fields
                        ] + soft_preferences
                        session.pending_question = None
                        return {
                            "query": previous,
                            "issue": None,
                            "tokens": tokens,
                            "mode": mode,
                            "trace": ["parse", "soft_description"],
                            "validation_status": "PASS",
                            "llm_usage": usage,
                            "soft_preferences": session.soft_preferences,
                        }
                    if usage.get("noop_preference"):
                        cleared_fields = set(usage.pop("noop_preference_fields", []))
                        session.skipped_slots.update(opt_out_fields)
                        session.soft_preferences = [
                            hint for hint in session.soft_preferences
                            if hint["field"] not in cleared_fields and hint.get("source_field") not in cleared_fields
                        ]
                        session.pending_question = None
                        return {
                            "query": previous,
                            "issue": None,
                            "tokens": tokens,
                            "mode": mode,
                            "trace": ["parse", "verified_no_preference"],
                            "validation_status": "PASS",
                            "llm_usage": usage,
                            "soft_preferences": session.soft_preferences,
                        }
                    turn_id = f"{state['session_id']}:{session.calls}"
                    normalized = self.request_adapter.normalize(structured, request.message)
                    # Preserve rejected proposals as rejected evidence. Only the
                    # adapter's accepted updates can supply normalized values.
                    canonical = []
                    rejected_change_ids: dict[str, list[str]] = {}
                    for update_index, update in enumerate(structured.updates):
                        single = self.request_adapter.normalize(
                            structured.model_copy(update={"updates": [update], "issues": []}),
                            request.message,
                        )
                        canonical.append(single.updates[0] if single.updates else update)
                        if not single.updates or single.issues:
                            rejected_change_ids.setdefault(update.field, []).append(f"{turn_id}:{update_index}")
                    current = structured.model_copy(
                        update={
                            "updates": canonical,
                            "issues": normalized.issues,
                        }
                    )
                    fresh = normalize_proposal(
                        current,
                        turn_id=turn_id,
                        message=request.message,
                        constraint_operators=self.request_adapter.constraint_operators,
                    )
                    accepted = normalize_proposal(
                        normalized,
                        turn_id=turn_id,
                        message=request.message,
                        constraint_operators=self.request_adapter.constraint_operators,
                    )
                    accepted = self._declared_relations(accepted)
                    implied = tuple(c for c in accepted.changes if ":implied:" in c.id)
                    fresh = fresh.model_copy(update={"changes": (*fresh.changes, *implied)})
                    base = self._proposal_base(accepted, session.constraint_state)
                    proposal = self._merge_pending(fresh, base, reset=False)
                    validation = validate_proposal(
                        proposal,
                        message=request.message,
                        known_fields=set(type(previous).model_fields),
                        coverage_surfaces=self._coverage_surfaces(),
                        state=base,
                        coverage_exempt_spans=(u.source_text for u in normalized.updates if u.field == "seed_title"),
                        reference_fields={"navigation": "seed_title", "similar": "seed_title"},
                    )
                    # The DTO is a compatibility view. Apply against an empty
                    # view only to check shape/ranges, not to reconstruct state
                    # or re-cite evidence staged on a previous turn.
                    _, adapter_issues = self.request_adapter.apply(normalized, Query(), request.message)
                    findings = [*validation.findings]
                    findings.extend(
                        ValidationFinding(
                            code="adapter_" + entry.kind,
                            status="uncertain",
                            field=entry.field,
                            details=entry.message,
                            change_ids=tuple(rejected_change_ids.get(entry.field, ())),
                        )
                        for entry in adapter_issues
                    )
                    adapter_issue_fields = {issue.field for issue in adapter_issues}
                    resolved_fields = {u.field for u in normalized.updates} - adapter_issue_fields
                    resolved_assertions = {
                        u.field for u in normalized.updates if u.operation in {"set", "clear"}
                    } - adapter_issue_fields
                    resolved_conflict_values = {
                        (u.field, str(u.value))
                        for u in normalized.updates
                        if u.operation in {"include", "exclude"} and u.field not in adapter_issue_fields
                    }
                    pending = base.pending
                    if pending is not None:
                        prior_wishes = self._unsupported_wishes(pending.findings, "", prior_findings=pending.findings)
                        repeated_prior = self._collapsed_unsupported(prior_wishes)
                        for finding in pending.findings:
                            if finding.code in {"empty_patch", "intent_reference_missing", "constraint_conflict"}:
                                continue  # These are recomputed against the merged proposal.
                            prior_conflict = False
                            prior_conflict_value = None
                            if finding.code == "interpretation_issue":
                                try:
                                    prior_issue = json.loads(finding.details)
                                    prior_conflict = prior_issue.get("kind") == "conflict"
                                    prior_conflict_value = str(prior_issue.get("value") or "")
                                    if prior_conflict and not prior_conflict_value:
                                        # A model may emit an empty-value conflict while
                                        # the verifier also records the exact conflicting
                                        # value. Both refer to the same pending question.
                                        known_values = set()
                                        for other in pending.findings:
                                            if other.code != "interpretation_issue" or other.field != finding.field:
                                                continue
                                            try:
                                                other_issue = json.loads(other.details)
                                            except (TypeError, ValueError):
                                                continue
                                            if other_issue.get("kind") == "conflict" and other_issue.get("value"):
                                                known_values.add(str(other_issue["value"]))
                                        if len(known_values) == 1:
                                            prior_conflict_value = next(iter(known_values))
                                except (ValueError, AttributeError):
                                    prior_conflict = False
                            # An extra exclusion does not answer an unknown positive
                            # preference; an explicit value/clear can resolve it.
                            resolved = (
                                resolved_assertions
                                | (
                                    {finding.field}
                                    if prior_conflict and (finding.field, prior_conflict_value) in resolved_conflict_values
                                    else set()
                                )
                                if not finding.change_ids
                                and (
                                    finding.code == "interpretation_issue"
                                    or (finding.code.startswith("adapter_") and finding.code != "adapter_conflict")
                                )
                                else resolved_fields
                            )
                            if finding.field not in resolved or (repeated_prior and finding.field in {f for f, _ in prior_wishes}):
                                findings.append(finding)
                    status = (
                        "REJECT"
                        if any(f.status == "reject" for f in findings)
                        else "UNCERTAIN"
                        if any(f.status == "uncertain" for f in findings)
                        else "PASS"
                    )
                    validation = validation.model_copy(update={"status": status, "findings": tuple(findings)})
                    validation_status = status
                    if status != "PASS":
                        # An empty extraction contains nothing to keep pending.
                        # Retain an earlier pending request, if any, so the next
                        # turn can still answer its concrete question.
                        if proposal.changes or proposal.issues:
                            session.constraint_state = reduce_state(session.constraint_state, validation)
                            issues = self._findings_as_issues(validation.findings)
                            session.unresolved = list({(i.field, i.value): i for i in issues}.values())
                        elif pending is None:
                            session.unresolved = []
                        prior_findings = pending.findings if pending is not None else ()
                        issue = self._clarification_message(
                            validation.findings, request.message, prior_findings=prior_findings, has_domain=previous.kind is not None
                        )
                        validation_slot = next(
                            (finding.field for finding in validation.findings if finding.status != "pass" and finding.field in self.request_adapter.field_questions),
                            None,
                        )
                        if validation_slot is None and previous.kind is None and issue == self.request_adapter.domain_question(request.message):
                            validation_slot = self.request_adapter.domain_field
                        unsupported = self._unsupported_wishes(
                            validation.findings, request.message, prior_findings=prior_findings
                        )
                        # Persist only wishes cited in this turn. Pending wishes
                        # remain visible to the user but must not inflate demand.
                        catalog_gaps = [
                            {"reason": "unsupported_catalog_constraint", "field_hint": field, "wish": source}
                            for field, source in unsupported
                            if self.request_adapter._source_is_cited(source, request.message)
                        ]
                        if self._collapsed_unsupported(unsupported):
                            # Several unknown wishes collapsed into one enum field
                            # cannot be independently amended in pending state.
                            # End this request without committing it. The next
                            # turn starts fresh, so a short answer cannot reuse
                            # the old format and produce a false match.
                            session.constraint_state = session.constraint_state.model_copy(update={"pending": None})
                            session.unresolved.clear()
                            session.restart_on_next_turn = True
                            unsupported_catalog_request = True
                    else:
                        session.unresolved = []
                        session.constraint_state = reduce_state(base, validation)
                        query = self._project_query(session.constraint_state.constraints)
                        query = query.model_copy(update={"intent": session.constraint_state.intent})
                        # An explicit "any" is an answer, not a missing preference.
                        # Remember it so the optional question is not repeated.
                        session.skipped_slots.update(
                            update.field
                            for update in normalized.updates
                            if update.operation == "clear"
                            and update.field in self.request_adapter.no_preference_markers
                            and update.field != self.request_adapter.domain_field
                        )
                        session.skipped_slots.update(opt_out_fields)
                        if previous.kind is not None and query.kind != previous.kind:
                            session.soft_preferences.clear()
                        replaced_fields = {update.field for update in normalized.updates}
                        replaced_fields.update(
                            field for hint in soft_preferences for field in (hint["field"], hint.get("source_field"))
                        )
                        session.soft_preferences = [
                            hint for hint in session.soft_preferences
                            if hint["field"] not in replaced_fields and hint.get("source_field") not in replaced_fields
                        ] + soft_preferences
                        soft_preferences = session.soft_preferences
                except Exception as exc:
                    warnings.append(f"LLM недоступна или вернула неверную структуру ({type(exc).__name__}): разбор по правилам.")
                    mode = "rules_fallback"
                    fallback_reason = f"llm_failure:{type(exc).__name__}"
                usage = usage or getattr(self.llm, "last_usage", {})
                tokens = max(tokens, int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
        if self.mode == "rules" or mode == "rules_fallback":
            query, issue = rule_parse(request.message, previous)
            session.unresolved = [
                entry
                for entry in session.unresolved
                if not hasattr(query, entry.field) or getattr(query, entry.field) == getattr(previous, entry.field)
            ]
            if session.unresolved:
                pending = session.constraint_state.pending
                issue = (
                    self._clarification_message(
                        pending.findings, request.message, prior_findings=pending.findings, has_domain=previous.kind is not None
                    )
                    if pending is not None
                    else self.request_adapter.field_questions.get(session.unresolved[0].field)
                    or "Не понял новое пожелание. Напишите его другими словами."
                )
            if not issue:
                # Only the explicitly degraded path projects a rules Query
                # back into state. Preserve canonical-only exclusions unless
                # the parser has accepted a change of domain.
                residual = tuple(c for c in session.constraint_state.constraints if c.op == "neq" and c.field != "genre")
                if query.kind != previous.kind:
                    residual = ()
                session.constraint_state = ConstraintState(
                    intent=query.intent,
                    constraints=(*self._canonical_constraints(query, state["session_id"]), *residual),
                    version=session.constraint_state.version + 1,
                )

        if not issue and query.kind != previous.kind:
            if previous.kind is not None:
                session.skipped_slots.clear()
            session.skipped_slots.update(opt_out_fields)
        session.pending_question = None
        session.query = query
        session.preferences.update(query, previous, request.message, mode)
        if not issue and not query.kind and not query.seed_title:
            issue = "Что подбираем: фильм, сериал или курс?"
        elif (
            self.question_policy == "legacy"
            and not issue
            and not (
                query.genre
                or query.tone
                or query.seed_title
                or query.level
                or any(constraint.field in {"genre", "tone", "seed_title", "level"} for constraint in session.constraint_state.constraints)
            )
            and session.clarifications < 2
        ):
            issue = "Какой жанр или настроение вам ближе? Для курса можно назвать тему или уровень."
        if query.kind != "course" and (query.level or query.practical is not None):
            issue = "Уровень и практика относятся к курсам. Уточните формат или напишите «сброс»."
        slot = session.unresolved[0].field if session.unresolved else validation_slot or ("kind" if issue and "Что подбираем" in issue else None)
        return {
            "query": query,
            "issue": issue,
            "warnings": warnings,
            "tokens": tokens,
            "mode": mode,
            "trace": ["parse", "shape_validation", "proposal_validation", "state_reducer"],
            "clarification_slot": slot,
            "degradation": "NO_LLM" if mode == "rules_fallback" else "FULL",
            "llm_usage": usage,
            "structured_request": structured.model_dump() if structured is not None else None,
            "validation_status": validation_status,
            "fallback_reason": fallback_reason,
            "retry_rejection_reason": ("non_monotonic_structured_patch" if usage.get("retry_rejected") else None),
            "soft_preferences": soft_preferences,
            "unsupported_catalog_request": unsupported_catalog_request,
            "catalog_gaps": catalog_gaps,
        }

    def _recommend(self, state):
        """Render the order produced by filtered, seed-aware retrieval."""
        session, seed, history = state["session"], state.get("seed"), state["history"]
        ranked = state["candidates"][:5]
        recommendations = [
            explain(
                item,
                state["query"],
                history,
                1.0 / (rank + 1),
                seed,
                tone=state["request"].explanation_tone,
                length=state["request"].explanation_length,
            )
            for rank, item in enumerate(ranked)
        ]
        session.last_ids = [item.id for item in ranked]
        session.shown.update(session.last_ids)
        state["trace"] += ["deterministic_rank", "grounding_check"]
        if not ranked:
            return {
                "response": self._response(
                    state,
                    "no_results",
                    "Подходящих вариантов не найдено. Можно снять ограничение или изменить запрос.",
                ),
                "generation_fallback_reason": state.get("generation_fallback_reason"),
            }
        plan_recommendations = recommendations
        if state.get("broad_discovery"):
            # The visible cards may contain a second item of one genre, but a
            # short opening answer should demonstrate the available choices.
            seen_genres = set()
            plan_recommendations = []
            for rec in recommendations:
                if rec.item.genre in seen_genres:
                    continue
                seen_genres.add(rec.item.genre)
                plan_recommendations.append(rec)
        options = [
            GroundedOption(id=rec.item.id, title=rec.item.title, claims=rec.claim_texts or [rec.explanation])
            for rec in plan_recommendations
        ]
        generator = self.response_generator
        if getattr(generator, "requires_llm", False) and (
            session.calls >= self.max_calls or session.tokens + state.get("tokens", 0) >= self.max_tokens
        ):
            generator = EvidenceResponseGenerator()
            state["warnings"].append("Бюджет LLM исчерпан: ответ собран напрямую из проверенного evidence.")
            state["generation_fallback_reason"] = "response_budget_exhausted"
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
            state["tokens"] = state.get("tokens", 0) + max(
                response_tokens,
                int(response_usage.get("input_tokens", 0) + response_usage.get("output_tokens", 0)),
            )
            state["llm_usage"] = self._merged_usage(state.get("llm_usage", {}), response_usage)
        except Exception as exc:
            state["generation_fallback_reason"] = f"response_generation_failure:{type(exc).__name__}"
            response_usage = getattr(getattr(generator, "backend", None), "last_usage", {})
            state["tokens"] = state.get("tokens", 0) + int(response_usage.get("input_tokens", 0) + response_usage.get("output_tokens", 0))
            state["llm_usage"] = self._merged_usage(state.get("llm_usage", {}), response_usage)
            state["warnings"].append(f"Генератор ответа недоступен ({type(exc).__name__}): использован grounded fallback.")
            message, _ = EvidenceResponseGenerator().generate(
                original_request=state["request"].message,
                intent=state["query"].intent,
                accepted_constraints=state["query"].model_dump(mode="json"),
                options=options,
                unresolved=[issue.model_dump() for issue in session.unresolved],
            )
        if state.get("generation_fallback_reason"):
            state["mode"] = "rules_fallback"
            state["degradation"] = "NO_LLM"
        if state.get("soft_preferences"):
            wishes = ", ".join(dict.fromkeys(str(hint["source_text"]) for hint in state["soft_preferences"]))
            message = f"Учёл пожелание «{wishes}» как ориентир. Точное соответствие ему каталог не подтверждает, поэтому подбор приблизительный.\n" + message
        if state.get("broad_discovery"):
            note = (
                "Жанр не ограничен. Показываю варианты разных жанров."
                if len({item.genre for item in ranked}) > 1
                else "Жанр не ограничен; вот доступные варианты."
            )
            if history:
                note += " При выборе первого варианта учитывается история этого профиля."
            message = f"{note}\n{message}"
        return {
            "response": self._response(state, "recommend", message, recommendations),
            "generation_fallback_reason": state.get("generation_fallback_reason"),
        }
