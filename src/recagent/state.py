"""Pure atomic reducer for validated workflow proposals."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import Constraint, ConstraintState, PendingState, ProposedChange, ValidationResult


def merge_pending_changes(pending: Iterable[ProposedChange], fresh: Iterable[ProposedChange]) -> tuple[ProposedChange, ...]:
    """Replace the addressed constraint, not every staged value of its field.

    Scalar corrections supersede the same operator. Negative values accumulate;
    include removes only its exact exclusion, and clear addresses the whole
    field. Retain fresh contradictory updates together for validation.
    """
    fresh = tuple(fresh)

    def superseded(old: ProposedChange) -> bool:
        previous = old.constraint
        if previous is None:
            return False
        for new in fresh:
            current = new.constraint
            if new.operation == "clear" and (
                previous.id in new.target_constraint_ids or (current is not None and current.field == previous.field)
            ):
                return True
            if current is None or current.field != previous.field:
                continue
            if (
                new.operation in {"add", "replace"}
                and old.operation in {"add", "replace"}
                and current.value == previous.value
                and {current.op, previous.op} == {"eq", "neq"}
            ):
                # A later explicit correction supersedes the opposite staged
                # polarity. Contradictions inside the *new* turn remain intact.
                return True
            if current.op != previous.op:
                continue
            if new.operation == "remove" and current.value == previous.value:
                return old.operation in {"add", "replace"}
            if (
                new.operation in {"add", "replace"}
                and old.operation in {"add", "replace"}
                and (current.op != "neq" or current.value == previous.value)
            ):
                return True
        return False

    return (*(old for old in pending if not superseded(old)), *fresh)


def apply_changes(constraints: Iterable[Constraint], changes: Iterable[ProposedChange]) -> tuple[Constraint, ...]:
    """Preview and commit exactly the same constraint transition.

    A new scalar or bound supersedes only the previous value for that field and
    operator.  Other bounds and exclusions survive.  Proposed values are kept
    separate from the previous state so two incompatible updates in one proposal
    stay visible to validation instead of silently making the last update win.
    """
    previous = list(constraints)
    proposed: list[Constraint] = []
    for change in changes:
        constraint = change.constraint
        if change.operation in ("add", "replace") and constraint is not None:
            if constraint.op != "neq":
                previous = [current for current in previous if (current.field, current.op) != (constraint.field, constraint.op)]
            else:
                key = (constraint.field, constraint.op, constraint.value)
                previous = [current for current in previous if (current.field, current.op, current.value) != key]
                proposed = [current for current in proposed if (current.field, current.op, current.value) != key]
            proposed.append(constraint)
        elif change.operation == "remove" and constraint is not None:
            key = (constraint.field, constraint.op, constraint.value)
            previous = [current for current in previous if (current.field, current.op, current.value) != key]
            proposed = [current for current in proposed if (current.field, current.op, current.value) != key]
        elif change.operation == "clear":
            targets = set(change.target_constraint_ids)
            # IDs name individual constraints (for example one range bound).
            # Clearing a whole field is a separate, explicit operation.
            fields = {constraint.field} if constraint is not None else set()
            previous = [current for current in previous if current.id not in targets and current.field not in fields]
            proposed = [current for current in proposed if current.id not in targets and current.field not in fields]
    return (*previous, *proposed)


def reduce_state(state: ConstraintState, result: ValidationResult) -> ConstraintState:
    """Commit a passing proposal; otherwise keep active state and stage it.

    A blocked proposal cannot partially change the retrieval constraints.
    Its pending version identifies the state that a later answer may amend.
    """
    if result.status != "PASS":
        blocker = next((finding for finding in result.findings if finding.status != "pass"), None)
        target_ids = blocker.change_ids if blocker is not None else ()
        target_field = blocker.field if blocker is not None else None
        question_id = f"{result.proposal.raw_request_hash}:{target_field or (target_ids[0] if target_ids else 'proposal')}"
        return state.model_copy(
            update={
                "pending": PendingState(
                    proposal=result.proposal,
                    findings=result.findings,
                    question_id=question_id,
                    target_change_ids=target_ids,
                    target_field=target_field,
                    base_version=state.version,
                )
            }
        )
    return ConstraintState(
        intent=result.proposal.intent or state.intent,
        constraints=apply_changes(state.constraints, result.proposal.changes),
        pending=None,
        version=state.version + 1,
    )
