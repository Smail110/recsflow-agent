"""Conservative, display-only view of fresh independently verified changes."""

from .contracts import ConstraintState, NormalizedProposal
from .models import PendingChangePreview, PendingPreview


def build_pending_preview(
    state: ConstraintState,
    verified: NormalizedProposal,
    *,
    turn_id: str,
    known_fields: set[str],
    domain_field: str | None = None,
    labels: dict[str, str] | None = None,
) -> PendingPreview | None:
    """Show a subset of pending without applying it or implying acceptance.

    `verified` must come from the adapter's ordinary evidence normalization of
    this turn. Pending alone is untrusted. Require agreement with that fresh
    canonical result and scoped validation findings. Historical/implied changes
    are deliberately omitted. A global finding or blocked domain hides all.
    """
    pending = state.pending
    if pending is None or pending.base_version != state.version:
        return None
    blockers = [finding for finding in pending.findings if finding.status != "pass"]
    fields_by_id = {change.id: change.constraint.field for change in pending.proposal.changes if change.constraint is not None}
    # Unknown/global diagnostics cannot establish independence of any subset.
    if not blockers or any(
        (finding.field is None and not finding.change_ids)
        or (finding.field == "request")
        or (finding.field is not None and finding.field not in known_fields)
        or any(change_id not in fields_by_id for change_id in finding.change_ids)
        or (domain_field is not None and finding.field == domain_field)
        or (domain_field is not None and any(fields_by_id.get(change_id) == domain_field for change_id in finding.change_ids))
        for finding in blockers
    ):
        return None

    def signature(change):
        c = change.constraint
        return (change.operation, c.field, c.op, type(c.value), c.value, tuple(span.text for span in change.source_spans)) if c else None

    safe = {signature(change) for change in verified.changes if change.constraint is not None}
    blocked_fields = {finding.field for finding in blockers if finding.field is not None}
    blocked_ids = {change_id for finding in blockers for change_id in finding.change_ids}
    blocked_fields.update(fields_by_id[change_id] for change_id in blocked_ids)
    changes = []
    for change in pending.proposal.changes:
        c = change.constraint
        if (
            c is None
            or c.field in blocked_fields
            or change.id in blocked_ids
            or ":implied:" in change.id
            or signature(change) not in safe
            or not change.source_spans
            or any(span.origin != "current" or span.turn_id != turn_id for span in change.source_spans)
        ):
            continue
        changes.append(
            PendingChangePreview(
                field=c.field,
                label=(labels or {}).get(c.field, c.field),
                operation=change.operation,
                operator=c.op,
                value=c.value,
                source_text=[span.text for span in change.source_spans],
            )
        )
    return PendingPreview(question_id=pending.question_id, base_version=pending.base_version, changes=changes) if changes else None
