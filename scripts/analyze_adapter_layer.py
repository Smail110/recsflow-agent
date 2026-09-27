"""Adapter-only offline attribution: zero inference, frozen DEV artifacts.

Question this script answers: when the LLM already produced a semantically
usable StructuredRequest, how much of that information does
SchemaRequestAdapter destroy, and through which guard?

Per-update verdicts (scope = fields expected by the cohort turn):
  preserved_correctly  semantically correct update survived into Query
  veto_of_correct      semantically correct update was destroyed by a guard
  null_recoverable     value=null but source_text is itself canonical -> adapter
                       could resolve it without guessing
  null_unrecoverable   value=null and source_text is not canonical
  veto_of_wrong        guard correctly stopped a wrong value (safety success)
  false_accept         wrong value passed through
  formatting_defect    value accepted but not canonically normalized

Guard subtype for every veto is re-derived from the adapter's own conditions so
the taxonomy separates alias veto from source-span/numeric/scalar guards.

Model omissions (expected field with NO update at all) are reported separately
and are explicitly outside adapter scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recagent.domains.demo import request_adapter  # noqa: E402
from recagent.interpretation import InterpretationIssue, StructuredRequest  # noqa: E402
from recagent.models import Query  # noqa: E402
from recagent.request_mapping import canonical_text  # noqa: E402

QUOTE_CHARS = "«»\"“”'‘’"


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_semantically_correct(value, expected) -> bool:
    """LLM understood the field correctly, judged against cohort ground truth."""
    if value is None:
        return False
    if isinstance(expected, bool) or isinstance(value, bool):
        return value is expected or value == expected
    return canonical_text(str(value)) == canonical_text(str(expected))


def guard_subtype(adapter, name: str, value, source_text: str, message: str) -> str:
    """Re-derive which adapter guard rejects this update (diagnostic only)."""
    if canonical_text(source_text) not in canonical_text(message):
        return "source_span_guard"
    if type(value) is int and not re.search(
        r"(?<![\d.,])" + re.escape(canonical_text(source_text)) + r"(?!\d|[.,]\d)",
        canonical_text(message),
    ):
        return "numeric_guard"
    if name in adapter.enums:
        permitted = adapter.enums[name] | {canonical_text(k): v for k, v in adapter.aliases.get(name, {}).items()}
        raw = canonical_text(source_text)
        if raw not in permitted:
            return "alias_veto:source_not_lexical"
        if not isinstance(value, str) or canonical_text(value) not in permitted:
            return "alias_veto:value_not_permitted"
        if permitted[canonical_text(value)] != permitted[raw]:
            return "alias_veto:value_source_mismatch"
    elif not adapter._scalar_matches(name, value, source_text):
        return "scalar_guard"
    return "not_rejected_by_guards"


def analyze_turn(adapter, turn: dict, expected_query: dict) -> dict:
    calls = turn.get("structured_calls") or []
    raw = (calls[-1].get("validated_output") if calls else None) or {}
    payload = (calls[-1].get("payload") if calls else None) or {}
    message = payload.get("message") or turn.get("user", "")
    final = turn.get("final_query") or {}

    # Offline replay through the shipped adapter, then a second replay with the
    # candidate logic injected, so both verdicts come from the same code path.
    previous = Query.model_validate(payload.get("previous") or {})
    unresolved = [InterpretationIssue.model_validate(i) for i in (payload.get("unresolved") or [])]
    request = StructuredRequest.model_validate(raw)
    replayed, replayed_issues = adapter.apply(request, previous, message, unresolved)
    replayed_dump = replayed.model_dump(mode="json")
    replay_matches = all(replayed_dump.get(k) == final.get(k) for k in expected_query)

    updates = [u for u in raw.get("updates", []) if isinstance(u, dict)]
    by_field: dict[str, list[dict]] = {}
    for u in updates:
        by_field.setdefault(u["field"], []).append(u)

    rows = []
    for field, expected in expected_query.items():
        if field == "intent":
            # intent is carried on StructuredRequest.intent, not via updates
            got = final.get(field)
            rows.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": got,
                    "verdict": "preserved_correctly" if got == expected else "intent_mismatch",
                    "guard": None,
                    "update": None,
                    "note": "intent comes from StructuredRequest.intent, adapter copies it verbatim",
                }
            )
            continue
        field_updates = by_field.get(field, [])
        if not field_updates:
            rows.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": final.get(field),
                    "verdict": "model_omission",
                    "guard": None,
                    "update": None,
                    "note": "no update emitted by the model; outside adapter scope",
                }
            )
            continue
        for u in field_updates:
            value, source_text = u.get("value"), u.get("source_text", "")
            operation = u.get("operation", "set")
            got = final.get(field)
            preserved = got == expected or (field == "excluded_genres" and expected in (got or []))
            note = ""
            if value is None and operation != "clear":
                source_canonical = field in adapter.enums and canonical_text(source_text) in adapter.enums[field]
                verdict = "null_recoverable" if source_canonical else "null_unrecoverable"
                guard = guard_subtype(adapter, field, value, source_text, message) if not source_canonical else None
                note = (
                    "source_text is itself a canonical schema value -> resolvable without guessing"
                    if source_canonical
                    else "source_text is not canonical; unresolved is the honest answer"
                )
            elif is_semantically_correct(value, expected):
                if preserved:
                    verdict, guard = "preserved_correctly", None
                else:
                    verdict = "veto_of_correct"
                    guard = guard_subtype(adapter, field, value, source_text, message)
                    note = "semantically correct canonical value destroyed by a lexical guard"
            else:
                if preserved:
                    verdict, guard = "false_accept", None
                    note = "wrong value passed through"
                else:
                    verdict = "veto_of_wrong"
                    guard = guard_subtype(adapter, field, value, source_text, message)
                    note = "guard correctly stopped an unsupported value (safety success)"
            # Formatting defect overrides: value accepted but not canonical text form.
            if isinstance(got, str) and got.strip(QUOTE_CHARS) != got:
                verdict = "formatting_defect"
                guard = None
                note = (
                    "value reached Query with quoting preserved; provider.find_title "
                    "normalizes case/ё only, so the fuzzy path masks it downstream"
                )
            rows.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": got,
                    "verdict": verdict,
                    "guard": guard,
                    "operation": operation,
                    "update": {"value": value, "source_text": source_text, "operation": operation},
                    "note": note,
                }
            )

    return {
        "dialogue": None,  # filled by caller
        "index": None,
        "user": turn.get("user"),
        "message_sent_to_model": message,
        "expected_state": turn.get("expected_state"),
        "actual_state": turn.get("actual_state"),
        "runner_failures": turn.get("failures", []),
        "replay_matches_saved_query": replay_matches,
        "replayed_issues": [{"kind": i.kind, "field": i.field, "value": i.value} for i in replayed_issues],
        "raw_issues": raw.get("issues", []),
        "raw_clarification_required": raw.get("clarification_required"),
        "rows": rows,
    }


def summarize(turns: list[dict]) -> dict:
    all_rows = [dict(r, dialogue=t["dialogue"], index=t["index"]) for t in turns for r in t["rows"]]
    verdicts = Counter(r["verdict"] for r in all_rows)
    guards = Counter(r["guard"] for r in all_rows if r["verdict"] == "veto_of_correct" and r["guard"])
    wrong_guards = Counter(r["guard"] for r in all_rows if r["verdict"] == "veto_of_wrong" and r["guard"])
    veto_fields = Counter(r["field"] for r in all_rows if r["verdict"] == "veto_of_correct")
    omission_fields = Counter(r["field"] for r in all_rows if r["verdict"] == "model_omission")

    correct_updates = verdicts["preserved_correctly"] + verdicts["veto_of_correct"]
    preservation = (verdicts["preserved_correctly"] / correct_updates) if correct_updates else None
    return {
        "total_expected_field_slots": len(all_rows),
        "verdict_counts": dict(sorted(verdicts.items())),
        "semantically_correct_updates": correct_updates,
        "adapter_preserved_correctly": verdicts["preserved_correctly"],
        "adapter_destroyed_correct": verdicts["veto_of_correct"],
        "adapter_preservation_rate": round(preservation, 4) if preservation is not None else None,
        "false_accepts": verdicts["false_accept"],
        "safety_successes_veto_of_wrong": verdicts["veto_of_wrong"],
        "null_recoverable": verdicts["null_recoverable"],
        "null_unrecoverable": verdicts["null_unrecoverable"],
        "formatting_defects": verdicts["formatting_defect"],
        "model_omissions_outside_adapter_scope": verdicts["model_omission"],
        "veto_guards_on_correct_values": dict(sorted(guards.items())),
        "veto_guards_on_wrong_values_safety": dict(sorted(wrong_guards.items())),
        "veto_of_correct_by_field": dict(sorted(veto_fields.items())),
        "model_omission_by_field": dict(sorted(omission_fields.items())),
        "replay_matches_saved_query": sum(bool(t["replay_matches_saved_query"]) for t in turns),
    }


def load_adapter(baseline: Path | None):
    """Load the adapter that produced the saved report.

    Required for evidence integrity: this script attributes a saved run, so it
    must replay the adapter as it was at recording time. Running it against a
    newer adapter silently mixes two versions and inflates apparent veto counts.
    """
    if baseline is None:
        return request_adapter()
    import importlib.util
    from importlib.machinery import SourceFileLoader

    name = "recagent._attribution_baseline"
    loader = SourceFileLoader(name, str(baseline))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    base = request_adapter()
    return module.SchemaRequestAdapter(
        base.model,
        aliases=base.aliases,
        exclusions=base.exclusions,
        domain_field=base.domain_field,
        field_labels=base.field_labels,
        scalar_aliases=base.scalar_aliases,
        numeric_units=base.numeric_units,
    )


def analyze_report(report_path: Path, cohort_path: Path, baseline: Path | None = None) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_id = {d["id"]: d for d in cohort["dialogues"]}
    adapter = load_adapter(baseline)
    turns = []
    for dialogue in report["dialogues"]:
        expected_turns = cohort_by_id[dialogue["id"]]["turns"]
        for index, turn in enumerate(dialogue["turns"]):
            expected_query = expected_turns[index].get("expected_query", {})
            if not expected_query:
                continue
            result = analyze_turn(adapter, turn, expected_query)
            result["dialogue"] = dialogue["id"]
            result["index"] = index
            result["tags"] = dialogue.get("tags", [])
            turns.append(result)
    return {
        "schema_version": 1,
        "analysis": "adapter_layer_attribution",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inference_calls": 0,
        "report": {
            "path": str(report_path),
            "sha256": sha256_bytes(report_path),
            "label": report.get("label"),
            "model_identity": report.get("model_identity_after") or report.get("model_identity_before"),
            "cohort_sha256": report.get("cohort", {}).get("sha256"),
        },
        "cohort": {"path": str(cohort_path), "sha256": sha256_bytes(cohort_path)},
        # adapter_source_sha256 always names the adapter that DID the attribution,
        # not the one currently in the working copy: otherwise a before-attribution
        # looks like it was produced by the after code.
        "adapter_source_sha256": sha256_bytes(baseline if baseline is not None else ROOT / "src" / "recagent" / "request_mapping.py"),
        "attributed_with": (
            {"baseline_snapshot": str(baseline), "sha256": sha256_bytes(baseline)}
            if baseline is not None
            else {"current_working_copy": str(ROOT / "src" / "recagent" / "request_mapping.py")}
        ),
        "summary": summarize(turns),
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "product_llm_first_dev_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-baseline",
        type=Path,
        default=None,
        help="replay with this saved adapter source instead of the working copy; "
        "use the pre-change snapshot when attributing an older report",
    )
    args = parser.parse_args()
    results = [analyze_report(p, args.cohort, args.adapter_baseline) for p in args.report]
    payload = results[0] if len(results) == 1 else {"schema_version": 1, "runs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for result in results:
        print(json.dumps({"report": result["report"]["path"], "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
