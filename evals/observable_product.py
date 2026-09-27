"""Versioned saved-response evaluation using dataset-owned spoken truth only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.explanations import audit_text
from evals.oracle import SpokenConstraints, satisfies_spoken

VERSION = "observable-product-v1"
STATES = {"recommend", "clarify", "no_results"}


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def safe_read_json(path: Path):
    resolved = path.resolve()
    if any(any(marker in part.casefold() for marker in ("blind", "final_holdout", "final-holdout")) for part in resolved.parts):
        raise ValueError("Closed-data path is forbidden")
    raw = resolved.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _ids(value, label):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{label} must be an explicit list of nonempty IDs")
    return value


def state_contract(turn):
    explicit = "expected_state" in turn or "recommendation_exists" in turn
    allowed = "allowed_states" in turn or "presence_by_state" in turn
    if explicit == allowed:
        raise ValueError("Exactly one state/presence annotation form is required")
    if explicit:
        if turn.get("expected_state") not in STATES or type(turn.get("recommendation_exists")) is not bool:
            raise ValueError("Invalid expected state/presence")
        return {turn["expected_state"]: turn["recommendation_exists"]}
    states = turn.get("allowed_states")
    presence = turn.get("presence_by_state")
    if (
        not isinstance(states, list)
        or not states
        or len(states) != len(set(states))
        or not set(states) <= STATES
        or not isinstance(presence, dict)
        or set(presence) != set(states)
        or any(type(v) is not bool for v in presence.values())
    ):
        raise ValueError("Invalid allowed states/presence mapping")
    return presence


def validate_cohort(data):
    if data.get("split") not in {"dev", "contract"} or data.get("origin") not in {"synthetic_code", "ai_authored_synthetic"}:
        raise ValueError("Only explicitly open synthetic DEV/CONTRACT cohorts are supported")
    dialogues = data.get("dialogues")
    if not isinstance(dialogues, list) or not dialogues:
        raise ValueError("Empty/missing cohort")
    ids = [d.get("id") for d in dialogues]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate/missing dialogue IDs")
    for dialogue in dialogues:
        if not isinstance(dialogue.get("turns"), list) or not dialogue["turns"]:
            raise ValueError("Dialogue turns cannot be empty")
        for index, turn in enumerate(dialogue["turns"]):
            if not isinstance(turn.get("user"), str) or not turn["user"]:
                raise ValueError("Every turn needs original user text")
            state_contract(turn)
            references = turn.get("disjoint_from_turns", [])
            if (
                not isinstance(references, list)
                or any(type(i) is not int or not 0 <= i < index for i in references)
                or len(references) != len(set(references))
            ):
                raise ValueError("Disjoint references must be unique preceding turn indices")
            SpokenConstraints.model_validate(turn.get("spoken", {}))
            for key in ("exact_recommendation_ids", "forbidden_recommendation_ids"):
                if key in turn:
                    _ids(turn[key], key)
            for key in ("expected_query", "forbidden_query"):
                if key in turn and not isinstance(turn[key], dict):
                    raise ValueError("Query annotations must be mappings")


def assess_turn(turn, observed, catalog):
    """Uniform checks for all IDs; Query never supplies spoken ground truth."""
    states = state_contract(turn)
    ids = _ids(observed.get("shown_ids"), "shown_ids")
    actual_state = observed.get("actual_state")
    if actual_state not in STATES or not isinstance(observed.get("final_query"), dict):
        raise ValueError("Malformed observed state/Query")
    failures = []
    if actual_state not in states:
        failures.append("state_mismatch")
    elif bool(ids) != states[actual_state]:
        failures.append("recommendation_presence")
    if len(ids) != len(set(ids)):
        failures.append("duplicate_ids")
    unknown = [i for i in ids if i not in catalog]
    if unknown:
        failures.append("unknown_id")
    spoken = SpokenConstraints.model_validate(turn.get("spoken", {}))
    violations = {
        i: list(satisfies_spoken(catalog[i], spoken)[1]) for i in ids if i in catalog and not satisfies_spoken(catalog[i], spoken)[0]
    }
    if violations:
        failures.append("spoken_constraint_violation")
    if "exact_recommendation_ids" in turn and ids != turn["exact_recommendation_ids"]:
        failures.append("exact_ids")
    if set(ids) & set(turn.get("forbidden_recommendation_ids", [])):
        failures.append("forbidden_ids")
    query = observed["final_query"]

    def equal(a, b):
        return type(a) is type(b) and a == b

    diagnostics = [f"query:{k}" for k, v in turn.get("expected_query", {}).items() if not equal(query.get(k), v)]
    diagnostics += [f"forbidden_query:{k}" for k, v in turn.get("forbidden_query", {}).items() if equal(query.get(k), v)]
    return {
        "pass": not failures,
        "failures": failures,
        "unknown_ids": unknown,
        "spoken_violations": violations,
        "active_query_checks_separate": diagnostics,
    }


def audit_recommendation_text(observed, catalog):
    evidence = observed.get("recommendation_evidence")
    if evidence is None:
        return {"available": False, "claims": None, "unsupported": None, "rate": None}
    if not isinstance(evidence, list):
        raise ValueError("Malformed recommendation evidence")
    if [rec.get("item_id") for rec in evidence] != observed.get("shown_ids"):
        return {"available": False, "claims": None, "unsupported": None, "rate": None}
    claims = unsupported = 0
    for rec in evidence:
        if not isinstance(rec.get("claim_texts"), list) or any(not isinstance(v, str) for v in rec["claim_texts"]):
            raise ValueError("Malformed claim text list")
        item = catalog.get(rec.get("item_id"))
        if item is None:
            # Catalog identity failure is already a main verdict failure. Do not
            # invent a claim denominator for an unknown object.
            return {"available": False, "claims": None, "unsupported": None, "rate": None}
        audit = audit_text(" ".join(rec["claim_texts"]), item, history=(), catalog=catalog.values())
        claims += audit["claims"]
        unsupported += len(audit["invalid"])
    return {"available": True, "claims": claims, "unsupported": unsupported, "rate": unsupported / claims if claims else None}


def evaluate_report(data, report, *, cohort_sha256, catalog):
    validate_cohort(data)
    if report.get("complete") is not True or report.get("cohort", {}).get("sha256") != cohort_sha256:
        raise ValueError("Incomplete report or mismatched cohort hash")
    if report.get("report_sha256") != canonical_hash({k: v for k, v in report.items() if k != "report_sha256"}):
        raise ValueError("Report content hash mismatch")
    if report.get("model_identity_before") != report.get("model_identity_after"):
        raise ValueError("Model identity changed during the recorded run")
    actual_catalog_hash = canonical_hash([item.model_dump(mode="json") for item in catalog.values()])
    recorded_catalog_hash = report.get("catalog_sha256")
    if data["split"] == "contract" and recorded_catalog_hash is None:
        raise ValueError("CONTRACT report requires an explicit catalog hash")
    if recorded_catalog_hash is not None and recorded_catalog_hash != actual_catalog_hash:
        raise ValueError("Report catalog hash mismatch")
    actual = report.get("dialogues")
    if not isinstance(actual, list) or [d.get("id") for d in actual] != [d["id"] for d in data["dialogues"]]:
        raise ValueError("Missing, duplicate, reordered or extra dialogues")
    rows = []
    for expected, observed in zip(data["dialogues"], actual, strict=True):
        if (observed.get("success") is not None and type(observed.get("success")) is not bool) or len(observed.get("turns", [])) != len(
            expected["turns"]
        ):
            raise ValueError("Missing turn or malformed historical verdict")
        turns = []
        for index, (turn, result) in enumerate(zip(expected["turns"], observed["turns"], strict=True)):
            if result.get("index") != index or type(result.get("index")) is not int or result.get("user") != turn["user"]:
                raise ValueError("Turn index/text differs from frozen cohort")
            verdict = assess_turn(turn, result, catalog)
            if any(set(result["shown_ids"]) & set(observed["turns"][i]["shown_ids"]) for i in turn.get("disjoint_from_turns", [])):
                verdict["failures"].append("repeated_previous_ids")
                verdict["pass"] = False
            turns.append({"index": index, **verdict, "grounding_templates": audit_recommendation_text(result, catalog)})
        rows.append(
            {
                "id": expected["id"],
                "pass": all(t["pass"] for t in turns),
                "legacy_recorded_success": observed.get("success"),
                "turns": turns,
            }
        )
    legacy_present = all(type(r["legacy_recorded_success"]) is bool for r in rows)
    if data["split"] == "dev" and not legacy_present:
        raise ValueError("Historical DEV requires its original recorded verdicts")
    if not legacy_present and any(r["legacy_recorded_success"] is not None for r in rows):
        raise ValueError("Partial legacy verdicts are invalid")
    legacy = sum(r["legacy_recorded_success"] for r in rows) if legacy_present else None
    if legacy_present and report.get("summary", {}).get("full_dialog_successes") != legacy:
        raise ValueError("Historical summary differs from case verdicts")
    observed_turns = [turn for row in actual for turn in row["turns"]]
    clarification_turns = sum(t["actual_state"] == "clarify" for t in observed_turns)
    accounting = {}
    for field in ("llm_calls", "llm_tokens"):
        values = [t.get(field) for t in observed_turns]
        available = all(type(v) is int and v >= 0 for v in values)
        accounting[field] = {"available": available, "total": sum(values) if available else None}
    return {
        "metric_version": VERSION,
        "catalog_identity": {
            "verified": recorded_catalog_hash is not None,
            "recorded_sha256": recorded_catalog_hash,
            "scoring_sha256": actual_catalog_hash,
        },
        "lanes": {
            "observable_delivery": {
                "dialogues_pass": sum(r["pass"] for r in rows),
                "dialogues": len(rows),
                "turns_pass": sum(t["pass"] for r in rows for t in r["turns"]),
                "turns": sum(len(r["turns"]) for r in rows),
            },
            "legacy_recorded": {"available": legacy_present, "dialogues_pass": legacy, "dialogues": len(rows), "rescored": False},
            "active_query_diagnostics": {
                "failed_turns": sum(bool(t["active_query_checks_separate"]) for r in rows for t in r["turns"]),
                "affects_observable_verdict": False,
            },
        },
        "observed_accounting": {
            "clarification_turns": clarification_turns,
            "mean_clarifications_per_dialogue": clarification_turns / len(rows),
            **accounting,
        },
        "rows": rows,
    }


def compare_reports(data, baseline, candidate, *, cohort_sha256, catalog, require_matching_run_configuration=True):
    if require_matching_run_configuration:
        for field in ("configuration", "model_identity_before", "model_identity_after"):
            if field not in baseline or field not in candidate or baseline[field] != candidate[field]:
                raise ValueError(f"Comparison has mismatched/missing {field}")
    before = evaluate_report(data, baseline, cohort_sha256=cohort_sha256, catalog=catalog)
    after = evaluate_report(data, candidate, cohort_sha256=cohort_sha256, catalog=catalog)
    return {
        "before": before,
        "after": after,
        "improved": [a["id"] for b, a in zip(before["rows"], after["rows"], strict=True) if not b["pass"] and a["pass"]],
        "regressed": [a["id"] for b, a in zip(before["rows"], after["rows"], strict=True) if b["pass"] and not a["pass"]],
    }
