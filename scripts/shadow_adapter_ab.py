"""Offline shadow A/B for the adapter semantic-resolution boundary. Zero inference.

Replays the SAME saved validated StructuredRequests from a frozen DEV report
through the wired SchemaRequestAdapter under two injected resolvers and scores
field-level outcomes, explicit safety probes, and the cohort's forbidden_query
constraints:

  exact         ExactSurfaceResolver — the historical literal alias/enum veto
  inflectional  InflectionalResolver — accepts forms of declared surfaces,
                refuses unknown vocabulary, resolves value=None when the cited
                evidence denotes exactly one declared value

Safety probes are hand-written from the shipped unit tests and observed failure
classes; they never come from expected_query, and no resolver receives labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recagent.domains.demo import request_adapter  # noqa: E402
from recagent.interpretation import InterpretationIssue, StructuredRequest  # noqa: E402
from recagent.models import Query  # noqa: E402
from recagent.resolution import ExactSurfaceResolver, InflectionalResolver  # noqa: E402

VARIANTS = ("shipped", "exact", "inflectional")
SNAPSHOT = ROOT / "artifacts" / "llm-first-product" / "baseline-snapshot" / "request_mapping.py.bak"


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shipped_adapter():
    """Load the frozen pre-change adapter so the baseline arm is the real one.

    Without this arm the comparison would silently credit the resolver with gains
    that come from shared scalar/quote normalization, because both resolver arms
    run through the current module.
    """
    if not SNAPSHOT.exists():
        return None
    import importlib.util
    from importlib.machinery import SourceFileLoader

    # .bak is not a recognized source suffix, so the loader must be explicit, and
    # the module name must live in the recagent namespace because the snapshot
    # uses relative imports.
    name = "recagent._ab_baseline_mapping"
    loader = SourceFileLoader(name, str(SNAPSHOT))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module.SchemaRequestAdapter


def make_variant(kind: str):
    base = request_adapter()
    if kind == "shipped":
        snapshot = load_shipped_adapter()
        if snapshot is None:
            return None
        return snapshot(
            base.model,
            aliases=base.aliases,
            exclusions=base.exclusions,
            domain_field=base.domain_field,
            field_labels=base.field_labels,
            scalar_aliases=base.scalar_aliases,
            numeric_units=base.numeric_units,
        )
    resolver = ExactSurfaceResolver() if kind == "exact" else InflectionalResolver()
    return type(base)(
        base.model,
        aliases=base.aliases,
        exclusions=base.exclusions,
        domain_field=base.domain_field,
        field_labels=base.field_labels,
        scalar_aliases=base.scalar_aliases,
        numeric_units=base.numeric_units,
        resolver=resolver,
    )


def replay_turn(variant, turn, expected_query):
    calls = turn.get("structured_calls") or []
    if not calls or not calls[-1].get("validated_output"):
        return None
    raw = calls[-1]["validated_output"]
    payload = calls[-1].get("payload") or {}
    message = payload.get("message") or turn.get("user", "")
    previous = Query.model_validate(payload.get("previous") or {})
    unresolved = [InterpretationIssue.model_validate(i) for i in (payload.get("unresolved") or [])]
    request = StructuredRequest.model_validate(raw)
    query, issues = variant.apply(request, previous, message, unresolved)
    dumped = query.model_dump(mode="json")
    saved = turn.get("final_query") or {}
    rows = {}
    for field_name, want in expected_query.items():
        got = dumped.get(field_name)
        if field_name != "intent" and not any(u.get("field") == field_name for u in raw.get("updates", [])):
            rows[field_name] = "model_omission"
            continue
        rows[field_name] = "ok" if got == want else "mismatch"
    return {
        "rows": rows,
        "query": dumped,
        "saved_query_matches": all(dumped.get(k) == saved.get(k) for k in expected_query),
        "issues": [{"kind": i.kind, "field": i.field, "value": i.value} for i in issues],
        "n_issues": len(issues),
    }


def check_forbidden(variant, turn, forbidden_query) -> list[str]:
    """Any forbidden value that reaches Query is an automatic reject."""
    calls = turn.get("structured_calls") or []
    if not calls or not calls[-1].get("validated_output"):
        return []
    raw = calls[-1]["validated_output"]
    payload = calls[-1].get("payload") or {}
    message = payload.get("message") or turn.get("user", "")
    previous = Query.model_validate(payload.get("previous") or {})
    unresolved = [InterpretationIssue.model_validate(i) for i in (payload.get("unresolved") or [])]
    query, _ = variant.apply(StructuredRequest.model_validate(raw), previous, message, unresolved)
    dumped = query.model_dump(mode="json")
    return [
        f"{field_name}={dumped.get(field_name)!r} must not be {forbidden!r}"
        for field_name, forbidden in forbidden_query.items()
        if dumped.get(field_name) == forbidden
    ]


SAFETY_PROBES = [
    (
        "middle_must_not_become_nearest_enum",
        "level",
        None,
        {"updates": [{"field": "level", "value": "начальный", "source_text": "middle"}]},
        Query(kind="course"),
        "Ищу курс уровня middle",
    ),
    (
        "middle_must_not_become_advanced",
        "level",
        None,
        {"updates": [{"field": "level", "value": "продвинутый", "source_text": "middle-разработчика"}]},
        Query(kind="course", genre="python"),
        "Нужна программа по Python для middle-разработчика",
    ),
    (
        "fabricated_evidence_not_normalization",
        "level",
        None,
        {"updates": [{"field": "level", "value": "начальный", "source_text": "начальный"}]},
        Query(kind="course"),
        "Ищу курс уровня middle",
    ),
    (
        "numeric_citation_must_match_900",
        "max_minutes",
        None,
        {"updates": [{"field": "max_minutes", "value": 900, "source_text": "90"}]},
        Query(),
        "До 90 минут",
    ),
    (
        "numeric_citation_must_match_90",
        "max_minutes",
        None,
        {"updates": [{"field": "max_minutes", "value": 90, "source_text": "90"}]},
        Query(),
        "До 900 минут",
    ),
    (
        "numeric_citation_must_match_60",
        "max_minutes",
        None,
        {"updates": [{"field": "max_minutes", "value": 60, "source_text": "1 час"}]},
        Query(),
        "До 21 часа",
    ),
    (
        "numeric_citation_must_match_5",
        "max_minutes",
        None,
        {"updates": [{"field": "max_minutes", "value": 5, "source_text": "5"}]},
        Query(),
        "До 1.5 минут",
    ),
    (
        "invented_title_not_accepted",
        "seed_title",
        None,
        {"updates": [{"field": "seed_title", "value": "Выдуманный объект", "source_text": "Настоящий объект"}]},
        Query(),
        "Найди Настоящий объект",
    ),
    (
        "practical_false_from_practice",
        "practical",
        None,
        {"updates": [{"field": "practical", "value": False, "source_text": "практика"}]},
        Query(),
        "Нужна практика",
    ),
    (
        "genre_conflict_detected",
        None,
        True,
        {
            "updates": [
                {"field": "genre", "value": "драма", "source_text": "драма"},
                {"field": "genre", "value": "драма", "source_text": "драмы", "operation": "exclude"},
            ]
        },
        Query(kind="film"),
        "Драма, но без драмы",
    ),
    (
        "negated_tone_not_inverted",
        "tone",
        None,
        {"updates": [{"field": "tone", "value": "лёгкий", "source_text": "не мрачный"}]},
        Query(kind="film"),
        "Что-нибудь не мрачное",
    ),
    (
        "unknown_synonym_stays_unresolved",
        "kind",
        None,
        {"updates": [{"field": "kind", "value": "course", "source_text": "программа"}]},
        Query(),
        "Нужна программа по Python",
    ),
]

POSITIVE_PROBES = [
    (
        "level_canonical_from_inflected_source",
        "level",
        "начальный",
        {"updates": [{"field": "level", "value": "начальный", "source_text": "для начинающих"}]},
        Query(kind="course"),
        "Нужна программа обучения Python для начинающих.",
    ),
    (
        "level_advanced_from_inflected_source",
        "level",
        "продвинутый",
        {"updates": [{"field": "level", "value": "продвинутый", "source_text": "для опытного специалиста"}]},
        Query(kind="course"),
        "Подберите обучение для опытного специалиста",
    ),
    (
        "level_from_inflected_noun_phrase",
        "level",
        "начальный",
        {"updates": [{"field": "level", "value": "начальный", "source_text": "начального уровня"}]},
        Query(kind="course"),
        "Покажите курс по машинному обучению начального уровня.",
    ),
    (
        "kind_from_phrase_containing_canonical",
        "kind",
        "course",
        {"updates": [{"field": "kind", "value": "course", "source_text": "практическом курсе"}]},
        Query(),
        "Хочу освоить Python на практическом курсе.",
    ),
    (
        "genre_from_inflected_source",
        "genre",
        "комедия",
        {"updates": [{"field": "genre", "value": "комедия", "source_text": "комедии"}]},
        Query(kind="film"),
        "Оставьте комедии, нужен фильм.",
    ),
    (
        "null_value_recovered_from_canonical_source",
        "genre",
        "python",
        {"updates": [{"field": "genre", "value": None, "source_text": "python"}]},
        Query(kind="course"),
        "Найти курс по теме python.",
    ),
    (
        "genre_multiword_surface_inflected",
        "genre",
        "машинное обучение",
        {"updates": [{"field": "genre", "value": "машинное обучение", "source_text": "по машинному обучению"}]},
        Query(kind="course"),
        "Подберите обучение по машинному обучению",
    ),
    (
        "numeric_with_unit_phrase",
        "max_minutes",
        1,
        {"updates": [{"field": "max_minutes", "value": 1, "source_text": "не дольше 1 минуты"}]},
        Query(kind="film"),
        "Найди фильм-комедию не дольше 1 минуты.",
    ),
    (
        "practical_from_inflected_phrase",
        "practical",
        True,
        {"updates": [{"field": "practical", "value": True, "source_text": "с заданиями"}]},
        Query(kind="course"),
        "Подберите курс «python» по теме с заданиями.",
    ),
    (
        "quoted_title_normalized",
        "seed_title",
        "Декоратор в проде",
        {"updates": [{"field": "seed_title", "value": "«Декоратор в проде»", "source_text": "«Декоратор в проде»"}]},
        Query(),
        "Найди «Декоратор в проде».",
    ),
]


def run_safety_probes(variant):
    """Every probe asserts a rejection: the unsafe value must not reach Query."""
    results = []
    for name, field_name, require_issue, payload, previous, message in SAFETY_PROBES:
        query, issues = variant.apply(StructuredRequest.model_validate(payload), previous, message)
        got = getattr(query, field_name) if field_name else None
        if name == "genre_conflict_detected":
            passed = any(i.kind == "conflict" for i in issues)
        elif require_issue:
            passed = bool(issues) and got is None
        else:
            passed = got is None
        results.append(
            {
                "probe": name,
                "passed": bool(passed),
                "got": got,
                "expected": "rejected",
                "issues": [{"kind": i.kind, "field": i.field, "value": i.value} for i in issues],
            }
        )
    return results


def run_positive_probes(variant):
    """Every probe asserts a clean accept: the canonical value must reach Query."""
    results = []
    for name, field_name, expected_value, payload, previous, message in POSITIVE_PROBES:
        query, issues = variant.apply(StructuredRequest.model_validate(payload), previous, message)
        got = getattr(query, field_name)
        results.append(
            {
                "probe": name,
                "passed": bool(got == expected_value and not issues),
                "got": got,
                "expected": expected_value,
                "issues": [{"kind": i.kind, "field": i.field, "value": i.value} for i in issues],
            }
        )
    return results


def analyze(report_path: Path, cohort_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_id = {d["id"]: d for d in cohort["dialogues"]}
    out = {}
    for name in VARIANTS:
        variant = make_variant(name)
        if variant is None:
            out[name] = {"skipped": "baseline snapshot not found"}
            continue
        rows, per_turn, forbidden_violations = [], [], []
        for dialogue in report["dialogues"]:
            expected_turns = cohort_by_id[dialogue["id"]]["turns"]
            for index, turn in enumerate(dialogue["turns"]):
                expected_query = expected_turns[index].get("expected_query", {})
                forbidden_query = expected_turns[index].get("forbidden_query", {}) or {}
                violations = check_forbidden(variant, turn, forbidden_query)
                if violations:
                    forbidden_violations.append({"dialogue": dialogue["id"], "index": index, "violations": violations})
                if not expected_query:
                    continue
                result = replay_turn(variant, turn, expected_query)
                if result is None:
                    continue
                result["dialogue"] = dialogue["id"]
                result["index"] = index
                per_turn.append(result)
                for field_name, verdict in result["rows"].items():
                    rows.append(
                        {
                            "dialogue": dialogue["id"],
                            "index": index,
                            "field": field_name,
                            "verdict": verdict,
                            "value": result["query"].get(field_name),
                            "expected": expected_query[field_name],
                            "issues": result["issues"],
                        }
                    )
            counts = Counter(r["verdict"] for r in rows)
        scored = counts["ok"] + counts["mismatch"]
        safety = run_safety_probes(variant)
        positive = run_positive_probes(variant)
        out[name] = {
            "expected_field_slots": len(rows),
            "verdict_counts": dict(sorted(counts.items())),
            "field_match_rate": round(counts["ok"] / len(rows), 4) if rows else None,
            "match_rate_excluding_model_omission": round(counts["ok"] / scored, 4) if scored else None,
            "model_omissions_outside_adapter_scope": counts["model_omission"],
            "turns_with_zero_issues": sum(1 for t in per_turn if t["n_issues"] == 0),
            "total_issues": sum(t["n_issues"] for t in per_turn),
            "forbidden_query_violations": forbidden_violations,
            "safety_probes_passed": sum(1 for s in safety if s["passed"]),
            "safety_probes_total": len(safety),
            "safety_failures": [s for s in safety if not s["passed"]],
            "positive_probes_passed": sum(1 for p in positive if p["passed"]),
            "positive_probes_total": len(positive),
            "positive_failures": [f"{p['probe']}: got {p['got']!r} want {p['expected']!r}" for p in positive if not p["passed"]],
            "mismatch_rows": [r for r in rows if r["verdict"] == "mismatch"],
            "_rows": rows,
        }

    # Paired regression: a field that shipped resolved correctly must not break.
    baseline_rows = out.get("shipped", {}).get("_rows")
    if baseline_rows:
        baseline_index = {(r["dialogue"], r["index"], r["field"]): r["verdict"] for r in baseline_rows}
        for name in VARIANTS:
            arm = out.get(name) or {}
            rows = arm.get("_rows")
            if not rows:
                continue
            regressed = [
                {"dialogue": r["dialogue"], "index": r["index"], "field": r["field"], "value": r["value"], "expected": r["expected"]}
                for r in rows
                if baseline_index.get((r["dialogue"], r["index"], r["field"])) == "ok" and r["verdict"] == "mismatch"
            ]
            improved = [
                {"dialogue": r["dialogue"], "index": r["index"], "field": r["field"], "value": r["value"], "expected": r["expected"]}
                for r in rows
                if baseline_index.get((r["dialogue"], r["index"], r["field"])) == "mismatch" and r["verdict"] == "ok"
            ]
            arm["regressions_vs_shipped"] = regressed
            arm["improvements_vs_shipped_count"] = len(improved)
            arm["improvements_vs_shipped"] = improved
    for name in VARIANTS:
        (out.get(name) or {}).pop("_rows", None)
    return {
        "schema_version": 2,
        "analysis": "adapter_shadow_ab",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inference_calls": 0,
        "report": {"path": str(report_path), "sha256": sha256_bytes(report_path), "label": report.get("label")},
        "cohort": {"path": str(cohort_path), "sha256": sha256_bytes(cohort_path)},
        "adapter_source_sha256": sha256_bytes(ROOT / "src" / "recagent" / "request_mapping.py"),
        "resolution_source_sha256": sha256_bytes(ROOT / "src" / "recagent" / "resolution.py"),
        "shipped_baseline_snapshot": ({"path": str(SNAPSHOT), "sha256": sha256_bytes(SNAPSHOT)} if SNAPSHOT.exists() else None),
        "arms": {
            "shipped": "frozen pre-change adapter snapshot (true baseline)",
            "exact": "current module, ENUM path literal-only (ExactSurfaceResolver), scalar/quote path shared with "
            "inflectional -> isolates the enum resolver contribution",
            "inflectional": "current module, enum and scalar paths both new -> the proposed change",
        },
        "arm_notes": [
            "resolve_scalar is shared by the exact and inflectional arms, so quote normalization, numeric-literal "
            "acceptance and inflectional boolean aliases are already active in the exact arm. Passing profile=None to "
            "resolve_scalar restores literal-only scalar behaviour if that isolation is ever needed.",
            "Attribution of a saved report (scripts/analyze_adapter_layer.py) scores the SAVED final_query, so it is "
            "only valid for the adapter that produced the report; before/after comes from this script and from "
            "scripts/replay_adapter_dialogues.py.",
        ],
        "variants": out,
        "decision_rule": (
            "KEEP inflectional only if safety_probes_passed == total for every arm, "
            "forbidden_query_violations == [], ok strictly increases vs shipped, and no "
            "field that was ok under shipped becomes mismatch."
        ),
        "limitations": [
            "Field-level replay only: policy/retrieval/rerank are NOT re-executed, so this does not measure dialog success.",
            "Replay reuses the captured payload.previous, so state divergence after a changed answer is not propagated.",
            "Safety and positive probes are hand-written from shipped unit tests and observed failure classes, not from expected_query.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "product_llm_first_dev_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [analyze(p, args.cohort) for p in args.report]
    payload = runs[0] if len(runs) == 1 else {"schema_version": 2, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for run in runs:
        print(
            json.dumps(
                {
                    "report": run["report"]["path"],
                    "variants": {k: {kk: vv for kk, vv in v.items() if kk != "mismatch_rows"} for k, v in run["variants"].items()},
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
