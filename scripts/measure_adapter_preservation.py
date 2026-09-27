"""Adapter preservation rate: the milestone metric, computed identically for both arms.

Denominator = updates the model emitted for a field the cohort expects, where the
proposed value is semantically correct against ground truth (a property of the
saved model output, identical for both arms).
Numerator = how many of those survived into Query.

Residual losses are listed individually so the claim is auditable and so a loss
that is actually the model's operation choice is not credited to the adapter.
"""

import argparse
import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recagent.domains.demo import request_adapter  # noqa: E402
from recagent.interpretation import InterpretationIssue, StructuredRequest  # noqa: E402
from recagent.models import Query  # noqa: E402
from recagent.request_mapping import canonical_text  # noqa: E402

SNAPSHOT = ROOT / "artifacts" / "llm-first-product" / "baseline-snapshot" / "request_mapping.py.bak"
DEFAULT_COHORT = ROOT / "data" / "product_llm_first_dev_v2.json"


def load_adapter(baseline):
    base = request_adapter()
    if not baseline:
        return base
    name = "recagent._metric_baseline"
    loader = SourceFileLoader(name, str(SNAPSHOT))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod.SchemaRequestAdapter(
        base.model,
        aliases=base.aliases,
        exclusions=base.exclusions,
        domain_field=base.domain_field,
        field_labels=base.field_labels,
        scalar_aliases=base.scalar_aliases,
        numeric_units=base.numeric_units,
    )


def correct(value, expected):
    if value is None:
        return False
    if isinstance(expected, bool) or isinstance(value, bool):
        return value == expected
    return canonical_text(str(value)) == canonical_text(str(expected))


def measure(report_path, cohort_path, baseline):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cohort = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in cohort["dialogues"]}
    adapter = load_adapter(baseline)
    total = preserved = 0
    residual = []
    for dialogue in report["dialogues"]:
        exp_turns = by_id[dialogue["id"]]["turns"]
        for index, turn in enumerate(dialogue["turns"]):
            eq = exp_turns[index].get("expected_query", {})
            calls = turn.get("structured_calls") or []
            if not calls or not calls[-1].get("validated_output"):
                continue
            raw = calls[-1]["validated_output"]
            payload = calls[-1].get("payload") or {}
            msg = payload.get("message") or turn.get("user", "")
            prev = Query.model_validate(payload.get("previous") or {})
            unres = [InterpretationIssue.model_validate(i) for i in (payload.get("unresolved") or [])]
            q, issues = adapter.apply(StructuredRequest.model_validate(raw), prev, msg, unres)
            dumped = q.model_dump(mode="json")
            for u in raw.get("updates", []):
                f = u.get("field")
                if f not in eq or f == "intent":
                    continue
                if not correct(u.get("value"), eq[f]):
                    continue  # model proposed a wrong value: not adapter's loss
                total += 1
                if dumped.get(f) == eq[f]:
                    preserved += 1
                else:
                    residual.append(
                        {
                            "dialogue": dialogue["id"],
                            "index": index,
                            "field": f,
                            "value": u.get("value"),
                            "source_text": u.get("source_text"),
                            "expected": eq[f],
                            "issues": [f"{i.kind}:{i.field}" for i in issues],
                        }
                    )
    return total, preserved, residual


DEFAULT_RUNS = (
    ("8B", ROOT / "artifacts" / "llm-first-product" / "after.json"),
    ("14B", ROOT / "artifacts" / "llm-first-product" / "model14b.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument(
        "--report", type=Path, action="append", default=None, help="saved runner report; defaults to the frozen 8B and 14B product reports"
    )
    parser.add_argument("--arms", choices=("both", "shipped", "inflectional"), default="both")
    parser.add_argument("--quiet-residuals", action="store_true")
    args = parser.parse_args()

    reports = [(p.stem, p) for p in args.report] if args.report else list(DEFAULT_RUNS)
    arms = (
        (("shipped", True), ("inflectional", False))
        if args.arms == "both"
        else (("shipped", True) if args.arms == "shipped" else (("inflectional", False),))
    )

    for label, path in reports:
        for arm, baseline in arms:
            total, preserved, residual = measure(path, args.cohort, baseline)
            rate = preserved / total if total else 0.0
            print(
                f"{label} {arm:<13} semantically_correct_updates={total} preserved={preserved} "
                f"adapter_preservation_rate={rate:.4f} residual_losses={len(residual)}"
            )
            if not args.quiet_residuals:
                for r in residual:
                    print(
                        f"    LOST {r['dialogue']:<34} t{r['index']} {r['field']:<11} value={r['value']!r} "
                        f"src={r['source_text']!r} want={r['expected']!r} issues={r['issues']}"
                    )


if __name__ == "__main__":
    main()
