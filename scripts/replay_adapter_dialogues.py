"""Offline end-to-end dialog replay through the real Agent. Zero inference.

Field-level replay shows what the adapter preserved; it does NOT show whether the
dialog then succeeded, because state, policy and retrieval run after it. This
script closes that gap without any new model call: the interpreter is replaced by
a replay that serves the SAVED validated StructuredRequest for each turn, while
everything downstream (SchemaRequestAdapter, Agent, LangGraph, clarification
policy, provider retrieval, filter, rerank, grounding) executes for real.

Arms:
  shipped       frozen pre-change adapter snapshot
  inflectional  current adapter (proposed change)

Honest limitation, recorded in the artifact: because downstream state now
diverges from the recorded run, the saved raw output for turn t+1 is no longer
necessarily what the model would have produced given the changed state. This
measures the adapter's contribution under a frozen interpretation, not a
re-inferred dialog.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.oracle import SpokenConstraints, satisfies_spoken  # noqa: E402

from recagent.agent import Agent  # noqa: E402
from recagent.catalog import generate_catalog  # noqa: E402
from recagent.domains.demo import request_adapter  # noqa: E402
from recagent.interpretation import StructuredRequest  # noqa: E402
from recagent.models import ChatRequest  # noqa: E402
from recagent.providers import DemoProvider  # noqa: E402
from recagent.resolution import InflectionalResolver  # noqa: E402

SNAPSHOT = ROOT / "artifacts" / "llm-first-product" / "baseline-snapshot" / "request_mapping.py.bak"


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayInterpreter:
    """Serves the saved StructuredRequest for a fixed turn sequence."""

    def __init__(self, saved: list[dict]):
        self.saved = saved
        self.index = 0
        self.served = 0
        self.missing = 0

    def interpret(self, message: str, previous: dict, *, pending_question=None, unresolved=None):
        calls = self.saved[self.index] if self.index < len(self.saved) else []
        self.index += 1
        if not calls or not calls[-1].get("validated_output"):
            self.missing += 1
            return StructuredRequest(), 0
        self.served += 1
        return StructuredRequest.model_validate(calls[-1]["validated_output"]), 0


def make_adapter(kind: str):
    base = request_adapter()
    if kind == "shipped":
        name = "recagent._ab_replay_baseline"
        loader = SourceFileLoader(name, str(SNAPSHOT))
        spec = importlib.util.spec_from_loader(name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module.SchemaRequestAdapter(
            base.model,
            aliases=base.aliases,
            exclusions=base.exclusions,
            domain_field=base.domain_field,
            field_labels=base.field_labels,
            scalar_aliases=base.scalar_aliases,
            numeric_units=base.numeric_units,
        )
    return type(base)(
        base.model,
        aliases=base.aliases,
        exclusions=base.exclusions,
        domain_field=base.domain_field,
        field_labels=base.field_labels,
        scalar_aliases=base.scalar_aliases,
        numeric_units=base.numeric_units,
        resolver=InflectionalResolver(),
    )


def assess(turn: dict, response, by_id: dict) -> tuple[dict, list[str]]:
    """Same frozen semantic checks as scripts/evaluate_llm_first_product.py."""
    failures: list[str] = []
    query = response.query.model_dump(mode="json")
    if response.state != turn["expected_state"]:
        failures.append(f"state:{response.state}!={turn['expected_state']}")
    for field, expected in turn.get("expected_query", {}).items():
        if query.get(field) != expected:
            failures.append(f"query:{field}={query.get(field)!r}!={expected!r}")
    for field, forbidden in turn.get("forbidden_query", {}).items():
        if query.get(field) == forbidden:
            failures.append(f"forbidden_query:{field}={forbidden!r}")
    shown = [r.item.id for r in response.recommendations]
    if turn["recommendation_exists"] and not shown:
        failures.append("missing_recommendation")
    if not turn["recommendation_exists"] and shown:
        failures.append("unexpected_recommendation")
    if [i for i in shown if i not in by_id]:
        failures.append("unknown_item_id")
    spoken = SpokenConstraints.model_validate(turn.get("spoken", {}))
    if [i for i in shown if i in by_id and not satisfies_spoken(by_id[i], spoken)[0]]:
        failures.append("spoken_constraint_violation")
    if turn.get("exact_recommendation_ids") is not None and shown != turn["exact_recommendation_ids"]:
        failures.append("literal_title_not_exact")
    if set(shown).intersection(turn.get("forbidden_recommendation_ids", [])):
        failures.append("forbidden_recommendation_shown")
    return {
        "expected_state": turn["expected_state"],
        "actual_state": response.state,
        "final_query": query,
        "shown_ids": shown,
        "message": response.message,
        "clarification_slot": response.clarification_slot,
        "adapter_semantic_pass": not any(r.startswith(("query:", "forbidden_query:")) for r in failures),
        "policy_state_pass": not any(r.startswith("state:") for r in failures),
        "recommendation_contract_pass": not any(
            r in {"missing_recommendation", "unexpected_recommendation", "literal_title_not_exact"} for r in failures
        ),
    }, failures


def run_arm(kind: str, report: dict, cohort: dict, catalog: list, config: dict) -> dict:
    by_id = {item.id: item for item in catalog}
    saved_by_dialogue = {
        dialogue["id"]: [turn.get("structured_calls") or [] for turn in dialogue["turns"]] for dialogue in report["dialogues"]
    }
    rows = []
    for dialogue in cohort["dialogues"]:
        saved_turns = saved_by_dialogue.get(dialogue["id"], [])
        interpreter = ReplayInterpreter(saved_turns)
        agent = Agent(
            provider=DemoProvider(list(catalog)),
            mode="ollama",
            interpreter=interpreter,
            request_adapter=make_adapter(kind),
            question_policy=config["question_policy"],
            max_questions=config["max_questions"],
        )
        session_id = None
        turns_out, failures = [], []
        for index, turn in enumerate(dialogue["turns"]):
            response = agent.chat(ChatRequest(user_id=f"replay-{dialogue['id']}", message=turn["user"], session_id=session_id))
            session_id = response.session_id
            scored, turn_failures = assess(turn, response, by_id)
            scored.update(index=index, user=turn["user"], failures=turn_failures)
            turns_out.append(scored)
            failures.extend(f"turn{index}:{r}" for r in turn_failures)
        rows.append(
            {
                "id": dialogue["id"],
                "tags": dialogue["tags"],
                "turns": turns_out,
                "failures": failures,
                "success": not failures,
                "replay_served": interpreter.served,
                "replay_missing": interpreter.missing,
            }
        )

    all_turns = [t for r in rows for t in r["turns"]]
    counts = Counter(f.split(":", 2)[-1] for r in rows for f in r["failures"])
    return {
        "full_dialogues": len(rows),
        "full_dialog_successes": sum(r["success"] for r in rows),
        "turn_count": len(all_turns),
        "turn_successes": sum(1 for t in all_turns if not t["failures"]),
        "adapter_semantic_pass_turns": sum(1 for t in all_turns if t["adapter_semantic_pass"]),
        "policy_state_pass_turns": sum(1 for t in all_turns if t["policy_state_pass"]),
        "recommendation_contract_pass_turns": sum(1 for t in all_turns if t["recommendation_contract_pass"]),
        "missing_recommendation_turns": counts.get("missing_recommendation", 0),
        "spoken_violation_turns": counts.get("spoken_constraint_violation", 0),
        "failure_counts": dict(sorted(counts.items())),
        "replay_served_total": sum(r["replay_served"] for r in rows),
        "replay_missing_total": sum(r["replay_missing"] for r in rows),
        "dialogues": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="saved runner report with raw structured calls")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "product_llm_first_dev_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    catalog = generate_catalog(int(cohort["catalog"]["seed"]))
    arms = {kind: run_arm(kind, report, cohort, catalog, cohort["configuration"]) for kind in ("shipped", "inflectional")}

    changed = []
    base_rows = {r["id"]: r for r in arms["shipped"]["dialogues"]}
    for row in arms["inflectional"]["dialogues"]:
        old = base_rows[row["id"]]
        if old["success"] != row["success"] or old["failures"] != row["failures"]:
            changed.append(
                {
                    "id": row["id"],
                    "tags": row["tags"],
                    "shipped_success": old["success"],
                    "inflectional_success": row["success"],
                    "shipped_failures": old["failures"],
                    "inflectional_failures": row["failures"],
                    "turn_changes": [
                        {
                            "index": i,
                            "shipped_state": a["actual_state"],
                            "inflectional_state": b["actual_state"],
                            "shipped_query": a["final_query"],
                            "inflectional_query": b["final_query"],
                            "shipped_shown": a["shown_ids"],
                            "inflectional_shown": b["shown_ids"],
                        }
                        for i, (a, b) in enumerate(zip(old["turns"], row["turns"], strict=True))
                        if a["actual_state"] != b["actual_state"]
                        or a["final_query"] != b["final_query"]
                        or a["shown_ids"] != b["shown_ids"]
                    ],
                    "new_failures": [f for f in row["failures"] if f not in old["failures"]],
                    "fixed_failures": [f for f in old["failures"] if f not in row["failures"]],
                }
            )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "adapter_offline_dialog_replay",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inference_calls": 0,
        "report": {
            "path": str(args.report),
            "sha256": sha256_bytes(args.report),
            "label": report.get("label"),
            "model_identity": report.get("model_identity_after") or report.get("model_identity_before"),
        },
        "cohort": {"path": str(args.cohort), "sha256": sha256_bytes(args.cohort), "dialogues": len(cohort["dialogues"])},
        "adapter_source_sha256": sha256_bytes(ROOT / "src" / "recagent" / "request_mapping.py"),
        "resolution_source_sha256": sha256_bytes(ROOT / "src" / "recagent" / "resolution.py"),
        "shipped_snapshot": {"path": str(SNAPSHOT), "sha256": sha256_bytes(SNAPSHOT)},
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "dialogues"} for k, v in arms.items()},
        "changed_dialogue_count": len(changed),
        "changed_dialogues": changed,
        "limitations": [
            "Interpretation is replayed verbatim from the saved report: downstream state divergence means the saved "
            "raw output for a later turn is not necessarily what the model would produce given the changed state.",
            "This measures the adapter's contribution under frozen interpretation, not a re-inferred dialog.",
            "Frozen DEV cohort, AI-authored; not a production or real-traffic claim. FINAL HOLDOUT untouched.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"report": payload["report"]["path"], "arms": payload["arms"], "changed_dialogue_count": len(changed)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
