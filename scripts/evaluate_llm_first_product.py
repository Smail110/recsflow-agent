"""Small frozen DEV-only dialogue replay for the LLM-first product track."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from evals.oracle import SpokenConstraints, satisfies_spoken

from recagent.catalog import generate_catalog
from recagent.factory import build_agent, component_identity
from recagent.models import ChatRequest, Query
from recagent.parsing import OllamaClient, normalize
from recagent.providers import DemoProvider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "data" / "product_llm_first_dev.json"


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: object) -> str:
    payload = value if isinstance(value, bytes) else canonical(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[union-attr]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def source_revision() -> dict[str, object]:
    paths = [
        ROOT / "src/recagent/agent.py",
        ROOT / "src/recagent/contracts.py",
        ROOT / "src/recagent/config/settings.py",
        ROOT / "src/recagent/domains/__init__.py",
        ROOT / "src/recagent/domains/demo.py",
        ROOT / "src/recagent/factory.py",
        ROOT / "src/recagent/filtering.py",
        ROOT / "src/recagent/grounding.py",
        ROOT / "src/recagent/interpretation.py",
        ROOT / "src/recagent/models.py",
        ROOT / "src/recagent/parsing.py",
        ROOT / "src/recagent/questions.py",
        ROOT / "src/recagent/query_compilation.py",
        ROOT / "src/recagent/providers.py",
        ROOT / "src/recagent/request_mapping.py",
        ROOT / "src/recagent/response_generation.py",
        ROOT / "src/recagent/ranking.py",
        Path(__file__),
        ROOT / "src/recagent/state.py",
        ROOT / "src/recagent/validation.py",
        ROOT / "src/recagent/workflow.py",
    ]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "commit": commit,
        "dirty": dirty,
        "files_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def model_identity(base_url: str, model: str, *, timeout: float) -> dict[str, str]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, trust_env=False) as client:
        tags = client.get("/api/tags")
        tags.raise_for_status()
        version = client.get("/api/version")
        version.raise_for_status()
    matches = [entry for entry in tags.json().get("models", []) if isinstance(entry, dict) and entry.get("name") == model]
    if len(matches) != 1 or not isinstance(matches[0].get("digest"), str) or not matches[0]["digest"]:
        raise ValueError(f"exact digest for {model!r} not found in /api/tags")
    if not isinstance(version.json().get("version"), str) or not version.json()["version"]:
        raise ValueError("/api/version has no version")
    return {"model": model, "digest": matches[0]["digest"], "ollama_version": version.json()["version"], "base_url": base_url.rstrip("/")}


def select_model(data: dict[str, Any], model_override: str | None, expected_digest_override: str | None) -> dict[str, object]:
    """Resolve the only permitted experimental variable without altering the cohort."""
    cohort_model = data["model"]["name"]
    effective_model = model_override or cohort_model
    expected_digest = expected_digest_override
    if expected_digest is None and effective_model == cohort_model:
        expected_digest = data["model"]["expected_digest"]
    if expected_digest is None:
        raise ValueError("--expected-digest is required when --model differs from the frozen cohort model")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("expected model digest must be a 64-character lowercase SHA-256 hex string")
    return {
        "cohort_model": cohort_model,
        "cohort_expected_digest": data["model"]["expected_digest"],
        "effective_model": effective_model,
        "expected_digest": expected_digest,
        "model_override": model_override is not None,
    }


def public_dev_utterances() -> set[str]:
    path = ROOT / "artifacts/eval-v2-20260914/recagent-eval-v2.0-dev.jsonl"
    return {
        normalize(turn["utterance"])
        for line in path.read_text(encoding="utf-8").splitlines()
        for turn in json.loads(line)["scenario"]["turns"]
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "frozen_dev_only_before_inference" or data.get("split") != "dev":
        raise ValueError("cohort must remain frozen DEV-only before inference")
    dialogues = data.get("dialogues")
    if not isinstance(dialogues, list) or len(dialogues) != 20:
        raise ValueError("cohort must contain exactly 20 bounded dialogues")
    ids = [dialogue.get("id") for dialogue in dialogues]
    if len(ids) != len(set(ids)) or any(not isinstance(case_id, str) for case_id in ids):
        raise ValueError("dialogue IDs must be unique strings")

    def tagged(name: str) -> int:
        return sum(name in dialogue.get("tags", []) for dialogue in dialogues)

    if tagged("mandatory") != 6 or tagged("unseen_paraphrase") != 8 or tagged("missing_metadata") != 2:
        raise ValueError("cohort composition differs from the frozen 6+8+2 contract")
    seen = public_dev_utterances()
    unseen = [turn["user"] for dialogue in dialogues if "unseen_paraphrase" in dialogue["tags"] for turn in dialogue["turns"]]
    duplicates = sorted(user for user in unseen if normalize(user) in seen)
    if duplicates:
        raise ValueError(f"unseen paraphrase is literal public-dev duplicate: {duplicates}")
    catalog = generate_catalog(int(data["catalog"]["seed"]))
    by_id = {item.id: item for item in catalog}
    for title_kind, expected in data["catalog"]["known_titles"].items():
        item = by_id.get(expected["id"])
        if item is None or item.title != expected["title"] or item.kind != expected["kind"]:
            raise ValueError(f"known {title_kind} title no longer matches catalog")
    for dialogue in dialogues:
        if not dialogue.get("turns"):
            raise ValueError(f"dialogue has no turns: {dialogue['id']}")
        for turn in dialogue["turns"]:
            if turn.get("expected_state") not in {"clarify", "recommend", "no_results"}:
                raise ValueError(f"invalid expected state in {dialogue['id']}")
            SpokenConstraints.model_validate(turn.get("spoken", {}))
    return data


def select_dialogues(data: dict[str, Any], requested_ids: list[str] | None) -> tuple[dict[str, Any], dict[str, object]]:
    """Select whole frozen dialogues after validating the original cohort."""
    original = data["dialogues"]
    if not requested_ids:
        return data, {"scope": "full", "selected_ids": [row["id"] for row in original], "original_dialogues": len(original)}
    duplicates = [case_id for index, case_id in enumerate(requested_ids) if case_id in requested_ids[:index]]
    if duplicates:
        raise ValueError(f"--dialogue-id repeats IDs: {sorted(set(duplicates))}")
    by_id = {row["id"]: row for row in original}
    unknown = [case_id for case_id in requested_ids if case_id not in by_id]
    if unknown:
        raise ValueError(f"unknown --dialogue-id: {unknown}; available IDs are frozen cohort IDs")
    selected = [by_id[case_id] for case_id in requested_ids]
    return {**data, "dialogues": selected}, {
        "scope": "subset",
        "selected_ids": list(requested_ids),
        "original_dialogues": len(original),
        "subset_dialogues": len(selected),
    }


def install_capture(client: OllamaClient) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Capture generic structured schema/output and parse result without product edits."""
    structured_calls: list[dict[str, object]] = []
    parse_calls: list[dict[str, object]] = []
    original_structured = client.structured
    original_parse = client.parse

    def structured_role(schema_name: str) -> str:
        """Keep dynamic canonical interpretation schemas out of response metrics."""

        if schema_name == "StructuredRequest" or schema_name.endswith(("FlatStructuredRequest", "FieldOrientedRequest")):
            return "interpretation"
        return "response_planning"

    def captured_structured(schema, system, payload):
        started = time.perf_counter()
        schema_name = getattr(schema, "__name__", str(schema))
        role = structured_role(schema_name)
        try:
            output, tokens = original_structured(schema, system, payload)
        except Exception as exc:
            structured_calls.append(
                {
                    "role": role,
                    "schema": schema_name,
                    "system_prompt": system,
                    "payload": jsonable(payload),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            raise
        structured_calls.append(
            {
                "role": role,
                "schema": schema_name,
                "system_prompt": system,
                "payload": jsonable(payload),
                "validated_output": jsonable(output),
                "tokens": tokens,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return output, tokens

    def captured_parse(message: str, previous: Query):
        start = len(structured_calls)
        output, tokens = original_parse(message, previous)
        parse_calls.append(
            {
                "message": message,
                "previous_query": previous.model_dump(mode="json"),
                "raw_parse": output.model_dump(mode="json"),
                "tokens": tokens,
                "structured_call_indexes": list(range(start, len(structured_calls))),
                "last_interpretation": jsonable(getattr(client, "last_interpretation", None)),
            }
        )
        return output, tokens

    client.structured = captured_structured  # type: ignore[method-assign]
    client.parse = captured_parse  # type: ignore[method-assign]
    return structured_calls, parse_calls


def raw_extraction_evidence(calls: list[dict[str, object]]) -> dict[str, object]:
    """Describe validated raw extraction without treating omitted inherited fields as errors."""
    extraction_calls = [call for call in calls if call.get("role") == "interpretation"]
    outputs = [call.get("validated_output") for call in extraction_calls if isinstance(call.get("validated_output"), dict)]
    updates = [update for output in outputs for update in output.get("updates", []) if isinstance(update, dict)]
    return {
        "structured_calls": len(extraction_calls),
        "validated_outputs": len(outputs),
        "errors": sum(1 for call in extraction_calls if call.get("error")),
        "update_fields": dict(sorted(Counter(str(update.get("field")) for update in updates).items())),
        "issue_count": sum(len(output.get("issues", [])) for output in outputs),
        "clarification_required": sum(bool(output.get("clarification_required")) for output in outputs),
    }


def assess_turn(turn: dict[str, Any], response, by_id: dict[str, object]) -> tuple[dict[str, object], list[str]]:
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
    shown_ids = [recommendation.item.id for recommendation in response.recommendations]
    expects_recommendation = bool(turn["recommendation_exists"])
    if expects_recommendation and not shown_ids:
        failures.append("missing_recommendation")
    if not expects_recommendation and shown_ids:
        failures.append("unexpected_recommendation")
    unknown = [item_id for item_id in shown_ids if item_id not in by_id]
    if unknown:
        failures.append("unknown_item_id")
    spoken = SpokenConstraints.model_validate(turn.get("spoken", {}))
    violations = [item_id for item_id in shown_ids if item_id in by_id and not satisfies_spoken(by_id[item_id], spoken)[0]]
    if violations:
        failures.append("spoken_constraint_violation")
    if turn.get("exact_recommendation_ids") is not None and shown_ids != turn["exact_recommendation_ids"]:
        failures.append("literal_title_not_exact")
    if set(shown_ids).intersection(turn.get("forbidden_recommendation_ids", [])):
        failures.append("forbidden_recommendation_shown")
    adapter_semantic_pass = not any(reason.startswith(("query:", "forbidden_query:")) for reason in failures)
    policy_state_pass = not any(reason.startswith("state:") for reason in failures)
    recommendation_contract_pass = not any(
        reason in {"missing_recommendation", "unexpected_recommendation", "literal_title_not_exact"} for reason in failures
    )
    return {
        "expected_state": turn["expected_state"],
        "actual_state": response.state,
        "expected_query": turn.get("expected_query", {}),
        "final_query": query,
        "shown_ids": shown_ids,
        "unknown_ids": unknown,
        "spoken_constraint_violations": violations,
        "recommendation_evidence": [
            {
                "item_id": recommendation.item.id,
                "title": recommendation.item.title,
                "claim_texts": list(recommendation.claim_texts),
                "evidence": [evidence.model_dump(mode="json") for evidence in recommendation.evidence],
            }
            for recommendation in response.recommendations
        ],
        "recommendation_exists": bool(shown_ids),
        "latency_ms": response.latency_ms,
        "llm_calls": response.llm_calls,
        "llm_tokens": response.llm_tokens,
        "clarification_count": response.clarification_count,
        "clarification_slot": response.clarification_slot,
        "message": response.message,
        "response_mode": response.mode,
        "warnings": response.warnings,
        "adapter_semantic_pass": adapter_semantic_pass,
        "policy_state_pass": policy_state_pass,
        "recommendation_contract_pass": recommendation_contract_pass,
        "raw_interpretation": jsonable(getattr(response, "structured_request", None)),
    }, failures


def assess_observed_turn(turn: dict[str, Any], observed: dict[str, Any], by_id: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    """Reapply only frozen semantic checks to an already captured response."""
    failures: list[str] = []
    state = observed["actual_state"]
    query = observed["final_query"]
    shown_ids = observed["shown_ids"]
    if state != turn["expected_state"]:
        failures.append(f"state:{state}!={turn['expected_state']}")
    for field, expected in turn.get("expected_query", {}).items():
        if query.get(field) != expected:
            failures.append(f"query:{field}={query.get(field)!r}!={expected!r}")
    for field, forbidden in turn.get("forbidden_query", {}).items():
        if query.get(field) == forbidden:
            failures.append(f"forbidden_query:{field}={forbidden!r}")
    if turn["recommendation_exists"] and not shown_ids:
        failures.append("missing_recommendation")
    if not turn["recommendation_exists"] and shown_ids:
        failures.append("unexpected_recommendation")
    unknown = [item_id for item_id in shown_ids if item_id not in by_id]
    if unknown:
        failures.append("unknown_item_id")
    spoken = SpokenConstraints.model_validate(turn.get("spoken", {}))
    violations = [item_id for item_id in shown_ids if item_id in by_id and not satisfies_spoken(by_id[item_id], spoken)[0]]
    if violations:
        failures.append("spoken_constraint_violation")
    if turn.get("exact_recommendation_ids") is not None and shown_ids != turn["exact_recommendation_ids"]:
        failures.append("literal_title_not_exact")
    if set(shown_ids).intersection(turn.get("forbidden_recommendation_ids", [])):
        failures.append("forbidden_recommendation_shown")
    return {
        "expected_state": turn["expected_state"],
        "actual_state": state,
        "expected_query": turn.get("expected_query", {}),
        "final_query": query,
        "shown_ids": shown_ids,
        "unknown_ids": unknown,
        "spoken_constraint_violations": violations,
        "recommendation_exists": bool(shown_ids),
        "legacy_label_note": turn.get("legacy_label_note"),
    }, failures


def run_dialogue(
    dialogue: dict[str, Any],
    data: dict[str, Any],
    catalog: list,
    by_id: dict[str, object],
    *,
    model: str,
    mode: str,
    timeout: float,
    implementation: str = "baseline-v1",
    workflow_config: dict[str, object] | None = None,
) -> dict[str, object]:
    client = OllamaClient(model=model, base_url=data["model"]["base_url"], timeout=timeout)
    structured_calls, parse_calls = install_capture(client)
    agent = build_agent(
        provider=DemoProvider(list(catalog)),
        mode=mode,
        llm=client,
        question_policy=data["configuration"]["question_policy"],
        max_questions=data["configuration"]["max_questions"],
        implementation=implementation,
        workflow_config=workflow_config,
    )
    session_id = None
    results: list[dict[str, object]] = []
    failures: list[str] = []
    try:
        for index, turn in enumerate(dialogue["turns"]):
            before_parse, before_structured = len(parse_calls), len(structured_calls)
            response = agent.chat(ChatRequest(user_id=f"llm-first-{dialogue['id']}", message=turn["user"], session_id=session_id))
            session_id = response.session_id
            result, turn_failures = assess_turn(turn, response, by_id)
            turn_structured = structured_calls[before_structured:]
            result.update(
                {
                    "index": index,
                    "user": turn["user"],
                    "raw_parse": parse_calls[before_parse:],
                    "structured_calls": turn_structured,
                    "raw_extraction": raw_extraction_evidence(turn_structured),
                    "failures": turn_failures,
                }
            )
            results.append(result)
            failures.extend(f"turn{index}:{reason}" for reason in turn_failures)
    except Exception as exc:
        failures.append(f"runner_error:{type(exc).__name__}:{exc}")
    return {"id": dialogue["id"], "tags": dialogue["tags"], "turns": results, "failures": failures, "success": not failures}


def stage_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    turns = [turn for row in rows for turn in row["turns"]]
    evidence = [turn["raw_extraction"] for turn in turns]
    adapter_pass = sum(bool(turn["adapter_semantic_pass"]) for turn in turns)
    policy_eligible = [turn for turn in turns if turn["adapter_semantic_pass"]]
    return {
        "turns": len(turns),
        "raw_extraction": {
            "structured_calls": sum(item["structured_calls"] for item in evidence),
            "validated_outputs": sum(item["validated_outputs"] for item in evidence),
            "errors": sum(item["errors"] for item in evidence),
            "issue_count": sum(item["issue_count"] for item in evidence),
            "clarification_required": sum(item["clarification_required"] for item in evidence),
            "update_fields": dict(
                sorted(Counter(field for item in evidence for field, count in item["update_fields"].items() for _ in range(count)).items())
            ),
        },
        "adapter": {"semantic_pass_turns": adapter_pass, "semantic_mismatch_turns": len(turns) - adapter_pass},
        "policy": {
            "eligible_after_adapter": len(policy_eligible),
            "expected_state_pass": sum(bool(turn["policy_state_pass"]) for turn in policy_eligible),
            "recommendation_contract_pass": sum(bool(turn["recommendation_contract_pass"]) for turn in policy_eligible),
        },
        "note": "Raw evidence is descriptive: an omitted update is not automatically an error when the field is inherited or derived from a known title.",
    }


def compare(before_path: Path | None, rows: list[dict[str, object]]) -> dict[str, object] | None:
    if before_path is None:
        return None
    before = json.loads(before_path.read_text(encoding="utf-8"))
    old_rows = {row["id"]: row for row in before.get("dialogues", [])}
    examples = []
    for row in rows:
        old = old_rows.get(row["id"])
        if old is None:
            continue
        before_turns = old.get("turns", [])
        after_turns = row.get("turns", [])
        turn_changes = []
        for index, (before_turn, after_turn) in enumerate(zip(before_turns, after_turns, strict=False)):
            # A corrected-fixture re-score preserves the captured turn below
            # ``reused_observation``; compare the observation, not its scoring wrapper.
            before_observation = before_turn.get("reused_observation", before_turn)
            old_state = before_observation.get("actual_state")
            new_state = after_turn.get("actual_state")
            old_query = before_observation.get("final_query")
            new_query = after_turn.get("final_query")
            if old_state != new_state or old_query != new_query:
                turn_changes.append(
                    {
                        "index": index,
                        "before_state": old_state,
                        "after_state": new_state,
                        "before_query": old_query,
                        "after_query": new_query,
                    }
                )
        if old.get("success") != row["success"] or old.get("failures") != row.get("failures") or turn_changes:
            examples.append(
                {
                    "id": row["id"],
                    "before_success": old.get("success"),
                    "after_success": row["success"],
                    "before_failures": old.get("failures"),
                    "after_failures": row.get("failures"),
                    "turn_changes": turn_changes,
                }
            )
    return {
        "before_path": str(before_path),
        "before_report_sha256": before.get("report_sha256"),
        "changed_full_dialogues": examples[:8],
        "changed_dialogue_count": len(examples),
    }


def rescore(source_path: Path, cohort_path: Path, data: dict[str, Any]) -> dict[str, object]:
    """Score saved before responses under a corrected fixture without inference."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    catalog = generate_catalog(int(data["catalog"]["seed"]))
    by_id = {item.id: item for item in catalog}
    source_by_id = {row["id"]: row for row in source.get("dialogues", [])}
    rows: list[dict[str, object]] = []
    for dialogue in data["dialogues"]:
        original = source_by_id.get(dialogue["id"])
        if original is None or len(original.get("turns", [])) != len(dialogue["turns"]):
            raise ValueError(f"source report does not contain matching turns for {dialogue['id']}")
        turns, failures = [], []
        for index, (expected, observed) in enumerate(zip(dialogue["turns"], original["turns"], strict=True)):
            scored, turn_failures = assess_observed_turn(expected, observed, by_id)
            turns.append(
                {"index": index, "user": expected["user"], "reused_observation": observed, "rescore": scored, "failures": turn_failures}
            )
            failures.extend(f"turn{index}:{reason}" for reason in turn_failures)
        rows.append({"id": dialogue["id"], "tags": dialogue["tags"], "turns": turns, "failures": failures, "success": not failures})
    counts = Counter(failure.split(":", 2)[-1] for row in rows for failure in row["failures"])
    result: dict[str, object] = {
        "schema_version": 1,
        "label": "before_rescored",
        "complete": bool(source.get("complete")),
        "rescore_only": True,
        "source_report": {
            "path": str(source_path),
            "report_sha256": source.get("report_sha256"),
            "cohort_sha256": source.get("cohort", {}).get("sha256"),
        },
        "cohort": {
            "dataset_id": data["dataset_id"],
            "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            "dialogues": len(rows),
            "origin": data["origin"],
            "split": data["split"],
        },
        "model_identity_before": source.get("model_identity_before"),
        "model_identity_after": source.get("model_identity_after"),
        "source": source_revision(),
        "dialogues": rows,
        "summary": {
            "full_dialogues": len(rows),
            "full_dialog_successes": sum(row["success"] for row in rows),
            "failure_counts": dict(sorted(counts.items())),
        },
        "limitations": [
            "No Ollama call was made: saved v1 responses were rescored only under corrected v2 semantic criteria.",
            *data["limitations"],
        ],
    }
    result["report_sha256"] = sha256({key: value for key, value in result.items() if key != "report_sha256"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--mode", choices=("ollama", "rules"), default="ollama")
    parser.add_argument("--implementation", choices=("baseline-v1", "workflow-v2"), default="baseline-v1")
    parser.add_argument("--workflow-config", type=Path, help="versioned YAML options applied only to workflow-v2")
    parser.add_argument("--model", help="effective Ollama model; changing it is the only permitted experiment variable")
    parser.add_argument("--expected-digest", help="required pinned SHA-256 digest when --model overrides the frozen cohort model")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--rescore-report", type=Path, help="offline semantic rescore of a saved report; performs zero inference")
    parser.add_argument("--refresh-comparison", type=Path, help="offline rebuild of comparison in a saved report; performs zero inference")
    parser.add_argument(
        "--dialogue-id", action="append", dest="dialogue_ids", help="whole frozen DEV dialogue ID; repeat for a focused subset"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    original_data = load_and_validate(args.cohort)
    data, scope = select_dialogues(original_data, args.dialogue_ids)
    selected_model = select_model(data, args.model, args.expected_digest)
    if args.validate_only:
        print(
            canonical(
                {
                    "valid": True,
                    "dialogues": len(data["dialogues"]),
                    "experiment_scope": scope,
                    "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
                    "model_selection": selected_model,
                }
            )
        )
        return
    if args.rescore_report:
        result = rescore(args.rescore_report, args.cohort, data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            canonical(
                {
                    "complete": result["complete"],
                    "full_dialog_successes": result["summary"]["full_dialog_successes"],
                    "rescore_only": True,
                    "output": str(args.output),
                }
            )
        )
        return
    if args.refresh_comparison:
        if args.compare is None:
            raise ValueError("--refresh-comparison requires --compare")
        result = json.loads(args.refresh_comparison.read_text(encoding="utf-8"))
        result["comparison"] = compare(args.compare, result["dialogues"])
        result["comparison_refresh"] = {
            "mode": "offline",
            "reason": "comparison against corrected v2 rescore",
            "refreshed_at_utc": datetime.now(UTC).isoformat(),
        }
        result["report_sha256"] = sha256({key: value for key, value in result.items() if key != "report_sha256"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(canonical({"complete": result["complete"], "comparison_refresh_only": True, "output": str(args.output)}))
        return
    if args.mode != "ollama":
        raise ValueError("frozen before/after product measurements require --mode ollama")
    workflow_config = None
    if args.workflow_config:
        workflow_config = yaml.safe_load(args.workflow_config.read_text(encoding="utf-8")) or {}
        if not isinstance(workflow_config, dict):
            raise ValueError("workflow config must be a YAML mapping")
    if workflow_config and args.implementation != "workflow-v2":
        raise ValueError("--workflow-config requires --implementation workflow-v2")
    configuration = {
        **data["configuration"],
        "mode": args.mode,
        "timeout_s": args.timeout,
        "implementation": args.implementation,
        "workflow_config_sha256": hashlib.sha256(args.workflow_config.read_bytes()).hexdigest() if args.workflow_config else None,
    }
    source = source_revision()
    identity_before = model_identity(data["model"]["base_url"], selected_model["effective_model"], timeout=args.timeout)
    if identity_before["digest"] != selected_model["expected_digest"]:
        raise ValueError("Ollama digest differs from the explicitly pinned effective model before inference")
    catalog = generate_catalog(int(data["catalog"]["seed"]))
    by_id = {item.id: item for item in catalog}
    rows = [
        run_dialogue(
            dialogue,
            data,
            catalog,
            by_id,
            model=selected_model["effective_model"],
            mode=args.mode,
            timeout=args.timeout,
            implementation=args.implementation,
            workflow_config=workflow_config,
        )
        for dialogue in data["dialogues"]
    ]
    identity_after = model_identity(data["model"]["base_url"], selected_model["effective_model"], timeout=args.timeout)
    complete = identity_after == identity_before and all(
        not any(failure.startswith("runner_error:") for failure in row["failures"]) for row in rows
    )
    failure_counts = Counter(failure.split(":", 2)[-1] for row in rows for failure in row["failures"])
    result: dict[str, object] = {
        "schema_version": 1,
        "label": args.label,
        "complete": complete,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort": {
            "path": str(args.cohort),
            "sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
            "dialogues": len(rows),
            "origin": data["origin"],
            "split": data["split"],
        },
        "model_identity_before": identity_before,
        "model_identity_after": identity_after,
        "model_selection": selected_model,
        "configuration": configuration,
        "component_identity": component_identity(args.implementation),
        "source": source,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "dialogues": rows,
        "summary": {
            "full_dialogues": len(rows),
            "full_dialog_successes": sum(row["success"] for row in rows),
            "failure_counts": dict(sorted(failure_counts.items())),
            "stages": stage_summary(rows),
        },
        "comparison": compare(args.compare, rows),
        "experiment_scope": {
            "cohort_role": "fixed public DEV-only synthetic measurement fixture",
            **scope,
            "original_cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
            "interpretation": "A paired comparison requires the same cohort, decoding, prompt/schema/pipeline and product-source hashes; it does not establish production or traffic performance.",
        },
        "limitations": [
            *data["limitations"],
            "The frozen cohort is reused as a fixed fixture for this controlled comparison; it is not a new independent model-evaluation split.",
        ],
        "command": ["python", "-m", "scripts.evaluate_llm_first_product", *sys.argv[1:]],
    }
    result["report_sha256"] = sha256({key: value for key, value in result.items() if key != "report_sha256"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "complete": complete,
                "full_dialog_successes": result["summary"]["full_dialog_successes"],
                "full_dialogues": len(rows),
                "output": str(args.output),
            }
        )
    )
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
