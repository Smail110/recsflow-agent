"""Run a manifest-pinned open CONTRACT cohort without sending labels to the agent."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import platform
import sys
import textwrap
import time
from pathlib import Path

from evals.observable_product import canonical_hash, compare_reports, evaluate_report, safe_read_json, validate_cohort

from recagent.catalog import catalog_sha256, generate_catalog
from recagent.factory import build_agent
from recagent.models import ChatRequest
from recagent.parsing import OllamaClient
from recagent.providers import DemoProvider
from scripts.evaluate_llm_first_product import install_capture, jsonable, model_identity, source_revision
from scripts.evaluate_observable_product import write_json


def transport_configuration():
    """Read literal wire options from the actual client method, without a call."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(OllamaClient.structured)))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "json" and isinstance(node.value, ast.Dict):
            values = {ast.literal_eval(k): v for k, v in zip(node.value.keys, node.value.values, strict=True)}
            return {key: ast.literal_eval(values[key]) for key in ("options", "think", "stream")}
    raise ValueError("Ollama transport settings are not statically observable")


def make_agent(data, catalog, mode, implementation, client):
    return build_agent(
        provider=DemoProvider(list(catalog)),
        mode=mode,
        llm=client,
        question_policy=data["configuration"]["question_policy"],
        max_questions=data["configuration"]["max_questions"],
        implementation=implementation,
    )


def run_cases(data, catalog, *, mode, implementation, client_factory, rows=None):
    """Only public catalog, configuration and user text cross the product boundary."""
    rows = [] if rows is None else rows
    for case in data["dialogues"]:
        client = client_factory()
        structured, parsed = install_capture(client)
        agent = make_agent(data, catalog, mode, implementation, client)
        session_id, turns = None, []
        rows.append({"id": case["id"], "turns": turns})
        for index, turn in enumerate(case["turns"]):
            before_structured, before_parsed = len(structured), len(parsed)
            started = time.perf_counter()
            response = agent.chat(ChatRequest(user_id=f"contract-{case['id']}", message=turn["user"], session_id=session_id))
            elapsed = (time.perf_counter() - started) * 1000
            session_id = response.session_id
            raw = response.model_dump(mode="json")
            turns.append(
                {
                    "index": index,
                    "user": turn["user"],
                    "actual_state": response.state,
                    "final_query": response.query.model_dump(mode="json"),
                    "shown_ids": [r.item.id for r in response.recommendations],
                    "recommendation_evidence": [
                        {
                            "item_id": r.item.id,
                            "title": r.item.title,
                            "claim_texts": list(r.claim_texts),
                            "evidence": [e.model_dump(mode="json") for e in r.evidence],
                        }
                        for r in response.recommendations
                    ],
                    "response": raw,
                    "message": response.message,
                    "structured_calls": jsonable(structured[before_structured:]),
                    "raw_parse": jsonable(parsed[before_parsed:]),
                    "wall_latency_ms": elapsed,
                    "llm_calls": response.llm_calls,
                    "llm_tokens": response.llm_tokens,
                    "latency_ms": response.latency_ms,
                    "response_mode": response.mode,
                }
            )
        # No invented legacy success for a cohort which never used that scorer.
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/product_contract_ru_v1.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("rules", "ollama"), required=True)
    parser.add_argument("--implementation", choices=("baseline-v1", "workflow-v2"), required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    data, cohort_hash = safe_read_json(args.cohort)
    manifest, manifest_hash = safe_read_json(args.manifest)
    validate_cohort(data)
    if (
        data["split"] != "contract"
        or manifest["sha256"] != cohort_hash
        or manifest["seed"] != data["catalog"]["seed"]
        or manifest["dialogs"] != len(data["dialogues"])
        or manifest["turns"] != sum(len(d["turns"]) for d in data["dialogues"])
    ):
        raise ValueError("Contract manifest mismatch")
    catalog = generate_catalog(data["catalog"]["seed"])
    catalog_hash = catalog_sha256(data["catalog"]["seed"])
    by_id = {i.id: i for i in catalog}
    baseline = None
    if args.compare:
        baseline, _ = safe_read_json(args.compare)
        evaluate_report(data, baseline, cohort_sha256=cohort_hash, catalog=by_id)
        if baseline.get("catalog_sha256") != catalog_hash:
            raise ValueError("Comparison catalog hash mismatch")
    source = source_revision()
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        *sorted((root / "src/recagent").rglob("*.py")),
        Path(__file__),
        root / "evals/observable_product.py",
        root / "evals/oracle.py",
        root / "evals/explanations.py",
        root / "scripts/build_product_contract.py",
    ]
    source["files_sha256"].update({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    assembly = make_agent(data, catalog, args.mode, args.implementation, OllamaClient())
    workflow_config = getattr(assembly, "workflow_config", None)
    config = {
        "mode": args.mode,
        "implementation": args.implementation,
        "timeout_s": args.timeout,
        "base_url": args.base_url,
        "question_policy": assembly.question_policy,
        "max_questions": assembly.max_questions,
        "history": "cold_start",
        "max_calls": assembly.max_calls,
        "max_tokens": assembly.max_tokens,
        "workflow_config": workflow_config,
        "workflow_config_sha256": canonical_hash(workflow_config) if workflow_config is not None else None,
        "ollama_transport": transport_configuration() if args.mode == "ollama" else None,
    }
    if data["configuration"]["history"] != "cold_start":
        raise ValueError("Runner supports explicitly cold-start cohorts only")

    def identity():
        return (
            model_identity(args.base_url, data["model"]["name"], timeout=args.timeout)
            if args.mode == "ollama"
            else {"provider": "rules", "model": None}
        )

    before = identity()
    if args.mode == "ollama" and before["digest"] != data["model"]["expected_digest"]:
        raise ValueError("Model digest differs from frozen contract")
    protocol = {
        "cohort_sha256": cohort_hash,
        "manifest_sha256": manifest_hash,
        "catalog_sha256": catalog_hash,
        "configuration": config,
        "source": source,
        "model_identity_before": before,
        "command": ["python", "-m", "scripts.evaluate_product_contract", *sys.argv[1:]],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "limitations": data["limitations"],
    }
    protocol_path = args.output.with_suffix(".protocol.json")
    if args.output.exists() or protocol_path.exists():
        raise ValueError("Refusing to overwrite an existing run")
    write_json(protocol_path, protocol)
    result = {**protocol, "complete": False, "cohort": {"sha256": cohort_hash}, "dialogues": []}
    try:
        result["dialogues"] = run_cases(
            data,
            catalog,
            mode=args.mode,
            implementation=args.implementation,
            client_factory=lambda: OllamaClient(model=data["model"]["name"], base_url=args.base_url, timeout=args.timeout),
            rows=result["dialogues"],
        )
        result["model_identity_after"] = identity()
        result["complete"] = result["model_identity_after"] == before
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["report_sha256"] = canonical_hash(result)
    write_json(args.output, result)
    if not result["complete"]:
        raise SystemExit(2)
    evaluation = evaluate_report(data, result, cohort_sha256=cohort_hash, catalog=by_id)
    if baseline is not None:
        # This is explicitly a system comparison: rules vs LLM may differ in
        # implementation/model. All deltas are retained rather than hidden.
        evaluation["comparison"] = compare_reports(
            data, baseline, result, cohort_sha256=cohort_hash, catalog=by_id, require_matching_run_configuration=False
        )
        evaluation["system_comparison"] = {
            "baseline_configuration": baseline["configuration"],
            "candidate_configuration": config,
            "causal_single_variable_claim": False,
        }
    write_json(args.output.with_suffix(".evaluation.json"), evaluation)
    print(evaluation["lanes"])


if __name__ == "__main__":
    main()
