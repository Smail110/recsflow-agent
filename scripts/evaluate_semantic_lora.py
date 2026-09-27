"""Evaluate base or LoRA Ollama tags on the frozen schema-conditioned protocol."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from recagent.interpretation import LLMRequestInterpreter, StructuredRequest
from scripts.semantic_lora_common import (
    ROOT,
    canonical,
    file_sha256,
    load_config,
    randomize_enums,
    read_jsonl,
    remove_schema,
    rename_fields,
    resolve,
    score_request,
    seen_schema_holdout,
    select_cases,
    sha256_bytes,
    summarize,
    verify_frozen_protocol,
)


class EvaluationOllamaClient:
    """Production-equivalent request body with experiment-only model selection and telemetry."""

    def __init__(self, model: str, base_url: str, timeout: float, options: dict[str, Any]):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = options
        self.last_meta: dict[str, Any] = {}

    def structured(self, schema, system: str, payload: dict) -> tuple[StructuredRequest, int]:
        schema_json = schema.model_json_schema()
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema_json,
            "options": self.options,
            "messages": [
                {
                    "role": "system",
                    "content": system + "\nJSON schema: " + json.dumps(schema_json, ensure_ascii=False),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.post(self.base_url + "/api/chat", json=body)
            response.raise_for_status()
            raw = response.json()
        message = raw.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        thinking = message.get("thinking", "") if isinstance(message, dict) else ""
        self.last_meta = {
            "body_sha256_without_model": sha256_bytes(canonical(body | {"model": "<MODEL>"}).encode()),
            "done": raw.get("done"),
            "done_reason": raw.get("done_reason"),
            "prompt_eval_count": int(raw.get("prompt_eval_count", 0)),
            "eval_count": int(raw.get("eval_count", 0)),
            "prompt_eval_seconds": raw.get("prompt_eval_duration", 0) / 1e9,
            "eval_seconds": raw.get("eval_duration", 0) / 1e9,
            "total_seconds": raw.get("total_duration", 0) / 1e9,
            "wall_seconds": time.perf_counter() - started,
            "content_chars": len(content) if isinstance(content, str) else 0,
            "content_sha256": sha256_bytes(content.encode()) if isinstance(content, str) else None,
            "thinking_chars": len(thinking) if isinstance(thinking, str) else 0,
        }
        if not isinstance(content, str) or not content:
            raise ValueError("Ollama response has empty message.content")
        result = schema.model_validate_json(content)
        return result, self.last_meta["prompt_eval_count"] + self.last_meta["eval_count"]


def model_identity(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, trust_env=False) as client:
        tags = client.get("/api/tags")
        version = client.get("/api/version")
        tags.raise_for_status()
        version.raise_for_status()
    matches = [item for item in tags.json().get("models", []) if item.get("name") == model]
    if len(matches) != 1:
        raise ValueError(f"Ollama tag is not uniquely available: {model}")
    return {
        "tag": model,
        "digest": matches[0].get("digest"),
        "size": matches[0].get("size"),
        "ollama_version": version.json().get("version"),
    }


def verify_baseline(path: Path, config_hash: str, evaluator_hash: str, blind_hash: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    integrity = report.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(report).encode()):
        raise ValueError("baseline report integrity mismatch")
    report["report_sha256"] = integrity
    if report["arm"] != "base" or not report["complete"]:
        raise ValueError("LoRA evaluation requires a complete frozen base report")
    if report["protocol"]["config_sha256"] != config_hash:
        raise ValueError("baseline config hash mismatch")
    if report["protocol"]["evaluator_sha256"] != evaluator_hash:
        raise ValueError("baseline evaluator hash mismatch")
    common_hash = file_sha256(Path(__file__).with_name("semantic_lora_common.py"))
    if report["protocol"].get("common_sha256") != common_hash:
        raise ValueError("baseline common scorer/transform hash mismatch")
    if report["protocol"]["blind_sha256"] != blind_hash:
        raise ValueError("baseline blind hash mismatch")
    return report


def verify_conversion_gate(path: Path, config_hash: str, model_digest: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    integrity = report.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(report).encode()):
        raise ValueError("conversion gate report integrity mismatch")
    if report["status"] != "PASS" or report["config_sha256"] != config_hash:
        raise ValueError("LoRA blind evaluation requires a passing conversion gate")
    if report["ollama_model"]["digest"] != model_digest:
        raise ValueError("conversion gate model digest mismatch")
    report["report_sha256"] = integrity
    return report


def evaluate_rows(rows: list[dict], client: EvaluationOllamaClient) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        client.last_meta = {}
        incoming = row["input"]
        try:
            request, _ = LLMRequestInterpreter(client, incoming["domain"]).interpret(
                incoming["message"],
                incoming.get("previous", {}),
                pending_question=incoming.get("pending_question"),
                unresolved=incoming.get("unresolved", []),
            )
            results.append(
                {
                    "id": row["id"],
                    "schema_id": row["schema_id"],
                    "domain_id": row["domain_id"],
                    "domain_family": row["domain_family"],
                    "tags": row.get("tags", []),
                    "counterfactual_group_id": row.get("counterfactual_group_id"),
                    "request": request.model_dump(mode="json"),
                    "score": score_request(row, request),
                    "telemetry": client.last_meta,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": row["id"],
                    "schema_id": row["schema_id"],
                    "domain_id": row["domain_id"],
                    "domain_family": row["domain_family"],
                    "tags": row.get("tags", []),
                    "counterfactual_group_id": row.get("counterfactual_group_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "telemetry": client.last_meta,
                }
            )
    return results


def counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("counterfactual_group_id"):
            groups[row["counterfactual_group_id"]].append(row)
    eligible = {key: value for key, value in groups.items() if len(value) >= 2}
    correct = 0
    for group in eligible.values():
        valid = all("score" in item and item["score"]["exact_request"] for item in group)
        signatures = {canonical(item.get("request", {})) for item in group}
        correct += int(valid and len(signatures) == len(group))
    return {
        "groups": len(eligible),
        "correct_groups": correct,
        "schema_conditional_correctness": correct / len(eligible) if eligible else 0.0,
    }


def telemetry_summary(suites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    telemetry = [row.get("telemetry", {}) for suite in suites.values() for row in suite["cases"]]
    return {
        "calls": len(telemetry),
        "prompt_tokens": sum(item.get("prompt_eval_count", 0) for item in telemetry),
        "generated_tokens": sum(item.get("eval_count", 0) for item in telemetry),
        "eval_seconds": sum(item.get("eval_seconds", 0) for item in telemetry),
        "total_backend_seconds": sum(item.get("total_seconds", 0) for item in telemetry),
        "wall_seconds": sum(item.get("wall_seconds", 0) for item in telemetry),
        "length_exhaustions": sum(item.get("done_reason") == "length" for item in telemetry),
        "empty_outputs": sum(item.get("content_chars", 0) == 0 for item in telemetry),
        "reasoning_tokens": None,
        "reasoning_observability": "unavailable; think=false and Ollama exposes no separate token count",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "lora"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--conversion-gate-report", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config_hash = file_sha256(args.config)
    evaluator_hash = file_sha256(Path(__file__))
    protocol_seal = verify_frozen_protocol(args.config, config, Path(__file__))
    evaluation = config["evaluation"]
    model = config["base_model"]["ollama_tag"] if args.arm == "base" else evaluation["lora_ollama_tag"]
    identity = model_identity(evaluation["base_url"], model, evaluation["timeout_seconds"])
    if args.arm == "base" and identity["digest"] != config["base_model"]["ollama_digest"]:
        raise ValueError("base Ollama digest drift")
    if identity["ollama_version"] != config["base_model"]["ollama_version"]:
        raise ValueError("Ollama version drift")
    blind_hash = file_sha256(resolve(config["dataset"]["blind"]["path"]))
    if args.arm == "lora":
        if args.baseline_report is None:
            parser.error("--baseline-report is required for the lora arm")
        baseline = verify_baseline(args.baseline_report, config_hash, evaluator_hash, blind_hash)
        baseline_integrity = baseline["report_sha256"]
        if args.conversion_gate_report is None:
            parser.error("--conversion-gate-report is required for the lora arm")
        conversion_gate = verify_conversion_gate(args.conversion_gate_report, config_hash, identity["digest"])
        conversion_gate_integrity = conversion_gate["report_sha256"]
    else:
        baseline_integrity = None
        conversion_gate_integrity = None

    train = read_jsonl(resolve(config["dataset"]["train"]["path"]))
    dev = read_jsonl(resolve(config["dataset"]["dev"]["path"]))
    blind = read_jsonl(resolve(config["dataset"]["blind"]["path"]))
    _, seen = seen_schema_holdout(train, config["dataset"]["seen_schema_eval_fraction"], config["seeds"]["split"])
    seed = config["seeds"]["evaluation"]
    renamed_source = select_cases(blind, evaluation["renamed_cases"], seed, "rename")
    enum_source = select_cases(blind, evaluation["enum_randomized_cases"], seed, "enum")
    suite_rows = {
        "seen_schemas_seen_domains": seen,
        "unseen_schemas_seen_domains": dev,
        "unseen_schemas_unseen_domains": blind,
        "field_renaming_original": renamed_source,
        "field_renaming": [rename_fields(row) for row in renamed_source],
        "enum_randomization_original": enum_source,
        "enum_randomization": [randomize_enums(row) for row in enum_source],
        "without_schema": [remove_schema(row) for row in select_cases(blind, evaluation["without_schema_cases"], seed, "no-schema")],
    }
    options = {
        "temperature": evaluation["temperature"],
        "seed": evaluation["seed"],
        "num_predict": evaluation["num_predict"],
        "num_ctx": evaluation["num_ctx"],
    }
    client = EvaluationOllamaClient(model, evaluation["base_url"], evaluation["timeout_seconds"], options)
    suites = {}
    for name, rows in suite_rows.items():
        evaluated = evaluate_rows(rows, client)
        block = {"summary": summarize(evaluated, len(rows)), "cases": evaluated}
        if name == "without_schema":
            valid = [item for item in evaluated if "request" in item]
            unsafe = sum(bool(item["request"]["updates"]) for item in valid)
            block["without_schema_safety"] = {
                "valid_turns": len(valid),
                "unsafe_guess_turns": unsafe,
                "unsafe_guess_rate": unsafe / len(valid) if valid else 1.0,
                "safe_unresolved_turns": sum(not item["request"]["updates"] and bool(item["request"]["issues"]) for item in valid),
            }
        suites[name] = block
    suites["unseen_schemas_unseen_domains"]["counterfactual"] = counterfactual_summary(suites["unseen_schemas_unseen_domains"]["cases"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "complete": all(len(suite["cases"]) == suite["summary"]["turns"] for suite in suites.values()),
        "model": identity,
        "decoding": {**options, "think": False, "stream": False, "format": "StructuredRequest JSON schema"},
        "protocol": {
            "config": str(args.config),
            "config_sha256": config_hash,
            "evaluator_sha256": evaluator_hash,
            "common_sha256": file_sha256(Path(__file__).with_name("semantic_lora_common.py")),
            "production_interpretation_sha256": file_sha256(ROOT / "src/recagent/interpretation.py"),
            "structured_request_schema_sha256": sha256_bytes(canonical(StructuredRequest.model_json_schema()).encode()),
            "train_sha256": config["dataset"]["train"]["sha256"],
            "dev_sha256": config["dataset"]["dev"]["sha256"],
            "blind_sha256": blind_hash,
            "final_holdout_used": False,
            "prompt_tuned": False,
            "baseline_report_sha256": baseline_integrity,
            "conversion_gate_report_sha256": conversion_gate_integrity,
            "protocol_seal_sha256": protocol_seal["report_sha256"],
        },
        "source": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "python": platform.python_version(),
        },
        "suites": suites,
        "telemetry": telemetry_summary(suites),
        "structured_output_failures": sum(
            suite["summary"]["turns"] - suite["summary"]["valid_structured_outputs"] for suite in suites.values()
        ),
        "limitations": [
            "Synthetic schema-first data is not production traffic or human annotation.",
            "wrong_kind_proxy counts wrong-field events only on cases tagged kind because StructuredRequest has no kind field.",
            "CUDA/Ollama fixed-seed execution is not promised bit-identical.",
        ],
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "arm": args.arm,
                "complete": result["complete"],
                "model": identity,
                "output": str(args.output),
                "report_sha256": result["report_sha256"],
                "suite_summaries": {name: suite["summary"] for name, suite in suites.items()},
                "telemetry": result["telemetry"],
            }
        )
    )


if __name__ == "__main__":
    main()
