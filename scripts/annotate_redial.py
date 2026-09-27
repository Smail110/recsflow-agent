"""Independently annotate public ReDial prefixes with local models and exact replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from evals import redial_annotation as annotation

VERSION = "redial-annotation-transport-v1.1"
ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def immutable_json(path, value):
    payload = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace immutable file: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)
    return digest(payload)


def read_open(path):
    resolved = path.resolve()
    if any(marker in str(resolved).casefold() for marker in ("blind", "final_holdout", "final-holdout")):
        raise ValueError("Closed-data path is forbidden")
    return resolved.read_bytes()


def load_inputs(manifest_path, *, input_file="inputs.en.jsonl", limit=None, split=None):
    if Path(input_file).name != input_file:
        raise ValueError("Input filename must be a direct manifest child")
    raw_manifest = read_open(manifest_path)
    manifest = json.loads(raw_manifest)
    expected = manifest["files"][input_file]
    raw = read_open(manifest_path.parent / input_file)
    if digest(raw) != expected["sha256"] or len(raw) != expected["bytes"]:
        raise ValueError("Input file size/SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    seen, valid = set(), []
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Duplicate or missing input row ID")
        seen.add(key)
        # Validate every row before selecting a limit or split; future annotation
        # files named elsewhere in the manifest are deliberately never opened.
        annotation.annotation_payload(row)
        valid.append(row)
    if split is not None:
        valid = [row for row in valid if row.get("split") == split]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("Limit must be positive")
        valid = valid[:limit]
    if not valid:
        raise ValueError("No selected inputs")
    return valid, {
        "manifest_sha256": digest(raw_manifest),
        "input_file": input_file,
        "input_file_sha256": digest(raw),
        "total_rows_verified": len(rows),
        "selected_ids": [r["id"] for r in valid],
        "limit": limit,
        "split": split,
    }


def load_config(path):
    raw = read_open(path)
    config = json.loads(raw)
    parsed = urlparse(config["base_url"])
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Only an explicit unauthenticated local Ollama URL is allowed")
    if not 0 < config.get("timeout_s", 120) <= 600:
        raise ValueError("Invalid timeout")
    for role in ("primary", "verifier"):
        spec = config[role]
        if not isinstance(spec.get("model"), str) or not re.fullmatch(r"[a-f0-9]{64}", spec.get("digest", "")):
            raise ValueError("Each role needs an explicit model and pinned digest")
        if not isinstance(spec.get("options"), dict) or set(spec["options"]) != {"seed", "temperature", "num_predict"}:
            raise ValueError("Explicit seed, temperature and num_predict are required")
        options = spec["options"]
        if (
            type(options["seed"]) is not int
            or options["temperature"] != 0
            or type(options["num_predict"]) is not int
            or not 1 <= options["num_predict"] <= 8192
        ):
            raise ValueError("Invalid deterministic generation options")
        if spec.get("think") is not None and type(spec["think"]) is not bool:
            raise ValueError("think must be boolean or null (omit wire field)")
    if config["primary"]["digest"] == config["verifier"]["digest"]:
        raise ValueError("Independent annotators require distinct pinned model digests")
    return config, digest(raw)


def identity(client, spec):
    response = client.get("/api/tags")
    response.raise_for_status()
    matches = [row for row in response.json().get("models", []) if row.get("name") == spec["model"]]
    if len(matches) != 1 or matches[0].get("digest") != spec["digest"]:
        raise ValueError("Configured model digest does not match local Ollama tags")
    version = client.get("/api/version")
    version.raise_for_status()
    if not isinstance(version.json().get("version"), str):
        raise ValueError("Missing Ollama version")
    return {
        "model": spec["model"],
        "digest": spec["digest"],
        "ollama_version": version.json()["version"],
        "details": matches[0].get("details"),
    }


def wire_schema(value):
    """Ollama grammar cannot compile large bounded string repetitions.

    Remove only maxLength from generation grammar; the untouched Pydantic
    schema still enforces all semantic validation limits after generation.
    """
    if isinstance(value, dict):
        return {key: wire_schema(item) for key, item in value.items() if key != "maxLength"}
    if isinstance(value, list):
        return [wire_schema(item) for item in value]
    return value


def request_material(prefix, role, config, source_hashes):
    payload = annotation.annotation_payload(prefix)
    validation_schema = annotation.Annotation.model_json_schema()
    schema = wire_schema(validation_schema)
    prompt = annotation.build_prompt(role=role)
    spec = config[role]
    wire = {
        "model": spec["model"],
        "stream": False,
        "format": schema,
        "options": spec["options"],
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": canonical(payload).decode()}],
    }
    if spec.get("think") is not None:
        wire["think"] = spec["think"]
    return {
        "transport_version": VERSION,
        "role": role,
        "input_sha256": digest(payload),
        "wire_schema_sha256": digest(schema),
        "validation_schema_sha256": digest(validation_schema),
        "prompt_sha256": digest(prompt.encode()),
        "model": spec["model"],
        "expected_digest": spec["digest"],
        "base_url": config["base_url"],
        "source_sha256": source_hashes,
        "wire_request": wire,
    }


def analyze_response(prefix, raw):
    """Invalid model output is a stable observation, never a reason to resample."""
    if raw.get("done_reason") == "length":
        return None, {"valid": False, "errors": ["truncated_model_output"], "annotation": None}
    content = raw.get("message", {}).get("content") if isinstance(raw.get("message"), dict) else None
    if not isinstance(content, str):
        return None, {"valid": False, "errors": ["missing_message_content"], "annotation": None}
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return None, {"valid": False, "errors": ["invalid_json"], "annotation": None}
    validation = annotation.validate_annotation(prefix, parsed)
    value = annotation.Annotation.model_validate(parsed).model_dump(mode="json") if validation["valid"] else None
    return value, validation


def verified_cache(path, material, prefix):
    entry = json.loads(path.read_bytes())
    if entry.get("entry_sha256") != digest({k: v for k, v in entry.items() if k != "entry_sha256"}):
        raise ValueError("Annotation cache entry hash mismatch")
    if entry.get("request") != material or entry.get("request_sha256") != digest(material) or entry.get("status") != "complete":
        raise ValueError("Annotation cache request mismatch")
    if (
        entry.get("identity_before") != entry.get("identity_after")
        or entry.get("identity_before", {}).get("digest") != material["expected_digest"]
    ):
        raise ValueError("Annotation cache model identity mismatch")
    raw = entry["raw_response"]
    if raw.get("done") is not True or raw.get("model") != material["model"]:
        raise ValueError("Cached transport response is incomplete or has a different model")
    value, validation = analyze_response(prefix, raw)
    if value != entry["annotation"] or validation != entry["validation"] or validation["valid"] != entry["annotation_valid"]:
        raise ValueError("Cached annotation/diagnostics differ from raw response")
    return entry


def annotate_one(prefix, role, config, cache, *, mode, source_hashes, client=None):
    material = request_material(prefix, role, config, source_hashes)
    key = digest(material)
    path = cache / "entries" / f"{key}.json"
    if path.exists():
        entry = verified_cache(path, material, prefix)
        return entry["annotation"], {
            "status": "complete",
            "request_sha256": key,
            "annotation_valid": entry["annotation_valid"],
            "validation": entry["validation"],
            "cache_file_sha256": digest(path.read_bytes()),
            "cache_hit": True,
        }
    if mode == "replay":
        return None, {"status": "cache_miss", "request_sha256": key, "cache_hit": False}
    if client is None:
        raise ValueError("Record requires an explicit HTTP client")
    started = time.perf_counter()
    receipt = {
        "request": material,
        "request_sha256": key,
        "status": "failed",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "raw_response": None,
        "usage": {"prompt_tokens": None, "output_tokens": None, "total_tokens": None},
    }
    try:
        receipt["identity_before"] = identity(client, config[role])
        response = client.post("/api/chat", json=material["wire_request"])
        receipt["http_status"] = response.status_code
        receipt["raw_response_text"] = response.text
        response.raise_for_status()
        raw = response.json()
        receipt["raw_response"] = raw
        receipt["identity_after"] = identity(client, config[role])
        if receipt["identity_before"] != receipt["identity_after"]:
            raise ValueError("Model identity changed during annotation")
        if raw.get("model") != material["model"] or raw.get("done") is not True:
            raise ValueError("Unexpected model or incomplete Ollama response")
        for source, target in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "output_tokens")):
            value = raw.get(source)
            if type(value) is int and value >= 0:
                receipt["usage"][target] = value
        if all(receipt["usage"][key] is not None for key in ("prompt_tokens", "output_tokens")):
            receipt["usage"]["total_tokens"] = receipt["usage"]["prompt_tokens"] + receipt["usage"]["output_tokens"]
        value, validated = analyze_response(prefix, raw)
        receipt["validation"] = validated
        receipt["annotation_valid"] = validated["valid"]
        receipt["annotation"] = value
        receipt["status"] = "complete"
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000
    receipt["entry_sha256"] = digest(receipt)
    if receipt["status"] == "complete":
        immutable_json(path, receipt)
        return receipt["annotation"], {
            "status": "complete",
            "request_sha256": key,
            "annotation_valid": receipt["annotation_valid"],
            "validation": receipt["validation"],
            "cache_file_sha256": digest(path.read_bytes()),
            "cache_hit": False,
        }
    failure = cache / "failures" / key / f"{uuid.uuid4().hex}.json"
    immutable_json(failure, receipt)
    return None, {
        "status": "failed",
        "request_sha256": key,
        "cache_hit": False,
        "failure_receipt": str(failure),
        "failure_sha256": digest(failure.read_bytes()),
        "error_type": receipt["error"]["type"],
    }


def execute(rows, config, cache, output, *, mode, input_identity, config_sha256, client_factory=httpx.Client, progress_every=1):
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; resume by reusing cache")
    if config["primary"]["digest"] == config["verifier"]["digest"]:
        raise ValueError("Independent annotators require distinct pinned model digests")
    sources = {
        "scripts/annotate_redial.py": digest(Path(__file__).read_bytes()),
        "evals/redial_annotation.py": digest(Path(annotation.__file__).read_bytes()),
    }
    protocol = {
        "version": VERSION,
        "mode": mode,
        "input": input_identity,
        "config_sha256": config_sha256,
        "config": config,
        "source_sha256": sources,
        "scheduling": "all primary, then all verifier",
        "validation_schema_sha256": digest(annotation.Annotation.model_json_schema()),
        "wire_schema_sha256": digest(wire_schema(annotation.Annotation.model_json_schema())),
        "wire_schema_transform": "recursively remove maxLength only; semantic validation unchanged",
        "circuit_breaker": {
            "consecutive_transport_failures": 3,
            "scope": "entire run; cached observations retained, remaining network calls unattempted",
        },
        "label_origin": "automatic_silver",
        "no_agent_outputs": True,
        "no_human_gold": True,
    }
    output.mkdir(parents=True)
    immutable_json(output / "protocol.json", protocol)
    results, statuses = {}, {}
    consecutive_failures, circuit_open = 0, False
    for role in ("primary", "verifier"):
        client = (
            client_factory(base_url=config["base_url"], timeout=config.get("timeout_s", 120), trust_env=False)
            if mode == "record" and not circuit_open
            else None
        )
        try:
            for index, row in enumerate(rows, 1):
                if circuit_open:
                    try:
                        value, status = annotate_one(row, role, config, cache, mode="replay", source_hashes=sources)
                        if status["status"] == "cache_miss":
                            status.update(status="unattempted", reason="transport_circuit_open")
                    except Exception as exc:
                        value, status = (
                            None,
                            {
                                "status": "cache_error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "cache_hit": False,
                                "request_sha256": digest(request_material(row, role, config, sources)),
                            },
                        )
                else:
                    try:
                        value, status = annotate_one(row, role, config, cache, mode=mode, source_hashes=sources, client=client)
                    except Exception as exc:
                        value, status = (
                            None,
                            {
                                "status": "cache_error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "cache_hit": False,
                                "request_sha256": digest(request_material(row, role, config, sources)),
                            },
                        )
                    if status["status"] == "failed":
                        consecutive_failures += 1
                        circuit_open = consecutive_failures >= 3
                    elif status["status"] == "complete":
                        consecutive_failures = 0
                results[(row["id"], role)] = value
                statuses[(row["id"], role)] = status
                if progress_every > 0 and (index % progress_every == 0 or index == len(rows)):
                    print(
                        json.dumps(
                            {
                                "progress": {
                                    "role": role,
                                    "completed_rows": index,
                                    "total_rows": len(rows),
                                    "status": status["status"],
                                    "annotation_valid": status.get("annotation_valid"),
                                    "cache_hit": status["cache_hit"],
                                }
                            }
                        ),
                        flush=True,
                    )
        finally:
            if client is not None:
                client.close()
    outputs = []
    for row in rows:
        primary, verifier = (results[(row["id"], role)] for role in ("primary", "verifier"))
        decision = (
            annotation.consensus(row, primary, verifier)
            if primary is not None and verifier is not None
            else {
                "status": "REVIEW",
                "label_origin": "automatic_silver",
                "eligible_for_scoring": False,
                "eligible_scope": "assertion_extraction_only",
                "unknown": True,
                "reason": "invalid_annotation"
                if all(statuses[(row["id"], role)]["status"] == "complete" for role in ("primary", "verifier"))
                else "transport_unavailable",
                "annotation": None,
            }
        )
        outputs.append(
            {
                "id": row["id"],
                "input_sha256": digest(annotation.annotation_payload(row)),
                "split": row.get("split"),
                "consensus": decision,
                "annotation_status": {
                    role: {
                        "transport_complete": statuses[(row["id"], role)]["status"] == "complete",
                        "attempted": statuses[(row["id"], role)]["status"] != "unattempted",
                        "status": statuses[(row["id"], role)]["status"],
                        "annotation_valid": statuses[(row["id"], role)].get("annotation_valid"),
                    }
                    for role in ("primary", "verifier")
                },
                "annotation_diagnostics": {role: statuses[(row["id"], role)].get("validation") for role in ("primary", "verifier")},
                "request_sha256": {role: statuses[(row["id"], role)]["request_sha256"] for role in ("primary", "verifier")},
            }
        )
    if any(digest((ROOT / path).read_bytes()) != expected for path, expected in sources.items()):
        raise ValueError("Source changed during annotation; receipts preserved but no dataset published")
    payload = b"".join(canonical(row) + b"\n" for row in outputs)
    with (output / "annotations.jsonl").open("xb") as handle:
        handle.write(payload)
    receipt = {
        "status": "complete" if all(v["status"] == "complete" for v in statuses.values()) else "incomplete",
        "annotations_sha256": digest(payload),
        "rows": len(outputs),
        "decisions": dict(Counter(row["consensus"]["status"] for row in outputs)),
        "valid_annotations": sum(v.get("annotation_valid") is True for v in statuses.values()),
        "invalid_complete_annotations": sum(v["status"] == "complete" and v.get("annotation_valid") is False for v in statuses.values()),
        "circuit_open": circuit_open,
        "unattempted_role_calls": sum(v["status"] == "unattempted" for v in statuses.values()),
        "eligible_for_scoring": sum(row["consensus"]["eligible_for_scoring"] for row in outputs),
        "calls": [{"id": key, "role": role, **value} for (key, role), value in statuses.items()],
        "protocol_file_sha256": digest((output / "protocol.json").read_bytes()),
        "unknown_is_not_zero": True,
        "scope": "automatic assertion/extraction silver only; no recommendation success score",
    }
    immutable_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("record", "replay"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split")
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    config, config_hash = load_config(args.config)
    rows, inputs = load_inputs(
        args.input_manifest, input_file=config.get("input_file", "inputs.en.jsonl"), limit=args.limit, split=args.split
    )
    receipt = execute(
        rows,
        config,
        args.cache_dir,
        args.output_dir,
        mode=args.mode,
        input_identity=inputs,
        config_sha256=config_hash,
        progress_every=args.progress_every,
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "calls"}))
    if receipt["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
