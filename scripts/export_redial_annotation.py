"""Export saved automatic-silver decisions and all review rows without relabeling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.annotate_redial import canonical, digest, load_inputs, read_open


def read_json(path):
    return json.loads(read_open(path))


def export(input_manifest, run_dir, cache_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError("Export directory exists; choose a fresh path")
    protocol_bytes = read_open(run_dir / "protocol.json")
    protocol = json.loads(protocol_bytes)
    receipt_bytes = read_open(run_dir / "receipt.json")
    receipt = json.loads(receipt_bytes)
    annotation_bytes = read_open(run_dir / "annotations.jsonl")
    if digest(protocol_bytes) != receipt["protocol_file_sha256"] or digest(annotation_bytes) != receipt["annotations_sha256"]:
        raise ValueError("Recorded protocol/annotation byte hash mismatch")
    if receipt.get("status") not in {"complete", "incomplete"}:
        raise ValueError("Unsupported receipt status")
    inputs, input_identity = load_inputs(
        input_manifest, input_file=protocol["input"]["input_file"], limit=protocol["input"]["limit"], split=protocol["input"]["split"]
    )
    if input_identity != protocol["input"]:
        raise ValueError("Input manifest/selection differs from annotation run")
    records = [json.loads(line) for line in annotation_bytes.splitlines() if line.strip()]
    if [row.get("id") for row in records] != input_identity["selected_ids"] or len(records) != receipt["rows"]:
        raise ValueError("Missing, duplicate, reordered or extra annotation rows")
    calls = {}
    for call in receipt["calls"]:
        key = (call["id"], call["role"])
        if key in calls:
            raise ValueError("Duplicate role receipt")
        calls[key] = call
    if set(calls) != {(row["id"], role) for row in inputs for role in ("primary", "verifier")}:
        raise ValueError("Missing/extra role receipts")
    silver, queue = [], []
    for prefix, record in zip(inputs, records, strict=True):
        # Hash exactly the input whitelist used by the frozen annotation pipeline.
        from evals.redial_annotation import annotation_payload

        expected_input_hash = digest(annotation_payload(prefix))
        if record.get("input_sha256") != expected_input_hash or record.get("split") != prefix.get("split"):
            raise ValueError("Annotation row differs from source prefix")
        annotations = {}
        for role in ("primary", "verifier"):
            call = calls[(record["id"], role)]
            key = record["request_sha256"][role]
            if call["request_sha256"] != key or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("Invalid role request hash")
            diagnostics = record["annotation_diagnostics"][role]
            if diagnostics != call.get("validation"):
                raise ValueError("Row diagnostics differ from role receipt")
            raw_text = None
            parsed = diagnostics.get("annotation") if diagnostics else None
            if call["status"] == "complete":
                entry_bytes = read_open(cache_dir / "entries" / f"{key}.json")
                entry = json.loads(entry_bytes)
                if digest(entry_bytes) != call["cache_file_sha256"]:
                    raise ValueError("Cached annotation byte hash mismatch")
                if entry.get("entry_sha256") != digest({k: v for k, v in entry.items() if k != "entry_sha256"}):
                    raise ValueError("Cached annotation internal hash mismatch")
                if (
                    entry.get("request_sha256") != key
                    or digest(entry["request"]) != key
                    or entry["request"].get("role") != role
                    or entry["request"].get("input_sha256") != expected_input_hash
                    or entry.get("validation") != diagnostics
                ):
                    raise ValueError("Cached annotation provenance differs from row")
                raw = entry.get("raw_response", {}).get("message", {})
                raw_text = raw.get("content") if isinstance(raw, dict) else None
            annotations[role] = {
                "transport_status": call["status"],
                "annotation_valid": call.get("annotation_valid"),
                "parsed_annotation": parsed,
                "raw_model_text": raw_text,
                "validation": diagnostics,
                "request_sha256": key,
            }
        decision = record["consensus"]
        if decision.get("status") not in {"ACCEPT", "REVIEW"} or type(decision.get("eligible_for_scoring")) is not bool:
            raise ValueError("Malformed recorded consensus")
        eligible = decision["eligible_for_scoring"]
        if eligible:
            annotation = decision.get("annotation") or {}
            if (
                decision["status"] != "ACCEPT"
                or annotation.get("status") != "explicit"
                or not annotation.get("facts")
                or any(f.get("predicate") == "unknown" for f in annotation["facts"])
                or decision.get("eligible_scope") != "assertion_extraction_only"
                or any(value.get("annotation_valid") is not True for value in annotations.values())
            ):
                raise ValueError("Recorded eligibility contradicts the saved consensus contract")
        status = "SILVER_ELIGIBLE" if eligible else "ACCEPT_INELIGIBLE" if decision["status"] == "ACCEPT" else "REVIEW"
        exported = {
            "id": record["id"],
            "split": prefix.get("split"),
            "label_origin": "automatic_silver",
            "export_status": status,
            "prefix": prefix,
            "consensus": decision,
            "annotations": annotations,
            "reason": decision["reason"],
            "input_sha256": expected_input_hash,
            "request_sha256": record["request_sha256"],
        }
        (silver if eligible else queue).append(exported)
    if len(silver) != receipt["eligible_for_scoring"]:
        raise ValueError("Eligible row count differs from recorded receipt")
    files = {
        "silver.jsonl": b"".join(canonical(row) + b"\n" for row in silver),
        "review-queue.jsonl": b"".join(canonical(row) + b"\n" for row in queue),
    }
    manifest = {
        "version": "redial-annotation-export-v1",
        "label_origin": "automatic_silver",
        "source_input_manifest_sha256": input_identity["manifest_sha256"],
        "source_annotations_sha256": digest(annotation_bytes),
        "source_receipt_sha256": digest(receipt_bytes),
        "source_protocol_sha256": digest(protocol_bytes),
        "exporter_sha256": digest(Path(__file__).read_bytes()),
        "rows": len(records),
        "silver_rows": len(silver),
        "review_queue_rows": len(queue),
        "export_status_counts": dict(Counter(row["export_status"] for row in silver + queue)),
        "source_selected_ids": input_identity["selected_ids"],
        "files": {name: {"sha256": digest(payload), "bytes": len(payload)} for name, payload in files.items()},
        "relabeling": False,
        "inference": False,
        "limitations": [
            "Agreement is not accuracy or human gold",
            "Assertion/extraction only; no recommendation success",
            "ACCEPT_INELIGIBLE remains in review queue; all selected rows accounted exactly once",
        ],
    }
    output_dir.mkdir(parents=True)
    for name, payload in {**files, "manifest.json": canonical(manifest) + b"\n"}.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(payload)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.input_manifest, args.run_dir, args.cache_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("rows", "silver_rows", "review_queue_rows", "export_status_counts")}))


if __name__ == "__main__":
    main()
