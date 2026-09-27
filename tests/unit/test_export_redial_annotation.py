from __future__ import annotations

import copy
import json

import httpx
import pytest
from evals.redial_annotation import annotation_payload, consensus, validate_annotation
from scripts.annotate_redial import canonical, digest, load_inputs, request_material
from scripts.export_redial_annotation import export


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def make_fixture(tmp_path, *, unknown=False, invalid_primary=False):
    source, run, cache = (tmp_path / name for name in ("source", "run", "cache"))
    source.mkdir()
    prefixes = [
        {
            "id": f"redial-train-{i}",
            "split": "calibration",
            "source_language": "en",
            "mentioned_entities": [],
            "messages": [{"source_message_id": i, "role": "user", "text": "I want comedy."}],
        }
        for i in range(1, 4)
    ]
    inputs = b"".join(canonical(row) + b"\n" for row in prefixes)
    (source / "inputs.en.jsonl").write_bytes(inputs)
    write(source / "manifest.json", {"files": {"inputs.en.jsonl": {"sha256": digest(inputs), "bytes": len(inputs)}}})
    _, input_identity = load_inputs(source / "manifest.json")
    config = {
        "base_url": "http://127.0.0.1:11434",
        **{
            role: {"model": role, "digest": char * 64, "options": {"temperature": 0, "seed": 1, "num_predict": 100}}
            for role, char in [("primary", "a"), ("verifier", "b")]
        },
    }
    protocol = {"input": input_identity}
    write(run / "protocol.json", protocol)
    records, calls = [], []
    for prefix in prefixes:
        i = prefix["messages"][0]["source_message_id"]
        primary = {
            "status": "explicit",
            "facts": [{"predicate": "inclusion", "attribute": "genre", "target": "comedy", "message_id": i, "quote": "I want comedy."}],
        }
        if i == 2:
            if unknown:
                primary["facts"][0]["predicate"] = "unknown"
            else:
                primary = {"status": "no_explicit_preferences", "facts": []}
        verifier = copy.deepcopy(primary)
        if i == 3:
            verifier["facts"][0]["predicate"] = "preference"
        diagnostics, requests = {}, {}
        for role, value in [("primary", primary), ("verifier", verifier)]:
            material = request_material(prefix, role, config, {})
            key = digest(material)
            validation = validate_annotation(prefix, value)
            raw_text = json.dumps(value)
            if invalid_primary and i == 1 and role == "primary":
                validation = {"valid": False, "errors": ["invalid_json"], "annotation": None}
                raw_text = "not JSON, retained exactly"
            entry = {
                "request": material,
                "request_sha256": key,
                "validation": validation,
                "raw_response": {"message": {"content": raw_text}},
            }
            entry["entry_sha256"] = digest(entry)
            path = cache / "entries" / f"{key}.json"
            write(path, entry)
            calls.append(
                {
                    "id": prefix["id"],
                    "role": role,
                    "status": "complete",
                    "request_sha256": key,
                    "cache_file_sha256": digest(path.read_bytes()),
                    "validation": validation,
                    "annotation_valid": validation["valid"],
                }
            )
            diagnostics[role], requests[role] = validation, key
        decision = consensus(prefix, primary, verifier)
        if invalid_primary and i == 1:
            decision = {"status": "REVIEW", "eligible_for_scoring": False, "reason": "invalid_annotation", "annotation": None}
        records.append(
            {
                "id": prefix["id"],
                "split": prefix["split"],
                "input_sha256": digest(annotation_payload(prefix)),
                "consensus": decision,
                "annotation_diagnostics": diagnostics,
                "request_sha256": requests,
            }
        )
    payload = b"".join(canonical(row) + b"\n" for row in records)
    (run / "annotations.jsonl").write_bytes(payload)
    receipt = {
        "protocol_file_sha256": digest((run / "protocol.json").read_bytes()),
        "annotations_sha256": digest(payload),
        "status": "complete",
        "rows": 3,
        "calls": calls,
        "eligible_for_scoring": sum(row["consensus"]["eligible_for_scoring"] for row in records),
    }
    write(run / "receipt.json", receipt)
    return source / "manifest.json", run, cache


def test_export_is_deterministic_accounts_all_rows_and_never_calls_model(tmp_path, monkeypatch):
    source, run, cache = make_fixture(tmp_path)

    def no_client(*_args, **_kwargs):
        pytest.fail("Exporter attempted network")

    monkeypatch.setattr(httpx, "Client", no_client)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json*")}
    first = export(source, run, cache, tmp_path / "export-a")
    second = export(source, run, cache, tmp_path / "export-b")
    assert first == second
    assert first["rows"] == 3 and first["silver_rows"] == 1 and first["review_queue_rows"] == 2
    assert first["export_status_counts"] == {"SILVER_ELIGIBLE": 1, "ACCEPT_INELIGIBLE": 1, "REVIEW": 1}
    for name in ["silver.jsonl", "review-queue.jsonl", "manifest.json"]:
        assert (tmp_path / "export-a" / name).read_bytes() == (tmp_path / "export-b" / name).read_bytes()
    silver = json.loads((tmp_path / "export-a/silver.jsonl").read_bytes())
    assert silver["prefix"]["messages"][0]["text"] == "I want comedy."
    assert all(silver["annotations"][role]["raw_model_text"] for role in ("primary", "verifier"))
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(FileExistsError):
        export(source, run, cache, tmp_path / "export-a")


@pytest.mark.parametrize("mutation", ["annotations", "input", "cache", "protocol"])
def test_tampering_is_rejected_without_publishing_export(tmp_path, mutation):
    source, run, cache = make_fixture(tmp_path)
    paths = {
        "annotations": run / "annotations.jsonl",
        "input": source.parent / "inputs.en.jsonl",
        "cache": next((cache / "entries").glob("*.json")),
        "protocol": run / "protocol.json",
    }
    paths[mutation].write_bytes(paths[mutation].read_bytes() + b" ")
    with pytest.raises(ValueError):
        export(source, run, cache, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_accepted_unknown_is_not_counted_as_eligible(tmp_path):
    source, run, cache = make_fixture(tmp_path, unknown=True)
    result = export(source, run, cache, tmp_path / "export")
    rows = [json.loads(line) for line in (tmp_path / "export/review-queue.jsonl").read_bytes().splitlines()]
    unknown = next(row for row in rows if row["id"] == "redial-train-2")
    assert unknown["export_status"] == "ACCEPT_INELIGIBLE"
    assert unknown["consensus"]["unknown"] is True
    assert result["silver_rows"] == 1


def test_invalid_json_keeps_both_raw_model_outputs_in_review_queue(tmp_path):
    source, run, cache = make_fixture(tmp_path, invalid_primary=True)
    result = export(source, run, cache, tmp_path / "export")
    rows = [json.loads(line) for line in (tmp_path / "export/review-queue.jsonl").read_bytes().splitlines()]
    first = next(row for row in rows if row["id"] == "redial-train-1")
    assert first["annotations"]["primary"]["raw_model_text"] == "not JSON, retained exactly"
    assert first["annotations"]["primary"]["parsed_annotation"] is None
    assert first["annotations"]["verifier"]["parsed_annotation"]["status"] == "explicit"
    assert result["rows"] == result["review_queue_rows"] == 3


@pytest.mark.parametrize("mutation", ["missing_row", "reordered_rows", "duplicate_call", "wrong_input_hash", "fake_eligible"])
def test_rehashed_alignment_errors_still_rejected(tmp_path, mutation):
    source, run, cache = make_fixture(tmp_path)
    rows = [json.loads(line) for line in (run / "annotations.jsonl").read_bytes().splitlines()]
    receipt = json.loads((run / "receipt.json").read_bytes())
    if mutation == "missing_row":
        rows.pop()
    elif mutation == "reordered_rows":
        rows.reverse()
    elif mutation == "duplicate_call":
        receipt["calls"].append(receipt["calls"][0])
    elif mutation == "wrong_input_hash":
        rows[0]["input_sha256"] = "0" * 64
    else:
        rows[1]["consensus"]["eligible_for_scoring"] = True
    payload = b"".join(canonical(row) + b"\n" for row in rows)
    (run / "annotations.jsonl").write_bytes(payload)
    receipt["annotations_sha256"] = digest(payload)
    write(run / "receipt.json", receipt)
    with pytest.raises(ValueError):
        export(source, run, cache, tmp_path / "export")
