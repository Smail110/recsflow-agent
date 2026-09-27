from __future__ import annotations

import copy
import json

import httpx
import pytest
from scripts import annotate_redial as runner


def prefix(index=1):
    return {
        "id": f"redial-train-{index}",
        "source_language": "en",
        "mentioned_entities": [],
        "messages": [{"role": "user", "source_message_id": index, "text": "I want comedy."}],
    }


def answer(index=1):
    return {
        "status": "explicit",
        "facts": [{"predicate": "inclusion", "attribute": "genre", "target": "comedy", "message_id": index, "quote": "I want comedy."}],
    }


def config():
    return {
        "base_url": "http://127.0.0.1:11434",
        "timeout_s": 5,
        "primary": {
            "model": "llama3.1:8b",
            "digest": "a" * 64,
            "options": {"temperature": 0, "seed": 20260923, "num_predict": 1600},
            "think": None,
        },
        "verifier": {
            "model": "qwen3:8b",
            "digest": "b" * 64,
            "options": {"temperature": 0, "seed": 20260923, "num_predict": 1600},
            "think": False,
        },
    }


class Server:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.bad_digest = False
        self.bad_content = False
        self.bad_evidence = False
        self.bad_schema = False
        self.truncated = False
        self.usage = True
        self.changed_after_chat = False

    def __call__(self, request):
        self.calls.append((request.method, request.url.path, request.content))
        if request.url.path == "/api/tags":
            entries = [
                {"name": spec["model"], "digest": spec["digest"], "details": {"family": "test"}}
                for spec in (config()["primary"], config()["verifier"])
            ]
            if self.bad_digest or (self.changed_after_chat and any(path == "/api/chat" for _, path, _ in self.calls)):
                entries[0]["digest"] = "0" * 64
            return httpx.Response(200, json={"models": entries})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-v1"})
        assert request.url.path == "/api/chat"
        if self.fail:
            return httpx.Response(503, text="controlled unavailable")
        wire = json.loads(request.content)
        payload = json.loads(wire["messages"][1]["content"])
        result = answer(payload["messages"][0]["source_message_id"])
        if self.bad_evidence:
            result["facts"][0]["quote"] = "invented quote"
        if self.bad_schema:
            result["unexpected_field"] = "not permitted"
        body = {
            "model": wire["model"],
            "done": True,
            "done_reason": "length" if self.truncated else "stop",
            "message": {"content": "not JSON" if self.bad_content else json.dumps(result)},
        }
        if self.usage:
            body.update(prompt_eval_count=30, eval_count=20)
        return httpx.Response(200, json=body)

    def factory(self, **kwargs):
        return httpx.Client(transport=httpx.MockTransport(self), **kwargs)


def execute(tmp_path, server, *, name="record", mode="record", rows=None):
    return runner.execute(
        rows or [prefix()],
        config(),
        tmp_path / "cache",
        tmp_path / name,
        mode=mode,
        input_identity={"manifest_sha256": "test"},
        config_sha256="test",
        client_factory=server.factory,
    )


def test_record_replay_byte_identical_and_no_network_on_replay(tmp_path):
    server = Server()
    recorded = execute(tmp_path, server, rows=[prefix(1), prefix(2)])
    assert recorded["status"] == "complete"
    assert recorded["eligible_for_scoring"] == 2
    before = len(server.calls)
    replayed = execute(tmp_path, server, name="replay", mode="replay", rows=[prefix(1), prefix(2)])
    assert len(server.calls) == before
    assert replayed["status"] == "complete"
    assert (tmp_path / "record/annotations.jsonl").read_bytes() == (tmp_path / "replay/annotations.jsonl").read_bytes()
    chats = [json.loads(body) for _, path, body in server.calls if path == "/api/chat"]
    assert [body["model"] for body in chats] == ["llama3.1:8b"] * 2 + ["qwen3:8b"] * 2
    assert "think" not in chats[0] and chats[-1]["think"] is False
    for chat in chats:
        payload = json.loads(chat["messages"][1]["content"])
        assert set(payload) == {"id", "messages", "source_language", "mentioned_entities"}
        assert "facts" not in payload
    for entry in (tmp_path / "cache/entries").glob("*.json"):
        raw = entry.read_bytes()
        assert b"\r\n" not in raw
        content = json.loads(raw)
        assert content["usage"]["total_tokens"] == 50
        assert content["identity_before"] == content["identity_after"]
        assert content["elapsed_ms"] >= 0
    assert recorded["protocol_file_sha256"] == runner.digest((tmp_path / "record/protocol.json").read_bytes())


def test_resume_failures_preserves_old_receipts_and_completes_new_snapshot(tmp_path):
    server = Server()
    server.fail = True
    failed = execute(tmp_path, server)
    assert failed["status"] == "incomplete" and failed["rows"] == 1
    old_files = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    server.fail = False
    resumed = execute(tmp_path, server, name="resumed")
    assert resumed["status"] == "complete"
    assert all(path.read_bytes() == contents for path, contents in old_files.items())
    assert len(list((tmp_path / "cache/failures").rglob("*.json"))) == 2
    with pytest.raises(FileExistsError):
        execute(tmp_path, server)


@pytest.mark.parametrize("flag", ["bad_digest", "changed_after_chat"])
def test_transport_identity_failures_remain_auditable(tmp_path, flag):
    server = Server()
    setattr(server, flag, True)
    result = execute(tmp_path, server)
    assert result["status"] == "incomplete"
    row = json.loads((tmp_path / "record/annotations.jsonl").read_bytes())
    assert row["consensus"]["eligible_for_scoring"] is False
    failures = [json.loads(p.read_bytes()) for p in (tmp_path / "cache/failures").rglob("*.json")]
    assert failures and all(x["status"] == "failed" for x in failures)


@pytest.mark.parametrize("flag", ["bad_content", "bad_schema", "bad_evidence", "truncated"])
def test_invalid_complete_outputs_cached_replayed_and_never_resampled(tmp_path, flag):
    server = Server()
    setattr(server, flag, True)
    result = execute(tmp_path, server)
    assert result["status"] == "complete"
    assert result["invalid_complete_annotations"] == 2
    assert result["eligible_for_scoring"] == 0
    before = len(server.calls)
    replay = execute(tmp_path, server, name="replay", mode="replay")
    assert len(server.calls) == before
    assert replay["invalid_complete_annotations"] == 2
    assert (tmp_path / "record/annotations.jsonl").read_bytes() == (tmp_path / "replay/annotations.jsonl").read_bytes()
    # A record resume must also reuse invalid observations, even if the server
    # would now return a valid answer. Otherwise retries would bias the cohort.
    setattr(server, flag, False)
    resumed = execute(tmp_path, server, name="resume")
    assert len(server.calls) == before
    assert resumed["invalid_complete_annotations"] == 2
    assert all(call["cache_hit"] for call in resumed["calls"])
    entries = [json.loads(p.read_bytes()) for p in (tmp_path / "cache/entries").glob("*.json")]
    assert len(entries) == 2
    assert all(e["status"] == "complete" and e["annotation_valid"] is False for e in entries)
    if flag == "truncated":
        assert all(e["validation"]["errors"] == ["truncated_model_output"] for e in entries)


def test_missing_usage_is_null_not_zero(tmp_path):
    server = Server()
    server.usage = False
    assert execute(tmp_path, server)["status"] == "complete"
    for path in (tmp_path / "cache/entries").glob("*.json"):
        assert set(json.loads(path.read_bytes())["usage"].values()) == {None}


def test_cache_tampering_is_not_silently_overwritten(tmp_path):
    server = Server()
    execute(tmp_path, server)
    entry = next((tmp_path / "cache/entries").glob("*.json"))
    content = json.loads(entry.read_bytes())
    content["annotation"]["facts"][0]["target"] = "invented"
    entry.write_text(json.dumps(content), encoding="utf8")
    old = entry.read_bytes()
    result = execute(tmp_path, server, name="tampered-replay", mode="replay")
    assert result["status"] == "incomplete"
    assert entry.read_bytes() == old
    assert any(call["status"] == "cache_error" for call in result["calls"])


def test_replay_cache_miss_does_not_create_http_client(tmp_path):
    class NoClient:
        def factory(self, **_kwargs):
            pytest.fail("Replay attempted to construct a network client")

    result = execute(tmp_path, NoClient(), mode="replay")
    assert result["status"] == "incomplete"
    assert all(call["status"] == "cache_miss" for call in result["calls"])


def input_manifest(tmp_path, rows):
    contents = b"".join(runner.canonical(row) + b"\n" for row in rows)
    (tmp_path / "inputs.en.jsonl").write_bytes(contents)
    manifest = {
        "files": {
            "inputs.en.jsonl": {"sha256": runner.digest(contents), "bytes": len(contents)},
            "observed-annotations.DO-NOT-PROMPT.jsonl": {"sha256": "do not read", "bytes": 0},
        }
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(runner.canonical(manifest))
    return path


def test_all_input_rows_verified_before_limit_and_no_future_file_open(tmp_path):
    path = input_manifest(tmp_path, [prefix(1), prefix(2)])
    rows, receipt = runner.load_inputs(path, limit=1)
    assert len(rows) == 1 and receipt["total_rows_verified"] == 2
    path = input_manifest(tmp_path, [prefix(1), prefix(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_inputs(path, limit=1)


def test_input_hash_mismatch_and_closed_path_rejected(tmp_path):
    path = input_manifest(tmp_path, [prefix()])
    (tmp_path / "inputs.en.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        runner.load_inputs(path)
    with pytest.raises(ValueError, match="Closed-data"):
        runner.read_open(tmp_path / "BLIND.json")


def test_split_filter_before_limit_and_split_not_sent_to_model(tmp_path):
    rows = [{**prefix(1), "split": "calibration"}, {**prefix(2), "split": "dev"}, {**prefix(3), "split": "dev"}]
    selected, _ = runner.load_inputs(input_manifest(tmp_path, rows), split="dev", limit=1)
    assert [r["id"] for r in selected] == ["redial-train-2"]
    material = runner.request_material(selected[0], "primary", config(), {})
    assert "split" not in json.loads(material["wire_request"]["messages"][1]["content"])


@pytest.mark.parametrize("mutation", ["remote", "digest", "sampling", "same_model"])
def test_config_rejects_unpinned_or_remote_generation(tmp_path, mutation):
    cfg = copy.deepcopy(config())
    if mutation == "remote":
        cfg["base_url"] = "https://paid.example/api"
    elif mutation == "digest":
        cfg["primary"]["digest"] = "latest"
    elif mutation == "sampling":
        cfg["primary"]["options"]["temperature"] = 0.5
    else:
        cfg["verifier"]["digest"] = cfg["primary"]["digest"]
    path = tmp_path / "config.json"
    path.write_bytes(runner.canonical(cfg))
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_wire_schema_removes_only_maxlength_without_mutating_semantic_schema():
    schema = runner.annotation.Annotation.model_json_schema()
    original = copy.deepcopy(schema)
    wire = runner.wire_schema(schema)

    def compare(left, right):
        if isinstance(left, dict):
            assert set(right) == set(left) - {"maxLength"}
            for key in right:
                compare(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for first, second in zip(left, right, strict=True):
                compare(first, second)
        else:
            assert left == right

    compare(schema, wire)
    assert schema == original
    assert schema["$defs"]["Fact"]["properties"]["quote"]["maxLength"] == 4000
    assert schema["$defs"]["Fact"]["properties"]["target"]["maxLength"] == 500
    assert wire["$defs"]["Fact"]["properties"]["quote"]["minLength"] == 1
    material = runner.request_material(prefix(), "primary", config(), {})
    assert material["wire_request"]["format"] == wire
    assert material["wire_schema_sha256"] == runner.digest(wire)
    assert material["validation_schema_sha256"] == runner.digest(schema)
    assert material["wire_schema_sha256"] != material["validation_schema_sha256"]


def test_quote_4001_remains_invalid_after_wire_relaxation():
    row = prefix()
    row["messages"][0]["text"] = "I want comedy." + "x" * 3987
    assert len(row["messages"][0]["text"]) == 4001
    value = answer()
    value["facts"][0]["quote"] = row["messages"][0]["text"]
    raw = {"message": {"content": json.dumps(value)}, "done_reason": "stop"}
    parsed, validation = runner.analyze_response(row, raw)
    assert parsed is None and validation["valid"] is False


def test_three_transport_failures_stop_all_further_network_calls_keep_rows(tmp_path):
    server = Server()
    server.fail = True
    result = execute(tmp_path, server, rows=[prefix(i) for i in range(1, 6)])
    assert sum(path == "/api/chat" for _, path, _ in server.calls) == 3
    assert result["circuit_open"] is True
    assert result["unattempted_role_calls"] == 7
    assert result["rows"] == 5 and len(result["calls"]) == 10
    rows = [json.loads(line) for line in (tmp_path / "record/annotations.jsonl").read_bytes().splitlines()]
    assert all(row["consensus"]["status"] == "REVIEW" for row in rows)
    assert all(row["annotation_status"]["verifier"]["status"] == "unattempted" for row in rows)


def test_invalid_complete_output_does_not_trigger_circuit(tmp_path):
    server = Server()
    server.bad_content = True
    result = execute(tmp_path, server, rows=[prefix(i) for i in range(1, 5)])
    assert sum(path == "/api/chat" for _, path, _ in server.calls) == 8
    assert result["circuit_open"] is False
    assert result["invalid_complete_annotations"] == 8


def test_complete_invalid_output_resets_consecutive_transport_failure_streak(tmp_path):
    class AlternatingServer(Server):
        chats = 0

        def __call__(self, request):
            if request.url.path == "/api/chat":
                self.chats += 1
                self.fail = self.chats % 3 != 0
                self.bad_content = self.chats % 3 == 0
            return super().__call__(request)

    server = AlternatingServer()
    result = execute(tmp_path, server, rows=[prefix(i) for i in range(1, 4)])
    assert server.chats == 6
    assert result["circuit_open"] is False
    assert result["invalid_complete_annotations"] == 2


def test_circuit_keeps_existing_cached_observations_without_http(tmp_path):
    server = Server()
    execute(tmp_path, server, name="seed-cache", rows=[prefix(4)])
    server.calls.clear()
    server.fail = True
    result = execute(tmp_path, server, rows=[prefix(i) for i in range(1, 5)])
    assert sum(path == "/api/chat" for _, path, _ in server.calls) == 3
    assert result["circuit_open"] is True
    cached_calls = [c for c in result["calls"] if c["id"] == "redial-train-4"]
    assert len(cached_calls) == 2
    assert all(c["status"] == "complete" and c["cache_hit"] for c in cached_calls)
