import json
from pathlib import Path

import httpx
import pytest
from evals.llm_replay import (
    CacheIntegrityError,
    CacheMissError,
    ModelIdentityError,
    OllamaReplayClient,
    ReplayError,
    ReplayRoles,
    RoleConfig,
)
from pydantic import BaseModel


class Answer(BaseModel):
    value: str


@pytest.mark.parametrize("version", [2, True, "1"])
@pytest.mark.parametrize("target", ["manifest", "entry"])
def test_unsupported_hashed_version_is_rejected(tmp_path, version, target):
    from evals.llm_replay import _sha256

    writer = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport([]))
    writer.structured(Answer, "s", {})
    manifest = tmp_path / "frozen.json"
    writer.write_identity_manifest(manifest)
    path = manifest if target == "manifest" else next((tmp_path / "entries").glob("*.json"))
    field = "manifest_sha256" if target == "manifest" else "entry_sha256"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cache_schema_version"] = version
    value[field] = _sha256({key: val for key, val in value.items() if key != field})
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CacheIntegrityError, match="cache_schema_version"):
        reader = OllamaReplayClient(_config(), cache_dir=tmp_path, mode="require_cache", frozen_manifest=manifest)
        reader.structured(Answer, "s", {})


def test_post_inference_serialization_failure_is_materialized(monkeypatch, tmp_path):
    def broken_dump(*args, **kwargs):
        raise ValueError("serializer failed")

    monkeypatch.setattr(Answer, "model_dump", broken_dump)
    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport([]))
    with pytest.raises(ReplayError, match="serializer failed"):
        client.structured(Answer, "s", {})
    record = json.loads(next((tmp_path / "incomplete").glob("*.json")).read_text(encoding="utf-8"))
    assert record["complete"] is False
    assert "serializer failed" in record["error"]
    assert not (tmp_path / "entries").exists()


def _config(**changes) -> RoleConfig:
    return RoleConfig(role="simulator", model="qwen3:8b", base_url="http://mock", **changes)


def _transport(calls: list[httpx.Request], *, chat_statuses=(200,), content='{"value":"ok"}'):
    responses = iter(chat_statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "digest": "sha256:exact"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.3"})
        if request.url.path == "/api/chat":
            status = next(responses)
            if status != 200:
                return httpx.Response(status, json={"error": "temporary"})
            return httpx.Response(200, json={"message": {"content": content}, "prompt_eval_count": 11, "eval_count": 7})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_record_materializes_identity_key_raw_response_and_exact_offline_replay(tmp_path: Path):
    calls: list[httpx.Request] = []
    client = OllamaReplayClient(
        _config(seed=91, temperature=0.2, num_predict=12, think=True), cache_dir=tmp_path, transport=_transport(calls)
    )
    result, tokens = client.structured(Answer, "system prompt", {"input": "x"})

    assert result == Answer(value="ok") and tokens == 18
    assert [request.url.path for request in calls] == ["/api/tags", "/api/version", "/api/chat"]
    request = json.loads(calls[-1].content)
    assert request["model"] == "qwen3:8b"
    assert request["think"] is True
    assert request["options"] == {"seed": 91, "temperature": 0.2, "num_predict": 12}

    manifest = tmp_path / "frozen.json"
    client.write_identity_manifest(manifest)
    entries = list((tmp_path / "entries").glob("*.json"))
    assert len(entries) == 1
    entry = json.loads(entries[0].read_text(encoding="utf-8"))
    assert entry["raw_response"]["message"]["content"] == '{"value":"ok"}'
    assert entry["request"]["role"] == "simulator"
    assert entry["request"]["model_identity"]["digest"] == "sha256:exact"
    assert entry["request"]["system_prompt"] == "system prompt"
    assert entry["request"]["schema"] == Answer.model_json_schema()

    def offline(_: httpx.Request) -> httpx.Response:
        raise AssertionError("require_cache must never call HTTP, including metadata")

    replay = OllamaReplayClient(
        _config(seed=91, temperature=0.2, num_predict=12, think=True),
        cache_dir=tmp_path,
        mode="require_cache",
        frozen_manifest=manifest,
        transport=httpx.MockTransport(offline),
    )
    replayed, replayed_tokens = replay.structured(Answer, "system prompt", {"input": "x"})
    assert replayed == result and replayed_tokens == tokens


def test_expected_digest_fails_before_inference(tmp_path: Path):
    calls: list[httpx.Request] = []
    client = OllamaReplayClient(_config(), cache_dir=tmp_path, expected_digest="sha256:other", transport=_transport(calls))
    with pytest.raises(ModelIdentityError, match="expected_digest"):
        client.structured(Answer, "s", {})
    assert [request.url.path for request in calls] == ["/api/tags", "/api/version"]


def test_roles_are_explicit_and_part_of_the_cache_key(tmp_path: Path):
    simulator = _config()
    judge = RoleConfig(role="judge", model="qwen3:8b", base_url="http://mock")
    assert ReplayRoles(simulator, judge).judge.model == "qwen3:8b"
    simulator_calls: list[httpx.Request] = []
    judge_calls: list[httpx.Request] = []
    OllamaReplayClient(simulator, cache_dir=tmp_path, transport=_transport(simulator_calls)).structured(Answer, "same", {"x": 1})
    OllamaReplayClient(judge, cache_dir=tmp_path, transport=_transport(judge_calls)).structured(Answer, "same", {"x": 1})
    assert len(list((tmp_path / "entries").glob("*.json"))) == 2


def test_require_cache_miss_is_offline(tmp_path: Path):
    manifest = tmp_path / "frozen.json"
    payload = {
        "cache_schema_version": 1,
        "role": "simulator",
        "model_identity": {"model": "qwen3:8b", "digest": "sha256:exact", "ollama_version": "0.12.3", "base_url": "http://mock"},
        "options": {"seed": 42, "temperature": 0.0, "num_predict": 700, "think": False},
    }
    from evals.llm_replay import _sha256

    manifest.write_text(json.dumps({**payload, "manifest_sha256": _sha256(payload)}), encoding="utf-8")

    client = OllamaReplayClient(_config(), cache_dir=tmp_path, mode="require_cache", frozen_manifest=manifest)
    with pytest.raises(CacheMissError):
        client.structured(Answer, "s", {"new": "input"})


def test_retry_is_limited_to_two_attempts_and_attempts_are_recorded(tmp_path: Path):
    calls: list[httpx.Request] = []
    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(calls, chat_statuses=(503, 200)))
    client.structured(Answer, "s", {})
    assert [request.url.path for request in calls] == ["/api/tags", "/api/version", "/api/chat", "/api/chat"]
    entry = json.loads(next((tmp_path / "entries").glob("*.json")).read_text(encoding="utf-8"))
    chat_attempts = [attempt for attempt in entry["attempts"] if attempt["path"] == "/api/chat"]
    assert [attempt["status_code"] for attempt in chat_attempts] == [503, 200]


def test_exhausted_post_records_its_attempts_in_incomplete_entry(tmp_path: Path):
    calls: list[httpx.Request] = []
    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(calls, chat_statuses=(503, 503)))
    with pytest.raises(ReplayError):
        client.structured(Answer, "s", {})
    incomplete = json.loads(next((tmp_path / "incomplete").glob("*.json")).read_text(encoding="utf-8"))
    chat_attempts = [attempt for attempt in incomplete["attempts"] if attempt["path"] == "/api/chat"]
    assert [attempt["status_code"] for attempt in chat_attempts] == [503, 503]


def test_schema_error_is_incomplete_not_a_retry_or_success(tmp_path: Path):
    calls: list[httpx.Request] = []
    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(calls, content='{"wrong":"field"}'))
    with pytest.raises(ReplayError, match="invalid"):
        client.structured(Answer, "s", {})
    assert [request.url.path for request in calls] == ["/api/tags", "/api/version", "/api/chat"]
    incomplete = json.loads(next((tmp_path / "incomplete").glob("*.json")).read_text(encoding="utf-8"))
    assert incomplete["complete"] is False
    assert "ValidationError" in incomplete["error"]


@pytest.mark.parametrize("counters", [{}, {"prompt_eval_count": None, "eval_count": 7}, {"prompt_eval_count": 1.5, "eval_count": 7}])
def test_missing_or_noninteger_token_counters_fail_incomplete(tmp_path: Path, counters: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "digest": "sha256:exact"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.3"})
        return httpx.Response(200, json={"message": {"content": '{"value":"ok"}'}, **counters})

    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(ReplayError, match="token counters"):
        client.structured(Answer, "s", {})
    assert json.loads(next((tmp_path / "incomplete").glob("*.json")).read_text(encoding="utf-8"))["complete"] is False


def test_identity_failure_is_marked_incomplete_before_any_inference(tmp_path: Path):
    calls: list[httpx.Request] = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"error": "offline"})

    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=httpx.MockTransport(unavailable))
    with pytest.raises(ReplayError):
        client.structured(Answer, "s", {})
    assert [request.url.path for request in calls] == ["/api/tags", "/api/tags"]
    incomplete = json.loads(next((tmp_path / "incomplete").glob("*.json")).read_text(encoding="utf-8"))
    assert incomplete["complete"] is False
    assert [attempt["status_code"] for attempt in incomplete["attempts"]] == [503, 503]


def test_cache_is_verified_and_never_silently_replaced(tmp_path: Path):
    calls: list[httpx.Request] = []
    client = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(calls))
    client.structured(Answer, "s", {})
    entry_path = next((tmp_path / "entries").glob("*.json"))
    entry_path.write_text(entry_path.read_text(encoding="utf-8").replace('"ok"', '"tampered"'), encoding="utf-8")

    second_calls: list[httpx.Request] = []
    second = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(second_calls))
    with pytest.raises(CacheIntegrityError, match="entry hash mismatch"):
        second.structured(Answer, "s", {})
    assert [request.url.path for request in second_calls] == ["/api/tags", "/api/version"]


def test_require_cache_rejects_a_hashed_but_incomplete_entry(tmp_path: Path):
    from evals.llm_replay import _sha256

    calls: list[httpx.Request] = []
    writer = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(calls))
    writer.structured(Answer, "s", {})
    manifest = tmp_path / "frozen.json"
    writer.write_identity_manifest(manifest)
    entry_path = next((tmp_path / "entries").glob("*.json"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["complete"] = False
    entry["entry_sha256"] = _sha256({key: value for key, value in entry.items() if key != "entry_sha256"})
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    reader = OllamaReplayClient(_config(), cache_dir=tmp_path, mode="require_cache", frozen_manifest=manifest)
    with pytest.raises(CacheIntegrityError, match="not a complete replay"):
        reader.structured(Answer, "s", {})


def test_frozen_manifest_locks_role_options_and_record_identity(tmp_path: Path):
    calls: list[httpx.Request] = []
    initial = OllamaReplayClient(_config(), cache_dir=tmp_path, transport=_transport(calls))
    initial.structured(Answer, "s", {})
    manifest = tmp_path / "frozen.json"
    initial.write_identity_manifest(manifest)

    with pytest.raises(ModelIdentityError, match="frozen role"):
        OllamaReplayClient(
            RoleConfig(role="judge", model="qwen3:8b", base_url="http://mock"),
            cache_dir=tmp_path,
            mode="require_cache",
            frozen_manifest=manifest,
        )
    with pytest.raises(ModelIdentityError, match="frozen options"):
        OllamaReplayClient(_config(seed=99), cache_dir=tmp_path, mode="require_cache", frozen_manifest=manifest)

    recorded_calls: list[httpx.Request] = []

    def changed_identity(request: httpx.Request) -> httpx.Response:
        recorded_calls.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "digest": "sha256:changed"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.4"})
        raise AssertionError("digest/runtime mismatch must fail before inference")

    pinned = OllamaReplayClient(_config(), cache_dir=tmp_path, frozen_manifest=manifest, transport=httpx.MockTransport(changed_identity))
    with pytest.raises(ModelIdentityError, match="differs from frozen"):
        pinned.structured(Answer, "new", {})
    assert [request.url.path for request in recorded_calls] == ["/api/tags", "/api/version"]
