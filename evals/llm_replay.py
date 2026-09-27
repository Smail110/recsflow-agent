"""Воспроизводимый кэш для структурированных вызовов Ollama в evaluation.

Модуль намеренно не подключён к product runtime: он предназначен для фиксированных
экспериментов. ``require_cache`` не открывает HTTP-клиент и не читает metadata
Ollama, поэтому воспроизводит только ранее материализованный exact response.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel

CACHE_SCHEMA_VERSION = 1
CacheMode = Literal["record", "require_cache"]


class ReplayError(RuntimeError):
    """Base error for an incomplete or non-reproducible replay."""


class CacheMissError(ReplayError):
    pass


class CacheIntegrityError(ReplayError):
    pass


class ModelIdentityError(ReplayError):
    pass


@dataclass(frozen=True)
class RoleConfig:
    """One independently configurable evaluation role."""

    role: str
    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 45.0
    seed: int | None = 42
    temperature: float = 0.0
    num_predict: int = 700
    think: bool = False
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.model.strip():
            raise ValueError("role and model must be non-empty")
        if self.timeout_s <= 0 or self.num_predict < 1 or not 1 <= self.max_attempts <= 2:
            raise ValueError("timeout_s/num_predict/max_attempts are outside the replay protocol")

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def options(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        if self.seed is not None:
            values["seed"] = self.seed
        return values


@dataclass(frozen=True)
class ModelIdentity:
    model: str
    digest: str
    ollama_version: str
    base_url: str


@dataclass(frozen=True)
class FrozenReplayManifest:
    role: str
    options: dict[str, Any]
    model_identity: ModelIdentity


@dataclass(frozen=True)
class ReplayRoles:
    """Keep simulator and judge identities explicit even when they share a model."""

    simulator: RoleConfig
    judge: RoleConfig

    def __post_init__(self) -> None:
        if self.simulator.role != "simulator" or self.judge.role != "judge":
            raise ValueError("ReplayRoles requires simulator and judge role names")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _entry_payload(entry: Mapping[str, object]) -> dict[str, object]:
    result = dict(entry)
    result.pop("entry_sha256", None)
    return result


def _frozen_manifest_from_path(path: Path) -> FrozenReplayManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if type(data.get("cache_schema_version")) is not int or data["cache_schema_version"] != CACHE_SCHEMA_VERSION:
            raise CacheIntegrityError("unsupported manifest cache_schema_version")
        if data.get("manifest_sha256") != _sha256({key: value for key, value in data.items() if key != "manifest_sha256"}):
            raise CacheIntegrityError("frozen manifest hash mismatch")
        identity = data["model_identity"]
        model_identity = ModelIdentity(
            model=str(identity["model"]),
            digest=str(identity["digest"]),
            ollama_version=str(identity["ollama_version"]),
            base_url=str(identity["base_url"]),
        )
        role, options = data["role"], data["options"]
        if not isinstance(role, str) or not role.strip() or not isinstance(options, dict):
            raise CacheIntegrityError("frozen manifest has invalid role or options")
        if not all((model_identity.model, model_identity.digest, model_identity.ollama_version, model_identity.base_url)):
            raise CacheIntegrityError("frozen manifest has an empty model identity field")
        return FrozenReplayManifest(role=role, options=options, model_identity=model_identity)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"invalid frozen replay manifest: {path}") from exc


class OllamaReplayClient:
    """A generic ``structured`` client with record and exact-offline replay modes."""

    def __init__(
        self,
        config: RoleConfig,
        *,
        cache_dir: Path,
        mode: CacheMode = "record",
        expected_digest: str | None = None,
        frozen_manifest: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if mode not in {"record", "require_cache"}:
            raise ValueError("mode must be record or require_cache")
        if mode == "require_cache" and frozen_manifest is None:
            raise ValueError("require_cache needs a local frozen_manifest and performs zero network requests")
        self.config = config
        self.cache_dir = cache_dir
        self.mode = mode
        self.expected_digest = expected_digest
        self.transport = transport
        self._frozen = _frozen_manifest_from_path(frozen_manifest) if frozen_manifest else None
        self._identity = self._frozen.model_identity if self._frozen else None
        if self._frozen is not None:
            self._validate_frozen_manifest(self._frozen)

    def _validate_identity(self, identity: ModelIdentity) -> None:
        if identity.model != self.config.model:
            raise ModelIdentityError(f"frozen model {identity.model!r} differs from configured {self.config.model!r}")
        if identity.base_url.rstrip("/") != self.config.normalized_base_url:
            raise ModelIdentityError("frozen base_url differs from configured base_url")
        if self.expected_digest is not None and identity.digest != self.expected_digest:
            raise ModelIdentityError("model digest differs from expected_digest")

    def _validate_frozen_manifest(self, frozen: FrozenReplayManifest) -> None:
        self._validate_identity(frozen.model_identity)
        if frozen.role != self.config.role:
            raise ModelIdentityError("frozen role differs from configured role")
        if frozen.options != {**self.config.options, "think": self.config.think}:
            raise ModelIdentityError("frozen options differ from configured options")

    def _request_json(
        self, client: httpx.Client, method: str, path: str, *, payload: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for number in range(1, self.config.max_attempts + 1):
            try:
                response = client.request(method, path, json=payload)
                retryable = response.status_code == 429 or response.status_code >= 500
                attempts.append({"attempt": number, "path": path, "status_code": response.status_code, "retryable": retryable})
                if response.is_error:
                    response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    error = ReplayError(f"{path} returned a non-object JSON payload")
                    error.attempts = attempts  # type: ignore[attr-defined]
                    raise error
                return data, attempts
            except (json.JSONDecodeError, ValueError) as exc:
                error = ReplayError(f"Ollama returned invalid JSON at {path}: {exc}")
                error.attempts = attempts  # type: ignore[attr-defined]
                raise error from exc
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError) and (exc.response.status_code == 429 or exc.response.status_code >= 500)
                )
                if not attempts or attempts[-1].get("attempt") != number:
                    attempts.append({"attempt": number, "path": path, "error": type(exc).__name__, "retryable": retryable})
                if not retryable or number == self.config.max_attempts:
                    error = ReplayError(f"Ollama request failed at {path}: {exc}")
                    error.attempts = attempts  # type: ignore[attr-defined]
                    raise error from exc
        error = ReplayError(f"Ollama request failed at {path}: {last_error}")
        error.attempts = attempts  # type: ignore[attr-defined]
        raise error

    def _fetch_identity(self) -> tuple[ModelIdentity, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        with httpx.Client(
            base_url=self.config.normalized_base_url, timeout=self.config.timeout_s, trust_env=False, transport=self.transport
        ) as client:
            tags, tag_attempts = self._request_json(client, "GET", "/api/tags")
            attempts.extend(tag_attempts)
            versions, version_attempts = self._request_json(client, "GET", "/api/version")
            attempts.extend(version_attempts)
        matches = [item for item in tags.get("models", []) if isinstance(item, dict) and item.get("name") == self.config.model]
        if len(matches) != 1 or not isinstance(matches[0].get("digest"), str) or not matches[0]["digest"]:
            error = ModelIdentityError(f"exact digest for configured model {self.config.model!r} was not found in /api/tags")
            error.attempts = attempts  # type: ignore[attr-defined]
            raise error
        if not isinstance(versions.get("version"), str) or not versions["version"]:
            error = ModelIdentityError("/api/version has no exact Ollama version")
            error.attempts = attempts  # type: ignore[attr-defined]
            raise error
        identity = ModelIdentity(self.config.model, matches[0]["digest"], versions["version"], self.config.normalized_base_url)
        try:
            self._validate_identity(identity)
        except ModelIdentityError as exc:
            exc.attempts = attempts  # type: ignore[attr-defined]
            raise
        return identity, attempts

    def _material(
        self, schema: type[BaseModel], system_prompt: str, payload: Mapping[str, object], identity: ModelIdentity
    ) -> dict[str, object]:
        schema_json = schema.model_json_schema()
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "role": self.config.role,
            "model_identity": asdict(identity),
            "options": {**self.config.options, "think": self.config.think},
            "system_prompt": system_prompt,
            "schema": schema_json,
            "input": dict(payload),
            "input_sha256": _sha256(dict(payload)),
            "schema_sha256": _sha256(schema_json),
        }

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / "entries" / f"{key}.json"

    def _incomplete_path(self, key: str) -> Path:
        return self.cache_dir / "incomplete" / f"{key}.json"

    def _read_entry(self, path: Path, *, key: str, material: Mapping[str, object], schema: type[BaseModel]) -> tuple[BaseModel, int]:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict) or entry.get("entry_sha256") != _sha256(_entry_payload(entry)):
                raise CacheIntegrityError("entry hash mismatch")
            if entry.get("complete") is not True:
                raise CacheIntegrityError("entry is not a complete replay")
            if type(entry.get("cache_schema_version")) is not int or entry["cache_schema_version"] != CACHE_SCHEMA_VERSION:
                raise CacheIntegrityError("unsupported entry cache_schema_version")
            if entry.get("cache_key") != key or entry.get("request") != material:
                raise CacheIntegrityError("entry key or protocol material mismatch")
            model = schema.model_validate(entry["validated_output"])
            tokens = entry["total_tokens"]
            if type(tokens) is not int or tokens < 0:
                raise CacheIntegrityError("entry total_tokens is invalid")
            return model, tokens
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            if isinstance(exc, CacheIntegrityError):
                raise
            raise CacheIntegrityError(f"invalid cache entry: {path}") from exc

    @staticmethod
    def _write_immutable(path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (_canonical(value) + "\n").encode("utf-8")
        temporary: str | None = None
        try:
            handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return
            except OSError:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _write_incomplete(self, key: str, material: Mapping[str, object], attempts: list[dict[str, Any]], error: Exception) -> None:
        record = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "request": material,
            "attempts": attempts,
            "complete": False,
            "error": f"{type(error).__name__}: {error}",
        }
        record["entry_sha256"] = _sha256(record)
        self._write_immutable(self._incomplete_path(key), record)

    def identity_manifest(self) -> dict[str, object]:
        if self._identity is None:
            raise ReplayError("model identity is unavailable until record mode resolves /api/tags and /api/version")
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "role": self.config.role,
            "model_identity": asdict(self._identity),
            "options": {**self.config.options, "think": self.config.think},
        }
        return {**manifest, "manifest_sha256": _sha256(manifest)}

    def write_identity_manifest(self, path: Path) -> None:
        manifest = self.identity_manifest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical(manifest) + "\n", encoding="utf-8")

    def structured(self, schema: type[BaseModel], system_prompt: str, payload: Mapping[str, object]) -> tuple[BaseModel, int]:
        """Return a validated model and total Ollama tokens, compatible with old client usage."""

        if self.mode == "require_cache":
            assert self._identity is not None  # established by __init__, no network fallback is permitted
            material = self._material(schema, system_prompt, payload, self._identity)
            key = _sha256(material)
            path = self._cache_path(key)
            if not path.is_file():
                raise CacheMissError(f"required replay entry is absent: {key}")
            return self._read_entry(path, key=key, material=material, schema=schema)

        provisional_material = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "role": self.config.role,
            "configured_model": self.config.model,
            "configured_base_url": self.config.normalized_base_url,
            "options": {**self.config.options, "think": self.config.think},
            "system_prompt": system_prompt,
            "schema": schema.model_json_schema(),
            "input": dict(payload),
        }
        try:
            identity, identity_attempts = self._fetch_identity()
        except Exception as exc:
            self._write_incomplete(_sha256(provisional_material), provisional_material, getattr(exc, "attempts", []), exc)
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError(f"could not resolve Ollama model identity: {type(exc).__name__}: {exc}") from exc
        self._identity = identity
        if self._frozen is not None and identity != self._frozen.model_identity:
            error = ModelIdentityError("recorded Ollama identity differs from frozen manifest")
            self._write_incomplete(_sha256(provisional_material), provisional_material, identity_attempts, error)
            raise error
        material = self._material(schema, system_prompt, payload, identity)
        key = _sha256(material)
        path = self._cache_path(key)
        if path.is_file():
            return self._read_entry(path, key=key, material=material, schema=schema)
        request = {
            "model": self.config.model,
            "stream": False,
            "think": self.config.think,
            "format": schema.model_json_schema(),
            "options": self.config.options,
            "messages": [
                {"role": "system", "content": system_prompt + "\nJSON schema: " + _canonical(schema.model_json_schema())},
                {"role": "user", "content": _canonical(dict(payload))},
            ],
        }
        attempts = list(identity_attempts)
        try:
            with httpx.Client(
                base_url=self.config.normalized_base_url, timeout=self.config.timeout_s, trust_env=False, transport=self.transport
            ) as client:
                raw, chat_attempts = self._request_json(client, "POST", "/api/chat", payload=request)
            attempts.extend(chat_attempts)
            content = raw.get("message", {}).get("content") if isinstance(raw.get("message"), dict) else None
            if not isinstance(content, str):
                raise ReplayError("/api/chat has no message.content string")
            validated = schema.model_validate_json(content)
            input_tokens, output_tokens = raw.get("prompt_eval_count"), raw.get("eval_count")
            if type(input_tokens) is not int or type(output_tokens) is not int or input_tokens < 0 or output_tokens < 0:
                raise ReplayError("Ollama token counters must be non-negative integers")
            entry: dict[str, object] = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "request": material,
                "raw_response": raw,
                "validated_output": validated.model_dump(mode="json"),
                "total_tokens": input_tokens + output_tokens,
                "attempts": attempts,
                "complete": True,
            }
            entry["entry_sha256"] = _sha256(entry)
        except Exception as exc:
            attempts.extend(getattr(exc, "attempts", []))
            self._write_incomplete(key, material, attempts, exc)
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError(f"Ollama structured response is invalid: {type(exc).__name__}: {exc}") from exc
        self._write_immutable(path, entry)
        return self._read_entry(path, key=key, material=material, schema=schema)
