"""Central configuration with fail-fast validation.

Everything tunable lives here so that a production switch is a config change,
not a code change. Values are read from environment variables (prefix RECAGENT_)
or a YAML file pointed to by RECAGENT_CONFIG.

Design note: unknown environment keys are rejected on purpose. A typo like
RECAGENT_LLM_BUDGET=8 silently doing nothing is worse than a startup failure.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ParseMode = Literal["rules", "ollama"]
ProviderKind = Literal["memory", "recsflow"]


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    timeout_s: float = Field(default=45.0, gt=0, le=300)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=700, gt=0, le=8192)
    seed: int = 42


class SessionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sessions: int = Field(default=500, gt=0)
    ttl_s: int = Field(default=3600, gt=0)
    max_llm_calls: int = Field(default=8, gt=0)
    max_token_budget: int = Field(default=100_000, gt=0)
    max_clarifications: int = Field(default=3, ge=0)


class ProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind = "memory"
    base_url: str = "http://127.0.0.1:8090"
    api_key: str | None = None
    timeout_s: float = Field(default=5.0, gt=0, le=60)
    max_retries: int = Field(default=2, ge=0, le=5)
    candidate_limit: int = Field(default=100, gt=0, le=1000)
    circuit_breaker_threshold: int = Field(default=5, gt=0)
    circuit_breaker_cooldown_s: float = Field(default=30.0, gt=0)
    metadata_cache_ttl_s: float = Field(default=300.0, ge=0)
    metadata_cache_size: int = Field(default=10_000, gt=0)


class ObservabilitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_json: bool = True
    otel_enabled: bool = False
    otel_endpoint: str | None = None
    service_name: str = "recagent"
    service_version: str = "0.2.0"


class CostSettings(BaseModel):
    """Prices per 1M tokens, used only for reporting a documented cost estimate."""

    model_config = ConfigDict(extra="forbid")

    price_in_per_mtok: float = Field(default=0.0, ge=0)
    price_out_per_mtok: float = Field(default=0.0, ge=0)
    gpu_hour_cost: float = Field(default=0.0, ge=0)
    currency: str = "USD"


class Settings(BaseModel):
    """Root settings object. Immutable after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_env: Literal["dev", "staging", "prod"] = "dev"
    parse_mode: ParseMode = "rules"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    cost: CostSettings = Field(default_factory=CostSettings)

    @field_validator("llm", "session", "provider", "observability", "cost", mode="before")
    @classmethod
    def _none_to_default(cls, value: Any) -> Any:
        # An explicit `key:` with no value in YAML parses as None; treat as default.
        return {} if value is None else value

    @model_validator(mode="after")
    def _check_cross_field(self) -> Settings:
        # Debug logs can contain user utterances and extracted slots. This must not
        # depend on the parse mode: the risk is the log level itself, in any mode.
        if self.app_env == "prod" and self.observability.log_level == "debug":
            raise ValueError("debug logging is not allowed in prod (may leak user content)")
        if self.provider.kind == "recsflow" and not self.provider.base_url.startswith(("http://", "https://")):
            raise ValueError(f"provider.base_url must be an absolute URL, got {self.provider.base_url!r}")
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _env_overrides() -> dict[str, Any]:
    """Map a flat set of RECAGENT_* environment variables onto the nested model."""
    env_map: dict[str, tuple[str, ...]] = {
        "APP_ENV": ("app_env",),
        "MODE": ("parse_mode",),
        "OLLAMA_URL": ("llm", "url"),
        "OLLAMA_MODEL": ("llm", "model"),
        "LLM_TIMEOUT_S": ("llm", "timeout_s"),
        "PROVIDER": ("provider", "kind"),
        "PROVIDER_URL": ("provider", "base_url"),
        "PROVIDER_API_KEY": ("provider", "api_key"),
        "LOG_LEVEL": ("observability", "log_level"),
        "LOG_JSON": ("observability", "log_json"),
        "OTEL_ENABLED": ("observability", "otel_enabled"),
        "OTEL_ENDPOINT": ("observability", "otel_endpoint"),
        "MAX_LLM_CALLS": ("session", "max_llm_calls"),
        "SESSION_TTL_S": ("session", "ttl_s"),
    }
    out: dict[str, Any] = {}
    for env_name, path in env_map.items():
        raw = os.getenv(f"RECAGENT_{env_name}")
        if raw is None:
            continue
        node = out
        for key in path[:-1]:
            node = node.setdefault(key, {})
        leaf = path[-1]
        if leaf in ("log_json", "otel_enabled"):
            node[leaf] = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            node[leaf] = raw
    return out


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from RECAGENT_CONFIG (YAML) then environment overrides.

    Raises on any invalid or unknown key: fail fast at startup, never at request time.
    """
    data: dict[str, Any] = {}
    config_path = os.getenv("RECAGENT_CONFIG")
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"RECAGENT_CONFIG points to a missing file: {path}")
        data = _load_yaml(path)
    data = _deep_merge(data, _env_overrides())
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        # Surface a single actionable message rather than a pydantic traceback.
        lines = [f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()]
        raise RuntimeError("Invalid RecAgent configuration:\n" + "\n".join(lines)) from exc


def reset_settings_cache() -> None:
    """Clear the memoized settings. Used by tests and config reloads."""
    get_settings.cache_clear()
