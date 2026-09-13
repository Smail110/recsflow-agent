"""Tests for configuration loading and fail-fast validation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from recagent.config.settings import (
    Settings,
    _deep_merge,
    _env_overrides,
    get_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Each test starts from a clean RECAGENT_* environment."""
    for key in list(os.environ):
        if key.startswith("RECAGENT_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()



def test_defaults_are_safe_for_dev():
    settings = Settings()
    assert settings.app_env == "dev"
    assert settings.parse_mode == "rules"
    assert settings.provider.kind == "memory"
    # A demo must not silently talk to a platform that was never configured.
    assert settings.provider.base_url.startswith("http")


def test_settings_are_immutable():
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.parse_mode = "ollama"  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {"nonexistent_key": 1},
        {"llm": {"timeout_s": -5}},
        {"llm": {"unknown": "x"}},
        {"provider": {"kind": "kafka"}},
        {"session": {"max_llm_calls": 0}},
        {"app_env": "production"},
    ],
)
def test_invalid_configuration_is_rejected(payload: dict):
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_none_section_falls_back_to_defaults():
    """`llm:` with no value in YAML parses as None and must not crash."""
    settings = Settings.model_validate({"llm": None, "provider": None})
    assert settings.llm.model == "qwen3:8b"
    assert settings.provider.kind == "memory"


def test_prod_disallows_debug_logging():
    with pytest.raises(ValidationError):
        Settings.model_validate({"app_env": "prod", "observability": {"log_level": "debug"}})


def test_prod_allows_info_logging():
    settings = Settings.model_validate({"app_env": "prod", "observability": {"log_level": "info"}})
    assert settings.app_env == "prod"


def test_relative_provider_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate({"provider": {"kind": "recsflow", "base_url": "localhost:8090"}})


def test_env_overrides_map_onto_nested_fields(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RECAGENT_MODE", "ollama")
    monkeypatch.setenv("RECAGENT_OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("RECAGENT_LOG_JSON", "false")
    monkeypatch.setenv("RECAGENT_MAX_LLM_CALLS", "4")
    overrides = _env_overrides()
    assert overrides["parse_mode"] == "ollama"
    assert overrides["llm"]["model"] == "llama3.1:8b"
    assert overrides["observability"]["log_json"] is False
    assert overrides["session"]["max_llm_calls"] == "4"

    settings = Settings.model_validate(_deep_merge(Settings().model_dump(), overrides))
    assert settings.parse_mode == "ollama"
    assert settings.session.max_llm_calls == 4


def test_env_beats_yaml_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "config.yaml"
    config.write_text("parse_mode: rules\nllm:\n  model: qwen3:8b\n", encoding="utf-8")
    monkeypatch.setenv("RECAGENT_CONFIG", str(config))
    monkeypatch.setenv("RECAGENT_MODE", "ollama")
    settings = get_settings()
    assert settings.parse_mode == "ollama"
    assert settings.llm.model == "qwen3:8b"


def test_missing_config_file_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RECAGENT_CONFIG", str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError):
        get_settings()


def test_invalid_config_reports_actionable_message(monkeypatch: pytest.MonkeyPatch):
    config = Path(os.environ.get("TEMP", ".")) / "recagent-bad-config.json"
    # A YAML parser reads JSON too, so this doubles as a format check.
    config.write_text(json.dumps({"session": {"max_llm_calls": -1}}), encoding="utf-8")
    monkeypatch.setenv("RECAGENT_CONFIG", str(config))
    with pytest.raises(RuntimeError, match="Invalid RecAgent configuration"):
        get_settings()
    config.unlink(missing_ok=True)


def test_deep_merge_does_not_mutate_inputs():
    base = {"llm": {"model": "a", "timeout_s": 1}}
    patch = {"llm": {"model": "b"}}
    merged = _deep_merge(base, patch)
    assert merged == {"llm": {"model": "b", "timeout_s": 1}}
    assert base["llm"]["model"] == "a"


def test_yaml_file_with_non_mapping_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "bad.yaml"
    config.write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setenv("RECAGENT_CONFIG", str(config))
    with pytest.raises(ValueError, match="YAML mapping"):
        get_settings()
