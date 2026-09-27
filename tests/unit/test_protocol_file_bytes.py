from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from evals.llm_protocol import canonical_sha256
from scripts.evaluate_llm_protocol import canonical_json, freeze_protocol_manifest


def _manifest() -> dict[str, object]:
    content = {"protocol_version": "byte-regression", "description": "Фильм\nСериал"}
    return {**content, "protocol_sha256": canonical_sha256(content)}


def test_new_manifest_roundtrip_hashes_written_utf8_bytes(tmp_path: Path) -> None:
    path = tmp_path / "new" / "protocol.json"
    manifest = _manifest()

    digest = freeze_protocol_manifest(path, manifest)

    payload = path.read_bytes()
    assert payload == (canonical_json(manifest) + "\n").encode("utf-8")
    assert b"\r\n" not in payload
    assert json.loads(payload) == manifest
    assert digest == hashlib.sha256(payload).hexdigest()
    assert freeze_protocol_manifest(path, manifest) == digest
    assert path.read_bytes() == payload


@pytest.mark.parametrize("formatting", ["crlf", "pretty", "no_final_newline"])
def test_existing_equal_manifest_preserves_and_hashes_actual_bytes(tmp_path: Path, formatting: str) -> None:
    path = tmp_path / "protocol.json"
    manifest = _manifest()
    normalized = (canonical_json(manifest) + "\n").encode("utf-8")
    if formatting == "crlf":
        payload = normalized.replace(b"\n", b"\r\n")
    elif formatting == "pretty":
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    else:
        payload = normalized.rstrip(b"\n")
    path.write_bytes(payload)

    digest = freeze_protocol_manifest(path, manifest)

    assert path.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    # Semantic equality must not turn a normalized content hash into a file hash.
    assert digest != hashlib.sha256(normalized).hexdigest()
    assert json.loads(payload) == manifest


def test_changed_manifest_is_rejected_without_rewriting_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    manifest = _manifest()
    freeze_protocol_manifest(path, manifest)
    original = path.read_bytes()
    changed = {**manifest, "description": "Другое условие"}
    changed["protocol_sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "protocol_sha256"})

    with pytest.raises(ValueError, match="existing protocol manifest differs"):
        freeze_protocol_manifest(path, changed)

    assert path.read_bytes() == original


def test_invalid_semantic_hash_is_rejected_before_file_creation(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    manifest = {**_manifest(), "protocol_sha256": "0" * 64}

    with pytest.raises(ValueError, match="protocol manifest hash mismatch before inference"):
        freeze_protocol_manifest(path, manifest)

    assert not path.exists()
