import json
from pathlib import Path

import pytest
from scripts import prepare_external_language as importer
from scripts.prepare_external_language import EXPECTED_FIELDS, prepare


def _write_train(path: Path) -> list[dict[str, str]]:
    rows = [
        {"id": str(i), "id_1": str(100 + i), "id_2": str(200 + i), "text_1": left, "text_2": right, "class": label}
        for i, (left, right, label) in enumerate(
            [
                ("Лёгкий фильм на вечер.", "Фильм для спокойного вечера.", "1"),
                ("Лёгкий фильм на вечер.", "Фильм для спокойного вечера.", "1"),
                ("Курс по данным.", "Обучение анализу данных.", "0"),
                ("Сериал про космос.", "Космический сериал.", "1"),
            ]
        )
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def test_offline_import_preserves_provenance_and_labels_without_test_leakage(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    rows = _write_train(source)
    manifest = prepare(source=str(source), output=tmp_path / "out", seed=42, max_pairs=3)

    assert manifest["task"] == "paraphrase identification"
    assert manifest["recommendation_ground_truth"] is False
    assert manifest["source_split"] is None
    assert manifest["source_kind"] == "unverified_local_file"
    assert manifest["source_revision"] is None
    assert not manifest["original_upstream_split_preserved"]
    assert manifest["source_row_count"] == len(rows)
    assert manifest["sample_row_count"] == 3
    assert manifest["user_count"] is None
    assert manifest["original_row_ids_labels_text_unchanged"] is True
    assert manifest["sample_file"].startswith("train_sample_seed42")
    assert not (tmp_path / "out" / "test.jsonl").exists()

    sample = [json.loads(line) for line in (tmp_path / "out" / manifest["sample_file"]).read_text(encoding="utf-8").splitlines()]
    assert all(tuple(row) == EXPECTED_FIELDS for row in sample)
    assert all(row in rows for row in sample)
    assert {row["class"] for row in sample} <= {"0", "1", "-1"}

    audit = json.loads((tmp_path / "out" / "metadata-audit.json").read_text(encoding="utf-8"))
    assert audit["normalized_pair_duplicate_group_count"] == 1
    assert audit["reversed_pair_group_count"] == 0


def test_sample_hash_and_rows_are_reproducible(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    _write_train(source)
    first = prepare(source=str(source), output=tmp_path / "first", seed=42, max_pairs=4)
    second = prepare(source=str(source), output=tmp_path / "second", seed=42, max_pairs=4)

    assert first["source_sha256"] == second["source_sha256"]
    assert first["sample_sha256"] == second["sample_sha256"]
    first_bytes = (tmp_path / "first" / first["sample_file"]).read_bytes()
    second_bytes = (tmp_path / "second" / second["sample_file"]).read_bytes()
    assert first_bytes == second_bytes


def test_reversed_pair_audit_is_reported_once(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    rows = _write_train(source)
    rows[1]["text_1"], rows[1]["text_2"] = rows[1]["text_2"], rows[1]["text_1"]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    prepare(source=str(source), output=tmp_path / "out", seed=42, max_pairs=4)
    audit = json.loads((tmp_path / "out" / "metadata-audit.json").read_text(encoding="utf-8"))

    assert audit["reversed_pair_group_count"] == 1
    assert audit["reversed_pair_rows"] == 2


def test_pinned_download_cannot_silently_change_source(tmp_path, monkeypatch):
    source = tmp_path / "fixture.jsonl"
    _write_train(source)
    monkeypatch.setattr(importer, "read_source", lambda *_args: (source.read_bytes(), importer.SOURCE_URL))
    with pytest.raises(ValueError, match="download hash"):
        prepare(output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cached_source_is_verified_by_content_not_filename(tmp_path, monkeypatch):
    source = tmp_path / "arbitrary-cache-name"
    _write_train(source)
    monkeypatch.setattr(importer, "SOURCE_SHA256", importer.sha256_bytes(source.read_bytes()))
    manifest = prepare(str(source), tmp_path / "out")
    assert manifest["source_split"] == "train"
    assert manifest["source_kind"] == "verified_upstream_train"
