import hashlib
from contextlib import contextmanager

import httpx
import pytest
from scripts import download_reference_data as downloader


def test_download_rejects_wrong_hash_and_oversized_payload(monkeypatch):
    @contextmanager
    def stream(*args, **kwargs):
        yield httpx.Response(200, content=b"reference", request=httpx.Request("GET", "https://example.org"))

    monkeypatch.setattr(httpx, "stream", stream)
    digest = hashlib.sha256(b"reference").hexdigest()
    assert downloader.fetch_verified("https://example.org", digest) == b"reference"
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.fetch_verified("https://example.org", "wrong")
    with pytest.raises(ValueError, match="size limit"):
        downloader.fetch_verified("https://example.org", digest, max_bytes=3)


def test_failed_second_file_does_not_publish_partial_dataset(monkeypatch, tmp_path):
    def fetch(url, expected):
        if url.endswith("u.item"):
            raise ValueError("reference download SHA-256 mismatch")
        return b"first file"

    monkeypatch.setattr(downloader, "fetch_verified", fetch)
    destination = tmp_path / "download"
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.prepare("movielens", destination)
    assert not destination.exists()


def test_existing_destination_is_not_refreshed(monkeypatch, tmp_path):
    (tmp_path / "u.data").write_bytes(b"old")

    def unexpected(*args):
        raise AssertionError("must fail before network")

    monkeypatch.setattr(downloader, "fetch_verified", unexpected)
    with pytest.raises(FileExistsError):
        downloader.prepare("movielens", tmp_path)
    assert (tmp_path / "u.data").read_bytes() == b"old"


def test_calibration_rejects_one_byte_mutation_before_parsing(monkeypatch, tmp_path):
    from scripts.calibrate_catalog import calibrate

    original = b"reference"
    monkeypatch.setitem(
        downloader.SOURCES["movielens"],
        "files",
        {
            "u.data": hashlib.sha256(original).hexdigest(),
            "u.item": hashlib.sha256(original).hexdigest(),
        },
    )
    (tmp_path / "u.data").write_bytes(original)
    (tmp_path / "u.item").write_bytes(original)
    assert downloader.verify_movielens(tmp_path)["revision"] == downloader.MOVIELENS_REVISION
    (tmp_path / "u.item").write_bytes(b"Reference")
    with pytest.raises(ValueError, match="MovieLens SHA-256 mismatch"):
        calibrate(tmp_path)
