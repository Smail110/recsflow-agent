"""Download pinned public reference data; never commit the downloaded files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

MOVIELENS_REVISION = "4f9df57cf6bde70d17372fed165d3331b23e7c68"
CCPE_REVISION = "2c9cd30f33f3a154b5a27d015333679262ff36f5"
SOURCES = {
    "movielens": {
        "revision": MOVIELENS_REVISION,
        "origin": "external_real_ratings_and_metadata",
        "license": "GroupLens research terms; not the mirror's Apache-2.0 label",
        "license_url": "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        "license_readable_mirror": "https://huggingface.co/datasets/harisarang/ml-100k/raw/main/raw/README",
        "files": {
            "u.data": "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490",
            "u.item": "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701",
        },
        "base_url": f"https://huggingface.co/datasets/includeno/movielens-100k/resolve/{MOVIELENS_REVISION}/",
        "purpose": "Optional reproduction of derived catalog calibration; not recommendation ground truth.",
    },
    "ccpe": {
        "revision": CCPE_REVISION,
        "origin": "external_human_wizard_of_oz_dialogues",
        "license": "CC-BY-4.0",
        "license_url": f"https://raw.githubusercontent.com/google-research-datasets/ccpe/{CCPE_REVISION}/README.md",
        "files": {"data.json": "4ff051ea7ea60cf0f480c911c7e2cfed56434e2e2c9ea8965ac5e26365773f0a"},
        "base_url": f"https://raw.githubusercontent.com/google-research-datasets/ccpe/{CCPE_REVISION}/",
        "purpose": "English dialogue research; not Russian labels or hidden-preference utility ground truth.",
    },
}


def fetch_verified(url: str, expected_sha256: str, *, max_bytes: int = 8_000_000, timeout: float = 30) -> bytes:
    if max_bytes < 1 or timeout <= 0:
        raise ValueError("positive size and timeout required")
    chunks, size = [], 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("reference download exceeds size limit")
            chunks.append(chunk)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("reference download SHA-256 mismatch")
    return payload


def prepare(name: str, output_dir: Path) -> dict:
    source = SOURCES[name]
    if any((output_dir / filename).exists() for filename in [*source["files"], f"{name}.manifest.json"]):
        raise FileExistsError("reference destination already contains dataset files; choose a fresh output directory")
    # Verify every file before any data file is published to the output directory.
    downloaded = {filename: fetch_verified(source["base_url"] + filename, expected) for filename, expected in source["files"].items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, payload in downloaded.items():
        path = output_dir / filename
        path.write_bytes(payload)
        files.append(
            {"file": filename, "url": source["base_url"] + filename, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        )
    manifest = {key: value for key, value in source.items() if key not in {"files", "base_url"}}
    manifest.update(dataset=name, files=files, preprocessing="none; source bytes preserved", seed=None)
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_movielens(raw_dir: Path) -> dict:
    """Bind derived calibration to the same pinned bytes as the downloader."""
    source = SOURCES["movielens"]
    files = []
    for filename, expected in source["files"].items():
        payload = (raw_dir / filename).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"MovieLens SHA-256 mismatch: {filename}")
        files.append({"file": filename, "sha256": digest, "bytes": len(payload), "url": source["base_url"] + filename})
    provenance = {"dataset": "movielens", "revision": source["revision"], "files": files, "license": source["license"]}
    canonical = json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    provenance["input_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(SOURCES))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or (Path("data/raw") if args.dataset == "movielens" else Path("data/raw/ccpe"))
    print(json.dumps(prepare(args.dataset, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
