"""Prepare a pinned, research-only ParaPhraser language challenge sample.

Only the upstream ``train.jsonl`` split is read.  The source rows are copied
byte-for-byte and the sample contains original rows without added labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_REVISION = "9842f8af30e47fd0e24dd967e0f48ebd755d98a7"
SOURCE_SHA256 = "c5b486526eb6eefe3d723e31ea80114cf1f7a60529024af610c3628ee5271c18"
SOURCE_URL = f"https://huggingface.co/datasets/merionum/ru_paraphraser/resolve/{SOURCE_REVISION}/train.jsonl?download=true"
SOURCE_CARD_URL = f"https://huggingface.co/datasets/merionum/ru_paraphraser/tree/{SOURCE_REVISION}"
DEFAULT_OUTPUT = Path("artifacts/external-language-20260914")
EXPECTED_FIELDS = ("id", "id_1", "id_2", "text_1", "text_2", "class")
MAX_SOURCE_BYTES = 10 * 1024 * 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def parse_source(raw: bytes) -> tuple[list[bytes], list[dict[str, Any]]]:
    lines = raw.splitlines(keepends=True)
    if not lines or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("source is empty or exceeds maxbytes")
    rows: list[dict[str, Any]] = []
    data_lines: list[bytes] = []
    for line_number, raw_line in enumerate(lines, 1):
        payload = raw_line.rstrip(b"\r\n")
        if not payload.strip():
            continue
        try:
            row = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL at line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != set(EXPECTED_FIELDS):
            raise ValueError(f"unexpected fields at line {line_number}: {list(row) if isinstance(row, dict) else type(row)}")
        if not all(isinstance(row[field], str) for field in ("id", "id_1", "id_2", "text_1", "text_2", "class")):
            raise ValueError(f"non-string source value at line {line_number}")
        if row["class"] not in {"-1", "0", "1"}:
            raise ValueError(f"unknown paraphrase label at line {line_number}")
        data_lines.append(raw_line)
        rows.append(row)
    if not rows:
        raise ValueError("source has no rows")
    return data_lines, rows


def audit_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oriented: dict[tuple[str, str], list[int]] = defaultdict(list)
    canonical: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        left, right = normalize_text(row["text_1"]), normalize_text(row["text_2"])
        oriented[(left, right)].append(index)
        canonical[tuple(sorted((left, right)))].append(index)
    duplicate_groups = [indices for indices in canonical.values() if len(indices) > 1]
    reversed_groups = []
    for key, indices in canonical.items():
        reverse = (key[1], key[0])
        if key[0] < key[1] and reverse in oriented and key in oriented:
            reversed_groups.append(sorted(set(indices + oriented[reverse])))
    return {
        "rows_audited": len(rows),
        "normalized_pair_duplicate_group_count": len(duplicate_groups),
        "normalized_pair_duplicate_rows": sum(len(group) for group in duplicate_groups),
        "reversed_pair_group_count": len(reversed_groups),
        "reversed_pair_rows": sum(len(group) for group in reversed_groups),
        "label_counts": dict(sorted(Counter(row["class"] for row in rows).items())),
    }


def read_source(source: str | None, timeout: float, maxbytes: int) -> tuple[bytes, str]:
    if source:
        source_path = Path(source)
        if source_path.stat().st_size > maxbytes:
            raise ValueError(f"offline source exceeds maxbytes={maxbytes}")
        raw = source_path.read_bytes()
        if len(raw) > maxbytes:
            raise ValueError(f"offline source exceeds maxbytes={maxbytes}")
        return raw, str(source_path.resolve())
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "recagent-external-language/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > maxbytes:
            raise ValueError(f"remote source exceeds maxbytes={maxbytes}")
        chunks: list[bytes] = []
        total = 0
        while chunk := response.read(64 * 1024):
            total += len(chunk)
            if total > maxbytes:
                raise ValueError(f"remote source exceeds maxbytes={maxbytes}")
            chunks.append(chunk)
    return b"".join(chunks), SOURCE_URL


def prepare(
    source: str | None = None,
    output: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    max_pairs: int = 200,
    timeout: float = 30.0,
    maxbytes: int = MAX_SOURCE_BYTES,
) -> dict[str, Any]:
    if max_pairs < 1 or max_pairs > 200:
        raise ValueError("max_pairs must be between 1 and 200")
    if timeout <= 0 or maxbytes < 1 or maxbytes > MAX_SOURCE_BYTES:
        raise ValueError("timeout and maxbytes must be positive and maxbytes within the source limit")
    raw, source_used = read_source(source, timeout, maxbytes)
    source_hash = sha256_bytes(raw)
    verified = source_hash == SOURCE_SHA256
    if source is None and not verified:
        raise ValueError("download hash differs from pinned upstream train source")
    lines, rows = parse_source(raw)
    indices = sorted(random.Random(seed).sample(range(len(rows)), min(max_pairs, len(rows))))
    sample_rows = [rows[index] for index in indices]
    sample_lines = [lines[index] for index in indices]
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "train.jsonl"
    sample_path = output / f"train_sample_seed{seed}_max{max_pairs}.jsonl"
    raw_path.write_bytes(raw)
    sample_path.write_bytes(b"".join(sample_lines))
    sample_hash = sha256_bytes(sample_path.read_bytes())
    audit = audit_pairs(sample_rows)
    (output / "metadata-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "manifest_version": "1.0",
        "task": "paraphrase identification",
        "recommendation_ground_truth": False,
        "source_kind": "verified_upstream_train" if verified else "unverified_local_file",
        "source_attribution": "ParaPhraser, Pivovarova et al. (2017); upstream maintainer merionum"
        if verified
        else "Непроверенный локальный источник",
        "source_url": SOURCE_URL if verified else None,
        "source_card_url": SOURCE_CARD_URL if verified else None,
        "source_revision": SOURCE_REVISION if verified else None,
        "source_split": "train" if verified else None,
        "source_file": "train.jsonl",
        "source_used": source_used,
        "source_license_statement": "MIT заявлена upstream dataset card; это заявление карточки, а не самостоятельная гарантия прав на исходные новостные заголовки."
        if verified
        else "Не установлена для локального файла.",
        "source_license_statement_source": SOURCE_CARD_URL if verified else None,
        "source_sha256": source_hash,
        "source_bytes": len(raw),
        "source_row_count": len(rows),
        "source_fields": list(EXPECTED_FIELDS),
        "sample_file": sample_path.name,
        "sample_seed": seed,
        "sample_max_pairs": max_pairs,
        "sample_row_count": len(sample_rows),
        "sample_sha256": sample_hash,
        "sample_indices_zero_based": indices,
        "sample_labels": dict(sorted(Counter(row["class"] for row in sample_rows).items())),
        "original_row_ids_labels_text_unchanged": True,
        "original_upstream_split_preserved": verified,
        "user_count": None,
        "user_count_note": "Dataset has no user field; user count is unknown and must not be inferred.",
        "development_policy": "Только train и только language inspection; validation/test не читались."
        if verified
        else "Исходный split локального файла неизвестен; не считать защищённым от утечки.",
        "oracle_policy": "Не использовать для acceptable_set(theta, query), recommendation labels или success oracle.",
        "audit_file": "metadata-audit.json",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="offline train.jsonl path; skips network")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--maxbytes", type=int, default=MAX_SOURCE_BYTES)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.source, args.output, args.seed, args.max_pairs, args.timeout, args.maxbytes)
    except (OSError, ValueError, TimeoutError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_rows": manifest["source_row_count"],
                "sample_rows": manifest["sample_row_count"],
                "source_sha256": manifest["source_sha256"],
                "sample_sha256": manifest["sample_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
