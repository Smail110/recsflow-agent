"""Reproduce the tracked legacy synthetic intent snapshot without migrating it.

This module preserves the algorithm from commit ``271baf9``, which added the
tracked snapshot.  Commit ``24cd191`` already consumed an extra catalog-title
random draw and therefore cannot reproduce that file.  This module is separate
from ``prepare_hf_dataset`` because the current training format requires
provenance fields and group-separated train/validation splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

LEGACY_SOURCE_COMMIT = "271baf958b4e28cc370d45b6129ac8abd1b650b3"
LEGACY_SEED = 42
LEGACY_EXAMPLES_PER_LABEL = 40
LEGACY_CANONICAL_SHA256 = "b03da67e46a8a3009332ea616db9bb0cd8d405f14d25441adbbf070080397877"
LEGACY_REFERENCE = Path("data/dialogues.jsonl")
DEFAULT_OUTPUT = Path("artifacts/reproducibility/legacy-dialogues.jsonl")

TEMPLATES = {
    "discovery": [
        "Подбери {kind} {genre}",
        "Что посмотреть сегодня?",
        "Хочу {genre} без лишней мрачности",
    ],
    "similar": ["Найди похожее на Тайна старого маяка", "Что-то в духе этого сериала"],
    "mood": ["Хочу лёгкое на вечер", "Подбери что-нибудь после тяжёлого дня"],
    "navigation": ["Покажи в каталоге {genre}", "Найди курс по {genre}"],
}


class LegacyReproductionError(RuntimeError):
    """The historical generator or its tracked reference no longer matches."""


def build_legacy_rows() -> list[dict[str, object]]:
    """Run the fixed seed-42, 40-examples-per-label historical algorithm."""

    rng = random.Random(LEGACY_SEED)
    genres = ["детектив", "комедию", "фантастику", "машинному обучению", "python"]
    kinds = ["сериал", "фильм", "курс"]
    rows: list[dict[str, object]] = []
    for label, templates in TEMPLATES.items():
        for _ in range(LEGACY_EXAMPLES_PER_LABEL):
            template = rng.choice(templates)
            text = template.format(
                kind=rng.choice(kinds),
                genre=rng.choice(genres),
            )
            rows.append({"text": text, "label": label, "synthetic": True, "seed": LEGACY_SEED})
    rng.shuffle(rows)
    return rows


def render_legacy_bytes(rows: list[dict[str, object]]) -> bytes:
    """Render repository-canonical UTF-8 JSONL with a final LF."""

    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")


def _canonicalize_line_endings(payload: bytes) -> bytes:
    # A stale Windows worktree may retain CRLF even though .gitattributes says
    # eol=lf.  Line endings are the only normalization allowed here.
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_legacy_reference(reference: Path = LEGACY_REFERENCE) -> dict[str, object]:
    """Fail unless generator and tracked legacy content equal the frozen hash."""

    rows = build_legacy_rows()
    generated = render_legacy_bytes(rows)
    generated_hash = _sha256(generated)
    if generated_hash != LEGACY_CANONICAL_SHA256:
        raise LegacyReproductionError(f"legacy generator drift: expected {LEGACY_CANONICAL_SHA256}, got {generated_hash}")

    reference_bytes = _canonicalize_line_endings(reference.read_bytes())
    reference_hash = _sha256(reference_bytes)
    if reference_hash != LEGACY_CANONICAL_SHA256 or reference_bytes != generated:
        raise LegacyReproductionError(f"legacy reference mismatch at {reference}: expected {LEGACY_CANONICAL_SHA256}, got {reference_hash}")

    labels = Counter(str(row["label"]) for row in rows)
    unique_texts = len({str(row["text"]) for row in rows})
    return {
        "source_commit": LEGACY_SOURCE_COMMIT,
        "seed": LEGACY_SEED,
        "examples_per_label": LEGACY_EXAMPLES_PER_LABEL,
        "rows": len(rows),
        "labels": dict(sorted(labels.items())),
        "unique_texts": unique_texts,
        "duplicate_rows_by_text": len(rows) - unique_texts,
        "canonical_sha256": generated_hash,
    }


def reproduce(output: Path, reference: Path = LEGACY_REFERENCE) -> dict[str, object]:
    """Verify the tracked reference first, then write an identical LF artifact."""

    if output.resolve() == reference.resolve():
        raise LegacyReproductionError("output must differ from the tracked legacy reference")
    report = verify_legacy_reference(reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(render_legacy_bytes(build_legacy_rows()))
    return {**report, "reference": str(reference), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=LEGACY_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = reproduce(args.output, args.reference)
    except (LegacyReproductionError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "reproduced", **report}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
