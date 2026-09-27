"""Audit lexical train/dev proximity for the experimental evidence pairs.

This is a string-level diagnostic, not a semantic independence test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {re.sub(r"\W+", " ", row["premise"].casefold()).strip() for row in rows}


def trigrams(message: str) -> set[str]:
    padded = f" {message} "
    return {padded[index : index + 3] for index in range(max(0, len(padded) - 2))}


def audit(train_path: Path, dev_path: Path, *, threshold: float = 0.8) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    train, dev = messages(train_path), messages(dev_path)
    if not train or not dev or "" in train or "" in dev:
        raise ValueError("splits must contain non-empty messages")
    train_grams = {message: trigrams(message) for message in sorted(train)}
    nearest = []
    for dev_message in sorted(dev):
        grams = trigrams(dev_message)
        train_message, score = max(
            ((message, len(grams & other) / len(grams | other)) for message, other in train_grams.items()),
            key=lambda item: item[1],
        )
        nearest.append({"dev_message": dev_message, "nearest_train_message": train_message, "jaccard_char3": score})
    scores = [item["jaccard_char3"] for item in nearest]
    return {
        "method": "casefold, Unicode word normalization, character trigram Jaccard; maximum train match for each unique dev message",
        "caveat": "Post-run lexical audit; does not establish semantic independence or select a model.",
        "threshold_exploratory": threshold,
        "train_path": train_path.as_posix(),
        "train_sha256": sha256(train_path),
        "dev_path": dev_path.as_posix(),
        "dev_sha256": sha256(dev_path),
        "train_unique_messages": len(train),
        "dev_unique_messages": len(dev),
        "exact_matches": len(train & dev),
        "near_matches_at_threshold": sum(score >= threshold for score in scores),
        "max_similarity": max(scores),
        "median_nearest_similarity": statistics.median(scores),
        "top5": sorted(nearest, key=lambda item: item["jaccard_char3"], reverse=True)[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.train, args.dev)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "top5"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
