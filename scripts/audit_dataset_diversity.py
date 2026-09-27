"""Audit lexical and semantic-path diversity in a public v2 dataset split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from itertools import combinations, pairwise
from pathlib import Path

from evals.dataset_v2 import (
    PUBLIC_SPLITS,
    V2Case,
    V2Manifest,
    load_cases,
    load_manifest,
    normalize_text,
    normalized_dialogue,
    sha256_value,
)

NEAR_THRESHOLD = 0.8


def load_public_dataset(manifest_path: Path, *, expected_split: str | None = None) -> tuple[V2Manifest, list[V2Case]]:
    """Validate a public manifest before opening its case file."""
    manifest = load_manifest(manifest_path)
    if manifest.split not in PUBLIC_SPLITS:
        raise ValueError("dataset audit accepts only public dev or validation manifests")
    if expected_split is not None and manifest.split != expected_split:
        raise ValueError(f"expected {expected_split} manifest, got {manifest.split}")
    return manifest, load_cases(manifest_path, manifest)


def semantic_key(case: V2Case) -> str:
    """Hash the spoken constraints and expected state along the whole dialogue."""
    return sha256_value([{"spoken": turn.spoken.model_dump(mode="json"), "expected": turn.expected.value} for turn in case.scenario.turns])


def word_bigrams(text: str) -> set[tuple[str, str]]:
    words = normalize_text(text).split()
    return set(pairwise(words))


def dialogue_bigrams(case: V2Case) -> set[tuple[str, str]]:
    """Preserve the historical ``||`` turn marker as a bigram token."""
    return set(pairwise(normalized_dialogue(case).split()))


def bigram_jaccard(left_bigrams: set[tuple[str, str]], right_bigrams: set[tuple[str, str]]) -> float:
    union = left_bigrams | right_bigrams
    return len(left_bigrams & right_bigrams) / len(union) if union else 1.0


def build_report(manifest: V2Manifest, cases: Sequence[V2Case], *, manifest_path: Path | None = None) -> dict[str, object]:
    families: dict[str, dict[str, int]] = {}
    near: list[dict[str, object]] = []
    for family in sorted({case.scenario_family for case in cases}):
        group = [case for case in cases if case.scenario_family == family]
        semantics = Counter(semantic_key(case) for case in group)
        families[family] = {
            "cases": len(group),
            "unique_spoken_paths": len(semantics),
            "largest_same_spoken_group": max(semantics.values()),
            "grammar_cores": len({case.surface_family_id for case in group}),
            "surface_patterns": len({normalize_text(case.surface_pattern) for case in group}),
        }
        for left, right in combinations(group, 2):
            score = bigram_jaccard(dialogue_bigrams(left), dialogue_bigrams(right))
            if score >= NEAR_THRESHOLD:
                near.append(
                    {
                        "a": left.case_id,
                        "b": right.case_id,
                        "jaccard_bigrams": round(score, 4),
                        "same_spoken_path": semantic_key(left) == semantic_key(right),
                    }
                )

    report: dict[str, object] = {
        "dataset_sha256": manifest.cases_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "split": manifest.split,
        "cases": len(cases),
        "exact_normalized_dialogue_duplicates": len(cases) - len({normalized_dialogue(case) for case in cases}),
        "unique_spoken_paths": len({semantic_key(case) for case in cases}),
        "by_family": families,
        "within_family_near_pairs_jaccard_ge_08": len(near),
        "near_pairs": near,
        "near_threshold": NEAR_THRESHOLD,
        "near_metric": "Jaccard over normalized word bigrams within each scenario family",
        "limitations": [
            "Повтор озвученных условий может иметь другой скрытый профиль и oracle; это не новый смысл запроса.",
            "Jaccard не понимает отрицания и числовые ограничения; близкие пары требуют семантического review.",
            "Количество grammar core и уникальных профилей не доказывает независимость русского языка.",
        ],
    }
    if manifest_path is not None:
        report["input_manifest"] = str(manifest_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="public dev or validation manifest")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, cases = load_public_dataset(args.manifest)
    report = build_report(manifest, cases, manifest_path=args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "near_pairs"}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
