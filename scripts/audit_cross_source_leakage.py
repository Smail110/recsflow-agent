"""Audit exact and near lexical overlap across explicit public text sources."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from evals.adversarial_ru import CASES, corpus_sha256
from evals.dataset_v2 import normalize_text, sha256_bytes

from scripts.audit_dataset_diversity import load_public_dataset

NEAR_THRESHOLD = 0.8
MIN_NEAR_UNION_TOKENS = 4
TextRows = list[tuple[str, str]]


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    payload = path.read_bytes()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: source is empty")
    return payload, rows


def _required_text(row: dict[str, object], field: str, *, path: Path, row_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{row_number}: {field} must be a non-empty string")
    return value


def load_training_smoke(path: Path) -> tuple[bytes, TextRows]:
    payload, rows = _read_jsonl(path)
    texts = [(str(index), _required_text(row, "text", path=path, row_number=index + 1)) for index, row in enumerate(rows)]
    return payload, texts


def load_external_language(path: Path) -> tuple[bytes, TextRows]:
    payload, rows = _read_jsonl(path)
    texts: TextRows = []
    for index, row in enumerate(rows, start=1):
        if "id" not in row:
            raise ValueError(f"{path}:{index}: id is required")
        row_id = str(row["id"])
        for side in (1, 2):
            texts.append(
                (
                    f"{row_id}:{side}",
                    _required_text(row, f"text_{side}", path=path, row_number=index),
                )
            )
    return payload, texts


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def compare_groups(left: TextRows, right: TextRows) -> dict[str, object]:
    prepared_left = [(key, normalize_text(value), set(normalize_text(value).split())) for key, value in left]
    prepared_right = [(key, normalize_text(value), set(normalize_text(value).split())) for key, value in right]
    exact: list[list[str]] = []
    near: list[dict[str, object]] = []
    for key_left, text_left, tokens_left in prepared_left:
        for key_right, text_right, tokens_right in prepared_right:
            if text_left == text_right:
                exact.append([key_left, key_right])
                continue
            union = tokens_left | tokens_right
            if len(union) < MIN_NEAR_UNION_TOKENS:
                continue
            score = len(tokens_left & tokens_right) / len(union)
            if score >= NEAR_THRESHOLD:
                near.append({"left": key_left, "right": key_right, "token_jaccard": round(score, 4)})
    return {
        "exact_pair_count": len(exact),
        "near_pair_count": len(near),
        "exact_pairs": exact,
        "near_pairs": near,
    }


def build_report(
    *,
    dev_manifest: Path,
    validation_manifest: Path,
    training_smoke: Path,
    external_language: Path,
) -> dict[str, object]:
    dev_meta, dev_cases = load_public_dataset(dev_manifest, expected_split="dev")
    validation_meta, validation_cases = load_public_dataset(validation_manifest, expected_split="validation")
    training_payload, training_rows = load_training_smoke(training_smoke)
    external_payload, external_rows = load_external_language(external_language)

    groups: dict[str, TextRows] = {
        "v2_dev": [(f"{case.case_id}:{turn.index}", turn.utterance) for case in dev_cases for turn in case.scenario.turns],
        "v2_validation": [(f"{case.case_id}:{turn.index}", turn.utterance) for case in validation_cases for turn in case.scenario.turns],
        "training_smoke": training_rows,
        "external_language": external_rows,
        "adversarial_dev": [(case.case_id, case.utterance) for case in CASES],
    }
    source_hashes = {
        "v2_dev": dev_meta.cases_sha256,
        "v2_validation": validation_meta.cases_sha256,
        "training_smoke": sha256_bytes(training_payload),
        "external_language": sha256_bytes(external_payload),
        "adversarial_dev": corpus_sha256(),
    }
    comparisons = {f"{left}__{right}": compare_groups(groups[left], groups[right]) for left, right in combinations(groups, 2)}
    return {
        "unit": "individual utterance/text, not full scenario",
        "inputs": {
            "v2_dev_manifest": str(dev_manifest),
            "v2_validation_manifest": str(validation_manifest),
            "training_smoke_jsonl": str(training_smoke),
            "external_language_jsonl": str(external_language),
            "adversarial_dev": "evals.adversarial_ru:CASES",
        },
        "source_hashes": source_hashes,
        "source_text_counts": {name: len(rows) for name, rows in groups.items()},
        "comparisons": comparisons,
        "near_threshold": NEAR_THRESHOLD,
        "near_metric": "Jaccard over normalized token sets across distinct sources",
        "minimum_near_union_tokens": MIN_NEAR_UNION_TOKENS,
        "limitations": [
            "Только лексический аудит; совпадение короткого ответа не доказывает утечку полного сценария.",
            "V2-входы ограничены публичными dev и validation; final holdout не читается.",
            "Роль и split вспомогательных JSONL подтверждаются их собственными manifests, не этим лексическим аудитом.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--training-smoke", type=Path, required=True)
    parser.add_argument("--external-language", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_report(
        dev_manifest=args.dev_manifest,
        validation_manifest=args.validation_manifest,
        training_smoke=args.training_smoke,
        external_language=args.external_language,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "counts": report["source_text_counts"],
        "pairs": {
            name: {key: value for key, value in row.items() if key.endswith("_count")} for name, row in report["comparisons"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
