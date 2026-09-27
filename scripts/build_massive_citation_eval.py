"""Build an open, citation-aware component check from MASSIVE Russian slot spans.

The original slot spans were annotated in MASSIVE. The support/wrong-citation
pairs below are deterministic *silver* transformations, not human NLI labels.
No product evaluation data, model predictions or hidden splits are read.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_REVISION = "6d9e43756289aa637fe5927c51a9e4329528deec"
SOURCE_URL = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    f"{SOURCE_REVISION}/ru-RU/massive-validation.parquet"
)
SOURCE_SHA256 = "49e6bcaa82d747583052e04f150b99a059eaa15834fced08aaec5548c58a8be4"
SOURCE_BYTES = 201_738
SOURCE_ROWS = 2_033
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://github.com/alexa/massive/blob/main/NOTICE.md"
SOURCE_CITATION = "FitzGerald et al. (2022), MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset."
SLOT_MARKUP = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+)", flags=re.DOTALL)


@dataclass(frozen=True)
class SlotSpan:
    role: str
    value: str
    start: int
    end: int


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_annotated_utterance(utterance: str, annotated: str) -> list[SlotSpan]:
    """Recover exact Unicode offsets; reject malformed or lossy annotations."""
    pieces: list[str] = []
    spans: list[SlotSpan] = []
    cursor = 0
    output_length = 0
    while cursor < len(annotated):
        char = annotated[cursor]
        if char == "]":
            raise ValueError("unmatched closing bracket")
        if char != "[":
            pieces.append(char)
            cursor += 1
            output_length += 1
            continue
        close = annotated.find("]", cursor + 1)
        if close < 0 or "[" in annotated[cursor + 1 : close]:
            raise ValueError("nested or unclosed slot bracket")
        match = SLOT_MARKUP.fullmatch(annotated[cursor + 1 : close])
        if match is None:
            raise ValueError("invalid slot markup")
        role, value = match.groups()
        if not value.strip():
            raise ValueError("empty slot value")
        pieces.append(value)
        spans.append(SlotSpan(role, value, output_length, output_length + len(value)))
        output_length += len(value)
        cursor = close + 1
    if "".join(pieces) != utterance:
        raise ValueError("annotated utterance does not reproduce raw utterance")
    if any(utterance[span.start : span.end] != span.value for span in spans):
        raise ValueError("slot offset mismatch")
    return spans


def tag_citation(message: str, span: SlotSpan) -> str:
    if message[span.start : span.end] != span.value or "<evidence>" in message.casefold():
        raise ValueError("invalid citation span or reserved marker in utterance")
    return message[: span.start] + "<evidence>" + span.value + "</evidence>" + message[span.end :]


def slot_quality(judgments: Any) -> bool:
    """Require a strict majority of supplied MASSIVE slot-validity judgments."""
    if not isinstance(judgments, dict):
        return False
    scores = judgments.get("slots_score")
    if not isinstance(scores, list) or len(scores) < 2:
        return False
    return sum(score == 1 for score in scores) > len(scores) / 2


def _claim(span: SlotSpan) -> str:
    # The identical hypothesis is paired with two different highlighted spans.
    return f"Выделенная цитата подтверждает: значение «{span.value}» относится к параметру «{span.role}»."


def build_pairs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce one matched positive/negative pair per eligible annotated span."""
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    cohort_ids: set[str] = set()
    source_ids: set[str] = set()
    considered_spans = 0
    for record in sorted(records, key=lambda row: str(row.get("id", ""))):
        source_id = str(record.get("id", ""))
        if not source_id or source_id in source_ids:
            raise ValueError("missing or duplicate MASSIVE source id")
        source_ids.add(source_id)
        if record.get("locale") != "ru-RU" or record.get("partition") != "dev":
            raise ValueError("source is not the pinned Russian validation partition")
        utterance = record.get("utt")
        annotated = record.get("annot_utt")
        if not isinstance(utterance, str) or not isinstance(annotated, str) or not utterance:
            excluded["invalid_text"] += 1
            continue
        if "<evidence>" in utterance.casefold() or "</evidence>" in utterance.casefold():
            excluded["reserved_marker"] += 1
            continue
        try:
            spans = parse_annotated_utterance(utterance, annotated)
        except ValueError:
            excluded["invalid_markup_or_alignment"] += 1
            continue
        if not slot_quality(record.get("judgments")):
            excluded["slot_quality_not_majority_valid"] += 1
            continue
        if len(spans) < 2:
            excluded["fewer_than_two_slot_spans"] += 1
            continue
        record_pairs = 0
        for index, primary in enumerate(spans):
            considered_spans += 1
            # A different annotated slot gives a within-utterance citation
            # counterfactual. This is silver; its semantic validity is not gold.
            alternatives = [
                other for other in spans if other.role != primary.role and other.value.casefold() != primary.value.casefold()
            ]
            if not alternatives:
                continue
            wrong = alternatives[0]
            group_id = f"massive-ru-validation:{source_id}:{index}"
            if group_id in cohort_ids:
                raise ValueError("duplicate derived group id")
            cohort_ids.add(group_id)
            common = {
                "group_id": group_id,
                "source_id": source_id,
                "locale": "ru-RU",
                "source_partition": "dev",
                "message": utterance,
                "role": primary.role,
                "value": primary.value,
                "hypothesis": _claim(primary),
                "label_origin": "human_slot_span_plus_deterministic_silver_counterfactual",
            }
            for label, cited in (("support", primary), ("unknown", wrong)):
                rows.append(
                    {
                        **common,
                        "id": f"{group_id}:{label}",
                        "label": label,
                        "source_start": cited.start,
                        "source_end": cited.end,
                        "source_text": cited.value,
                        "premise": tag_citation(utterance, cited),
                    }
                )
            record_pairs += 1
        if not record_pairs:
            excluded["no_distinct_role_value_counterfactual"] += 1
    coverage = {
        "source_utterances": len(records),
        "eligible_utterances": len({row["source_id"] for row in rows}),
        "excluded_utterances_by_reason": dict(sorted(excluded.items())),
        "considered_spans_after_row_filters": considered_spans,
        "matched_groups": len(cohort_ids),
        "derived_rows": len(rows),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "cluster_unit": "MASSIVE utterance id",
    }
    if coverage["eligible_utterances"] + sum(excluded.values()) != len(records):
        raise AssertionError("coverage denominator mismatch")
    return rows, coverage


def _source_bytes(path: Path | None) -> bytes:
    if path is not None:
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read(SOURCE_BYTES + 1)
    if len(content) != SOURCE_BYTES or sha256(content) != SOURCE_SHA256:
        raise ValueError("MASSIVE source byte count or SHA-256 does not match pinned revision")
    return content


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build(output_dir: Path, *, source_path: Path | None = None) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    # Parquet is only an input dependency of this external evaluation builder.
    import pyarrow.parquet as pq

    content = _source_bytes(source_path)
    table = pq.read_table(io.BytesIO(content))
    if table.num_rows != SOURCE_ROWS:
        raise ValueError("MASSIVE validation row count mismatch")
    rows, coverage = build_pairs(table.to_pylist())
    if not rows:
        raise ValueError("no eligible citation pairs")
    output_dir.mkdir(parents=True)
    output = b"".join(_json_bytes(row) for row in rows)
    (output_dir / "pairs.jsonl").write_bytes(output)
    manifest = {
        "schema_version": "massive-citation-eval-v1",
        "source": {
            "name": "MASSIVE 1.1 ru-RU validation",
            "repository": "AmazonScience/massive",
            "revision": SOURCE_REVISION,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
            "rows": SOURCE_ROWS,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "citation": SOURCE_CITATION,
        },
        "derivation": {
            "version": "deterministic-human-slot-span-citation-pair-v1",
            "builder_sha256": sha256(Path(__file__).read_bytes()),
            "positive": "original annotated slot span highlighted; role and value from same human annotation",
            "negative": "same hypothesis, different annotated slot with distinct role and surface value, highlighted in same utterance",
            "negative_label": "unknown (silver citation non-support, not contradiction)",
            "judgment_filter": "strict majority slots_score == 1",
            "selection": "all eligible spans in source-id order; first other distinct-role/value span in source order",
            "seed": None,
            "training_use": False,
            "product_dev_contract_use": False,
            "gold_status": "human source slot spans; transformed NLI labels are silver",
        },
        "coverage": coverage,
        "files": {"pairs.jsonl": {"sha256": sha256(output), "bytes": len(output)}},
        "limitations": [
            "Localized voice-assistant requests, not recommendation dialogue or organic Russian product traffic.",
            "MASSIVE slot annotation supports positive span-role/value links; wrong-citation labels are automatic silver.",
            "No contradiction class, no paraphrase-only values, and no complete recommendation success measure.",
            "Open source split may have been seen during pretraining; independence from model training cannot be proven.",
            "Repeated slot values, ambiguous spans, and imperfect source annotation can make silver labels noisy.",
        ],
    }
    (output_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Optional local copy of the pinned parquet bytes")
    args = parser.parse_args()
    manifest = build(args.output_dir, source_path=args.source)
    print(json.dumps({"output_dir": str(args.output_dir), "coverage": manifest["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
