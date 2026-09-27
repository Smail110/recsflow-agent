"""Convert frozen schema-first synthetic requests into three-way evidence pairs.

The converter uses only generator targets and declared schema aliases. It never
reads product evaluations, agent output, or a sealed split. Its labels describe
the synthetic generator's intent, not human annotation or real traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "lora" / "semantic_extraction_v2"
LABELS = ("support", "contradiction", "unknown")
OPERATIONS = ("set", "exclude", "include")
OPPOSITE = {"set": "exclude", "exclude": "set", "include": "exclude"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", text.casefold().replace("ё", "е")).split())


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load(path: Path, split: str) -> list[dict[str, Any]]:
    if any(token in part.casefold() for part in path.parts for token in ("blind", "holdout")):
        raise ValueError("sealed or holdout input is prohibited")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty {split} input")
    ids: set[str] = set()
    for row in rows:
        if row.get("split") != split or row.get("source") != "deterministic_schema_first_synthetic":
            raise ValueError(f"unexpected split/source in {split}")
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in ids:
            raise ValueError(f"missing or duplicate row id in {split}: {row_id}")
        ids.add(row_id)
        message = row["input"]["message"]
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"empty message: {row_id}")
        for update in row["target"]["updates"]:
            source = update["source_text"]
            if not isinstance(source, str) or not source or source.casefold() not in message.casefold():
                raise ValueError(f"source_text not present in message: {row_id}")
    return rows


def _alias_map(row: dict[str, Any], field: str) -> dict[str, str]:
    schema = row["input"]["domain"]["schema"]
    property_schema = schema["properties"].get(field, {})
    if property_schema.get("type") != "string" or "enum" not in property_schema:
        return {}
    aliases = row["input"]["domain"]["aliases"].get(field, {})
    allowed = set(property_schema["enum"])
    result: dict[str, str] = {}
    for surface, canonical in sorted(aliases.items()):
        if canonical in allowed:
            result.setdefault(canonical, surface)
    return result


def _field_label(row: dict[str, Any], field: str) -> str:
    description = row["input"]["domain"]["schema"]["properties"][field].get("description", "")
    label = description.split(";")[0].rstrip(". ").strip()
    if not label:
        raise ValueError(f"missing human-readable field description: {row['id']}:{field}")
    return label


def _claim(row: dict[str, Any], field: str, value: str, operation: str) -> tuple[str, str]:
    alias = _alias_map(row, field).get(value)
    if not alias:
        raise ValueError(f"missing declared alias: {row['id']}:{field}:{value}")
    label = _field_label(row, field)
    if operation == "set":
        if field == row["input"]["domain"].get("capabilities", {}).get("preference_field"):
            text = f"Пользователь предпочёл бы для подбираемого варианта значение «{alias}» параметра «{label}»."
        else:
            text = f"Пользователь задал для подбираемого варианта параметр «{label}»: «{alias}»."
    elif operation == "exclude":
        text = f"Пользователь исключил для подбора значение «{alias}» параметра «{label}»."
    elif operation == "include":
        text = f"Пользователь снова разрешил для подбора значение «{alias}» параметра «{label}»."
    else:
        raise ValueError(f"unsupported operation: {operation}")
    return text, alias


def _example(
    row: dict[str, Any], field: str, value: str, operation: str,
    label: str, source_text: str | None, reason: str, ordinal: int,
) -> dict[str, Any]:
    claim, alias = _claim(row, field, value, operation)
    return {
        "id": f"{row['id']}:{ordinal:02d}",
        "split": row["split"],
        "family": row["domain_family_id"],
        "premise": row["input"]["message"],
        "hypothesis": claim,
        "label": label,
        "source_text": source_text,
        "field_id": field,
        "value_id": value,
        "value_alias": alias,
        "operation": operation,
        "reason": reason,
        "provenance": {
            "source_row_id": row["id"],
            "source": row["source"],
            "generator_version": row["generator_version"],
            "domain_family_id": row["domain_family_id"],
            "schema_family_id": row["schema_family_id"],
            "template_family_id": row["template_family_id"],
            "counterfactual_group_id": row.get("counterfactual_group_id"),
        },
    }


def _row_exclusion_reason(row: dict[str, Any]) -> str | None:
    if "mood" in row.get("tags", ()):
        return "ambiguous_mood_role"
    enum_sets: dict[str, set[str]] = {}
    for update in row["target"]["updates"]:
        if (
            update["operation"] == "set"
            and isinstance(update["value"], str)
            and update["value"] in _alias_map(row, update["field"])
        ):
            enum_sets.setdefault(update["field"], set()).add(update["value"])
    if "conflict" in row.get("tags", ()) and any(len(values) > 1 for values in enum_sets.values()):
        return "conflicting_single_valued_enum"
    return None


def _update_exclusion_reason(row: dict[str, Any], update: dict[str, Any]) -> str | None:
    if update["operation"] != "set":
        return None
    message = row["input"]["message"].casefold()
    surface = re.escape(update["source_text"].casefold())
    if re.search(rf"(?<!\w){surface}\s+допустим\w*", message):
        return "permission_is_not_selection"
    if re.search(rf"наверное\s+{surface}(?!\w)", message):
        return "hedged_value_is_not_certain_selection"
    return None


def _convert_row_with_exclusions(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a synthetic row, retaining every excluded row/update reason."""
    row_reason = _row_exclusion_reason(row)
    if row_reason:
        return [], [{"source_row_id": row["id"], "scope": "row", "reason": row_reason}]
    updates = row["target"]["updates"]
    blocked_fields = {update["field"] for update in updates}
    blocked_fields.update(issue.get("field") for issue in row["target"]["issues"])
    result: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for update in updates:
        field, value, operation = update["field"], update["value"], update["operation"]
        if operation not in OPERATIONS or not isinstance(value, str):
            continue
        aliases = _alias_map(row, field)
        if value not in aliases:
            continue
        source = update["source_text"].casefold()
        if source not in {
            surface.casefold()
            for surface, canonical in row["input"]["domain"]["aliases"][field].items()
            if canonical == value
        }:
            raise ValueError(f"enum target source is not its declared alias: {row['id']}:{field}")
        if any(
            surface.casefold() == source
            for other_field, surfaces in row["input"]["domain"]["aliases"].items()
            if other_field != field
            for surface in surfaces
        ):
            raise ValueError(f"enum target alias is ambiguous across fields: {row['id']}:{field}")
        update_reason = _update_exclusion_reason(row, update)
        if update_reason:
            exclusions.append(
                {"source_row_id": row["id"], "scope": "update", "reason": update_reason,
                 "field_id": field, "value_id": value, "operation": operation}
            )
            continue
        result.append(_example(row, field, value, operation, "support", update["source_text"], "generator_target", len(result)))
        result.append(
            _example(
                row, field, value, OPPOSITE[operation], "contradiction", update["source_text"],
                "incompatible_operation_same_field_value", len(result),
            )
        )

    # Choose a deterministic unrelated field. A missing target alone is not
    # evidence of contradiction; this is deliberately labelled unknown.
    for field in sorted(row["input"]["domain"]["aliases"]):
        if field in blocked_fields:
            continue
        for value, alias in sorted(_alias_map(row, field).items()):
            if alias.casefold() in row["input"]["message"].casefold():
                continue
            result.append(_example(row, field, value, "set", "unknown", None, "unmentioned_other_field", len(result)))
            return result, exclusions
    return result, exclusions


def convert_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Emit supported, incompatible and unrelated claims after noise filtering."""
    return _convert_row_with_exclusions(row)[0]


def _check_split_isolation(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, int]:
    groups = ("id", "domain_family_id", "schema_family_id", "template_family_id", "counterfactual_group_id")
    result: dict[str, int] = {}
    for key in groups:
        left = {row.get(key) for row in train if row.get(key)}
        right = {row.get(key) for row in dev if row.get(key)}
        overlap = len(left & right)
        result[key] = overlap
        if overlap:
            raise ValueError(f"train/dev overlap on {key}: {overlap}")
    left_messages = {_normal(row["input"]["message"]) for row in train}
    right_messages = {_normal(row["input"]["message"]) for row in dev}
    result["normalized_message"] = len(left_messages & right_messages)
    if result["normalized_message"]:
        raise ValueError("train/dev normalized message overlap")
    return result


def build(train_path: Path, dev_path: Path, output_dir: Path) -> dict[str, Any]:
    train, dev = _load(train_path, "train"), _load(dev_path, "dev")
    isolation = _check_split_isolation(train, dev)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"train": train_path, "dev": dev_path}
    source = {"train": train, "dev": dev}
    splits: dict[str, Any] = {}
    converted_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev"):
        converted: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for row in source[split]:
            cases, rejected = _convert_row_with_exclusions(row)
            converted.extend(cases)
            exclusions.extend(rejected)
        counts = Counter(case["label"] for case in converted)
        if any(counts[label] == 0 for label in LABELS):
            raise ValueError(f"empty label class in {split}: {dict(counts)}")
        converted_by_split[split] = converted
        path = output_dir / f"{split}.jsonl"
        path.write_text("".join(_json_line(case) + "\n" for case in converted), encoding="utf-8", newline="\n")
        exclusions_path = output_dir / f"{split}.exclusions.jsonl"
        exclusions_path.write_text("".join(_json_line(item) + "\n" for item in exclusions), encoding="utf-8", newline="\n")
        splits[split] = {
            "input": _portable_path(inputs[split]),
            "input_sha256": _sha256(inputs[split]),
            "input_rows": len(source[split]),
            "output": path.name,
            "output_sha256": _sha256(path),
            "output_rows": len(converted),
            "labels": {label: counts[label] for label in LABELS},
            "exclusions": {
                "file": exclusions_path.name,
                "sha256": _sha256(exclusions_path),
                "records": len(exclusions),
                "by_reason": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
            },
        }
    sample = {
        "status": "unreviewed_synthetic_generator_labels; structural sample, not human gold",
        "selection": "first three per split and label sorted by SHA-256 of example ID",
        "cases": [
            {
                key: case[key]
                for key in ("id", "split", "family", "premise", "hypothesis", "label", "source_text", "reason", "provenance")
            }
            for split in ("train", "dev")
            for label in LABELS
            for case in sorted(
                (item for item in converted_by_split[split] if item["label"] == label),
                key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest(),
            )[:3]
        ],
    }
    sample_path = output_dir / "sample-audit.json"
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "converter": {"path": "scripts/build_slot_evidence_dataset.py", "sha256": _sha256(Path(__file__))},
        "origin": "deterministic_schema_first_synthetic; derived from generator targets and schema aliases, not human annotation",
        "label_policy": {
            "support": "declared enum target update with literal source_text in full message",
            "contradiction": "opposite operation for same field and value; no unrelated enum alternatives",
            "unknown": "schema enum value from field absent from target/issues and literal message",
        },
        "noise_filter": "exclude conflicting enum/mood rows and only permission/hedged set updates; retain explicit excludes even if other classes are permitted",
        "premise": "full user message",
        "split_isolation_overlap": isolation,
        "splits": splits,
        "sample_audit": {"file": sample_path.name, "sha256": _sha256(sample_path), "cases": len(sample["cases"])},
        "exclusions": "blind, final holdout, product DEV/CONTRACT, agent Query and model outputs",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA / "train.jsonl")
    parser.add_argument("--dev", type=Path, default=DEFAULT_DATA / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.train, args.dev, args.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
