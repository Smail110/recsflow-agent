"""Generate citation-aware synthetic NLI pairs from isolated schema families.

The source is only the schema/aliases of the frozen v2 train and dev splits.
User messages, claims, citations, and labels are deterministic constructions;
none are real traffic or human annotation. Product DEV/CONTRACT, role fixtures,
BLIND, and the final holdout are neither imported nor read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_role_nli_dataset import (
    LABELS,
    ROOT,
    SOURCE,
    _check_isolation,
    _line,
    _normal,
    _portable,
    _read_schemas,
    _sha256,
    _slots,
)

OPERATIONS = ("set", "exclude", "include")
SUPPORTED_ROLES = frozenset({"Класс объекта", "Режим объекта", "Контекст использования", "Мягкое пожелание"})


def _claim(role: str, alias: str, operation: str, *, is_preference: bool) -> str:
    if operation == "set" and is_preference:
        body = f"Пользователь предпочёл бы для подбираемого варианта значение «{alias}» параметра «{role}»."
    else:
        verb = {"set": "выбрал", "exclude": "исключил", "include": "снова разрешил"}[operation]
        body = f"Пользователь {verb} для подбора значение «{alias}» параметра «{role}»."
    return "Выделенная цитата подтверждает: " + body


def _primary(split: str, role: str, alias: str, operation: str) -> str:
    """Use role cues, not copied v2 utterances or a source-text parser."""
    if operation == "include":
        if split == "train":
            return f"По параметру «{role}» прежний запрет на «{alias}» снимаю; теперь допускаю."
        return f"Прежнее исключение «{alias}» снимаю для параметра «{role}»: вариант снова допустим."
    train = {
        ("Класс объекта", "set"): f"Мне нужен объект категории «{alias}».",
        ("Класс объекта", "exclude"): f"Объект «{alias}» мне не предлагайте.",
        ("Режим объекта", "set"): f"По исполнению хочу «{alias}».",
        ("Режим объекта", "exclude"): f"Исполнение «{alias}» исключаю.",
        ("Контекст использования", "set"): f"Сценарий использования — «{alias}».",
        ("Контекст использования", "exclude"): f"Сценарий использования «{alias}» исключаю.",
        ("Мягкое пожелание", "set"): f"Если возможно, предпочту свойство «{alias}».",
        ("Мягкое пожелание", "exclude"): f"Свойство «{alias}» не подходит, исключите его.",
    }
    dev = {
        ("Класс объекта", "set"): f"Для нового выбора хочу «{alias}».",
        ("Класс объекта", "exclude"): f"Уберите из вариантов объект «{alias}».",
        ("Режим объекта", "set"): f"Нужно исполнение «{alias}».",
        ("Режим объекта", "exclude"): f"Вариант с исполнением «{alias}» не нужен.",
        ("Контекст использования", "set"): f"Использовать собираюсь «{alias}».",
        ("Контекст использования", "exclude"): f"Для сценария «{alias}» вариант не ищу.",
        ("Мягкое пожелание", "set"): f"Желательно свойство «{alias}».",
        ("Мягкое пожелание", "exclude"): f"Свойства «{alias}» не хочу, отсейте такие варианты.",
    }
    try:
        return (train if split == "train" else dev)[(role, operation)]
    except KeyError as exc:
        raise ValueError(f"no unambiguous constructor for {role}:{operation}") from exc


def _distractor(split: str, kind: str, role: str, alias: str) -> str:
    if kind == "other_role":
        if split == "train":
            return f"В черновике обсуждалось слово «{alias}» для поля «{role}»."
        return f"В старой заметке упоминалось «{alias}» по параметру «{role}»."
    if kind == "same_alias_quote":
        if split == "train":
            return f"В заголовке обзора встречалось слово «{alias}»."
        return f"Чужая статья называлась «{alias}»."
    raise ValueError(f"unsupported distractor family: {kind}")


def _label(action: str, claim_operation: str, *, cited_primary: bool, is_preference: bool) -> str:
    if not cited_primary:
        return "unknown"
    if action == claim_operation:
        return "support"
    if is_preference and action == "set" and claim_operation == "exclude":
        # A soft wish alone does not establish that the user has categorically
        # forbidden or accepted that attribute as a hard constraint.
        return "unknown"
    if {action, claim_operation} == {"set", "include"}:
        return "unknown"
    return "contradiction"


def _tag(message: str, source_text: str, start: int, end: int) -> str:
    if start < 0 or end <= start or message[start:end] != source_text:
        raise ValueError("citation offsets do not match the message")
    return message[:start] + "<evidence>" + source_text + "</evidence>" + message[end:]


def _make_case(
    schema: dict[str, Any],
    split: str,
    field: str,
    role: str,
    value_id: str,
    alias: str,
    action: str,
    kind: str,
    span_family: str,
    message: str,
    source_text: str,
    source_start: int,
    claim_operation: str,
    *,
    cited_primary: bool,
    primary_first: bool,
) -> dict[str, Any]:
    source_end = source_start + len(source_text)
    premise = _tag(message, source_text, source_start, source_end)
    return {
        "id": (
            f"cited-nli-v4-{split}-{schema['schema_family_id']}-{field}-{value_id}-"
            f"{action}-{kind}-{span_family}-{claim_operation}-{'primary' if cited_primary else 'counterfactual'}"
        ),
        "split": split,
        "message": message,
        "source_text": source_text,
        "source_start": source_start,
        "source_end": source_end,
        "premise": premise,
        "hypothesis": _claim(role, alias, claim_operation, is_preference=field == schema["preference_field"]),
        "label": _label(
            action,
            claim_operation,
            cited_primary=cited_primary,
            is_preference=field == schema["preference_field"],
        ),
        "field_id": field,
        "role": role,
        "value_id": value_id,
        "value_alias": alias,
        "operation": claim_operation,
        "actual_operation": action,
        "citation_kind": "primary" if cited_primary else "counterfactual",
        "marked_clause_position": "first" if cited_primary == primary_first else "second",
        "citation_span_family": span_family,
        "distractor_family": kind,
        "template_family_id": f"cited-nli-v4-{split}-{action}-{kind}",
        "schema_family_id": schema["schema_family_id"],
        "domain_family_id": schema["domain_family_id"],
        "domain_id": schema["domain_id"],
        "provenance": "deterministic_schema_first_synthetic_v4; not human annotation",
    }


def convert_schema(schema: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in ("train", "dev"):
        raise ValueError(f"unsupported split: {split}")
    # The v2 train schemas also contain a budget-package enum absent from the
    # isolated dev domains. Exclude it rather than fabricate a role constructor
    # that cannot be checked across splits.
    slots = [slot for slot in _slots(schema) if slot[1] in SUPPORTED_ROLES]
    if len(slots) < 3:
        raise ValueError(f"fewer than three supported roles: {schema['schema_family_id']}")
    family_seed = int(hashlib.sha256(schema["schema_family_id"].encode("utf-8")).hexdigest()[:8], 16)
    rows: list[dict[str, Any]] = []
    for slot_index, (field, role, values) in enumerate(slots):
        _, other_role, other_values = slots[(slot_index + 1) % len(slots)]
        for action_index, action in enumerate(OPERATIONS):
            value_id, alias = values[(family_seed + slot_index + action_index) % len(values)]
            _, other_alias = other_values[(family_seed + slot_index + action_index) % len(other_values)]
            primary = _primary(split, role, alias, action)
            for kind_index, kind in enumerate(("other_role", "same_alias_quote")):
                distractor = _distractor(
                    split,
                    kind,
                    other_role,
                    other_alias if kind == "other_role" else alias,
                )
                primary_first = (family_seed + slot_index + action_index + kind_index) % 2 == 0
                message = primary + " " + distractor if primary_first else distractor + " " + primary
                distractor_alias = other_alias if kind == "other_role" else alias
                primary_clause_start = 0 if primary_first else len(distractor) + 1
                distractor_clause_start = len(primary) + 1 if primary_first else 0
                primary_alias_start = primary_clause_start + primary.index(f"«{alias}»") + 1
                distractor_alias_start = distractor_clause_start + distractor.index(f"«{distractor_alias}»") + 1
                for claim_operation in OPERATIONS:
                    for span_family, primary_source, source_start, other_source, other_start in (
                        ("clause", primary, primary_clause_start, distractor, distractor_clause_start),
                        ("value", alias, primary_alias_start, distractor_alias, distractor_alias_start),
                    ):
                        rows.append(
                            _make_case(
                                schema,
                                split,
                                field,
                                role,
                                value_id,
                                alias,
                                action,
                                kind,
                                span_family,
                                message,
                                primary_source,
                                source_start,
                                claim_operation,
                                cited_primary=True,
                                primary_first=primary_first,
                            )
                        )
                        rows.append(
                            _make_case(
                                schema,
                                split,
                                field,
                                role,
                                value_id,
                                alias,
                                action,
                                kind,
                                span_family,
                                message,
                                other_source,
                                other_start,
                                claim_operation,
                                cited_primary=False,
                                primary_first=primary_first,
                            )
                        )
    return rows


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    labels_by_pair: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (_normal(row["premise"]), _normal(row["hypothesis"]))
        old = labels_by_pair.get(key)
        if old is not None:
            if old != row["label"]:
                raise ValueError(f"conflicting text-pair labels: {row['id']}")
            continue
        labels_by_pair[key] = row["label"]
        unique.append(row)
    return unique, len(rows) - len(unique)


def _audit_pairs(rows: list[dict[str, Any]]) -> dict[str, int]:
    paired: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["message"][row["source_start"] : row["source_end"]] != row["source_text"]:
            raise ValueError(f"invalid citation offset: {row['id']}")
        if row["premise"] != _tag(row["message"], row["source_text"], row["source_start"], row["source_end"]):
            raise ValueError(f"invalid marked premise: {row['id']}")
        key = (row["message"], row["hypothesis"], row["citation_span_family"])
        paired.setdefault(key, {})[row["citation_kind"]] = row["label"]
    missing = sum(set(labels) != {"primary", "counterfactual"} for labels in paired.values())
    if missing:
        raise ValueError(f"unpaired citation counterfactuals: {missing}")
    support_flips = sum(labels == {"primary": "support", "counterfactual": "unknown"} for labels in paired.values())
    contradiction_flips = sum(labels == {"primary": "contradiction", "counterfactual": "unknown"} for labels in paired.values())
    if not support_flips or not contradiction_flips:
        raise ValueError("missing citation-driven support/contradiction counterfactuals")
    return {
        "matched_message_claim_pairs": len(paired),
        "support_to_unknown": support_flips,
        "contradiction_to_unknown": contradiction_flips,
    }


def build(train_path: Path, dev_path: Path, output_dir: Path) -> dict[str, Any]:
    schemas = {"train": _read_schemas(train_path, "train"), "dev": _read_schemas(dev_path, "dev")}
    overlap = _check_isolation(schemas["train"], schemas["dev"])
    source_paths = {"train": train_path, "dev": dev_path}
    outputs: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {
        "schema_version": 4,
        "origin": "deterministic schema-first synthetic; no human annotations or agent outputs",
        "task": "citation-aware support/contradiction/unknown for field+value+operation",
        "citation_format": "full message with exactly one <evidence>...</evidence> span; start/end are Unicode codepoint offsets in unmarked message; clause and value spans are separate paired families",
        "label_policy": {
            "support": "marked primary clause performs exactly the claimed operation",
            "contradiction": "marked primary clause explicitly performs an incompatible operation for the same role/value",
            "unknown": "marked clause does not support this claim; set and include are not equivalent; a soft wish does not prove a hard exclusion",
        },
        "excluded_sources": ["product DEV", "product CONTRACT", "NLI role fixture", "BLIND", "final holdout", "agent Query"],
        "role_scope": sorted(SUPPORTED_ROLES),
        "excluded_schema_role": "Категория бюджета: exists only in v2 train schemas, so no disjoint-domain dev check",
        "template_isolation_note": (
            "Train/dev have different wording and split-prefixed IDs for the same semantic scenario types; "
            "zero ID overlap is not independent semantic-template coverage or real-traffic validation"
        ),
        "split_isolation_overlap": overlap,
        "generator": {"path": _portable(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "schema_reader": {"path": "scripts/build_role_nli_dataset.py", "sha256": _sha256(ROOT / "scripts" / "build_role_nli_dataset.py")},
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        candidates = [row for schema in schemas[split].values() for row in convert_schema(schema, split)]
        rows, removed = _deduplicate(candidates)
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {split} IDs")
        pair_audit = _audit_pairs(rows)
        cross = Counter((row["operation"], row["label"]) for row in rows)
        if any(cross[(operation, label)] == 0 for operation in OPERATIONS for label in LABELS):
            raise ValueError(f"operation predicts a missing label in {split}")
        output = output_dir / f"{split}.jsonl"
        output.write_text("".join(_line(row) for row in rows), encoding="utf-8", newline="\n")
        outputs[split] = rows
        manifest["splits"][split] = {
            "source": _portable(source_paths[split]),
            "source_sha256": _sha256(source_paths[split]),
            "schema_families": len(schemas[split]),
            "domain_families": sorted({schema["domain_family_id"] for schema in schemas[split].values()}),
            "template_families": sorted({row["template_family_id"] for row in rows}),
            "output": output.name,
            "output_sha256": _sha256(output),
            "candidate_rows": len(candidates),
            "duplicate_text_pairs_removed": removed,
            "rows": len(rows),
            "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
            "label_by_operation": {operation: {label: cross[(operation, label)] for label in LABELS} for operation in OPERATIONS},
            "citation_pair_audit": pair_audit,
            "citation_span_families": dict(sorted(Counter(row["citation_span_family"] for row in rows).items())),
            "marked_position_by_citation_kind": {
                kind: {
                    position: sum(row["citation_kind"] == kind and row["marked_clause_position"] == position for row in rows)
                    for position in ("first", "second")
                }
                for kind in ("primary", "counterfactual")
            },
        }
    left = {(_normal(row["premise"]), _normal(row["hypothesis"])) for row in outputs["train"]}
    right = {(_normal(row["premise"]), _normal(row["hypothesis"])) for row in outputs["dev"]}
    overlap["normalized_pair"] = len(left & right)
    if overlap["normalized_pair"]:
        raise ValueError("normalized train/dev text-pair overlap")
    left_templates = set(manifest["splits"]["train"]["template_families"])
    right_templates = set(manifest["splits"]["dev"]["template_families"])
    overlap["template_family_id"] = len(left_templates & right_templates)
    if overlap["template_family_id"]:
        raise ValueError("train/dev generated template overlap")
    sample = {
        "status": "deterministic synthetic sample; requires human audit before use; not human gold",
        "selection": "four per split and label sorted by SHA-256 of ID",
        "cases": [
            {
                key: row[key]
                for key in (
                    "id",
                    "message",
                    "source_text",
                    "source_start",
                    "source_end",
                    "premise",
                    "hypothesis",
                    "label",
                    "operation",
                    "actual_operation",
                    "citation_kind",
                )
            }
            for split in ("train", "dev")
            for label in LABELS
            for row in sorted(
                (item for item in outputs[split] if item["label"] == label),
                key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest(),
            )[:4]
        ],
    }
    sample_path = output_dir / "sample-audit.json"
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest["sample_audit"] = {"path": sample_path.name, "sha256": _sha256(sample_path), "cases": len(sample["cases"])}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=SOURCE / "train.jsonl")
    parser.add_argument("--dev", type=Path, default=SOURCE / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.train, args.dev, args.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
