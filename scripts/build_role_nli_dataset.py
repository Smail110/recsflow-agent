"""Build synthetic role-aware NLI pairs from frozen schema families.

Only schema descriptions and declared enum aliases are taken from the v2
schema-first train/dev splits. The original messages, targets and issues are
never used as labels or templates. Labels follow the meaning of the sentence
constructor below; they are synthetic intent, not human annotation.
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
SOURCE = ROOT / "data" / "lora" / "semantic_extraction_v2"
LABELS = ("support", "contradiction", "unknown")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _normal(value: str) -> str:
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", value.casefold().replace("ё", "е")).split())


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_schemas(path: Path, split: str) -> dict[str, dict[str, Any]]:
    if any("blind" in part.casefold() or "holdout" in part.casefold() for part in path.parts):
        raise ValueError("sealed or holdout source is prohibited")
    schemas: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("split") != split
            or row.get("source") != "deterministic_schema_first_synthetic"
            or row.get("generator_version") != "semantic-lora-schema-first-v2"
        ):
            raise ValueError(f"unexpected {split} source metadata")
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in ids:
            raise ValueError(f"missing or duplicate {split} source ID: {row_id}")
        ids.add(row_id)
        family = row["schema_family_id"]
        domain = row["input"]["domain"]
        if family != domain["schema_family_id"]:
            raise ValueError(f"schema family mismatch: {row_id}")
        schema = {
            "schema_family_id": family,
            "domain_family_id": row["domain_family_id"],
            "domain_id": row["domain_id"],
            "aliases": domain["aliases"],
            "properties": domain["schema"]["properties"],
            "preference_field": domain["capabilities"].get("preference_field"),
        }
        if family in schemas and schemas[family] != schema:
            raise ValueError(f"schema drift within family: {family}")
        schemas[family] = schema
    if not ids or not schemas:
        raise ValueError(f"empty {split} source")
    return schemas


def _slots(schema: dict[str, Any]) -> list[tuple[str, str, list[tuple[str, str]]]]:
    result: list[tuple[str, str, list[tuple[str, str]]]] = []
    for field, aliases in sorted(schema["aliases"].items()):
        property_schema = schema["properties"].get(field, {})
        if property_schema.get("type") != "string" or "enum" not in property_schema:
            continue
        by_value: dict[str, str] = {}
        for surface, canonical in sorted(aliases.items()):
            if canonical in property_schema["enum"]:
                by_value.setdefault(canonical, surface)
        if len(by_value) < 2:
            continue
        label = property_schema.get("description", "").split(";")[0].rstrip(". ").strip()
        if not label:
            raise ValueError(f"missing role label: {schema['schema_family_id']}:{field}")
        result.append((field, label, sorted(by_value.items())))
    if len(result) < 3:
        raise ValueError(f"need three distinct enum roles: {schema['schema_family_id']}")
    return result


# The split-specific sentence families differ in syntax and cue placement.
# Every family has matched claims, so a classifier cannot solve the task from
# class priors or the presence of a particular premise alone.
_TRAIN = {
    "direct": lambda role, value: f"Для нового подбора зафиксируйте параметр «{role}»: «{value}».",
    "exclude": lambda role, value: f"При подборе по параметру «{role}» исключите «{value}».",
    "include": lambda role, value: f"Прежний запрет по параметру «{role}» отменяю: «{value}» снова допускается.",
    "uncertain": lambda role, value: f"По параметру «{role}» пока не решил; «{value}» — только один из вариантов.",
    "permitted": lambda role, value: f"По параметру «{role}» значение «{value}» возможно, но специально его не выбираю.",
    "reported": lambda role, value: f"В чужом отзыве упоминалось «{value}» для параметра «{role}»; к моему подбору это не относится.",
    "implicit_set": None,
    "implicit_exclude": None,
}
_DEV = {
    "direct": lambda role, value: f"Из условий мне нужен «{role}» со значением «{value}».",
    "exclude": lambda role, value: f"Значение «{value}» в «{role}» мне не предлагайте.",
    "include": lambda role, value: f"Для «{role}» снимаю ограничение на «{value}»: теперь такой вариант разрешён.",
    "uncertain": lambda role, value: f"По параметру «{role}» решения ещё нет; «{value}» — лишь предположение.",
    "permitted": lambda role, value: f"Не запрещаю «{value}» в «{role}», хотя предпочтения у меня здесь нет.",
    "reported": lambda role, value: f"«{value}» по «{role}» было написано в старой заметке, не в этом запросе.",
    "implicit_set": None,
    "implicit_exclude": None,
}


def _implicit_phrase(split: str, role: str, value: str, operation: str) -> str | None:
    """Render only role cues whose schema meaning is unambiguous in isolation."""
    train = {
        ("Класс объекта", "set"): f"Ищу именно «{value}».",
        ("Класс объекта", "exclude"): f"Сам объект «{value}» мне не предлагайте.",
        ("Режим объекта", "set"): f"По исполнению хочу вариант «{value}».",
        ("Режим объекта", "exclude"): f"По исполнению вариант «{value}» не подходит.",
        ("Контекст использования", "set"): f"Для использования нужен вариант «{value}».",
        ("Контекст использования", "exclude"): f"Сценарий использования «{value}» исключаю.",
        ("Мягкое пожелание", "set"): f"Если получится, предпочту характеристику «{value}».",
        ("Мягкое пожелание", "exclude"): f"Характеристика «{value}» мне не подходит, исключите её.",
    }
    dev = {
        ("Класс объекта", "set"): f"Выбираю себе «{value}».",
        ("Класс объекта", "exclude"): f"Объект «{value}» сразу отсекайте.",
        ("Режим объекта", "set"): f"По типу исполнения устроит «{value}».",
        ("Режим объекта", "exclude"): f"Исполнение «{value}» не рассматриваю.",
        ("Контекст использования", "set"): f"Сценарий применения — «{value}».",
        ("Контекст использования", "exclude"): f"Для сценария «{value}» ничего не ищу.",
        ("Мягкое пожелание", "set"): f"В идеале у варианта будет свойство «{value}».",
        ("Мягкое пожелание", "exclude"): f"Свойство «{value}» нежелательно, уберите такие варианты.",
    }
    return (train if split == "train" else dev).get((role, operation))


def _hypothesis(split: str, role: str, value: str, operation: str, *, is_preference: bool) -> str:
    if split == "train":
        if operation == "set" and is_preference:
            return f"Пользователь предпочёл бы для подбираемого варианта значение «{value}» параметра «{role}»."
        verb = {"set": "выбрал", "exclude": "исключил", "include": "снова разрешил"}[operation]
        return f"Пользователь {verb} для подбора значение «{value}» параметра «{role}»."
    if operation == "set" and is_preference:
        return f"Для подбираемого варианта пользователь указал как пожелание «{value}» по полю «{role}»."
    verb = {"set": "задал", "exclude": "запретил", "include": "вернул в допустимые"}[operation]
    return f"Для подбираемого варианта пользователь {verb} «{value}» по полю «{role}»."


def _case(
    schema: dict[str, Any], split: str, family: str, ordinal: int,
    premise: str, field: str, role: str, value_id: str, value: str,
    operation: str, label: str, reason: str,
) -> dict[str, Any]:
    return {
        "id": f"role-nli-v3-{split}-{schema['schema_family_id']}-{family}-{ordinal:02d}",
        "split": split,
        "premise": premise,
        "hypothesis": _hypothesis(split, role, value, operation, is_preference=field == schema["preference_field"]),
        "label": label,
        "field_id": field,
        "role": role,
        "value_id": value_id,
        "value_alias": value,
        "operation": operation,
        "reason": reason,
        "template_family_id": f"role-nli-v3-{split}-{family}",
        "schema_family_id": schema["schema_family_id"],
        "domain_family_id": schema["domain_family_id"],
        "domain_id": schema["domain_id"],
        "provenance": "deterministic_schema_first_synthetic_v3; not human annotation",
    }


def convert_schema(schema: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Create matched supports, contradictions and unknowns per premise."""
    if split not in ("train", "dev"):
        raise ValueError(f"unsupported split: {split}")
    templates = _TRAIN if split == "train" else _DEV
    slots = _slots(schema)
    alias_counts = Counter(_normal(surface) for _, _, values in slots for _, surface in values)
    family_seed = int(hashlib.sha256(schema["schema_family_id"].encode("utf-8")).hexdigest()[:8], 16)
    result: list[dict[str, Any]] = []
    for index, (field, role, values) in enumerate(slots):
        other_field, other_role, other_values = slots[(index + 1) % len(slots)]
        absent_field, absent_role, absent_values = slots[(index + 2) % len(slots)]
        for family_index, family in enumerate(templates):
            chosen_id, chosen = values[(family_seed + index + family_index) % len(values)]
            alternatives = [item for item in values if item[0] != chosen_id]
            mentioned_id, mentioned = other_values[(family_seed + index + family_index) % len(other_values)]
            absent_id, absent = absent_values[(family_seed + index + family_index) % len(absent_values)]
            if family in {"direct", "exclude", "include", "implicit_set", "implicit_exclude"}:
                if family.startswith("implicit"):
                    if alias_counts[_normal(chosen)] != 1:
                        continue
                    implicit_operation = "set" if family == "implicit_set" else "exclude"
                    premise = _implicit_phrase(split, role, chosen, implicit_operation)
                    if premise is None:
                        continue
                else:
                    premise = templates[family](role, chosen)
                unknown = (absent_field, absent_role, absent_id, absent, "set", "unmentioned_other_role")
            else:
                # An independently selected role gives every such premise a
                # genuine support and contradiction while the mentioned value
                # remains neither selected nor excluded for the other role.
                premise = (
                    templates["direct"](role, chosen) + " "
                    + templates[family](other_role, mentioned)
                )
                unknown = (other_field, other_role, mentioned_id, mentioned, "set", f"{family}_is_not_selection")
            operation = "exclude" if family == "implicit_exclude" else family if family in {"exclude", "include"} else "set"
            opposite = "set" if operation == "exclude" else "exclude"
            if operation == "include":
                opposite = "exclude"
            claims = [
                (field, role, chosen_id, chosen, operation, "support", "explicit_current_operation"),
                (field, role, chosen_id, chosen, opposite, "contradiction", "incompatible_current_operation"),
                (*unknown[:5], "unknown", unknown[5]),
            ]
            # An unmentioned alternative is unknown, even for a schema field
            # with one stored value. The user can express more than one wish;
            # storage cardinality does not prove semantic contradiction.
            if family in {"direct", "implicit_set"}:
                claims.extend(
                    (field, role, alternative_id, alternative, "set", "unknown", "unmentioned_alternative_same_role")
                    for alternative_id, alternative in alternatives
                )
            if family == "include":
                claims.append(
                    (field, role, chosen_id, chosen, "set", "unknown", "permission_is_not_selection")
                )
            for ordinal, claim in enumerate(claims):
                result.append(_case(schema, split, family, index * 10 + ordinal, premise, *claim))
    return result


def _check_isolation(train: dict[str, dict[str, Any]], dev: dict[str, dict[str, Any]]) -> dict[str, int]:
    keys = ("schema_family_id", "domain_family_id", "domain_id")
    overlap = {}
    for key in keys:
        left = {schema[key] for schema in train.values()}
        right = {schema[key] for schema in dev.values()}
        overlap[key] = len(left & right)
        if overlap[key]:
            raise ValueError(f"train/dev {key} overlap: {overlap[key]}")
    return overlap


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Do not count identical text pairs from schema variants as new cases."""
    seen: dict[tuple[str, str], str] = {}
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (_normal(row["premise"]), _normal(row["hypothesis"]))
        prior = seen.get(key)
        if prior is not None:
            if prior != row["label"]:
                raise ValueError(f"conflicting labels for same text pair: {row['id']}")
            continue
        seen[key] = row["label"]
        unique.append(row)
    return unique, len(rows) - len(unique)


def build(train_path: Path, dev_path: Path, output_dir: Path) -> dict[str, Any]:
    train = _read_schemas(train_path, "train")
    dev = _read_schemas(dev_path, "dev")
    overlap = _check_isolation(train, dev)
    data = {"train": train, "dev": dev}
    paths = {"train": train_path, "dev": dev_path}
    converted: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "origin": "deterministic schema-first synthetic; labels from sentence constructor, not human or agent output",
        "premise": "full generated user utterance",
        "claim": "role + value + operation",
        "label_semantics": {
            "support": "the constructed utterance explicitly performs the claimed current operation",
            "contradiction": "the constructed utterance explicitly performs an incompatible current operation for the same role and value",
            "unknown": "the utterance does not select this role/value; mention, permission and uncertainty are not selection",
        },
        "source_exclusions": ["product DEV", "product CONTRACT", "NLI role fixture", "BLIND", "final holdout", "agent Query"],
        "split_isolation_overlap": overlap,
        "generator": {"path": _portable(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        candidate_rows = [case for schema in data[split].values() for case in convert_schema(schema, split)]
        rows, duplicates = _deduplicate(candidate_rows)
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {split} case ID")
        counts = Counter(row["label"] for row in rows)
        if any(counts[label] == 0 for label in LABELS):
            raise ValueError(f"missing {split} label class")
        premise_labels: dict[str, set[str]] = {}
        for row in rows:
            premise_labels.setdefault(_normal(row["premise"]), set()).add(row["label"])
        if any(labels != set(LABELS) for labels in premise_labels.values()):
            raise ValueError(f"unmatched {split} premise: each must have all three labels")
        output = output_dir / f"{split}.jsonl"
        output.write_text("".join(_line(row) for row in rows), encoding="utf-8", newline="\n")
        converted[split] = rows
        manifest["splits"][split] = {
            "source": _portable(paths[split]),
            "source_sha256": _sha256(paths[split]),
            "schema_families": len(data[split]),
            "template_families": sorted({row["template_family_id"] for row in rows}),
            "output": output.name,
            "output_sha256": _sha256(output),
            "rows": len(rows),
            "candidate_rows": len(candidate_rows),
            "duplicate_text_pairs_removed": duplicates,
            "matched_premises": len(premise_labels),
            "labels": {label: counts[label] for label in LABELS},
            "operations": dict(sorted(Counter(row["operation"] for row in rows).items())),
            "reasons": dict(sorted(Counter(row["reason"] for row in rows).items())),
            "domains": sorted({row["domain_id"] for row in rows}),
        }
    train_pairs = {(_normal(row["premise"]), _normal(row["hypothesis"])) for row in converted["train"]}
    dev_pairs = {(_normal(row["premise"]), _normal(row["hypothesis"])) for row in converted["dev"]}
    overlap["normalized_pair"] = len(train_pairs & dev_pairs)
    if overlap["normalized_pair"]:
        raise ValueError("normalized train/dev pair overlap")
    train_templates = set(manifest["splits"]["train"]["template_families"])
    dev_templates = set(manifest["splits"]["dev"]["template_families"])
    overlap["template_family_id"] = len(train_templates & dev_templates)
    if overlap["template_family_id"]:
        raise ValueError("template family overlap")
    sample = {
        "status": "synthetic construction audit; not human gold",
        "cases": [
            {key: row[key] for key in ("id", "premise", "hypothesis", "label", "reason", "operation", "domain_id")}
            for split in ("train", "dev")
            for label in LABELS
            for row in sorted(
                (item for item in converted[split] if item["label"] == label),
                key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest(),
            )[:4]
        ],
    }
    sample_path = output_dir / "sample-audit.json"
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest["sample_audit"] = {"path": sample_path.name, "sha256": _sha256(sample_path), "cases": len(sample["cases"])}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=SOURCE / "train.jsonl")
    parser.add_argument("--dev", type=Path, default=SOURCE / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.train, arguments.dev, arguments.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
