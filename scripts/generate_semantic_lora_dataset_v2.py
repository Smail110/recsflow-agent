"""Generate schema-first LoRA dataset v2 with group-level split isolation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.generate_semantic_lora_dataset import (
    SEEN_DOMAINS,
    DomainSpec,
    build_schema,
    render_case,
)

GENERATOR_VERSION = "semantic-lora-schema-first-v2"
DATASET_VERSION = "semantic-lora-pilot-v2"
DEFAULT_SEED = 20260916
SPLIT_COUNTS = {"train": 960, "dev": 160, "blind": 120}
NEAR_DUPLICATE_THRESHOLD = 0.80

DEV_DOMAINS = (
    DomainSpec(
        "fitness_equipment",
        "recreation_goods",
        ("тренажёр", "гантели", "коврик"),
        ("складной", "устойчивый", "компактный"),
        ("для квартиры", "для зала", "для улицы"),
        ("тихая работа", "простая сборка", "малый вес"),
        "с регулировкой",
        "без регулировки",
        "тыс. руб",
        1000,
    ),
    DomainSpec(
        "event_venues",
        "event_services",
        ("зал", "лофт", "площадка"),
        ("камерный", "торжественный", "неформальный"),
        ("у метро", "за городом", "в центре"),
        ("своя парковка", "гибкая рассадка", "позднее закрытие"),
        "с кейтерингом",
        "без кейтеринга",
        "тыс. руб",
        1000,
    ),
    DomainSpec(
        "pet_services",
        "animal_services",
        ("передержка", "груминг", "выгул"),
        ("индивидуальный", "бережный", "активный"),
        ("рядом с домом", "с выездом", "за городом"),
        ("фотоотчёт", "гибкое время", "знакомство заранее"),
        "с ветеринаром",
        "без ветеринара",
        "часов",
        1,
    ),
    DomainSpec(
        "home_repairs",
        "maintenance_services",
        ("диагностика", "ремонт", "монтаж"),
        ("срочный", "плановый", "комплексный"),
        ("в квартире", "в офисе", "на даче"),
        ("фиксированная смета", "уборка после работ", "вечерний выезд"),
        "с гарантией",
        "без гарантии",
        "тыс. руб",
        1000,
    ),
)

BLIND_DOMAINS = (
    DomainSpec(
        "energy_plans",
        "utilities",
        ("тариф", "контракт", "пакет энергии"),
        ("фиксированный", "гибкий", "ночной"),
        ("для квартиры", "для мастерской", "для офиса"),
        ("прозрачный расчёт", "месячный отчёт", "автоплатёж"),
        "с зелёной энергией",
        "без зелёной энергии",
        "руб/мес",
        1,
        True,
    ),
    DomainSpec(
        "coworking",
        "workspaces",
        ("рабочее место", "переговорная", "кабинет"),
        ("тихий", "командный", "представительский"),
        ("у метро", "в деловом квартале", "у парка"),
        ("доступ ночью", "локер", "гостевые пропуска"),
        "с парковкой",
        "без парковки",
        "тыс. руб/мес",
        1000,
        True,
    ),
    DomainSpec(
        "vehicle_rental",
        "rental_mobility",
        ("автомобиль", "фургон", "скутер"),
        ("городской", "вместительный", "экономичный"),
        ("у аэропорта", "в центре", "у вокзала"),
        ("быстрая выдача", "второй водитель", "без залога"),
        "со страховкой",
        "без страховки",
        "тыс. руб/день",
        1000,
        True,
    ),
    DomainSpec(
        "cloud_storage",
        "data_services",
        ("хранилище", "архив", "пространство обмена"),
        ("персональный", "командный", "регулируемый"),
        ("для отдела", "для подрядчиков", "для филиалов"),
        ("журнал доступа", "быстрый перенос", "единый вход"),
        "с шифрованием",
        "без шифрования",
        "тыс. руб/мес",
        1000,
        True,
    ),
)

PARTITIONS = {
    "train": {"domains": SEEN_DOMAINS, "schemas_per_domain": 4, "scenarios": tuple(range(30)), "schema_start": 0, "template_start": 0},
    "dev": {
        "domains": DEV_DOMAINS,
        "schemas_per_domain": 5,
        "scenarios": (0, 3, 4, 10, 13, 15, 18, 25),
        "schema_start": 100,
        "template_start": 100,
    },
    "blind": {
        "domains": BLIND_DOMAINS,
        "schemas_per_domain": 3,
        "scenarios": (0, 3, 4, 10, 12, 14, 16, 17, 26, 27),
        "schema_start": 200,
        "template_start": 200,
    },
}

PARTITION_FRAMES = {
    "train": (
        "Требования к очередному подбору фиксирую заранее:",
        "Для учебного запроса перечисляю критерии:",
        "В новой заявке задаю такие параметры:",
        "До начала поиска записываю условия:",
    ),
    "dev": (
        "Независимая контрольная заявка содержит следующее:",
        "Для отдельной проверки передаю новые ориентиры:",
        "В контрольном обращении условия сформулированы так:",
        "Перед сравнением контрольных ответов уточняю критерии:",
    ),
    "blind": (
        "В закрытой оценке сообщаю самостоятельные вводные:",
        "Для запечатанного набора формулирую отдельные границы:",
        "Перед скрытой проверкой называю незнакомые условия:",
        "В изолированном обращении передаю требования выбора:",
    ),
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", text.casefold().replace("ё", "е")).split())


def _replace_exact(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {mapping.get(str(key), str(key)): _replace_exact(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_exact(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _token(prefix: str, identity: str, slot: str) -> str:
    return f"{prefix}_{hashlib.sha256(f'{identity}:{slot}'.encode()).hexdigest()[:9]}"


def _compact_schema(raw: dict[str, Any], family_id: str, schema_id: str) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    field_mapping = {field: _token("p", schema_id, slot) for slot, field in result["field_names"].items()}
    enum_mapping = {
        value: _token("e", schema_id, f"{slot}:{index}")
        for slot, values in result["enum_labels"].items()
        for index, value in enumerate(values)
    }
    result = _replace_exact(result, field_mapping | enum_mapping)
    result["schema_id"] = schema_id
    result["schema_family_id"] = family_id
    descriptions = {
        "category": "Класс объекта; canonical value указан в aliases.",
        "style": "Режим объекта; canonical value указан в aliases.",
        "region": "Контекст использования; canonical value указан в aliases.",
        "preference": "Мягкое пожелание; canonical value указан в aliases.",
        "feature": "Запрошенная возможность; boolean указан в scalar_aliases.",
        "maximum": "Верхняя граница в базовых единицах.",
        "minimum": "Нижняя граница в базовых единицах.",
        "budget_enum": "Категория бюджета; сумму без границ не угадывать.",
        "extra": "Служебное поле; не заполнять.",
    }
    for slot, field in result["field_names"].items():
        if field in result["schema"]["properties"]:
            result["schema"]["properties"][field]["description"] = descriptions[slot]
    result["schema"]["title"] = f"Request_{family_id}"
    result["capabilities"] = {
        "allowed_operations": ["set", "clear", "exclude", "include"],
        "preference_field": result["field_names"]["preference"],
        "unknown_value_policy": "preserve_as_issue_without_guessing",
    }
    return result


def _schema_mode(position: int) -> str:
    return ("numeric", "enum_budget", "absent", "numeric_alt")[position % 4]


def generate_split(split: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    spec = PARTITIONS[split]
    rows: list[dict[str, Any]] = []
    schemas: dict[str, dict[str, Any]] = {}
    registry = {"template_families": {}, "schema_families": {}, "domain_families": {}}
    schema_position = 0
    for domain_position, domain in enumerate(spec["domains"]):
        registry["domain_families"][domain.family] = {"partition": split, "domain_id": domain.domain_id}
        counterfactual_amount = 50 + domain_position * 5
        counterfactual_text = {
            "train": f"Не дороже {counterfactual_amount} {domain.amount_unit}.",
            "dev": f"Потолок расходов — {counterfactual_amount} {domain.amount_unit}.",
            "blind": f"Сумма выше {counterfactual_amount} {domain.amount_unit} не подходит.",
        }[split]
        for local_schema in range(spec["schemas_per_domain"]):
            ordinal = spec["schema_start"] + schema_position
            family_id = f"sf_{ordinal:03d}"
            schema_id = f"sc_{ordinal:03d}_{domain.domain_id}"
            registry["schema_families"][family_id] = {"partition": split, "domain_family_id": domain.family}
            rng = random.Random(seed + ordinal * 101)
            raw = build_schema(split, domain, ordinal, _schema_mode(local_schema), rng)
            schema_info = _compact_schema(raw, family_id, schema_id)
            public = {
                key: value for key, value in schema_info.items() if key not in {"field_names", "enum_surfaces", "enum_labels", "mode"}
            }
            schemas[schema_id] = public
            for case_position, scenario in enumerate(spec["scenarios"]):
                family_ordinal = spec["template_start"] + scenario
                template_id = f"tf_{family_ordinal:03d}"
                registry["template_families"][template_id] = {
                    "partition": split,
                    "semantic_scenario": scenario,
                    "renderer_variant": {"train": "directive", "dev": "criteria", "blind": "boundary"}[split],
                }
                text, target, tags = render_case(
                    split,
                    schema_info,
                    domain,
                    scenario,
                    ordinal * 31 + case_position,
                    counterfactual_text if scenario == 0 else None,
                )
                frame = PARTITION_FRAMES[split][ordinal % len(PARTITION_FRAMES[split])]
                text = f"{frame} {text}"
                rows.append(
                    {
                        "id": f"v2-{split}-{ordinal:03d}-{case_position:02d}",
                        "split": split,
                        "schema_id": schema_id,
                        "schema_family_id": family_id,
                        "domain_id": domain.domain_id,
                        "domain_family_id": domain.family,
                        "domain_family": domain.family,
                        "template_family_id": template_id,
                        "counterfactual_group_id": f"v2-{split}-{domain.domain_id}-schema-dependence" if scenario == 0 else None,
                        "source": "deterministic_schema_first_synthetic",
                        "generator_version": GENERATOR_VERSION,
                        "input": {
                            "message": text,
                            "previous": {},
                            "domain": public,
                            "pending_question": None,
                            "unresolved": [],
                        },
                        "target": target,
                        "tags": tags,
                    }
                )
            schema_position += 1
    return rows, schemas, registry


def _schema_fingerprint(schema: dict[str, Any]) -> str:
    properties = schema["schema"]["properties"]
    normalized = {
        "fields": {
            field: {
                "type": value.get("type"),
                "enum_cardinality": len(value.get("enum", [])),
                "minimum": value.get("minimum"),
                "description": normalize_text(value.get("description", "")),
            }
            for field, value in sorted(properties.items())
        },
        "required": sorted(schema["schema"].get("required", [])),
        "capabilities": schema["capabilities"],
    }
    return sha256_bytes(canonical_json(normalized))


def _enum_fingerprints(schema: dict[str, Any]) -> set[str]:
    return {sha256_bytes(canonical_json(sorted(prop["enum"]))) for prop in schema["schema"]["properties"].values() if prop.get("enum")}


def audit_dataset(splits: dict[str, list[dict]], schemas: dict[str, dict], registry: dict[str, Any]) -> dict[str, Any]:
    pairs = (("train", "dev"), ("train", "blind"), ("dev", "blind"))
    texts = {name: [row["input"]["message"] for row in rows] for name, rows in splits.items()}
    normalized = {name: [normalize_text(text) for text in values] for name, values in texts.items()}
    token_sets = {name: [set(text.split()) for text in values] for name, values in normalized.items()}
    report: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "grouping_keys": ["template_family_id", "schema_family_id", "domain_family_id"],
        "thresholds": {"token_set_jaccard": NEAR_DUPLICATE_THRESHOLD, "minimum_union_tokens": 5},
    }
    checks: dict[str, dict[str, Any]] = defaultdict(dict)
    maximum_jaccard: dict[str, float] = {}
    for left, right in pairs:
        pair = f"{left}__{right}"
        checks["exact_text_overlap"][pair] = len(set(texts[left]) & set(texts[right]))
        checks["normalized_text_overlap"][pair] = len(set(normalized[left]) & set(normalized[right]))
        for key in ("template_family_id", "schema_family_id", "domain_family_id"):
            checks[f"{key}_overlap"][pair] = len({row[key] for row in splits[left]} & {row[key] for row in splits[right]})
        left_schema_ids = {row["schema_id"] for row in splits[left]}
        right_schema_ids = {row["schema_id"] for row in splits[right]}
        checks["schema_id_overlap"][pair] = len(left_schema_ids & right_schema_ids)
        checks["field_name_set_overlap"][pair] = len(
            {tuple(sorted(schemas[key]["schema"]["properties"])) for key in left_schema_ids}
            & {tuple(sorted(schemas[key]["schema"]["properties"])) for key in right_schema_ids}
        )
        checks["schema_fingerprint_overlap"][pair] = len(
            {_schema_fingerprint(schemas[key]) for key in left_schema_ids} & {_schema_fingerprint(schemas[key]) for key in right_schema_ids}
        )
        left_enums = set().union(*(_enum_fingerprints(schemas[key]) for key in left_schema_ids))
        right_enums = set().union(*(_enum_fingerprints(schemas[key]) for key in right_schema_ids))
        checks["enum_set_overlap"][pair] = len(left_enums & right_enums)
        near = 0
        maximum = 0.0
        for left_tokens in token_sets[left]:
            for right_tokens in token_sets[right]:
                union = left_tokens | right_tokens
                score = len(left_tokens & right_tokens) / len(union) if union else 1.0
                maximum = max(maximum, score)
                near += int(len(union) >= 5 and score >= NEAR_DUPLICATE_THRESHOLD)
        checks["token_jaccard_near_duplicate_pairs"][pair] = near
        maximum_jaccard[pair] = round(maximum, 6)
    report.update({key: dict(value) for key, value in checks.items()})
    report["maximum_cross_split_token_jaccard"] = maximum_jaccard
    report["registry"] = registry
    report["repeated_example_ids"] = sum(map(len, splits.values())) - len({row["id"] for rows in splits.values() for row in rows})
    report["repeated_item_ids"] = 0
    report["repeated_catalog_entities"] = 0
    report["final_holdout_used"] = False
    numeric_checks = [value for check in checks.values() for value in check.values()]
    report["passed"] = all(value == 0 for value in numeric_checks) and report["repeated_example_ids"] == 0
    return report


def _validate(splits: dict[str, list[dict]], schemas: dict[str, dict], audit: dict[str, Any]) -> None:
    from recagent.interpretation import StructuredRequest

    for split, expected in SPLIT_COUNTS.items():
        if len(splits[split]) != expected:
            raise ValueError(f"{split}: {len(splits[split])} != {expected}")
        for row in splits[split]:
            StructuredRequest.model_validate(row["target"])
            if row["schema_id"] not in schemas:
                raise ValueError(f"unknown schema in {row['id']}")
            for update in row["target"]["updates"]:
                if update["source_text"].casefold() not in row["input"]["message"].casefold():
                    raise ValueError(f"ungrounded source in {row['id']}")
    if not audit["passed"]:
        raise ValueError("dataset v2 isolation audit failed")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def _write(path: Path, payload: bytes, sealed: bool = False) -> None:
    if sealed and path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"sealed artifact differs and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def generate(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    splits: dict[str, list[dict]] = {}
    schemas: dict[str, dict] = {}
    registries: dict[str, Any] = {}
    for split in ("train", "dev", "blind"):
        rows, split_schemas, registry = generate_split(split, seed)
        splits[split] = rows
        schemas.update(split_schemas)
        registries[split] = registry
    audit = audit_dataset(splits, schemas, registries)
    _validate(splits, schemas, audit)
    payloads = {split: _jsonl(rows) for split, rows in splits.items()}
    manifest_splits = {}
    for split, rows in splits.items():
        schema_ids = sorted({row["schema_id"] for row in rows})
        manifest_splits[split] = {
            "file": f"{split}.jsonl",
            "examples": len(rows),
            "sha256": sha256_bytes(payloads[split]),
            "domains": dict(sorted(Counter(row["domain_id"] for row in rows).items())),
            "domain_family_ids": sorted({row["domain_family_id"] for row in rows}),
            "schema_ids": schema_ids,
            "schema_family_ids": sorted({row["schema_family_id"] for row in rows}),
            "template_family_ids": sorted({row["template_family_id"] for row in rows}),
            "fact_and_difficulty_tags": dict(sorted(Counter(tag for row in rows for tag in row["tags"]).items())),
            "schema_fingerprints_sha256": sorted(_schema_fingerprint(schemas[key]) for key in schema_ids),
            "source": "AI-authored deterministic synthetic; no human-label claim",
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "sealed": split == "blind",
        }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generation_order": ["schema", "semantic_target", "user_utterance"],
        "split_unit": "group families, never individual examples",
        "grouping_keys": audit["grouping_keys"],
        "seed": seed,
        "expected_output_contract": "recagent.interpretation.StructuredRequest",
        "splits": manifest_splits,
        "TRAIN_SCHEMA_IDS": manifest_splits["train"]["schema_ids"],
        "DEV_SCHEMA_IDS": manifest_splits["dev"]["schema_ids"],
        "BLIND_SCHEMA_IDS": manifest_splits["blind"]["schema_ids"],
        "leakage_report_file": "leakage-report.json",
        "leakage_passed": audit["passed"],
        "prohibited_sources": {"final_holdout_used": False, "v1_blind_used": False, "catalog_used": False},
    }
    _write(output_dir / "train.jsonl", payloads["train"])
    _write(output_dir / "dev.jsonl", payloads["dev"])
    _write(output_dir / "blind.jsonl", payloads["blind"], sealed=True)
    _write(output_dir / "schemas.json", canonical_json(schemas) + b"\n")
    _write(output_dir / "leakage-report.json", canonical_json(audit) + b"\n")
    _write(output_dir / "manifest.json", canonical_json(manifest) + b"\n")
    seal = {
        "dataset_version": DATASET_VERSION,
        "split": "blind",
        "sealed_before_baseline": True,
        "examples": len(splits["blind"]),
        "sha256": sha256_bytes(payloads["blind"]),
        "schema_ids_sha256": sha256_bytes(canonical_json(manifest["BLIND_SCHEMA_IDS"])),
        "audit_sha256": sha256_bytes(canonical_json(audit) + b"\n"),
        "policy": "immutable; use a new dataset version for any change",
    }
    _write(output_dir / "blind.seal.json", canonical_json(seal) + b"\n", sealed=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/lora/semantic_extraction_v2"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = generate(args.output_dir, args.seed)
    print(
        json.dumps(
            {
                "dataset_version": manifest["dataset_version"],
                "counts": {split: block["examples"] for split, block in manifest["splits"].items()},
                "hashes": {split: block["sha256"] for split, block in manifest["splits"].items()},
                "leakage_passed": manifest["leakage_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
