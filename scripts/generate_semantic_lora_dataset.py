"""Generate and audit the schema-conditioned semantic LoRA pilot dataset.

The generator is deliberately closed-world and schema-first: it only uses the
domain specifications in this file, creates a schema, derives a semantic plan
from that schema, and finally renders an utterance.  It never reads repository
evaluation data, catalogs, FINAL holdouts, or the previous blind fixture.

The assistant target is exactly ``recagent.interpretation.StructuredRequest``.
All annotations outside ``target`` are provenance/evaluation metadata and are
not part of the model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "semantic-lora-schema-first-v1"
DATASET_VERSION = "semantic-lora-pilot-v1"
DEFAULT_SEED = 20260915
SPLIT_COUNTS = {"train": 1200, "dev": 180, "blind": 120}
NEAR_DUPLICATE_THRESHOLD = 0.80

CONTEXT_PARTS = {
    "train": (
        (
            "Критерии перечисляю",
            "Условия записываю",
            "Запрос формулирую",
            "Параметры уточняю",
            "Ограничения называю",
            "Пожелания отмечаю",
            "Выбор описываю",
            "Требования задаю",
        ),
        ("перед первым поиском", "для отдельного списка", "до просмотра выдачи", "для нового подбора", "в рамках одного обращения"),
    ),
    "dev": (
        (
            "Сведения фиксирую",
            "Рамки обозначаю",
            "Задачу описываю",
            "Указания собираю",
            "Ориентиры сообщаю",
            "Границы определяю",
            "Приоритеты передаю",
            "Детали свожу",
        ),
        ("до сравнения кандидатов", "для свежей подборки", "перед оценкой ответа", "для независимой проверки", "в новом сеансе выбора"),
    ),
    "blind": (
        (
            "Исходные данные даю",
            "Мои условия следуют",
            "Основания выбора такие",
            "Вводные перечислены",
            "Решение ищу",
            "Фильтры называю",
            "Нужды формулирую",
            "Контекст передаю",
        ),
        (
            "для закрытого просмотра",
            "перед самостоятельной оценкой",
            "для незнакомого набора",
            "в отдельной попытке",
            "до получения результата",
        ),
    ),
}


def schema_context(split: str, schema_index: int) -> str:
    starts, ends = CONTEXT_PARTS[split]
    return f"{starts[schema_index % len(starts)]} {ends[(schema_index // len(starts)) % len(ends)]}."


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    family: str
    nouns: tuple[str, str, str]
    styles: tuple[str, str, str]
    regions: tuple[str, str, str]
    preferences: tuple[str, str, str]
    feature_positive: str
    feature_negative: str
    amount_unit: str
    amount_multiplier: int
    unseen: bool = False


SEEN_DOMAINS = (
    DomainSpec(
        "electronics",
        "consumer_goods",
        ("ноутбук", "планшет", "монитор"),
        ("компактный", "мощный", "ремонтопригодный"),
        ("для дома", "для поездок", "для студии"),
        ("тихий", "лёгкий", "энергоэффективный"),
        "с автономной работой",
        "без автономной работы",
        "тыс. руб",
        1000,
    ),
    DomainSpec(
        "travel",
        "mobility",
        ("тур", "маршрут", "экскурсия"),
        ("спокойный", "насыщенный", "приключенческий"),
        ("у моря", "в горах", "в старом городе"),
        ("без пересадок", "с местной кухней", "свободный график"),
        "с трансфером",
        "без трансфера",
        "тыс. руб",
        1000,
    ),
    DomainSpec(
        "hotels",
        "hospitality",
        ("отель", "апарт-отель", "гостевой дом"),
        ("тихий", "семейный", "деловой"),
        ("у вокзала", "в центре", "за городом"),
        ("поздний заезд", "вид из окна", "просторный номер"),
        "с завтраком",
        "без завтрака",
        "тыс. руб",
        1000,
    ),
    DomainSpec(
        "books",
        "media",
        ("роман", "сборник", "справочник"),
        ("ироничный", "напряжённый", "созерцательный"),
        ("для дороги", "для дома", "для клуба"),
        ("короткие главы", "много диалогов", "без спойлеров"),
        "с иллюстрациями",
        "без иллюстраций",
        "страниц",
        1,
    ),
    DomainSpec(
        "dining",
        "hospitality",
        ("кафе", "бистро", "ресторан"),
        ("уютный", "оживлённый", "камерный"),
        ("у парка", "в центре", "на набережной"),
        ("быстрая подача", "тихая музыка", "открытая кухня"),
        "с верандой",
        "без веранды",
        "руб",
        1,
    ),
    DomainSpec(
        "housing",
        "property",
        ("студия", "квартира", "дом"),
        ("современный", "исторический", "минималистичный"),
        ("у метро", "у парка", "за городом"),
        ("тихий двор", "много света", "высокие потолки"),
        "с мебелью",
        "без мебели",
        "тыс. руб",
        1000,
    ),
    DomainSpec(
        "education",
        "learning",
        ("курс", "практикум", "семинар"),
        ("вводный", "прикладной", "исследовательский"),
        ("для дома", "в кампусе", "в группе"),
        ("обратная связь", "короткие уроки", "свободный темп"),
        "с практикой",
        "без практики",
        "часов",
        1,
    ),
    DomainSpec(
        "services",
        "professional_services",
        ("консультация", "аудит", "сопровождение"),
        ("экспресс", "углублённый", "пошаговый"),
        ("дистанционно", "в офисе", "с выездом"),
        ("письменный итог", "единое окно", "гибкое время"),
        "с поддержкой",
        "без поддержки",
        "часов",
        1,
    ),
)

BLIND_DOMAINS = (
    DomainSpec(
        "jobs",
        "employment",
        ("вакансия", "стажировка", "проектная роль"),
        ("исследовательская", "операционная", "творческая"),
        ("удалённо", "в гибриде", "в офисе"),
        ("гибкий график", "наставничество", "небольшая команда"),
        "с обучением",
        "без обучения",
        "тыс. руб",
        1000,
        True,
    ),
    DomainSpec(
        "saas",
        "business_software",
        ("сервис аналитики", "система поддержки", "платформа автоматизации"),
        ("модульный", "безкодовый", "корпоративный"),
        ("для команды", "для филиалов", "для подрядчиков"),
        ("быстрый запуск", "единый вход", "подробный аудит"),
        "с пробным периодом",
        "без пробного периода",
        "тыс. руб",
        1000,
        True,
    ),
    DomainSpec(
        "agriculture",
        "agritech",
        ("система полива", "датчик почвы", "полевой контроллер"),
        ("автономный", "модульный", "защищённый"),
        ("для теплицы", "для сада", "для поля"),
        ("редкое обслуживание", "точные замеры", "простая установка"),
        "с телеметрией",
        "без телеметрии",
        "тыс. руб",
        1000,
        True,
    ),
    DomainSpec(
        "logistics",
        "supply_chain",
        ("доставка", "складская услуга", "маршрут перевозки"),
        ("срочный", "бережный", "консолидированный"),
        ("по городу", "между регионами", "до пункта выдачи"),
        ("окно прибытия", "отслеживание", "единая накладная"),
        "со страхованием",
        "без страхования",
        "тыс. руб",
        1000,
        True,
    ),
)

FIELD_NAME_POOLS = {
    "category": ("kind", "offering_class", "selection_axis", "f_17", "profile", "object_family", "attribute_q"),
    "style": ("style", "experience_band", "presentation_mode", "attribute_b", "profile", "quality_axis", "f_29"),
    "region": ("location", "usage_context", "placement", "property_3", "zone", "context_band", "f_41"),
    "preference": ("preferred_trait", "soft_priority", "wish_axis", "attribute_p", "profile", "preference_band", "f_53"),
    "feature": ("feature_flag", "option_enabled", "capability_x", "attribute_t", "support_mode", "toggle_7", "f_61"),
    "maximum": ("max_cost", "upper_limit", "ceiling_value", "f_71", "amount_cap", "threshold_high", "limit"),
    "minimum": ("min_cost", "lower_limit", "floor_value", "f_73", "amount_floor", "threshold_low", "limit"),
    "budget_enum": ("budget_tier", "spend_band", "cost_class", "attribute_c", "profile", "f_79", "price_band"),
    "extra": ("internal_rank", "display_hint", "provider_note", "f_97", "debug_bucket", "sort_key", "tenant_marker"),
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    folded = text.casefold().replace("ё", "е")
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", folded).split())


def _enum_labels(split: str, schema_index: int, slot: str) -> list[str]:
    prefix = {"train": "tr", "dev": "dv", "blind": "zb"}[split]
    digest = hashlib.sha256(f"{split}:{schema_index}:{slot}".encode()).hexdigest()
    return [f"{prefix}_{digest[i : i + 5]}" for i in (0, 5, 10)]


def _field_name(slot: str, schema_index: int, offset: int) -> str:
    pool = FIELD_NAME_POOLS[slot]
    base = pool[(schema_index + offset) % len(pool)]
    # Some opaque fields remain opaque; descriptive aliases are schema-local too.
    return base if base.startswith(("f_", "attribute_")) else f"{base}_{schema_index % 11}"


def build_schema(split: str, domain: DomainSpec, schema_index: int, mode: str, rng: random.Random) -> dict[str, Any]:
    slot_surfaces = {
        "category": domain.nouns,
        "style": domain.styles,
        "region": domain.regions,
        "preference": domain.preferences,
    }
    field_names: dict[str, str] = {}
    used: set[str] = set()
    for offset, slot in enumerate(("category", "style", "region", "preference", "feature", "maximum", "minimum", "extra")):
        candidate = _field_name(slot, schema_index, offset)
        if candidate in used:
            candidate = f"{candidate}_{slot[:2]}"
        field_names[slot] = candidate
        used.add(candidate)
    if mode == "enum_budget":
        candidate = _field_name("budget_enum", schema_index, 3)
        if candidate in used:
            candidate = f"{candidate}_bu"
        field_names["budget_enum"] = candidate
        used.add(candidate)

    properties: dict[str, Any] = {}
    aliases: dict[str, dict[str, Any]] = {}
    scalar_aliases: dict[str, dict[str, bool]] = {}
    numeric_units: dict[str, dict[str, int]] = {}
    enum_surfaces: dict[str, tuple[str, str, str]] = {}
    enum_labels: dict[str, list[str]] = {}

    for slot in ("category", "style", "region", "preference"):
        field = field_names[slot]
        labels = _enum_labels(split, schema_index, slot)
        surfaces = slot_surfaces[slot]
        pairs = list(zip(surfaces, labels, strict=True))
        rng.shuffle(pairs)
        aliases[field] = dict(pairs)
        enum_surfaces[slot] = surfaces
        enum_labels[slot] = labels
        descriptions = {
            "category": "Класс объекта, явно названный пользователем; каноническое значение определяется aliases.",
            "style": "Характер или режим впечатления; не выводить его из названия класса объекта.",
            "region": "Контекст использования или расположения, только если он явно указан.",
            "preference": "Мягко предпочитаемое свойство; не превращать его в другое обязательное поле.",
        }
        properties[field] = {"type": "string", "enum": [label for _, label in pairs], "description": descriptions[slot]}

    feature = field_names["feature"]
    properties[feature] = {
        "type": "boolean",
        "description": "Наличие явно запрошенной доменной возможности; отрицательная форма означает false.",
    }
    scalar_aliases[feature] = {domain.feature_positive: True, domain.feature_negative: False}

    if mode in {"numeric", "numeric_alt"}:
        for slot, description in (
            ("maximum", "Верхняя граница величины в базовых единицах; фразы 'не больше' и 'до' задают это поле."),
            ("minimum", "Нижняя граница величины в базовых единицах; фразы 'не меньше' и 'от' задают это поле."),
        ):
            field = field_names[slot]
            properties[field] = {"type": "integer", "minimum": 0, "description": description}
            numeric_units[field] = {domain.amount_unit: domain.amount_multiplier}
    elif mode == "enum_budget":
        field = field_names["budget_enum"]
        labels = _enum_labels(split, schema_index, "budget_enum")
        properties[field] = {
            "type": "string",
            "enum": list(reversed(labels)),
            "description": "Категориальная политика бюджета. Точная сумма не определяет категорию без явно заданных границ.",
        }
        aliases[field] = {"экономный пакет": labels[0], "обычный пакет": labels[1], "расширенный пакет": labels[2]}

    extra = field_names["extra"]
    properties[extra] = {"type": "string", "description": "Служебное необязательное поле провайдера; пользовательская речь его не задаёт."}
    shuffled = list(properties.items())
    rng.shuffle(shuffled)
    properties = dict(shuffled)
    possible_required = [field_names["category"], field_names["region"], field_names["feature"]]
    required = rng.sample(possible_required, k=schema_index % 3)
    schema_id = f"{split}-{domain.domain_id}-schema-{schema_index:03d}"
    family_id = f"{split}-{domain.family}-family-{schema_index // 3:02d}"
    return {
        "schema_id": schema_id,
        "schema_family_id": family_id,
        "domain_id": domain.domain_id,
        "domain_family": domain.family,
        "mode": mode,
        "field_names": field_names,
        "enum_surfaces": enum_surfaces,
        "enum_labels": enum_labels,
        "schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"CustomerRequest_{schema_index:03d}",
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        },
        "aliases": aliases,
        "scalar_aliases": scalar_aliases,
        "numeric_units": numeric_units,
        "capabilities": {
            "allowed_operations": ["set", "clear", "exclude", "include"],
            "preference_field": field_names["preference"],
            "hard_constraint_fields": [name for slot, name in field_names.items() if slot != "extra"],
            "unknown_value_policy": "preserve_as_issue_without_guessing",
        },
    }


def _target(
    intent: str | None = "discovery",
    updates: list[dict[str, Any]] | None = None,
    issues: list[dict[str, str]] | None = None,
    *,
    reset: bool = False,
) -> dict[str, Any]:
    issues = issues or []
    return {
        "intent": intent,
        "updates": updates or [],
        "reset_constraints": reset,
        "issues": issues,
        "clarification_required": bool(issues),
    }


def _update(field: str, value: str | int | bool | None, source: str, operation: str = "set") -> dict[str, Any]:
    return {"field": field, "operation": operation, "value": value, "source_text": source}


def _issue(kind: str, field: str, value: str, message: str) -> dict[str, str]:
    return {"kind": kind, "field": field, "value": value, "message": message}


def render_case(
    split: str, schema_info: dict[str, Any], domain: DomainSpec, scenario: int, local_index: int, counterfactual_text: str | None = None
) -> tuple[str, dict[str, Any], list[str]]:
    fields = schema_info["field_names"]
    mode = schema_info["mode"]
    noun_a, noun_b, noun_c = domain.nouns
    style_a, style_b, style_c = domain.styles
    region_a, region_b, region_c = domain.regions
    pref_a, pref_b, pref_c = domain.preferences
    alias = schema_info["aliases"]

    def enum_value(slot: str, surface: str) -> str:
        return alias[fields[slot]][surface]

    amount_a = 20 + (local_index % 7) * 5
    amount_b = amount_a + 30
    numeric_a = amount_a * domain.amount_multiplier
    numeric_b = amount_b * domain.amount_multiplier
    max_field = fields.get("maximum")
    min_field = fields.get("minimum")
    split_lead = {"train": "Подбери", "dev": "Помоги выбрать", "blind": "Ищу подходящий"}[split]

    if scenario == 0:
        text = counterfactual_text or f"Не дороже {amount_b} {domain.amount_unit}."
        counterfactual_amount = int(re.search(r"\d+", text).group())
        counterfactual_numeric = counterfactual_amount * domain.amount_multiplier
        source = {
            "train": f"не дороже {counterfactual_amount} {domain.amount_unit}",
            "dev": f"потолок расходов — {counterfactual_amount} {domain.amount_unit}",
            "blind": f"выше {counterfactual_amount} {domain.amount_unit}",
        }[split]
        if mode in {"numeric", "numeric_alt"}:
            return text, _target(updates=[_update(max_field, counterfactual_numeric, source)]), ["counterfactual", "max", "hard_constraint"]
        if mode == "enum_budget":
            field = fields["budget_enum"]
            issue = _issue(
                "unsupported_constraint",
                field,
                str(counterfactual_numeric),
                "Точная сумма не определяет категорию бюджета без границ в schema.",
            )
            return text, _target(issues=[issue]), ["counterfactual", "unsupported", "no_map", "hard_constraint"]
        issue = _issue("unsupported_constraint", "request", str(counterfactual_numeric), "Schema не содержит поля для ограничения суммы.")
        return text, _target(issues=[issue]), ["counterfactual", "unsupported", "no_map", "hard_constraint"]
    if scenario == 1:
        return (
            f"{split_lead} {noun_a}.",
            _target(updates=[_update(fields["category"], enum_value("category", noun_a), noun_a)]),
            ["single_fact", "kind"],
        )
    if scenario == 2:
        return (
            f"Хочется что-то {style_b}.",
            _target(updates=[_update(fields["style"], enum_value("style", style_b), style_b)]),
            ["single_fact", "style"],
        )
    if scenario == 3:
        text = {
            "train": f"{split_lead} {noun_b}: {style_a}, {region_b}.",
            "dev": f"Среди вариантов нужен {noun_b}; атмосфера {style_a}, место — {region_b}.",
            "blind": f"Рассматриваю {noun_b}, причём {region_b} и обязательно {style_a}.",
        }[split]
        updates = [
            _update(fields["category"], enum_value("category", noun_b), noun_b),
            _update(fields["style"], enum_value("style", style_a), style_a),
            _update(fields["region"], enum_value("region", region_b), region_b),
        ]
        return text, _target(updates=updates), ["compound", "kind", "hard_constraint"]
    if scenario == 4:
        text = {
            "train": f"Только не {noun_c}, остальные варианты можно.",
            "dev": f"Исключите {noun_c}; другие классы допустимы.",
            "blind": f"{noun_c.capitalize()} прошу не предлагать, запрет только на него.",
        }[split]
        return (
            text,
            _target(updates=[_update(fields["category"], enum_value("category", noun_c), noun_c, "exclude")]),
            ["negation", "exclusion", "polarity"],
        )
    if scenario == 5:
        text = f"{noun_c} снова можно включить в поиск."
        return (
            text,
            _target(updates=[_update(fields["category"], enum_value("category", noun_c), noun_c, "include")]),
            ["include", "polarity"],
        )
    if scenario == 6:
        source = domain.feature_positive
        return f"Важно, чтобы было {source}.", _target(updates=[_update(fields["feature"], True, source)]), ["boolean", "hard_constraint"]
    if scenario == 7:
        source = domain.feature_negative
        return f"Мне нужно {source}.", _target(updates=[_update(fields["feature"], False, source)]), ["boolean", "negation", "polarity"]
    if scenario in {8, 9, 10} and mode not in {"numeric", "numeric_alt"}:
        issue = _issue(
            "unsupported_constraint",
            "request",
            f"{amount_a}..{amount_b}",
            "Schema не представляет числовые границы в запрошенных единицах.",
        )
        templates = {
            8: {
                "train": f"Верхний предел — {amount_b} {domain.amount_unit}.",
                "dev": f"Выше {amount_b} {domain.amount_unit} ничего не рассматриваю.",
                "blind": f"Порог сверху равен {amount_b} {domain.amount_unit}.",
            },
            9: {
                "train": f"Нижний предел — {amount_a} {domain.amount_unit}.",
                "dev": f"Начинать нужно с {amount_a} {domain.amount_unit}.",
                "blind": f"Порог снизу равен {amount_a} {domain.amount_unit}.",
            },
            10: {
                "train": f"Нужен диапазон от {amount_a} до {amount_b} {domain.amount_unit}.",
                "dev": f"Задаю нижнюю планку {amount_a} и верхнюю {amount_b} {domain.amount_unit}.",
                "blind": f"Вилка начинается с {amount_a}, заканчивается на {amount_b} {domain.amount_unit}.",
            },
        }
        text = templates[scenario][split]
        return text, _target(issues=[issue]), ["range", "unsupported", "no_map"]
    if scenario == 8:
        source = f"до {amount_b} {domain.amount_unit}"
        return f"Ограничение — {source}.", _target(updates=[_update(max_field, numeric_b, source)]), ["max", "units", "hard_constraint"]
    if scenario == 9:
        source = f"от {amount_a} {domain.amount_unit}"
        return (
            f"Рассматриваю варианты {source}.",
            _target(updates=[_update(min_field, numeric_a, source)]),
            ["min", "units", "hard_constraint"],
        )
    if scenario == 10:
        source_min = f"от {amount_a}"
        source_max = f"до {amount_b} {domain.amount_unit}"
        text = {
            "train": f"Нужен диапазон {source_min} {source_max}.",
            "dev": f"Нижняя планка {source_min}, верхняя — {source_max}.",
            "blind": f"Вилка значений: {source_min}; при этом предел {source_max}.",
        }[split]
        return (
            text,
            _target(updates=[_update(min_field, numeric_a, source_min), _update(max_field, numeric_b, source_max)]),
            ["range", "compound", "units", "hard_constraint"],
        )
    if scenario == 11:
        source = pref_a
        return (
            f"Если получится, предпочту {source}.",
            _target(updates=[_update(fields["preference"], enum_value("preference", source), source)]),
            ["preference", "soft_constraint"],
        )
    if scenario == 12:
        pieces = [noun_a, style_c, region_a, pref_b, domain.feature_positive]
        text = {
            "train": f"Нужен {pieces[0]}, по характеру {pieces[1]}, {pieces[2]}; желательно {pieces[3]} и обязательно {pieces[4]}.",
            "dev": f"Выбираю {pieces[0]}: место {pieces[2]}, режим {pieces[1]}; было бы хорошо {pieces[3]}, но {pieces[4]} строго обязательно.",
            "blind": f"Для меня подходит {pieces[0]} только {pieces[2]} и с признаком {pieces[4]}; дополнительно ценю {pieces[3]}, настроение — {pieces[1]}.",
        }[split]
        updates = [
            _update(fields["category"], enum_value("category", noun_a), noun_a),
            _update(fields["style"], enum_value("style", style_c), style_c),
            _update(fields["region"], enum_value("region", region_a), region_a),
            _update(fields["preference"], enum_value("preference", pref_b), pref_b),
            _update(fields["feature"], True, domain.feature_positive),
        ]
        return text, _target(updates=updates), ["compound", "long", "kind", "preference", "hard_constraint"]
    if scenario == 13:
        value = "что-нибудь среднее"
        issue = _issue("ambiguity", fields["style"], value, "Фраза не различает несколько значений schema.")
        text = {
            "train": f"По характеру хочу {value}, но точнее пока не скажу.",
            "dev": f"Пусть режим будет {value}; выбрать один из близких вариантов не могу.",
            "blind": f"Остановился на формулировке '{value}', без уточнения категории.",
        }[split]
        return text, _target(issues=[issue]), ["ambiguity", "no_map", "explicit_uncertainty"]
    if scenario == 14:
        text = {
            "train": f"Хочу одновременно {style_a} и {style_c}, хотя понимаю, что это противоречиво.",
            "dev": f"Поставьте два режима сразу: {style_a}, а ещё {style_c}; выбирать между ними не буду.",
            "blind": f"Требования сталкиваются: вариант должен быть {style_a} и в то же время {style_c}.",
        }[split]
        updates = [
            _update(fields["style"], enum_value("style", style_a), style_a),
            _update(fields["style"], enum_value("style", style_c), style_c),
        ]
        issue = _issue("conflict", fields["style"], f"{style_a} | {style_c}", "Одно поле получило два несовместимых значения.")
        return text, _target(updates=updates, issues=[issue]), ["conflict", "compound", "polarity"]
    if scenario == 15:
        concept = "гарантированная доставка за десять минут"
        issue = _issue("unsupported_constraint", "request", concept, "В schema отсутствует поле для этого требования.")
        text = {
            "train": f"Ещё нужна {concept}.",
            "dev": f"Отдельное требование — {concept}; не уверен, поддерживается ли оно.",
            "blind": f"Без условия '{concept}' вариант не подойдёт.",
        }[split]
        return text, _target(issues=[issue]), ["unsupported", "no_map", "absent_field"]
    if scenario == 16:
        raw = "ультраредкий режим"
        issue = _issue(
            "unsupported_constraint", fields["style"], raw, "Значение отсутствует в enum и aliases; ближайшее значение выбирать нельзя."
        )
        return (
            f"По характеру нужен {raw}.",
            _target(updates=[_update(fields["style"], raw, raw)], issues=[issue]),
            ["unsupported", "unknown_enum", "no_canonical_map"],
        )
    if scenario == 17:
        value = "конкретный тип"
        issue = _issue("ambiguity", fields["category"], value, "Обязательное значение не названо.")
        text = {
            "train": "Подберите, пожалуйста, но конкретный тип я пока не выбрал.",
            "dev": "Начните поиск, хотя класс объекта я назвать ещё не готов.",
            "blind": "Нужен результат, но категорию оставляю неопределённой.",
        }[split]
        return text, _target(issues=[issue]), ["missing_value", "ambiguity", "no_map"]
    if scenario == 18:
        text = {
            "train": "Я сравниваю варианты вечером и никуда не тороплюсь.",
            "dev": "Пока просто изучаю, как устроен процесс выбора.",
            "blind": "Сообщаю лишь, что решение приму не сегодня.",
        }[split]
        return text, _target(intent=None), ["irrelevant", "no_map", "negative_control"]
    if scenario == 19:
        if mode not in {"numeric", "numeric_alt"}:
            issue = _issue("unsupported_constraint", "request", str(numeric_b), "Schema не содержит надёжного числового отображения.")
            return (
                f"{noun_b}. До {amount_b} {domain.amount_unit}.",
                _target(updates=[_update(fields["category"], enum_value("category", noun_b), noun_b)], issues=[issue]),
                ["terse", "compound", "unsupported"],
            )
        text = f"{noun_b}. До {amount_b} {domain.amount_unit}."
        return (
            text,
            _target(
                updates=[
                    _update(fields["category"], enum_value("category", noun_b), noun_b),
                    _update(max_field, numeric_b, f"до {amount_b} {domain.amount_unit}"),
                ]
            ),
            ["terse", "compound", "kind", "max"],
        )
    if scenario == 20:
        text = f"Эм, наверное {noun_a}, точно {region_c}; {style_b}, и, пожалуйста, {domain.feature_negative}."
        updates = [
            _update(fields["category"], enum_value("category", noun_a), noun_a),
            _update(fields["region"], enum_value("region", region_c), region_c),
            _update(fields["style"], enum_value("style", style_b), style_b),
            _update(fields["feature"], False, domain.feature_negative),
        ]
        return text, _target(updates=updates), ["noisy", "compound", "negation", "kind"]
    if scenario == 21:
        source = "характер больше не важен"
        return (
            f"{source.capitalize()}, снимите это условие.",
            _target(updates=[_update(fields["style"], None, source, "clear")]),
            ["clear", "polarity"],
        )
    if scenario == 22:
        return (
            f"Под настроение хочется чего-то {style_a}.",
            _target(intent="mood", updates=[_update(fields["style"], enum_value("style", style_a), style_a)]),
            ["intent", "mood"],
        )
    if scenario == 23:
        text = f"Скорее {pref_c}, но {region_b} — обязательное условие."
        updates = [
            _update(fields["preference"], enum_value("preference", pref_c), pref_c),
            _update(fields["region"], enum_value("region", region_b), region_b),
        ]
        return text, _target(updates=updates), ["preference", "soft_constraint", "hard_constraint", "compound"]
    if scenario == 24:
        text = f"Не {noun_a} и не {noun_b}; {noun_c} допустим."
        updates = [
            _update(fields["category"], enum_value("category", noun_a), noun_a, "exclude"),
            _update(fields["category"], enum_value("category", noun_b), noun_b, "exclude"),
            _update(fields["category"], enum_value("category", noun_c), noun_c),
        ]
        return text, _target(updates=updates), ["negation", "exclusion", "compound", "polarity"]
    if scenario == 25:
        if mode not in {"numeric", "numeric_alt"}:
            issue = _issue("unsupported_constraint", "request", f">={amount_b}, <={amount_a}", "Schema не представляет числовой конфликт.")
            text = {
                "train": f"Не меньше {amount_b}, но и не больше {amount_a} {domain.amount_unit}.",
                "dev": f"Нижняя граница — не меньше {amount_b}, верхняя — не больше {amount_a} {domain.amount_unit}.",
                "blind": f"Требую не меньше {amount_b}; одновременно предел — не больше {amount_a} {domain.amount_unit}.",
            }[split]
            return text, _target(issues=[issue]), ["conflict", "range", "unsupported", "no_map"]
        text = {
            "train": f"Не меньше {amount_b}, но и не больше {amount_a} {domain.amount_unit}.",
            "dev": f"Нижняя граница — не меньше {amount_b}, верхняя — не больше {amount_a} {domain.amount_unit}.",
            "blind": f"Требую не меньше {amount_b}; одновременно предел — не больше {amount_a} {domain.amount_unit}.",
        }[split]
        updates = [
            _update(min_field, numeric_b, f"не меньше {amount_b}"),
            _update(max_field, numeric_a, f"не больше {amount_a} {domain.amount_unit}"),
        ]
        issue = _issue("conflict", "request", f"{numeric_b}>{numeric_a}", "Нижняя граница превышает верхнюю.")
        return text, _target(updates=updates, issues=[issue]), ["conflict", "range", "compound"]
    if scenario == 26:
        value = "удобный"
        issue = _issue("ambiguity", "request", value, "Описание одинаково правдоподобно для нескольких полей schema.")
        return (
            f"Главное, чтобы вариант был {value}; не знаю, к какому свойству это отнести.",
            _target(issues=[issue]),
            ["ambiguity", "multiple_mapping", "no_map"],
        )
    if scenario == 27:
        raw = f"до {amount_b} попугаев"
        issue = _issue(
            "unsupported_constraint",
            max_field if mode in {"numeric", "numeric_alt"} else "request",
            raw,
            "Единица измерения отсутствует в numeric_units.",
        )
        return f"Лимит странный: {raw}.", _target(issues=[issue]), ["unsupported", "units", "no_map"]
    if scenario == 28:
        issue = _issue("ambiguity", fields["category"], "", "Schema требует класс объекта, но пользователь его не указал.")
        return (
            "Хочу посмотреть доступные варианты, подробностей пока нет.",
            _target(issues=[issue]),
            ["missing_required", "ambiguity", "no_map"],
        )
    text = f"{noun_c} обязателен, а {pref_a} — просто пожелание."
    updates = [
        _update(fields["category"], enum_value("category", noun_c), noun_c),
        _update(fields["preference"], enum_value("preference", pref_a), pref_a),
    ]
    return text, _target(updates=updates), ["compound", "kind", "preference", "hard_constraint", "soft_constraint"]


def _modes(split: str) -> tuple[str, ...]:
    return ("numeric", "enum_budget", "absent") if split != "train" else ("numeric", "enum_budget", "absent", "numeric_alt", "numeric")


def generate_split(
    split: str, domains: tuple[DomainSpec, ...], schema_count_per_domain: int, examples_per_schema: list[int], seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    schemas: dict[str, dict[str, Any]] = {}
    schema_global = 0
    for domain_index, domain in enumerate(domains):
        counterfactual_amount = 50 + domain_index * 5
        counterfactual_text = {
            "train": f"Не дороже {counterfactual_amount} {domain.amount_unit}.",
            "dev": f"Потолок расходов — {counterfactual_amount} {domain.amount_unit}.",
            "blind": f"Сумма выше {counterfactual_amount} {domain.amount_unit} не подходит.",
        }[split]
        for local_schema in range(schema_count_per_domain):
            schema_seed = seed + {"train": 1000, "dev": 2000, "blind": 3000}[split] + schema_global
            schema_rng = random.Random(schema_seed)
            mode = _modes(split)[local_schema % len(_modes(split))]
            schema_info = build_schema(split, domain, schema_global, mode, schema_rng)
            schema_id = schema_info["schema_id"]
            public_schema = {
                key: value for key, value in schema_info.items() if key not in {"field_names", "enum_surfaces", "enum_labels", "mode"}
            }
            schemas[schema_id] = public_schema
            count = examples_per_schema[schema_global]
            if split == "train":
                scenarios = list(range(30))
            elif split == "dev":
                base = [0, 3, 4, 10, 13, 15, 18, 25]
                scenarios = base[:count]
            else:
                scenarios = [0, 3, 4, 10, 12, 13, 14, 15, 17, 25]
            for case_pos, scenario in enumerate(scenarios):
                text, target, tags = render_case(
                    split,
                    schema_info,
                    domain,
                    scenario,
                    schema_global * 31 + case_pos,
                    counterfactual_text if scenario == 0 else None,
                )
                if scenario != 0:
                    text = f"{text} {schema_context(split, schema_global)}"
                case_id = f"{split}-{schema_global:03d}-{case_pos:02d}"
                cf_group = f"{split}-{domain.domain_id}-amount-schema-dependence" if scenario == 0 else None
                rows.append(
                    {
                        "id": case_id,
                        "split": split,
                        "schema_id": schema_id,
                        "schema_family_id": schema_info["schema_family_id"],
                        "domain_id": domain.domain_id,
                        "domain_family": domain.family,
                        "template_family_id": f"{split}-template-{scenario:02d}",
                        "counterfactual_group_id": cf_group,
                        "source": "deterministic_schema_first_synthetic",
                        "generator_version": GENERATOR_VERSION,
                        "input": {
                            "message": text,
                            "previous": {},
                            "domain": public_schema,
                            "pending_question": None,
                            "unresolved": [],
                        },
                        "target": target,
                        "tags": tags,
                    }
                )
            schema_global += 1
    return rows, schemas


def _schema_fingerprint(schema: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json({key: schema[key] for key in ("schema", "aliases", "scalar_aliases", "numeric_units", "capabilities")})
    )


def _enum_set_fingerprints(schema: dict[str, Any]) -> set[str]:
    results = set()
    for spec in schema["schema"]["properties"].values():
        values = spec.get("enum")
        if values:
            results.add(sha256_bytes(canonical_json(sorted(values))))
    return results


def leakage_report(splits: dict[str, list[dict[str, Any]]], schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = tuple(splits)
    exact_overlap: dict[str, int] = {}
    normalized_overlap: dict[str, int] = {}
    template_overlap: dict[str, int] = {}
    schema_overlap: dict[str, int] = {}
    enum_overlap: dict[str, int] = {}
    near_duplicate_counts: dict[str, int] = {}
    max_jaccard: dict[str, float] = {}
    texts = {name: [row["input"]["message"] for row in rows] for name, rows in splits.items()}
    normalized = {name: [normalize_text(text) for text in values] for name, values in texts.items()}
    token_sets = {name: [set(text.split()) for text in values] for name, values in normalized.items()}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pair = f"{left}__{right}"
            exact_overlap[pair] = len(set(texts[left]) & set(texts[right]))
            normalized_overlap[pair] = len(set(normalized[left]) & set(normalized[right]))
            template_overlap[pair] = len({r["template_family_id"] for r in splits[left]} & {r["template_family_id"] for r in splits[right]})
            left_schema_ids = {r["schema_id"] for r in splits[left]}
            right_schema_ids = {r["schema_id"] for r in splits[right]}
            left_fps = {_schema_fingerprint(schemas[schema_id]) for schema_id in left_schema_ids}
            right_fps = {_schema_fingerprint(schemas[schema_id]) for schema_id in right_schema_ids}
            schema_overlap[pair] = len(left_fps & right_fps)
            left_enums = set().union(*(_enum_set_fingerprints(schemas[schema_id]) for schema_id in left_schema_ids))
            right_enums = set().union(*(_enum_set_fingerprints(schemas[schema_id]) for schema_id in right_schema_ids))
            enum_overlap[pair] = len(left_enums & right_enums)
            count = 0
            maximum = 0.0
            for left_tokens in token_sets[left]:
                for right_tokens in token_sets[right]:
                    union = left_tokens | right_tokens
                    score = len(left_tokens & right_tokens) / len(union) if union else 1.0
                    maximum = max(maximum, score)
                    if len(union) >= 5 and score >= NEAR_DUPLICATE_THRESHOLD:
                        count += 1
            near_duplicate_counts[pair] = count
            max_jaccard[pair] = round(maximum, 6)

    all_ids = [row["id"] for rows in splits.values() for row in rows]
    duplicate_groups: dict[str, int] = {}
    for split, values in normalized.items():
        groups = Counter(values)
        duplicate_groups[split] = sum(1 for count in groups.values() if count > 1)
        allowed = defaultdict(set)
        for row, value in zip(splits[split], values, strict=True):
            if groups[value] > 1:
                allowed[value].add(row["counterfactual_group_id"])
        if any(None in group_ids or len(group_ids) != 1 for group_ids in allowed.values()):
            raise ValueError(f"unapproved within-{split} exact duplicate")

    report = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "thresholds": {"token_set_jaccard": NEAR_DUPLICATE_THRESHOLD, "minimum_union_tokens": 5},
        "exact_text_overlap": exact_overlap,
        "normalized_text_overlap": normalized_overlap,
        "token_jaccard_near_duplicate_pairs": near_duplicate_counts,
        "maximum_cross_split_token_jaccard": max_jaccard,
        "template_family_overlap": template_overlap,
        "schema_fingerprint_overlap": schema_overlap,
        "enum_set_fingerprint_overlap": enum_overlap,
        "within_split_exact_duplicate_groups": duplicate_groups,
        "repeated_example_ids": len(all_ids) - len(set(all_ids)),
        "repeated_item_ids": 0,
        "repeated_catalog_entities": 0,
        "catalog_identity_fields_present": 0,
        "external_corpora_loaded": [],
        "final_holdout_used": False,
        "old_blind_used": False,
        "catalog_used": False,
    }
    zero_checks = (
        exact_overlap,
        normalized_overlap,
        near_duplicate_counts,
        template_overlap,
        schema_overlap,
        enum_overlap,
    )
    report["passed"] = all(value == 0 for check in zero_checks for value in check.values()) and report["repeated_example_ids"] == 0
    return report


def validate_dataset(splits: dict[str, list[dict[str, Any]]], schemas: dict[str, dict[str, Any]], report: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from recagent.interpretation import StructuredRequest

    for split, expected_count in SPLIT_COUNTS.items():
        rows = splits[split]
        if len(rows) != expected_count:
            raise ValueError(f"{split}: expected {expected_count} rows, got {len(rows)}")
        for row in rows:
            if row["split"] != split or row["schema_id"] not in schemas:
                raise ValueError(f"invalid provenance in {row['id']}")
            try:
                StructuredRequest.model_validate(row["target"])
            except ValidationError as exc:
                raise ValueError(f"invalid StructuredRequest target in {row['id']}: {exc}") from exc
            if set(row["target"]) != {"intent", "updates", "reset_constraints", "issues", "clarification_required"}:
                raise ValueError(f"training-only target key in {row['id']}")
            message = row["input"]["message"]
            for update in row["target"]["updates"]:
                if update["source_text"].casefold() not in message.casefold():
                    raise ValueError(f"ungrounded update source in {row['id']}")
            forbidden_keys = {"item_id", "product_id", "catalog_id", "sku", "title"}
            if forbidden_keys & set(row):
                raise ValueError(f"catalog identity key in {row['id']}")
    train_domains = {row["domain_id"] for row in splits["train"]}
    blind_domains = {row["domain_id"] for row in splits["blind"]}
    train_families = {row["domain_family"] for row in splits["train"]}
    blind_families = {row["domain_family"] for row in splits["blind"]}
    if train_domains & blind_domains or train_families & blind_families:
        raise ValueError("blind domain/schema family is not unseen")
    if not report["passed"]:
        raise ValueError(f"cross-split leakage detected: {json.dumps(report, ensure_ascii=False)}")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def _write_frozen(path: Path, payload: bytes, *, frozen: bool) -> None:
    if frozen and path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"sealed artifact differs and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_manifest(
    splits: dict[str, list[dict[str, Any]]],
    schemas: dict[str, dict[str, Any]],
    payloads: dict[str, bytes],
    leakage: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    manifest_splits: dict[str, Any] = {}
    for split, rows in splits.items():
        tag_counts = Counter(tag for row in rows for tag in row["tags"])
        domain_counts = Counter(row["domain_id"] for row in rows)
        schema_ids = sorted({row["schema_id"] for row in rows})
        manifest_splits[split] = {
            "file": f"{split}.jsonl",
            "examples": len(rows),
            "sha256": sha256_bytes(payloads[split]),
            "domains": dict(sorted(domain_counts.items())),
            "domain_families": sorted({row["domain_family"] for row in rows}),
            "schema_ids": schema_ids,
            "schema_family_ids": sorted({row["schema_family_id"] for row in rows}),
            "schema_fingerprints_sha256": sorted(_schema_fingerprint(schemas[schema_id]) for schema_id in schema_ids),
            "fact_and_difficulty_tags": dict(sorted(tag_counts.items())),
            "source": "AI-authored deterministic synthetic; no human-label claim",
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "sealed": split == "blind",
        }
    return {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generation_order": ["schema", "semantic_target", "user_utterance"],
        "seed": seed,
        "expected_output_contract": "recagent.interpretation.StructuredRequest",
        "target_contract_sha256": sha256_bytes(
            canonical_json(
                {
                    "intent": ["discovery", "similar", "mood", "navigation", None],
                    "update_operations": ["set", "clear", "exclude", "include"],
                    "issue_kinds": ["ambiguity", "unsupported_constraint", "conflict"],
                }
            )
        ),
        "splits": manifest_splits,
        "TRAIN_SCHEMA_IDS": manifest_splits["train"]["schema_ids"],
        "DEV_SCHEMA_IDS": manifest_splits["dev"]["schema_ids"],
        "BLIND_SCHEMA_IDS": manifest_splits["blind"]["schema_ids"],
        "blind_isolation": {
            "unseen_domains": sorted({row["domain_id"] for row in splits["blind"]}),
            "unseen_domain_families": sorted({row["domain_family"] for row in splits["blind"]}),
            "schema_overlap_with_train": 0,
            "template_overlap_with_train": 0,
        },
        "counterfactual_groups": {
            split: len({row["counterfactual_group_id"] for row in rows if row["counterfactual_group_id"]}) for split, rows in splits.items()
        },
        "leakage_report_file": "leakage-report.json",
        "leakage_passed": leakage["passed"],
        "prohibited_sources": {
            "final_holdout_used": False,
            "old_blind_used": False,
            "catalog_used": False,
            "repository_examples_ingested": False,
        },
    }


def generate(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    train_counts = [30] * 40
    dev_counts = [8 if index < 12 else 7 for index in range(24)]
    blind_counts = [10] * 12
    train, train_schemas = generate_split("train", SEEN_DOMAINS, 5, train_counts, seed)
    dev, dev_schemas = generate_split("dev", SEEN_DOMAINS, 3, dev_counts, seed)
    blind, blind_schemas = generate_split("blind", BLIND_DOMAINS, 3, blind_counts, seed)
    splits = {"train": train, "dev": dev, "blind": blind}
    schemas = {**train_schemas, **dev_schemas, **blind_schemas}
    leak = leakage_report(splits, schemas)
    validate_dataset(splits, schemas, leak)
    payloads = {split: _jsonl_bytes(rows) for split, rows in splits.items()}
    manifest = build_manifest(splits, schemas, payloads, leak, seed)
    _write_frozen(output_dir / "train.jsonl", payloads["train"], frozen=False)
    _write_frozen(output_dir / "dev.jsonl", payloads["dev"], frozen=False)
    _write_frozen(output_dir / "blind.jsonl", payloads["blind"], frozen=True)
    _write_frozen(output_dir / "schemas.json", canonical_json(schemas) + b"\n", frozen=False)
    _write_frozen(output_dir / "leakage-report.json", canonical_json(leak) + b"\n", frozen=False)
    _write_frozen(output_dir / "manifest.json", canonical_json(manifest) + b"\n", frozen=False)
    seal = {
        "dataset_version": DATASET_VERSION,
        "split": "blind",
        "sealed_before_baseline": True,
        "examples": len(blind),
        "sha256": sha256_bytes(payloads["blind"]),
        "schema_ids_sha256": sha256_bytes(canonical_json(manifest["BLIND_SCHEMA_IDS"])),
        "modification_policy": "immutable; regenerate under a new dataset version instead of overwriting",
    }
    _write_frozen(output_dir / "blind.seal.json", canonical_json(seal) + b"\n", frozen=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/lora/semantic_extraction_v1"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = generate(args.output_dir, args.seed)
    summary = {
        "dataset_version": manifest["dataset_version"],
        "counts": {split: data["examples"] for split, data in manifest["splits"].items()},
        "hashes": {split: data["sha256"] for split, data in manifest["splits"].items()},
        "leakage_passed": manifest["leakage_passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
