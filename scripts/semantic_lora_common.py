"""Shared, production-contract helpers for the semantic LoRA research pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from recagent.interpretation import LLMRequestInterpreter, StructuredRequest

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LoRA config must be a mapping")
    for split in ("train", "dev", "blind"):
        declared = config["dataset"][split]
        actual = file_sha256(resolve(declared["path"]))
        if actual != declared["sha256"]:
            raise ValueError(f"{split} dataset hash mismatch: {actual}")
    return config


def protocol_hashes(config: dict[str, Any], evaluator_path: Path) -> dict[str, str]:
    """Return every mutable protocol component that must be frozen before baseline."""
    return {
        "evaluator_sha256": file_sha256(evaluator_path),
        "scorer_transforms_normalization_sha256": file_sha256(Path(__file__)),
        "generator_sha256": file_sha256(resolve(config["dataset"]["generator"])),
        "generator_dependency_sha256": file_sha256(Path(__file__).with_name("generate_semantic_lora_dataset.py")),
        "manifest_sha256": file_sha256(resolve(config["dataset"]["manifest"])),
        "schemas_sha256": file_sha256(resolve(config["dataset"]["schemas"])),
        "leakage_report_sha256": file_sha256(resolve(config["dataset"]["leakage_report"])),
        "acceptance_sha256": sha256_bytes(canonical(config["acceptance"]).encode()),
        "production_interpretation_sha256": file_sha256(ROOT / "src/recagent/interpretation.py"),
        "structured_request_schema_sha256": sha256_bytes(canonical(StructuredRequest.model_json_schema()).encode()),
    }


def training_execution_sha256(config: dict[str, Any]) -> str:
    """Hash the model/data/training inputs while excluding the later protocol seal fields."""
    payload = {key: config[key] for key in ("base_model", "dataset", "seeds", "training", "tiny_overfit", "environment")}
    return sha256_bytes(canonical(payload).encode())


def verify_frozen_protocol(config_path: Path, config: dict[str, Any], evaluator_path: Path) -> dict[str, Any]:
    """Refuse baseline/training/evaluation after any frozen protocol component drifts."""
    declared = config["protocol_integrity"]
    if declared.get("status") != "FROZEN_BEFORE_BASELINE":
        raise ValueError("protocol must be FROZEN_BEFORE_BASELINE")
    actual = protocol_hashes(config, evaluator_path)
    mismatches = {key: {"declared": declared.get(key), "actual": value} for key, value in actual.items() if declared.get(key) != value}
    if mismatches:
        raise ValueError(f"frozen protocol drift: {canonical(mismatches)}")
    seal_path = resolve(declared["seal"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    integrity = seal.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(seal).encode()):
        raise ValueError("protocol seal integrity mismatch")
    seal["report_sha256"] = integrity
    if seal["config_sha256"] != file_sha256(config_path):
        raise ValueError("protocol seal config hash mismatch")
    if seal["hashes"] != actual:
        raise ValueError("protocol seal component hash mismatch")
    return seal


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        StructuredRequest.model_validate(item["target"])
        rows.append(item)
    return rows


def latent_isolation_summary(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Audit split-independent template IDs and field-name shapes after the v1 freeze."""
    template_families = {split: {row["template_family_id"].rsplit("-", 1)[-1] for row in rows} for split, rows in splits.items()}
    field_sets: dict[str, set[str]] = {}
    for split, rows in splits.items():
        unique: dict[str, tuple[str, ...]] = {}
        for row in rows:
            unique.setdefault(
                row["schema_id"],
                tuple(sorted(row["input"]["domain"]["schema"]["properties"])),
            )
        field_sets[split] = {canonical(value) for value in unique.values()}
    pairs = (("train", "dev"), ("train", "blind"), ("dev", "blind"))
    template_overlap = {f"{left}__{right}": sorted(template_families[left] & template_families[right]) for left, right in pairs}
    field_set_overlap = {f"{left}__{right}": len(field_sets[left] & field_sets[right]) for left, right in pairs}
    passed = not any(template_overlap.values()) and not any(field_set_overlap.values())
    return {
        "passed": passed,
        "split_independent_template_family_overlap": template_overlap,
        "field_name_set_overlap_counts": field_set_overlap,
        "interpretation": ("v1 split-prefixed template_family_id and enum-sensitive schema fingerprint hid these overlaps"),
    }


class CaptureBackend:
    def __init__(self, target: dict[str, Any]):
        self.target = target
        self.system = ""
        self.payload: dict[str, Any] = {}

    def structured(self, schema, system: str, payload: dict) -> tuple[StructuredRequest, int]:
        self.system = system
        self.payload = payload
        return schema.model_validate(self.target), 0


def production_exchange(row: dict[str, Any]) -> tuple[str, str]:
    incoming = row["input"]
    backend = CaptureBackend(row["target"])
    LLMRequestInterpreter(backend, incoming["domain"]).interpret(
        incoming["message"],
        incoming.get("previous", {}),
        pending_question=incoming.get("pending_question"),
        unresolved=incoming.get("unresolved", []),
        pending_context=incoming.get("pending_context"),
    )
    schema_json = StructuredRequest.model_json_schema()
    system = backend.system + "\nJSON schema: " + json.dumps(schema_json, ensure_ascii=False)
    user = json.dumps(backend.payload, ensure_ascii=False)
    return system, user


def training_texts(row: dict[str, Any], tokenizer, prompt_mode: str = "production") -> tuple[str, str]:
    if prompt_mode != "production":
        raise ValueError(f"unknown training prompt mode: {prompt_mode}")
    system, user = production_exchange(row)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target = canonical(StructuredRequest.model_validate(row["target"]).model_dump(mode="json"))
    full = prompt + target + tokenizer.eos_token
    return prompt, full


def tokenize_training_row(row: dict[str, Any], tokenizer, max_length: int, prompt_mode: str = "production") -> dict[str, Any]:
    """Tokenize one exact production exchange and mask all non-assistant tokens."""
    prompt, full = training_texts(row, tokenizer, prompt_mode)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"token boundary drift for {row['id']}")
    if len(full_ids) > max_length:
        raise ValueError(f"overlength example {row['id']}: {len(full_ids)} > {max_length}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not labels or all(label == -100 for label in labels):
        raise ValueError(f"empty assistant loss for {row['id']}")
    return {
        "id": row["id"],
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "prompt_length": len(prompt_ids),
        "target_length": len(full_ids) - len(prompt_ids),
    }


def stable_key(seed: int, value: str) -> str:
    return sha256_bytes(f"{seed}:{value}".encode())


def seen_schema_holdout(rows: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_schema: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_schema[row["schema_id"]].append(row)
    training, holdout = [], []
    for schema_rows in by_schema.values():
        ordered = sorted(schema_rows, key=lambda row: stable_key(seed, row["id"]))
        held = max(1, math.floor(len(ordered) * fraction))
        holdout.extend(ordered[:held])
        training.extend(ordered[held:])
    return (
        sorted(training, key=lambda row: stable_key(seed + 1, row["id"])),
        sorted(holdout, key=lambda row: stable_key(seed + 2, row["id"])),
    )


def select_cases(rows: list[dict[str, Any]], count: int, seed: int, label: str) -> list[dict]:
    return sorted(rows, key=lambda row: stable_key(seed, f"{label}:{row['id']}"))[:count]


def _replace_exact(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {mapping.get(str(key), str(key)): _replace_exact(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_exact(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def rename_fields(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    properties = result["input"]["domain"]["schema"]["properties"]
    mapping = {field: f"f_{index:02d}" for index, field in enumerate(sorted(properties))}
    result["input"]["domain"] = _replace_exact(result["input"]["domain"], mapping)
    result["target"] = _replace_exact(result["target"], mapping)
    result["id"] += "--renamed"
    return result


def randomize_enums(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    properties = result["input"]["domain"]["schema"]["properties"]
    mapping: dict[str, str] = {}
    for field_index, field in enumerate(sorted(properties)):
        for value_index, value in enumerate(properties[field].get("enum", [])):
            mapping[value] = f"enum_{field_index:02d}_{value_index:02d}"
    result["input"]["domain"] = _replace_exact(result["input"]["domain"], mapping)
    result["target"] = _replace_exact(result["target"], mapping)
    result["id"] += "--enum-randomized"
    return result


def remove_schema(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    domain = result["input"]["domain"]
    domain["schema"] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    domain["aliases"] = {}
    domain["scalar_aliases"] = {}
    domain["numeric_units"] = {}
    domain["capabilities"] = {
        "allowed_operations": [],
        "unknown_value_policy": "preserve_as_issue_without_guessing",
    }
    result["id"] += "--without-schema"
    return result


def normalize(value: object) -> str:
    return " ".join(str(value).casefold().replace("ё", "е").strip(" \t\r\n«»\"'").split())


def values_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    return normalize(left) == normalize(right) if isinstance(left, str) else left == right


def source_matches(actual: str, expected: str, message: str) -> bool:
    actual_norm, expected_norm, message_norm = map(normalize, (actual, expected, message))
    if not actual_norm or not expected_norm:
        return False
    cited = re.search(r"(?<!\w)" + re.escape(actual_norm) + r"(?!\w)", message_norm) is not None
    return cited and (actual_norm in expected_norm or expected_norm in actual_norm)


def semantic_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return (
        expected["field"] == actual.get("field")
        and expected["operation"] == actual.get("operation")
        and values_equal(expected.get("value"), actual.get("value"))
    )


def exact_matching(expected: list[dict], actual: list[dict], message: str) -> list[tuple[int, int]]:
    @cache
    def solve(expected_index: int, used_actual: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        if expected_index == len(expected):
            return 0, 0, ()
        best = solve(expected_index + 1, used_actual)
        for actual_index, item in enumerate(actual):
            if used_actual & (1 << actual_index) or not semantic_matches(expected[expected_index], item):
                continue
            count, grounded, pairs = solve(expected_index + 1, used_actual | (1 << actual_index))
            candidate = (
                count + 1,
                grounded + int(source_matches(item["source_text"], expected[expected_index]["source_text"], message)),
                ((expected_index, actual_index), *pairs),
            )
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    return list(solve(0, 0)[2])


def issue_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return (
        expected["kind"] == actual.get("kind")
        and expected["field"] == actual.get("field")
        and (not expected.get("value") or normalize(expected["value"]) == normalize(actual.get("value", "")))
    )


def score_request(row: dict[str, Any], request: StructuredRequest) -> dict[str, Any]:
    expected = row["target"]["updates"]
    actual = [item.model_dump(mode="json") for item in request.updates]
    message = row["input"]["message"]
    pairs = exact_matching(expected, actual, message)
    by_expected = dict(pairs)
    remaining = set(range(len(actual))) - {actual_index for _, actual_index in pairs}
    grounded = 0
    wrong_field = wrong_value = wrong_polarity = wrong_kind = 0
    omitted = 0
    for expected_index, fact in enumerate(expected):
        if expected_index in by_expected:
            item = actual[by_expected[expected_index]]
            grounded += int(source_matches(item["source_text"], fact["source_text"], message))
            continue
        related = next(
            (
                index
                for index in remaining
                if source_matches(actual[index]["source_text"], fact["source_text"], message) or actual[index].get("field") == fact["field"]
            ),
            None,
        )
        if related is None:
            omitted += 1
            continue
        remaining.remove(related)
        item = actual[related]
        if item.get("field") != fact["field"]:
            wrong_field += 1
            wrong_kind += int("kind" in row.get("tags", []))
        elif item.get("operation") != fact["operation"]:
            wrong_polarity += 1
        elif not values_equal(item.get("value"), fact.get("value")):
            wrong_value += 1
    expected_issues = row["target"].get("issues", [])
    actual_issues = [item.model_dump(mode="json") for item in request.issues]
    issue_hits = Counter()
    issue_totals = Counter(issue["kind"] for issue in expected_issues)
    for issue in expected_issues:
        if any(issue_matches(issue, item) for item in actual_issues):
            issue_hits[issue["kind"]] += 1
    semantic = len(pairs)
    invented = len(remaining)
    precision = semantic / len(actual) if actual else float(not expected)
    recall = semantic / len(expected) if expected else float(not actual)
    return {
        "expected_facts": len(expected),
        "actual_facts": len(actual),
        "semantic_matches": semantic,
        "grounded_correct": grounded,
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "omissions": omitted,
        "invented": invented,
        "wrong_field": wrong_field,
        "wrong_value": wrong_value,
        "wrong_kind_proxy": wrong_kind,
        "wrong_polarity": wrong_polarity,
        "expected_issues": dict(issue_totals),
        "preserved_issues": dict(issue_hits),
        "compound": len(expected) >= 2,
        "compound_complete": len(expected) >= 2 and grounded == len(expected),
        "intent_correct": request.intent == row["target"].get("intent"),
        "exact_request": request.model_dump(mode="json") == row["target"],
    }


def summarize(scored_rows: list[dict[str, Any]], total_rows: int) -> dict[str, Any]:
    scores = [row["score"] for row in scored_rows if "score" in row]
    expected = sum(score["expected_facts"] for score in scores)
    actual = sum(score["actual_facts"] for score in scores)
    semantic = sum(score["semantic_matches"] for score in scores)
    grounded = sum(score["grounded_correct"] for score in scores)
    compound = [score for score in scores if score["compound"]]
    expected_issues, preserved_issues = Counter(), Counter()
    for score in scores:
        expected_issues.update(score["expected_issues"])
        preserved_issues.update(score["preserved_issues"])
    precision = semantic / actual if actual else float(expected == 0)
    recall = semantic / expected if expected else float(actual == 0)
    return {
        "turns": total_rows,
        "valid_structured_outputs": len(scores),
        "valid_structured_output_rate": len(scores) / total_rows if total_rows else 0.0,
        "expected_facts": expected,
        "actual_facts": actual,
        "semantic_matches": semantic,
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "grounded_correct": grounded,
        "grounded_correct_rate": grounded / expected if expected else float(actual == 0),
        "omissions": sum(score["omissions"] for score in scores),
        "omission_rate": sum(score["omissions"] for score in scores) / expected if expected else 0.0,
        "invented": sum(score["invented"] for score in scores),
        "invented_fact_rate": sum(score["invented"] for score in scores) / actual if actual else 0.0,
        "wrong_field": sum(score["wrong_field"] for score in scores),
        "wrong_value": sum(score["wrong_value"] for score in scores),
        "wrong_kind_proxy": sum(score["wrong_kind_proxy"] for score in scores),
        "wrong_polarity": sum(score["wrong_polarity"] for score in scores),
        "compound_turns": len(compound),
        "compound_complete": sum(score["compound_complete"] for score in compound),
        "compound_complete_rate": (sum(score["compound_complete"] for score in compound) / len(compound) if compound else 0.0),
        "expected_issues": dict(sorted(expected_issues.items())),
        "preserved_issues": dict(sorted(preserved_issues.items())),
        "issue_preservation": {kind: preserved_issues[kind] / count if count else 0.0 for kind, count in sorted(expected_issues.items())},
        "intent_correct_rate": sum(score["intent_correct"] for score in scores) / len(scores) if scores else 0.0,
        "exact_request_rate": sum(score["exact_request"] for score in scores) / len(scores) if scores else 0.0,
    }


def seeded() -> None:
    random.seed(41004)
