"""Extraction-only controlled comparison on the frozen public synthetic DEV facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from recagent.interpretation import (
    ConstraintUpdate,
    InterpretationIssue,
    LLMRequestInterpreter,
    StructuredRequest,
)
from recagent.parsing import OllamaClient, normalize

ROOT = Path(__file__).resolve().parents[1]


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AtomicFact(ExperimentModel):
    source_text: str = Field(min_length=1)
    meaning: str = Field(min_length=1)


class FactInventory(ExperimentModel):
    intent: Literal["discovery", "similar", "mood", "navigation"] | None = None
    reset_constraints: bool = False
    facts: list[AtomicFact] = Field(default_factory=list, max_length=20)


class FactMapping(ExperimentModel):
    fact_index: int = Field(ge=0)
    field: str = Field(min_length=1)
    operation: Literal["set", "clear", "exclude", "include"] = "set"
    value: str | int | bool | None = None
    issue_kind: Literal["ambiguity", "unsupported_constraint", "conflict"] | None = None
    issue_value: str = ""
    issue_message: str = ""


class BatchMapping(ExperimentModel):
    mappings: list[FactMapping] = Field(default_factory=list, max_length=20)


class SingleMapping(ExperimentModel):
    mapping: FactMapping


INVENTORY_SYSTEM = (
    "Разложи только текущее сообщение пользователя на все независимые semantic facts. "
    "Каждый факт должен иметь точную минимальную цитату source_text из сообщения и краткое meaning обычными словами. "
    "Отдельными facts считаются тип искомого объекта, тема/категория, свойства, ограничения, отрицания и название. "
    "Не объединяй несколько условий в один fact и ничего не выводи из previous. Общую команду поиска не добавляй как fact: "
    "она отражается в intent. navigation — только точное название; similar — похожее на название; discovery — подбор по свойствам. "
    "Если пользователь явно меняет предмет запроса, reset_constraints=true. Не сопоставляй facts с customer fields. Верни только JSON."
)

MAPPING_SYSTEM = (
    "Сопоставь каждый переданный atomic fact с customer schema/config. Один fact должен появиться ровно один раз по fact_index. "
    "Используй только schema fields, allowed values, aliases, scalar_aliases, numeric_units и capabilities из domain. "
    "Не анализируй исходное сообщение заново и не объединяй facts. Для boolean false используй operation=set и value=false. "
    "exclude/include применяй только когда domain capability допускает исключение поля. Если факт нельзя безопасно применить, "
    "сохрани raw value и укажи issue_kind; не выбирай ближайший enum. source_text добавит deterministic composer. Верни только JSON."
)

SINGLE_MAPPING_SYSTEM = (
    "Сопоставь ровно один atomic fact с customer schema/config. Используй только domain schema, aliases, scalar_aliases, "
    "numeric_units и capabilities. Не анализируй другие части исходного сообщения. Для boolean false используй set/false; "
    "exclude/include — только при заявленной capability. Непредставимый факт сохрани с raw value и issue_kind, не угадывай enum. "
    "fact_index должен совпасть с переданным. source_text добавит deterministic composer. Верни только JSON."
)


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[union-attr]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def source_revision() -> dict[str, object]:
    paths = [
        ROOT / "data/model_extraction_dev_facts.json",
        ROOT / "src/recagent/interpretation.py",
        ROOT / "src/recagent/parsing.py",
        Path(__file__),
    ]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "commit": commit,
        "dirty": dirty,
        "files_sha256": {path.relative_to(ROOT).as_posix(): sha256_bytes(path.read_bytes()) for path in paths},
    }


def load_inputs(facts_path: Path, context_path: Path) -> tuple[dict, dict[tuple[str, int], dict]]:
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    cases = facts.get("cases", [])
    if len(cases) != facts.get("expected_turns", 25):
        raise ValueError("fact fixture turn count differs from expected_turns")
    if facts.get("source_dataset"):
        expected_sha = facts.get("source_dataset_sha256")
        source_dataset = ROOT / facts["source_dataset"]
        if source_dataset.exists() and sha256_bytes(source_dataset.read_bytes()) != expected_sha:
            raise ValueError("source DEV hash differs from the annotation provenance")
        if not source_dataset.exists() and not facts.get("embedded_contexts"):
            raise FileNotFoundError("source DEV is unavailable and the fact fixture has no embedded contexts")
    contexts: dict[tuple[str, int], dict] = {}
    if facts.get("embedded_contexts"):
        for entry in facts["embedded_contexts"]:
            contexts[(entry["dialogue_id"], entry["turn"])] = {
                "payload": entry["payload"],
                "structured": entry.get("structured"),
            }
    else:
        context = json.loads(context_path.read_text(encoding="utf-8"))
        for dialogue in context.get("dialogues", []):
            for index, turn in enumerate(dialogue.get("turns", [])):
                calls = [
                    call
                    for call in turn.get("structured_calls", [])
                    if call.get("schema") == "StructuredRequest" or "updates" in call.get("validated_output", {})
                ]
                if not calls:
                    raise ValueError(f"missing raw StructuredRequest: {dialogue['id']}:{index}")
                contexts[(dialogue["id"], index)] = {
                    "payload": calls[-1]["payload"],
                    "structured": calls[-1]["validated_output"],
                }
    for case in cases:
        key = (case["dialogue_id"], case["turn"])
        if key not in contexts or contexts[key]["payload"]["message"] != case["user"]:
            raise ValueError(f"fixture/context mismatch: {key}")
    return facts, contexts


def issue_for(mapping: FactMapping, fact: AtomicFact) -> InterpretationIssue | None:
    if mapping.issue_kind is None:
        return None
    value = mapping.issue_value or str(mapping.value if mapping.value is not None else fact.meaning)
    message = mapping.issue_message or f"Требуется уточнить факт: {fact.meaning}."
    return InterpretationIssue(kind=mapping.issue_kind, field=mapping.field, value=value, message=message)


def schema_issue(mapping: FactMapping, fact: AtomicFact, domain: dict | None) -> InterpretationIssue | None:
    if domain is None:
        return None
    properties = domain.get("schema", {}).get("properties", {})
    if mapping.field not in properties or mapping.field == "intent":
        return InterpretationIssue(
            kind="unsupported_constraint",
            field=mapping.field,
            value=str(mapping.value or fact.meaning),
            message=f"Fact mapping targets a field unavailable for updates: {mapping.field}.",
        )
    property_schema = properties[mapping.field]
    options = property_schema.get("anyOf", [property_schema])
    non_null = [option for option in options if option.get("type") != "null"]
    allowed_enum = next((option.get("enum") for option in non_null if option.get("enum")), None)
    allowed_types = {option.get("type") for option in non_null}
    value_type = {str: "string", int: "integer", bool: "boolean"}.get(type(mapping.value)) if mapping.value is not None else "null"
    unsupported = (
        (mapping.operation in {"exclude", "include"} and mapping.field not in domain.get("exclusion_fields", {}))
        or (mapping.operation != "clear" and allowed_enum is not None and mapping.value not in allowed_enum)
        or (mapping.operation != "clear" and value_type not in allowed_types)
    )
    if not unsupported:
        return None
    return InterpretationIssue(
        kind="unsupported_constraint",
        field=mapping.field,
        value=str(mapping.value or fact.meaning),
        message=f"Fact cannot be applied by the supplied customer schema/capabilities: {fact.meaning}.",
    )


def compose(
    inventory: FactInventory, mappings: list[FactMapping], domain: dict | None = None
) -> tuple[StructuredRequest, dict[str, object]]:
    by_index: dict[int, FactMapping] = {}
    composition_issues: list[str] = []
    for mapping in mappings:
        if mapping.fact_index >= len(inventory.facts):
            composition_issues.append(f"out_of_range:{mapping.fact_index}")
            continue
        if mapping.fact_index in by_index:
            composition_issues.append(f"duplicate:{mapping.fact_index}")
            continue
        by_index[mapping.fact_index] = mapping
    updates: list[ConstraintUpdate] = []
    issues: list[InterpretationIssue] = []
    for index, fact in enumerate(inventory.facts):
        mapping = by_index.get(index)
        if mapping is None:
            composition_issues.append(f"unmapped:{index}")
            issues.append(
                InterpretationIssue(
                    kind="ambiguity",
                    field="request",
                    value=fact.meaning,
                    message=f"Не удалось типизировать явно выделенный факт: {fact.meaning}.",
                )
            )
            continue
        explicit_issue = issue_for(mapping, fact)
        contract_issue = schema_issue(mapping, fact, domain)
        issue = explicit_issue or contract_issue
        if contract_issue is None:
            updates.append(
                ConstraintUpdate(
                    field=mapping.field,
                    operation=mapping.operation,
                    value=mapping.value,
                    source_text=fact.source_text,
                )
            )
        if issue is not None:
            issues.append(issue)
    operations = {(update.field, normalize(str(update.value))): update.operation for update in updates}
    for update in updates:
        opposite = "exclude" if update.operation == "set" else "set"
        if operations.get((update.field, normalize(str(update.value)))) == opposite:
            issues.append(
                InterpretationIssue(
                    kind="conflict",
                    field=update.field,
                    value=str(update.value),
                    message=f"Fact batch both selects and excludes {update.field}={update.value}.",
                )
            )
    request = StructuredRequest(
        intent=inventory.intent,
        updates=updates,
        reset_constraints=inventory.reset_constraints,
        issues=issues,
        clarification_required=bool(issues),
    )
    return request, {"inventory_count": len(inventory.facts), "mapped_count": len(by_index), "composition_issues": composition_issues}


def run_current(client: OllamaClient, payload: dict) -> tuple[StructuredRequest, list[dict], dict]:
    started = time.perf_counter()
    result, tokens = LLMRequestInterpreter(client, payload["domain"]).interpret(
        payload["message"],
        payload["previous"],
        pending_question=payload.get("pending_question"),
        unresolved=payload.get("unresolved", []),
    )
    call = {
        "stage": "one_shot",
        "schema": "StructuredRequest",
        "tokens": tokens,
        "usage": jsonable(client.last_usage),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "validated_output": result.model_dump(mode="json"),
    }
    return result, [call], {}


def inventory(client: OllamaClient, payload: dict) -> tuple[FactInventory, dict]:
    started = time.perf_counter()
    result, tokens = client.structured(
        FactInventory,
        INVENTORY_SYSTEM,
        {
            "message": payload["message"],
        },
    )
    validated = FactInventory.model_validate(result.model_dump())
    return validated, {
        "stage": "inventory",
        "schema": "FactInventory",
        "tokens": tokens,
        "usage": jsonable(client.last_usage),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "validated_output": validated.model_dump(mode="json"),
    }


def run_batch(client: OllamaClient, payload: dict) -> tuple[StructuredRequest, list[dict], dict]:
    facts, inventory_call = inventory(client, payload)
    started = time.perf_counter()
    result, tokens = client.structured(
        BatchMapping,
        MAPPING_SYSTEM,
        {
            "facts": [dict(fact_index=index, **fact.model_dump()) for index, fact in enumerate(facts.facts)],
            "domain": payload["domain"],
        },
    )
    batch = BatchMapping.model_validate(result.model_dump())
    mapping_call = {
        "stage": "batch_mapping",
        "schema": "BatchMapping",
        "tokens": tokens,
        "usage": jsonable(client.last_usage),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "validated_output": batch.model_dump(mode="json"),
    }
    structured, diagnostics = compose(facts, batch.mappings, payload["domain"])
    return structured, [inventory_call, mapping_call], diagnostics


def run_per_fact(client: OllamaClient, payload: dict) -> tuple[StructuredRequest, list[dict], dict]:
    facts, inventory_call = inventory(client, payload)
    calls = [inventory_call]
    mappings: list[FactMapping] = []
    for index, fact in enumerate(facts.facts):
        started = time.perf_counter()
        result, tokens = client.structured(
            SingleMapping,
            SINGLE_MAPPING_SYSTEM,
            {
                "fact": {"fact_index": index, **fact.model_dump()},
                "domain": payload["domain"],
            },
        )
        single = SingleMapping.model_validate(result.model_dump())
        mappings.append(single.mapping)
        calls.append(
            {
                "stage": "single_mapping",
                "schema": "SingleMapping",
                "tokens": tokens,
                "usage": jsonable(client.last_usage),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "validated_output": single.model_dump(mode="json"),
            }
        )
    structured, diagnostics = compose(facts, mappings, payload["domain"])
    return structured, calls, diagnostics


def fact_key(fact: dict) -> tuple[str, str, str]:
    value = fact.get("value")
    normalized = normalize(value) if isinstance(value, str) else canonical(value)
    return str(fact.get("field")), str(fact.get("operation", "set")), normalized


def issue_matches(expected: dict, actual: dict) -> bool:
    if expected["kind"] != actual.get("kind") or expected["field"] != actual.get("field"):
        return False
    expected_value = normalize(str(expected.get("value", "")))
    actual_value = normalize(str(actual.get("value", "")))
    return not expected_value or expected_value == actual_value or expected_value in actual_value


def score(case: dict, request: StructuredRequest) -> dict[str, object]:
    expected = list(case["expected_facts"])
    actual = [update.model_dump(mode="json") for update in request.updates]
    remaining_actual = set(range(len(actual)))
    matches: list[dict[str, object]] = []
    missing: list[dict] = []
    for fact in expected:
        index = next((i for i in remaining_actual if fact_key(actual[i]) == fact_key(fact)), None)
        if index is None:
            missing.append(fact)
            continue
        remaining_actual.remove(index)
        source = normalize(str(actual[index].get("source_text", "")))
        user = normalize(case["user"])
        matches.append({"expected": fact, "actual": actual[index], "source_cited": bool(source and source in user)})
    wrong: list[dict[str, object]] = []
    still_missing: list[dict] = []
    for fact in missing:
        index = next((i for i in remaining_actual if actual[i].get("field") == fact["field"]), None)
        if index is None:
            still_missing.append(fact)
            continue
        remaining_actual.remove(index)
        wrong.append({"expected": fact, "actual": actual[index]})
    extras = [actual[i] for i in sorted(remaining_actual)]
    expected_issues = case.get("expected_issues", [])
    actual_issues = [issue.model_dump(mode="json") for issue in request.issues]
    issue_hits = sum(any(issue_matches(item, actual_issue) for actual_issue in actual_issues) for item in expected_issues)
    return {
        "expected_fact_count": len(expected),
        "correct_fact_count": len(matches),
        "correct_cited_count": sum(bool(item["source_cited"]) for item in matches),
        "missing_facts": still_missing,
        "wrong_facts": wrong,
        "invented_facts": extras,
        "intent_correct": request.intent == case["expected_intent"],
        "actual_intent": request.intent,
        "reset_correct": request.reset_constraints == case["expected_reset"],
        "expected_issue_count": len(expected_issues),
        "preserved_issue_count": issue_hits,
        "extra_issue_count": max(0, len(actual_issues) - issue_hits),
        "compound": len(expected) >= 2,
        "compound_complete": (len(expected) >= 2 and len(matches) == len(expected) and all(bool(item["source_cited"]) for item in matches)),
    }


def summarize(rows: list[dict]) -> dict[str, object]:
    scores = [row["score"] for row in rows if not row.get("error")]
    expected = sum(item["expected_fact_count"] for item in scores)
    semantic_matches = sum(item["correct_fact_count"] for item in scores)
    grounded_correct = sum(item["correct_cited_count"] for item in scores)
    compound = [item for item in scores if item["compound"]]
    calls = [call for row in rows for call in row.get("calls", [])]
    taxonomy = Counter(row["primary_category"] for row in rows if row.get("primary_category"))
    return {
        "turns": len(rows),
        "completed_turns": len(scores),
        "expected_facts": expected,
        "correct_facts": grounded_correct,
        "fact_recall": grounded_correct / expected if expected else 0,
        "semantic_tuple_matches": semantic_matches,
        "uncited_semantic_matches": semantic_matches - grounded_correct,
        "omissions": sum(len(item["missing_facts"]) for item in scores),
        "wrong_facts": sum(len(item["wrong_facts"]) for item in scores),
        "invented_facts": sum(len(item["invented_facts"]) for item in scores),
        "intent_correct": sum(bool(item["intent_correct"]) for item in scores),
        "reset_correct": sum(bool(item["reset_correct"]) for item in scores),
        "compound_turns": len(compound),
        "compound_complete": sum(bool(item["compound_complete"]) for item in compound),
        "expected_issues": sum(item["expected_issue_count"] for item in scores),
        "preserved_issues": sum(item["preserved_issue_count"] for item in scores),
        "extra_issues": sum(item["extra_issue_count"] for item in scores),
        "structured_calls": len(calls),
        "tokens": sum(int(call.get("tokens", 0)) for call in calls),
        "elapsed_ms": round(sum(float(call.get("elapsed_ms", 0)) for call in calls), 3),
        "historical_failure_taxonomy": dict(sorted(taxonomy.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=ROOT / "data/model_extraction_dev_facts.json")
    parser.add_argument("--context-report", type=Path, default=ROOT / "artifacts/llm-first-product/after.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", choices=("saved", "current", "batch", "per_fact"), required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    fixture, contexts = load_inputs(args.facts, args.context_report)
    client = None if args.design == "saved" else OllamaClient(args.model, args.base_url, args.timeout)
    runners = {"current": run_current, "batch": run_batch, "per_fact": run_per_fact}
    rows: list[dict[str, object]] = []
    for case in fixture["cases"]:
        context = contexts[(case["dialogue_id"], case["turn"])]
        started = time.perf_counter()
        try:
            if args.design == "saved":
                request = StructuredRequest.model_validate(context["structured"])
                calls: list[dict] = []
                diagnostics = {"source": "saved raw StructuredRequest"}
            else:
                request, calls, diagnostics = runners[args.design](client, context["payload"])  # type: ignore[arg-type]
            row = {
                "dialogue_id": case["dialogue_id"],
                "turn": case["turn"],
                "user": case["user"],
                "primary_category": case.get("primary_category"),
                "representability": case["representability"],
                "expected_intent": case["expected_intent"],
                "expected_facts": case["expected_facts"],
                "expected_issues": case["expected_issues"],
                "actual_structured_request": request.model_dump(mode="json"),
                "score": score(case, request),
                "calls": calls,
                "diagnostics": diagnostics,
            }
        except Exception as exc:
            row = {
                "dialogue_id": case["dialogue_id"],
                "turn": case["turn"],
                "user": case["user"],
                "primary_category": case.get("primary_category"),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        rows.append(row)
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "design": args.design,
        "complete": all("error" not in row for row in rows),
        "fixture": {
            "path": str(args.facts),
            "sha256": sha256_bytes(args.facts.read_bytes()),
            "source_dataset_sha256": fixture.get("source_dataset_sha256"),
            "turns": len(rows),
        },
        "context_report": None
        if fixture.get("embedded_contexts")
        else {"path": str(args.context_report), "sha256": sha256_bytes(args.context_report.read_bytes())},
        "model": None
        if client is None
        else {
            "name": args.model,
            "base_url": args.base_url,
            "structured_options": {"temperature": 0, "seed": 42, "num_predict": 700, "think": False},
        },
        "source": source_revision(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "summary": summarize(rows),
        "cases": rows,
        "limitations": [
            "The fixture is reused public synthetic DEV, not an independent final test.",
            "Saved and live outputs may differ despite a fixed seed; conclusions require class-level, not phrase-level, gains.",
            "No downstream adapter, policy, retrieval, ranking or response code runs in this extraction-only experiment.",
        ],
    }
    result["report_sha256"] = hashlib.sha256(canonical(result).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"complete": result["complete"], "design": args.design, "summary": result["summary"], "output": str(args.output)}))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
