"""Validate docs/contract/openapi.yaml: YAML syntax + $ref resolution + required sections.

Deliberately dependency-light (PyYAML only) so it runs in CI without openapi-spec-validator.
When openapi-spec-validator IS available, full 3.1 validation is performed too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "contract" / "openapi.yaml"


def resolve(doc: dict, ref: str) -> object | None:
    """Walk a JSON Pointer ref. Returns None when unresolved."""
    if not ref.startswith("#/"):
        return None
    node: object = doc
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def collect_refs(node: object, path: str = "", out: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                out.append((value, path))
            else:
                collect_refs(value, f"{path}/{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            collect_refs(value, f"{path}[{index}]", out)
    return out


def main() -> int:
    if not SPEC.exists():
        print(f"FAIL: spec not found at {SPEC}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    try:
        doc = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"FAIL: YAML syntax error: {exc}")
        return 1

    if not isinstance(doc, dict):
        print("FAIL: spec root is not a mapping")
        return 1

    print("YAML parse: OK")

    for section in ("openapi", "info", "paths", "components"):
        if section not in doc:
            errors.append(f"missing top-level section: {section}")

    if doc.get("openapi") != "3.1.0":
        errors.append(f"openapi version is {doc.get('openapi')!r}, expected '3.1.0'")

    refs = collect_refs(doc)
    unresolved = [(r, p) for r, p in refs if resolve(doc, r) is None]
    for ref, path in unresolved:
        errors.append(f"unresolved $ref {ref!r} at {path}")
    print(f"$ref resolution: {len(refs)} refs, {len(unresolved)} unresolved")

    # Structural sanity that matters for the integration package.
    paths = doc.get("paths", {})
    components = doc.get("components", {})
    schemas = components.get("schemas", {})

    required_paths = ["/v1/recommendations", "/v1/items/{item_id}", "/v1/items:batchGet",
                      "/v1/users/{user_id}/history", "/health", "/ready"]
    for rp in required_paths:
        if rp not in paths:
            errors.append(f"missing required path: {rp}")

    required_schemas = ["Item", "Filters", "RecommendationRequest", "RecommendationResponse",
                        "ScoredItem", "HistoryEvent", "HistoryResponse", "Problem", "Health", "EventType"]
    for rs in required_schemas:
        if rs not in schemas:
            errors.append(f"missing required schema: {rs}")

    unused = {name for name in schemas if not any(name in r for r, _ in refs)}
    if unused:
        warnings.append(f"schemas never referenced: {sorted(unused)}")

    # Every non-200 response should be application/problem+json (RFC 7807 discipline).
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if not isinstance(op, dict) or method not in ("get", "post", "put", "delete", "patch"):
                continue
            for code, resp in (op.get("responses") or {}).items():
                if code.startswith("2") or not isinstance(resp, dict):
                    continue
                resolved = resp
                if "$ref" in resp:
                    target = resolve(doc, resp["$ref"])
                    resolved = target if isinstance(target, dict) else {}
                content = resolved.get("content", {})
                # Readiness probes legitimately return Health on 503 (a probe result, not an error).
                if path in ("/health", "/ready"):
                    continue
                if content and "application/problem+json" not in content:
                    warnings.append(f"{method.upper()} {path} {code}: error response is not application/problem+json")

    print(f"paths: {len(paths)} | schemas: {len(schemas)}")
    for path, item in sorted(paths.items()):
        if isinstance(item, dict):
            for method in item:
                if method in ("get", "post", "put", "delete", "patch"):
                    print(f"  {method.upper():6} {path}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    # Optional: full spec validation when the library happens to be installed.
    try:
        from openapi_spec_validator import validate  # type: ignore

        validate(doc)
        print("\nopenapi-spec-validator: OK (full 3.1 validation passed)")
    except ImportError:
        print("\nopenapi-spec-validator: not installed (skipped full validation)")
    except Exception as exc:
        errors.append(f"openapi-spec-validator: {type(exc).__name__}: {exc}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nCONTRACT VALIDATION: FAIL ({len(errors)} errors)")
        return 1

    print("\nCONTRACT VALIDATION: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
