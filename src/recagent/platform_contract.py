"""Проверка запросов и ответов по передаваемому контракту OpenAPI."""

from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker


@lru_cache
def specification():
    path = Path(__file__).resolve().parents[2] / "docs" / "contract" / "openapi.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache
def validator(name):
    schema = {"$ref": f"#/components/schemas/{name}", "components": specification()["components"]}
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_schema(payload, name):
    validator(name).validate(payload)
