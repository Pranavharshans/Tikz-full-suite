"""Minimal JSON-Schema validator for Stage 1 artifacts.

This intentionally supports only the subset of JSON Schema that the shipped
schemas use (``schemas/*.json``). It exists so that unit tests can assert that
every machine-readable artifact we write stays compatible with its documented
schema, without adding a third-party dependency to the CPU test environment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import SchemaError

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schemas"

_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def load_schema(name: str) -> dict:
    path = SCHEMA_ROOT / name
    if not path.is_file():
        raise SchemaError(f"Unknown schema {name!r}; looked at {path}")
    return json.loads(path.read_text())


def validate_artifact(value, schema: dict, name: str) -> None:
    """Raise ``SchemaError`` when ``value`` violates ``schema``."""
    violations = _validate(value, schema, "$")
    if violations:
        detail = "\n".join(f"  - {line}" for line in violations[:20])
        more = "" if len(violations) <= 20 else f"\n  ... and {len(violations) - 20} more"
        raise SchemaError(f"{name} does not match its schema:\n{detail}{more}")


def _validate(value, schema: dict, path: str) -> list:
    problems = []

    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS[option](value) for option in options):
            problems.append(f"{path}: expected type {expected!r}, got {type(value).__name__}")
            return problems

    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: expected constant {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']!r}")

    if isinstance(value, str):
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            problems.append(f"{path}: {value!r} does not match pattern {pattern!r}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than minLength {schema['minLength']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: more than maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                problems.extend(_validate(item, item_schema, f"{path}[{index}]"))

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                problems.extend(_validate(item, properties[key], f"{path}.{key}"))
            elif additional is False:
                problems.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                problems.extend(_validate(item, additional, f"{path}.{key}"))
    return problems
