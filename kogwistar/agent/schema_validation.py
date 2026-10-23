"""Host-side JSON Schema validation for structured model and tool payloads."""

from __future__ import annotations

from jsonschema import FormatChecker
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from kogwistar.json_types import JsonObject, JsonValue


class SchemaValidationError(ValueError):
    """A schema is invalid or a JSON value does not satisfy it."""


def _validator(schema: JsonObject) -> Validator:
    try:
        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
    except Exception as exc:
        raise SchemaValidationError("JSON Schema is invalid or unsupported") from exc
    return validator_type(schema, format_checker=FormatChecker())


def validate_json_schema(value: JsonValue, schema: JsonObject) -> None:
    """Validate a JSON value using its host-supplied JSON Schema."""

    validator = _validator(schema)
    try:
        if next(validator.iter_errors(value), None) is not None:
            raise SchemaValidationError("JSON value does not conform to its schema")
    except SchemaValidationError:
        raise
    except Exception as exc:
        raise SchemaValidationError("JSON Schema validation failed") from exc


def validate_json_object_schema(schema: JsonObject) -> None:
    """Validate a host schema used for an object-valued tool argument."""

    _validator(schema)
    if schema.get("type") != "object":
        raise SchemaValidationError("tool argument schema must declare an object")


__all__ = ["SchemaValidationError", "validate_json_object_schema", "validate_json_schema"]
