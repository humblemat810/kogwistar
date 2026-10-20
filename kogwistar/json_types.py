"""Dependency-free recursive JSON type aliases shared by public contracts."""

from __future__ import annotations

from collections.abc import Mapping

# TypeAliasType is provided by typing_extensions on Python 3.11, including
# the supported PyPy 3.11 runtime. The stdlib copy only arrived in Python 3.12.
from typing_extensions import TypeAliasType

JsonScalar = TypeAliasType("JsonScalar", None | bool | int | float | str)
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]


def decode_json_value(value: object, *, label: str = "value") -> JsonValue:
    """Validate and copy a value into the recursive JSON domain."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [
            decode_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        decoded: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} must contain only string keys")
            decoded[key] = decode_json_value(item, label=f"{label}.{key}")
        return decoded
    raise TypeError(f"{label} contains a non-JSON value")


def decode_json_object(value: object, *, label: str = "value") -> JsonObject:
    """Validate and copy an object into the recursive JSON domain."""

    decoded = decode_json_value(value, label=label)
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be a JSON object")
    return decoded
