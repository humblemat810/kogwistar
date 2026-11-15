"""Strict JSON conversion for Python/Rust runtime contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from kogwistar.json_types import JsonObject, JsonValue


def normalize_json_value(
    value: object,
    *,
    field: str = "value",
    allow_unbounded_integers: bool = False,
) -> JsonValue:
    """Copy supported Python values into the lossless JSON contract domain."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if not allow_unbounded_integers and (
            value < -(1 << 63) or value > (1 << 64) - 1
        ):
            raise ValueError(f"{field} integer is outside the Rust JSON number range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [
            normalize_json_value(
                item,
                field=f"{field}[{index}]",
                allow_unbounded_integers=allow_unbounded_integers,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} contains a non-string object key")
            normalized[key] = normalize_json_value(
                item,
                field=f"{field}.{key}",
                allow_unbounded_integers=allow_unbounded_integers,
            )
        return normalized
    raise TypeError(f"{field} contains a non-JSON value ({type(value).__name__})")


def strict_json_dumps(value: object, *, sort_keys: bool = True) -> str:
    """Encode a value as standards-compliant JSON with no coercive keys."""
    normalized = normalize_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    )
