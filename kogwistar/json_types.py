"""Dependency-free recursive JSON type aliases shared by public contracts."""

from __future__ import annotations

from typing import TypeAliasType

JsonScalar = TypeAliasType("JsonScalar", None | bool | int | float | str)
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]
