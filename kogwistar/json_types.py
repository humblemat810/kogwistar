"""Dependency-free recursive JSON type aliases shared by public contracts."""

from __future__ import annotations

# TypeAliasType is provided by typing_extensions on Python 3.11, including
# the supported PyPy 3.11 runtime. The stdlib copy only arrived in Python 3.12.
from typing_extensions import TypeAliasType

JsonScalar = TypeAliasType("JsonScalar", None | bool | int | float | str)
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]
