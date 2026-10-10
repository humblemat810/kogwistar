from __future__ import annotations

from kogwistar.json_types import JsonObject, JsonValue


def test_recursive_json_aliases_are_importable_on_supported_runtimes() -> None:
    payload: JsonObject = {"nested": [True, {"value": "ok"}]}
    value: JsonValue = payload

    assert value == payload
