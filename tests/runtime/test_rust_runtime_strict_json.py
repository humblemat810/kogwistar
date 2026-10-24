from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pytest

from kogwistar.runtime.strict_json import normalize_json_value, strict_json_dumps

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class _IntegerCase(TypedDict):
    name: str
    literal: str


class _NonStringKeyCase(TypedDict):
    key: int
    value: str


class _StrictJsonFixture(TypedDict):
    valid_integer_literals: list[_IntegerCase]
    invalid_integer_literals: list[_IntegerCase]
    non_finite_values: list[str]
    non_string_object_key: _NonStringKeyCase


def _fixture() -> _StrictJsonFixture:
    path = Path(__file__).resolve().parents[2] / "contracts/golden/strict-json-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_strict_json_integer_contract_matches_shared_rust_fixture() -> None:
    fixture = _fixture()
    for case in fixture["valid_integer_literals"]:
        literal = case["literal"]
        value = int(literal)
        assert normalize_json_value(value) == value
        assert strict_json_dumps(value) == literal

    for case in fixture["invalid_integer_literals"]:
        with pytest.raises(ValueError, match="outside the Rust JSON number range"):
            normalize_json_value(int(case["literal"]))


def test_strict_json_rejects_non_finite_numbers_and_non_string_keys() -> None:
    fixture = _fixture()
    for value in fixture["non_finite_values"]:
        parsed = float(value.lower())
        with pytest.raises(ValueError, match="non-finite"):
            normalize_json_value({"value": parsed})

    invalid_key = fixture["non_string_object_key"]
    with pytest.raises(TypeError, match="non-string object key"):
        normalize_json_value({"nested": {invalid_key["key"]: invalid_key["value"]}})
