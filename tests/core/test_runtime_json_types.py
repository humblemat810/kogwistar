from __future__ import annotations

import pytest

from kogwistar.runtime.serialize import JsonValue, to_jsonable


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def test_recursive_json_value_accepts_nested_structures() -> None:
    value: JsonValue = {
        "items": [
            {"name": "root", "children": [{"name": "leaf"}]},
            {"enabled": True, "value": None},
        ]
    }

    assert to_jsonable(value) == value


def test_conversion_boundary_keeps_opaque_values_out_of_recursive_json() -> None:
    converted = to_jsonable({"items": [object()]})

    assert isinstance(converted, dict)
    assert isinstance(converted["items"], list)
    assert converted["items"][0]["_ref_type"] == "repr_sha256"
