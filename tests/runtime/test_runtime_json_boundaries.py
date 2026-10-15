from __future__ import annotations

import pytest

from kogwistar.runtime.runtime import _json_int, _json_string_list


pytestmark = pytest.mark.ci


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, -1),
        (True, -1),
        ({"step_seq": 4}, -1),
        ("not-an-int", -1),
        ("7", 7),
        (7, 7),
    ],
)
def test_persisted_integer_boundary_rejects_structured_values(
    value: object, expected: int
) -> None:
    assert _json_int(value, default=-1) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ({"errors": ["hidden"]}, []),
        ("not-a-list", []),
        (["first", 2, {"detail": "third"}], ["first", "2", "{'detail': 'third'}"]),
    ],
)
def test_persisted_error_list_boundary_is_total(
    value: object, expected: list[str]
) -> None:
    assert _json_string_list(value) == expected
