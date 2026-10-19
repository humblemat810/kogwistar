from __future__ import annotations

import pytest

from kogwistar.runtime.join_state import (
    decode_runtime_join_snapshot,
    restore_runtime_join_snapshot,
)

pytestmark = pytest.mark.ci


def _snapshot(outstanding: object) -> dict[str, object]:
    return {
        "join_node_ids": ["join-a", "join-b"],
        "join_outstanding": outstanding,
        "join_waiters": {"join-a": [], "join-b": []},
        "pending": [],
        "suspended": [],
    }


def test_join_state_normalizes_legacy_short_counter_vectors() -> None:
    decoded = decode_runtime_join_snapshot(
        _snapshot([2]),
        ["join-a", "join-b"],
        allow_legacy_lengths=False,
        allow_single_mask_waiters=False,
    )

    assert decoded is not None
    assert decoded["join_outstanding"] == [2, 0]


@pytest.mark.parametrize("value", [[None, 1], [-1, 0], [True, 0], ["not-an-int", 0]])
def test_join_state_rejects_malformed_counter_values(value: object) -> None:
    assert (
        decode_runtime_join_snapshot(
            _snapshot(value),
            ["join-a", "join-b"],
            allow_legacy_lengths=False,
            allow_single_mask_waiters=False,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("join_waiters", {"join-a": [[1.5, "waiter", None]], "join-b": []}),
        ("join_waiters", {"join-a": [[1, object(), None]], "join-b": []}),
        ("pending", [["node", 1.5, "token", None]]),
        ("pending", [["node", -1, "token", None]]),
        ("suspended", [["node", 1, "token", object()]]),
    ],
)
def test_join_state_rejects_malformed_entries_without_dropping_them(
    field: str, value: object
) -> None:
    snapshot = _snapshot([0, 0])
    snapshot[field] = value

    assert (
        decode_runtime_join_snapshot(
            snapshot,
            ["join-a", "join-b"],
            allow_legacy_lengths=False,
            allow_single_mask_waiters=False,
        )
        is None
    )


def test_join_restore_treats_absent_state_as_fresh_run() -> None:
    assert (
        restore_runtime_join_snapshot(
            {},
            ["join-a", "join-b"],
            allow_legacy_lengths=False,
            allow_single_mask_waiters=False,
        )
        is None
    )


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {**_snapshot([0, 0]), "join_node_ids": ["old-join"]},
        {**_snapshot([0, 0]), "join_waiters": {"join-a": [[1.5, "bad", None]]}},
    ],
)
def test_join_restore_rejects_present_invalid_or_incompatible_state(
    snapshot: object,
) -> None:
    with pytest.raises(ValueError, match="malformed or incompatible"):
        restore_runtime_join_snapshot(
            {"_rt_join": snapshot},
            ["join-a", "join-b"],
            allow_legacy_lengths=False,
            allow_single_mask_waiters=False,
        )
