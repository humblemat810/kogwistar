from __future__ import annotations

import pytest

from kogwistar.runtime.join_state import decode_runtime_join_snapshot

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
