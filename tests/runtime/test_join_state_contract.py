from __future__ import annotations

import pytest

from kogwistar.runtime.join_state import (
    decode_runtime_join_snapshot,
    restore_runtime_join_snapshot,
)
from kogwistar.runtime.models import WorkflowNode
from kogwistar.runtime.runtime_join_state import RuntimeJoinState

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


def test_runtime_join_state_round_trips_frontier_and_counter_state() -> None:
    join_node = WorkflowNode.model_construct(
        id="join-a",
        metadata={"wf_join": True, "wf_join_is_merge": False},
    )
    state = RuntimeJoinState.for_workflow(
        nodes={
            "start": WorkflowNode.model_construct(id="start", metadata={}),
            "join-a": join_node,
        },
        adjacency={},
    )
    state.inc(state.bit_for_join("join-a"))
    state.join_waiters["join-a"].append((1, "waiter-1", None))
    snapshot = state.snapshot(
        [("next", 1, "pending-1", "parent-1")],
        suspended=[("parked", 0, "suspended-1", None)],
    )
    persisted: dict[str, object] = {}
    state.set_snapshot(persisted, snapshot)

    assert state.join_is_merge == {"join-a": False}
    assert persisted["_rt_join"] == {
        "join_node_ids": ["join-a"],
        "join_outstanding": [1],
        "join_waiters": {"join-a": [(1, "waiter-1", None)]},
        "pending": [("next", 1, "pending-1", "parent-1")],
        "suspended": [("parked", 0, "suspended-1", None)],
    }

    resumed = RuntimeJoinState.for_workflow(
        nodes={
            "start": WorkflowNode.model_construct(id="start", metadata={}),
            "join-a": join_node,
        },
        adjacency={},
    )
    assert resumed.restore(persisted) == [("next", 1, "pending-1", "parent-1")]
    assert resumed.join_outstanding == [1]
    assert resumed.join_waiters == {"join-a": [(1, "waiter-1", None)]}


def test_runtime_join_state_counter_decrement_stays_nonnegative() -> None:
    join_node = WorkflowNode.model_construct(id="join-a", metadata={"wf_join": True})
    state = RuntimeJoinState.for_workflow(nodes={"join-a": join_node}, adjacency={})

    state.dec(state.bit_for_join("join-a"))

    assert state.join_outstanding == [0]


def test_non_merge_join_releases_each_arrival_without_clearing_waiter() -> None:
    node = WorkflowNode.model_construct(
        id="join-a", metadata={"wf_join": True, "wf_join_is_merge": False}
    )
    state = RuntimeJoinState.for_workflow(nodes={"join-a": node}, adjacency={})
    bit = state.bit_for_join("join-a")
    state.inc(bit)

    outcome = state.arrive("join-a", bit, "token-a", "parent-a")

    assert outcome.mask == 0
    assert outcome.released == (0, "token-a", "parent-a")
    assert outcome.outstanding == 0
    assert state.join_waiters["join-a"] == [(0, "token-a", "parent-a")]


def test_merge_join_releases_one_token_after_all_arrivals() -> None:
    node = WorkflowNode.model_construct(
        id="join-a", metadata={"wf_join": True, "wf_join_is_merge": True}
    )
    state = RuntimeJoinState.for_workflow(nodes={"join-a": node}, adjacency={})
    bit = state.bit_for_join("join-a")
    state.inc(bit)
    state.inc(bit)

    first = state.arrive("join-a", bit, "token-a", "parent-a")
    second = state.arrive("join-a", bit, "token-b", "parent-b")

    assert first.released is None
    assert first.outstanding == 1
    assert second.released == (0, "token-a", "parent-a")
    assert second.collapsed_work_items == 1
    assert second.outstanding == 0
    assert state.join_waiters["join-a"] == []
    assert state.join_outstanding == [0]
