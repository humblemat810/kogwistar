"""Mutable join bookkeeping for one synchronous workflow execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kogwistar.runtime.join_state import (
    JoinToken,
    JoinWaiter,
    RuntimeJoinSnapshot,
    restore_runtime_join_snapshot,
)
from kogwistar.runtime.join_topology import (
    _compute_may_reach_join_bitsets,
    _iter_bits,
)
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode, WorkflowState

ReleasedJoinToken = tuple[int, str, str | None]


@dataclass(frozen=True)
class JoinArrivalOutcome:
    """State transition produced when one token reaches a join node."""

    mask: int
    released: ReleasedJoinToken | None
    collapsed_work_items: int
    outstanding: int


@dataclass
class RuntimeJoinState:
    """Per-run join topology and mutable token accounting.

    This object owns only the join counters and their durable snapshot shape;
    token scheduling remains in :class:`WorkflowRuntime`.
    """

    join_node_ids: list[str]
    join_is_merge: dict[str, bool]
    may_reach_join: dict[str, int]
    join_pos: dict[str, int]
    join_outstanding: list[int]
    join_waiters: dict[str, list[JoinWaiter]]

    @classmethod
    def for_workflow(
        cls,
        *,
        nodes: Mapping[str, WorkflowNode],
        adjacency: dict[str, list[WorkflowEdge]],
    ) -> RuntimeJoinState:
        node_ids = list(nodes)
        join_node_ids: list[str] = []
        join_is_merge: dict[str, bool] = {}
        for node_id, node in nodes.items():
            metadata = node.metadata
            if bool(metadata.get("wf_join", False)) or node.op == "join":
                stable_id = str(node_id)
                join_node_ids.append(stable_id)
                join_is_merge[stable_id] = bool(
                    metadata.get("wf_join_is_merge")
                    if metadata.get("wf_join_is_merge") is not None
                    else True
                )

        may_reach_join = (
            _compute_may_reach_join_bitsets(
                node_ids=node_ids,
                adj=adjacency,
                join_ids=join_node_ids,
            )
            if join_node_ids
            else {node_id: 0 for node_id in node_ids}
        )
        return cls(
            join_node_ids=join_node_ids,
            join_is_merge=join_is_merge,
            may_reach_join=may_reach_join,
            join_pos={join_id: index for index, join_id in enumerate(join_node_ids)},
            join_outstanding=[0 for _ in join_node_ids],
            join_waiters={join_id: [] for join_id in join_node_ids},
        )

    def inc(self, mask: int) -> None:
        for bit in _iter_bits(mask):
            self.join_outstanding[bit] += 1

    def dec(self, mask: int) -> None:
        for bit in _iter_bits(mask):
            self.join_outstanding[bit] -= 1
            if self.join_outstanding[bit] < 0:
                self.join_outstanding[bit] = 0

    def mask_without_join(self, mask: int, join_id: str) -> int:
        bit = self.join_pos.get(join_id)
        return mask if bit is None else mask & ~(1 << bit)

    def bit_for_join(self, join_id: str) -> int:
        bit = self.join_pos.get(join_id)
        return (1 << bit) if bit is not None else 0

    def arrive(
        self,
        join_id: str,
        mask: int,
        token_id: str,
        parent_token_id: str | None,
    ) -> JoinArrivalOutcome:
        """Apply the Python join-arrival transition used by the scheduler."""
        join_bit = self.bit_for_join(join_id)
        if mask & join_bit:
            self.dec(join_bit)
            mask = self.mask_without_join(mask, join_id)
        self.join_waiters[join_id].append((mask, token_id, parent_token_id))

        join_index = self.join_pos.get(join_id)
        outstanding = (
            self.join_outstanding[join_index] if join_index is not None else 0
        )
        if not self.join_is_merge[join_id]:
            return JoinArrivalOutcome(
                mask=mask,
                released=(mask, token_id, parent_token_id),
                collapsed_work_items=0,
                outstanding=outstanding,
            )

        released: ReleasedJoinToken | None = None
        collapsed_work_items = 0
        if outstanding == 0:
            waiters = list(self.join_waiters.get(join_id, []))
            if waiters:
                merged_mask = int(waiters[0][0])
                for waiter_mask, _waiter_token, _waiter_parent in waiters:
                    self.dec(int(waiter_mask))
                for waiter_mask, _waiter_token, _waiter_parent in waiters[1:]:
                    merged_mask &= int(waiter_mask)
                self.inc(merged_mask)
                released = (merged_mask, waiters[0][1], waiters[0][2])
                collapsed_work_items = max(0, len(waiters) - 1)
                self.join_waiters[join_id] = []

        return JoinArrivalOutcome(
            mask=mask,
            released=released,
            collapsed_work_items=collapsed_work_items,
            outstanding=outstanding,
        )

    def set_snapshot(self, state: WorkflowState, payload: RuntimeJoinSnapshot) -> None:
        state["_rt_join"] = payload

    def snapshot(
        self,
        pending: list[JoinToken],
        *,
        suspended: list[JoinToken] | None = None,
    ) -> RuntimeJoinSnapshot:
        """Serialize counters and frontier without changing scheduling state."""
        return RuntimeJoinSnapshot(
            join_node_ids=self.join_node_ids,
            join_outstanding=list(self.join_outstanding),
            join_waiters={
                join_id: [
                    (
                        int(mask),
                        str(token_id),
                        str(parent_token_id) if parent_token_id is not None else None,
                    )
                    for mask, token_id, parent_token_id in waiters
                ]
                for join_id, waiters in self.join_waiters.items()
            },
            pending=[
                (
                    node_id,
                    int(mask),
                    str(token_id),
                    str(parent_token_id) if parent_token_id is not None else None,
                )
                for node_id, mask, token_id, parent_token_id in pending
            ],
            suspended=[
                (
                    node_id,
                    int(mask),
                    str(token_id),
                    str(parent_token_id) if parent_token_id is not None else None,
                )
                for node_id, mask, token_id, parent_token_id in (suspended or [])
            ],
        )

    def restore(self, state: Mapping[str, object]) -> list[JoinToken] | None:
        """Restore mutable counters, rejecting malformed persisted snapshots."""
        payload = restore_runtime_join_snapshot(
            state,
            self.join_node_ids,
            allow_legacy_lengths=True,
            allow_single_mask_waiters=True,
        )
        if payload is None:
            return None
        for index, outstanding in enumerate(payload["join_outstanding"]):
            if index >= len(self.join_outstanding):
                break
            self.join_outstanding[index] = outstanding
        for join_id in self.join_node_ids:
            self.join_waiters[join_id] = payload["join_waiters"][join_id]
        # Suspended tokens are restored by resume_run, not rescheduled here.
        return payload["pending"]
