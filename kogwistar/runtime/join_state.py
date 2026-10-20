"""Typed persisted join-state contracts shared by workflow runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias
from typing_extensions import TypedDict
import uuid

JoinWaiter: TypeAlias = tuple[int, str, str | None]
JoinToken: TypeAlias = tuple[str, int, str, str | None]


class RuntimeJoinSnapshot(TypedDict):
    """Persisted join state used to resume a workflow frontier."""

    join_node_ids: list[str]
    join_outstanding: list[int]
    join_waiters: dict[str, list[JoinWaiter]]
    pending: list[JoinToken]
    suspended: list[JoinToken]


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_join_outstanding(
    value: object, *, expected_count: int
) -> list[int] | None:
    """Decode join counters without turning malformed values into zero."""
    if not isinstance(value, list):
        return None
    counters: list[int] = []
    for item in value:
        counter = _coerce_int(item)
        if counter is None or counter < 0:
            return None
        counters.append(counter)
    if len(counters) < expected_count:
        counters.extend(0 for _ in range(expected_count - len(counters)))
    return counters[:expected_count]


def normalize_join_waiter(
    value: object, *, allow_single_mask: bool
) -> JoinWaiter | None:
    """Decode one persisted waiter without trusting an external payload shape."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 3:
        mask = _coerce_int(value[0])
        if (
            mask is None
            or mask < 0
            or not isinstance(value[1], str)
            or (value[2] is not None and not isinstance(value[2], str))
        ):
            return None
        return mask, value[1], value[2]
    if allow_single_mask and len(value) == 1:
        mask = _coerce_int(value[0])
        return (mask, uuid.uuid4().hex, None) if mask is not None and mask >= 0 else None
    return None


def normalize_join_token(
    value: object, *, allow_legacy_lengths: bool
) -> JoinToken | None:
    """Decode a persisted token, retaining only a complete known shape."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4:
        mask = _coerce_int(value[1])
        if (
            mask is None
            or mask < 0
            or not isinstance(value[0], str)
            or not isinstance(value[2], str)
            or (value[3] is not None and not isinstance(value[3], str))
        ):
            return None
        return (
            value[0],
            mask,
            value[2],
            value[3],
        )
    if not allow_legacy_lengths:
        return None
    if len(value) == 3:
        mask = _coerce_int(value[1])
        return (
            (value[0], mask, value[2], None)
            if mask is not None
            and mask >= 0
            and isinstance(value[0], str)
            and isinstance(value[2], str)
            else None
        )
    if len(value) == 2:
        mask = _coerce_int(value[1])
        return (
            (value[0], mask, uuid.uuid4().hex, None)
            if mask is not None and mask >= 0 and isinstance(value[0], str)
            else None
        )
    return None


def decode_runtime_join_snapshot(
    value: object,
    expected_join_node_ids: Sequence[str],
    *,
    allow_legacy_lengths: bool,
    allow_single_mask_waiters: bool,
) -> RuntimeJoinSnapshot | None:
    """Validate and normalize a persisted join snapshot at one typed boundary."""
    if not isinstance(value, Mapping):
        return None
    if value.get("join_node_ids") != list(expected_join_node_ids):
        return None
    outstanding_value = value.get("join_outstanding")
    waiters_value = value.get("join_waiters")
    pending_value = value.get("pending")
    suspended_value = value.get("suspended", [])
    if (
        not isinstance(outstanding_value, list)
        or not isinstance(waiters_value, Mapping)
        or not isinstance(pending_value, list)
        or not isinstance(suspended_value, list)
    ):
        return None
    outstanding = _normalize_join_outstanding(
        outstanding_value, expected_count=len(expected_join_node_ids)
    )
    if outstanding is None:
        return None
    waiters: dict[str, list[JoinWaiter]] = {}
    for join_id in expected_join_node_ids:
        items_value = waiters_value.get(join_id, [])
        if not isinstance(items_value, list):
            return None
        decoded_waiters: list[JoinWaiter] = []
        for item in items_value:
            waiter = normalize_join_waiter(
                item, allow_single_mask=allow_single_mask_waiters
            )
            if waiter is None:
                return None
            decoded_waiters.append(waiter)
        waiters[join_id] = decoded_waiters
    pending: list[JoinToken] = []
    for item in pending_value:
        token = normalize_join_token(
            item, allow_legacy_lengths=allow_legacy_lengths
        )
        if token is None:
            return None
        pending.append(token)
    suspended: list[JoinToken] = []
    for item in suspended_value:
        token = normalize_join_token(
            item, allow_legacy_lengths=allow_legacy_lengths
        )
        if token is None:
            return None
        suspended.append(token)
    return RuntimeJoinSnapshot(
        join_node_ids=list(expected_join_node_ids),
        join_outstanding=outstanding,
        join_waiters=waiters,
        pending=pending,
        suspended=suspended,
    )


def restore_runtime_join_snapshot(
    state: Mapping[str, object],
    expected_join_node_ids: Sequence[str],
    *,
    allow_legacy_lengths: bool,
    allow_single_mask_waiters: bool,
) -> RuntimeJoinSnapshot | None:
    """Return absent state as ``None`` and reject present invalid state."""
    if "_rt_join" not in state:
        return None
    snapshot = decode_runtime_join_snapshot(
        state["_rt_join"],
        expected_join_node_ids,
        allow_legacy_lengths=allow_legacy_lengths,
        allow_single_mask_waiters=allow_single_mask_waiters,
    )
    if snapshot is None:
        raise ValueError(
            "persisted join snapshot is malformed or incompatible with the "
            "current workflow design"
        )
    return snapshot
