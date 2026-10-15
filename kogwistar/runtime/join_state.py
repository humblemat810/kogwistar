"""Typed persisted join-state contracts shared by workflow runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypedDict
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
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_join_waiter(
    value: object, *, allow_single_mask: bool
) -> JoinWaiter | None:
    """Decode one persisted waiter without trusting an external payload shape."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) >= 3:
        mask = _coerce_int(value[0])
        if mask is None:
            return None
        return mask, str(value[1]), None if value[2] is None else str(value[2])
    if allow_single_mask and len(value) == 1:
        mask = _coerce_int(value[0])
        return (mask, uuid.uuid4().hex, None) if mask is not None else None
    return None


def normalize_join_token(
    value: object, *, allow_legacy_lengths: bool
) -> JoinToken | None:
    """Decode a persisted token, retaining only a complete known shape."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) >= 4:
        mask = _coerce_int(value[1])
        if mask is None:
            return None
        return (
            str(value[0]),
            mask,
            str(value[2]),
            None if value[3] is None else str(value[3]),
        )
    if not allow_legacy_lengths:
        return None
    if len(value) == 3:
        mask = _coerce_int(value[1])
        return (str(value[0]), mask, str(value[2]), None) if mask is not None else None
    if len(value) == 2:
        mask = _coerce_int(value[1])
        return (
            (str(value[0]), mask, uuid.uuid4().hex, None)
            if mask is not None
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
    outstanding = [
        _coerce_int(item) or 0
        for item in outstanding_value
    ]
    waiters: dict[str, list[JoinWaiter]] = {}
    for join_id in expected_join_node_ids:
        items_value = waiters_value.get(join_id, [])
        if not isinstance(items_value, list):
            return None
        waiters[join_id] = [
            normalized
            for item in items_value
            if (
                normalized := normalize_join_waiter(
                    item, allow_single_mask=allow_single_mask_waiters
                )
            )
            is not None
        ]
    pending = [
        normalized
        for item in pending_value
        if (
            normalized := normalize_join_token(
                item, allow_legacy_lengths=allow_legacy_lengths
            )
        )
        is not None
    ]
    suspended = [
        normalized
        for item in suspended_value
        if (
            normalized := normalize_join_token(
                item, allow_legacy_lengths=allow_legacy_lengths
            )
        )
        is not None
    ]
    return RuntimeJoinSnapshot(
        join_node_ids=list(expected_join_node_ids),
        join_outstanding=outstanding,
        join_waiters=waiters,
        pending=pending,
        suspended=suspended,
    )
