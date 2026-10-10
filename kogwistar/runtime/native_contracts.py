"""Typed adapters for native runtime JSON results.

The Rust bridge deliberately exposes JSON objects.  These adapters validate the
small result protocols at the boundary before the schedulers consume them.
"""

from __future__ import annotations

from typing import TypedDict

from ..json_types import JsonObject, JsonValue


class PlannedToken(TypedDict):
    node_id: str
    join_mask: int
    token_id: str
    parent_token_id: str | None
    spawned: bool


class JoinWaiter(TypedDict):
    join_mask: int
    token_id: str
    parent_token_id: str | None


class JoinArrivalResult(TypedDict):
    join_outstanding: list[int]
    waiters: list[JoinWaiter]
    released: PlannedToken | None
    collapsed_work_items: int
    outstanding: int


def _required_int(value: JsonValue | None, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"native runtime result field {field!r} must be an integer")
    return value


def _required_text(value: JsonValue | None, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"native runtime result field {field!r} must be a string")
    return value


def _optional_text(value: JsonValue | None, *, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise RuntimeError(
            f"native runtime result field {field!r} must be a string or null"
        )
    return value


def _object(value: JsonValue | None, *, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RuntimeError(f"native runtime result field {field!r} must be an object")
    return value


def _int_list(value: JsonValue | None, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise RuntimeError(f"native runtime result field {field!r} must be a list")
    return [_required_int(item, field=f"{field}[]") for item in value]


def _token(value: JsonValue | None, *, field: str) -> PlannedToken:
    item = _object(value, field=field)
    return {
        "node_id": _required_text(item.get("node_id"), field=f"{field}.node_id"),
        "join_mask": _required_int(item.get("join_mask"), field=f"{field}.join_mask"),
        "token_id": _required_text(item.get("token_id"), field=f"{field}.token_id"),
        "parent_token_id": _optional_text(
            item.get("parent_token_id"), field=f"{field}.parent_token_id"
        ),
        "spawned": item.get("spawned") is True,
    }


def planned_tokens(value: JsonValue | None) -> list[PlannedToken]:
    if not isinstance(value, list):
        raise RuntimeError("native runtime result field 'tokens' must be a list")
    return [_token(item, field=f"tokens[{index}]") for index, item in enumerate(value)]


def successor_plan(value: JsonObject) -> tuple[list[PlannedToken], list[int]]:
    return planned_tokens(value.get("tokens")), _int_list(
        value.get("join_outstanding"), field="join_outstanding"
    )


def join_arrival_result(value: JsonObject) -> JoinArrivalResult:
    released_value = value.get("released")
    released = None if released_value is None else _token(released_value, field="released")
    raw_waiters = value.get("waiters")
    if not isinstance(raw_waiters, list):
        raise RuntimeError("native runtime result field 'waiters' must be a list")
    waiters: list[JoinWaiter] = []
    for index, raw_waiter in enumerate(raw_waiters):
        waiter = _object(raw_waiter, field=f"waiters[{index}]")
        waiters.append(
            {
                "join_mask": _required_int(
                    waiter.get("join_mask"), field=f"waiters[{index}].join_mask"
                ),
                "token_id": _required_text(
                    waiter.get("token_id"), field=f"waiters[{index}].token_id"
                ),
                "parent_token_id": _optional_text(
                    waiter.get("parent_token_id"),
                    field=f"waiters[{index}].parent_token_id",
                ),
            }
        )
    return {
        "join_outstanding": _int_list(
            value.get("join_outstanding"), field="join_outstanding"
        ),
        "waiters": waiters,
        "released": released,
        "collapsed_work_items": _required_int(
            value.get("collapsed_work_items"), field="collapsed_work_items"
        ),
        "outstanding": _required_int(value.get("outstanding"), field="outstanding"),
    }
