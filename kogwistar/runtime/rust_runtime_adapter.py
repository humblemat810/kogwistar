"""Private ADR-015 Phase-4 recorded-runtime SQLite facade.

This is intentionally not imported by :mod:`kogwistar.runtime`.  It accepts
JSON-shaped recorded worker results only and never holds/calls a resolver,
provider, tool, lane, graph, or runtime object.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .._rust_bridge import store_sqlite


_TRANSITION_FIELDS = frozenset(
    {
        "contract_version",
        "transition_id",
        "expected_event_seq",
        "kind",
        "run_id",
        "workflow_id",
        "conversation_id",
        "user_id",
        "user_turn_node_id",
        "step_seq",
        "node_id",
        "token_id",
        "parent_token_id",
        "initial_state",
        "state_update",
        "update",
        "state_schema",
        "frontier",
        "result",
        "wait_reason",
        "resume_payload",
        "errors",
    }
)
_WORKER_HANDOFF_FIELDS = frozenset(
    {"message_id", "claimed_by", "run_id", "step_id", "correlation_id"}
)


def _require_json(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise TypeError(f"{field} must contain finite JSON numbers")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json(item, field=f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} JSON object keys must be strings")
            _require_json(item, field=f"{field}.{key}")
        return
    raise TypeError(f"{field} must be JSON-only, got {type(value).__name__}")


def _require_transition(transition: dict[str, Any]) -> None:
    if not isinstance(transition, dict):
        raise TypeError("transition must be a JSON object")
    _require_json(transition, field="transition")
    unknown = set(transition).difference(_TRANSITION_FIELDS)
    if unknown:
        raise ValueError(f"transition has unknown fields: {sorted(unknown)!r}")
    required = {
        "contract_version",
        "transition_id",
        "expected_event_seq",
        "kind",
        "run_id",
        "workflow_id",
        "conversation_id",
        "step_seq",
    }
    missing = required.difference(transition)
    if missing:
        raise ValueError(f"transition missing fields: {sorted(missing)!r}")


def _require_worker_handoff(handoff: dict[str, Any]) -> None:
    if not isinstance(handoff, dict):
        raise TypeError("handoff must be a JSON object")
    _require_json(handoff, field="handoff")
    unknown = set(handoff).difference(_WORKER_HANDOFF_FIELDS)
    if unknown:
        raise ValueError(f"handoff has unknown fields: {sorted(unknown)!r}")
    missing = _WORKER_HANDOFF_FIELDS.difference(handoff)
    if missing:
        raise ValueError(f"handoff missing fields: {sorted(missing)!r}")
    for field in _WORKER_HANDOFF_FIELDS:
        value = handoff[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"handoff {field} must be a non-empty string")


def apply_recorded_transition(
    *,
    path: str | os.PathLike[str],
    transition: dict[str, Any],
    abort_after_writes: bool = False,
) -> dict[str, Any]:
    """Atomically record one previously-obtained runtime transition.

    `abort_after_writes` exists only for fault-injection tests. Public
    `WorkflowRuntime` remains Python-owned until later explicit cutover.
    """
    _require_transition(transition)
    if not isinstance(abort_after_writes, bool):
        raise TypeError("abort_after_writes must be bool")
    value = store_sqlite(
        path=os.fspath(path),
        operation={
            "kind": "apply_recorded_runtime_transition",
            "transition": transition,
            "abort_after_writes": abort_after_writes,
        },
    )
    if not isinstance(value, dict):
        raise RuntimeError("recorded runtime transition returned non-object JSON")
    return value


def apply_claimed_worker_result(
    *,
    path: str | os.PathLike[str],
    handoff: dict[str, Any],
    transition: dict[str, Any],
    abort_after_writes: bool = False,
) -> dict[str, Any]:
    """Atomically accept one claimed Python-worker result.

    Rust validates durable claim ownership and request identity, records the
    transition/checkpoint/status, then acknowledges the lane request in the
    same SQLite transaction. No Python callback occurs inside that boundary.
    """
    _require_worker_handoff(handoff)
    _require_transition(transition)
    if not isinstance(abort_after_writes, bool):
        raise TypeError("abort_after_writes must be bool")
    value = store_sqlite(
        path=os.fspath(path),
        operation={
            "kind": "apply_claimed_recorded_runtime_transition",
            "handoff": handoff,
            "transition": transition,
            "abort_after_writes": abort_after_writes,
        },
    )
    if not isinstance(value, dict):
        raise RuntimeError("worker result handoff returned non-object JSON")
    return value


def read_recorded_runtime_state(
    *,
    path: str | os.PathLike[str],
    run_id: str,
    workflow_id: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    """Read restart state; never dispatch pending or suspended tokens."""
    for field, value in (
        ("run_id", run_id),
        ("workflow_id", workflow_id),
        ("conversation_id", conversation_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    value = store_sqlite(
        path=os.fspath(path),
        operation={
            "kind": "read_recorded_runtime_state",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
        },
    )
    if value is not None and not isinstance(value, dict):
        raise RuntimeError("recorded runtime read returned non-object JSON")
    return value
