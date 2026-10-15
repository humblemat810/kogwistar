from __future__ import annotations

"""Durable ownership fencing for resumable workflow runs."""

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from kogwistar.json_types import JsonObject, JsonValue


class RunOwnershipStore(Protocol):
    def get_named_projection(self, namespace: str, key: str) -> JsonObject | None: ...

    def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: JsonObject,
        *,
        expected_last_authoritative_seq: int | None,
        expected_last_materialized_seq: int | None,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> bool: ...


RUN_OWNERSHIP_SCHEMA_VERSION = 1
DEFAULT_RUN_OWNERSHIP_LEASE_MS = 15 * 60 * 1000


class RunOwnershipConflict(RuntimeError):
    """Raised when another live worker owns a resumable run."""


@dataclass(frozen=True, slots=True)
class RunOwnership:
    owner_id: str
    lease_until_ms: int
    generation: int


def _row_sequence(row: Mapping[str, object] | None, key: str) -> int | None:
    value = row.get(key) if row is not None else None
    return _coerce_int(value)


def _coerce_int(value: object) -> int | None:
    """Convert scalar projection values without trusting JSON container types."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def claim_run_ownership(
    store: RunOwnershipStore,
    *,
    namespace: str,
    run_id: str,
    workflow_id: str,
    conversation_id: str,
    owner_id: str | None = None,
    now_ms: int | None = None,
    lease_ms: int = DEFAULT_RUN_OWNERSHIP_LEASE_MS,
    allow_resume_takeover: bool = False,
) -> RunOwnership:
    """Atomically claim a run, or fail while an unexpired owner holds it."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if lease_ms <= 0:
        raise ValueError("run ownership lease must be positive")
    current: JsonObject | None = store.get_named_projection(namespace, str(run_id))
    current_payload_value: JsonValue | None = (
        current.get("payload") if current is not None else None
    )
    current_payload: Mapping[str, JsonValue] = (
        current_payload_value if isinstance(current_payload_value, dict) else {}
    )
    current_owner_value: JsonValue | None = current_payload.get("owner_id")
    current_owner: str = (
        current_owner_value if isinstance(current_owner_value, str) else ""
    )
    current_lease: JsonValue | None = current_payload.get("lease_until_ms")
    current_lease_ms = _coerce_int(current_lease) or 0
    resumable_value: JsonValue | None = current_payload.get("resumable")
    if (
        current_owner
        and current_owner != owner_id
        and current_lease_ms > now
        and not (allow_resume_takeover and resumable_value is True)
    ):
        raise RunOwnershipConflict(
            f"workflow run {run_id!r} is owned until {current_lease_ms}"
        )

    next_owner = str(owner_id or uuid.uuid4())
    generation_value: JsonValue | None = current_payload.get("generation")
    previous_generation: int = _coerce_int(generation_value) or 0
    generation: int = previous_generation + 1
    latest_authoritative = _row_sequence(current, "last_authoritative_seq")
    latest_materialized = _row_sequence(current, "last_materialized_seq")
    payload: JsonObject = {
        "run_id": str(run_id),
        "workflow_id": str(workflow_id),
        "conversation_id": str(conversation_id),
        "owner_id": next_owner,
        "lease_until_ms": now + int(lease_ms),
        "generation": generation,
        "resumable": False,
        "schema_version": RUN_OWNERSHIP_SCHEMA_VERSION,
    }
    if not store.compare_and_swap_named_projection(
        namespace,
        str(run_id),
        payload,
        expected_last_authoritative_seq=latest_authoritative,
        expected_last_materialized_seq=latest_materialized,
        last_authoritative_seq=(latest_authoritative or 0) + 1,
        last_materialized_seq=(latest_materialized or 0) + 1,
        projection_schema_version=RUN_OWNERSHIP_SCHEMA_VERSION,
        materialization_status="claimed",
    ):
        raise RunOwnershipConflict(f"workflow run {run_id!r} ownership changed")
    return RunOwnership(
        owner_id=next_owner,
        lease_until_ms=now + int(lease_ms),
        generation=generation,
    )


def release_run_ownership_for_resume(
    store: RunOwnershipStore,
    *,
    namespace: str,
    run_id: str,
    ownership: RunOwnership,
) -> bool:
    """Make a suspended run immediately claimable without dropping its fence."""
    current: JsonObject | None = store.get_named_projection(namespace, str(run_id))
    current_payload_value: JsonValue | None = (
        current.get("payload") if current is not None else None
    )
    current_payload: Mapping[str, JsonValue] = (
        current_payload_value if isinstance(current_payload_value, dict) else {}
    )
    current_owner_value: JsonValue | None = current_payload.get("owner_id")
    current_owner: str = (
        current_owner_value if isinstance(current_owner_value, str) else ""
    )
    generation_value: JsonValue | None = current_payload.get("generation", -1)
    current_generation: int | None = _coerce_int(generation_value)
    if (
        current_owner != ownership.owner_id
        or current_generation is None
        or current_generation != ownership.generation
    ):
        return False
    latest_authoritative = _row_sequence(current, "last_authoritative_seq")
    latest_materialized = _row_sequence(current, "last_materialized_seq")
    payload: JsonObject = dict(current_payload)
    payload["lease_until_ms"] = 0
    payload["resumable"] = True
    return store.compare_and_swap_named_projection(
        namespace,
        str(run_id),
        payload,
        expected_last_authoritative_seq=latest_authoritative,
        expected_last_materialized_seq=latest_materialized,
        last_authoritative_seq=(latest_authoritative or 0) + 1,
        last_materialized_seq=(latest_materialized or 0) + 1,
        projection_schema_version=RUN_OWNERSHIP_SCHEMA_VERSION,
        materialization_status="suspended",
    )


def renew_run_ownership(
    store: RunOwnershipStore,
    *,
    namespace: str,
    run_id: str,
    ownership: RunOwnership,
    now_ms: int | None = None,
    lease_ms: int = DEFAULT_RUN_OWNERSHIP_LEASE_MS,
) -> RunOwnership:
    """Extend a lease only when the caller still owns the current generation."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    current: JsonObject | None = store.get_named_projection(namespace, str(run_id))
    current_payload_value: JsonValue | None = (
        current.get("payload") if current is not None else None
    )
    current_payload: Mapping[str, JsonValue] = (
        current_payload_value if isinstance(current_payload_value, dict) else {}
    )
    current_owner_value: JsonValue | None = current_payload.get("owner_id")
    current_owner: str = (
        current_owner_value if isinstance(current_owner_value, str) else ""
    )
    generation_value: JsonValue | None = current_payload.get("generation", -1)
    current_generation: int | None = _coerce_int(generation_value)
    if (
        current_owner != ownership.owner_id
        or current_generation is None
        or current_generation != ownership.generation
    ):
        raise RunOwnershipConflict(f"workflow run {run_id!r} ownership was lost")
    latest_authoritative = _row_sequence(current, "last_authoritative_seq")
    latest_materialized = _row_sequence(current, "last_materialized_seq")
    payload: JsonObject = dict(current_payload)
    payload["lease_until_ms"] = now + int(lease_ms)
    if not store.compare_and_swap_named_projection(
        namespace,
        str(run_id),
        payload,
        expected_last_authoritative_seq=latest_authoritative,
        expected_last_materialized_seq=latest_materialized,
        last_authoritative_seq=(latest_authoritative or 0) + 1,
        last_materialized_seq=(latest_materialized or 0) + 1,
        projection_schema_version=RUN_OWNERSHIP_SCHEMA_VERSION,
        materialization_status="claimed",
    ):
        raise RunOwnershipConflict(f"workflow run {run_id!r} ownership changed")
    return RunOwnership(
        owner_id=ownership.owner_id,
        lease_until_ms=now + int(lease_ms),
        generation=ownership.generation,
    )
