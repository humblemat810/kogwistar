from __future__ import annotations

"""Durable ownership fencing for resumable workflow runs."""

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from kogwistar.json_types import JsonObject


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
    return int(value) if isinstance(value, (int, float, str)) else None


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
    current = store.get_named_projection(namespace, str(run_id))
    current_payload = current.get("payload") if isinstance(current, Mapping) else None
    current_payload = current_payload if isinstance(current_payload, Mapping) else {}
    current_owner = str(current_payload.get("owner_id") or "")
    current_lease = current_payload.get("lease_until_ms")
    current_lease_ms = (
        int(current_lease) if isinstance(current_lease, (int, float, str)) else 0
    )
    if (
        current_owner
        and current_owner != owner_id
        and current_lease_ms > now
        and not (allow_resume_takeover and current_payload.get("resumable") is True)
    ):
        raise RunOwnershipConflict(
            f"workflow run {run_id!r} is owned until {current_lease_ms}"
        )

    next_owner = str(owner_id or uuid.uuid4())
    generation_value = current_payload.get("generation")
    generation = (
        int(generation_value) + 1
        if isinstance(generation_value, (int, float, str))
        else 1
    )
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
    current = store.get_named_projection(namespace, str(run_id))
    current_payload = current.get("payload") if isinstance(current, Mapping) else None
    current_payload = current_payload if isinstance(current_payload, Mapping) else {}
    if (
        str(current_payload.get("owner_id") or "") != ownership.owner_id
        or int(current_payload.get("generation", -1)) != ownership.generation
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
    current = store.get_named_projection(namespace, str(run_id))
    current_payload = current.get("payload") if isinstance(current, Mapping) else None
    current_payload = current_payload if isinstance(current_payload, Mapping) else {}
    if (
        str(current_payload.get("owner_id") or "") != ownership.owner_id
        or int(current_payload.get("generation", -1)) != ownership.generation
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
