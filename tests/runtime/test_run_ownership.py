from __future__ import annotations

import pytest

from kogwistar.json_types import JsonObject
from kogwistar.runtime.run_ownership import (
    RunOwnershipConflict,
    claim_run_ownership,
    release_run_ownership_for_resume,
    renew_run_ownership,
)

pytestmark = pytest.mark.ci


class _ProjectionStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], JsonObject] = {}
        self.sequence = 0

    def get_named_projection(self, namespace: str, key: str) -> JsonObject | None:
        row = self.rows.get((namespace, key))
        return dict(row) if row is not None else None

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
    ) -> bool:
        current = self.rows.get((namespace, key))
        if current is None:
            if (
                expected_last_authoritative_seq is not None
                or expected_last_materialized_seq is not None
            ):
                return False
        else:
            if current["last_authoritative_seq"] != expected_last_authoritative_seq:
                return False
            if current["last_materialized_seq"] != expected_last_materialized_seq:
                return False
        self.rows[(namespace, key)] = {
            "namespace": namespace,
            "key": key,
            "payload": payload,
            "last_authoritative_seq": last_authoritative_seq,
            "last_materialized_seq": last_materialized_seq,
            "projection_schema_version": projection_schema_version,
            "materialization_status": materialization_status,
        }
        return True


def test_only_one_live_owner_can_claim_a_run() -> None:
    store = _ProjectionStore()
    first = claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-a",
        now_ms=100,
        lease_ms=1000,
    )

    with pytest.raises(RunOwnershipConflict):
        claim_run_ownership(
            store,
            namespace="conversation:ownership",
            run_id="run-1",
            workflow_id="ocr",
            conversation_id="doc-1",
            owner_id="worker-b",
            now_ms=500,
            lease_ms=1000,
        )

    renewed = renew_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        ownership=first,
        now_ms=700,
        lease_ms=1000,
    )
    assert renewed.owner_id == "worker-a"
    assert renewed.lease_until_ms == 1700


def test_expired_owner_can_be_reclaimed_with_a_new_generation() -> None:
    store = _ProjectionStore()
    claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-a",
        now_ms=100,
        lease_ms=100,
    )
    second = claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-b",
        now_ms=201,
        lease_ms=100,
    )
    assert second.owner_id == "worker-b"
    assert second.generation == 2


def test_stale_owner_cannot_renew_after_reclaim() -> None:
    store = _ProjectionStore()
    first = claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-a",
        now_ms=100,
        lease_ms=100,
    )
    claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-b",
        now_ms=201,
        lease_ms=100,
    )
    with pytest.raises(RunOwnershipConflict):
        renew_run_ownership(
            store,
            namespace="conversation:ownership",
            run_id="run-1",
            ownership=first,
            now_ms=250,
            lease_ms=100,
        )


def test_resume_takeover_uses_cas_fencing() -> None:
    store = _ProjectionStore()
    first = claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-a",
        now_ms=100,
        lease_ms=1000,
    )
    assert release_run_ownership_for_resume(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        ownership=first,
    )
    second = claim_run_ownership(
        store,
        namespace="conversation:ownership",
        run_id="run-1",
        workflow_id="ocr",
        conversation_id="doc-1",
        owner_id="worker-b",
        now_ms=200,
        lease_ms=1000,
        allow_resume_takeover=True,
    )
    assert second.owner_id == "worker-b"
    with pytest.raises(RunOwnershipConflict):
        claim_run_ownership(
            store,
            namespace="conversation:ownership",
            run_id="run-1",
            workflow_id="ocr",
            conversation_id="doc-1",
            owner_id="worker-c",
            now_ms=200,
            lease_ms=1000,
            allow_resume_takeover=True,
        )
