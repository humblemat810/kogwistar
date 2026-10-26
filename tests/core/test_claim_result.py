from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore

pytestmark = pytest.mark.core


def _enqueue(meta: object, job_id: str = "candidate-job") -> None:
    meta.enqueue_index_job(
        job_id=job_id,
        namespace="maintenance",
        entity_kind="maintenance",
        entity_id=job_id,
        index_kind="parse",
        op="UPSERT",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_first_live_claim_wins_and_result_is_immutable(backend: str, tmp_path: Path) -> None:
    meta = InMemoryMetaStore() if backend == "memory" else EngineSQLite(tmp_path / "meta")
    if backend == "sqlite":
        meta.ensure_initialized()
    _enqueue(meta)
    claim = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="maintenance")[0]
    winner = meta.accept_index_job_result(
        claim.job_id,
        claim_token=claim.claim_token or "",
        result_json='{"winner":1}',
        result_sha256="digest-1",
    )
    assert winner["status"] == "accepted"
    loser = meta.accept_index_job_result(
        claim.job_id,
        claim_token="stale-worker",
        result_json='{"winner":2}',
        result_sha256="digest-2",
    )
    assert loser["status"] == "existing"
    assert loser["result_json"] == '{"winner":1}'


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_expired_claim_cannot_admit_a_result(backend: str, tmp_path: Path) -> None:
    meta = InMemoryMetaStore() if backend == "memory" else EngineSQLite(tmp_path / "meta")
    if backend == "sqlite":
        meta.ensure_initialized()
    _enqueue(meta)

    claim = meta.claim_index_jobs(limit=1, lease_seconds=-1, namespace="maintenance")[0]
    rejected = meta.accept_index_job_result(
        claim.job_id,
        claim_token=claim.claim_token or "",
        result_json='{"late":1}',
        result_sha256="late-digest",
    )

    assert rejected["status"] == "rejected"
    assert meta.get_index_job_result(claim.job_id) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_expired_claim_token_cannot_mutate_queue_state(backend: str, tmp_path: Path) -> None:
    meta = InMemoryMetaStore() if backend == "memory" else EngineSQLite(tmp_path / "meta")
    if backend == "sqlite":
        meta.ensure_initialized()

    for job_id in ("done", "failed", "retried", "requeued"):
        _enqueue(meta, job_id)
    claims = {
        claim.job_id: claim
        for claim in meta.claim_index_jobs(
            limit=4, lease_seconds=-1, namespace="maintenance"
        )
    }
    assert set(claims) == {"done", "failed", "retried", "requeued"}

    assert not meta.mark_index_job_done("done", claim_token=claims["done"].claim_token)
    meta.mark_index_job_failed(
        "failed", "late failure", claim_token=claims["failed"].claim_token
    )
    meta.bump_retry_and_requeue(
        "retried",
        "late retry",
        next_run_at_seconds=0,
        claim_token=claims["retried"].claim_token,
    )
    meta.requeue_index_job_at_tail(
        "requeued",
        payload_json="{}",
        claim_token=claims["requeued"].claim_token,
    )

    reclaimed = meta.claim_index_jobs(
        limit=4, lease_seconds=30, namespace="maintenance"
    )
    assert {claim.job_id for claim in reclaimed} == set(claims)
    assert all(claim.retry_count == 0 for claim in reclaimed)
    assert all(claim.claim_token != claims[claim.job_id].claim_token for claim in reclaimed)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_expired_claim_keeps_legacy_tokenless_admin_completion(backend: str, tmp_path: Path) -> None:
    meta = InMemoryMetaStore() if backend == "memory" else EngineSQLite(tmp_path / "meta")
    if backend == "sqlite":
        meta.ensure_initialized()
    _enqueue(meta)
    claim = meta.claim_index_jobs(limit=1, lease_seconds=-1, namespace="maintenance")[0]

    assert meta.mark_index_job_done(claim.job_id)


def test_expired_postgres_claim_token_cannot_mutate_queue_across_languages(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    from kogwistar._rust_bridge import store_postgres
    from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore

    python = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    python.ensure_initialized()
    namespace = f"expired-claim-{uuid.uuid4().hex}"
    job_ids = ("done", "failed", "retried", "requeued")
    for job_id in job_ids:
        python.enqueue_index_job(
            job_id=job_id,
            namespace=namespace,
            entity_kind="node",
            entity_id=job_id,
            index_kind="test",
            op="UPSERT",
        )
    claims = {
        claim.job_id: claim
        for claim in python.claim_index_jobs(
            limit=4, lease_seconds=-1, namespace=namespace
        )
    }
    assert set(claims) == set(job_ids)

    assert store_postgres(
        dsn=pg_dsn,
        schema=pg_schema,
        operation={
            "kind": "mark_index_job_done",
            "job_id": "done",
            "claim_token": claims["done"].claim_token,
        },
    ) is False
    python.mark_index_job_failed(
        "failed", "late", claim_token=claims["failed"].claim_token
    )
    store_postgres(
        dsn=pg_dsn,
        schema=pg_schema,
        operation={
            "kind": "bump_retry_and_requeue",
            "job_id": "retried",
            "error": "late",
            "next_run_at_seconds": 0,
            "claim_token": claims["retried"].claim_token,
        },
    )
    python.requeue_index_job_at_tail(
        "requeued",
        payload_json="{}",
        claim_token=claims["requeued"].claim_token,
    )

    reclaimed = store_postgres(
        dsn=pg_dsn,
        schema=pg_schema,
        operation={
            "kind": "claim_index_jobs",
            "namespace": namespace,
            "limit": 4,
            "lease_seconds": 30,
        },
    )
    assert {claim["job_id"] for claim in reclaimed} == set(job_ids)
    assert all(claim["retry_count"] == 0 for claim in reclaimed)
    assert all(
        claim["claim_token"] != claims[claim["job_id"]].claim_token
        for claim in reclaimed
    )
