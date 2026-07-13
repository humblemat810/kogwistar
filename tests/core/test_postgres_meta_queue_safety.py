from __future__ import annotations

from contextlib import contextmanager

import pytest

from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.postgres_backend import _set_active_conn
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore


class _Result:
    def fetchone(self):
        return ("existing-job",)


class _Connection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.savepoints = 0

    @contextmanager
    def begin_nested(self):
        self.savepoints += 1
        try:
            yield
        except Exception:
            raise

    def execute(self, statement, params=None):
        del params
        self.statements.append(str(statement))
        return _Result()


def test_enqueue_uses_atomic_pending_coalesce_upsert_inside_savepoint() -> None:
    connection = _Connection()
    meta = EnginePostgresMetaStore(engine=object(), schema="public")

    with _set_active_conn(connection):
        job_id = meta.enqueue_index_job(
            job_id="new-job",
            namespace="source",
            entity_kind="node",
            entity_id="node-1",
            index_kind="node_docs",
            op="UPSERT",
        )

    assert job_id == "existing-job"
    assert connection.savepoints == 1
    sql = connection.statements[0]
    assert "ON CONFLICT (namespace, coalesce_key) WHERE status='PENDING'" in sql
    assert "FOR UPDATE" not in sql


def test_nested_metadata_failure_does_not_escape_as_outer_transaction_failure() -> None:
    connection = _Connection()
    meta = EnginePostgresMetaStore(engine=object(), schema="public")

    with _set_active_conn(connection):
        try:
            with meta.transaction():
                raise RuntimeError("queue operation failed")
        except RuntimeError:
            pass

    assert connection.savepoints == 1


def test_stale_claim_cannot_complete_reclaimed_in_memory_job() -> None:
    meta = InMemoryMetaStore()
    meta.enqueue_index_job(
        job_id="job-1",
        namespace="maintenance",
        entity_kind="node",
        entity_id="node-1",
        index_kind="maintenance",
        op="UPSERT",
    )
    first = meta.claim_index_jobs(limit=1, lease_seconds=1, namespace="maintenance")[0]
    meta._debug_force_job_lease(job_id="job-1", lease_until=0)
    second = meta.claim_index_jobs(limit=1, lease_seconds=60, namespace="maintenance")[0]

    meta.mark_index_job_done(first.job_id, claim_token=first.claim_token)
    assert meta.list_index_jobs(namespace="maintenance")[0].status == "DOING"
    meta.mark_index_job_done(second.job_id, claim_token=second.claim_token)
    assert meta.list_index_jobs(namespace="maintenance")[0].status == "DONE"


@pytest.mark.core
def test_postgres_claim_token_null_predicates_are_explicitly_typed() -> None:
    """Guard PostgreSQL's inability to infer the type of a NULL bind value."""
    meta = EnginePostgresMetaStore(engine=object(), schema="public")
    connection = _Connection()

    with _set_active_conn(connection):
        meta.mark_index_job_done("job-1", claim_token=None)
        meta.mark_index_job_failed("job-1", "failed", claim_token=None)
        meta.bump_retry_and_requeue("job-1", "retry", next_run_at_seconds=0, claim_token=None)
        meta.requeue_index_job_at_tail("job-1", payload_json="{}", claim_token=None)

    assert len(connection.statements) == 4
    assert all("CAST(:claim_token AS TEXT)" in sql for sql in connection.statements)
