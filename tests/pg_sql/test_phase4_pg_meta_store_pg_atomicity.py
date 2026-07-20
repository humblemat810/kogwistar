import json
import threading
import uuid

import pytest
pytestmark = pytest.mark.ci_full
pytest.importorskip("sqlalchemy")
import sqlalchemy as sa

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend

# New meta-store introduced by the patch
from kogwistar.engine_core.engine_postgres_meta import (
    EnginePostgresMetaStore,
)


def _require_postgres(sa_engine, pg_schema) -> None:
    if sa_engine is None or pg_schema is None:
        pytest.skip("PostgreSQL fixtures are unavailable")


def _fake_ef(texts):
    # deterministic, dimension=3
    return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.mark.parametrize("distance", ["cosine"])
def test_pg_backend_uses_postgres_meta_store(
    sa_engine, pg_schema, tmp_path, distance: str
):
    _require_postgres(sa_engine, pg_schema)
    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, distance=distance, schema=pg_schema
    )

    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "gke_meta"),
        embedding_function=_fake_ef,
        backend=backend,
        kg_graph_type="knowledge",
    )

    assert isinstance(eng.meta_sqlite, EnginePostgresMetaStore)


def _count_user_seq_rows(sa_engine, schema: str) -> int:
    q = sa.text(f"SELECT COUNT(*) FROM {schema}.user_seq")
    with sa_engine.connect() as conn:
        return int(conn.execute(q).scalar_one())


def _count_nodes(sa_engine, schema: str) -> int:
    q = sa.text(f"SELECT COUNT(*) FROM {schema}.gke_nodes")
    with sa_engine.connect() as conn:
        return int(conn.execute(q).scalar_one())


@pytest.mark.parametrize("distance", ["cosine"])
def test_engine_uow_rolls_back_meta_and_graph_writes_together(
    sa_engine, pg_schema, tmp_path, distance: str
):
    """
    Proves the new PG meta-store participates in the SAME PG transaction as pgvector writes.

    Inside engine.uow():
      - bump user_seq (meta write)
      - insert a node (graph write)
      - raise to simulate crash-before-commit

    After exception:
      - gke_user_seq should have no row (meta write rolled back)
      - gke_nodes should have no row (graph write rolled back)
    """
    _require_postgres(sa_engine, pg_schema)
    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, distance=distance, schema=pg_schema
    )
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "gke_meta"),
        embedding_function=_fake_ef,
        backend=backend,
        kg_graph_type="knowledge",
    )

    # Pre-assert empty
    assert _count_user_seq_rows(sa_engine, pg_schema) == 0
    assert _count_nodes(sa_engine, pg_schema) == 0

    with pytest.raises(RuntimeError):
        with eng.uow():
            # meta write
            _ = eng.meta_sqlite.next_user_seq("user1")
            # graph write (direct to backend to keep this test stable)
            eng.backend.node_add(
                ids=["X"],
                documents=["X"],
                metadatas=[{"name": "X"}],
                embeddings=[[1.0, 0.0, 0.0]],
            )
            raise RuntimeError("power out")

    # Both should be rolled back if they share the same PG transaction
    assert _count_user_seq_rows(sa_engine, pg_schema) == 0
    assert _count_nodes(sa_engine, pg_schema) == 0


@pytest.mark.ci_full
def test_append_entity_event_allocates_unique_sequences_for_new_namespace_concurrently(
    sa_engine, pg_schema
):
    _require_postgres(sa_engine, pg_schema)
    meta = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    meta.ensure_initialized()

    namespace = f"namespace-{uuid.uuid4().hex[:8]}"
    barrier = threading.Barrier(2)
    results: list[int] = []
    errors: list[BaseException] = []

    def _worker(ix: int) -> None:
        try:
            barrier.wait(timeout=10.0)
            seq = meta.append_entity_event(
                namespace=namespace,
                event_id=f"event-{ix}-{uuid.uuid4().hex[:8]}",
                entity_kind="node",
                entity_id=f"node-{ix}",
                op="UPSERT",
                payload_json=json.dumps({"ix": ix}, sort_keys=True, separators=(",", ":")),
            )
            results.append(seq)
        except BaseException as exc:  # pragma: no cover - thread capture
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=(1,), name="append-event-1")
    t2 = threading.Thread(target=_worker, args=(2,), name="append-event-2")
    t1.start()
    t2.start()
    t1.join(timeout=20.0)
    t2.join(timeout=20.0)

    assert not t1.is_alive(), "worker 1 did not finish"
    assert not t2.is_alive(), "worker 2 did not finish"
    assert not errors, f"unexpected concurrent append failure: {errors!r}"
    assert sorted(results) == [1, 2]


@pytest.mark.ci_full
def test_postgres_queue_accepts_null_claim_token_on_terminal_updates(sa_engine, pg_schema):
    """Execute nullable claim-token predicates through real psycopg/Postgres."""
    if sa_engine is None or pg_schema is None:
        pytest.skip("Postgres fixture is not available")

    meta = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    meta.ensure_initialized()
    job_id = f"null-claim-{uuid.uuid4().hex}"
    meta.enqueue_index_job(
        job_id=job_id,
        namespace="maintenance",
        entity_kind="node",
        entity_id=job_id,
        index_kind="maintenance",
        op="UPSERT",
    )
    claimed = meta.claim_index_jobs(limit=1, namespace="maintenance")
    assert [job.job_id for job in claimed] == [job_id]

    assert meta.mark_index_job_done(job_id, claim_token=None) is True
    row = meta.list_index_jobs(namespace="maintenance", status="DONE")
    assert any(job.job_id == job_id for job in row)
