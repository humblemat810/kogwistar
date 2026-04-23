import time
import time
import pathlib
import sqlite3

import pytest
pytestmark = pytest.mark.ci_full

pytest.importorskip("sqlalchemy")

import sqlalchemy as sa

from kogwistar.engine_core.chroma_backend import ChromaBackend
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.engine_core.engine_postgres_meta import (
    EnginePostgresMetaStore,
)

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Node, Edge, Grounding, Span
from tests.conftest import FakeEmbeddingFunction
from tests._helpers.meta_job_state import set_index_job_state
from tests._helpers.fake_backend import InMemoryBackend, build_fake_backend

# Reuse raw helpers to avoid duplicating Chroma plumbing.
# from tests.conftest import add_node_raw, add_edge_raw


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return sp


EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)


def _mk_node(node_id: str, *, doc_id: str) -> Node:

    node = Node(
        id=node_id,
        label=f"Node {node_id}",
        type="entity",
        summary=f"Summary {node_id}",
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        embedding=[0.1] * EMBEDDING_DIM,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )
    return node


def _mk_edge(edge_id: str, *, src: str, tgt: str, doc_id: str) -> Edge:
    return Edge(
        id=edge_id,
        label=f"Edge {edge_id}",
        type="relationship",
        summary=f"Summary {edge_id}",
        relation="related_to",
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=None,
        target_edge_ids=None,
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"level_from_root": 0, "entity_type": "kg_relation"},
        embedding=[0.1] * EMBEDDING_DIM,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


@pytest.fixture(
    params=[
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
    ids=["fake", "chroma", "pg"],
)
def e2e_engine(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
) -> GraphKnowledgeEngine:
    """Run the same E2E outbox/reconcile tests against both selectors."""

    if request.param == "fake":
        persist_dir = tmp_path / "fake"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(
            persist_directory=str(persist_dir),
            embedding_function=TEST_EMBEDDING,
            backend_factory=build_fake_backend,
        )
    elif request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(
            persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING
        )
    else:
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
        # Skip cleanly if the pgvector dependency isn't available in this environment.
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
        eng = GraphKnowledgeEngine(backend=backend, embedding_function=TEST_EMBEDDING)

    # Test helper (not used by production code): lets tests assert backend type.
    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def _ids(coll_get: dict) -> list[str]:
    return list(coll_get.get("ids") or [])


def _assert_backend_kind(eng: GraphKnowledgeEngine) -> None:
    kind = getattr(eng, "_test_backend_kind", None)
    if kind == "pg":
        assert isinstance(eng.backend, PgVectorBackend)
    elif kind == "fake":
        assert isinstance(eng.backend, InMemoryBackend)
    else:
        assert isinstance(eng.backend, ChromaBackend)


def _job_by_id(eng: GraphKnowledgeEngine, job_id: str):
    for job in eng.meta_sqlite.list_index_jobs(limit=1000):
        if job.job_id == job_id:
            return job
    raise AssertionError(f"missing job_id={job_id}")


def _make_job_runnable_now(eng: GraphKnowledgeEngine, job_id: str) -> None:
    if hasattr(eng.meta_sqlite, "transaction"):
        with eng.meta_sqlite.transaction() as txn:
            if isinstance(eng.meta_sqlite, EnginePostgresMetaStore):
                schema = eng.meta_sqlite.schema
                table = getattr(eng.meta_sqlite, "index_jobs_table", "index_jobs")
                ij = f"{schema}.{table}"
                txn.execute(
                    sa.text(
                        f"UPDATE {ij} "
                        "SET status='PENDING', lease_until=NULL, next_run_at=NOW(), updated_at=NOW() "
                        "WHERE job_id=:job_id"
                    ),
                    {"job_id": job_id},
                )
            elif isinstance(txn, sqlite3.Connection):
                now = int(time.time())
                txn.execute(
                    "UPDATE index_jobs SET status='PENDING', lease_until=NULL, next_run_at=?, updated_at=? WHERE job_id=?",
                    (now, now, job_id),
                )
            else:
                set_index_job_state(
                    eng.meta_sqlite,
                    txn,
                    job_id=str(job_id),
                    status="PENDING",
                    lease_until=None,
                    next_run_at=int(time.time()),
                    updated_at=int(time.time()),
                )


def test_reconcile_builds_all_joins_from_raw_entities(e2e_engine: GraphKnowledgeEngine):
    """E2E: if join collections are empty (crash between base write and join write), reconcile must converge."""
    eng = e2e_engine
    _assert_backend_kind(eng)

    n1 = _mk_node("n1", doc_id="d1")
    n2 = _mk_node("n2", doc_id="d1")
    eng.add_node(n1)
    eng.add_node(n2)

    e1 = _mk_edge("e1", src="n1", tgt="n2", doc_id="d1")
    eng.add_edge(e1)

    # Enqueue the "slow path" derived index rebuilds
    for nid in ("n1", "n2"):
        eng.enqueue_index_job(
            entity_kind="node", entity_id=nid, index_kind="node_docs", op="UPSERT"
        )
        eng.enqueue_index_job(
            entity_kind="node", entity_id=nid, index_kind="node_refs", op="UPSERT"
        )
    eng.enqueue_index_job(
        entity_kind="edge", entity_id="e1", index_kind="edge_endpoints", op="UPSERT"
    )
    eng.enqueue_index_job(
        entity_kind="edge", entity_id="e1", index_kind="edge_refs", op="UPSERT"
    )

    processed = eng.reconcile_indexes(max_jobs=50)
    assert processed >= 1

    # node_docs contains n1<->d1, n2<->d1
    got = eng.backend.node_docs_get(where={"doc_id": "d1"})
    ids = _ids(got)
    assert any(i.startswith("n1::") for i in ids)
    assert any(i.startswith("n2::") for i in ids)

    # node_refs contain doc_id=d1 rows
    got = eng.backend.node_refs_get(where={"doc_id": "d1"})
    assert len(_ids(got)) >= 2

    # edge_endpoints contain src/tgt rows
    got = eng.backend.edge_endpoints_get(where={"edge_id": "e1"})
    ids = _ids(got)
    assert any("::src::" in i for i in ids)
    assert any("::tgt::" in i for i in ids)

    # edge_refs contain doc_id=d1
    got = eng.backend.edge_refs_get(where={"doc_id": "d1"})
    assert len(_ids(got)) >= 1


def test_race_job_before_entity_exists_recovers_on_later_insert(
    e2e_engine: GraphKnowledgeEngine,
):
    """If reconcile runs before the entity exists, the job should requeue, then recover on a later insert."""
    eng = e2e_engine
    _assert_backend_kind(eng)

    jid = eng.enqueue_index_job(
        entity_kind="node", entity_id="n_race", index_kind="node_docs", op="UPSERT"
    )
    eng.reconcile_indexes(max_jobs=10)

    first = _job_by_id(eng, jid)
    assert first.status == "PENDING"
    assert first.retry_count == 1
    assert first.next_run_at is not None
    assert first.last_error is not None
    assert "document not found" in first.last_error

    failed_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="FAILED")}
    assert jid not in failed_ids

    # Later the base entity arrives
    new_node = _mk_node("n_race", doc_id="d1")
    eng.write.add_node(new_node)

    node_docs_jobs = eng.meta_sqlite.list_index_jobs(
        entity_kind="node",
        entity_id="n_race",
        index_kind="node_docs",
        limit=20,
    )
    assert {j.job_id for j in node_docs_jobs} == {jid}

    current = _job_by_id(eng, jid)
    if current.status != "DONE":
        _make_job_runnable_now(eng, jid)
        eng.reconcile_indexes(max_jobs=10)

    final = _job_by_id(eng, jid)
    assert final.status == "DONE"

    got = eng.backend.node_docs_get(where={"node_id": "n_race"})
    assert len(_ids(got)) >= 1


def test_tombstoned_entities_never_resurrect_derived_rows(
    e2e_engine: GraphKnowledgeEngine,
):
    eng = e2e_engine
    _assert_backend_kind(eng)

    eng.add_node(_mk_node("n_dead", doc_id="d1"))

    # First build derived rows
    eng.enqueue_index_job(
        entity_kind="node", entity_id="n_dead", index_kind="node_docs", op="UPSERT"
    )
    eng.enqueue_index_job(
        entity_kind="node", entity_id="n_dead", index_kind="node_refs", op="UPSERT"
    )
    eng.reconcile_indexes(max_jobs=10)

    assert len(_ids(eng.backend.node_docs_get(where={"node_id": "n_dead"}))) >= 1

    # Tombstone base record
    assert eng.tombstone_node("n_dead") is True

    # Even if an UPSERT is enqueued, reconcile should delete derived rows instead of rebuilding.
    eng.enqueue_index_job(
        entity_kind="node", entity_id="n_dead", index_kind="node_docs", op="UPSERT"
    )
    eng.enqueue_index_job(
        entity_kind="node", entity_id="n_dead", index_kind="node_refs", op="UPSERT"
    )
    eng.reconcile_indexes(max_jobs=10)

    assert len(_ids(eng.backend.node_docs_get(where={"node_id": "n_dead"}))) == 0
    assert len(_ids(eng.backend.node_refs_get(where={"node_id": "n_dead"}))) == 0


def test_stuck_doing_job_is_stealable_after_lease_expiry(
    e2e_engine: GraphKnowledgeEngine,
):
    """Covers the 'halted forever' scenario: DOING with expired lease must be reclaimed."""
    eng = e2e_engine
    _assert_backend_kind(eng)
    import re

    eng.add_node(_mk_node("n_stuck", doc_id="d1"))
    jid = eng.enqueue_index_job(
        entity_kind="node", entity_id="n_stuck", index_kind="node_docs", op="UPSERT"
    )

    # Force the job into DOING with an expired lease to simulate a crashed worker.
    # (We do it at the metastore level; this is not monkeypatching runtime logic.)
    if hasattr(eng.meta_sqlite, "transaction"):
        if isinstance(eng.meta_sqlite, EnginePostgresMetaStore):
            with eng.meta_sqlite.transaction() as conn:
                # PG uses per-test schema; schema-qualify the table.
                schema = eng.meta_sqlite.schema
                table = getattr(eng.meta_sqlite, "index_jobs_table", "index_jobs")
                # Very defensive, keep schema/table safe even though fixtures should already sanitize schema.
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema):
                    raise AssertionError(f"invalid schema in test: {schema!r}")
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
                    raise AssertionError(f"invalid table in test: {table!r}")
                ij = f"{schema}.{table}"
                conn.execute(
                    sa.text(
                        f"UPDATE {ij} "
                        "SET status='DOING', "
                        "    lease_until=NOW() - (:secs || ' seconds')::interval, "
                        "    updated_at=NOW() "
                        "WHERE job_id=:job_id"
                    ),
                    {"secs": 10, "job_id": jid},
                )
        elif hasattr(eng.meta_sqlite, "_debug_force_job_lease"):
            # Fake/in-memory meta uses a transaction view, not a DB connection.
            with eng.meta_sqlite.transaction() as txn:
                set_index_job_state(
                    eng.meta_sqlite,
                    txn,
                    job_id=jid,
                    status="DOING",
                    lease_until=0,
                    updated_at=int(time.time()),
                )
        else:
            with eng.meta_sqlite.transaction() as conn:
                if isinstance(conn, sqlite3.Connection):
                    set_index_job_state(
                        eng.meta_sqlite,
                        conn,
                        job_id=jid,
                        status="DOING",
                        lease_until=int(time.time()) - 10,
                        updated_at=int(time.time()),
                    )
                else:
                    # SQLite-like in-memory fallback.
                    set_index_job_state(
                        eng.meta_sqlite,
                        conn,
                        job_id=jid,
                        status="DOING",
                        lease_until=int(time.time()) - 10,
                        updated_at=int(time.time()),
                    )

    eng.reconcile_indexes(max_jobs=10, lease_seconds=5)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids
    assert len(_ids(eng.backend.node_docs_get(where={"node_id": "n_stuck"}))) >= 1


def test_reconcile_edge_endpoints_is_noop_when_projection_already_synced(
    e2e_engine: GraphKnowledgeEngine,
    monkeypatch: pytest.MonkeyPatch,
):
    eng = e2e_engine
    _assert_backend_kind(eng)

    n1 = _mk_node("n_sync_1", doc_id="d_sync")
    n2 = _mk_node("n_sync_2", doc_id="d_sync")
    eng.add_node(n1)
    eng.add_node(n2)

    e1 = _mk_edge("e_sync", src=n1.safe_get_id(), tgt=n2.safe_get_id(), doc_id="d_sync")
    eng.add_edge(e1)

    delete_calls: list[dict] = []
    add_calls: list[dict] = []
    upsert_calls: list[dict] = []

    orig_delete = eng.backend.edge_endpoints_delete
    orig_add = eng.backend.edge_endpoints_add
    orig_upsert = eng.backend.edge_endpoints_upsert

    def _spy_delete(*args, **kwargs):
        delete_calls.append({"args": args, "kwargs": kwargs})
        return orig_delete(*args, **kwargs)

    def _spy_add(*args, **kwargs):
        add_calls.append({"args": args, "kwargs": kwargs})
        return orig_add(*args, **kwargs)

    def _spy_upsert(*args, **kwargs):
        upsert_calls.append({"args": args, "kwargs": kwargs})
        return orig_upsert(*args, **kwargs)

    monkeypatch.setattr(eng.backend, "edge_endpoints_delete", _spy_delete)
    monkeypatch.setattr(eng.backend, "edge_endpoints_add", _spy_add)
    monkeypatch.setattr(eng.backend, "edge_endpoints_upsert", _spy_upsert)

    eng.enqueue_index_job(
        entity_kind="edge",
        entity_id=e1.safe_get_id(),
        index_kind="edge_endpoints",
        op="UPSERT",
    )
    processed = eng.reconcile_indexes(max_jobs=10)

    assert processed == 1
    assert delete_calls == []
    assert add_calls == []
    assert upsert_calls == []
