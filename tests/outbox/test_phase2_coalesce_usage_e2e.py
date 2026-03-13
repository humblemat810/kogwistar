import time
import pathlib
import re

import pytest
import sqlalchemy as sa

from graph_knowledge_engine.engine_core.chroma_backend import ChromaBackend
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend
from graph_knowledge_engine.engine_core.engine_postgres_meta import (
    EnginePostgresMetaStore,
)
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Node, Grounding, Span
from tests.conftest import FakeEmbeddingFunction


EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return sp


def _mk_node(node_id: str, *, doc_id: str) -> Node:
    return Node(
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


def _ids(coll_get: dict) -> list[str]:
    return list(coll_get.get("ids") or [])


@pytest.fixture(params=["chroma", "pg"], ids=["chroma", "pg"])
def e2e_engine(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    sa_engine,  # provided by tests/conftest.py
    pg_schema,  # provided by tests/conftest.py
) -> GraphKnowledgeEngine:
    """Run the same Phase-2 E2E usage tests against both backends.

    Mirrors the Phase-1 E2E style: identical test code must pass for:
      - `chroma`
      - `pg` (PgVectorBackend)

    Kept local to this module so Phase-1 tests stay unchanged.
    """
    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(
            persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING
        )
    else:
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
        eng = GraphKnowledgeEngine(backend=backend, embedding_function=TEST_EMBEDDING)

    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def _assert_backend_kind(eng: GraphKnowledgeEngine) -> None:
    kind = getattr(eng, "_test_backend_kind", None)
    if kind == "pg":
        assert isinstance(eng.backend, PgVectorBackend)
    else:
        assert isinstance(eng.backend, ChromaBackend)


def test_phase2_coalescing_sample_usage(e2e_engine: GraphKnowledgeEngine):
    """Sample usage: enqueue repeatedly, but only one PENDING job exists.

    Intended usage pattern:

    - A "hot" entity (node/edge) may be updated frequently.
    - Call `enqueue_index_job(...)` each time you update base state.
    - Phase 2 coalescing collapses repeated enqueues into **one** PENDING job per:
        (entity_kind, entity_id, index_kind)

    This keeps the outbox bounded and ensures the derived "join" indexes converge.
    """
    eng = e2e_engine
    _assert_backend_kind(eng)

    # 1) Base write (normal engine usage)
    eng.add_node(_mk_node("n_hot", doc_id="d1"))

    # 2) Many triggers happen quickly (hot entity)
    job_ids = [
        eng.enqueue_index_job(
            entity_kind="node", entity_id="n_hot", index_kind="node_docs", op="UPSERT"
        )
        for _ in range(10)
    ]

    # Coalescing: all calls should return the SAME effective job_id (the pending row's job_id).
    assert len(set(job_ids)) == 1
    jid = job_ids[0]

    pending = eng.meta_sqlite.list_index_jobs(
        status="PENDING", entity_kind="node", entity_id="n_hot", index_kind="node_docs"
    )
    assert len(pending) == 1
    assert pending[0].job_id == jid

    # 3) Drain the outbox (can be background worker, or fast-path call)
    eng.reconcile_indexes(max_jobs=10)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids

    # Sanity: join-like derived rows exist
    got = eng.backend.node_docs_get(where={"node_id": "n_hot"})
    assert len(_ids(got)) >= 1


def test_phase2_enqueue_while_doing_creates_new_pending(
    e2e_engine: GraphKnowledgeEngine,
):
    """Sample usage: enqueue while DOING should create a NEW PENDING job.

    Scenario:

    - J1 is already DOING with a *valid* lease (active worker).
    - Another base update happens for the same (entity_kind, entity_id, index_kind).
    - Enqueue again -> should create J2 (PENDING), because uniqueness is enforced only
      for PENDING rows.

    This avoids losing updates that arrive while a worker is mid-flight.
    """
    eng = e2e_engine
    _assert_backend_kind(eng)

    eng.add_node(_mk_node("n_busy", doc_id="d1"))

    jid1 = eng.enqueue_index_job(
        entity_kind="node", entity_id="n_busy", index_kind="node_docs", op="UPSERT"
    )

    # Force J1 into DOING with a *future* lease_until to simulate an active worker.
    if hasattr(eng.meta_sqlite, "transaction"):
        with eng.meta_sqlite.transaction() as conn:
            if isinstance(eng.meta_sqlite, EnginePostgresMetaStore):
                schema = eng.meta_sqlite.schema
                table = getattr(eng.meta_sqlite, "index_jobs_table", "index_jobs")
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema):
                    raise AssertionError(f"invalid schema in test: {schema!r}")
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
                    raise AssertionError(f"invalid table in test: {table!r}")
                ij = f"{schema}.{table}"
                conn.execute(
                    sa.text(
                        f"UPDATE {ij} "
                        "SET status='DOING', "
                        "    lease_until=NOW() + (:secs || ' seconds')::interval, "
                        "    updated_at=NOW() "
                        "WHERE job_id=:job_id"
                    ),
                    {"secs": 60, "job_id": jid1},
                )
            else:
                conn.execute(
                    "UPDATE index_jobs SET status='DOING', lease_until=?, updated_at=? WHERE job_id=?",
                    (time.time() + 60.0, time.time(), jid1),
                )

    # Enqueue again while J1 is DOING.
    jid2 = eng.enqueue_index_job(
        entity_kind="node", entity_id="n_busy", index_kind="node_docs", op="UPSERT"
    )
    assert jid2 != jid1

    # Exactly one PENDING for that key, plus one DOING.
    pending = eng.meta_sqlite.list_index_jobs(
        status="PENDING", entity_kind="node", entity_id="n_busy", index_kind="node_docs"
    )
    doing = eng.meta_sqlite.list_index_jobs(
        status="DOING", entity_kind="node", entity_id="n_busy", index_kind="node_docs"
    )
    assert len(pending) == 1 and pending[0].job_id == jid2
    assert len(doing) == 1 and doing[0].job_id == jid1

    # Drain should process the new pending job.
    eng.reconcile_indexes(max_jobs=10)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid2 in done_ids

    got = eng.backend.node_docs_get(where={"node_id": "n_busy"})
    assert len(_ids(got)) >= 1
