import time
import json
import pathlib

import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Node, Edge, Grounding, Span
from graph_knowledge_engine.storage_backend import ChromaBackend
from graph_knowledge_engine.postgres_backend import PgVectorBackend

# Reuse raw helpers to avoid duplicating Chroma plumbing.
from tests.conftest import add_node_raw, add_edge_raw


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
        embedding=None,
        level_from_root=0,
        domain_id = None,
        canonical_entity_id=None,
        properties = None
    )


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
        embedding=None,
        domain_id = None,
        canonical_entity_id=None,
        properties = None
    )


@pytest.fixture(params=["chroma", "pgvector"], ids=["chroma", "pgvector"])
def e2e_engine(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    sa_engine,  # provided by tests/conftest.py
    pg_schema,  # provided by tests/conftest.py
) -> GraphKnowledgeEngine:
    """E2E engine fixture that runs the same test suite against both backends."""

    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
    else:
        # Skip cleanly if pgvector isn't installed in this environment.
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
        eng = GraphKnowledgeEngine(backend=backend)

    # Test helper (not used by production code): lets tests assert backend type.
    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def _ids(coll_get: dict) -> list[str]:
    return list(coll_get.get("ids") or [])


def test_reconcile_builds_all_joins_from_raw_entities(e2e_engine: GraphKnowledgeEngine):
    """E2E: if join collections are empty (crash between base write and join write), reconcile must converge."""
    eng = e2e_engine

    if getattr(eng, "_test_backend_kind", None) == "pgvector":
        assert isinstance(eng.backend, PgVectorBackend)
    else:
        assert isinstance(eng.backend, ChromaBackend)

    n1 = _mk_node("n1", doc_id="d1")
    n2 = _mk_node("n2", doc_id="d1")
    add_node_raw(eng, n1)
    add_node_raw(eng, n2)

    e1 = _mk_edge("e1", src="n1", tgt="n2", doc_id="d1")
    add_edge_raw(eng, e1)

    # Enqueue the "slow path" derived index rebuilds
    for nid in ("n1", "n2"):
        eng.enqueue_index_job(entity_kind="node", entity_id=nid, index_kind="node_docs", op="UPSERT")
        eng.enqueue_index_job(entity_kind="node", entity_id=nid, index_kind="node_refs", op="UPSERT")
    eng.enqueue_index_job(entity_kind="edge", entity_id="e1", index_kind="edge_endpoints", op="UPSERT")
    eng.enqueue_index_job(entity_kind="edge", entity_id="e1", index_kind="edge_refs", op="UPSERT")

    processed = eng.reconcile_indexes(max_jobs=50)
    assert processed >= 1

    # node_docs contains n1<->d1, n2<->d1
    got = eng.backend.node_docs_get(where={"doc_id": "d1"}, include=["metadatas"])
    ids = _ids(got)
    assert any(i.startswith("n1::") for i in ids)
    assert any(i.startswith("n2::") for i in ids)

    # node_refs contain doc_id=d1 rows
    got = eng.backend.node_refs_get(where={"doc_id": "d1"})
    assert len(_ids(got)) >= 2

    # edge_endpoints contain src/tgt rows
    got = eng.backend.edge_endpoints_get(where={"edge_id": "e1"}, include=["metadatas"])
    ids = _ids(got)
    assert any("::src::" in i for i in ids)
    assert any("::tgt::" in i for i in ids)

    # edge_refs contain doc_id=d1
    got = eng.backend.edge_refs_get(where={"doc_id": "d1"})
    assert len(_ids(got)) >= 1


def test_race_job_before_entity_exists_recovers_on_later_insert(e2e_engine: GraphKnowledgeEngine):
    """If reconcile runs while a job references an entity not yet present, it should fail + retry later."""
    eng = e2e_engine

    if getattr(eng, "_test_backend_kind", None) == "pgvector":
        assert isinstance(eng.backend, PgVectorBackend)
    else:
        assert isinstance(eng.backend, ChromaBackend)

    jid = eng.enqueue_index_job(entity_kind="node", entity_id="n_race", index_kind="node_docs", op="UPSERT")
    eng.reconcile_indexes(max_jobs=10)

    failed = {j.job_id: j for j in eng.meta_sqlite.list_index_jobs(status="FAILED")}
    assert jid in failed

    # Later the base entity arrives
    new_node = _mk_node("n_race", doc_id="d1")
    add_node_raw(eng, new_node)

    eng.reconcile_indexes(max_jobs=10)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids

    # node_docs rows are keyed by node_id (not a generic entity_id)
    got = eng.backend.node_docs_get(where={"node_id": "n_race"})
    assert len(_ids(got)) >= 1


def test_tombstoned_entities_never_resurrect_derived_rows(e2e_engine: GraphKnowledgeEngine):
    eng = e2e_engine

    if getattr(eng, "_test_backend_kind", None) == "pgvector":
        assert isinstance(eng.backend, PgVectorBackend)
    else:
        assert isinstance(eng.backend, ChromaBackend)

    add_node_raw(eng, _mk_node("n_dead", doc_id="d1"))

    # First build derived rows
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_docs", op="UPSERT")
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_refs", op="UPSERT")
    eng.reconcile_indexes(max_jobs=10)

    assert len(_ids(eng.backend.node_docs_get(where={"node_id": "n_dead"}))) >= 1

    # Tombstone base record
    assert eng.tombstone_node("n_dead") is True

    # Even if an UPSERT is enqueued, reconcile should delete derived rows instead of rebuilding.
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_docs", op="UPSERT")
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_refs", op="UPSERT")
    eng.reconcile_indexes(max_jobs=10)

    assert len(_ids(eng.backend.node_docs_get(where={"node_id": "n_dead"}))) == 0
    assert len(_ids(eng.backend.node_refs_get(where={"node_id": "n_dead"}))) == 0


def test_stuck_doing_job_is_stealable_after_lease_expiry(e2e_engine: GraphKnowledgeEngine):
    """Covers the 'halted forever' scenario: DOING with expired lease must be reclaimed."""
    eng = e2e_engine

    if getattr(eng, "_test_backend_kind", None) == "pgvector":
        assert isinstance(eng.backend, PgVectorBackend)
    else:
        assert isinstance(eng.backend, ChromaBackend)

    add_node_raw(eng, _mk_node("n_stuck", doc_id="d1"))
    jid = eng.enqueue_index_job(entity_kind="node", entity_id="n_stuck", index_kind="node_docs", op="UPSERT")

    # Force the job into DOING with an expired lease to simulate a crashed worker.
    # (We do it at the metastore level; this is not monkeypatching runtime logic.)
    if hasattr(eng.meta_sqlite, "transaction"):
        with eng.meta_sqlite.transaction() as conn:
            conn.execute(
                "UPDATE index_jobs SET status='DOING', lease_until=?, updated_at=? WHERE job_id=?",
                (time.time() - 10.0, time.time(), jid),
            )

    eng.reconcile_indexes(max_jobs=10, lease_seconds=5)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids
    assert len(_ids(eng.backend.node_docs_get(where={"node_id": "n_stuck"}))) >= 1
