import time
import pathlib

import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Node, Edge, Grounding, Span
from graph_knowledge_engine.postgres_backend import PgVectorBackend


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


def add_node_raw(engine: GraphKnowledgeEngine, node: Node, *, embedding_dim: int = 384) -> None:
    """Backend-agnostic 'raw' insert: write base entity only (no derived joins)."""
    doc, meta = engine._node_doc_and_meta(node)
    if node.embedding is None:
        node.embedding = [0.1] * embedding_dim
    engine.backend.node_add(
        ids=[node.safe_get_id()],
        documents=[doc],
        embeddings=[list(node.embedding)],
        metadatas=[meta],
    )


def add_edge_raw(engine: GraphKnowledgeEngine, edge: Edge, *, embedding_dim: int = 384) -> None:
    doc, meta = engine._edge_doc_and_meta(edge)
    if edge.embedding is None:
        edge.embedding = [0.1] * embedding_dim
    engine.backend.edge_add(
        ids=[edge.safe_get_id()],
        documents=[doc],
        embeddings=[list(edge.embedding)],
        metadatas=[meta],
    )


def _ids(coll_get: dict) -> list[str]:
    return list(coll_get.get("ids") or [])


@pytest.fixture(params=["chroma", "pgvector"])
def e2e_engine(request, tmp_path: pathlib.Path, sa_engine, pg_schema) -> GraphKnowledgeEngine:
    """Parametrized engine: Chroma (SQLite meta) and PgVector (Postgres meta)."""
    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        return GraphKnowledgeEngine(persist_directory=str(persist_dir))

    # pgvector
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=384, schema=pg_schema)
    return GraphKnowledgeEngine(backend=backend)


def test_reconcile_builds_all_joins_from_raw_entities(e2e_engine: GraphKnowledgeEngine):
    """E2E: if join collections are empty (crash between base write and join write), reconcile must converge."""
    eng = e2e_engine

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
    got = eng.backend.node_docs_get(where={"doc_id": "d1"}, include=["ids", "metadatas"])
    ids = _ids(got)
    assert any(i.startswith("n1::") for i in ids)
    assert any(i.startswith("n2::") for i in ids)

    # node_refs contain doc_id=d1 rows
    got = eng.backend.node_refs_get(where={"doc_id": "d1"}, include=["ids"])
    assert len(_ids(got)) >= 2

    # edge_endpoints contain src/tgt rows
    got = eng.backend.edge_endpoints_get(where={"edge_id": "e1"}, include=["ids", "metadatas"])
    ids = _ids(got)
    assert any("::src::" in i for i in ids)
    assert any("::tgt::" in i for i in ids)

    # edge_refs contain doc_id=d1
    got = eng.backend.edge_refs_get(where={"doc_id": "d1"}, include=["ids"])
    assert len(_ids(got)) >= 1


def test_race_job_before_entity_exists_recovers_on_later_insert(e2e_engine: GraphKnowledgeEngine):
    """Race: job refers to entity not yet present. It should fail, then succeed once base exists."""
    eng = e2e_engine

    jid = eng.enqueue_index_job(entity_kind="node", entity_id="n_race", index_kind="node_docs", op="UPSERT")
    eng.reconcile_indexes(max_jobs=10)

    failed = {j.job_id: j for j in eng.meta_sqlite.list_index_jobs(status="FAILED")}
    assert jid in failed

    # Later the base entity arrives
    add_node_raw(eng, _mk_node("n_race", doc_id="d1"))

    eng.reconcile_indexes(max_jobs=10)

    done_ids = {j.job_id for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids

    got = eng.backend.node_docs_get(where={"node_id": "n_race"})
    assert len(_ids(got)) >= 1


def test_tombstoned_entities_never_resurrect_derived_rows(e2e_engine: GraphKnowledgeEngine):
    eng = e2e_engine

    add_node_raw(eng, _mk_node("n_dead", doc_id="d1"))

    # First build derived rows
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_docs", op="UPSERT")
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_refs", op="UPSERT")
    eng.reconcile_indexes(max_jobs=10)

    assert len(_ids(eng.backend.node_docs_get(where={"entity_id": "n_dead"}, include=["ids"]))) >= 1

    # Tombstone base record
    assert eng.tombstone_node("n_dead") is True

    # Even if an UPSERT is enqueued, reconcile should delete derived rows instead of rebuilding.
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_docs", op="UPSERT")
    eng.enqueue_index_job(entity_kind="node", entity_id="n_dead", index_kind="node_refs", op="UPSERT")
    eng.reconcile_indexes(max_jobs=10)

    assert len(_ids(eng.backend.node_docs_get(where={"entity_id": "n_dead"}, include=["ids"]))) == 0
    assert len(_ids(eng.backend.node_refs_get(where={"entity_id": "n_dead"}, include=["ids"]))) == 0


def test_stuck_doing_job_is_stealable_after_lease_expiry(e2e_engine: GraphKnowledgeEngine):
    """Covers the 'halted forever' scenario: DOING with expired lease must be reclaimed."""
    eng = e2e_engine

    add_node_raw(eng, _mk_node("n_stuck", doc_id="d1"))
    jid = eng.enqueue_index_job(entity_kind="node", entity_id="n_stuck", index_kind="node_docs", op="UPSERT")

    # Force the job into DOING with an expired lease to simulate a crashed worker.
    # (We do it at the metastore level; this is not monkeypatching runtime logic.)
    with eng.meta_sqlite.transaction() as conn:
        conn.execute(
            "UPDATE index_jobs SET status='DOING', lease_until=?, updated_at=? WHERE job_id=?",
            (time.time() - 10.0, time.time(), jid),
        )

    eng.reconcile_indexes(max_jobs=10, lease_seconds=5)

    done_ids = {j["job_id"] for j in eng.meta_sqlite.list_index_jobs(status="DONE")}
    assert jid in done_ids
    assert len(_ids(eng.backend.node_docs_get(where={"entity_id": "n_stuck"}, include=["ids"]))) >= 1
