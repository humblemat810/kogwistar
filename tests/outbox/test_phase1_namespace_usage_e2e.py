import pathlib

import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend

from graph_knowledge_engine.engine_core.models import Node, Edge, Grounding, Span


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return sp


EMBEDDING_DIM = 3

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


@pytest.fixture(params=["chroma", "pgvector"], ids=["chroma", "pgvector"])
def e2e_engine(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    sa_engine,  # provided by tests/conftest.py
    pg_schema,  # provided by tests/conftest.py
) -> GraphKnowledgeEngine:
    """Run the same tests against both backends (Chroma+SQLite meta and PGVector+PG meta)."""

    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
    else:
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
        eng = GraphKnowledgeEngine(backend=backend)

    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def test_namespace_scopes_index_jobs_and_reconcile(e2e_engine: GraphKnowledgeEngine):
    """Sample usage: multi-tenant/namespace isolation for index_jobs.

    Why this exists
    --------------
    In Phase 1, join-like derived indexes (node_docs/node_refs/edge_refs/edge_endpoints)
    converge via an outbox table (index_jobs). As soon as you introduce a runtime layer
    (workflow steps, per-user agents, etc.), you'll want a cheap way to segregate drain
    pressure and "must be current" checks by tenant/user/conversation.

    Design
    ------
    - index_jobs now has a `namespace` column (default: "default").
    - enqueue_index_job(...) and reconcile_indexes(...) accept an optional namespace.
    - claim_index_jobs(...) and list_index_jobs(...) are namespace-aware.

    Guarantee
    ---------
    Reconciling namespace A will NOT steal or apply jobs from namespace B.
    """

    eng = e2e_engine

    # Base write (same physical node table for now; namespace is for the outbox / drains).
    eng.add_node(_mk_node("n_ns", doc_id="d_ns"))

    # Same derived-index work, two namespaces.
    eng.enqueue_index_job(entity_kind="node", entity_id="n_ns", index_kind="node_docs", op="UPSERT", namespace="ns_a")
    eng.enqueue_index_job(entity_kind="node", entity_id="n_ns", index_kind="node_docs", op="UPSERT", namespace="ns_b")

    jobs_a = eng.meta_sqlite.list_index_jobs(namespace="ns_a")
    jobs_b = eng.meta_sqlite.list_index_jobs(namespace="ns_b")
    assert len(jobs_a) == 1
    assert len(jobs_b) == 1

    # Drain only ns_a
    applied_a = eng.reconcile_indexes(max_jobs=10, namespace="ns_a")
    assert applied_a == 1

    # ns_b job should still be pending
    pending_b = eng.meta_sqlite.list_index_jobs(namespace="ns_b", status="PENDING")
    assert len(pending_b) == 1

    applied_b = eng.reconcile_indexes(max_jobs=10, namespace="ns_b")
    assert applied_b == 1

    pending_b2 = eng.meta_sqlite.list_index_jobs(namespace="ns_b", status="PENDING")
    assert len(pending_b2) == 0
