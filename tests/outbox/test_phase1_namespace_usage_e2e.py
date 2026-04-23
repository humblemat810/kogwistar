import pathlib

import pytest
pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.fake_backend import build_fake_backend

from kogwistar.engine_core.models import Node
from tests.conftest import FakeEmbeddingFunction
from tests._helpers.graph_builders import build_entity_node

EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)


def _mk_node(node_id: str, *, doc_id: str) -> Node:
    return build_entity_node(
        node_id=node_id,
        doc_id=doc_id,
        embedding=[0.1] * EMBEDDING_DIM,
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
    """Run the same tests against both selectors; `pg` uses PgVectorBackend + PG meta."""

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
        pytest.importorskip("pgvector")
        from kogwistar.engine_core.postgres_backend import PgVectorBackend
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
        eng = GraphKnowledgeEngine(backend=backend, embedding_function=TEST_EMBEDDING)

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
    eng.enqueue_index_job(
        entity_kind="node",
        entity_id="n_ns",
        index_kind="node_docs",
        op="UPSERT",
        namespace="ns_a",
    )
    eng.enqueue_index_job(
        entity_kind="node",
        entity_id="n_ns",
        index_kind="node_docs",
        op="UPSERT",
        namespace="ns_b",
    )

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
