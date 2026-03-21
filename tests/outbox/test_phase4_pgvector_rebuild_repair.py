import json
import pytest
import sqlalchemy as sa

from kogwistar.engine_core.models import Node, Edge, Grounding, Span
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.engine_core.engine import GraphKnowledgeEngine
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


def _mk_edge(edge_id: str, src: str, tgt: str, doc_id: str) -> Edge:
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


@pytest.fixture()
def pg_engine(sa_engine, pg_schema) -> GraphKnowledgeEngine:
    pytest.importorskip("pgvector")
    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=EMBEDDING_DIM, schema=pg_schema
    )
    eng = GraphKnowledgeEngine(backend=backend, embedding_function=TEST_EMBEDDING)
    backend.ensure_schema()
    return eng


@pytest.mark.ci_full
def test_pgvector_normal_replay_overwrites_tampered_rows(
    pg_engine, sa_engine, pg_schema
):
    eng = pg_engine
    ns = getattr(eng, "namespace", "default")

    # write a tiny graph
    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    # Sanity: event log exists
    events_before = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events_before) >= 3

    # Tamper backend row directly (simulate drift / bug)
    schema = pg_schema
    with sa_engine.begin() as conn:
        conn.execute(
            sa.text(
                f'UPDATE "{schema}".gke_nodes SET metadata = CAST(:m AS JSONB) WHERE id = :id'
            ),
            {"m": json.dumps({"tampered": True}), "id": "n1"},
        )

    # Confirm tamper visible pre-replay
    got0 = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got0["metadatas"][0].get("tampered") is True

    # Normal replay should restore correct payload (pgvector uses upsert semantics)
    last_seq = eng.replay_namespace(
        namespace=ns, apply_indexes=False, repair_backend=False
    )
    assert last_seq >= 3

    got1 = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    md = got1["metadatas"][0]
    assert md.get("tampered") is None
    assert md.get("entity_type") == "kg_entity"

    # Replay must not append new events
    events_after = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events_after) == len(events_before)


@pytest.mark.ci_full
def test_pgvector_rebuild_from_event_log_after_truncate(
    pg_engine, sa_engine, pg_schema
):
    eng = pg_engine
    ns = getattr(eng, "namespace", "default")

    # write a tiny graph
    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    before_nodes = {n.safe_get_id() for n in eng.get_nodes()}
    before_edges = {e.safe_get_id() for e in eng.get_edges()}
    assert before_nodes == {"n1", "n2"}
    assert before_edges == {"e1"}

    events_before = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events_before) >= 3

    # Truncate projection tables (simulate lost projection)
    schema = pg_schema
    with sa_engine.begin() as conn:
        conn.execute(sa.text(f'TRUNCATE TABLE "{schema}".gke_edges'))
        conn.execute(sa.text(f'TRUNCATE TABLE "{schema}".gke_nodes'))

    # Verify projection empty now
    assert (
        eng.backend.node_get(ids=["n1"], include=["documents"]).get("ids") in ([], None)
        or len(eng.backend.node_get(ids=["n1"], include=["documents"]).get("ids", []))
        == 0
    )

    # Replay should rebuild missing rows
    last_seq = eng.replay_namespace(
        namespace=ns, apply_indexes=True, repair_backend=False
    )
    assert last_seq >= 3

    after_nodes = {n.safe_get_id() for n in eng.get_nodes()}
    after_edges = {e.safe_get_id() for e in eng.get_edges()}

    assert after_nodes == before_nodes
    assert after_edges == before_edges

    # Vector query should work after rebuild (embedding row exists)
    q = eng.backend.node_query(
        query_embeddings=[[0.1] * EMBEDDING_DIM],
        n_results=2,
        where={"entity_type": "kg_entity"},
        include=["distances"],
    )
    assert "n1" in q["ids"][0] and "n2" in q["ids"][0]

    # Replay must not append new events
    events_after = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events_after) == len(events_before)
