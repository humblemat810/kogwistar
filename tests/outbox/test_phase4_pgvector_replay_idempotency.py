import pytest

from graph_knowledge_engine.engine_core.models import Node, Edge, Grounding, Span
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
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


@pytest.mark.integration
def test_pgvector_replay_is_idempotent_no_new_events(pg_engine):
    eng = pg_engine
    ns = getattr(eng, "namespace", "default")

    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    # Baseline: capture events + current projection snapshot (IDs + metadatas)
    events_before = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events_before) >= 3

    snap0 = eng.backend.node_get(ids=["n1", "n2"], include=["metadatas", "documents"])
    md0 = {i: m for i, m in zip(snap0.get("ids") or [], snap0.get("metadatas") or [])}

    # Replay once (should be no-op logically)
    last1 = eng.replay_namespace(namespace=ns, apply_indexes=True, repair_backend=False)
    assert last1 >= 3

    # Replay twice (idempotent)
    last2 = eng.replay_namespace(namespace=ns, apply_indexes=True, repair_backend=False)
    assert last2 == last1

    # No new truth should be appended by replay
    events_after = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events_after) == len(events_before)

    # Projection stable
    snap1 = eng.backend.node_get(ids=["n1", "n2"], include=["metadatas", "documents"])
    md1 = {i: m for i, m in zip(snap1.get("ids") or [], snap1.get("metadatas") or [])}
    assert md1["n1"].get("entity_type") == "kg_entity"
    assert md1["n2"].get("entity_type") == "kg_entity"
    # at least ensure keys we care about didn't disappear
    assert md1["n1"].get("level_from_root") == md0["n1"].get("level_from_root")
    assert md1["n2"].get("level_from_root") == md0["n2"].get("level_from_root")
