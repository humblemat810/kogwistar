import uuid
import pathlib
import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine, _node_doc_and_meta
from graph_knowledge_engine.engine_core.models import Node, Grounding, Span

EMBEDDING_DIM = 3

def _emb(*args, **kwargs):
    return [0.1] * EMBEDDING_DIM

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

@pytest.fixture
def chroma_engine(tmp_path: pathlib.Path) -> GraphKnowledgeEngine:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
    eng._test_backend_kind = "chroma"  # type: ignore[attr-defined]
    eng._ef._emb = _emb
    return eng


def test_phase3_chroma_replay_repairs_missing_vector_state(chroma_engine):
    """
    Chroma cannot rollback, so the Phase-3 correctness story is:
      - meta event log is durable
      - replay can repair vector/index state after partial failure

    We simulate “vector state lost” by deleting from backend directly (NO new event),
    then replay should restore it from entity_events.
    """
    eng = chroma_engine
    ns = f"phase3_repair_{uuid.uuid4().hex}"
    eng.namespace = ns

    nid = "n_repair_1"
    eng.add_node(_mk_node(nid, doc_id="d1"))

    # Ensure it exists
    got1 = eng.backend.node_get(ids=[nid], include=["metadatas", "documents"])
    assert got1.get("ids") and nid in got1["ids"]

    # Simulate “vector store lost the row” without logging a new event
    eng.backend.node_delete(ids=[nid])

    got2 = eng.backend.node_get(ids=[nid], include=["metadatas", "documents"])
    assert got2.get("ids") in ([], None) or len(got2.get("ids", [])) == 0

    # Repair by replay
    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=True)
    assert last_seq >= 1

    got3 = eng.backend.node_get(ids=[nid], include=["metadatas", "documents"])
    assert got3.get("ids") and nid in got3["ids"]

def test_phase3b_chroma_replay_repair_restores_tampered_node(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_tamper_{uuid.uuid4().hex}"
    eng.namespace = ns

    node = _mk_node("n1", doc_id="d1")
    eng.add_node(node)

    # Canonical expectation from *event payload* (embedding excluded in your event log)
    expected_label = node.label
    expected_summary = node.summary

    # Tamper backend: delete + re-add wrong content directly (no new event)
    eng.backend.node_delete(ids=["n1"])
    node = _mk_node("n1", doc_id="d1")
    node.label = "TAMPERED"
    node.summary = "TAMPERED"
    doc, meta = _node_doc_and_meta(node)
    eng.backend.node_add(
            ids=[node.safe_get_id()],
            documents=[doc],
            embeddings=[node.embedding] if node.embedding is not None else [eng._iterative_defensive_emb(str(doc))],
            metadatas=[meta],
        )
    got_tampered = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got_tampered["metadatas"][0]["label"] == "TAMPERED"

    # Repair replay (overwrite-capable)
    eng.replay_namespace(namespace=ns, apply_indexes=False, repair_backend=True)

    got_repaired = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got_repaired["metadatas"][0]["label"] == expected_label
    assert got_repaired["metadatas"][0]["summary"] == expected_summary