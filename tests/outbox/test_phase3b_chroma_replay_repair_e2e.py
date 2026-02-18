import pathlib
import uuid

import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Node, Edge, Grounding, Span

EMBEDDING_DIM = 3


def _emb(*_a, **_kw):
    return [0.1] * EMBEDDING_DIM


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return sp


def _mk_node(node_id: str, *, doc_id: str, label: str) -> Node:
    return Node(
        id=node_id,
        label=label,
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

def _mk_edge(edge_id: str, *, doc_id: str, label: str, source_id: str, target_id: str) -> Edge:
    return Edge(
        id=edge_id,
        label=label,
        type="relationship",
        summary=f"Summary {edge_id}",
        relation="related_to",
        source_ids=[source_id],
        target_ids=[target_id],
        source_edge_ids=[],
        target_edge_ids=[],
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"level_from_root": 0, "entity_type": "kg_edge"},
        embedding=[0.1] * EMBEDDING_DIM,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


@pytest.fixture()
def chroma_engine(tmp_path: pathlib.Path) -> GraphKnowledgeEngine:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
    eng._ef._emb = _emb
    eng._test_backend_kind = "chroma"  # type: ignore[attr-defined]
    return eng


def test_phase3b_chroma_replay_repair_overwrites_tampered_row(chroma_engine: GraphKnowledgeEngine):
    """Phase 3b: repair replay must overwrite tampered materialized state.

    Scenario:
      1) Add node -> event log has the canonical payload
      2) Delete from backend, then re-add a *wrong* payload directly (no new event)
      3) replay_repair_namespace() must restore the canonical payload

    This specifically targets Chroma's "add is not upsert" behavior.
    """

    eng = chroma_engine
    ns = f"phase3b_repair_{uuid.uuid4().hex}"
    eng.namespace = ns

    nid = "n_repair_1"
    eng.add_node(_mk_node(nid, doc_id="d1", label="Correct Label"))

    got1 = eng.backend.node_get(ids=[nid], include=["documents", "metadatas"])
    assert got1.get("ids") and nid in got1["ids"]

    # Simulate tamper: delete then re-add wrong row directly to the backend (no new entity_event).
    eng.backend.node_delete(ids=[nid])

    wrong = _mk_node(nid, doc_id="d1", label="WRONG LABEL")
    doc = wrong.model_dump_json(field_mode="backend", exclude=["embedding"])
    # mimic add_node metadata as best as we can (label is the key assertion)
    eng.backend.node_add(
        ids=[nid],
        documents=[doc],
        embeddings=[wrong.embedding] if wrong.embedding is not None else [eng._iterative_defensive_emb(doc)],
        metadatas=[{"label": wrong.label, "doc_id": "d1"}],
    )

    got2 = eng.backend.node_get(ids=[nid], include=["metadatas"])
    meta2 = (got2.get("metadatas") or [None])[0] or {}
    assert meta2.get("label") == "WRONG LABEL"

    # Repair by replay (forced overwrite)
    last_seq = eng.replay_repair_namespace(namespace=ns, apply_indexes=True)
    assert last_seq >= 1

    got3 = eng.backend.node_get(ids=[nid], include=["metadatas"])
    meta3 = (got3.get("metadatas") or [None])[0] or {}
    assert meta3.get("label") == "Correct Label"



def test_phase3b_chroma_normal_replay_does_not_overwrite_tamper(chroma_engine: GraphKnowledgeEngine):
    """Normal replay_namespace() is *not* a repair mechanism for Chroma.

    If the materialized row is tampered (but still exists), Chroma's 'add' won't upsert,
    so replay_namespace() should leave the tampered row unchanged.

    This test documents the intentional divergence:
      - replay_namespace(): re-apply events, best-effort, no forced overwrite
      - replay_repair_namespace(): delete+reapply (forced overwrite)
    """

    eng = chroma_engine
    ns = f"phase3b_normal_{uuid.uuid4().hex}"
    eng.namespace = ns

    nid = "n_normal_1"
    eng.add_node(_mk_node(nid, doc_id="d1", label="Correct Label"))

    upd = getattr(eng.backend, "node_update", None)
    if upd is None:
        pytest.skip("backend.node_update not available; cannot simulate in-place tamper")

    upd(ids=[nid], metadatas=[{"label": "TAMPERED", "doc_id": "d1"}])

    got2 = eng.backend.node_get(ids=[nid], include=["metadatas"])
    meta2 = (got2.get("metadatas") or [None])[0] or {}
    assert meta2.get("label") == "TAMPERED"

    eng.replay_namespace(namespace=ns, apply_indexes=False)

    got3 = eng.backend.node_get(ids=[nid], include=["metadatas"])
    meta3 = (got3.get("metadatas") or [None])[0] or {}
    assert meta3.get("label") == "TAMPERED"


def test_phase3b_chroma_replay_repair_is_idempotent(chroma_engine: GraphKnowledgeEngine):
    """repair replay can be safely re-run (idempotent in outcome)."""

    eng = chroma_engine
    ns = f"phase3b_idem_{uuid.uuid4().hex}"
    eng.namespace = ns

    nid = "n_idem_1"
    eng.add_node(_mk_node(nid, doc_id="d1", label="Correct Label"))

    upd = getattr(eng.backend, "node_update", None)
    if upd is None:
        pytest.skip("backend.node_update not available; cannot simulate in-place tamper")
    upd(ids=[nid], metadatas=[{"label": "TAMPERED", "doc_id": "d1"}])

    eng.replay_repair_namespace(namespace=ns, apply_indexes=False)
    eng.replay_repair_namespace(namespace=ns, apply_indexes=False)

    got = eng.backend.node_get(ids=[nid], include=["metadatas"])
    meta = (got.get("metadatas") or [None])[0] or {}
    assert meta.get("label") == "Correct Label"


def test_phase3b_chroma_replay_repair_overwrites_tampered_edge(chroma_engine: GraphKnowledgeEngine):
    """Phase 3b edge parity: repair replay should overwrite tampered edge rows too."""

    eng = chroma_engine
    ns = f"phase3b_edge_{uuid.uuid4().hex}"  # unique per run
    eng.namespace = ns

    eng.add_node(_mk_node("n_src", doc_id="d1", label="SRC"))
    eng.add_node(_mk_node("n_tgt", doc_id="d1", label="TGT"))

    eid = "e1"
    e = _mk_edge(eid, doc_id="d1", label="Correct Edge", source_id="n_src", target_id="n_tgt")
    eng.add_edge(e)

    got1 = eng.backend.edge_get(ids=[eid], include=["metadatas"])
    assert got1.get("ids") and eid in got1["ids"]

    eupd = getattr(eng.backend, "edge_update", None)
    if eupd is None:
        pytest.skip("backend.edge_update not available; cannot simulate in-place edge tamper")
    doc, meta = eng._edge_doc_and_meta(e)
    # tamper document
    import json
    doc = json.loads(doc)
    doc.update({'label': "TAMPERED_EDGE"})
    meta.update({"label": "TAMPERED_EDGE", "doc_id": "d1"})
    eng.backend.edge_update(ids=[eid], documents = [json.dumps(doc)],  metadatas=[meta])

    got2 = eng.backend.edge_get(ids=[eid], include=["metadatas"])
    meta2 = (got2.get("metadatas") or [None])[0] or {}
    assert meta2.get("label") == "TAMPERED_EDGE" and eng.get_edges(ids=[eid])[0].label== "TAMPERED_EDGE"

    eng.replay_repair_namespace(namespace=ns, apply_indexes=False)

    assert eng.get_edges(ids=[eid])[0].label == "Correct Edge"
