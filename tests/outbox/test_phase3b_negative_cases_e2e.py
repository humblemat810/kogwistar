import uuid
import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core import models
from tests.conftest import FakeEmbeddingFunction

EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)


def _mk_node(node_id: str, *, doc_id: str) -> models.Node:
    sp = models.Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return models.Node(
        id=node_id,
        label=f"Node {node_id}",
        type="entity",
        summary=f"Summary {node_id}",
        doc_id=doc_id,
        mentions=[models.Grounding(spans=[sp])],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        embedding=[0.1] * EMBEDDING_DIM,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _mk_edge(edge_id: str, src: str, dst: str, *, doc_id: str) -> models.Edge:
    sp = models.Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return models.Edge(
        id=edge_id,
        label=f"Edge {edge_id}",
        type="relationship",
        summary=f"Summary {edge_id}",
        doc_id=doc_id,
        mentions=[models.Grounding(spans=[sp])],
        metadata={"edge_type": "rel"},
        embedding=[0.1] * EMBEDDING_DIM,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        # EdgeMixin fields (current schema)
        source_ids=[src],
        target_ids=[dst],
        relation="rel",
        source_edge_ids=None,
        target_edge_ids=None,
    )


@pytest.fixture
def chroma_engine(tmp_path) -> GraphKnowledgeEngine:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    eng = GraphKnowledgeEngine(
        persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING
    )
    return eng


def _count_events(eng: GraphKnowledgeEngine, ns: str) -> int:
    return sum(1 for _ in eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))


def _insert_raw_event(
    eng: GraphKnowledgeEngine, ns: str, *, ek: str, eid: str, op: str, payload_json: str
):
    eng.meta_sqlite.append_entity_event(
        namespace=ns,
        event_id=f"ev_{uuid.uuid4().hex}",
        entity_kind=ek,
        entity_id=eid,
        op=op,
        payload_json=payload_json,
    )


def _repair_replay(eng: GraphKnowledgeEngine, ns: str):
    # Supports either replay_namespace(..., repair_backend=True) or wrapper replay_repair_namespace
    if "repair_backend" in eng.replay_namespace.__code__.co_varnames:
        eng.replay_namespace(namespace=ns, apply_indexes=False, repair_backend=True)
    else:
        eng.replay_repair_namespace(namespace=ns, apply_indexes=False)  # type: ignore[attr-defined]


# (2) Normal replay does NOT overwrite tampered-but-present rows (Chroma add semantics)
def test_phase3b_chroma_normal_replay_does_not_overwrite_tampered(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_norm_nooverwrite_{uuid.uuid4().hex}"
    eng.namespace = ns

    node = _mk_node("n1", doc_id="d1")
    eng.add_node(node)

    # Tamper WITHOUT new events: disable event log, delete+readd wrong content
    prev = getattr(eng, "_disable_event_log", False)
    eng._disable_event_log = True
    try:
        eng.backend.node_delete(ids=["n1"])
        wrong = _mk_node("n1", doc_id="d1")
        wrong.label = "TAMPERED"
        wrong.summary = "TAMPERED"
        eng.add_node(wrong)  # event log disabled, so this is "out-of-band"
    finally:
        eng._disable_event_log = prev

    got = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got["metadatas"][0]["label"] == "TAMPERED"

    eng.replay_namespace(namespace=ns, apply_indexes=False)

    got2 = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got2["metadatas"][0]["label"] == "TAMPERED"


# (3) Repair replay is idempotent (Chroma)
def test_phase3b_chroma_repair_replay_is_idempotent(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_repair_idem_{uuid.uuid4().hex}"
    eng.namespace = ns

    node = _mk_node("n1", doc_id="d1")
    eng.add_node(node)

    # Tamper WITHOUT new events
    prev = getattr(eng, "_disable_event_log", False)
    eng._disable_event_log = True
    try:
        eng.backend.node_delete(ids=["n1"])
        wrong = _mk_node("n1", doc_id="d1")
        wrong.label = "TAMPERED"
        eng.add_node(wrong)
    finally:
        eng._disable_event_log = prev

    _repair_replay(eng, ns)
    got = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got["metadatas"][0]["label"] == node.label

    _repair_replay(eng, ns)
    got2 = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got2["metadatas"][0]["label"] == node.label


# (4) Edge repair semantics (Chroma)
def test_phase3b_chroma_repair_replay_repairs_tampered_edge(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_edge_repair_{uuid.uuid4().hex}"
    eng.namespace = ns

    eng.add_node(_mk_node("a", doc_id="d1"))
    eng.add_node(_mk_node("b", doc_id="d1"))

    edge = _mk_edge("e1", "a", "b", doc_id="d1")
    eng.add_edge(edge)

    prev = getattr(eng, "_disable_event_log", False)
    eng._disable_event_log = True
    try:
        eng.backend.edge_delete(ids=["e1"])
        wrong = _mk_edge("e1", "a", "b", doc_id="d1")
        wrong.label = "TAMPERED"
        eng.add_edge(wrong)
    finally:
        eng._disable_event_log = prev
    import json

    got = eng.backend.edge_get(ids=["e1"], include=["metadatas", "documents"])
    assert json.loads(got["documents"][0])["label"] == "TAMPERED"

    _repair_replay(eng, ns)

    got2 = eng.backend.edge_get(ids=["e1"], include=["metadatas", "documents"])
    assert json.loads(got2["documents"][0])["label"] == edge.label


# Corrupt payload policy
# Why not both? You *can* test both if the engine exposes two modes (e.g., on_corrupt="halt"/"skip").
# If your engine only has one policy today, the other test is skipped.
def test_phase3b_replay_corrupt_payload_halts(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_corrupt_halt_{uuid.uuid4().hex}"

    _insert_raw_event(
        eng, ns, ek="node", eid="n_bad", op="ADD", payload_json="{not-json"
    )

    # Either it raises, or it must explicitly skip. Default should be "halt" unless you changed it.
    with pytest.raises(Exception):
        eng.replay_namespace(namespace=ns, apply_indexes=False)


def test_phase3b_replay_corrupt_payload_can_skip_if_supported(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_corrupt_skip_{uuid.uuid4().hex}"

    _insert_raw_event(
        eng, ns, ek="node", eid="n_bad", op="ADD", payload_json="{not-json"
    )
    good = _mk_node("n_ok", doc_id="d1")
    _insert_raw_event(
        eng,
        ns,
        ek="node",
        eid="n_ok",
        op="ADD",
        payload_json=good.model_dump_json(exclude={"embedding"}),
    )

    if "on_corrupt" not in eng.replay_namespace.__code__.co_varnames:
        pytest.skip(
            "Engine has no on_corrupt=... mode; only one policy is implemented."
        )

    last = eng.replay_namespace(namespace=ns, apply_indexes=False, on_corrupt="skip")
    assert last >= 2
    got = eng.backend.node_get(ids=["n_ok"], include=["metadatas"])
    assert got.get("ids") and "n_ok" in got["ids"]


# Tombstone missing is idempotent (Chroma path)
def test_phase3b_tombstone_missing_is_idempotent(chroma_engine):
    eng = chroma_engine
    ns = f"phase3b_tombstone_missing_{uuid.uuid4().hex}"
    eng.namespace = ns

    eng.tombstone_node("missing-node-id-123")
    eng.tombstone_edge("missing-edge-id-123")


# Cross-store non-2PC expectation:
# If meta append fails after backend write, PG should rollback; Chroma likely persists.
# This test only runs for Chroma here; you likely already have PG atomicity coverage.
def test_phase3b_chroma_meta_failure_after_backend_write_persists(
    chroma_engine, monkeypatch
):
    """Chroma projection is not 2PC with meta.

    If meta append fails after the backend write, we do NOT abort the write in Chroma mode.
    Instead, the backend write persists and the system relies on later replay/repair to converge.
    """
    eng = chroma_engine
    ns = f"phase3b_meta_fail_{uuid.uuid4().hex}"
    eng.namespace = ns

    node = _mk_node("n1", doc_id="d1")

    def boom(*args, **kwargs):
        raise RuntimeError("meta append failed")

    monkeypatch.setattr(eng.meta_sqlite, "append_entity_event", boom)

    # Should NOT raise: meta failure is swallowed in Chroma mode (non-2PC).
    with eng.uow():
        eng.add_node(node)

    # Backend write persisted.
    got = eng.backend.node_get(ids=["n1"], include=["metadatas"])
    assert got.get("ids") and "n1" in got["ids"]

    # Meta event did NOT get appended (best-effort logging here).
    events = list(eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))
    assert len(events) == 0
