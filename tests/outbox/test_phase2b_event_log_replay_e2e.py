import pytest
import pathlib

from kogwistar.engine_core.models import Node, Edge
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
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
            persist_directory=str(persist_dir),
            embedding_function=TEST_EMBEDDING,
        )
    else:
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
        eng = GraphKnowledgeEngine(
            backend=backend,
            embedding_function=TEST_EMBEDDING,
        )

    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def test_phase2b_event_log_replay_e2e(e2e_engine):
    """Phase 2b: append-only event log + replay for node/edge primitives.

    This test runs for both backends via the existing e2e_engine fixture.
    """
    eng = e2e_engine
    ns = getattr(eng, "namespace", "default")
    # Write a tiny graph
    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    # Capture current visible state
    before_nodes = {n.safe_get_id() for n in eng.get_nodes()}
    before_edges = {e.safe_get_id() for e in eng.get_edges()}

    # Replaying into the same engine under suppression (idempotent re-apply).
    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=True)

    assert last_seq >= 3

    after_nodes = {n.safe_get_id() for n in eng.get_nodes()}
    after_edges = {e.safe_get_id() for e in eng.get_edges()}

    assert before_nodes == after_nodes
    assert before_edges == after_edges


def _read_all_events(eng: GraphKnowledgeEngine, namespace: str):
    return list(eng.meta_sqlite.iter_entity_events(namespace=namespace, from_seq=1))


def test_phase2b_event_log_no_duplicate_and_payload_sanity(e2e_engine):
    """Phase 2b: replay must NOT append new entity_events; payload must be node/edge shaped."""
    import json

    eng = e2e_engine
    ns = getattr(eng, "namespace", "default")

    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    events_before = _read_all_events(eng, ns)
    assert len(events_before) >= 3

    # Payload sanity for ADD events: must include an 'id' matching the entity_id (or at least present).
    for seq, ek, eid, op, payload_json in events_before:
        if op in ("ADD", "REPLACE"):
            payload = json.loads(payload_json)
            assert isinstance(payload, dict)
            assert "id" in payload
            assert str(payload["id"]) == str(eid)

    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=True)
    assert last_seq >= 3

    events_after = _read_all_events(eng, ns)
    assert len(events_after) == len(events_before), "Replay must not append new events"


def test_phase2b_event_log_tombstone_and_cursor_roundtrip(e2e_engine):
    """Phase 2b: tombstones must be logged and cursor_get/set must work."""
    eng = e2e_engine
    ns = getattr(eng, "namespace", "default")

    eng.add_node(_mk_node("n1", doc_id="d1"))
    eng.add_node(_mk_node("n2", doc_id="d2"))
    eng.add_edge(_mk_edge("e1", "n1", "n2", doc_id="d1"))

    # Tombstone one entity (best-effort: signature differs across versions)
    tomb = getattr(eng, "tombstone_node", None)
    assert callable(tomb)
    try:
        ok = tomb("n2", reason="phase2b-test")
    except TypeError:
        ok = tomb("n2")
    assert ok is True or ok is None  # some implementations return None

    events = _read_all_events(eng, ns)
    # Expect at least one TOMBSTONE for node n2
    assert any(
        (ek == "node" and eid == "n2" and op in ("TOMBSTONE", "DELETE"))
        for _, ek, eid, op, _ in events
    )

    # Replay should preserve tombstone state and not double events
    before_n = len(events)
    last_seq = eng.replay_namespace(namespace=ns, apply_indexes=False)
    assert last_seq >= 4

    after_events = _read_all_events(eng, ns)
    assert len(after_events) == before_n

    # Cursor set/get smoke
    consumer = "phase2b-test-consumer"
    eng.meta_sqlite.cursor_set(namespace=ns, consumer=consumer, last_seq=last_seq)
    got = eng.meta_sqlite.cursor_get(namespace=ns, consumer=consumer)
    assert int(got) == int(last_seq)
