import json
import pathlib

import pytest

pytestmark = pytest.mark.core

from kogwistar.engine_core.models import Node, Edge
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.fake_backend import build_fake_backend
from tests.conftest import FakeEmbeddingFunction
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge

EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)

def _mk_node(node_id: str, *, doc_id: str) -> Node:
    return build_entity_node(
        node_id=node_id,
        doc_id=doc_id,
        embedding=[0.1] * EMBEDDING_DIM,
    )


def _mk_edge(edge_id: str, src: str, tgt: str, doc_id: str) -> Edge:
    return build_relationship_edge(
        edge_id=edge_id,
        src=src,
        tgt=tgt,
        doc_id=doc_id,
        embedding=[0.1] * EMBEDDING_DIM,
        source_edge_ids=None,
        target_edge_ids=None,
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
    """Run the same Phase-2 E2E usage tests against both backends.

    Mirrors the Phase-1 E2E style: identical test code must pass for:
      - `chroma`
      - `pg` (PgVectorBackend)

    Kept local to this module so Phase-1 tests stay unchanged.
    """
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
            persist_directory=str(persist_dir),
            embedding_function=TEST_EMBEDDING,
        )
    else:
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
        if sa_engine is None or pg_schema is None:
            pytest.skip("PostgreSQL fixtures are unavailable")
        pytest.importorskip("pgvector")
        from kogwistar.engine_core.postgres_backend import PgVectorBackend

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


def test_node_projection_failure_leaves_recoverable_canonical_event(tmp_path, monkeypatch):
    """Canonical admission precedes an external backend projection write."""
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=TEST_EMBEDDING,
        backend_factory=build_fake_backend,
    )

    async def fail_node_add(**_kwargs):
        raise RuntimeError("projection unavailable")

    monkeypatch.setattr(eng.backend, "node_add", fail_node_add)
    with pytest.raises(RuntimeError, match="projection unavailable"):
        eng.add_node(_mk_node("canonical-before-projection", doc_id="d1"))

    events = _read_all_events(eng, "default")
    assert any(
        entity_kind == "node" and entity_id == "canonical-before-projection" and op == "ADD"
        for _, entity_kind, entity_id, op, _ in events
    )


def test_node_event_append_failure_does_not_create_projection(tmp_path, monkeypatch):
    """A node is never projection-visible when required canonical append fails."""
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=TEST_EMBEDDING,
        backend_factory=build_fake_backend,
    )

    def fail_append(**_kwargs):
        raise OSError("event store unavailable")

    monkeypatch.setattr(eng.meta_sqlite, "append_entity_event", fail_append)
    with pytest.raises(OSError, match="event store unavailable"):
        eng.add_node(_mk_node("no-projection-without-event", doc_id="d1"))

    assert {node.safe_get_id() for node in eng.get_nodes()} == set()


def test_lifecycle_event_append_failure_does_not_patch_projection(tmp_path, monkeypatch):
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=TEST_EMBEDDING,
        backend_factory=build_fake_backend,
    )
    eng.write.add_node(_mk_node("lifecycle-event-required", doc_id="d1"))

    def fail_append(**_kwargs):
        raise OSError("event store unavailable")

    monkeypatch.setattr(eng.meta_sqlite, "append_entity_event", fail_append)
    with pytest.raises(OSError, match="event store unavailable"):
        eng.lifecycle.tombstone_node("lifecycle-event-required")

    raw = eng.backend.node_get(ids=["lifecycle-event-required"], include=["metadatas"])
    assert raw["metadatas"][0].get("lifecycle_status") != "tombstoned"


def test_lifecycle_projection_failure_leaves_recoverable_tombstone_event(
    tmp_path, monkeypatch
):
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=TEST_EMBEDDING,
        backend_factory=build_fake_backend,
    )
    eng.write.add_node(_mk_node("tombstone-before-projection", doc_id="d1"))

    original_lifecycle_update = eng._backend_update_record_lifecycle
    monkeypatch.setattr(
        eng,
        "_backend_update_record_lifecycle",
        lambda **_kwargs: False,
    )
    assert eng.lifecycle.tombstone_node("tombstone-before-projection") is False

    assert any(
        entity_kind == "node"
        and entity_id == "tombstone-before-projection"
        and op == "TOMBSTONE"
        and "lifecycle_patch" in json.loads(payload_json)
        for _, entity_kind, entity_id, op, payload_json in _read_all_events(eng, "default")
    )

    monkeypatch.setattr(
        eng, "_backend_update_record_lifecycle", original_lifecycle_update
    )
    eng.replay_namespace(namespace="default")
    raw = eng.backend.node_get(
        ids=["tombstone-before-projection"], include=["metadatas"]
    )
    assert raw["metadatas"][0]["lifecycle_status"] == "tombstoned"


def test_redirect_is_recorded_as_replayable_lifecycle_replace(tmp_path):
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=TEST_EMBEDDING,
        backend_factory=build_fake_backend,
    )
    eng.write.add_node(_mk_node("redirect-source", doc_id="d1"))
    eng.write.add_node(_mk_node("redirect-target", doc_id="d1"))

    assert eng.lifecycle.redirect_node("redirect-source", "redirect-target")
    events = _read_all_events(eng, "default")
    assert any(
        entity_kind == "node"
        and entity_id == "redirect-source"
        and op == "REPLACE"
        and json.loads(payload_json)["lifecycle_patch"]["redirect_to_id"]
        == "redirect-target"
        for _, entity_kind, entity_id, op, payload_json in events
    )

    eng.replay_namespace(namespace="default")
    raw = eng.backend.node_get(ids=["redirect-source"], include=["metadatas"])
    assert raw["metadatas"][0]["redirect_to_id"] == "redirect-target"
