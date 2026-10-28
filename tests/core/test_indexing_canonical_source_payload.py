from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.engine_core.rust_postgres_session import RustEnginePostgresMetaStore
from kogwistar.implementation_selection import capture_implementation_selection
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node


def test_canonical_source_payload_distinguishes_active_tombstoned_and_missing(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path), backend_factory=build_fake_backend
    )
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=[0.1, 0.2, 0.3])
    engine.write.add_node(node)

    active = json.loads(
        engine.indexing.canonical_source_payload(entity_kind="node", entity_id="n1")
    )
    assert active["source_state"] == "active"

    assert engine.lifecycle.tombstone_node("n1") is True
    deleted = json.loads(
        engine.indexing.canonical_source_payload(entity_kind="node", entity_id="n1")
    )
    assert deleted["source_state"] == "deleted"
    assert deleted["source_fingerprint"] != active["source_fingerprint"]

    missing = json.loads(
        engine.indexing.canonical_source_payload(entity_kind="node", entity_id="missing")
    )
    assert missing["source_state"] == "missing"
    assert missing["source_fingerprint"] != deleted["source_fingerprint"]


def test_canonical_revision_batch_matches_single_results_with_one_event_scan(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path), backend_factory=build_fake_backend
    )
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1"))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2"))
    engine.lifecycle.tombstone_node("n2")

    original_iter = engine.metadata.iter_entity_events
    calls = 0

    def counted_iter_entity_events(**kwargs: object):
        nonlocal calls
        calls += 1
        return original_iter(**kwargs)

    engine.metadata.iter_entity_events = counted_iter_entity_events
    batch = engine.indexing.canonical_revision_payloads(
        entity_kind="node", entity_ids=["n1", "n2", "missing"]
    )
    assert calls == 1
    for entity_id, payload in batch.items():
        assert json.loads(payload) == json.loads(
            engine.indexing.canonical_revision_payload(
                entity_kind="node", entity_id=entity_id
            )
        )


def test_rust_metadata_source_payload_uses_native_projection_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    metadata = object.__new__(RustEnginePostgresMetaStore)
    calls: list[dict[str, object]] = []

    def graph_projection_records(**values: object) -> list[dict[str, object]]:
        calls.append(values)
        return [
            {
                "id": "n1",
                "document": "canonical node document",
                "metadata": {"lifecycle_status": "active", "doc_id": "d1"},
            }
        ]

    metadata.graph_projection_records = graph_projection_records  # type: ignore[method-assign]
    backend = SimpleNamespace(
        nodes=SimpleNamespace(name="gke_nodes"),
        edges=SimpleNamespace(name="gke_edges"),
        node_get=lambda **_: pytest.fail("Python backend read must not be used"),
    )
    engine = SimpleNamespace(
        metadata=metadata,
        backend=backend,
        namespace="workspace-ns",
        _implementation_selection_snapshot=capture_implementation_selection(),
    )

    payload = json.loads(
        IndexingSubsystem(engine=engine).canonical_source_payload(
            entity_kind="node", entity_id="n1"
        )
    )

    assert payload["source_state"] == "active"
    assert calls == [
        {
            "namespace": "workspace-ns",
            "workspace_id": None,
            "graph_space": None,
            "table": "gke_nodes",
            "ids": ["n1"],
            "metadata": {},
            "limit": 1,
                "for_update": False,
        }
    ]
