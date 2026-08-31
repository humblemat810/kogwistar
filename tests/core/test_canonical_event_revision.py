from __future__ import annotations

import pytest

from kogwistar.engine_core.canonical_events import read_canonical_entity_revision
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node
from tests.conftest import FakeEmbeddingFunction


def test_canonical_revision_folds_active_delete_and_reactivation() -> None:
    events = [
        (1, "node", "n1", "ADD", '{"id":"n1","summary":"v1"}'),
        (
            2,
            "node",
            "n1",
            "REPLACE",
            '{"lifecycle_patch":{"lifecycle_status":"tombstoned"}}',
        ),
        (3, "node", "n1", "ADD", '{"id":"n1","summary":"v2"}'),
        (4, "node", "other", "ADD", '{"id":"other"}'),
    ]

    revision = read_canonical_entity_revision(
        events=events,
        namespace="default",
        entity_kind="node",
        entity_id="n1",
    )

    assert revision is not None
    assert revision.revision == 3
    assert revision.state == "active"
    assert not revision.is_deleted
    assert revision.payload["summary"] == "v2"


def test_canonical_revision_tombstone_is_deleted() -> None:
    revision = read_canonical_entity_revision(
        events=[
            (1, "edge", "e1", "ADD", '{"id":"e1"}'),
            (2, "edge", "e1", "TOMBSTONE", '{"entity_id":"e1"}'),
        ],
        namespace="tenant-a",
        entity_kind="edge",
        entity_id="e1",
    )

    assert revision is not None
    assert revision.namespace == "tenant-a"
    assert revision.revision == 2
    assert revision.is_deleted


@pytest.mark.parametrize(
    "events, match",
    [
        ([(1, "node", "n1", "ADD", "not-json")], "invalid canonical event payload"),
        ([(1, "node", "n1", "MERGE", "{}")], "unsupported canonical event op"),
    ],
)
def test_canonical_revision_fails_closed_for_unusable_event(events, match) -> None:
    with pytest.raises(ValueError, match=match):
        read_canonical_entity_revision(
            events=events,
            namespace="default",
            entity_kind="node",
            entity_id="n1",
        )


def test_indexing_reads_canonical_revision_from_existing_event_store(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddingFunction(dim=3),
        backend_factory=build_fake_backend,
    )
    node = build_entity_node(
        node_id="n1", doc_id="d1", embedding=[0.1, 0.2, 0.3]
    )
    engine.write.add_node(node)

    active = engine.indexing.canonical_entity_revision(
        entity_kind="node", entity_id="n1"
    )
    assert active is not None
    assert active.state == "active"

    assert engine.lifecycle.tombstone_node("n1")
    deleted = engine.indexing.canonical_entity_revision(
        entity_kind="node", entity_id="n1"
    )
    assert deleted is not None
    assert deleted.revision > active.revision
    assert deleted.is_deleted
