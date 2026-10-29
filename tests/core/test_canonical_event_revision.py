from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from kogwistar.engine_core.acl_protocol import CreateOnceWriteProtocol
from kogwistar.engine_core.canonical_events import (
    canonical_revision_job_payload,
    read_canonical_entity_revision,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node
from tests._helpers.engine_factories import FakeEmbeddingFunction


def test_canonical_revision_and_job_payload_match_shared_rust_golden() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (repo_root / "contracts/golden/canonical-entity-revision-v1.json").read_text(
            encoding="utf-8"
        )
    )
    for case in fixture["cases"]:
        events = [
            (
                row["seq"],
                row["entity_kind"],
                row["entity_id"],
                row["op"],
                json.dumps(row["payload"], separators=(",", ":")),
            )
            for row in case["events"]
            if row["namespace"] == case["namespace"]
        ]
        revision = read_canonical_entity_revision(
            events=events,
            namespace=case["namespace"],
            entity_kind=case["entity_kind"],
            entity_id=case["entity_id"],
        )
        assert revision is not None
        expected = case["expected"]
        assert revision.revision == expected["revision"]
        assert revision.state == expected["state"]
        assert revision.payload == expected["payload"]
        assert canonical_revision_job_payload(revision) == expected["job_payload"]


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


@pytest.mark.parametrize("metadata_implementation", ["python", "rust"])
def test_create_node_if_absent_is_idempotent_and_rejects_payload_conflicts(
    tmp_path, monkeypatch, metadata_implementation
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", metadata_implementation)
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddingFunction(dim=3),
        backend_factory=build_fake_backend,
    )
    first = build_entity_node(
        node_id="create-once", doc_id="d1", embedding=[0.1, 0.2, 0.3]
    )
    same = first.model_copy(deep=True)
    different = first.model_copy(update={"summary": "different"}, deep=True)

    create_once = cast(CreateOnceWriteProtocol, engine.write).create_node_if_absent
    created = create_once(first)
    repeated = create_once(same)
    conflicting = create_once(different)

    assert created.status == "created"
    assert repeated.status == "already_exists"
    assert conflicting.status == "conflict"
    events = [
        row
        for row in engine.meta_sqlite.iter_entity_events(namespace=engine.namespace)
        if row[1] == "node" and row[2] == "create-once" and row[3] == "ADD"
    ]
    assert len(events) == 1
