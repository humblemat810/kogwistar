from __future__ import annotations

import json

from kogwistar.engine_core.engine import GraphKnowledgeEngine
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
