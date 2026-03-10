from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile

import pytest

from graph_knowledge_engine.cdc.change_event import ChangeEvent
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.search_index.models import IndexingItem, make_index_key_for_item


class MockSink:
    def __init__(self):
        self.events: list[ChangeEvent] = []
        self.name = "mock-sink"

    def publish(self, event: ChangeEvent):
        self.events.append(event)


@pytest.fixture
def temp_engine_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _read_entity_events(engine: GraphKnowledgeEngine):
    iter_events = getattr(engine.meta_sqlite, "iter_entity_events", None)
    assert iter_events is not None
    return list(iter_events(namespace="default"))


def _replay_search_index_events(engine: GraphKnowledgeEngine) -> None:
    items: list[IndexingItem] = []
    for _, entity_kind, _entity_id, op, payload_json in _read_entity_events(engine):
        if entity_kind != "search_index" or op != "search_index.upsert":
            continue
        payload = json.loads(payload_json)
        items.append(IndexingItem(**payload))
    assert items, "Expected at least one search_index event to replay"
    engine.search_index.upsert_entries(items)


def test_search_index_cdc_and_event_sourcing(temp_engine_dir):
    engine = GraphKnowledgeEngine(persist_directory=temp_engine_dir)
    sink = MockSink()
    engine.changes.add_sink(sink)

    item = IndexingItem(
        node_id="test-node-1",
        canonical_title="Test Entity",
        keywords=["test", "demo"],
        aliases=["T1"],
        provision="manual",
        doc_id="doc-123",
    )

    engine.search_index.upsert_entries([item])

    assert len(sink.events) == 1
    ev = sink.events[0]
    assert ev.op == "search_index.upsert"
    assert ev.entity is not None
    assert ev.entity.get("kind") == "search_index"
    assert ev.payload["node_id"] == "test-node-1"
    assert ev.payload["canonical_title"] == "Test Entity"

    found = False
    for _, ek, _, op, payload_json in _read_entity_events(engine):
        if ek == "search_index" and op == "search_index.upsert":
            p = json.loads(payload_json)
            if p["node_id"] == "test-node-1":
                found = True
                break
    assert found, "Event not found in meta event log"


def test_search_index_replay(temp_engine_dir):
    engine = GraphKnowledgeEngine(persist_directory=temp_engine_dir)
    item = IndexingItem(
        node_id="replayed-node",
        canonical_title="Replay Title",
        keywords=[],
        aliases=[],
        provision="manual",
        doc_id="doc-replay",
    )
    engine.search_index.upsert_entries([item])

    index_db_path = os.path.join(temp_engine_dir, "index.db")
    conn = sqlite3.connect(index_db_path)
    try:
        # External-content FTS should now support base-table delete via triggers.
        conn.execute("DELETE FROM semantic_index")
        conn.commit()

        row_count = conn.execute("SELECT COUNT(*) FROM semantic_index").fetchone()[0]
        assert row_count == 0
    finally:
        conn.close()

    # Clear vector side too so hybrid search is genuinely empty before replay.
    index_key = make_index_key_for_item(item)
    delete_fn = getattr(engine.backend, "node_index_delete", None)
    assert delete_fn is not None, "backend must support node_index_delete for replay test"
    delete_fn(ids=[f"idx:{index_key}"])

    search_res = engine.search_index.search_hybrid("Replay", limit=5)
    assert len(search_res["results"]) == 0

    # Replay from the persisted entity event log into the projection.
    _replay_search_index_events(engine)

    search_res2 = engine.search_index.search_hybrid("Replay", limit=5)
    assert len(search_res2["results"]) > 0
    assert search_res2["results"][0]["node_id"] == "replayed-node"
    assert search_res2["results"][0]["canonical_title"] == "Replay Title"

def test_search_index_retrieval(temp_engine_dir):
    engine = GraphKnowledgeEngine(persist_directory=temp_engine_dir)
    
    items = [
        IndexingItem(
            node_id="node-alpha",
            canonical_title="Alpha Protocol",
            keywords=["protocol", "security", "alpha"],
            aliases=["AP-1"],
            provision="The primary security protocol for the system.",
            doc_id="doc-1"
        ),
        IndexingItem(
            node_id="node-beta",
            canonical_title="Beta Protocol",
            keywords=["protocol", "network", "beta"],
            aliases=["BP-1"],
            provision="Secondary network protocol.",
            doc_id="doc-1"
        ),
        IndexingItem(
            node_id="node-gamma",
            canonical_title="Gamma Ray",
            keywords=["radiation", "physics"],
            aliases=["GR"],
            provision="High energy electromagnetic radiation.",
            doc_id="doc-2"
        )
    ]
    engine.search_index.upsert_entries(items)
    
    # 1. Search by exact keyword (FTS dominant)
    res_alpha = engine.search_index.search_hybrid("security alpha", limit=2)
    assert len(res_alpha["results"]) > 0
    assert res_alpha["results"][0]["node_id"] == "node-alpha"
    
    # 2. Search by title
    res_gamma = engine.search_index.search_hybrid("Gamma Ray", limit=2)
    assert len(res_gamma["results"]) > 0
    assert res_gamma["results"][0]["node_id"] == "node-gamma"
    
    # 3. Semantic / Vector dominant search
    res_semantic = engine.search_index.search_hybrid("electromagnetic", limit=2)
    assert len(res_semantic["results"]) > 0
    assert res_semantic["results"][0]["node_id"] == "node-gamma"