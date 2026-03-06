
from graph_knowledge_engine.conversation.models import ContextSnapshotMetadata
from graph_knowledge_engine.engine_core.models import ContextCost

def test_context_snapshot_metadata_flattens_cost():
    meta = ContextSnapshotMetadata(
        run_id="r1",
        run_step_seq=1,
        attempt_seq=0,
        stage="s",
        rendered_context_hash="h",
        used_node_ids=["n1","n2"],
        cost=ContextCost(char_count=123, token_count=45),
    )
    d = meta.to_chroma_metadata()
    assert "cost" not in d
    assert d["cost.char_count"] == 123
    assert d["cost.token_count"] == 45

def test_context_snapshot_metadata_roundtrip():
    d = {
        "entity_type":"context_snapshot",
        "level_from_root":0,
        "run_id":"r1",
        "run_step_seq":2,
        "attempt_seq":1,
        "stage":"x",
        "rendered_context_hash":"hh",
        "used_node_ids":["a"],
        "cost.char_count": 10,
        "cost.token_count": None,
    }
    meta = ContextSnapshotMetadata.from_chroma_metadata(d)
    assert meta.cost.char_count == 10
    assert meta.cost.token_count is None
