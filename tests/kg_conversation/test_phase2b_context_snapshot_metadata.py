import pytest

from graph_knowledge_engine.conversation.models import ContextSnapshotMetadata
from graph_knowledge_engine.engine_core.models import ContextCost


def test_context_cost_addition_token_optional():
    a = ContextCost(char_count=10, token_count=None)
    b = ContextCost(char_count=5, token_count=3)
    c = a + b
    assert c.char_count == 15
    assert c.token_count is None

    a2 = ContextCost(char_count=10, token_count=4)
    c2 = a2 + b
    assert c2.char_count == 15
    assert c2.token_count == 7


def test_context_snapshot_metadata_flattens_cost_for_chroma():
    meta = ContextSnapshotMetadata(
        run_id="r1",
        run_step_seq=12,
        attempt_seq=0,
        stage="answer",
        model_name="m",
        budget_tokens=100,
        tail_turn_index=7,
        used_node_ids=["n1", "n2"],
        rendered_context_hash="h",
        cost=ContextCost(char_count=123, token_count=45),
    )

    flat = meta.to_chroma_metadata()
    assert flat["entity_type"] == "context_snapshot"
    assert flat["run_id"] == "r1"
    assert flat["run_step_seq"] == 12
    assert flat["cost.char_count"] == 123
    assert flat["cost.token_count"] == 45
    assert "cost" not in flat  # must be flat for chroma

    back = ContextSnapshotMetadata.from_chroma_metadata(flat)
    assert back.cost.char_count == 123
    assert back.cost.token_count == 45


def test_context_snapshot_metadata_accepts_flat_cost_on_validate():
    meta = ContextSnapshotMetadata.from_chroma_metadata(
        {
            "run_id": "r2",
            "run_step_seq": 1,
            "attempt_seq": 2,
            "stage": "select_used_evidence",
            "rendered_context_hash": "hh",
            "cost.char_count": 9,
            "cost.token_count": 3,
        }
    )
    assert meta.cost.char_count == 9
    assert meta.cost.token_count == 3
