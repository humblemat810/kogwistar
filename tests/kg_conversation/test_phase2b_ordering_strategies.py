
import pytest
from graph_knowledge_engine.conversation.conversation_context import ContextItem, apply_ordering

def _mk(kind, node_id, pinned=False, priority=100, turn_index=None):
    extra = {}
    if turn_index is not None:
        extra["turn_index"] = turn_index
    return ContextItem(kind=kind, text=kind, role="system" if kind=="system_prompt" else "user",
                       node_id=node_id, pinned=pinned, priority=priority, extra=extra)

def test_ordering_default_is_deterministic():
    items = [
        _mk("tail_turn", "t2", pinned=False, priority=5, turn_index=2),
        _mk("system_prompt", None, pinned=True, priority=0),
        _mk("tail_turn", "t1", pinned=False, priority=4, turn_index=1),
        _mk("pinned_kg_ref", "k1", pinned=True, priority=10),
    ]
    a = apply_ordering(items=list(items), ordering="default", phase="pre_pack")
    b = apply_ordering(items=list(items), ordering="default", phase="pre_pack")
    assert [x.node_id for x in a] == [x.node_id for x in b]

def test_ordering_grouped_policy_orders_turns_chronologically_post_pack():
    kept = [
        _mk("tail_turn", "t2", turn_index=2),
        _mk("tail_turn", "t1", turn_index=1),
        _mk("pinned_kg_ref", "k1", pinned=True, priority=10),
        _mk("system_prompt", None, pinned=True, priority=0),
    ]
    out = apply_ordering(items=list(kept), ordering="grouped_policy", phase="post_pack")
    # system first, then pinned_kg_ref, then turns by turn_index
    assert out[0].kind == "system_prompt"
    assert out[1].kind == "pinned_kg_ref"
    assert [x.node_id for x in out if x.kind=="tail_turn"] == ["t1","t2"]
