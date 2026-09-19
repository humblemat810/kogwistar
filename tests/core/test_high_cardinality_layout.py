from __future__ import annotations

import copy
import pickle
from dataclasses import asdict, replace

from kogwistar.conversation.conversation_context import ContextMessage, DroppedItem
from kogwistar.conversation.retrieval_orchestrator import RetrievalOutcome


def test_context_records_use_strict_slots_without_changing_dataclass_behavior() -> None:
    message = ContextMessage(
        role="user",
        content="remember this",
        node_id="node-1",
        source="live_turn",
    )
    dropped = DroppedItem(
        kind="tail_turn",
        node_id="node-2",
        reason="over_budget",
        token_cost=12,
    )

    assert not hasattr(message, "__dict__")
    assert not hasattr(dropped, "__dict__")
    assert asdict(message)["content"] == "remember this"
    assert asdict(dropped)["token_cost"] == 12
    assert replace(message, content="updated").content == "updated"
    assert copy.copy(message) == message
    assert pickle.loads(pickle.dumps(dropped)) == dropped


def test_retrieval_records_use_strict_slots() -> None:
    outcome = RetrievalOutcome(
        memory=object(),
        knowledge=object(),
        memory_pin=None,
        pinned_kg_pointer_node_ids=["node-1"],
        pinned_kg_edge_ids=["edge-1"],
    )
    assert not hasattr(outcome, "__dict__")
    assert outcome.memory_context_node_id is None
