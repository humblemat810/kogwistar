from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, List

import pytest

from kogwistar.conversation.models import (
    ConversationEdge,
    ConversationNode,
)
from kogwistar.conversation.resolvers import default_resolver
from kogwistar.runtime.models import RunSuccess
from kogwistar.conversation.conversation_orchestrator import (
    get_id_for_conversation_turn,
    get_id_for_conversation_turn_edge,
)

pytestmark = [
    pytest.mark.ci,
    pytest.mark.conversation,
    pytest.mark.ci,
]


class FakeConversationEngine:
    """Minimal conversation engine stub for resolver unit tests."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Any] = {}
        self.edges: Dict[str, Any] = {}

    def _iterative_defensive_emb(self, _text: str):
        return [0.0, 0.1]

    def add_node(self, node: Any) -> None:
        self.nodes[str(getattr(node, "id"))] = node

    def add_edge(self, edge: Any) -> None:
        self.edges[str(getattr(edge, "id"))] = edge

    def get_nodes(self, ids: List[str]) -> List[Any]:
        return [self.nodes[str(i)] for i in ids]


class FakeStepContext:
    def __init__(self, state: Dict[str, Any]) -> None:
        self._state = state

    @property
    def state_view(self) -> Mapping[str, Any]:
        return MappingProxyType(self._state)

    @property
    def state_write(self):
        return _StateWriteTxn(self)


class _StateWriteTxn:
    def __init__(self, ctx: FakeStepContext) -> None:
        self._ctx = ctx

    def __enter__(self) -> Dict[str, Any]:
        return self._ctx._state

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _fake_add_link_to_new_turn(engine: FakeConversationEngine):
    def _fn(
        edge_id,
        turn_node,
        prev_node,
        conversation_id,
        *,
        span,
        prev_turn_meta_summary,
        causal_type="chain",
        clock=None,
    ):
        # A lightweight edge-like object. Real impl adds ConversationEdge to engine.
        edge = type("E", (), {})()
        edge.id = edge_id
        edge.source_ids = [prev_node.id]
        edge.target_ids = [turn_node.id]
        edge.relation = "next_turn"
        engine.add_edge(edge)
        return edge

    return _fn


def test_b1_ops_are_registered():
    for op in ("add_user_turn", "link_prev_turn", "link_assistant_turn"):
        assert op in default_resolver.handlers


def test_add_user_turn_creates_deterministic_node_id_and_updates_state():
    ce = FakeConversationEngine()
    ctx = FakeStepContext(
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "role": "user",
            "user_text": "hello",
            "turn_index": 7,
            "turn_id": "t7",
            "_deps": {"conversation_engine": ce},
        }
    )

    fn = default_resolver.handlers["add_user_turn"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)

    expected_id = get_id_for_conversation_turn(
        ConversationNode.id_kind,
        "u1",
        "c1",
        "hello",
        "7",
        "user",
        "conversation_turn",
        "True",
    )
    assert expected_id in ce.nodes

    # state_update should set turn_node_id and turn_index
    upd = dict(res.state_update[0][1])
    assert upd["turn_node_id"] == expected_id
    assert upd["turn_index"] == 7
    assert "self_span" in upd
    assert "embedding" in upd


def test_link_prev_turn_creates_next_turn_edge():
    ce = FakeConversationEngine()

    # create prev and current nodes in engine
    prev = type("N", (), {})()
    prev.id = "prev1"
    ce.add_node(prev)

    cur = type("N", (), {})()
    cur.id = "cur1"
    ce.add_node(cur)

    deps = {
        "conversation_engine": ce,
        "add_link_to_new_turn": _fake_add_link_to_new_turn(ce),
        # minimal mts-like object
        "prev_turn_meta_summary": type(
            "M",
            (),
            {
                "prev_node_char_distance_from_last_summary": 0,
                "prev_node_distance_from_last_summary": 0,
                "tail_turn_index": 0,
            },
        )(),
    }

    ctx = FakeStepContext(
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "prev_turn_id": "prev1",
            "turn_node_id": "cur1",
            "turn_index": 7,
            "self_span": None,
            "_deps": deps,
        }
    )

    fn = default_resolver.handlers["link_prev_turn"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)

    expected_edge_id = get_id_for_conversation_turn_edge(
        ConversationEdge.id_kind,
        "u1",
        "c1",
        "next_turn",
        7,
        ["prev1"],
        ["cur1"],
        [],
        [],
        "conversation_edge",
    )
    assert expected_edge_id in ce.edges


def test_link_assistant_turn_creates_next_turn_edge():
    ce = FakeConversationEngine()

    user_turn = type("N", (), {})()
    user_turn.id = "user1"
    ce.add_node(user_turn)

    assistant_turn = type("N", (), {})()
    assistant_turn.id = "assist1"
    ce.add_node(assistant_turn)

    deps = {
        "conversation_engine": ce,
        "add_link_to_new_turn": _fake_add_link_to_new_turn(ce),
        "prev_turn_meta_summary": type(
            "M",
            (),
            {
                "prev_node_char_distance_from_last_summary": 0,
                "prev_node_distance_from_last_summary": 0,
                "tail_turn_index": 0,
            },
        )(),
    }
    ctx = FakeStepContext(
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "turn_node_id": "user1",
            "turn_index": 7,
            "self_span": None,
            "answer": {"response_node_id": "assist1"},
            "_deps": deps,
        }
    )

    fn = default_resolver.handlers["link_assistant_turn"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)

    expected_edge_id = get_id_for_conversation_turn_edge(
        ConversationEdge.id_kind,
        "u1",
        "c1",
        "next_turn",
        7,
        ["user1"],
        ["assist1"],
        [],
        [],
        "conversation_edge",
    )
    assert expected_edge_id in ce.edges
