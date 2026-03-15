from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, List
import pytest

from graph_knowledge_engine.conversation.models import ConversationAIResponse
from graph_knowledge_engine.conversation.resolvers import default_resolver
from graph_knowledge_engine.runtime.models import RunSuccess


class FakeConversationEngine:
    def __init__(self) -> None:
        self.nodes: Dict[str, Any] = {}

    def _iterative_defensive_emb(self, _text: str):
        return [0.0, 0.1]

    def add_node(self, node: Any) -> None:
        self.nodes[str(getattr(node, "id"))] = node

    def get_nodes(self, ids: List[str]) -> List[Any]:
        return [self.nodes[str(i)] for i in ids]

    # Phase D hook
    def latest_context_snapshot_cost(self, *, conversation_id: str, stage: str):
        return type("Cost", (), {"char_count": 99999, "token_count": 12345})()


class FakeStepContext:
    def __init__(self, state: Dict[str, Any]) -> None:
        self._state = state
        self.events = None

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


def test_add_user_turn_respects_provided_turn_node_id():
    ce = FakeConversationEngine()
    ctx = FakeStepContext(
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "role": "user",
            "user_text": "hello",
            "turn_index": 1,
            "turn_id": "t1",
            "turn_node_id": "provided_node_id",
            "_deps": {"conversation_engine": ce},
        }
    )
    fn = default_resolver.handlers["add_user_turn"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)
    assert "provided_node_id" in ce.nodes


def test_answer_prefers_agentic_when_available():
    class FakeAgent:
        def answer_workflow_v2(
            self,
            *,
            conversation_id: str,
            user_id: str | None,
            prev_turn_meta_summary,
            workflow_engine=None,
            events=None,
            trace=True,
        ):
            return {"assistant_turn_node_id": "assist123", "assistant_text": "hi"}

    def fake_answer_only(*arg, **kwarg) -> ConversationAIResponse:
        return ConversationAIResponse(text="fake answer", response_node_id="assist123")

    ctx = FakeStepContext(
        {
            "conversation_id": "c1",
            "user_id": "u1",
            "turn_node_id": "user1",
            "_deps": {
                "agent": FakeAgent(),
                "prev_turn_meta_summary": type(
                    "M",
                    (),
                    {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                        "tail_turn_index": 0,
                    },
                )(),
                "answer_only": fake_answer_only,
            },
        }
    )
    fn = default_resolver.handlers["answer"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)
    # state update contains answer.response_node_id
    upd = res.state_update[0][1]["answer"]
    assert upd["response_node_id"] == "assist123"


def test_answer_does_not_implicitly_fallback_on_agent_error():
    class RaisingAgent:
        def answer_workflow_v2(
            self,
            *,
            conversation_id: str,
            user_id: str | None,
            prev_turn_meta_summary,
            workflow_engine=None,
            events=None,
            trace=True,
        ):
            _ = (
                conversation_id,
                user_id,
                prev_turn_meta_summary,
                workflow_engine,
                events,
                trace,
            )
            raise RuntimeError("agent failed")

    def fake_answer_only(*arg, **kwarg) -> ConversationAIResponse:
        return ConversationAIResponse(text="fallback answer", response_node_id="assist999")

    ctx = FakeStepContext(
        {
            "conversation_id": "c1",
            "user_id": "u1",
            "turn_node_id": "user1",
            "_deps": {
                "agent": RaisingAgent(),
                "prev_turn_meta_summary": type(
                    "M",
                    (),
                    {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                        "tail_turn_index": 0,
                    },
                )(),
                "answer_only": fake_answer_only,
            },
        }
    )
    fn = default_resolver.handlers["answer"]
    with pytest.raises(RuntimeError, match="agent failed"):
        fn(ctx)


def test_answer_can_be_forced_to_use_answer_only():
    class RaisingAgent:
        def answer_workflow_v2(
            self,
            *,
            conversation_id: str,
            user_id: str | None,
            prev_turn_meta_summary,
            workflow_engine=None,
            events=None,
            trace=True,
        ):
            _ = (
                conversation_id,
                user_id,
                prev_turn_meta_summary,
                workflow_engine,
                events,
                trace,
            )
            raise RuntimeError("agent should not run")

    def fake_answer_only(*arg, **kwarg) -> ConversationAIResponse:
        return ConversationAIResponse(text="fallback answer", response_node_id="assist999")

    ctx = FakeStepContext(
        {
            "conversation_id": "c1",
            "user_id": "u1",
            "turn_node_id": "user1",
            "_deps": {
                "agent": RaisingAgent(),
                "force_answer_only": True,
                "prev_turn_meta_summary": type(
                    "M",
                    (),
                    {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                        "tail_turn_index": 0,
                    },
                )(),
                "answer_only": fake_answer_only,
            },
        }
    )
    fn = default_resolver.handlers["answer"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)
    upd = res.state_update[0][1]["answer"]
    assert upd["response_node_id"] == "assist999"


def test_decide_summarize_uses_snapshot_cost_when_available():
    ce = FakeConversationEngine()
    ctx = FakeStepContext(
        {
            "conversation_id": "c1",
            "answer": {"llm_decision_need_summary": False},
            "_deps": {
                "conversation_engine": ce,
                "summary_char_threshold": 1000,
                "summary_token_threshold": 1000,
                "prev_turn_meta_summary": type(
                    "M",
                    (),
                    {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                        "tail_turn_index": 0,
                    },
                )(),
            },
        }
    )
    fn = default_resolver.handlers["decide_summarize"]
    res = fn(ctx)
    assert isinstance(res, RunSuccess)
    upd = res.state_update[0][1]["summary"]
    assert upd["should_summarize"] is True
