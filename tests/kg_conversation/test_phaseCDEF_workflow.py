from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

pytestmark = pytest.mark.ci_full

from kogwistar.conversation.models import ConversationAIResponse
from kogwistar.conversation.resolvers import default_resolver
from kogwistar.runtime.models import RunSuccess


class FakeConversationEngine:
    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self.write = self

    def _iterative_defensive_emb(self, _text: str):
        return [0.0, 0.1]

    def add_node(self, node: Any) -> None:
        self.nodes[str(node.id)] = node

    def get_nodes(self, ids: list[str]) -> list[Any]:
        return [self.nodes[str(i)] for i in ids]

    # Phase D hook
    def latest_context_snapshot_cost(self, *, conversation_id: str, stage: str):
        return type("Cost", (), {"char_count": 99999, "token_count": 12345})()


class FakeStepContext:
    def __init__(self, state: dict[str, Any]) -> None:
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

    def __enter__(self) -> dict[str, Any]:
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


def test_rust_worker_answer_fallback_returns_materialized_node_effect():
    from types import SimpleNamespace

    from kogwistar.engine_core.graph_mutation_contract import (
        GraphMutationBatchPayload,
    )
    from kogwistar.engine_core.utils.refs import node_doc_and_meta

    class WorkerWrite:
        def node_doc_and_meta(self, node):
            document, metadata = node_doc_and_meta(node)
            return document, {**metadata, "namespace": "conv:c1"}

        def add_node(self, *_args, **_kwargs):
            pytest.fail("Rust worker answer fallback must return effects, not write")

    ce = FakeConversationEngine()
    ce.write = WorkerWrite()
    ce.namespace = "conv:c1"
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.backend = SimpleNamespace(
        node_get=lambda **_kwargs: {"ids": []},
        nodes=SimpleNamespace(name="gke_nodes"),
        embedding_dim=2,
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: (
            '{"source_fingerprint":"answer-fallback"}'
        )
    )
    ce.pre_add_node_hooks = [
        lambda node: node.properties.update({"prepared_by_hook": True})
    ]

    def answer_only(**_kwargs):
        return {"text": "fallback answer"}

    ctx = FakeStepContext(
        {
            "conversation_id": "c1",
            "user_id": "u1",
            "turn_node_id": "user1",
            "turn_index": 3,
            "_deps": {
                "conversation_engine": ce,
                "answer_only": answer_only,
                "force_answer_only": True,
                "_rust_worker_effects": True,
                "prev_turn_meta_summary": type(
                    "M",
                    (),
                    {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                        "tail_turn_index": 3,
                    },
                )(),
            },
        }
    )
    ctx.idempotency_key = "worker-answer-fallback"
    result = default_resolver.handlers["answer"](ctx)

    assert isinstance(result, RunSuccess)
    batch = GraphMutationBatchPayload.model_validate(result.graph_mutation_batch)
    assert len(batch.mutations) == 1
    assert batch.mutations[0].entity_kind == "node"
    assert batch.mutations[0].record.id == result.conversation_node_id
    assert batch.mutations[0].payload["properties"]["prepared_by_hook"] is True
    assert {job.index_kind for job in batch.index_jobs} == {"node_docs", "node_refs"}
    answer_update = next(
        update["answer"]
        for mode, update in result.state_update
        if mode == "u" and "answer" in update
    )
    assert answer_update["response_node_id"] == result.conversation_node_id


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
