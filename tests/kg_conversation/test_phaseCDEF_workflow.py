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
        self.edges: dict[str, Any] = {}
        self.write = self

    def _iterative_defensive_emb(self, _text: str):
        return [0.0, 0.1]

    def add_node(self, node: Any) -> None:
        self.nodes[str(node.id)] = node

    def add_edge(self, edge: Any) -> None:
        self.edges[str(edge.id)] = edge

    def node_doc_and_meta(self, node: Any):
        return str(node.summary or node.label or ""), dict(node.metadata)

    def edge_doc_and_meta(self, edge: Any):
        return str(edge.summary or edge.label or ""), dict(edge.metadata)

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


@pytest.mark.parametrize("acl_enabled, expected_acl_nodes", [(False, 1), (True, 6)])
def test_memory_pin_batches_graph_and_acl_effects_for_rust_worker(
    monkeypatch, acl_enabled, expected_acl_nodes
):
    from types import SimpleNamespace

    from kogwistar.conversation import resolvers
    from kogwistar.conversation.models import MetaFromLastSummary, RetrievalResult
    from kogwistar.engine_core.models import Span
    from kogwistar.acl.graph import ACLGraph
    from kogwistar.engine_core.subsystems.acl import ACLSubsystem
    from kogwistar.engine_core.graph_mutation_contract import (
        validate_graph_mutation_batch,
    )

    ce = FakeConversationEngine()
    ce.kg_graph_type = "conversation"
    ce.acl_enabled = acl_enabled
    ce.acl_graph = ACLGraph()
    ce.raw_read = SimpleNamespace(get_nodes=lambda **_kwargs: [])
    ce.acl = ACLSubsystem(ce)
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.namespace = "conversation"
    ce.backend = SimpleNamespace(
        nodes=SimpleNamespace(name="gke_nodes"),
        edges=SimpleNamespace(name="gke_edges"),
        embedding_dim=2,
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: "{}"
    )
    memory = SimpleNamespace(
        selected=RetrievalResult(nodes=[], edges=[]),
        memory_context_text="remembered context",
    )
    monkeypatch.setattr(resolvers, "_memory_result_from_state", lambda _state: memory)
    summary = MetaFromLastSummary(4, 2, 1)
    state = {
        "user_id": "u1",
        "conversation_id": "c1",
        "turn_node_id": "turn-1",
        "turn_index": 1,
        "mem_id": "memory-context-1",
        "prev_turn_meta_summary": {
            "prev_node_char_distance_from_last_summary": 4,
            "prev_node_distance_from_last_summary": 2,
            "tail_turn_index": 1,
        },
        "self_span": Span.from_dummy_for_conversation().model_dump(
            field_mode="backend", dump_format="json"
        ),
        "_deps": {
            "conversation_engine": ce,
            "llm_tasks": object(),
            "filtering_callback": lambda *_args: None,
            "_rust_worker_effects": True,
            "prev_turn_meta_summary": summary,
        },
    }

    ctx = FakeStepContext(state)
    ctx.idempotency_key = "memory-pin-worker"
    result = default_resolver.handlers["memory_pin"](ctx)

    assert isinstance(result, RunSuccess)
    assert ce.nodes == {}
    assert ce.edges == {}
    batch = validate_graph_mutation_batch(result.graph_mutation_batch)
    assert sum(
        mutation.payload.get("metadata", {}).get("entity_type") == "acl_record"
        for mutation in batch.mutations
    ) == expected_acl_nodes
    assert len(batch.mutations) == 2 + expected_acl_nodes
    assert len(batch.index_jobs) == 2 * len(batch.mutations)
    acl_context_mutation = next(
        mutation
        for mutation in batch.mutations
        if mutation.payload.get("metadata", {}).get("entity_type") == "acl_record"
        and mutation.payload["metadata"].get("acl_target_entity_id")
        == "memory-context-1"
        and mutation.payload["metadata"].get("acl_target_grain") == "node"
    )
    assert isinstance(
        batch.mutations[0].model_envelope["payload"], dict
    )
    assert acl_context_mutation.payload["metadata"]["acl_mode"] == "private"
    assert batch.mutations[0].record.id == "memory-context-1"
    assert any(mutation.entity_kind == "edge" for mutation in batch.mutations)
    assert (summary.prev_node_char_distance_from_last_summary,
            summary.prev_node_distance_from_last_summary) == (4, 2)
    summary_update = next(
        update["prev_turn_meta_summary"]
        for mode, update in result.state_update
        if mode == "u" and "prev_turn_meta_summary" in update
    )
    assert summary_update == {
        "prev_node_char_distance_from_last_summary": 22,
        "prev_node_distance_from_last_summary": 3,
        "tail_turn_index": 1,
    }


def test_memory_pin_preserves_python_graph_write_path(monkeypatch):
    from types import SimpleNamespace

    from kogwistar.conversation import resolvers
    from kogwistar.conversation.models import MetaFromLastSummary, RetrievalResult
    from kogwistar.engine_core.models import Span

    ce = FakeConversationEngine()
    memory = SimpleNamespace(
        selected=RetrievalResult(nodes=[], edges=[]),
        memory_context_text="remembered context",
    )
    monkeypatch.setattr(resolvers, "_memory_result_from_state", lambda _state: memory)
    summary = MetaFromLastSummary(4, 2, 1)
    state = {
        "user_id": "u1",
        "conversation_id": "c1",
        "turn_node_id": "turn-1",
        "turn_index": 1,
        "mem_id": "memory-context-1",
        "self_span": Span.from_dummy_for_conversation().model_dump(
            field_mode="backend", dump_format="json"
        ),
        "_deps": {
            "conversation_engine": ce,
            "llm_tasks": object(),
            "filtering_callback": lambda *_args: None,
            "prev_turn_meta_summary": summary,
        },
    }

    result = default_resolver.handlers["memory_pin"](FakeStepContext(state))

    assert isinstance(result, RunSuccess)
    assert list(ce.nodes) == ["memory-context-1"]
    assert list(ce.edges) == ["turn-1::hm::memory-context-1"]
    assert summary.prev_node_distance_from_last_summary == 3
    assert summary.prev_node_char_distance_from_last_summary == 22


def test_memory_pin_acl_worker_effects_supersede_and_replay_idempotently():
    from types import SimpleNamespace

    from kogwistar.acl.graph import ACLGraph, ACLRecord, ACLTarget
    from kogwistar.acl.models import ACLEdge, ACLNode
    from kogwistar.conversation.models import ConversationNode
    from kogwistar.conversation.resolvers import _memory_pin_worker_entities
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.engine_core.subsystems.acl import ACLSubsystem

    persisted_acl_nodes = []
    ce = FakeConversationEngine()
    ce.kg_graph_type = "conversation"
    ce.acl_enabled = True
    ce.acl_graph = ACLGraph()
    ce.raw_read = SimpleNamespace(
        get_nodes=lambda **_kwargs: list(persisted_acl_nodes)
    )
    ce.acl = ACLSubsystem(ce)
    memory_node = ConversationNode(
        id="memory-context-replay",
        label="Memory context",
        type="entity",
        doc_id="memory-context-replay",
        summary="saved context",
        role="system",
        turn_index=1,
        conversation_id="c1",
        user_id="u1",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "memory_context",
            "visibility": "private",
            "owner_agent_id": "agent-a",
            "security_scope": "scope-a",
            "level_from_root": 0,
        },
        domain_id=None,
        canonical_entity_id=None,
    )
    prior_acl = ACLRecord(
        target=ACLTarget(
            truth_graph="conversation",
            entity_id="memory-context-replay",
            grain="node",
        ),
        version=1,
        mode="public",
        created_by="agent-a",
        owner_id="agent-a",
        security_scope="scope-a",
    )
    persisted_acl_nodes.append(ce.acl._build_acl_record_node(prior_acl))

    effects = _memory_pin_worker_entities(
        ce,
        [memory_node],
        user_id="u1",
        memory_context_node=memory_node,
    )
    current_acl_node = next(
        entity
        for entity, _kind in effects
        if isinstance(entity, ACLNode)
        and entity.metadata.get("acl_target_grain") == "node"
    )
    assert current_acl_node.metadata["acl_version"] == 2
    assert any(
        isinstance(entity, ACLEdge) and entity.relation == "acl_supersedes"
        for entity, _kind in effects
    )

    persisted_acl_nodes.extend(
        entity for entity, _kind in effects if isinstance(entity, ACLNode)
    )
    replay_effects = _memory_pin_worker_entities(
        ce,
        [memory_node],
        user_id="u1",
        memory_context_node=memory_node,
    )
    assert [entity.id for entity, _kind in replay_effects] == [
        "memory-context-replay"
    ]


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


def test_rust_worker_kg_pin_returns_deterministic_ordered_graph_batch():
    from types import SimpleNamespace

    from kogwistar.conversation.models import FilteringResult, MetaFromLastSummary
    from kogwistar.engine_core.graph_mutation_contract import (
        GraphMutationBatchPayload,
    )
    from kogwistar.engine_core.models import Span
    from kogwistar.engine_core.utils.refs import edge_doc_and_meta, node_doc_and_meta

    class WorkerWrite:
        def node_doc_and_meta(self, node):
            document, metadata = node_doc_and_meta(node)
            return document, {**metadata, "namespace": "conv:c1"}

        def edge_doc_and_meta(self, edge):
            document, metadata = edge_doc_and_meta(edge)
            return document, {**metadata, "namespace": "conv:c1"}

        def add_node(self, *_args, **_kwargs):
            pytest.fail("Rust worker KG pin must return effects, not write nodes")

        def add_edge(self, *_args, **_kwargs):
            pytest.fail("Rust worker KG pin must return effects, not write edges")

    ce = FakeConversationEngine()
    ce.write = WorkerWrite()
    ce.namespace = "conv:c1"
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.backend = SimpleNamespace(
        nodes=SimpleNamespace(name="gke_nodes"),
        edges=SimpleNamespace(name="gke_edges"),
        embedding_dim=2,
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: "{}"
    )
    ce.pre_add_node_hooks = []
    ce.pre_add_edge_hooks = []

    kg_node = SimpleNamespace(
        id="kg-node",
        label="KG node",
        metadata={"label": "KG node", "summary": "node summary", "type": "entity"},
        safe_get_id=lambda: "kg-node",
    )
    kg_edge = SimpleNamespace(
        id="kg-edge",
        label="KG edge",
        metadata={"label": "KG edge", "summary": "edge summary", "type": "relationship"},
        safe_get_id=lambda: "kg-edge",
    )
    ref_knowledge_engine = SimpleNamespace(
        kg_graph_type="knowledge",
        read=SimpleNamespace(
            get_nodes=lambda _ids, **_kwargs: [kg_node],
            get_edges=lambda _ids, **_kwargs: [kg_edge],
        ),
    )
    selected = FilteringResult(node_ids=["kg-node"], edge_ids=["kg-edge"])
    state = {
        "conversation_id": "c1",
        "user_id": "u1",
        "turn_node_id": "turn-1",
        "turn_index": 1,
        "self_span": Span.from_dummy_for_conversation().model_dump(
            field_mode="backend", dump_format="json"
        ),
        "kg": {
            "node_id_entry": None,
            "candidate": None,
            "selected": selected,
            "reasoning": "selected both kinds",
        },
        "prev_turn_meta_summary": {
            "prev_node_char_distance_from_last_summary": 12,
            "prev_node_distance_from_last_summary": 1,
            "tail_turn_index": 1,
        },
        "_deps": {
            "conversation_engine": ce,
            "ref_knowledge_engine": ref_knowledge_engine,
            "llm_tasks": object(),
            "filtering_callback": lambda *_args: None,
            "_rust_worker_effects": True,
            "prev_turn_meta_summary": MetaFromLastSummary(12, 1, 1),
        },
    }
    ctx = FakeStepContext(state)
    ctx.idempotency_key = "worker-kg-pin"
    result = default_resolver.handlers["kg_pin"](ctx)

    assert isinstance(result, RunSuccess)
    batch = GraphMutationBatchPayload.model_validate(result.graph_mutation_batch)
    assert [mutation.entity_kind for mutation in batch.mutations] == [
        "node",
        "edge",
        "node",
        "edge",
    ]
    assert len(batch.index_jobs) == 8
    summary_update = next(
        update["prev_turn_meta_summary"]
        for mode, update in result.state_update
        if mode == "u" and "prev_turn_meta_summary" in update
    )
    assert summary_update["prev_node_distance_from_last_summary"] == 3
    pin_update = next(
        update["kg_pin"]
        for mode, update in result.state_update
        if mode == "u" and "kg_pin" in update
    )
    assert len(pin_update["pinned_pointer_node_ids"]) == 2
    assert len(pin_update["pinned_edge_ids"]) == 2
    assert ce.nodes == {}
    retry = default_resolver.handlers["kg_pin"](ctx)
    retry_batch = GraphMutationBatchPayload.model_validate(retry.graph_mutation_batch)
    assert [
        (mutation.record.id, mutation.event_id) for mutation in retry_batch.mutations
    ] == [(mutation.record.id, mutation.event_id) for mutation in batch.mutations]

    from kogwistar.conversation.knowledge_retriever import KnowledgeRetriever

    python_engine = FakeConversationEngine()
    python_engine.nodes = {}
    python_engine.edges = {}
    python_engine.write = python_engine
    python_engine.add_node = lambda node: python_engine.nodes.__setitem__(node.id, node)
    python_engine.add_edge = lambda edge: python_engine.edges.__setitem__(edge.id, edge)
    python_retriever = KnowledgeRetriever(
        conversation_engine=python_engine,
        ref_knowledge_engine=ref_knowledge_engine,
        llm_tasks=object(),
        filtering_callback=lambda *_args: None,
    )
    python_pointer_ids, python_edge_ids = python_retriever.pin_selected(
        user_id="u1",
        conversation_id="c1",
        turn_node_id="turn-1",
        turn_index=1,
        self_span=Span.from_dummy_for_conversation(),
        selected_knowledge=selected,
        prev_turn_meta_summary=MetaFromLastSummary(12, 1, 1),
    )
    assert python_pointer_ids == [
        mutation.record.id
        for mutation in batch.mutations
        if mutation.entity_kind == "node"
    ]
    assert python_edge_ids == [
        mutation.record.id
        for mutation in batch.mutations
        if mutation.entity_kind == "edge"
    ]

    ce.pre_add_node_hooks = [lambda _node: None]
    with pytest.raises(RuntimeError, match="cannot preserve hooks"):
        default_resolver.handlers["kg_pin"](ctx)


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
