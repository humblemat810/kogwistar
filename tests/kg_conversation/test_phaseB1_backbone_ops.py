from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from kogwistar.conversation.conversation_orchestrator import (
    get_id_for_conversation_turn,
    get_id_for_conversation_turn_edge,
)
from kogwistar.conversation.models import (
    ConversationEdge,
    ConversationNode,
)
from kogwistar.conversation.resolvers import default_resolver
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)
from kogwistar.runtime.models import RunSuccess

pytestmark = [
    pytest.mark.ci,
    pytest.mark.conversation,
    pytest.mark.ci,
]


class FakeConversationEngine:
    """Minimal conversation engine stub for resolver unit tests."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, Any] = {}
        self.read = _ReadShim(self)
        self.write = _WriteShim(self)

    def _iterative_defensive_emb(self, _text: str):
        return [0.0, 0.1]

    def add_node(self, node: Any) -> None:
        self.nodes[str(node.id)] = node

    def add_edge(self, edge: Any) -> None:
        self.edges[str(edge.id)] = edge

    def get_nodes(self, ids: list[str]) -> list[Any]:
        return [self.nodes[str(i)] for i in ids]


class _ReadShim:
    def __init__(self, engine: FakeConversationEngine) -> None:
        self._engine = engine

    def get_nodes(self, ids: list[str], **kwargs) -> list[Any]:
        return self._engine.get_nodes(ids)


class _WriteShim:
    def __init__(self, engine: FakeConversationEngine) -> None:
        self._engine = engine

    def add_node(self, node: Any, *args, **kwargs) -> None:
        self._engine.add_node(node)

    def add_edge(self, edge: Any, *args, **kwargs) -> None:
        self._engine.add_edge(edge)


class FakeStepContext:
    def __init__(self, state: dict[str, Any]) -> None:
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

    def __enter__(self) -> dict[str, Any]:
        return self._ctx._state

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _noop_task_set() -> LLMTaskSet:
    return LLMTaskSet(
        extract_graph=lambda request: ExtractGraphTaskResult(
            raw=None, parsed_payload={}, parsing_error=None
        ),
        adjudicate_pair=lambda request: AdjudicatePairTaskResult(
            verdict_payload={}, raw=None, parsing_error=None
        ),
        adjudicate_batch=lambda request: AdjudicateBatchTaskResult(
            verdict_payloads=[], raw=None, parsing_error=None
        ),
        filter_candidates=lambda request: FilterCandidatesTaskResult(
            node_ids=[], edge_ids=[], reasoning=""
        ),
        summarize_context=lambda request: SummarizeContextTaskResult(text=""),
        answer_with_citations=lambda request: AnswerWithCitationsTaskResult(
            answer_payload={}, raw=None, parsing_error=None
        ),
        repair_citations=lambda request: RepairCitationsTaskResult(
            answer_payload={}, raw=None, parsing_error=None
        ),
    )


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
        engine.write.add_edge(edge)
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


def test_add_user_turn_returns_idempotent_rust_graph_effect_without_direct_write():
    from kogwistar._rust_bridge import validate_graph_model_family
    from kogwistar.engine_core.graph_mutation_contract import GraphMutationBatchPayload
    from kogwistar.engine_core.utils.refs import node_doc_and_meta
    from kogwistar.runtime.rust_worker import RustStepResolverAdapter

    class WorkerWrite(_WriteShim):
        def node_doc_and_meta(self, node):
            document, metadata = node_doc_and_meta(node)
            metadata = dict(metadata)
            metadata["namespace"] = "conv:c1"
            return document, metadata

        def add_node(self, *_args, **_kwargs):
            pytest.fail("Rust worker resolver must return a graph effect, not write")

    class WorkerEngine(FakeConversationEngine):
        def __init__(self):
            super().__init__()
            self.write = WorkerWrite(self)
            self.namespace = "conv:c1"
            self.persistence_mode = "single_stage"
            self._phase1_enable_index_jobs = True
            self._disable_event_log = False
            self.backend = type(
                "Backend",
                (),
                {
                    "nodes": type("Collection", (), {"name": "gke_nodes"})(),
                    "embedding_dim": 2,
                },
            )()
            self.indexing = type(
                "Indexing",
                (),
                {
                    "canonical_source_payload_for_projection": lambda _self, **_kwargs: (
                        '{"source_fingerprint":"stable"}'
                    ),
                },
            )()

    def run_once() -> tuple[dict[str, Any], str]:
        engine = WorkerEngine()
        worker = RustStepResolverAdapter(
            default_resolver,
            dependency_provider=lambda _work: {"conversation_engine": engine},
        )
        effect = worker(
            {
                "message_id": "lane-user-turn-7",
                "run_id": "run-1",
                "step_id": "add-user",
                "claimed_by": "worker-1",
                "correlation_id": "run-1",
                "worker_effect_capabilities": ["graph_mutation_batch_v1"],
                "payload": {
                    "workflow_id": "conversation.answer",
                    "conversation_id": "c1",
                    "run_id": "run-1",
                    "op": "add_user_turn",
                    "node_id": "add-user",
                    "op_version": "v1",
                    "step_seq": 0,
                    "token_id": "run-1",
                    "state": {
                        "user_id": "u1",
                        "conversation_id": "c1",
                        "role": "user",
                        "user_text": "hello",
                        "turn_index": 7,
                        "turn_id": "t7",
                    },
                },
            }
        )
        assert "graph_mutation_batch" in effect, effect
        batch = GraphMutationBatchPayload.model_validate(effect["graph_mutation_batch"])
        assert not engine.nodes
        state_update = effect["state_update"]
        turn_update = next(
            update
            for _, update in state_update
            if isinstance(update, dict) and "turn_node_id" in update
        )
        node_id = turn_update["turn_node_id"]
        assert isinstance(turn_update["self_span"], dict)
        return batch.model_dump(mode="json"), str(node_id)

    first_batch, first_node_id = run_once()
    second_batch, second_node_id = run_once()
    assert first_node_id == second_node_id
    assert first_batch == second_batch
    assert first_batch["mutations"][0]["payload"]["id"] == first_node_id
    assert first_batch["mutations"][0]["record"]["embedding"] == [0.0, 0.1]
    validate_graph_model_family(first_batch["mutations"][0]["model_envelope"])
    assert {job["index_kind"] for job in first_batch["index_jobs"]} == {
        "node_docs",
        "node_refs",
    }


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


def test_link_prev_turn_returns_idempotent_rust_edge_effect_without_direct_write():
    from kogwistar._rust_bridge import validate_graph_model_family
    from kogwistar.engine_core.graph_mutation_contract import GraphMutationBatchPayload
    from kogwistar.engine_core.utils.refs import edge_doc_and_meta
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.rust_worker import RustStepResolverAdapter

    class WorkerWrite(_WriteShim):
        def edge_doc_and_meta(self, edge):
            document, metadata = edge_doc_and_meta(edge)
            return document, {**metadata, "namespace": "conv:c1"}

        def add_edge(self, *_args, **_kwargs):
            pytest.fail("Rust worker resolver must return an edge effect, not write")

    class WorkerEngine(FakeConversationEngine):
        def __init__(self):
            super().__init__()
            self.write = WorkerWrite(self)
            self.namespace = "conv:c1"
            self.persistence_mode = "single_stage"
            self._phase1_enable_index_jobs = True
            self._disable_event_log = False
            self.backend = type(
                "Backend",
                (),
                {
                    "edges": type("Collection", (), {"name": "gke_edges"})(),
                    "embedding_dim": 2,
                },
            )()
            self.indexing = type(
                "Indexing",
                (),
                {
                    "canonical_source_payload_for_projection": lambda _self, **_kwargs: (
                        '{"source_fingerprint":"stable-edge"}'
                    ),
                },
            )()
            for node_id, turn_index in (("prev", 0), ("curr", 1)):
                self.nodes[node_id] = ConversationNode(
                    id=node_id,
                    label=node_id,
                    type="entity",
                    summary=node_id,
                    doc_id=f"conv:c1#{node_id}",
                    mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                    metadata={
                        "entity_type": "conversation_turn",
                        "conversation_id": "c1",
                        "user_id": "u1",
                        "level_from_root": 0,
                    },
                    role="user",
                    turn_index=turn_index,
                    conversation_id="c1",
                    user_id="u1",
                    properties=None,
                    embedding=[0.0, 0.1],
                )

        def _iterative_defensive_emb(self, _text: str):
            return [0.0, 0.1]

    engine = WorkerEngine()
    worker = RustStepResolverAdapter(
        default_resolver,
        dependency_provider=lambda _work: {
            "conversation_engine": engine,
            "add_link_to_new_turn": lambda *_args, **_kwargs: pytest.fail(
                "Rust worker resolver must not call the Python graph writer"
            ),
        },
    )
    span = Span.from_dummy_for_conversation()
    effect = worker(
        {
            "message_id": "lane-link-prev-7",
            "run_id": "run-1",
            "step_id": "link-prev",
            "claimed_by": "worker-1",
            "correlation_id": "run-1",
            "worker_effect_capabilities": ["graph_mutation_batch_v1"],
            "payload": {
                "workflow_id": "conversation.answer",
                "conversation_id": "c1",
                "run_id": "run-1",
                "op": "link_prev_turn",
                "node_id": "link-prev",
                "op_version": "v1",
                "step_seq": 1,
                "token_id": "run-1",
                "state": {
                    "user_id": "u1",
                    "conversation_id": "c1",
                    "turn_node_id": "curr",
                    "turn_index": 1,
                    "prev_turn_id": "prev",
                    "self_span": span.model_dump(
                        field_mode="backend", dump_format="json"
                    ),
                    "prev_turn_meta_summary": {
                        "prev_node_char_distance_from_last_summary": 5,
                        "prev_node_distance_from_last_summary": 1,
                        "tail_turn_index": 1,
                    },
                },
            },
        }
    )
    assert "graph_mutation_batch" in effect, effect
    batch = GraphMutationBatchPayload.model_validate(effect["graph_mutation_batch"])
    assert engine.edges == {}
    mutation = batch.mutations[0]
    validate_graph_model_family(mutation.model_envelope)
    assert mutation.entity_kind == "edge"
    assert mutation.op == "ADD"
    edge_update = next(
        update
        for _, update in effect["state_update"]
        if isinstance(update, dict) and "prev_turn_edge_id" in update
    )
    assert mutation.record.id == edge_update["prev_turn_edge_id"]
    assert mutation.payload["source_ids"] == ["prev"]
    assert mutation.payload["target_ids"] == ["curr"]
    assert {job.index_kind for job in batch.index_jobs} == {
        "edge_refs",
        "edge_endpoints",
    }


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


def test_link_assistant_turn_returns_rust_edge_effect_and_runs_edge_hooks():
    from types import SimpleNamespace

    from kogwistar.engine_core.graph_mutation_contract import (
        GraphMutationBatchPayload,
    )
    from kogwistar._rust_bridge import validate_graph_model_family
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.engine_core.utils.refs import edge_doc_and_meta

    class WorkerWrite(_WriteShim):
        def edge_doc_and_meta(self, edge):
            document, metadata = edge_doc_and_meta(edge)
            return document, {**metadata, "namespace": "conv:c1"}

        def add_edge(self, *_args, **_kwargs):
            pytest.fail("Rust worker resolver must return an edge effect, not write")

    ce = FakeConversationEngine()
    ce.write = WorkerWrite(ce)
    ce.namespace = "conv:c1"
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.backend = SimpleNamespace(
        edges=SimpleNamespace(name="gke_edges"), embedding_dim=2
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: (
            '{"source_fingerprint":"stable-assistant-edge"}'
        )
    )
    ce.pre_add_edge_hooks = [
        lambda edge: edge.metadata.update({"prepared_by_hook": True})
    ]
    for node_id, role in (("user1", "user"), ("assist1", "assistant")):
        ce.nodes[node_id] = ConversationNode(
            id=node_id,
            label=node_id,
            type="entity",
            summary=node_id,
            doc_id=f"conv:c1#{node_id}",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={"entity_type": "conversation_turn", "level_from_root": 0},
            role=role,
            turn_index=0,
            conversation_id="c1",
            user_id="u1",
            properties={},
            embedding=[0.0, 0.1],
        )

    state = {
        "user_id": "u1",
        "conversation_id": "c1",
        "turn_node_id": "user1",
        "turn_index": 7,
        "self_span": Span.from_dummy_for_conversation().model_dump(
            field_mode="backend", dump_format="json"
        ),
        "answer": {"response_node_id": "assist1"},
        "_deps": {
            "conversation_engine": ce,
            "_rust_worker_effects": True,
            "add_link_to_new_turn": lambda *_args, **_kwargs: pytest.fail(
                "Rust worker path must not call the Python graph writer"
            ),
            "prev_turn_meta_summary": SimpleNamespace(
                prev_node_char_distance_from_last_summary=8,
                prev_node_distance_from_last_summary=2,
                tail_turn_index=7,
            ),
        },
    }
    ctx = FakeStepContext(state)
    ctx.idempotency_key = "lane-link-assistant-7"
    result = default_resolver.handlers["link_assistant_turn"](ctx)

    assert isinstance(result, RunSuccess)
    assert ce.edges == {}
    batch = GraphMutationBatchPayload.model_validate(result.graph_mutation_batch)
    assert len(batch.mutations) == 1
    mutation = batch.mutations[0]
    validate_graph_model_family(mutation.model_envelope)
    assert mutation.entity_kind == "edge"
    assert mutation.payload["metadata"]["prepared_by_hook"] is True
    assert mutation.payload["source_ids"] == ["user1"]
    assert mutation.payload["target_ids"] == ["assist1"]
    assert {job.index_kind for job in batch.index_jobs} == {
        "edge_refs",
        "edge_endpoints",
    }

    ce.pre_add_edge_hooks = [lambda _edge: True]
    rejected = default_resolver.handlers["link_assistant_turn"](ctx)
    assert isinstance(rejected, RunSuccess)
    assert "graph_mutation_batch" not in rejected.model_dump()


def test_link_assistant_turn_falls_back_to_agentic_answering_result():
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
            "agentic_answering_result": {"assistant_turn_node_id": "assist1"},
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


def test_link_assistant_turn_accepts_missing_span_in_agentic_helper():
    from kogwistar.conversation.agentic_answering import AgenticAnsweringAgent

    ce = FakeConversationEngine()
    agent = AgenticAnsweringAgent(
        conversation_engine=ce,
        knowledge_engine=ce,
        llm_tasks=_noop_task_set(),
    )

    prev = type("N", (), {})()
    prev.id = "prev1"
    ce.add_node(prev)

    cur = type("N", (), {})()
    cur.id = "cur1"
    ce.add_node(cur)

    edge = agent._add_link_to_new_turn(
        edge_id="edge1",
        turn_node=cur,
        prev_node=prev,
        conversation_id="c1",
        span=None,
        prev_turn_meta_summary=type(
            "M",
            (),
            {
                "prev_node_char_distance_from_last_summary": 0,
                "prev_node_distance_from_last_summary": 0,
                "tail_turn_index": 0,
            },
        )(),
    )

    assert getattr(edge, "id", None) == "edge1"
    assert "edge1" in ce.edges
