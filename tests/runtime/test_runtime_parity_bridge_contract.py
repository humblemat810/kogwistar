from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import uuid

import pytest

from kogwistar.runtime import AsyncWorkflowRuntime, WorkflowRuntime
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    RunSuspended,
    WorkflowCompletedNode,
    WorkflowDesignArtifact,
    WorkflowEdge,
    WorkflowFailedNode,
    WorkflowInvocationRequest,
    WorkflowNode,
)
from kogwistar.runtime.replay import load_checkpoint, replay_to
from kogwistar.runtime.resolvers import AsyncMappingStepResolver, MappingStepResolver
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.ci, pytest.mark.runtime, pytest.mark.runtime_bridge_parity]

pytestmark = [pytest.mark.ci]


@dataclass
class _Node:
    id: str
    label: str
    op: str
    metadata: dict


@dataclass
class _Edge:
    id: str
    label: str
    source_ids: list[str]
    target_ids: list[str]
    metadata: dict

    def safe_get_id(self):
        return self.id

    @property
    def priority(self):
        return int(self.metadata.get("wf_priority", 100))

    @property
    def multiplicity(self):
        return str(self.metadata.get("wf_multiplicity", "one"))

    @property
    def is_default(self):
        return bool(self.metadata.get("wf_is_default", False))

    @property
    def predicate(self):
        return self.metadata.get("wf_predicate")


@dataclass
class _JoinNode:
    id: str
    op: str
    terminal: bool
    fanout: bool
    metadata: dict

    def safe_get_id(self):
        return self.id


@dataclass
class _JoinEdge:
    id: str
    label: str
    predicate: str | None
    source_ids: list[str]
    target_ids: list[str]
    multiplicity: str
    is_default: bool
    metadata: dict

    def safe_get_id(self):
        return self.id


class _JoinConversationEngine:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.read = self
        self.write = self

    def add_node(self, n):
        self.nodes.append(n)
        return n

    def add_edge(self, e):
        self.edges.append(e)
        return e

    def get_nodes(self, *args, **kwargs):
        return self.nodes

    def get_edges(self, *args, **kwargs):
        return self.edges


class _JoinWorkflowEngine:
    def __init__(self, nodes: list[_JoinNode], edges: list[_JoinEdge]) -> None:
        self._nodes = nodes
        self._edges = edges
        self.read = self
        self.write = self

    def get_nodes(self, where=None, limit=5000, **kwargs):
        return self._nodes

    def get_edges(self, where=None, limit=20000, **kwargs):
        return self._edges


class _TraceEmitter:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, object]]] = []

    def step_started(self, ctx):
        self.emitted.append(("started", ctx.as_fields()))

    def step_completed(self, ctx, *, status, duration_ms, extra=None):
        payload = dict(ctx.as_fields())
        payload["status"] = status
        payload["duration_ms"] = duration_ms
        self.emitted.append(("completed", payload))


def _mk_nodes():
    return {
        "n|left": _Node(
            id="n|left",
            label="left_label",
            op="left_op",
            metadata={"wf_terminal": False, "wf_fanout": False},
        ),
        "n|right": _Node(
            id="n|right",
            label="right_label",
            op="right_op",
            metadata={"wf_terminal": False, "wf_fanout": False},
        ),
        "n|fallback": _Node(
            id="n|fallback",
            label="fallback_label",
            op="fallback_op",
            metadata={"wf_terminal": True, "wf_fanout": False},
        ),
    }


def _mk_fanout_edges():
    return [
        _Edge(
            id="e-left",
            label="go_left",
            source_ids=["start"],
            target_ids=["n|left"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "many",
                "wf_is_default": False,
                "wf_predicate": None,
            },
        ),
        _Edge(
            id="e-right",
            label="go_right",
            source_ids=["start"],
            target_ids=["n|right"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "many",
                "wf_is_default": False,
                "wf_predicate": None,
            },
        ),
    ]


def _mk_default_edges():
    return [
        _Edge(
            id="e-pred",
            label="pred_path",
            source_ids=["start"],
            target_ids=["n|left"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "one",
                "wf_is_default": False,
                "wf_predicate": "if_true",
            },
        ),
        _Edge(
            id="e-default",
            label="default_path",
            source_ids=["start"],
            target_ids=["n|fallback"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "one",
                "wf_is_default": True,
                "wf_predicate": "if_false",
            },
        ),
    ]


def _join_node(
    node_id: str,
    *,
    workflow_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
    join: bool = False,
) -> _JoinNode:
    md = {
        "entity_type": "workflow_node",
        "workflow_id": workflow_id,
        "wf_op": op,
        "wf_version": "v1",
        "wf_start": bool(start),
        "wf_terminal": bool(terminal),
        "wf_fanout": bool(fanout),
    }
    if join:
        md["wf_join"] = True
    return _JoinNode(
        id=node_id,
        metadata=md,
        op=op,
        terminal=bool(terminal),
        fanout=bool(fanout),
    )


def _join_edge(
    edge_id: str,
    *,
    workflow_id: str,
    src: str,
    dst: str,
    predicate: str | None = None,
    priority: int = 100,
    is_default: bool = False,
    multiplicity: str = "one",
) -> _JoinEdge:
    md = {
        "entity_type": "workflow_edge",
        "workflow_id": workflow_id,
        "wf_predicate": predicate,
        "wf_priority": priority,
        "wf_is_default": bool(is_default),
        "wf_multiplicity": multiplicity,
    }
    return _JoinEdge(
        id=edge_id,
        label=f"{src} to {dst}",
        predicate=predicate,
        source_ids=[src],
        target_ids=[dst],
        multiplicity=multiplicity,
        is_default=bool(is_default),
        metadata=md,
    )


def test_runtime_parity_bridge_route_next_shared_semantics():
    """Bridge parity: sync `_route_next` and async native edge selection share explicit alias, fanout, and default fallback semantics."""
    nodes = _mk_nodes()

    sync_rt = WorkflowRuntime.__new__(WorkflowRuntime)
    sync_rt.predicate_registry = {}

    fanout_result = RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=["go_left", "right_op"],
    )
    sync_next, sync_decision = sync_rt._route_next(
        edges=_mk_fanout_edges(),
        state={},
        last_result=fanout_result,
        fanout=True,
        nodes=nodes,
    )

    async_node = _Node(
        id="start",
        label="start",
        op="start",
        metadata={"wf_fanout": True},
    )
    async_edges = AsyncWorkflowRuntime._select_next_edges(
        async_node,
        _mk_fanout_edges(),
        {},
        fanout_result,
        {},
        nodes=nodes,
    )
    async_next = [str(edge.target_ids[0]) for edge in async_edges]

    assert sync_next == ["n|left", "n|right"]
    assert async_next == sync_next
    assert sync_decision.selected == [
        ("e-left", "n|left", "explicit"),
        ("e-right", "n|right", "explicit"),
    ]

    default_result = RunSuccess(conversation_node_id=None, state_update=[])
    sync_default_next, sync_default_decision = sync_rt._route_next(
        edges=_mk_default_edges(),
        state={},
        last_result=default_result,
        fanout=False,
        nodes=nodes,
    )
    async_default_node = _Node(
        id="start",
        label="start",
        op="start",
        metadata={"wf_fanout": False},
    )
    async_default_edges = AsyncWorkflowRuntime._select_next_edges(
        async_default_node,
        _mk_default_edges(),
        {},
        default_result,
        {},
        nodes=nodes,
    )
    async_default_next = [str(edge.target_ids[0]) for edge in async_default_edges]

    assert sync_default_next == ["n|fallback"]
    assert async_default_next == sync_default_next
    assert sync_default_decision.selected == [
        ("e-default", "n|fallback", "default")
    ]


def _seed_join_bridge_graph(workflow_id: str):
    nodes = [
        _join_node("start", workflow_id=workflow_id, op="start", start=True, fanout=True),
        _join_node("a", workflow_id=workflow_id, op="a"),
        _join_node("b", workflow_id=workflow_id, op="b"),
        _join_node("join", workflow_id=workflow_id, op="join", join=True),
        _join_node("end", workflow_id=workflow_id, op="end", terminal=True),
    ]
    edges = [
        _join_edge("e1", workflow_id=workflow_id, src="start", dst="a"),
        _join_edge("e2", workflow_id=workflow_id, src="start", dst="b"),
        _join_edge("e3", workflow_id=workflow_id, src="a", dst="join"),
        _join_edge("e4", workflow_id=workflow_id, src="b", dst="join"),
        _join_edge("e5", workflow_id=workflow_id, src="join", dst="end"),
    ]
    return nodes, edges


def _normalize_join_bridge_state(state: dict) -> dict:
    return {
        "parts": sorted(list(state.get("parts", []) or [])),
        "join_calls": int(state.get("join_calls", 0)),
        "done": bool(state.get("done")),
    }


def _seed_trace_bridge_graph(workflow_id: str):
    nodes = [
        _join_node("start", workflow_id=workflow_id, op="start", start=True, terminal=True),
    ]
    return nodes, []


def _normalize_trace_payloads(emitted: list[tuple[str, dict[str, object]]]) -> dict:
    kinds = [kind for kind, _payload in emitted]
    started = emitted[0][1]
    completed = emitted[1][1]
    return {
        "kinds": kinds,
        "node_id": started["node_id"],
        "trace_id_matches_run": started["trace_id"] == started["run_id"],
        "same_run_id": started["run_id"] == completed["run_id"],
        "completed_status": completed["status"],
        "duration_non_negative": int(completed["duration_ms"]) >= 0,
    }


def _run_sync_join_bridge() -> dict:
    workflow_id = "wf_join_bridge_sync"
    nodes, edges = _seed_join_bridge_graph(workflow_id)
    engine = _JoinWorkflowEngine(nodes, edges)
    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "a"})])

    @resolver.register("b")
    def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "b"})])

    @resolver.register("join")
    def _join(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"join_calls": int(ctx.state_view.get("join_calls", 0)) + 1})],
        )

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"done": True})])

    runtime = WorkflowRuntime(
        workflow_engine=engine,
        conversation_engine=_JoinConversationEngine(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=2,
    )
    out = runtime.run(
        workflow_id=workflow_id,
        conversation_id="conv-join-sync",
        turn_node_id="turn-join-sync",
        initial_state={},
    )
    assert out.status == "succeeded"
    return _normalize_join_bridge_state(dict(out.final_state or {}))


def _run_async_join_bridge(monkeypatch) -> dict:
    workflow_id = "wf_join_bridge_async"

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return type("RunResultShim", (), {
                "run_id": str(kwargs.get("run_id") or "sync-run"),
                "final_state": dict(kwargs.get("initial_state") or {}),
                "status": "succeeded",
            })()

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        nodes, edges = _seed_join_bridge_graph(workflow_id)
        node_map = {node.id: node for node in nodes}
        adj = {node.id: [] for node in nodes}
        for edge in edges:
            adj[str(edge.source_ids[0])].append(edge)
        return node_map["start"], node_map, adj

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "a"})])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "b"})])

    @resolver.register("join")
    async def _join(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"join_calls": int(ctx.state_view.get("join_calls", 0)) + 1})],
        )

    @resolver.register("end")
    async def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"done": True})])

    runtime = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id=workflow_id,
            conversation_id="conv-join-async",
            turn_node_id="turn-join-async",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    return _normalize_join_bridge_state(dict(out.final_state or {}))


def _run_sync_trace_bridge() -> tuple[dict, dict]:
    workflow_id = "wf_trace_bridge_sync"
    nodes, edges = _seed_trace_bridge_graph(workflow_id)
    engine = _JoinWorkflowEngine(nodes, edges)
    emitter = _TraceEmitter()
    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"path": "trace"})])

    runtime = WorkflowRuntime(
        workflow_engine=engine,
        conversation_engine=_JoinConversationEngine(),
        step_resolver=resolver,
        predicate_registry={},
        trace=True,
        events=emitter,
    )
    out = runtime.run(
        workflow_id=workflow_id,
        conversation_id="conv-trace-sync",
        turn_node_id="turn-trace-sync",
        initial_state={},
    )
    return dict(out.final_state or {}), _normalize_trace_payloads(emitter.emitted)


def _run_async_trace_bridge(monkeypatch) -> tuple[dict, dict]:
    emitted: list[tuple[str, dict[str, object]]] = []

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.emitter = kwargs["events"]

        def run(self, **kwargs):
            return type("RunResultShim", (), {
                "run_id": str(kwargs.get("run_id") or "sync-run"),
                "final_state": dict(kwargs.get("initial_state") or {}),
                "status": "succeeded",
            })()

        def _should_step_uow(self, *args, **kwargs):
            return False

        def _maybe_step_uow(self):
            from contextlib import nullcontext

            return nullcontext()

        def _persist_step_exec(self, **kwargs):
            return object()

        def _persist_checkpoint(self, **kwargs):
            return None

    class _FakeEmitter:
        sink = None

        def step_started(self, ctx):
            emitted.append(("started", ctx.as_fields()))

        def step_completed(self, ctx, *, status, duration_ms, extra=None):
            payload = dict(ctx.as_fields())
            payload["status"] = status
            payload["duration_ms"] = duration_ms
            emitted.append(("completed", payload))

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=True, fanout=False)
        return start, {"start": start}, {"start": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"path": "trace"})])

    runtime = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=True,
        events=_FakeEmitter(),
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id="wf-trace-async",
            conversation_id="conv-trace-async",
            turn_node_id="turn-trace-async",
            initial_state={},
        )
    )
    return dict(out.final_state or {}), _normalize_trace_payloads(emitted)


def test_runtime_parity_bridge_join_barrier_waits_for_all_arrivals(monkeypatch):
    """Bridge parity: sync and async explicit join release once after all required arrivals, with same merged branch state."""
    sync_out = _run_sync_join_bridge()
    async_out = _run_async_join_bridge(monkeypatch)

    assert sync_out == {"parts": ["a", "b"], "join_calls": 1, "done": True}
    assert async_out == sync_out


def test_runtime_parity_bridge_trace_sink_parallel_nested_minimal(monkeypatch):
    """Bridge parity: sync and async emit compatible minimal trace step metadata for terminal execution."""
    sync_state, sync_trace = _run_sync_trace_bridge()
    async_state, async_trace = _run_async_trace_bridge(monkeypatch)

    assert sync_state["path"] == "trace"
    assert async_state == sync_state
    assert sync_trace == {
        "kinds": ["started", "completed"],
        "node_id": "start",
        "trace_id_matches_run": True,
        "same_run_id": True,
        "completed_status": "ok",
        "duration_non_negative": True,
    }
    assert async_trace == sync_trace


def _dummy_grounding() -> Grounding:
    sp = Span(
        collection_page_url="demo",
        document_page_url="demo",
        doc_id="demo",
        insertion_method="demo",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system",
            is_verified=True,
            score=1.0,
            notes="demo",
        ),
    )
    return Grounding(spans=[sp])


def _make_engine(tmp_path: Path, graph_type: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type=graph_type,
        backend_factory=build_fake_backend,
        embedding_function=ConstantEmbeddingFunction(dim=16),
    )


def _add_runtime_node(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    node_id: str,
    op: str,
    *,
    start: bool = False,
    terminal: bool = False,
) -> None:
    engine.write.add_node(
        WorkflowNode(
            id=node_id,
            label=op,
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            mentions=[_dummy_grounding()],
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": wf_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _add_runtime_edge(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    edge_id: str,
    src: str,
    dst: str,
    *,
    predicate: str | None = None,
    priority: int = 100,
    is_default: bool = True,
    multiplicity: str = "one",
) -> None:
    engine.write.add_edge(
        WorkflowEdge(
            id=edge_id,
            label="wf_next",
            type="entity",
            doc_id=edge_id,
            summary="next",
            properties={},
            source_ids=[src],
            target_ids=[dst],
            source_edge_ids=[],
            target_edge_ids=[],
            relation="wf_next",
            mentions=[_dummy_grounding()],
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": wf_id,
                "wf_predicate": predicate,
                "wf_priority": priority,
                "wf_is_default": is_default,
                "wf_multiplicity": multiplicity,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _latest_checkpoint_state(conv_engine: GraphKnowledgeEngine, run_id: str) -> dict:
    ckpts = conv_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1)))
    state_json = latest.metadata.get("state_json", {})
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    return dict(state_json or {})


def _seed_suspend_resume_workflow(engine: GraphKnowledgeEngine, wf_id: str) -> None:
    _add_runtime_node(engine, wf_id, "start", "start", start=True)
    _add_runtime_node(engine, wf_id, "gate", "gate")
    _add_runtime_node(engine, wf_id, "end", "end", terminal=True)
    _add_runtime_edge(engine, wf_id, "start->gate", "start", "gate")
    _add_runtime_edge(engine, wf_id, "gate->end", "gate", "end")


def _seed_linear_progress_workflow(engine: GraphKnowledgeEngine, wf_id: str) -> None:
    _add_runtime_node(engine, wf_id, "a", "a", start=True)
    _add_runtime_node(engine, wf_id, "b", "b")
    _add_runtime_node(engine, wf_id, "end", "end", terminal=True)
    _add_runtime_edge(engine, wf_id, "a->b", "a", "b")
    _add_runtime_edge(engine, wf_id, "b->end", "b", "end")


def _seed_gate_resume_workflow(engine: GraphKnowledgeEngine, wf_id: str) -> None:
    _add_runtime_node(engine, wf_id, "gate", "gate", start=True)
    _add_runtime_node(engine, wf_id, "a", "a")
    _add_runtime_node(engine, wf_id, "b", "b")
    _add_runtime_node(engine, wf_id, "end", "end", terminal=True)
    _add_runtime_edge(
        engine,
        wf_id,
        "gate->b",
        "gate",
        "b",
        predicate="has_done_a",
        priority=0,
        is_default=False,
    )
    _add_runtime_edge(engine, wf_id, "gate->a|default", "gate", "a")
    _add_runtime_edge(engine, wf_id, "a->gate", "a", "gate")
    _add_runtime_edge(engine, wf_id, "b->end", "b", "end")


def _initial_state(conv_id: str) -> dict:
    return {
        "conversation_id": conv_id,
        "user_id": "test",
        "turn_node_id": "turn_1",
        "turn_index": 0,
        "role": "user",
        "user_text": "",
        "mem_id": "mem_1",
    }


def _normalize_suspend_resume_result(
    *,
    suspended_status: str,
    checkpoint_state: dict,
    resumed_status: str,
    final_state: dict,
) -> dict:
    rt_join = dict(final_state.get("_rt_join", {}) or {})
    return {
        "suspended_status": suspended_status,
        "checkpoint_wait_reason": checkpoint_state.get("wait_reason"),
        "checkpoint_suspended_nodes": [item[0] for item in checkpoint_state.get("_rt_join", {}).get("suspended", [])],
        "resumed_status": resumed_status,
        "started": bool(final_state.get("started")),
        "resumed": bool(final_state.get("resumed")),
        "ended": bool(final_state.get("ended")),
        "final_suspended_nodes": [item[0] for item in rt_join.get("suspended", [])],
    }


def _resolve_progress_step(op: str):
    def _fn(ctx):
        with ctx.state_write as state:
            state.setdefault("op_log", [])
            state["op_log"].append(op)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("a", {"op": op}),
                ("u", {f"result.{op}": {"value": f"v_{op}"}}),
            ],
        )

    return _fn


def _normalize_checkpoint_replay_result(*, run_result, ckpt0: dict, reconstructed: dict) -> dict:
    rt_join = dict(ckpt0.get("_rt_join", {}) or {})
    pending = list(rt_join.get("pending", []) or [])
    return {
        "status": str(run_result.status),
        "final_op_log": list((run_result.final_state or {}).get("op_log", []) or []),
        "ckpt_op_log": list(ckpt0.get("op_log", []) or []),
        "ckpt_result_a": dict(ckpt0.get("result.a", {}) or {}),
        "ckpt_pending_nodes": [str(item[0]) for item in pending if isinstance(item, (list, tuple)) and item],
        "replay_op_log": list(reconstructed.get("op_log", []) or []),
        "replay_result_end": dict(reconstructed.get("result.end", {}) or {}),
    }


def _normalize_resume_result(*, first_result, checkpoint_state: dict, resumed_result) -> dict:
    rt_join = dict(checkpoint_state.get("_rt_join", {}) or {})
    pending = list(rt_join.get("pending", []) or [])
    final_state = dict(resumed_result.final_state or {})
    return {
        "first_status": str(first_result.status),
        "resume_status": str(resumed_result.status),
        "checkpoint_pending_nodes": [str(item[0]) for item in pending if isinstance(item, (list, tuple)) and item],
        "final_op_log": list(final_state.get("op_log", []) or []),
        "a_count": list(final_state.get("op_log", []) or []).count("a"),
        "result_a": dict(final_state.get("result.a", {}) or {}),
        "result_b": dict(final_state.get("result.b", {}) or {}),
        "result_end": dict(final_state.get("result.end", {}) or {}),
    }


def _run_sync_wait_reason_suspend_resume(tmp_path: Path) -> dict:
    wf_id = "wf_wait_bridge_sync"
    run_id = "run_wait_bridge_sync"
    conv_id = "conv_wait_bridge_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_suspend_resume_workflow(wf_engine, wf_id)

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("gate")
    def _gate(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            wait_reason="approval",
            resume_payload={"type": "recoverable_error", "category": "approval"},
        )

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"ended": True})],
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    out1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state=_initial_state(conv_id),
        run_id=run_id,
    )
    checkpoint_state = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = checkpoint_state["_rt_join"]["suspended"][0][2]
    out2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=suspended_token_id,
        client_result=RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"resumed": True})],
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )
    return _normalize_suspend_resume_result(
        suspended_status=out1.status,
        checkpoint_state=checkpoint_state,
        resumed_status=out2.status,
        final_state=dict(out2.final_state or {}),
    )


def _run_async_wait_reason_suspend_resume(tmp_path: Path) -> dict:
    wf_id = "wf_wait_bridge_async"
    run_id = "run_wait_bridge_async"
    conv_id = "conv_wait_bridge_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_suspend_resume_workflow(wf_engine, wf_id)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("gate")
    def _gate(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            wait_reason="approval",
            resume_payload={"type": "recoverable_error", "category": "approval"},
        )

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"ended": True})],
        )

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )

    out1 = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state=_initial_state(conv_id),
            run_id=run_id,
        )
    )
    checkpoint_state = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = checkpoint_state["_rt_join"]["suspended"][0][2]
    out2 = asyncio.run(
        runtime.resume_run(
            run_id=run_id,
            suspended_node_id="gate",
            suspended_token_id=suspended_token_id,
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"resumed": True})],
            ),
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
        )
    )
    return _normalize_suspend_resume_result(
        suspended_status=out1.status,
        checkpoint_state=checkpoint_state,
        resumed_status=out2.status,
        final_state=dict(out2.final_state or {}),
    )


def _run_sync_checkpoint_load_and_replay(tmp_path: Path) -> dict:
    wf_id = "wf_checkpoint_bridge_sync"
    run_id = "run_checkpoint_bridge_sync"
    conv_id = "conv_checkpoint_bridge_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_linear_progress_workflow(wf_engine, wf_id)

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_resolve_progress_step,
        predicate_registry={},
        checkpoint_every_n_steps=9999,
        max_workers=1,
    )
    out = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={},
        run_id=run_id,
    )
    ckpt0 = load_checkpoint(conversation_engine=conv_engine, run_id=run_id, step_seq=0)
    reconstructed = replay_to(conversation_engine=conv_engine, run_id=run_id, target_step_seq=2)
    return _normalize_checkpoint_replay_result(run_result=out, ckpt0=ckpt0, reconstructed=reconstructed)


def _run_async_checkpoint_load_and_replay(tmp_path: Path) -> dict:
    wf_id = "wf_checkpoint_bridge_async"
    run_id = "run_checkpoint_bridge_async"
    conv_id = "conv_checkpoint_bridge_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_linear_progress_workflow(wf_engine, wf_id)

    resolver = AsyncMappingStepResolver()

    @resolver.register("a")
    async def _a(ctx):
        return _resolve_progress_step("a")(ctx)

    @resolver.register("b")
    async def _b(ctx):
        return _resolve_progress_step("b")(ctx)

    @resolver.register("end")
    async def _end(ctx):
        return _resolve_progress_step("end")(ctx)

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=9999,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={},
            run_id=run_id,
        )
    )
    ckpt0 = load_checkpoint(conversation_engine=conv_engine, run_id=run_id, step_seq=0)
    reconstructed = replay_to(conversation_engine=conv_engine, run_id=run_id, target_step_seq=2)
    return _normalize_checkpoint_replay_result(run_result=out, ckpt0=ckpt0, reconstructed=reconstructed)


def _run_sync_resume_from_checkpoint(tmp_path: Path) -> dict:
    wf_id = "wf_resume_bridge_sync"
    conv_id = "conv_resume_bridge_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_gate_resume_workflow(wf_engine, wf_id)

    def done_a(_engine, state, _result):
        return "result.a" in state

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_resolve_progress_step,
        predicate_registry={"has_done_a": done_a},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    out1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={},
        run_id="run_resume_bridge_sync_1",
    )
    ckpt1 = load_checkpoint(conversation_engine=conv_engine, run_id=out1.run_id, step_seq=1)
    checkpoint_snapshot = copy.deepcopy(ckpt1)
    out2 = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"{conv_id}_2",
        turn_node_id="turn_2",
        initial_state=ckpt1,
        run_id="run_resume_bridge_sync_2",
    )
    return _normalize_resume_result(
        first_result=out1,
        checkpoint_state=checkpoint_snapshot,
        resumed_result=out2,
    )


def _run_async_resume_from_checkpoint(tmp_path: Path) -> dict:
    wf_id = "wf_resume_bridge_async"
    conv_id = "conv_resume_bridge_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_gate_resume_workflow(wf_engine, wf_id)

    def done_a(_engine, state, _result):
        return "result.a" in state

    resolver = AsyncMappingStepResolver()

    @resolver.register("gate")
    async def _gate(ctx):
        return _resolve_progress_step("gate")(ctx)

    @resolver.register("a")
    async def _a(ctx):
        return _resolve_progress_step("a")(ctx)

    @resolver.register("b")
    async def _b(ctx):
        return _resolve_progress_step("b")(ctx)

    @resolver.register("end")
    async def _end(ctx):
        return _resolve_progress_step("end")(ctx)

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"has_done_a": done_a},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    out1 = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={},
            run_id="run_resume_bridge_async_1",
        )
    )
    ckpt1 = load_checkpoint(conversation_engine=conv_engine, run_id=out1.run_id, step_seq=1)
    checkpoint_snapshot = copy.deepcopy(ckpt1)
    out2 = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=f"{conv_id}_2",
            turn_node_id="turn_2",
            initial_state=ckpt1,
            run_id="run_resume_bridge_async_2",
        )
    )
    return _normalize_resume_result(
        first_result=out1,
        checkpoint_state=checkpoint_snapshot,
        resumed_result=out2,
    )


def _run_sync_resume_from_checkpoint_frontier(tmp_path: Path) -> dict:
    wf_id = "wf_frontier_bridge_sync"
    conv_id = "conv_frontier_bridge_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_linear_progress_workflow(wf_engine, wf_id)

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_resolve_progress_step,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    out1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={},
        run_id="run_frontier_bridge_sync_1",
    )
    ckpt0 = load_checkpoint(conversation_engine=conv_engine, run_id=out1.run_id, step_seq=0)
    checkpoint_snapshot = copy.deepcopy(ckpt0)
    out2 = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"{conv_id}_2",
        turn_node_id="turn_2",
        initial_state=ckpt0,
        run_id="run_frontier_bridge_sync_2",
    )
    return _normalize_resume_result(
        first_result=out1,
        checkpoint_state=checkpoint_snapshot,
        resumed_result=out2,
    )


def _run_async_resume_from_checkpoint_frontier(tmp_path: Path) -> dict:
    wf_id = "wf_frontier_bridge_async"
    conv_id = "conv_frontier_bridge_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_linear_progress_workflow(wf_engine, wf_id)

    resolver = AsyncMappingStepResolver()

    @resolver.register("a")
    async def _a(ctx):
        return _resolve_progress_step("a")(ctx)

    @resolver.register("b")
    async def _b(ctx):
        return _resolve_progress_step("b")(ctx)

    @resolver.register("end")
    async def _end(ctx):
        return _resolve_progress_step("end")(ctx)

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    out1 = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={},
            run_id="run_frontier_bridge_async_1",
        )
    )
    ckpt0 = load_checkpoint(conversation_engine=conv_engine, run_id=out1.run_id, step_seq=0)
    checkpoint_snapshot = copy.deepcopy(ckpt0)
    out2 = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=f"{conv_id}_2",
            turn_node_id="turn_2",
            initial_state=ckpt0,
            run_id="run_frontier_bridge_async_2",
        )
    )
    return _normalize_resume_result(
        first_result=out1,
        checkpoint_state=checkpoint_snapshot,
        resumed_result=out2,
    )


def test_runtime_parity_bridge_wait_reason_suspend_resume():
    """Bridge parity: sync and async suspend/resume preserve wait reason, suspended token shape, and resumed terminal state."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"wait_reason_{uuid.uuid4().hex}"
    sync_out = _run_sync_wait_reason_suspend_resume(root / "sync")
    async_out = _run_async_wait_reason_suspend_resume(root / "async")

    assert sync_out == {
        "suspended_status": "suspended",
        "checkpoint_wait_reason": "approval",
        "checkpoint_suspended_nodes": ["gate"],
        "resumed_status": "succeeded",
        "started": True,
        "resumed": True,
        "ended": True,
        "final_suspended_nodes": [],
    }
    assert async_out == sync_out


def test_runtime_parity_bridge_checkpoint_load_and_replay():
    """Bridge parity: sync and async expose same normalized checkpoint payload and replayed state."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"checkpoint_{uuid.uuid4().hex}"
    sync_out = _run_sync_checkpoint_load_and_replay(root / "sync")
    async_out = _run_async_checkpoint_load_and_replay(root / "async")

    assert sync_out == {
        "status": "succeeded",
        "final_op_log": ["a", "b", "end"],
        "ckpt_op_log": ["a"],
        "ckpt_result_a": {"value": "v_a"},
        "ckpt_pending_nodes": ["b"],
        "replay_op_log": ["a"],
        "replay_result_end": {"value": "v_end"},
    }
    assert async_out == sync_out


def test_runtime_parity_bridge_resume_from_checkpoint():
    """Bridge parity: sync and async resume from loaded checkpoint without redoing completed work."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"resume_{uuid.uuid4().hex}"
    sync_out = _run_sync_resume_from_checkpoint(root / "sync")
    async_out = _run_async_resume_from_checkpoint(root / "async")

    assert sync_out == {
        "first_status": "succeeded",
        "resume_status": "succeeded",
        "checkpoint_pending_nodes": ["gate"],
        "final_op_log": ["gate", "a", "gate", "b", "end"],
        "a_count": 1,
        "result_a": {"value": "v_a"},
        "result_b": {"value": "v_b"},
        "result_end": {"value": "v_end"},
    }
    assert async_out == sync_out


def test_runtime_parity_bridge_resume_from_checkpoint_frontier():
    """Bridge parity: sync and async resume from frontier checkpoint at next pending node only."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"frontier_{uuid.uuid4().hex}"
    sync_out = _run_sync_resume_from_checkpoint_frontier(root / "sync")
    async_out = _run_async_resume_from_checkpoint_frontier(root / "async")

    assert sync_out == {
        "first_status": "succeeded",
        "resume_status": "succeeded",
        "checkpoint_pending_nodes": ["b"],
        "final_op_log": ["a", "b", "end"],
        "a_count": 1,
        "result_a": {"value": "v_a"},
        "result_b": {"value": "v_b"},
        "result_end": {"value": "v_end"},
    }
    assert async_out == sync_out


def _workflow_node_obj(
    wf_id: str,
    node_id: str,
    op: str,
    *,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
    join: bool = False,
) -> WorkflowNode:
    metadata = {
        "entity_type": "workflow_node",
        "workflow_id": wf_id,
        "wf_op": op,
        "wf_start": bool(start),
        "wf_terminal": bool(terminal),
        "wf_fanout": bool(fanout),
        "wf_version": "v1",
    }
    if join:
        metadata["wf_join"] = True
    return WorkflowNode(
        id=node_id,
        label=op,
        type="entity",
        doc_id=node_id,
        summary=op,
        properties={},
        mentions=[_dummy_grounding()],
        metadata=metadata,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )


def _workflow_edge_obj(
    wf_id: str,
    edge_id: str,
    src: str,
    dst: str,
    *,
    predicate: str | None = None,
    priority: int = 100,
    is_default: bool = True,
    multiplicity: str = "one",
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        label="wf_next",
        type="entity",
        doc_id=edge_id,
        summary="next",
        properties={},
        source_ids=[src],
        target_ids=[dst],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="wf_next",
        mentions=[_dummy_grounding()],
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": wf_id,
            "wf_predicate": predicate,
            "wf_priority": priority,
            "wf_is_default": bool(is_default),
            "wf_multiplicity": multiplicity,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )


def _persist_workflow_graph(
    engine: GraphKnowledgeEngine,
    *,
    nodes: list[WorkflowNode],
    edges: list[WorkflowEdge],
) -> None:
    for node in nodes:
        engine.write.add_node(node)
    for edge in edges:
        engine.write.add_edge(edge)


def _persisted_workflow_node_count(engine: GraphKnowledgeEngine, workflow_id: str) -> int:
    nodes = engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]},
        limit=100,
    )
    return len(nodes)


def _normalize_nested_design_result(
    *,
    result,
    workflow_engine: GraphKnowledgeEngine,
    child_workflow_id: str,
) -> dict:
    final_state = dict(result.final_state or {})
    child_state = dict(final_state.get("child_result", {}) or {})
    return {
        "status": str(result.status),
        "ended": bool(final_state.get("ended")),
        "child_done": bool(child_state.get("child_done")),
        "child_seed": child_state.get("child_seed"),
        "child_state_seed": child_state.get("seed"),
        "child_status": final_state.get("child_result__status"),
        "child_workflow_id": final_state.get("child_result__workflow_id"),
        "persisted_child_node_count": _persisted_workflow_node_count(
            workflow_engine, child_workflow_id
        ),
    }


def _build_nested_design_bridge_resolver(*, child_wf: str, child_design: WorkflowDesignArtifact):
    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["spawn"])

    @resolver.register("spawn")
    def _spawn(ctx):
        with ctx.state_write as state:
            state["spawn_seen"] = True
            state["seed"] = "propagated"
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"spawned": True})],
            _route_next=["end"],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=child_wf,
                    workflow_design=child_design,
                    result_state_key="child_result",
                    run_id="child-run",
                )
            ],
        )

    @resolver.register("child_body")
    def _child_body(ctx):
        with ctx.state_write as state:
            state["child_done"] = True
            state["child_seed"] = ctx.state_view.get("seed")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as state:
            state["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    return resolver


def _run_sync_nested_design_bridge(tmp_path: Path) -> dict:
    parent_wf = "wf_parent_bridge_nested"
    child_wf = "wf_child_bridge_nested"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _persist_workflow_graph(
        wf_engine,
        nodes=[
            _workflow_node_obj(parent_wf, "p|start", "start", start=True),
            _workflow_node_obj(parent_wf, "p|spawn", "spawn"),
            _workflow_node_obj(parent_wf, "p|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(parent_wf, "p|start->spawn", "p|start", "p|spawn"),
            _workflow_edge_obj(parent_wf, "p|spawn->end", "p|spawn", "p|end"),
        ],
    )
    child_design = WorkflowDesignArtifact(
        workflow_id=child_wf,
        workflow_version="v1",
        start_node_id="c|start",
        nodes=[
            _workflow_node_obj(child_wf, "c|start", "start", start=True),
            _workflow_node_obj(child_wf, "c|body", "child_body"),
            _workflow_node_obj(child_wf, "c|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(child_wf, "c|start->body", "c|start", "c|body"),
            _workflow_edge_obj(child_wf, "c|body->end", "c|body", "c|end"),
        ],
        source_run_id="parent-run",
        source_workflow_id=parent_wf,
        source_step_id="p|spawn",
        notes="bridge child design",
    )
    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_build_nested_design_bridge_resolver(
            child_wf=child_wf,
            child_design=child_design,
        ),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    out = runtime.run(
        workflow_id=parent_wf,
        conversation_id="conv-nested-sync",
        turn_node_id="turn-1",
        initial_state={"_deps": {}, "seed": "present"},
        run_id="run-nested-bridge",
    )
    return _normalize_nested_design_result(
        result=out,
        workflow_engine=wf_engine,
        child_workflow_id=child_wf,
    )


def _run_async_nested_design_bridge(tmp_path: Path) -> dict:
    parent_wf = "wf_parent_bridge_nested"
    child_wf = "wf_child_bridge_nested"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _persist_workflow_graph(
        wf_engine,
        nodes=[
            _workflow_node_obj(parent_wf, "p|start", "start", start=True),
            _workflow_node_obj(parent_wf, "p|spawn", "spawn"),
            _workflow_node_obj(parent_wf, "p|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(parent_wf, "p|start->spawn", "p|start", "p|spawn"),
            _workflow_edge_obj(parent_wf, "p|spawn->end", "p|spawn", "p|end"),
        ],
    )
    child_design = WorkflowDesignArtifact(
        workflow_id=child_wf,
        workflow_version="v1",
        start_node_id="c|start",
        nodes=[
            _workflow_node_obj(child_wf, "c|start", "start", start=True),
            _workflow_node_obj(child_wf, "c|body", "child_body"),
            _workflow_node_obj(child_wf, "c|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(child_wf, "c|start->body", "c|start", "c|body"),
            _workflow_edge_obj(child_wf, "c|body->end", "c|body", "c|end"),
        ],
        source_run_id="parent-run",
        source_workflow_id=parent_wf,
        source_step_id="p|spawn",
        notes="bridge child design",
    )
    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_build_nested_design_bridge_resolver(
            child_wf=child_wf,
            child_design=child_design,
        ),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id=parent_wf,
            conversation_id="conv-nested-async",
            turn_node_id="turn-1",
            initial_state={"_deps": {}, "seed": "present"},
            run_id="run-nested-bridge",
        )
    )
    return _normalize_nested_design_result(
        result=out,
        workflow_engine=wf_engine,
        child_workflow_id=child_wf,
    )


def test_runtime_parity_bridge_nested_workflow_synthesized_design_is_persisted_and_used():
    """Bridge parity: sync and async persist synthesized child workflow design and reuse normalized child result identically."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"nested_design_{uuid.uuid4().hex}"
    sync_out = _run_sync_nested_design_bridge(root / "sync")
    async_out = _run_async_nested_design_bridge(root / "async")

    assert sync_out == {
        "status": "succeeded",
        "ended": True,
        "child_done": True,
        "child_seed": "propagated",
        "child_state_seed": "propagated",
        "child_status": "succeeded",
        "child_workflow_id": "wf_child_bridge_nested",
        "persisted_child_node_count": 3,
    }
    assert async_out == sync_out


def _build_nested_failure_bridge_resolver(*, child_wf: str, child_design: WorkflowDesignArtifact):
    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], _route_next=["spawn"])

    @resolver.register("spawn")
    def _spawn(ctx):
        with ctx.state_write as state:
            state["spawn_seen"] = True
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            _route_next=["end"],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=child_wf,
                    workflow_design=child_design,
                    result_state_key="child_result",
                    run_id="child-run",
                )
            ],
        )

    @resolver.register("boom")
    def _boom(_ctx):
        return RunFailure(
            conversation_node_id=None,
            state_update=[],
            errors=["child boom"],
        )

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as state:
            state["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    return resolver


def _normalize_nested_failure_result(result) -> dict:
    final_state = dict(result.final_state or {})
    return {
        "status": str(result.status),
        "spawn_seen": bool(final_state.get("spawn_seen")),
        "ended": bool(final_state.get("ended")),
        "has_child_result": "child_result" in final_state,
    }


def _run_sync_nested_failure_bridge(tmp_path: Path) -> dict:
    parent_wf = "wf_parent_bridge_nested_failure"
    child_wf = "wf_child_bridge_nested_failure"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _persist_workflow_graph(
        wf_engine,
        nodes=[
            _workflow_node_obj(parent_wf, "p|start", "start", start=True),
            _workflow_node_obj(parent_wf, "p|spawn", "spawn"),
            _workflow_node_obj(parent_wf, "p|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(parent_wf, "p|start->spawn", "p|start", "p|spawn"),
            _workflow_edge_obj(parent_wf, "p|spawn->end", "p|spawn", "p|end"),
        ],
    )
    child_design = WorkflowDesignArtifact(
        workflow_id=child_wf,
        workflow_version="v1",
        start_node_id="c|start",
        nodes=[
            _workflow_node_obj(child_wf, "c|start", "start", start=True),
            _workflow_node_obj(child_wf, "c|boom", "boom"),
            _workflow_node_obj(child_wf, "c|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(child_wf, "c|start->boom", "c|start", "c|boom"),
            _workflow_edge_obj(child_wf, "c|boom->end", "c|boom", "c|end"),
        ],
    )
    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_build_nested_failure_bridge_resolver(
            child_wf=child_wf,
            child_design=child_design,
        ),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    out = runtime.run(
        workflow_id=parent_wf,
        conversation_id="conv-nested-fail-sync",
        turn_node_id="turn-1",
        initial_state={"_deps": {}},
        run_id="run-nested-fail-bridge",
    )
    return _normalize_nested_failure_result(out)


def _run_async_nested_failure_bridge(tmp_path: Path) -> dict:
    parent_wf = "wf_parent_bridge_nested_failure"
    child_wf = "wf_child_bridge_nested_failure"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _persist_workflow_graph(
        wf_engine,
        nodes=[
            _workflow_node_obj(parent_wf, "p|start", "start", start=True),
            _workflow_node_obj(parent_wf, "p|spawn", "spawn"),
            _workflow_node_obj(parent_wf, "p|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(parent_wf, "p|start->spawn", "p|start", "p|spawn"),
            _workflow_edge_obj(parent_wf, "p|spawn->end", "p|spawn", "p|end"),
        ],
    )
    child_design = WorkflowDesignArtifact(
        workflow_id=child_wf,
        workflow_version="v1",
        start_node_id="c|start",
        nodes=[
            _workflow_node_obj(child_wf, "c|start", "start", start=True),
            _workflow_node_obj(child_wf, "c|boom", "boom"),
            _workflow_node_obj(child_wf, "c|end", "end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(child_wf, "c|start->boom", "c|start", "c|boom"),
            _workflow_edge_obj(child_wf, "c|boom->end", "c|boom", "c|end"),
        ],
    )
    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_build_nested_failure_bridge_resolver(
            child_wf=child_wf,
            child_design=child_design,
        ),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id=parent_wf,
            conversation_id="conv-nested-fail-async",
            turn_node_id="turn-1",
            initial_state={"_deps": {}},
            run_id="run-nested-fail-bridge",
        )
    )
    return _normalize_nested_failure_result(out)


def test_runtime_parity_bridge_nested_workflow_child_failure_fails_parent():
    """Bridge parity: sync and async child workflow failure short-circuits parent routing with same normalized failure state."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"nested_failure_{uuid.uuid4().hex}"
    sync_out = _run_sync_nested_failure_bridge(root / "sync")
    async_out = _run_async_nested_failure_bridge(root / "async")

    assert sync_out == {
        "status": "failure",
        "spawn_seen": True,
        "ended": False,
        "has_child_result": False,
    }
    assert async_out == sync_out


def _normalize_roundtrip_resume_result(*, first_result, checkpoint_state: dict, resumed_result) -> dict:
    final_state = dict(resumed_result.final_state or {})
    return {
        "first_status": str(first_result.status),
        "suspended_nodes": [item[0] for item in checkpoint_state.get("_rt_join", {}).get("suspended", [])],
        "resume_status": str(resumed_result.status),
        "started": bool(final_state.get("started")),
        "pi": final_state.get("pi"),
        "ended": bool(final_state.get("ended")),
    }


def _run_sync_suspend_resume_roundtrip_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_roundtrip_bridge_sync"
    run_id = "run_roundtrip_bridge_sync"
    conv_id = "conv_roundtrip_bridge_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_suspend_resume_workflow(wf_engine, wf_id)
    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("gate")
    def _gate(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "calculate_pi"},
        )

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )
    first = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state=_initial_state(conv_id),
        run_id=run_id,
    )
    checkpoint_state = _latest_checkpoint_state(conv_engine, run_id)
    token_id = checkpoint_state["_rt_join"]["suspended"][0][2]
    resumed = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=token_id,
        client_result=RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"pi": 3.14})],
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )
    return _normalize_roundtrip_resume_result(
        first_result=first,
        checkpoint_state=checkpoint_state,
        resumed_result=resumed,
    )


def _run_async_suspend_resume_roundtrip_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_roundtrip_bridge_async"
    run_id = "run_roundtrip_bridge_async"
    conv_id = "conv_roundtrip_bridge_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_suspend_resume_workflow(wf_engine, wf_id)
    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("gate")
    def _gate(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "calculate_pi"},
        )

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    first = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state=_initial_state(conv_id),
            run_id=run_id,
        )
    )
    checkpoint_state = _latest_checkpoint_state(conv_engine, run_id)
    token_id = checkpoint_state["_rt_join"]["suspended"][0][2]
    resumed = asyncio.run(
        runtime.resume_run(
            run_id=run_id,
            suspended_node_id="gate",
            suspended_token_id=token_id,
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"pi": 3.14})],
            ),
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
        )
    )
    return _normalize_roundtrip_resume_result(
        first_result=first,
        checkpoint_state=checkpoint_state,
        resumed_result=resumed,
    )


def test_runtime_parity_bridge_workflow_suspend_and_resume_roundtrip():
    """Bridge parity: sync and async suspend checkpoint then resume to same normalized terminal state."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"roundtrip_{uuid.uuid4().hex}"
    sync_out = _run_sync_suspend_resume_roundtrip_bridge(root / "sync")
    async_out = _run_async_suspend_resume_roundtrip_bridge(root / "async")

    assert sync_out == {
        "first_status": "suspended",
        "suspended_nodes": ["gate"],
        "resume_status": "succeeded",
        "started": True,
        "pi": 3.14,
        "ended": True,
    }
    assert async_out == sync_out


def _seed_suspend_resume_branching_workflow(engine: GraphKnowledgeEngine, wf_id: str) -> None:
    engine.write.add_node(_workflow_node_obj(wf_id, "start", "start", start=True))
    engine.write.add_node(_workflow_node_obj(wf_id, "fork", "noop", fanout=True))
    engine.write.add_node(_workflow_node_obj(wf_id, "a", "suspend_op"))
    engine.write.add_node(_workflow_node_obj(wf_id, "b", "normal_b"))
    engine.write.add_node(_workflow_node_obj(wf_id, "join", "noop", join=True))
    engine.write.add_node(_workflow_node_obj(wf_id, "end", "end", terminal=True))
    edges = [
        ("start->fork", "start", "fork"),
        ("fork->a", "fork", "a"),
        ("fork->b", "fork", "b"),
        ("a->join", "a", "join"),
        ("b->join", "b", "join"),
        ("join->end", "join", "end"),
    ]
    for edge_id, src, dst in edges:
        engine.write.add_edge(_workflow_edge_obj(wf_id, edge_id, src, dst))


def _normalize_branching_resume_result(*, first_result, checkpoint_state: dict, resumed_result) -> dict:
    first_state = dict(first_result.final_state or {})
    final_state = dict(resumed_result.final_state or {})
    return {
        "first_status": str(first_result.status),
        "suspended_nodes": [item[0] for item in checkpoint_state.get("_rt_join", {}).get("suspended", [])],
        "b_done_before_resume": bool(first_state.get("b_done")),
        "resume_status": str(resumed_result.status),
        "a_done": bool(final_state.get("a_done")),
        "b_done": bool(final_state.get("b_done")),
        "ended": bool(final_state.get("ended")),
    }


def _run_sync_branching_suspend_resume_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_branch_roundtrip_sync"
    run_id = "run_branch_roundtrip_sync"
    conv_id = "conv_branch_roundtrip_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_suspend_resume_branching_workflow(wf_engine, wf_id)
    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("noop")
    def _noop(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("suspend_op")
    def _suspend(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "do_something"},
        )

    @resolver.register("normal_b")
    def _normal_b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"b_done": True})])

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=2,
    )
    first = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state=_initial_state(conv_id),
        run_id=run_id,
    )
    checkpoint_state = _latest_checkpoint_state(conv_engine, run_id)
    token_id = checkpoint_state["_rt_join"]["suspended"][0][2]
    resumed = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="a",
        suspended_token_id=token_id,
        client_result=RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"a_done": True})],
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )
    return _normalize_branching_resume_result(
        first_result=first,
        checkpoint_state=checkpoint_state,
        resumed_result=resumed,
    )


def _run_async_branching_suspend_resume_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_branch_roundtrip_async"
    run_id = "run_branch_roundtrip_async"
    conv_id = "conv_branch_roundtrip_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_suspend_resume_branching_workflow(wf_engine, wf_id)
    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])

    @resolver.register("noop")
    def _noop(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("suspend_op")
    def _suspend(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "do_something"},
        )

    @resolver.register("normal_b")
    def _normal_b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"b_done": True})])

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=2,
        experimental_native_scheduler=True,
    )
    first = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state=_initial_state(conv_id),
            run_id=run_id,
        )
    )
    checkpoint_state = _latest_checkpoint_state(conv_engine, run_id)
    token_id = checkpoint_state["_rt_join"]["suspended"][0][2]
    resumed = asyncio.run(
        runtime.resume_run(
            run_id=run_id,
            suspended_node_id="a",
            suspended_token_id=token_id,
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"a_done": True})],
            ),
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
        )
    )
    return _normalize_branching_resume_result(
        first_result=first,
        checkpoint_state=checkpoint_state,
        resumed_result=resumed,
    )


def test_runtime_parity_bridge_workflow_suspend_and_resume_branching_roundtrip():
    """Bridge parity: sync and async suspended branch/join frontier resumes to same normalized terminal state."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"branch_roundtrip_{uuid.uuid4().hex}"
    sync_out = _run_sync_branching_suspend_resume_bridge(root / "sync")
    async_out = _run_async_branching_suspend_resume_bridge(root / "async")

    assert sync_out == {
        "first_status": "suspended",
        "suspended_nodes": ["a"],
        "b_done_before_resume": True,
        "resume_status": "succeeded",
        "a_done": True,
        "b_done": True,
        "ended": True,
    }
    assert async_out == sync_out


def _seed_unreachable_join_workflow(engine: GraphKnowledgeEngine, wf_id: str) -> None:
    nodes = [
        _workflow_node_obj(wf_id, "start", "noop", start=True),
        _workflow_node_obj(wf_id, "fork", "noop", fanout=True),
        _workflow_node_obj(wf_id, "a", "fast"),
        _workflow_node_obj(wf_id, "end_a", "end_a", terminal=True),
        _workflow_node_obj(wf_id, "b", "slow"),
        _workflow_node_obj(wf_id, "join", "join", join=True),
        _workflow_node_obj(wf_id, "end", "end", terminal=True),
    ]
    edges = [
        _workflow_edge_obj(wf_id, "e1", "start", "fork"),
        _workflow_edge_obj(wf_id, "e2", "fork", "a"),
        _workflow_edge_obj(wf_id, "e3", "fork", "b"),
        _workflow_edge_obj(wf_id, "e4", "a", "end_a"),
        _workflow_edge_obj(wf_id, "e5", "b", "join"),
        _workflow_edge_obj(wf_id, "e6", "join", "end"),
    ]
    _persist_workflow_graph(engine, nodes=nodes, edges=edges)


def _normalize_unreachable_join_result(result) -> dict:
    final_state = dict(result.final_state or {})
    events = list(final_state.get("events", []) or [])

    def _idx(name: str) -> int:
        return events.index(name) if name in events else -1

    return {
        "status": str(result.status),
        "ended": bool(final_state.get("ended")),
        "event_set": sorted(set(events)),
        "join_before_end_a": _idx("join_done") >= 0 and _idx("end_a") >= 0 and _idx("join_done") < _idx("end_a"),
    }


def _build_sync_unreachable_join_resolver():
    resolver = MappingStepResolver()

    @resolver.register("noop")
    def _noop(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fast")
    def _fast(ctx):
        time.sleep(0.15)
        with ctx.state_write as state:
            state.setdefault("events", []).append("a_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end_a")
    def _end_a(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("end_a")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow")
    def _slow(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("b_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("join")
    def _join(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("join_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("end")
            state["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    return resolver


def _build_async_unreachable_join_resolver():
    resolver = AsyncMappingStepResolver()

    @resolver.register("noop")
    async def _noop(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fast")
    async def _fast(ctx):
        await asyncio.sleep(0.15)
        with ctx.state_write as state:
            state.setdefault("events", []).append("a_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end_a")
    async def _end_a(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("end_a")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow")
    async def _slow(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("b_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("join")
    async def _join(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("join_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    async def _end(ctx):
        with ctx.state_write as state:
            state.setdefault("events", []).append("end")
            state["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    return resolver


def _run_sync_unreachable_join_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_unreachable_join_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_unreachable_join_workflow(wf_engine, wf_id)
    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_build_sync_unreachable_join_resolver(),
        predicate_registry={},
        max_workers=2,
    )
    out = runtime.run(
        workflow_id=wf_id,
        conversation_id="conv-unreachable-sync",
        turn_node_id="turn-1",
        initial_state={},
        run_id="run-unreachable-bridge",
    )
    return _normalize_unreachable_join_result(out)


def _run_async_unreachable_join_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_unreachable_join_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_unreachable_join_workflow(wf_engine, wf_id)
    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=_build_async_unreachable_join_resolver(),
        predicate_registry={},
        max_workers=2,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id="conv-unreachable-async",
            turn_node_id="turn-1",
            initial_state={},
            run_id="run-unreachable-bridge",
        )
    )
    return _normalize_unreachable_join_result(out)


def test_runtime_parity_bridge_join_unreachable_branch_not_required():
    """Bridge parity: sync and async joins do not wait on branch outside reachable join region."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"unreachable_join_{uuid.uuid4().hex}"
    sync_out = _run_sync_unreachable_join_bridge(root / "sync")
    async_out = _run_async_unreachable_join_bridge(root / "async")

    assert sync_out["status"] == "succeeded"
    assert sync_out["ended"] is True
    assert sync_out["join_before_end_a"] is True
    assert async_out == sync_out


def _normalize_trace_metadata_payloads(emitted: list[tuple[str, dict[str, object]]]) -> dict:
    started = emitted[0][1]
    completed = emitted[1][1]
    return {
        "started_keys": sorted(
            key
            for key in (
                "run_id",
                "node_id",
                "conversation_id",
                "turn_node_id",
                "trace_id",
                "span_id",
                "parent_span_id",
            )
            if started.get(key)
        ),
        "completed_keys": sorted(
            key
            for key in (
                "run_id",
                "node_id",
                "conversation_id",
                "turn_node_id",
                "trace_id",
                "span_id",
                "parent_span_id",
                "status",
                "duration_ms",
            )
            if key in completed
        ),
        "same_ids": started["run_id"] == completed["run_id"]
        and started["node_id"] == completed["node_id"]
        and started["conversation_id"] == completed["conversation_id"]
        and started["turn_node_id"] == completed["turn_node_id"],
        "trace_matches_run": started["trace_id"] == started["run_id"],
        "completed_status": completed["status"],
        "duration_non_negative": int(completed["duration_ms"]) >= 0,
    }


def test_runtime_parity_bridge_runtime_trace_events_metadata_shape(monkeypatch):
    """Bridge parity: sync and async expose compatible trace event metadata keys and identifiers."""
    emitter_sync = _TraceEmitter()
    sync_nodes, sync_edges = _seed_trace_bridge_graph("wf_trace_bridge_metadata_sync")
    sync_runtime = WorkflowRuntime(
        workflow_engine=_JoinWorkflowEngine(sync_nodes, sync_edges),
        conversation_engine=_JoinConversationEngine(),
        step_resolver=MappingStepResolver({"start": lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])}),
        predicate_registry={},
        trace=True,
        events=emitter_sync,
    )
    sync_runtime.run(
        workflow_id="wf_trace_bridge_metadata_sync",
        conversation_id="conv-trace-meta",
        turn_node_id="turn-trace-meta",
        initial_state={},
    )
    _sync_state, _async_trace = _run_async_trace_bridge(monkeypatch)
    async_emitted = []
    async_state, _ = _run_async_trace_bridge(monkeypatch)
    del async_state
    async_emitted = []
    class _CaptureEmitter:
        sink = None
        def step_started(self, ctx):
            async_emitted.append(("started", ctx.as_fields()))
        def step_completed(self, ctx, *, status, duration_ms, extra=None):
            payload = dict(ctx.as_fields())
            payload["status"] = status
            payload["duration_ms"] = duration_ms
            async_emitted.append(("completed", payload))

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=True, fanout=False)
        return start, {"start": start}, {"start": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)
    runtime = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=AsyncMappingStepResolver({"start": lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])}),
        predicate_registry={},
        trace=True,
        events=_CaptureEmitter(),
        experimental_native_scheduler=True,
    )
    asyncio.run(
        runtime.run(
            workflow_id="wf_trace_bridge_metadata_async",
            conversation_id="conv-trace-meta",
            turn_node_id="turn-trace-meta",
            initial_state={},
        )
    )
    sync_out = _normalize_trace_metadata_payloads(emitter_sync.emitted)
    async_out = _normalize_trace_metadata_payloads(async_emitted)
    assert sync_out == {
        "started_keys": [
            "conversation_id",
            "node_id",
            "parent_span_id",
            "run_id",
            "span_id",
            "trace_id",
            "turn_node_id",
        ],
        "completed_keys": [
            "conversation_id",
            "duration_ms",
            "node_id",
            "parent_span_id",
            "run_id",
            "span_id",
            "status",
            "trace_id",
            "turn_node_id",
        ],
        "same_ids": True,
        "trace_matches_run": True,
        "completed_status": "ok",
        "duration_non_negative": True,
    }
    assert async_out == sync_out


def _seed_native_update_bridge_workflow(engine: GraphKnowledgeEngine, wf_id: str) -> None:
    _persist_workflow_graph(
        engine,
        nodes=[
            _workflow_node_obj(wf_id, "start", "start", start=True, terminal=True),
        ],
        edges=[],
    )


def _run_sync_native_update_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_native_update_bridge_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _seed_native_update_bridge_workflow(wf_engine, wf_id)
    resolver = MappingStepResolver()
    resolver.set_state_schema({"op_log": "a"})

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            update={"op_log": "x", "dyn": 1},
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        max_workers=1,
    )
    out = runtime.run(
        workflow_id=wf_id,
        conversation_id="conv-native-update-sync",
        turn_node_id="turn-1",
        initial_state={"op_log": []},
        run_id="run-native-update-bridge",
    )
    return {
        "status": str(out.status),
        "op_log": list((out.final_state or {}).get("op_log", []) or []),
        "dyn": (out.final_state or {}).get("dyn"),
    }


def _run_async_native_update_bridge(tmp_path: Path) -> dict:
    wf_id = "wf_native_update_bridge_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _seed_native_update_bridge_workflow(wf_engine, wf_id)
    resolver = AsyncMappingStepResolver()
    resolver.set_state_schema({"op_log": "a"})

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            update={"op_log": "x", "dyn": 1},
        )

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        max_workers=1,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        runtime.run(
            workflow_id=wf_id,
            conversation_id="conv-native-update-async",
            turn_node_id="turn-1",
            initial_state={"op_log": []},
            run_id="run-native-update-bridge",
        )
    )
    return {
        "status": str(out.status),
        "op_log": list((out.final_state or {}).get("op_log", []) or []),
        "dyn": (out.final_state or {}).get("dyn"),
    }


def test_runtime_parity_bridge_workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown():
    """Bridge parity: sync and async apply known native-update schema keys and fall back unknown keys compatibly."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"native_update_{uuid.uuid4().hex}"
    sync_out = _run_sync_native_update_bridge(root / "sync")
    async_out = _run_async_native_update_bridge(root / "async")

    assert sync_out == {"status": "succeeded", "op_log": ["x"], "dyn": 1}
    assert async_out == sync_out


def _normalize_parity_artifacts(conv_engine: GraphKnowledgeEngine, run_id: str) -> dict:
    nodes = conv_engine.read.get_nodes(where={"$and": [{"run_id": run_id}]}, limit=10_000)
    edges = conv_engine.read.get_edges(limit=10_000)
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    terminal_types: list[str] = []
    for node in nodes:
        md = dict(getattr(node, "metadata", {}) or {})
        entity_type = str(md.get("entity_type", ""))
        if entity_type.startswith("workflow_") and entity_type != "workflow_run":
            node_counts[entity_type] = int(node_counts.get(entity_type, 0)) + 1
            if entity_type in {"workflow_completed", "workflow_failed"}:
                terminal_types.append(entity_type)
    for edge in edges:
        md = dict(getattr(edge, "metadata", {}) or {})
        if md.get("run_id") != run_id:
            continue
        relation = str(getattr(edge, "relation", ""))
        edge_counts[relation] = int(edge_counts.get(relation, 0)) + 1
    return {
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "terminal_types": sorted(terminal_types),
    }


def _run_side_by_side_bridge(tmp_path: Path, terminal_case: str) -> tuple[dict, dict]:
    workflow_id = f"wf_side_by_side_bridge_{terminal_case}"
    run_id = f"run_side_by_side_bridge_{terminal_case}"
    sync_wf = _make_engine(tmp_path / "sync_wf", "workflow")
    sync_conv = _make_engine(tmp_path / "sync_conv", "conversation")
    async_wf = _make_engine(tmp_path / "async_wf", "workflow")
    async_conv = _make_engine(tmp_path / "async_conv", "conversation")
    graph_nodes = [
        _workflow_node_obj(workflow_id, "wf|start", "start", start=True),
        _workflow_node_obj(workflow_id, "wf|leaf", "leaf", terminal=True),
    ]
    graph_edges = [_workflow_edge_obj(workflow_id, "wf|start->leaf", "wf|start", "wf|leaf")]
    _persist_workflow_graph(sync_wf, nodes=graph_nodes, edges=graph_edges)
    _persist_workflow_graph(async_wf, nodes=graph_nodes, edges=graph_edges)

    def _resolve(op: str):
        def _fn(_ctx):
            if terminal_case == "failure":
                return RunFailure(
                    conversation_node_id=None,
                    state_update=[("u", {"last_op": op})],
                    errors=["boom"],
                )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"last_op": op})],
            )

        return _fn

    sync_runtime = WorkflowRuntime(
        workflow_engine=sync_wf,
        conversation_engine=sync_conv,
        step_resolver=_resolve,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    async_runtime = AsyncWorkflowRuntime(
        workflow_engine=async_wf,
        conversation_engine=async_conv,
        step_resolver=_resolve,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    sync_out = sync_runtime.run(
        workflow_id=workflow_id,
        conversation_id="conv-side-by-side",
        turn_node_id="turn-1",
        initial_state={},
        run_id=run_id,
    )
    async_out = asyncio.run(
        async_runtime.run(
            workflow_id=workflow_id,
            conversation_id="conv-side-by-side",
            turn_node_id="turn-1",
            initial_state={},
            run_id=run_id,
        )
    )
    sync_terminal = sync_conv.read.get_nodes(
        where={
            "$and": [
                {
                    "entity_type": (
                        "workflow_completed" if terminal_case == "success" else "workflow_failed"
                    )
                },
                {"run_id": run_id},
            ]
        },
        limit=10,
    )
    async_terminal = async_conv.read.get_nodes(
        where={
            "$and": [
                {
                    "entity_type": (
                        "workflow_completed" if terminal_case == "success" else "workflow_failed"
                    )
                },
                {"run_id": run_id},
            ]
        },
        limit=10,
    )
    sync_norm = {
        "status": str(sync_out.status),
        "last_op": (sync_out.final_state or {}).get("last_op"),
        **_normalize_parity_artifacts(sync_conv, run_id),
        "terminal_count": len(sync_terminal),
        "terminal_class": (
            isinstance(sync_terminal[0], WorkflowCompletedNode)
            if terminal_case == "success"
            else isinstance(sync_terminal[0], WorkflowFailedNode)
        ),
    }
    async_norm = {
        "status": str(async_out.status),
        "last_op": (async_out.final_state or {}).get("last_op"),
        **_normalize_parity_artifacts(async_conv, run_id),
        "terminal_count": len(async_terminal),
        "terminal_class": (
            isinstance(async_terminal[0], WorkflowCompletedNode)
            if terminal_case == "success"
            else isinstance(async_terminal[0], WorkflowFailedNode)
        ),
    }
    return sync_norm, async_norm


@pytest.mark.parametrize(
    "terminal_case, expected",
    [
        (
            "success",
            {
                "status": "succeeded",
                "last_op": "leaf",
                "terminal_types": ["workflow_completed"],
            },
        ),
        (
            "failure",
            {
                "status": "failure",
                "last_op": "start",
                "terminal_types": ["workflow_failed"],
            },
        ),
    ],
)
def test_runtime_parity_bridge_side_by_side_node_edge_and_terminal_parity(
    terminal_case: str,
    expected: dict[str, object],
):
    """Bridge parity: sync and async persist same normalized conversation node/edge effects and terminal node kind."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"side_by_side_{terminal_case}_{uuid.uuid4().hex}"
    sync_out, async_out = _run_side_by_side_bridge(root, terminal_case)

    assert sync_out["status"] == expected["status"]
    assert sync_out["last_op"] == expected["last_op"]
    assert sync_out["terminal_types"] == expected["terminal_types"]
    assert sync_out["terminal_count"] == 1
    assert sync_out["terminal_class"] is True
    assert async_out == sync_out


def _run_sync_parent_cancellation_bridge(tmp_path: Path) -> dict:
    child_wf = "wf_child_cancel_sync"
    wf_engine = _make_engine(tmp_path / "wf_sync", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_sync", "conversation")
    _persist_workflow_graph(
        wf_engine,
        nodes=[
            _workflow_node_obj(child_wf, "c|start", "child", start=True),
            _workflow_node_obj(child_wf, "c|end", "child_end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(child_wf, "c|start->end", "c|start", "c|end"),
        ],
    )
    cancel_flags = {"parent-run": False}
    seen_runs: list[str] = []

    def _cancel(run_id: str) -> bool:
        seen_runs.append(str(run_id))
        return bool(cancel_flags.get("parent-run", False) and str(run_id) in {"parent-run", "child-run"})

    resolver = MappingStepResolver()

    @resolver.register("child")
    def _child(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"child_started": True})])

    @resolver.register("child_end")
    def _child_end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"child_ended": True})])

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        cancel_requested=_cancel,
    )
    cancel_flags["parent-run"] = True
    _cancel("parent-run")
    out = runtime.run_subworkflow(
        workflow_id=child_wf,
        parent_state={},
        conversation_id="conv-parent-cancel-sync",
        turn_node_id="turn-1",
        parent_run_id="parent-run",
        result_state_key="child_result",
        run_id="child-run",
    )
    return {
        "status": str(out.status),
        "saw_parent": "parent-run" in seen_runs,
        "saw_child": "child-run" in seen_runs,
    }


def _run_async_parent_cancellation_bridge(tmp_path: Path) -> dict:
    child_wf = "wf_child_cancel_async"
    wf_engine = _make_engine(tmp_path / "wf_async", "workflow")
    conv_engine = _make_engine(tmp_path / "conv_async", "conversation")
    _persist_workflow_graph(
        wf_engine,
        nodes=[
            _workflow_node_obj(child_wf, "c|start", "child", start=True),
            _workflow_node_obj(child_wf, "c|end", "child_end", terminal=True),
        ],
        edges=[
            _workflow_edge_obj(child_wf, "c|start->end", "c|start", "c|end"),
        ],
    )
    cancel_flags = {"parent-run": False}
    seen_runs: list[str] = []

    def _cancel(run_id: str) -> bool:
        seen_runs.append(str(run_id))
        return bool(cancel_flags.get(str(run_id), False))

    resolver = AsyncMappingStepResolver()

    @resolver.register("child")
    async def _child(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"child_started": True})])

    @resolver.register("child_end")
    async def _child_end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"child_ended": True})])

    runtime = AsyncWorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
        cancel_requested=_cancel,
    )

    async def _run():
        cancel_flags["parent-run"] = True
        return await runtime._run_workflow_invocation_async(
            invocation=WorkflowInvocationRequest(
                workflow_id=child_wf,
                result_state_key="child_result",
                run_id="child-run",
            ),
            parent_state={},
            conversation_id="conv-parent-cancel-async",
            turn_node_id="turn-1",
            parent_run_id="parent-run",
        )

    out = asyncio.run(_run())
    return {
        "status": str(out.status),
        "saw_parent": "parent-run" in seen_runs,
        "saw_child": "child-run" in seen_runs,
    }


def test_runtime_parity_bridge_parent_cancellation_propagates_to_child():
    """Bridge parity: sync and async parent cancellation reaches child workflow run and lands same cancelled status."""
    root = Path.cwd() / ".tmp_runtime_parity_bridge" / f"parent_cancel_{uuid.uuid4().hex}"
    sync_out = _run_sync_parent_cancellation_bridge(root / "sync")
    async_out = _run_async_parent_cancellation_bridge(root / "async")

    assert sync_out == {"status": "cancelled", "saw_parent": True, "saw_child": True}
    assert async_out == sync_out
