from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Grounding, MentionVerification, Span
from graph_knowledge_engine.runtime.contract import BasePredicate
from graph_knowledge_engine.runtime.langgraph_converter import LGConverterOptions, to_langgraph
from graph_knowledge_engine.runtime.models import RunSuccess, RunSuspended, WorkflowEdge, WorkflowNode
from graph_knowledge_engine.runtime.resolvers import MappingStepResolver
from graph_knowledge_engine.runtime.runtime import StepContext, WorkflowRuntime


WORKFLOW_ID = "tutorial_runtime_pause_resume"
CONVERSATION_ID = "tutorial-runtime-conv"
TURN_NODE_ID = "tutorial-runtime-turn-1"
CDC_VIEWER_PATH = "graph_knowledge_engine/scripts/workflow.bundle.cdc.script.hl3.html"
RUNTIME_EVENT_ENDPOINT = "/api/workflow/runs/{run_id}/events"

warnings.filterwarnings("ignore", message=r"Using advanced underscore state key '_deps'.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r"Using advanced underscore state key '_rt_join'.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*PydanticSerializationUnexpectedValue.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic_extension\.model_slicing\.mixin")


class TinyEmbeddingFunction:
    _name="TinyEmbedding"
    def name(self):
        return self._name
    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [[float((len(str(text or "")) % 7) + 1)] for text in input]


class PredAlwaysTrue(BasePredicate):
    def __call__(self, e, state, result):  # type: ignore[override]
        return True


@dataclass
class _LGNode:
    id: str
    op: str
    terminal: bool
    fanout: bool
    metadata: dict[str, Any]

    def safe_get_id(self):
        return self.id


@dataclass
class _LGEdge:
    id: str
    label: str
    predicate: str | None
    source_ids: list[str]
    target_ids: list[str]
    multiplicity: str
    is_default: bool
    metadata: dict[str, Any]

    def safe_get_id(self):
        return self.id


class _LGWorkflowEngine:
    def __init__(self, nodes: list[_LGNode], edges: list[_LGEdge]) -> None:
        self._nodes = nodes
        self._edges = edges

    def get_nodes(self, where=None, limit=5000, **kwargs):
        return self._nodes

    def get_edges(self, where=None, limit=20000, **kwargs):
        return self._edges


class _LGRunResult:
    def __init__(self, state_update: list[tuple[str, dict[str, Any]]] | None = None, next_step_names: list[str] | None = None) -> None:
        self.state_update = state_update or []
        self.next_step_names = next_step_names or []


class _LGResolver:
    def __init__(self) -> None:
        self._state_schema = {"timeline": "a", "join_notes": "a", "custom_event_ops": "a"}
        self._handlers = {
            "start": lambda state: _LGRunResult([("u", {"started": True, "dep_echo": "runtime-tutorial"}), ("a", {"timeline": "start:audience=runtime-tutorial"}), ("a", {"custom_event_ops": "start"})]),
            "fork": lambda state: _LGRunResult([("u", {"fanout_seen": True}), ("a", {"timeline": "fork:fanout"})]),
            "branch_a_wait": lambda state: _LGRunResult([("u", {"branch_a_pending": True}), ("a", {"timeline": "branch_a_wait:suspended"}), ("a", {"custom_event_ops": "branch_a_wait"})]),
            "branch_b_complete": lambda state: _LGRunResult([("u", {"branch_b_done": True}), ("a", {"timeline": "branch_b_complete:completed"}), ("a", {"custom_event_ops": "branch_b_complete"})]),
            "join": lambda state: _LGRunResult([("u", {"joined": True}), ("a", {"join_notes": "join released"}), ("a", {"timeline": "join:released"})]),
            "end": lambda state: _LGRunResult([("u", {"ended": True}), ("a", {"timeline": "end:terminal"})]),
        }

    def resolve(self, op: str):
        return self._handlers[op]

    def describe_state(self) -> dict[str, str]:
        return dict(self._state_schema)


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _build_engines(data_dir: Path) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    data_dir.mkdir(parents=True, exist_ok=True)
    ef = TinyEmbeddingFunction()
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(data_dir / "workflow"),
        kg_graph_type="workflow",
        embedding_function=ef,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(data_dir / "conversation"),
        kg_graph_type="conversation",
        embedding_function=ef,
    )
    return workflow_engine, conversation_engine


def _span(doc_id: str, excerpt: str) -> Span:
    return Span(
        collection_page_url=f"tutorial/{doc_id}",
        document_page_url=f"tutorial/{doc_id}",
        doc_id=doc_id,
        insertion_method="tutorial_runtime",
        page_number=1,
        start_char=0,
        end_char=max(1, len(excerpt)),
        excerpt=excerpt[:256],
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="tutorial_runtime"),
    )


def _grounding(doc_id: str, excerpt: str) -> Grounding:
    return Grounding(spans=[_span(doc_id, excerpt)])


def _wf_node(*, node_id: str, op: str, start: bool = False, terminal: bool = False, fanout: bool = False, join: bool = False) -> WorkflowNode:
    summary = f"{node_id}::{op}"
    return WorkflowNode(
        id=node_id,
        label=node_id,
        type="entity",
        doc_id=f"wf-doc:{node_id}",
        summary=summary,
        mentions=[_grounding(f"wf-doc:{node_id}", summary)],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": WORKFLOW_ID,
            "wf_op": op,
            "wf_version": "v1",
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_fanout": fanout,
            "wf_join": join,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )


def _wf_edge(*, edge_id: str, src: str, dst: str, predicate: str | None = None, priority: int = 100, is_default: bool = False, multiplicity: str = "one") -> WorkflowEdge:
    summary = f"{src} -> {dst}"
    return WorkflowEdge(
        id=edge_id,
        label=edge_id,
        type="relationship",
        doc_id=f"wf-edge:{edge_id}",
        summary=summary,
        properties={},
        source_ids=[src],
        target_ids=[dst],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="wf_next",
        mentions=[_grounding(f"wf-edge:{edge_id}", summary)],
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": WORKFLOW_ID,
            "wf_predicate": predicate,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_multiplicity": multiplicity,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _workflow_seeded(workflow_engine: GraphKnowledgeEngine) -> bool:
    got = workflow_engine.backend.node_get(ids=["rt:start"], include=[])
    return bool(got.get("ids"))


def _resume_payload_for(node_id: str) -> dict[str, Any]:
    return {
        "action": "provide-approval",
        "node_id": node_id,
        "expected_result_key": "branch_a_result",
    }


def ensure_workflow_seed(data_dir: Path) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    workflow_engine, conversation_engine = _build_engines(data_dir)
    if _workflow_seeded(workflow_engine):
        return workflow_engine, conversation_engine

    nodes = [
        _wf_node(node_id="rt:start", op="start", start=True),
        _wf_node(node_id="rt:fork", op="fork", fanout=True),
        _wf_node(node_id="rt:branch_a", op="branch_a_wait"),
        _wf_node(node_id="rt:branch_b", op="branch_b_complete"),
        _wf_node(node_id="rt:join", op="join", join=True),
        _wf_node(node_id="rt:end", op="end", terminal=True),
    ]
    edges = [
        _wf_edge(edge_id="rt:start->fork", src="rt:start", dst="rt:fork", is_default=True, priority=100),
        _wf_edge(edge_id="rt:fork->branch_a", src="rt:fork", dst="rt:branch_a", predicate="always", priority=10, multiplicity="many"),
        _wf_edge(edge_id="rt:fork->branch_b", src="rt:fork", dst="rt:branch_b", predicate="always", priority=10, multiplicity="many"),
        _wf_edge(edge_id="rt:branch_a->join", src="rt:branch_a", dst="rt:join", is_default=True, priority=100),
        _wf_edge(edge_id="rt:branch_b->join", src="rt:branch_b", dst="rt:join", is_default=True, priority=100),
        _wf_edge(edge_id="rt:join->end", src="rt:join", dst="rt:end", is_default=True, priority=100),
    ]
    for node in nodes:
        workflow_engine.write.add_node(node)
    for edge in edges:
        workflow_engine.write.add_edge(edge)
    return workflow_engine, conversation_engine


def build_resolver() -> MappingStepResolver:
    resolver = MappingStepResolver()
    resolver.set_state_schema({"timeline": "a", "join_notes": "a", "custom_event_ops": "a"})

    @resolver.register("start")
    def _start(ctx: StepContext):
        deps = dict(ctx.state_view.get("_deps", {}) or {})
        audience = str(deps.get("audience", "runtime-tutorial"))
        if ctx.events is not None:
            ctx.events.emit(type="tutorial_resolver_note", ctx=ctx.trace_ctx, payload={"op": ctx.op, "audience": audience})
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("u", {"started": True, "dep_echo": audience}),
                ("a", {"timeline": f"{ctx.op}:audience={audience}"}),
                ("a", {"custom_event_ops": ctx.op}),
            ],
        )

    @resolver.register("fork")
    def _fork(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("u", {"fanout_seen": True}),
                ("a", {"timeline": f"{ctx.op}:fanout"}),
            ],
        )

    @resolver.register("branch_a_wait")
    def _branch_a_wait(ctx: StepContext):
        if ctx.events is not None:
            ctx.events.emit(type="tutorial_resolver_note", ctx=ctx.trace_ctx, payload={"op": ctx.op, "phase": "suspend"})
        return RunSuspended(
            conversation_node_id=None,
            state_update=[
                ("u", {"branch_a_pending": True}),
                ("a", {"timeline": f"{ctx.op}:suspended"}),
                ("a", {"custom_event_ops": ctx.op}),
            ],
            resume_payload=_resume_payload_for(ctx.workflow_node_id),
        )

    @resolver.register("branch_b_complete")
    def _branch_b_complete(ctx: StepContext):
        if ctx.events is not None:
            ctx.events.emit(type="tutorial_resolver_note", ctx=ctx.trace_ctx, payload={"op": ctx.op, "phase": "complete"})
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("u", {"branch_b_done": True}),
                ("a", {"timeline": f"{ctx.op}:completed"}),
                ("a", {"custom_event_ops": ctx.op}),
            ],
        )

    @resolver.register("join")
    def _join(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("u", {"joined": True}),
                ("a", {"join_notes": "join released"}),
                ("a", {"timeline": f"{ctx.op}:released"}),
            ],
        )

    @resolver.register("end")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("u", {"ended": True}),
                ("a", {"timeline": f"{ctx.op}:terminal"}),
            ],
        )

    return resolver


def build_runtime(data_dir: Path) -> tuple[WorkflowRuntime, MappingStepResolver, GraphKnowledgeEngine, GraphKnowledgeEngine]:
    workflow_engine, conversation_engine = ensure_workflow_seed(data_dir)
    resolver = build_resolver()
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={"always": PredAlwaysTrue()},
        checkpoint_every_n_steps=1,
        max_workers=2,
    )
    return runtime, resolver, workflow_engine, conversation_engine


def _base_initial_state() -> dict[str, Any]:
    return {
        "conversation_id": CONVERSATION_ID,
        "user_id": "tutorial-user",
        "turn_node_id": TURN_NODE_ID,
        "turn_index": 1,
        "role": "user",
        "user_text": "Explain runtime pause and continue.",
        "mem_id": "tutorial-mem",
        "self_span": {},
        "_deps": {"audience": "runtime-tutorial"},
    }


def _flush_trace(runtime: WorkflowRuntime) -> None:
    sink = getattr(runtime, "sink", None)
    if sink is not None:
        sink.close()


def _trace_db_path(workflow_engine: GraphKnowledgeEngine) -> Path:
    return Path(str(workflow_engine.persist_directory)) / "wf_trace.sqlite"


def _fetch_trace_events(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT ts_ms, type, step_seq, node_id, payload_json
            FROM wf_trace_events
            WHERE run_id = ?
            ORDER BY ts_ms ASC, step_seq ASC
            """,
            (run_id,),
        ).fetchall()
    finally:
        con.close()
    events: list[dict[str, Any]] = []
    for ts_ms, typ, step_seq, node_id, payload_json in rows:
        events.append(
            {
                "ts_ms": int(ts_ms),
                "type": str(typ),
                "step_seq": int(step_seq),
                "node_id": str(node_id),
                "payload": json.loads(payload_json) if payload_json else {},
            }
        )
    return events


def _workflow_nodes(conversation_engine: GraphKnowledgeEngine, entity_type: str, run_id: str) -> list[Any]:
    return conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": entity_type}, {"run_id": run_id}]},
        limit=1000,
    )


def _latest_checkpoint(conversation_engine: GraphKnowledgeEngine, run_id: str) -> tuple[Any | None, dict[str, Any]]:
    checkpoints = _workflow_nodes(conversation_engine, "workflow_checkpoint", run_id)
    if not checkpoints:
        return None, {}
    latest = max(checkpoints, key=lambda n: int(getattr(n, "metadata", {}).get("step_seq", -1)))
    state_json = getattr(latest, "metadata", {}).get("state_json", {})
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    return latest, dict(state_json or {})


def _build_langgraph_demo_engine() -> _LGWorkflowEngine:
    nodes = [
        _LGNode(id="rt_start", op="start", terminal=False, fanout=False, metadata={"wf_start": True, "wf_terminal": False, "wf_fanout": False}),
        _LGNode(id="rt_fork", op="fork", terminal=False, fanout=True, metadata={"wf_start": False, "wf_terminal": False, "wf_fanout": True}),
        _LGNode(id="rt_branch_a", op="branch_a_wait", terminal=False, fanout=False, metadata={"wf_start": False, "wf_terminal": False, "wf_fanout": False}),
        _LGNode(id="rt_branch_b", op="branch_b_complete", terminal=False, fanout=False, metadata={"wf_start": False, "wf_terminal": False, "wf_fanout": False}),
        _LGNode(id="rt_join", op="join", terminal=False, fanout=False, metadata={"wf_start": False, "wf_terminal": False, "wf_fanout": False, "wf_join": True}),
        _LGNode(id="rt_end", op="end", terminal=True, fanout=False, metadata={"wf_start": False, "wf_terminal": True, "wf_fanout": False}),
    ]
    edges = [
        _LGEdge(id="rt_start_to_fork", label="rt_start_to_fork", predicate=None, source_ids=["rt_start"], target_ids=["rt_fork"], multiplicity="one", is_default=True, metadata={"wf_predicate": None, "wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one"}),
        _LGEdge(id="rt_fork_to_branch_a", label="rt_fork_to_branch_a", predicate="always", source_ids=["rt_fork"], target_ids=["rt_branch_a"], multiplicity="many", is_default=False, metadata={"wf_predicate": "always", "wf_priority": 10, "wf_is_default": False, "wf_multiplicity": "many"}),
        _LGEdge(id="rt_fork_to_branch_b", label="rt_fork_to_branch_b", predicate="always", source_ids=["rt_fork"], target_ids=["rt_branch_b"], multiplicity="many", is_default=False, metadata={"wf_predicate": "always", "wf_priority": 10, "wf_is_default": False, "wf_multiplicity": "many"}),
        _LGEdge(id="rt_branch_a_to_join", label="rt_branch_a_to_join", predicate=None, source_ids=["rt_branch_a"], target_ids=["rt_join"], multiplicity="one", is_default=True, metadata={"wf_predicate": None, "wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one"}),
        _LGEdge(id="rt_branch_b_to_join", label="rt_branch_b_to_join", predicate=None, source_ids=["rt_branch_b"], target_ids=["rt_join"], multiplicity="one", is_default=True, metadata={"wf_predicate": None, "wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one"}),
        _LGEdge(id="rt_join_to_end", label="rt_join_to_end", predicate=None, source_ids=["rt_join"], target_ids=["rt_end"], multiplicity="one", is_default=True, metadata={"wf_predicate": None, "wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one"}),
    ]
    return _LGWorkflowEngine(nodes, edges)


def _run_to_suspension(data_dir: Path) -> tuple[dict[str, Any], WorkflowRuntime, MappingStepResolver, GraphKnowledgeEngine, GraphKnowledgeEngine]:
    runtime, resolver, workflow_engine, conversation_engine = build_runtime(data_dir)
    run_id = f"runtime-tutorial-{uuid.uuid4().hex}"
    run = runtime.run(
        workflow_id=WORKFLOW_ID,
        conversation_id=CONVERSATION_ID,
        turn_node_id=TURN_NODE_ID,
        initial_state=_base_initial_state(),
        run_id=run_id,
    )
    latest_ckpt, checkpoint_state = _latest_checkpoint(conversation_engine, run.run_id)
    return (
        {
            "run": run,
            "run_id": run.run_id,
            "latest_checkpoint": latest_ckpt,
            "checkpoint_state": checkpoint_state,
        },
        runtime,
        resolver,
        workflow_engine,
        conversation_engine,
    )


def reset_data(data_dir: Path) -> dict[str, Any]:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "action": "reset", "data_dir": str(data_dir)}


def level0_runtime_basics(data_dir: Path) -> dict[str, Any]:
    result, runtime, resolver, workflow_engine, conversation_engine = _run_to_suspension(data_dir)
    run = result["run"]
    run_id = result["run_id"]
    step_nodes = _workflow_nodes(conversation_engine, "workflow_step_exec", run_id)
    checkpoint_nodes = _workflow_nodes(conversation_engine, "workflow_checkpoint", run_id)
    _flush_trace(runtime)
    return {
        "level": 0,
        "workflow_id": WORKFLOW_ID,
        "canonical_workflow": "start -> fork -> branch_a(suspends) + branch_b(completes) -> join -> end",
        "run_id": run_id,
        "status": run.status,
        "final_state": run.final_state,
        "registered_ops": sorted(resolver.ops),
        "step_exec_count": len(step_nodes),
        "checkpoint_count": len(checkpoint_nodes),
        "trace_db_path": str(_trace_db_path(workflow_engine)),
        "checkpoint_pass": (
            run.status == "suspended"
            and run.final_state.get("started") is True
            and run.final_state.get("fanout_seen") is True
            and run.final_state.get("branch_b_done") is True
        ),
    }


def level1_resolvers_and_deps(data_dir: Path) -> dict[str, Any]:
    result, runtime, resolver, workflow_engine, _conversation_engine = _run_to_suspension(data_dir)
    run = result["run"]
    _flush_trace(runtime)
    events = _fetch_trace_events(_trace_db_path(workflow_engine), result["run_id"])
    custom_events = [evt for evt in events if evt["type"] == "tutorial_resolver_note"]
    return {
        "level": 1,
        "run_id": result["run_id"],
        "status": run.status,
        "dep_echo": run.final_state.get("dep_echo"),
        "state_schema": resolver.describe_state(),
        "registered_ops": sorted(resolver.ops),
        "custom_event_types": sorted({evt["type"] for evt in custom_events}),
        "custom_event_payloads": [evt["payload"] for evt in custom_events],
        "trace_db_path": str(_trace_db_path(workflow_engine)),
        "checkpoint_pass": (
            run.status == "suspended"
            and run.final_state.get("dep_echo") == "runtime-tutorial"
            and bool(custom_events)
        ),
    }


def level2_pause_and_resume(data_dir: Path) -> dict[str, Any]:
    result, runtime, _resolver, workflow_engine, conversation_engine = _run_to_suspension(data_dir)
    run = result["run"]
    checkpoint_state = result["checkpoint_state"]
    suspended = list((checkpoint_state.get("_rt_join", {}) or {}).get("suspended", []))
    if not suspended:
        _flush_trace(runtime)
        return {
            "level": 2,
            "run_id": result["run_id"],
            "status": run.status,
            "checkpoint_pass": False,
            "error": "No suspended tokens were found in checkpoint state.",
        }

    suspended_node_id, _mask, suspended_token_id, _parent_token_id = suspended[0]
    client_result = RunSuccess(
        conversation_node_id=None,
        state_update=[
            ("u", {"branch_a_pending": False, "branch_a_result": "approved"}),
            ("a", {"timeline": "client_result:branch_a_result=approved"}),
        ],
    )
    resumed = runtime.resume_run(
        run_id=result["run_id"],
        suspended_node_id=str(suspended_node_id),
        suspended_token_id=str(suspended_token_id),
        client_result=client_result,
        workflow_id=WORKFLOW_ID,
        conversation_id=CONVERSATION_ID,
        turn_node_id=TURN_NODE_ID,
    )
    _flush_trace(runtime)
    events = _fetch_trace_events(_trace_db_path(workflow_engine), result["run_id"])
    suspended_payloads = [evt["payload"] for evt in events if evt["type"] == "workflow_step_suspended"]
    checkpoint_nodes = _workflow_nodes(conversation_engine, "workflow_checkpoint", result["run_id"])
    resume_payload = suspended_payloads[0] if suspended_payloads else _resume_payload_for(str(suspended_node_id))
    return {
        "level": 2,
        "run_id": result["run_id"],
        "initial_status": run.status,
        "resumed_status": resumed.status,
        "suspended_node_id": str(suspended_node_id),
        "suspended_token_id": str(suspended_token_id),
        "resume_payload": resume_payload,
        "checkpoint_step_seqs": sorted(int(getattr(n, "metadata", {}).get("step_seq", -1)) for n in checkpoint_nodes),
        "final_state": resumed.final_state,
        "trace_db_path": str(_trace_db_path(workflow_engine)),
        "checkpoint_pass": (
            run.status == "suspended"
            and resumed.status == "succeeded"
            and resumed.final_state.get("branch_a_result") == "approved"
            and resumed.final_state.get("joined") is True
            and resumed.final_state.get("ended") is True
        ),
    }


def level3_observability_and_langgraph(data_dir: Path) -> dict[str, Any]:
    result, runtime, resolver, workflow_engine, conversation_engine = _run_to_suspension(data_dir)
    checkpoint_state = result["checkpoint_state"]
    suspended = list((checkpoint_state.get("_rt_join", {}) or {}).get("suspended", []))
    if suspended:
        suspended_node_id, _mask, suspended_token_id, _parent_token_id = suspended[0]
        runtime.resume_run(
            run_id=result["run_id"],
            suspended_node_id=str(suspended_node_id),
            suspended_token_id=str(suspended_token_id),
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"branch_a_pending": False, "branch_a_result": "approved"})],
            ),
            workflow_id=WORKFLOW_ID,
            conversation_id=CONVERSATION_ID,
            turn_node_id=TURN_NODE_ID,
        )
    _flush_trace(runtime)

    run_id = result["run_id"]
    events = _fetch_trace_events(_trace_db_path(workflow_engine), run_id)
    event_types = [evt["type"] for evt in events]
    saw_suspended_checkpoint = bool(suspended)
    try:
        lg_resolver = _LGResolver()
        visual = to_langgraph(
            workflow_engine=_build_langgraph_demo_engine(),
            workflow_id=WORKFLOW_ID,
            step_resolver=lg_resolver,
            predicate_registry={"always": PredAlwaysTrue()},
            options=LGConverterOptions(execution="visual"),
        )
        semantics = to_langgraph(
            workflow_engine=_build_langgraph_demo_engine(),
            workflow_id=WORKFLOW_ID,
            step_resolver=lg_resolver,
            predicate_registry={"always": PredAlwaysTrue()},
            options=LGConverterOptions(execution="semantics"),
        )
        visual_out = visual.invoke({"__blob__": {}})
        semantics_out = semantics.invoke({"__blob__": {}})
        langgraph_info = {
            "available": True,
            "visual_compiled": True,
            "semantics_compiled": True,
            "visual_blob_keys": sorted((visual_out.get("__blob__") or {}).keys()),
            "semantics_blob_keys": sorted((semantics_out.get("__blob__") or {}).keys()),
            "note": "LangGraph is interop/export here. Native WorkflowRuntime remains the source of truth for pause/resume and event sourcing.",
        }
    except Exception as exc:
        langgraph_info = {
            "available": False,
            "visual_compiled": False,
            "semantics_compiled": False,
            "note": f"Optional langgraph dependency unavailable: {exc}",
        }

    run_nodes = _workflow_nodes(conversation_engine, "workflow_run", run_id)
    step_nodes = _workflow_nodes(conversation_engine, "workflow_step_exec", run_id)
    checkpoint_nodes = _workflow_nodes(conversation_engine, "workflow_checkpoint", run_id)
    required_event_types = {
        "workflow_run_started",
        "step_attempt_started",
        "step_attempt_completed",
        "predicate_evaluated",
        "edge_selected",
        "join_waiting",
        "join_released",
        "checkpoint_saved",
        "workflow_step_suspended",
        "workflow_run_completed",
    }
    return {
        "level": 3,
        "run_id": run_id,
        "trace_db_path": str(_trace_db_path(workflow_engine)),
        "cdc_viewer_html": CDC_VIEWER_PATH,
        "viewer_asset_exists": Path(CDC_VIEWER_PATH).exists(),
        "runtime_event_endpoint": RUNTIME_EVENT_ENDPOINT.format(run_id=run_id),
        "trace_event_types": sorted(set(event_types)),
        "saw_suspended_checkpoint": saw_suspended_checkpoint,
        "step_exec_count": len(step_nodes),
        "checkpoint_count": len(checkpoint_nodes),
        "workflow_run_count": len(run_nodes),
        "langgraph": langgraph_info,
        "checkpoint_pass": (required_event_types - {"workflow_step_suspended"}).issubset(set(event_types))
        and Path(CDC_VIEWER_PATH).exists()
        and len(step_nodes) > 0
        and len(checkpoint_nodes) > 0
        and saw_suspended_checkpoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime tutorial ladder for resolver, pause/resume, CDC viewer, and LangGraph interop.")
    parser.add_argument("command", choices=["reset", "level0", "level1", "level2", "level3"])
    parser.add_argument("--data-dir", type=Path, default=Path(".gke-data/runtime-tutorial-ladder"))
    args = parser.parse_args()

    if args.command == "reset":
        payload = reset_data(args.data_dir)
    elif args.command == "level0":
        payload = level0_runtime_basics(args.data_dir)
    elif args.command == "level1":
        payload = level1_resolvers_and_deps(args.data_dir)
    elif args.command == "level2":
        payload = level2_pause_and_resume(args.data_dir)
    else:
        payload = level3_observability_and_langgraph(args.data_dir)
    _print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
