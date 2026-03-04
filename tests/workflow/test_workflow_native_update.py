from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from graph_knowledge_engine.runtime.runtime import WorkflowRuntime
from graph_knowledge_engine.runtime.resolvers import MappingStepResolver
from graph_knowledge_engine.runtime.contract import BasePredicate
from runtime.models import RunSuccess


# --- Minimum fake shapes (aligned with test_workflow_join.py) ---

@dataclass
class FakeNode:
    id: str
    op: str
    terminal: bool
    fanout: bool
    metadata: Dict[str, Any] | None = None

    def safe_get_id(self):
        return self.id


@dataclass
class FakeEdge:
    id: str
    label: str
    predicate: str | None
    source_ids: List[str]
    target_ids: List[str]
    multiplicity: str
    is_default: bool
    metadata: Dict[str, Any] | None = None

    def safe_get_id(self):
        return self.id


def _n(node_id: str, *, workflow_id: str, op: str, start=False, terminal=False, fanout=False) -> FakeNode:
    md = {
        "entity_type": "workflow_node",
        "workflow_id": workflow_id,
        "wf_op": op,
        "wf_version": "v1",
        "wf_start": bool(start),
        "wf_terminal": bool(terminal),
        "wf_fanout": bool(fanout),
    }
    return FakeNode(id=node_id, metadata=md, op=md["wf_op"], terminal=bool(md.get("wf_terminal")), fanout=bool(md.get("wf_fanout")))


def _e(edge_id: str, *, workflow_id: str, src: str, dst: str, predicate=None, priority=100, is_default=False, multiplicity="one") -> FakeEdge:
    md = {
        "entity_type": "workflow_edge",
        "workflow_id": workflow_id,
        "wf_predicate": predicate,
        "wf_priority": priority,
        "wf_is_default": bool(is_default),
        "wf_multiplicity": multiplicity,
    }
    return FakeEdge(
        id=edge_id,
        label=f"{src} to {dst}",
        predicate=md["wf_predicate"],
        multiplicity=md["wf_multiplicity"],
        is_default=md["wf_is_default"],
        source_ids=[src],
        target_ids=[dst],
        metadata=md,
    )


class FakeEngine:
    def __init__(self, nodes: List[FakeNode], edges: List[FakeEdge]):
        self._nodes = list(nodes)
        self._edges = list(edges)

    def add_node(self, n):  # for conversation_engine trace sink
        return n

    def get_nodes(self, where=None, limit=9999, node_type=None, ids=None):
        if ids:
            want = set(ids)
            return [n for n in self._nodes if n.id in want]
        if where and "$and" in where:
            cond = {k: v for d in where["$and"] for k, v in d.items()}
            wf_id = cond.get("workflow_id")
            et = cond.get("entity_type")
            if et == "workflow_node":
                return [n for n in self._nodes if (n.metadata or {}).get("workflow_id") == wf_id]
            if et == "workflow_checkpoint" or et == "workflow_step_exec":
                return []
        return list(self._nodes)

    def get_edges(self, where=None, limit=9999, edge_type=None, ids=None):
        if where and "$and" in where:
            cond = {k: v for d in where["$and"] for k, v in d.items()}
            wf_id = cond.get("workflow_id")
            et = cond.get("entity_type")
            if et == "workflow_edge":
                return [e for e in self._edges if (e.metadata or {}).get("workflow_id") == wf_id]
        return list(self._edges)

    def add_edge(self, e):
        return e


class PredAlwaysTrue(BasePredicate):
    def __call__(self, e, state, result):
        return True


def test_workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown():
    wid = "wf_native_update"
    nodes = [
        _n("start", workflow_id=wid, op="do", start=True),
        _n("end", workflow_id=wid, op="noop", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="end", priority=0),
    ]
    wf_engine = FakeEngine(nodes, edges)
    conv_engine = FakeEngine([], [])

    resolver = MappingStepResolver()
    resolver.set_state_schema({"op_log": "a"})

    @resolver.register("do")
    def _do(ctx):
        # op_log is known -> append; dyn is unknown -> fallback overwrite
        return RunSuccess(conversation_node_id=None, state_update=[], update={"op_log": "x", "dyn": 1})

    @resolver.register("noop")
    def _noop(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], update=None)

    rt = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=9999,
        max_workers=1,
    )

    run_result = rt.run(
        workflow_id=wid,
        conversation_id="c1",
        turn_node_id="t1",
        initial_state={},
    )
    final_state, _run_id = run_result.final_state, run_result.run_id
    assert final_state["op_log"] == ["x"]
    assert final_state["dyn"] == 1
