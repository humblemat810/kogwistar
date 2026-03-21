import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

pytest.importorskip("langgraph")

from graph_knowledge_engine.runtime.contract import BasePredicate
from graph_knowledge_engine.runtime.langgraph_converter import to_langgraph

pytestmark = pytest.mark.ci


# --- Minimum working fake shapes (MUST match tests/workflow/test_workflow_join.py) ---


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
    predicate: str
    source_ids: List[str]
    target_ids: List[str]
    multiplicity: Any
    is_default: bool
    metadata: Dict[str, Any] | None = None

    def safe_get_id(self):
        return self.id


class FakeConversationEngine:
    """
    Minimal sink for WorkflowRuntime tracing. It only needs to accept add_node/add_edge.
    """

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def add_node(self, n):
        self.nodes.append(n)
        return n

    def add_edge(self, e):
        self.edges.append(e)
        return e


class FakeWorkflowEngine:
    def __init__(self, nodes: List[FakeNode], edges: List[FakeEdge]) -> None:
        self._nodes = nodes
        self._edges = edges

    def get_nodes(self, where=None, limit=5000, **kwargs):
        return self._nodes

    def get_edges(self, where=None, limit=20000, **kwargs):
        return self._edges


def _n(
    node_id: str,
    *,
    workflow_id: str,
    op: str,
    start=False,
    terminal=False,
    fanout=False,
    join=False,
) -> FakeNode:
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
    return FakeNode(
        id=node_id,
        metadata=md,
        op=md["wf_op"],
        terminal=bool(md.get("wf_terminal")),
        fanout=bool(md.get("wf_fanout")),
    )


def _e(
    edge_id: str,
    *,
    workflow_id: str,
    src: str,
    dst: str,
    predicate=None,
    priority=100,
    is_default=False,
    multiplicity="one",
) -> FakeEdge:
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


class Resolver:
    def __init__(self, mapping):
        self.mapping = mapping

    def resolve(self, op):
        return self.mapping[op]


class RR:
    def __init__(self, state_update=None, next_step_names=None):
        self.state_update = state_update or []
        self.next_step_names = next_step_names or []


class PredAlwaysTrue(BasePredicate):
    def __call__(self, e, state, result):
        return True


def test_langgraph_parallel_fanout_merges_updates_end_to_end():
    wid = "wf_lg_parallel"
    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True),
        _n("fork", workflow_id=wid, op="noop", fanout=True),
        _n("x", workflow_id=wid, op="add_x"),
        _n("y", workflow_id=wid, op="add_y"),
        _n("end", workflow_id=wid, op="noop", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="fork", priority=0),
        _e("e2", workflow_id=wid, src="fork", dst="x", priority=0, multiplicity="many"),
        _e("e3", workflow_id=wid, src="fork", dst="y", priority=1, multiplicity="many"),
        _e("e4", workflow_id=wid, src="x", dst="end", priority=0),
        _e("e5", workflow_id=wid, src="y", dst="end", priority=0),
    ]
    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver(
        {
            "noop": lambda state: RR([]),
            "add_x": lambda state: RR([("a", {"events": ("x", time.time())})]),
            "add_y": lambda state: RR([("a", {"events": ("y", time.time())})]),
        }
    )

    compiled = to_langgraph(
        workflow_engine=engine,
        workflow_id=wid,
        step_resolver=resolver,
        predicate_registry={},
    )
    out = compiled.invoke({})

    names = (
        out.get("events", [])
        or [i[0] for i in (out.get("__blob__") and out["__blob__"].get("events")) or []]
        or []
    )
    assert "x" in names and "y" in names
