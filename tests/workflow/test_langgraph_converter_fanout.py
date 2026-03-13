import pytest
from dataclasses import dataclass
from typing import Any, Dict, List

pytest.importorskip("langgraph")

from graph_knowledge_engine.runtime.contract import BasePredicate
from graph_knowledge_engine.runtime.langgraph_converter import to_langgraph


# minimum fake shapes
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
    return FakeNode(
        id=node_id,
        metadata=md,
        op=md["wf_op"],
        terminal=bool(md.get("wf_terminal")),
        fanout=bool(md.get("wf_fanout")),
    )


def _e(
    edge_id: str, *, workflow_id: str, src: str, dst: str, multiplicity="one"
) -> FakeEdge:
    md = {
        "entity_type": "workflow_edge",
        "workflow_id": workflow_id,
        "wf_predicate": None,
        "wf_priority": 0,
        "wf_is_default": False,
        "wf_multiplicity": multiplicity,
    }
    return FakeEdge(
        id=edge_id,
        label=f"{src} to {dst}",
        predicate=None,
        multiplicity=md["wf_multiplicity"],
        is_default=False,
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


def test_converter_fanout_spawns_both_branches():
    wid = "wf_lg_fanout"
    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True, fanout=True),
        _n("a", workflow_id=wid, op="emit_a"),
        _n("b", workflow_id=wid, op="emit_b"),
        _n("end", workflow_id=wid, op="noop", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="a", multiplicity="many"),
        _e("e2", workflow_id=wid, src="start", dst="b", multiplicity="many"),
        _e("e3", workflow_id=wid, src="a", dst="end"),
        _e("e4", workflow_id=wid, src="b", dst="end"),
    ]
    engine = FakeWorkflowEngine(nodes, edges)
    resolver = Resolver(
        {
            "noop": lambda state: RR([]),
            "emit_a": lambda state: RR([("a", {"events": "a"})]),
            "emit_b": lambda state: RR([("a", {"events": "b"})]),
        }
    )

    compiled = to_langgraph(
        workflow_engine=engine,
        workflow_id=wid,
        step_resolver=resolver,
        predicate_registry={},
    )

    out = compiled.invoke({"__blob__": {}})
    events = out["__blob__"].get("events", [])
    assert set(events) == {"a", "b"}
