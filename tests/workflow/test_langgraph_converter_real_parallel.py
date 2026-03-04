

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Set

pytest.importorskip("langgraph")

from graph_knowledge_engine.runtime.langgraph_converter import to_langgraph
from graph_knowledge_engine.runtime.contract import BasePredicate


@dataclass
class FakeWorkflowNode:
    id: str
    op: str
    terminal: bool = False
    fanout: bool = False
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"wf_terminal": self.terminal, "wf_fanout": self.fanout}

    def safe_get_id(self) -> str:
        return self.id


@dataclass
class FakeWorkflowEdge:
    id: str
    label: str
    source_ids: List[str]
    target_ids: List[str]
    metadata: Dict[str, Any]

    def safe_get_id(self) -> str:
        return self.id


class FakeWorkflowEngine:
    def __init__(self, nodes: List[FakeWorkflowNode], edges: List[FakeWorkflowEdge]) -> None:
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self, where=None, limit=5000, node_type=None, **kwargs):
        return self.nodes

    def get_edges(self, where=None, limit=20000, edge_type=None, **kwargs):
        return self.edges


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


def _n(node_id: str, *, op: str, start: bool = False, terminal: bool = False, fanout: bool = False) -> FakeWorkflowNode:
    md = {"wf_start": start, "wf_terminal": terminal, "wf_fanout": fanout, "wf_op": op, "wf_version": "v1", "workflow_id": "x", "entity_type":"workflow_node"}
    return FakeWorkflowNode(id=node_id, op=op, terminal=terminal, fanout=fanout, metadata=md)


def _seen_nodes(compiled, init_state: Dict[str, Any]) -> Set[str]:
    seen: Set[str] = set()
    for ev in compiled.stream(init_state, stream_mode="updates"):
        for k in ev.keys():
            if isinstance(k, str) and not k.startswith("__"):
                seen.add(k)
    return seen


def test_predicate_signature_and_default_routing():
    wid = "wf1"
    nodes = [
        _n("start", op="noop", start=True),
        _n("a", op="set_a"),
        _n("b", op="set_b"),
        _n("end", op="noop", terminal=True),
    ]
    edges = [
        FakeWorkflowEdge(id="e1", label="to_a", source_ids=["start"], target_ids=["a"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_predicate":"p_true","wf_multiplicity":"one"}),
        FakeWorkflowEdge(id="e2", label="to_b", source_ids=["start"], target_ids=["b"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":1,"wf_is_default":True,"wf_multiplicity":"one"}),
        FakeWorkflowEdge(id="e3", label="to_end_from_a", source_ids=["a"], target_ids=["end"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_multiplicity":"one"}),
        FakeWorkflowEdge(id="e4", label="to_end_from_b", source_ids=["b"], target_ids=["end"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_multiplicity":"one"}),
    ]
    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver({
        "noop": lambda state: RR([]),
        "set_a": lambda state: RR([("u", {"path":"a"})]),
        "set_b": lambda state: RR([("u", {"path":"b"})]),
    })
    preds = {"p_true": PredAlwaysTrue()}

    compiled = to_langgraph(workflow_engine=engine, workflow_id=wid, step_resolver=resolver, predicate_registry=preds)
    out = compiled.invoke({})
    assert out.get("path") or (out["__blob__"]['path']) == "a"


def test_parallel_fanout_merges_appends_real_langgraph(backend_kind):
    wid = "wf2"
    nodes = [
        _n("start", op="noop", start=True),
        _n("fork", op="noop", fanout=True),
        _n("x", op="add_x"),
        _n("y", op="add_y"),
        _n("end", op="noop", terminal=True),
    ]
    edges = [
        FakeWorkflowEdge(id="e1", label="to_fork", source_ids=["start"], target_ids=["fork"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_multiplicity":"one"}),
        FakeWorkflowEdge(id="e2", label="to_x", source_ids=["fork"], target_ids=["x"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_multiplicity":"many"}),
        FakeWorkflowEdge(id="e3", label="to_y", source_ids=["fork"], target_ids=["y"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":1,"wf_multiplicity":"many"}),
        FakeWorkflowEdge(id="e4", label="x_to_end", source_ids=["x"], target_ids=["end"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_multiplicity":"one"}),
        FakeWorkflowEdge(id="e5", label="y_to_end", source_ids=["y"], target_ids=["end"],
                         metadata={"entity_type":"workflow_edge","workflow_id":wid,"wf_priority":0,"wf_multiplicity":"one"}),
    ]
    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver({
        "noop": lambda state: RR([]),
        "add_x": lambda state: RR([("a", {"events":"x"})]),
        "add_y": lambda state: RR([("a", {"events":"y"})]),
    })
    preds: Dict[str, BasePredicate] = {}

    compiled = to_langgraph(workflow_engine=engine, workflow_id=wid, step_resolver=resolver, predicate_registry=preds)

    seen = _seen_nodes(compiled, {})
    assert "x" in seen
    assert "y" in seen

    out = compiled.invoke({})
    assert sorted(out.get("events") or (out["__blob__"]['events'])) == ["x", "y"]
