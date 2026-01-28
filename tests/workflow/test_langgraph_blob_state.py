from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from graph_knowledge_engine.workflow.langgraph_converter import to_langgraph, LGConverterOptions
from graph_knowledge_engine.workflow.contract import BasePredicate


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


class FakeWorkflowEngine:
    def __init__(self, nodes: List[FakeNode], edges: List[FakeEdge]):
        self._nodes = list(nodes)
        self._edges = list(edges)

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
        return list(self._nodes)

    def get_edges(self, where=None, limit=9999, edge_type=None, ids=None):
        if ids:
            want = set(ids)
            return [e for e in self._edges if e.id in want]
        if where and "$and" in where:
            cond = {k: v for d in where["$and"] for k, v in d.items()}
            wf_id = cond.get("workflow_id")
            et = cond.get("entity_type")
            if et == "workflow_edge":
                return [e for e in self._edges if (e.metadata or {}).get("workflow_id") == wf_id]
        return list(self._edges)


class PredAlwaysTrue(BasePredicate):
    def __call__(self, e, state, result):
        return True


class RR:
    def __init__(self, state_update=None, update=None, next_step_names=None):
        self.state_update = state_update or []
        self.update = update
        self.next_step_names = next_step_names or []


class Resolver:
    def __init__(self, handlers: Dict[str, Any], schema: Optional[Dict[str, str]] = None):
        self._h = dict(handlers)
        self._schema = dict(schema or {})

    def resolve(self, op: str):
        return self._h[op]

    def describe_state(self) -> Dict[str, str]:
        return dict(self._schema)


def test_blob_state_routes_predicate_and_default_and_hides_apply_node():
    wid = "wf_lg_pred"
    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True),
        _n("a", workflow_id=wid, op="set_a"),
        _n("b", workflow_id=wid, op="set_b"),
        _n("end", workflow_id=wid, op="noop", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="a", predicate="p_true", priority=0),
        _e("e2", workflow_id=wid, src="start", dst="b", is_default=True, priority=1),
        _e("e3", workflow_id=wid, src="a", dst="end", priority=0),
        _e("e4", workflow_id=wid, src="b", dst="end", priority=0),
    ]
    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver(
        {
            "noop": lambda blob: RR([]),
            "set_a": lambda blob: RR([("u", {"path": "a"})]),
            "set_b": lambda blob: RR([("u", {"path": "b"})]),
        }
    )

    compiled = to_langgraph(
        workflow_engine=engine,
        workflow_id=wid,
        step_resolver=resolver,
        predicate_registry={"p_true": PredAlwaysTrue()},
        options=LGConverterOptions(mode="blob_state"),
    )

    # sanity: no apply node in graph structure
    g = compiled.get_graph()
    node_ids = {n.id for n in g.nodes}
    assert "__apply__" not in node_ids

    out = compiled.invoke({"__blob__": {}})
    assert out["__blob__"]["path"] == "a"
