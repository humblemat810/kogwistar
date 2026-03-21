
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from graph_knowledge_engine.runtime.langgraph_converter import (
    LGConverterOptions,
    to_langgraph,
)
from graph_knowledge_engine.runtime.models import RunSuccess

pytestmark = pytest.mark.ci


# ---- Minimum fake shapes (match tests/workflow/test_workflow_join.py) ----


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


class FakeWorkflowEngine:
    def __init__(self, nodes: List[FakeNode], edges: List[FakeEdge]) -> None:
        self._nodes = nodes
        self._edges = edges

    def get_nodes(self, where=None, limit=None, node_type=None):
        return list(self._nodes)

    def get_edges(self, where=None, limit=None, edge_type=None):
        return list(self._edges)


def _n(
    node_id: str,
    *,
    workflow_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
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
    edge_id: str,
    *,
    workflow_id: str,
    src: str,
    dst: str,
    predicate: None | str = None,
    priority: int = 100,
    is_default: bool = False,
    multiplicity: str = "one",
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
        predicate=str(md["wf_predicate"]) if md["wf_predicate"] is not None else None,
        source_ids=[src],
        target_ids=[dst],
        multiplicity=md["wf_multiplicity"],
        is_default=md["wf_is_default"],
        metadata=md,
    )


class Resolver:
    def __init__(self, fns: Dict[str, Any]):
        self._fns = dict(fns)

    def resolve(self, op: str):
        return self._fns[op]

    def describe_state(self) -> Dict[str, str]:
        # only static keys known; everything else will default to overwrite
        return {"log": "a"}


def test_blob_state_reducer_is_deterministic_under_parallel_fanout():
    """Two parallel branches append to the same list; order should be deterministic.

    We expect token ids root.0, root.1 -> branch 0 ops apply before branch 1.
    """

    wid = "wf_blob_parallel"

    nodes = [
        _n("start", workflow_id=wid, op="start", start=True, fanout=True),
        _n("a", workflow_id=wid, op="do_a"),
        _n("b", workflow_id=wid, op="do_b"),
        _n("end", workflow_id=wid, op="end", terminal=True),
    ]
    edges = [
        _e(
            "e1", workflow_id=wid, src="start", dst="a", priority=0, multiplicity="many"
        ),
        _e(
            "e2", workflow_id=wid, src="start", dst="b", priority=0, multiplicity="many"
        ),
        _e("e3", workflow_id=wid, src="a", dst="end", priority=0),
        _e("e4", workflow_id=wid, src="b", dst="end", priority=0),
    ]

    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver(
        {
            "start": lambda blob: RunSuccess(
                conversation_node_id=None, state_update=[]
            ),
            "do_a": lambda blob: RunSuccess(
                conversation_node_id=None, state_update=[("a", {"log": "a"})]
            ),
            "do_b": lambda blob: RunSuccess(
                conversation_node_id=None, state_update=[("a", {"log": "b"})]
            ),
            "end": lambda blob: RunSuccess(conversation_node_id=None, state_update=[]),
        }
    )

    compiled = to_langgraph(
        workflow_engine=engine,
        workflow_id=wid,
        step_resolver=resolver,
        predicate_registry={},
        options=LGConverterOptions(mode="blob_state"),
    )

    out = compiled.invoke({})
    blob = out.get("__blob__") or {}

    assert blob.get("log") == ["a", "b"]


def test_blob_state_mode_produces_no_apply_node():
    wid = "wf_no_apply"

    nodes = [
        _n("start", workflow_id=wid, op="start", start=True),
        _n("end", workflow_id=wid, op="end", terminal=True),
    ]
    edges = [_e("e1", workflow_id=wid, src="start", dst="end", priority=0)]

    engine = FakeWorkflowEngine(nodes, edges)
    resolver = Resolver(
        {
            "start": lambda blob: RunSuccess(
                conversation_node_id=None, state_update=[]
            ),
            "end": lambda blob: RunSuccess(conversation_node_id=None, state_update=[]),
        }
    )

    compiled = to_langgraph(
        workflow_engine=engine,
        workflow_id=wid,
        step_resolver=resolver,
        predicate_registry={},
        options=LGConverterOptions(mode="blob_state"),
    )

    # LangGraph compiled objects expose graph structure via get_graph().
    g = compiled.get_graph()
    # ensure there is no '__apply__' node name
    assert "__apply__" not in g.nodes
