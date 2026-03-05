import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

pytest.importorskip("langgraph")

from graph_knowledge_engine.runtime.contract import BasePredicate
from graph_knowledge_engine.runtime.langgraph_converter import LGConverterOptions, to_langgraph
from langgraph.graph.state import CompiledStateGraph


# -----------------------------
# Minimum fake shapes (match test_workflow_join.py)
# -----------------------------

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


def _n(node_id: str, *, workflow_id: str, op: str, start=False, terminal=False, fanout=False, join=False) -> FakeNode:
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
    priority=0,
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


def dump_langgraph_image(compiled: CompiledStateGraph, name="graph"):
    graph = compiled.get_graph()
    png_bytes = graph.draw_png()
    out = Path(f"{name}.png")
    out.write_bytes(png_bytes)


def _seen_nodes(compiled, init_state: Dict[str, Any]) -> Set[str]:
    seen: Set[str] = set()
    evs = []
    for ev in compiled.stream(init_state, stream_mode="updates"):
        evs.append(ev)
        for k in ev.keys():
            if isinstance(k, str) and not k.startswith("__"):
                seen.add(k)
    return seen


def test_converter_predicate_default_is_exclusive_choice_and_auto_inits_blob():
    wid = "wf_lg_pred_default"
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

    resolver = Resolver({
        "noop": lambda state: RR([]),
        "set_a": lambda state: RR([("u", {"path": "a"})]),
        "set_b": lambda state: RR([("u", {"path": "b"})]),
    })

    compiled = to_langgraph(
        workflow_engine=engine,
        workflow_id=wid,
        step_resolver=resolver,
        predicate_registry={"p_true": PredAlwaysTrue()},
    )
    dump_langgraph_image(compiled, "test_converter_predicate_default_is_exclusive_choice_and_auto_inits_blob")
    out = compiled.invoke({})
    assert out["__blob__"]["path"] == "a"

    seen = _seen_nodes(compiled, {"__blob__": {}})
    assert {"start", "a", "end"} <= seen
    assert "b" not in seen


def test_converter_fanout_no_join_spawns_both_branches():
    wid = "wf_lg_fanout_no_join"
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

    resolver = Resolver({
        "noop": lambda state: RR([]),
        "emit_a": lambda state: RR([("a", {"events": "a"})]),
        "emit_b": lambda state: RR([("a", {"events": "b"})]),
    })

    compiled = to_langgraph(workflow_engine=engine, workflow_id=wid, step_resolver=resolver, predicate_registry={})
    out = compiled.invoke({"__blob__": {}})
    assert set(out["__blob__"].get("events", [])) == {"a", "b"}

    seen = _seen_nodes(compiled, {"__blob__": {}})
    assert {"start", "a", "b", "end"} <= seen


def test_converter_fanout_then_join_barrier_emits_joined_once():
    wid = "wf_lg_fanout_join"
    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True, fanout=True),
        _n("a", workflow_id=wid, op="emit_a"),
        _n("b", workflow_id=wid, op="emit_b"),
        _n("join", workflow_id=wid, op="join_op", join=True),
        _n("end", workflow_id=wid, op="noop", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="a", multiplicity="many"),
        _e("e2", workflow_id=wid, src="start", dst="b", multiplicity="many"),
        _e("e3", workflow_id=wid, src="a", dst="join"),
        _e("e4", workflow_id=wid, src="b", dst="join"),
        _e("e5", workflow_id=wid, src="join", dst="end"),
    ]
    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver({
        "noop": lambda state: RR([]),
        "emit_a": lambda state: RR([("a", {"events": "a"})]),
        "emit_b": lambda state: RR([("a", {"events": "b"})]),
        "join_op": lambda state: RR([("a", {"events": "joined"})]),
    })
    options = LGConverterOptions(execution="semantics")
    compiled = to_langgraph(workflow_engine=engine, workflow_id=wid, 
                            step_resolver=resolver, 
                            predicate_registry={},
                            options=options)
    out = compiled.invoke({"__blob__": {}})

    events = out["__blob__"].get("events", [])
    assert "a" in events
    assert "b" in events
    assert events.count("joined") == 1

    seen = _seen_nodes(compiled, {"__blob__": {}})
    assert {"start", "a", "b", "join", "end"} <= seen


def test_converter_resolver_next_step_names_duplicates_repeat_downstream():
    wid = "wf_lg_repeat_downstream"
    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True),
        _n("fork", workflow_id=wid, op="decide", fanout=True),
        _n("x", workflow_id=wid, op="emit_x"),
        _n("end", workflow_id=wid, op="noop", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="fork"),
        _e("e2", workflow_id=wid, src="fork", dst="x"),
        _e("e3", workflow_id=wid, src="x", dst="end"),
    ]
    engine = FakeWorkflowEngine(nodes, edges)

    resolver = Resolver({
        "noop": lambda state: RR([]),
        # Explicit routing override (duplicates preserved)
        "decide": lambda state: RR([], next_step_names=["x", "x"]),
        "emit_x": lambda state: RR([("a", {"events": "x"})]),
    })

    options = LGConverterOptions(execution="semantics")
    compiled = to_langgraph(workflow_engine=engine, workflow_id=wid, 
                            step_resolver=resolver, 
                            predicate_registry={},
                            options=options)
    seen = _seen_nodes(compiled, {"__blob__": {}})
    out = compiled.invoke({"__blob__": {}})
    
    events = out["__blob__"].get("events", [])
    assert events.count("x") == 2
