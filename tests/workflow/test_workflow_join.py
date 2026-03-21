import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from kogwistar.runtime.runtime import StepContext, WorkflowRuntime
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.models import RunSuccess


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
    multiplicity: int
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

    def get_nodes(self, *arg, **kwargs):
        return self.nodes

    def get_edges(self, *arg, **kwargs):
        return self.edges


class FakeWorkflowEngine:
    def __init__(self, nodes: List[FakeNode], edges: List[FakeEdge]) -> None:
        self._nodes = nodes
        self._edges = edges

    def get_nodes(self, where=None, limit=5000, **kwarg):
        # ignore where in tests
        return self._nodes

    def get_edges(self, where=None, limit=20000, **kwarg):
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


@pytest.mark.parametrize("max_workers", [1, 2])
def test_join_barrier_waits_for_all_arrivals(max_workers: int):
    """
    Graph:
        start -> fork -> a -> join -> end
                     -> b -> join

    Expect:
      - join does not release to end until BOTH a and b have arrived.
    """
    wid = "wf_join_wait_all"

    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True),
        _n("fork", workflow_id=wid, op="noop", fanout=True),
        _n("a", workflow_id=wid, op="fast"),
        _n("b", workflow_id=wid, op="slow"),
        _n("c", workflow_id=wid, op="slow_c"),
        _n("join", workflow_id=wid, op="noop", join=True),
        _n("end", workflow_id=wid, op="end", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="fork"),
        _e("e2", workflow_id=wid, src="fork", dst="a"),
        _e("e3", workflow_id=wid, src="fork", dst="b"),
        _e("e3_2", workflow_id=wid, src="fork", dst="c"),
        _e("e4", workflow_id=wid, src="a", dst="join"),
        _e("e5", workflow_id=wid, src="b", dst="join"),
        _e("e6", workflow_id=wid, src="join", dst="end"),
    ]

    engine = FakeWorkflowEngine(nodes, edges)
    resolver = MappingStepResolver()

    @resolver.register("noop")
    def _noop(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fast")
    def _fast(ctx: StepContext):
        # mark arrival time

        # state.setdefault("op_log", []).append("start")
        # state["started"] = True
        # result = RunSuccess(conversation_node_id=None, state_update=[('u',{"started": True})])
        with ctx.state_write as st:
            st.setdefault("events", []).append(("a_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow")
    def _slow(ctx: StepContext):
        time.sleep(0.05)
        # st = ctx.state
        with ctx.state_write as st:
            st.setdefault("events", []).append(("b_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow_c")
    def _slow_c(ctx: StepContext):
        time.sleep(0.05)
        # st = ctx.state
        with ctx.state_write as st:
            st.setdefault("events", []).append(("c_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    def _end(ctx: StepContext):
        with ctx.state_write as st:
            st.setdefault("events", []).append(("end", time.time()))
            st["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    rt = WorkflowRuntime(
        workflow_engine=engine,
        conversation_engine=FakeConversationEngine(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=max_workers,
    )

    state: Dict[str, Any] = {}
    run_result = rt.run(
        workflow_id=wid,
        conversation_id="conv_test",
        turn_node_id="turn_test",
        initial_state=state,
    )
    final_state, run_id = run_result.final_state, run_result.run_id
    out = final_state, run_id
    assert state.get("ended") is True
    events = state.get("events", [])
    names = [x[0] for x in events]
    assert "a_done" in names and "b_done" in names and "end" in names

    # end must be after both a_done and b_done
    t_end = next(t for n, t in events if n == "end")
    t_a = next(t for n, t in events if n == "a_done")
    t_b = next(t for n, t in events if n == "b_done")
    t_c = next(t for n, t in events if n == "c_done")
    assert t_end >= t_c
    assert t_end >= t_a
    assert t_end >= t_b


def test_join_does_not_wait_for_branch_that_can_no_longer_reach_it():
    """
    Graph:
        start -> fork -> a -> end_a (terminal)
                     -> b -> join -> end

    Here 'a' path exits the join region; join must NOT wait for it.
    """
    wid = "wf_join_delta_exit"

    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True),
        _n("fork", workflow_id=wid, op="noop", fanout=True),
        _n("a", workflow_id=wid, op="fast"),
        _n("end_a", workflow_id=wid, op="end_a", terminal=True),
        _n("b", workflow_id=wid, op="slow"),
        _n("join", workflow_id=wid, op="join", join=True),
        _n("end", workflow_id=wid, op="end", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="fork"),
        _e("e2", workflow_id=wid, src="fork", dst="a"),
        _e("e3", workflow_id=wid, src="fork", dst="b"),
        _e("e4", workflow_id=wid, src="a", dst="end_a"),
        _e("e5", workflow_id=wid, src="b", dst="join"),
        _e("e6", workflow_id=wid, src="join", dst="end"),
    ]

    engine = FakeWorkflowEngine(nodes, edges)
    resolver = MappingStepResolver()

    @resolver.register("noop")
    def _fast(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fast")
    def _fast(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append("a_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("join")
    def _join(ctx):
        time.sleep(0.05)
        with ctx.state_write as st:
            st.setdefault("events", []).append("join_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow")
    def _slow(ctx):
        time.sleep(0.05)
        with ctx.state_write as st:
            st.setdefault("events", []).append("b_done")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append("end")
            st["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end_a")
    def _end_a(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append("end_a")
        return RunSuccess(conversation_node_id=None, state_update=[])

    rt = WorkflowRuntime(
        workflow_engine=engine,
        conversation_engine=FakeConversationEngine(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=2,
    )

    state: Dict[str, Any] = {}
    run_result = rt.run(
        workflow_id=wid,
        conversation_id="conv_test",
        turn_node_id="turn_test",
        initial_state=state,
    )
    final_state, _run_id = run_result.final_state, run_result.run_id
    assert final_state.get("ended") is True
    # join must have released even though 'a' never reaches join
    assert "b_done" in final_state.get("events", [])
    assert "end" in final_state.get("events", [])


def test_nested_joins_human_debug(capsys):
    """
    Graph (nested joins):
        start -> fork1 -> a -> join1 -> fork2 -> c -> join2 -> end
                     -> b -> join1          -> d -> join2

    Expect:
      - join1 waits for a and b
      - join2 waits for c and d (which only start after join1 released)
      - debug prints show join arrivals/releases
    """
    wid = "wf_nested_joins"

    nodes = [
        _n("start", workflow_id=wid, op="noop", start=True),
        _n("fork1", workflow_id=wid, op="noop", fanout=True),
        _n("a", workflow_id=wid, op="fast"),
        _n("b", workflow_id=wid, op="slow"),
        _n("join1", workflow_id=wid, op="noop", join=True),
        _n("fork2", workflow_id=wid, op="noop", fanout=True),
        _n("c", workflow_id=wid, op="fast2"),
        _n("d", workflow_id=wid, op="slow2"),
        _n("join2", workflow_id=wid, op="noop", join=True),
        _n("end", workflow_id=wid, op="end", terminal=True),
    ]
    edges = [
        _e("e1", workflow_id=wid, src="start", dst="fork1"),
        _e("e2", workflow_id=wid, src="fork1", dst="a"),
        _e("e3", workflow_id=wid, src="fork1", dst="b"),
        _e("e4", workflow_id=wid, src="a", dst="join1"),
        _e("e5", workflow_id=wid, src="b", dst="join1"),
        _e("e6", workflow_id=wid, src="join1", dst="fork2"),
        _e("e7", workflow_id=wid, src="fork2", dst="c"),
        _e("e8", workflow_id=wid, src="fork2", dst="d"),
        _e("e9", workflow_id=wid, src="c", dst="join2"),
        _e("e10", workflow_id=wid, src="d", dst="join2"),
        _e("e11", workflow_id=wid, src="join2", dst="end"),
    ]

    engine = FakeWorkflowEngine(nodes, edges)
    resolver = MappingStepResolver()

    @resolver.register("noop")
    def _noop(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append(("noop", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fast")
    def _fast(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append(("a_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow")
    def _slow(ctx):
        time.sleep(0.05)
        with ctx.state_write as st:
            st.setdefault("events", []).append(("b_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fast2")
    def _fast2(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append(("c_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow2")
    def _slow2(ctx):
        time.sleep(0.05)
        with ctx.state_write as st:
            st.setdefault("events", []).append(("d_done", time.time()))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as st:
            st.setdefault("events", []).append(("end", time.time()))
            st["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    rt = WorkflowRuntime(
        workflow_engine=engine,
        conversation_engine=FakeConversationEngine(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=2,
    )

    state: Dict[str, Any] = {"testcase_rt_join_debug": True}
    run_result = rt.run(
        workflow_id=wid,
        conversation_id="conv_test",
        turn_node_id="turn_test",
        initial_state=state,
    )
    final_state, _run_id = run_result.final_state, run_result.run_id
    assert final_state.get("ended") is True
    events = final_state.get("events", [])
    names = [n for n, _t in events]
    assert names.count("end") == 1
    assert (
        "a_done" in names
        and "b_done" in names
        and "c_done" in names
        and "d_done" in names
    )

    # end after all
    t_end = next(t for n, t in events if n == "end")
    assert t_end >= next(t for n, t in events if n == "a_done")
    assert t_end >= next(t for n, t in events if n == "b_done")
    assert t_end >= next(t for n, t in events if n == "c_done")
    assert t_end >= next(t for n, t in events if n == "d_done")

    # # Human readable debug evidence:
    # out = capsys.readouterr().out
    # assert "[wf.join]" in out
    # assert "release join join1" in out
    # assert "release join join2" in out
