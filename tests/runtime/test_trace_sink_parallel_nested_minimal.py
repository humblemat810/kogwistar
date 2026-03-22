import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import pytest
pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Span, Grounding
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.runtime import WorkflowRuntime, StepContext
from kogwistar.runtime.resolvers import MappingStepResolver
import logging

# Reuse your canonical engine factory (already parametrized in other tests)
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
from tests.conftest import _make_engine_pair, FakeEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend
import os

os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"


def _wid(workflow_id: str, suffix: str) -> str:
    return f"wf:{workflow_id}:{suffix}"


def _add_node(
    workflow_engine: GraphKnowledgeEngine,
    *,
    workflow_id: str,
    suffix: str,
    op: str | None,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
    wf_join: bool = False,
) -> str:
    node_id = _wid(workflow_id, suffix)
    sp = Span.from_dummy_for_workflow(workflow_id)
    n = WorkflowNode(
        id=node_id,
        label=suffix,
        type="entity",
        doc_id=node_id,
        summary=suffix,
        properties={},
        mentions=[Grounding(spans=[sp])],
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_fanout": bool(fanout),
            "wf_join": bool(wf_join),
            "wf_join_is_merge": True,
            "wf_version": "v_test",
        },
    )
    workflow_engine.add_node(n)
    return node_id


def _add_edge(
    workflow_engine: GraphKnowledgeEngine,
    *,
    workflow_id: str,
    edge_suffix: str,
    src: str,
    dst: str,
    pred: str | None = None,
    priority: int = 100,
    is_default: bool = False,
    multiplicity: str = "one",
) -> str:
    edge_id = f"wfe:{workflow_id}:{edge_suffix}"
    sp = Span.from_dummy_for_workflow(workflow_id)
    e = WorkflowEdge(
        id=edge_id,
        label="wf_next",
        type="entity",
        doc_id=edge_id,
        summary="wf_next",
        properties={},
        source_ids=[src],
        target_ids=[dst],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="wf_next",
        mentions=[Grounding(spans=[sp])],
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_edge_kind": "wf_next",
            "wf_predicate": pred,
            "wf_priority": int(priority),
            "wf_is_default": bool(is_default),
            "wf_multiplicity": multiplicity,
            "wf_version": "v_test",
        },
    )
    workflow_engine.add_edge(e)
    return edge_id


def _build_outer(workflow_engine: GraphKnowledgeEngine, workflow_id: str) -> None:
    """
    Outer graph:
      start -> fork(fanout) -> a -> join -> nested -> end
                           \\-> b -/
    """
    start = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="start", op="noop", start=True
    )
    fork = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="fork", op="noop", fanout=True
    )
    a = _add_node(workflow_engine, workflow_id=workflow_id, suffix="a", op="slow_a")
    b = _add_node(workflow_engine, workflow_id=workflow_id, suffix="b", op="slow_b")
    join = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="join", op="noop", wf_join=True
    )
    nested = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="nested", op="nested_call"
    )
    end = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="end", op="end", terminal=True
    )

    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="s_f", src=start, dst=fork
    )
    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="f_a", src=fork, dst=a
    )
    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="f_b", src=fork, dst=b
    )
    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="a_j", src=a, dst=join
    )
    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="b_j", src=b, dst=join
    )
    _add_edge(
        workflow_engine,
        workflow_id=workflow_id,
        edge_suffix="j_n",
        src=join,
        dst=nested,
    )
    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="n_e", src=nested, dst=end
    )


def _build_inner(workflow_engine: GraphKnowledgeEngine, workflow_id: str) -> None:
    """
    Inner graph:
      start -> inner_slow -> inner_end
    """
    start = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="start", op="noop", start=True
    )
    slow = _add_node(
        workflow_engine, workflow_id=workflow_id, suffix="slow", op="inner_slow"
    )
    end = _add_node(
        workflow_engine,
        workflow_id=workflow_id,
        suffix="end",
        op="inner_end",
        terminal=True,
    )

    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="s_s", src=start, dst=slow
    )
    _add_edge(
        workflow_engine, workflow_id=workflow_id, edge_suffix="s_e", src=slow, dst=end
    )


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
)
@pytest.mark.parametrize("tags", [["a"], ["a", "b", "c", "d"]], ids=["a", "abcd"])
@pytest.mark.parametrize("iterations", [1, 5])
def test_trace_sink_parallel_and_nested_minimal_sync(
    backend_kind: str,
    tmp_path,
    request: pytest.FixtureRequest,
    tags,
    iterations,
):
    # ---- engines: conversation engine varies by backend; workflow engine always local (sink is sqlite under persist_directory)
    if backend_kind == "fake":
        _kg_engine, conversation_engine = _make_engine_pair(
            backend_kind="fake",
            tmp_path=tmp_path,
            sa_engine=None,
            pg_schema=None,
            dim=8,
            use_fake=True,
        )
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=8),
            backend_factory=build_fake_backend,
        )
    else:
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
        _kg_engine, conversation_engine = _make_engine_pair(
            backend_kind=backend_kind,
            tmp_path=tmp_path,
            sa_engine=sa_engine,
            pg_schema=pg_schema,
            dim=8,
            use_fake=True,
        )
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=8),
        )

    outer_id = f"wf_test_outer_parallel_nested_min_sync_{backend_kind}"
    inner_id = f"wf_test_inner_parallel_nested_min_sync_{backend_kind}"
    _build_outer(workflow_engine, outer_id)
    _build_inner(workflow_engine, inner_id)

    # ---- per-run synchronization so multiple outer runs don't deadlock each other
    sync_by_run: dict[str, dict[str, Any]] = {}
    sync_lock = threading.Lock()

    def wait_branches_started_then_release(run_id: str, timeout: float = 5.0) -> None:
        sync = get_sync(run_id)
        deadline = time.time() + timeout
        a_is_set = None
        b_is_set = None
        while time.time() < deadline:
            a_is_set = sync["started_a"].is_set()
            b_is_set = sync["started_b"].is_set()
            if a_is_set and b_is_set:
                sync["release"].set()
                logging.info("release is set")
                return
            if sync["run_done"].is_set():
                raise AssertionError(
                    f"{run_id}: run finished before both branches started"
                )
            time.sleep(0.01)
        logging.info(f"release is not set. {a_is_set=} {b_is_set=}")
        raise AssertionError(f"{run_id}: branches did not both start")

    def get_sync(run_id: str) -> dict[str, Any]:
        with sync_lock:
            sync = sync_by_run.get(run_id)
            if sync is None:
                sync = {
                    "started_a": threading.Event(),
                    "started_b": threading.Event(),
                    "release": threading.Event(),
                    "run_done": threading.Event(),
                }
                sync_by_run[run_id] = sync
            return sync

    def _wait_evt(evt: threading.Event, *, timeout: float, abort_msg: str):
        if not evt.wait(timeout=timeout):
            raise AssertionError(abort_msg)

    resolver = MappingStepResolver()

    @resolver.register("noop")
    def _noop(ctx: StepContext):
        logging.info("noop started")
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow_a")
    def _slow_a(ctx: StepContext):
        logging.info(f"{ctx.run_id}-{ctx.workflow_id} slow_a started")
        sync = get_sync(ctx.run_id)
        sync["started_a"].set()

        # wait until other branch has at least started (no barrier deadlock)
        _wait_evt(
            sync["started_b"],
            timeout=5.0,
            abort_msg=f"{ctx.run_id}: slow_b never started",
        )
        logging.info("slow_a detected b started release")
        _wait_evt(
            sync["release"], timeout=5.0, abort_msg=f"{ctx.run_id}: release not set"
        )
        logging.info("slow_a detected release")
        time.sleep(0.1)
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("slow_b")
    def _slow_b(ctx: StepContext):
        logging.info(f"{ctx.run_id}-{ctx.workflow_id} slow_b started")
        sync = get_sync(ctx.run_id)
        sync["started_b"].set()
        _wait_evt(
            sync["started_a"],
            timeout=5.0,
            abort_msg=f"{ctx.run_id}: slow_a never started",
        )
        logging.info("slow_b detected a started release")
        _wait_evt(
            sync["release"], timeout=5.0, abort_msg=f"{ctx.run_id}: release not set"
        )
        logging.info("slow_b detected release")
        time.sleep(0.1)
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("inner_slow")
    def _inner_slow(ctx: StepContext):
        logging.info(f"{ctx.run_id}-{ctx.workflow_id} inner_slow started")
        time.sleep(0.20)
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("inner_end")
    def _inner_end(ctx: StepContext):
        logging.info(f"{ctx.run_id}-{ctx.workflow_id} inner_end started")
        with ctx.state_write as st:
            st["inner_ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    # Prefer marking nested ops if your resolver supports it (quickfix v2)
    if hasattr(resolver, "nested_ops"):
        resolver.nested_ops.add("nested_call")

    @resolver.register("nested_call")
    def _nested_call(ctx: StepContext):
        logging.info(f"{ctx.run_id}-{ctx.workflow_id} nested_call started")
        # Nested runtime. Prefer reusing ctx.events if runtime supports events injection.
        kwargs = dict(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,  # resolver(op)->fn is canonical
            predicate_registry={},
            max_workers=2,
            transaction_mode="step" if backend_kind == "pg" else "none",
        )
        try:
            if backend_kind == "pg":
                inner_rt = WorkflowRuntime(**kwargs, events=ctx.events)
            else:
                # chroma: avoid sharing EventEmitter until emitter is thread-safe
                inner_rt = WorkflowRuntime(**kwargs)
        except TypeError:
            inner_rt = WorkflowRuntime(**kwargs)

        inner_res = inner_rt.run(
            workflow_id=inner_id,
            conversation_id=ctx.conversation_id or f"conv_inner_{ctx.run_id}",
            turn_node_id=ctx.turn_node_id or f"turn_inner_{ctx.run_id}",
            initial_state={},
            run_id=f"{ctx.run_id}::inner",
        )
        assert inner_res.final_state.get("inner_ended") is True

        with ctx.state_write as st:
            st["nested_done"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("end")
    def _end(ctx: StepContext):
        logging.info(f"{ctx.run_id}-{ctx.workflow_id} end started")
        with ctx.state_write as st:
            st["ended"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    rt = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=4 * len(tags),
        transaction_mode="step" if backend_kind == "pg" else "none",
    )

    # ---- Stress: multiple outer runs concurrently + repeated iterations

    outer_tags = tags  # ["a"]#, "b", "c", "d"]
    # iterations = 5

    for it in range(iterations):
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        run_ids = [f"run_{backend_kind}_{it}_{t}" for t in outer_tags]

        def run_outer(tag: str, run_id: str):
            try:
                rr = rt.run(
                    workflow_id=outer_id,
                    conversation_id=f"conv_{backend_kind}_{it}_{tag}",
                    turn_node_id=f"turn_{backend_kind}_{it}_{tag}",
                    initial_state={},
                    run_id=run_id,
                )
                assert rr.final_state.get("ended") is True
                assert rr.final_state.get("nested_done") is True
            except BaseException as e:
                with errors_lock:
                    errors.append(e)
            finally:
                get_sync(run_id)["run_done"].set()

        with ThreadPoolExecutor(max_workers=len(outer_tags) * 4) as ex:
            futs = []
            for tag, rid in zip(outer_tags, run_ids):
                # precreate sync so coordinator can see it even before tasks start
                get_sync(rid)
                futs.append(ex.submit(run_outer, tag, rid))

            # coordinator: only release once both fork branches are confirmed running
            for rid in run_ids:
                wait_branches_started_then_release(rid, timeout=10.0)

            for f in futs:
                f.result()
        with sync_lock:
            for rid in run_ids:
                sync_by_run.pop(rid, None)

        assert not errors, (
            f"iter={it} errors (likely sqlite sink contention or deadlock): {errors!r}"
        )

    # ---- Sanity: trace db exists and has data
    trace_db = pathlib.Path(workflow_engine.persist_directory) / "wf_trace.sqlite"
    assert trace_db.exists(), "wf_trace.sqlite not created"
    assert trace_db.stat().st_size > 0, "wf_trace.sqlite is empty"
