import pytest
pytestmark = pytest.mark.core
import json
from pathlib import Path


from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Grounding,
    Span,
    MentionVerification,
)

from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.runtime.replay import load_checkpoint, replay_to
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
from tests.conftest import FakeEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend


BACKEND_PARAMS = [
    pytest.param("fake", id="fake", marks=pytest.mark.ci),
    pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
]


def _make_engine(tmp_path: Path, *, graph_type: str, backend_kind: str) -> GraphKnowledgeEngine:
    if backend_kind == "fake":
        return GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            kg_graph_type=graph_type,
            embedding_function=FakeEmbeddingFunction(),
            backend_factory=build_fake_backend,
        )
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type=graph_type,
    )


def _reopen_engine(tmp_path: Path, *, graph_type: str, backend_kind: str) -> GraphKnowledgeEngine:
    return _make_engine(tmp_path, graph_type=graph_type, backend_kind=backend_kind)


def _runtime_engine(
    engine: GraphKnowledgeEngine,
    tmp_path: Path,
    *,
    graph_type: str,
    backend_kind: str,
) -> GraphKnowledgeEngine:
    if backend_kind == "fake":
        return engine
    return _reopen_engine(tmp_path, graph_type=graph_type, backend_kind=backend_kind)


def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(
    *, workflow_id: str, node_id: str, op: str, start=False, terminal=False
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    )


def _wf_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None,
    priority: int,
    is_default: bool,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_runtime_checkpoint_load_and_replay(tmp_path: Path, backend_kind: str):
    """
    End-to-end:
      1) Create workflow design (saved in workflow_engine)
      2) Run via WorkflowRuntime (persists workflow_step_exec + workflow_checkpoint nodes)
      3) load_checkpoint(...) returns JSON state snapshot
      4) replay_to(...) reconstructs state up to target step using step exec results
    """
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_checkpoint_load_replay_smoke"

    workflow_engine = _make_engine(wf_dir, graph_type="workflow", backend_kind=backend_kind)
    conversation_engine = _make_engine(
        conv_dir, graph_type="conversation", backend_kind=backend_kind
    )

    # ---- Build a 3-step workflow: a -> b -> end ----
    n_a = _wf_node(
        workflow_id=workflow_id, node_id=f"wf|{workflow_id}|a", op="a", start=True
    )
    n_b = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|b", op="b")
    n_end = _wf_node(
        workflow_id=workflow_id,
        node_id=f"wf|{workflow_id}|end",
        op="end",
        terminal=True,
    )

    for n in [n_a, n_b, n_end]:
        workflow_engine.add_node(n)

    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|a->b",
            src=n_a.id,
            dst=n_b.id,
            predicate=None,
            priority=100,
            is_default=True,
        )
    )
    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|b->end",
            src=n_b.id,
            dst=n_end.id,
            predicate=None,
            priority=100,
            is_default=True,
        )
    )

    # workflow_engine.persist()

    # ---- Run workflow (persist checkpoints/traces) ----
    predicate_registry = {}  # no predicates used

    def resolve_step(op: str):
        def _fn(ctx):
            # keep state JSONable
            with ctx.state_write as state:
                state.setdefault("op_log", [])
                state["op_log"].append(op)
            # return JSONable result (will be stored at state["result.<op>"])
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("a", {"op": op}),
                    ("u", {f"result.{op}": {"value": f"v_{op}"}}),
                ],
            )

        return _fn

    # IMPORTANT: set large checkpoint interval so only checkpoint at step_seq=0
    # Runtime logic: checkpoint if (step_seq % N) == 0. With N=9999 => only at 0 for a short workflow.
    rt = WorkflowRuntime(
        workflow_engine=_runtime_engine(
            workflow_engine, wf_dir, graph_type="workflow", backend_kind=backend_kind
        ),
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=9999,
        max_workers=1,
    )

    run_result = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv1",
        turn_node_id="turn1",
        initial_state={},
    )
    final_state, run_id = run_result.final_state, run_result.run_id
    # Sanity: ensure it ran all steps
    assert final_state["op_log"] == ["a", "b", "end"]
    assert final_state["result.a"]["value"] == "v_a"
    assert final_state["result.b"]["value"] == "v_b"
    assert final_state["result.end"]["value"] == "v_end"

    # ---- Load checkpoint (step_seq=0) ----
    ckpt0 = load_checkpoint(
        conversation_engine=conversation_engine, run_id=run_id, step_seq=0
    )
    # checkpoint state_json is stored via try_serialize_with_ref(state), so must be valid JSON
    assert isinstance(ckpt0, dict)
    # at step_seq=0, state should already include result.a
    assert ckpt0["result.a"]["value"] == "v_a"

    # ---- Replay/reconstruct to a later step_seq ----
    # step_seq mapping in runtime:
    #   seq0 => node 'a'
    #   seq1 => node 'b'
    #   seq2 => node 'end'
    reconstructed = replay_to(
        conversation_engine=conversation_engine, run_id=run_id, target_step_seq=2
    )

    assert reconstructed["result.a"]["value"] == "v_a"
    assert reconstructed["result.b"]["value"] == "v_b"
    assert reconstructed["result.end"]["value"] == "v_end"

    # Ensure the reconstructed state is JSON serializable (resume-friendly)
    json.dumps(reconstructed)


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_runtime_resume_from_checkpoint(tmp_path: Path, backend_kind: str):
    """
    Resume semantics without needing a "start_from_step" API:

    Workflow has a gate start node that routes:
      - if state already has result.a -> go to b
      - else -> go to a

    Run #1: {} => gate -> a -> gate -> b -> end
    Load checkpoint after 'a', then Run #2 with initial_state=checkpoint_state:
      => gate -> b -> end (skipping a)
    """
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_resume_from_checkpoint_gate"

    workflow_engine = _make_engine(wf_dir, graph_type="workflow", backend_kind=backend_kind)
    conversation_engine = _make_engine(
        conv_dir, graph_type="conversation", backend_kind=backend_kind
    )

    # ---- Build workflow design in workflow_engine (no persist() needed) ----
    n_gate = _wf_node(
        workflow_id=workflow_id, node_id=f"wf|{workflow_id}|gate", op="gate", start=True
    )
    n_a = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|a", op="a")
    n_b = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|b", op="b")
    n_end = _wf_node(
        workflow_id=workflow_id,
        node_id=f"wf|{workflow_id}|end",
        op="end",
        terminal=True,
    )

    for n in [n_gate, n_a, n_b, n_end]:
        workflow_engine.add_node(n)

    # gate -> b if has_done_a
    e_gate_to_b = _wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|gate->b",
        src=n_gate.id,
        dst=n_b.id,
        predicate="has_done_a",
        priority=0,
        is_default=False,
    )
    workflow_engine.add_edge(e_gate_to_b)
    # gate -> a default
    e_gate_a_default = _wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|gate->a|default",
        src=n_gate.id,
        dst=n_a.id,
        predicate=None,
        priority=100,
        is_default=True,
    )
    workflow_engine.add_edge(e_gate_a_default)
    e_a_gate = _wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|a->gate",
        src=n_a.id,
        dst=n_gate.id,
        predicate=None,
        priority=100,
        is_default=True,
    )
    # a -> gate
    workflow_engine.add_edge(e_a_gate)
    e_b_end = _wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|b->end",
        src=n_b.id,
        dst=n_end.id,
        predicate=None,
        priority=100,
        is_default=True,
    )
    # b -> end
    workflow_engine.add_edge(e_b_end)

    # Re-open workflow engine from disk to prove the design is actually stored.
    # The fake backend keeps the lightweight path, but it does not guarantee
    # cross-instance persistence like the Chroma-backed row.
    workflow_engine2 = _runtime_engine(
        workflow_engine, wf_dir, graph_type="workflow", backend_kind=backend_kind
    )

    # predicate_registry = {
    #     "has_done_a": lambda st, _r: "result.a" in st,
    # }
    def done_a(e, st, r):
        return "result.a" in st

    predicate_registry = {
        "has_done_a": done_a,  # lambda e, st, r: "result.a" in st,
    }

    def resolve_step(op: str):
        def _fn(ctx):
            with ctx.state_write as state:
                state.setdefault("op_log", [])
                state["op_log"].append(op)
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("a", {"op": op}),
                    ("u", {f"result.{op}": {"value": f"v_{op}"}}),
                ],
            )

        return _fn

    # ---- Run #1 ----
    rt1 = WorkflowRuntime(
        workflow_engine=workflow_engine2,
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    run_result = rt1.run(
        workflow_id=workflow_id,
        conversation_id="conv1",
        turn_node_id="turn1",
        initial_state={},
    )
    final1, run_id1 = run_result.final_state, run_result.run_id
    assert final1["result.a"]["value"] == "v_a"

    # load checkpoint after step_seq=1 (with max_workers=1, step order is deterministic)
    ckpt_after_a = load_checkpoint(
        conversation_engine=conversation_engine, run_id=run_id1, step_seq=1
    )
    assert ckpt_after_a["result.a"]["value"] == "v_a"
    json.dumps(ckpt_after_a)

    # ---- Run #2: resume by starting a new run with loaded state ----
    rt2 = WorkflowRuntime(
        workflow_engine=workflow_engine2,
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    run_result = rt2.run(
        workflow_id=workflow_id,
        conversation_id="conv2",
        turn_node_id="turn2",
        initial_state=ckpt_after_a,
        run_id=None,  # new run id; avoids id collisions in persisted nodes
    )
    final2, run_id2 = run_result.final_state, run_result.run_id
    assert final2["result.a"]["value"] == "v_a"  # carried forward
    assert final2["result.b"]["value"] == "v_b"  # executed
    assert final2["result.end"]["value"] == "v_end"  # executed
    assert "a" in final2.get("op_log", [])  # continue, must contain a from save time


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_runtime_resume_from_checkpoint_frontier(
    tmp_path: Path, backend_kind: str
):
    """True resume from checkpoint using runtime frontier (_rt_join).

    This validates that checkpoints capture a resume-able scheduler frontier,
    so a new WorkflowRuntime run can continue from the *next pending node*
    without re-entering the workflow start node.

    Workflow: a -> b -> end

    Run #1 executes step_seq=0 (a). We load checkpoint at step_seq=0. That checkpoint must contain
    a pending token targeting node 'b'.

    Run #2 starts with initial_state=ckpt0 and a NEW run_id. It must execute b->end only.
    """
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_resume_from_checkpoint_frontier"

    workflow_engine = _make_engine(
        wf_dir, graph_type="workflow", backend_kind=backend_kind
    )
    conversation_engine = _make_engine(
        conv_dir, graph_type="conversation", backend_kind=backend_kind
    )

    # ---- Build workflow: a -> b -> end ----
    n_a = _wf_node(
        workflow_id=workflow_id, node_id=f"wf|{workflow_id}|a", op="a", start=True
    )
    n_b = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|b", op="b")
    n_end = _wf_node(
        workflow_id=workflow_id,
        node_id=f"wf|{workflow_id}|end",
        op="end",
        terminal=True,
    )

    for n in [n_a, n_b, n_end]:
        workflow_engine.add_node(n)

    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|a->b",
            src=n_a.id,
            dst=n_b.id,
            predicate=None,
            priority=100,
            is_default=True,
        )
    )
    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|b->end",
            src=n_b.id,
            dst=n_end.id,
            predicate=None,
            priority=100,
            is_default=True,
        )
    )

    predicate_registry = {}

    def resolve_step(op: str):
        def _fn(ctx):
            with ctx.state_write as state:
                state.setdefault("op_log", [])
                state["op_log"].append(op)
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("a", {"op": op}),
                    ("u", {f"result.{op}": {"value": f"v_{op}"}}),
                ],
            )

        return _fn

    rt1 = WorkflowRuntime(
        workflow_engine=_runtime_engine(
            workflow_engine, wf_dir, graph_type="workflow", backend_kind=backend_kind
        ),
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    run_result = rt1.run(
        workflow_id=workflow_id,
        conversation_id="conv1",
        turn_node_id="turn1",
        initial_state={},
    )
    final1, run_id1 = run_result.final_state, run_result.run_id
    assert final1["op_log"] == ["a", "b", "end"]

    ckpt0 = load_checkpoint(
        conversation_engine=conversation_engine, run_id=run_id1, step_seq=0
    )
    assert ckpt0["result.a"]["value"] == "v_a"

    # checkpoint must include runtime frontier
    assert "_rt_join" in ckpt0
    pend = (ckpt0.get("_rt_join") or {}).get("pending") or []
    # pending tokens are tuples like (node_id, mask, token_id, parent_token_id)
    assert any(
        (isinstance(t, (list, tuple)) and len(t) >= 1 and str(t[0]).endswith("|b"))
        for t in pend
    )

    rt2 = WorkflowRuntime(
        workflow_engine=_runtime_engine(
            workflow_engine, wf_dir, graph_type="workflow", backend_kind=backend_kind
        ),
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    run_result = rt2.run(
        workflow_id=workflow_id,
        conversation_id="conv2",
        turn_node_id="turn2",
        initial_state=ckpt0,
        run_id=None,  # new run id; avoids collisions
    )
    final2, run_id2 = run_result.final_state, run_result.run_id
    assert final2["result.a"]["value"] == "v_a"
    assert final2["result.b"]["value"] == "v_b"
    assert final2["result.end"]["value"] == "v_end"

    # ensure 'a' was not re-executed (it should only appear once, carried from ckpt)
    assert final2.get("op_log", []).count("a") == 1
