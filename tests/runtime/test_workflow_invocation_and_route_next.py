from __future__ import annotations

import pytest
import json

from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime import MappingStepResolver, WorkflowRuntime
from kogwistar.runtime.models import (
    RunSuccess,
    WorkflowDesignArtifact,
    WorkflowEdge,
    WorkflowInvocationRequest,
    WorkflowNode,
)
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.workflow, pytest.mark.runtime]


def _span(workflow_id: str) -> Span:
    return Span.from_dummy_for_workflow(workflow_id)


def _node(
    *,
    workflow_id: str,
    node_id: str,
    op: str | None,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op or node_id,
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_fanout": bool(fanout),
            "wf_version": "v_test",
        },
        mentions=[
            Grounding(
                spans=[
                    Span(
                        collection_page_url=f"workflow/{workflow_id}",
                        document_page_url=f"workflow/{workflow_id}",
                        doc_id=f"wf:{workflow_id}",
                        insertion_method="test",
                        page_number=1,
                        start_char=0,
                        end_char=1,
                        excerpt=node_id,
                        context_before="",
                        context_after="",
                        chunk_id=None,
                        source_cluster_id=None,
                        verification=MentionVerification(
                            method="human",
                            is_verified=True,
                            score=1.0,
                            notes="test",
                        ),
                    )
                ]
            )
        ],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    priority: int = 100,
    is_default: bool = True,
    predicate: str | None = None,
    multiplicity: str = "one",
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        label="wf_next",
        type="relationship",
        doc_id=edge_id,
        summary="next",
        properties={},
        source_ids=[src],
        target_ids=[dst],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="wf_next",
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_predicate": predicate,
            "wf_priority": priority,
            "wf_is_default": bool(is_default),
            "wf_multiplicity": multiplicity,
            "wf_version": "v_test",
        },
        mentions=[
            Grounding(
                spans=[
                    Span(
                        collection_page_url=f"workflow/{workflow_id}",
                        document_page_url=f"workflow/{workflow_id}",
                        doc_id=f"wf:{workflow_id}",
                        insertion_method="test",
                        page_number=1,
                        start_char=0,
                        end_char=1,
                        excerpt="edge",
                        context_before="",
                        context_after="",
                        chunk_id=None,
                        source_cluster_id=None,
                        verification=MentionVerification(
                            method="human",
                            is_verified=True,
                            score=1.0,
                            notes="test",
                        ),
                    )
                ]
            )
        ],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _build_engine_pair(tmp_path):
    emb = ConstantEmbeddingFunction(dim=8)
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf"),
        kg_graph_type="workflow",
        embedding_function=emb,
        backend_factory=build_fake_backend,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conv"),
        kg_graph_type="conversation",
        embedding_function=emb,
        backend_factory=build_fake_backend,
    )
    return workflow_engine, conversation_engine


@pytest.fixture
def engine_pair(tmp_path):
    pair = _build_engine_pair(tmp_path)
    try:
        yield pair
    finally:
        for engine in pair:
            engine.close()


@pytest.mark.e2e
def test_goal_loop_is_an_ordinary_cyclic_workflow_pattern(engine_pair):
    """Observe/decide/act/check needs no Goal runtime type or reserved metadata."""
    workflow_engine, conversation_engine = engine_pair
    control_id = "wf_goal_pattern"
    child_id = "wf_goal_action"

    control_nodes = [
        _node(workflow_id=control_id, node_id=f"wf|{control_id}|observe", op="observe", start=True),
        _node(workflow_id=control_id, node_id=f"wf|{control_id}|decide", op="decide"),
        _node(workflow_id=control_id, node_id=f"wf|{control_id}|act", op="act"),
        _node(workflow_id=control_id, node_id=f"wf|{control_id}|check", op="check"),
        _node(workflow_id=control_id, node_id=f"wf|{control_id}|done", op="done", terminal=True),
    ]
    control_edges = [
        _edge(workflow_id=control_id, edge_id=f"wf|{control_id}|observe->decide", src=f"wf|{control_id}|observe", dst=f"wf|{control_id}|decide"),
        _edge(workflow_id=control_id, edge_id=f"wf|{control_id}|decide->act", src=f"wf|{control_id}|decide", dst=f"wf|{control_id}|act"),
        _edge(workflow_id=control_id, edge_id=f"wf|{control_id}|act->check", src=f"wf|{control_id}|act", dst=f"wf|{control_id}|check"),
        _edge(workflow_id=control_id, edge_id=f"wf|{control_id}|check->observe", src=f"wf|{control_id}|check", dst=f"wf|{control_id}|observe"),
        _edge(workflow_id=control_id, edge_id=f"wf|{control_id}|check->done", src=f"wf|{control_id}|check", dst=f"wf|{control_id}|done"),
    ]
    child_nodes = [
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|run", op="action", start=True),
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|done", op="child_done", terminal=True),
    ]
    child_edges = [
        _edge(workflow_id=child_id, edge_id=f"wf|{child_id}|run->done", src=f"wf|{child_id}|run", dst=f"wf|{child_id}|done"),
    ]
    for node in [*control_nodes, *child_nodes]:
        workflow_engine.write.add_node(node)
    for edge in [*control_edges, *child_edges]:
        workflow_engine.write.add_edge(edge)

    resolver = MappingStepResolver()
    child_calls: list[int] = []
    child_lineage_counts: list[int] = []
    fake_llm_payloads = iter(
        [
            '{"action":"wf_goal_action","goal_status":"active"}',
            '{"action":"wf_goal_action","goal_status":"satisfied"}',
        ]
    )
    llm_calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        llm_calls.append(prompt)
        return next(fake_llm_payloads)

    @resolver.register("observe")
    def _observe(ctx):
        return RunSuccess(state_update=[], _route_next=["decide"])

    @resolver.register("decide")
    def _decide(ctx):
        iteration = int(ctx.state_view.get("goal_iteration", 0)) + 1
        decision = json.loads(fake_llm(f"choose action for iteration {iteration}"))
        with ctx.state_write as state:
            state["goal_iteration"] = iteration
            state["selected_action"] = decision["action"]
        return RunSuccess(state_update=[], _route_next=["act"])

    @resolver.register("act")
    def _act(ctx):
        iteration = int(ctx.state_view["goal_iteration"])
        return RunSuccess(
            state_update=[],
            _route_next=["check"],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=child_id,
                    invocation_key=f"goal-iteration-{iteration}",
                    result_state_key="latest_action",
                )
            ],
        )

    @resolver.register("check")
    def _check(ctx):
        satisfied = int(ctx.state_view["goal_iteration"]) >= 2
        with ctx.state_write as state:
            state["goal_status"] = "satisfied" if satisfied else "active"
        return RunSuccess(
            state_update=[], _route_next=["done" if satisfied else "observe"]
        )

    @resolver.register("done")
    def _done(ctx):
        return RunSuccess(state_update=[])

    @resolver.register("action")
    def _action(ctx):
        lineage = conversation_engine.read.get_edges(
            ids=[f"wf_invoked|run_goal_pattern|{ctx.run_id}"], limit=1
        )
        child_lineage_counts.append(len(lineage))
        child_calls.append(int(ctx.state_view["goal_iteration"]))
        return RunSuccess(state_update=[], _route_next=["child_done"])

    @resolver.register("child_done")
    def _child_done(ctx):
        return RunSuccess(state_update=[])

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )
    result = runtime.run(
        workflow_id=control_id,
        conversation_id="conv_goal_pattern",
        turn_node_id="turn_goal_pattern",
        initial_state={"goal_objective": "complete two actions"},
        run_id="run_goal_pattern",
    )

    assert result.status == "succeeded", result.final_state
    assert result.final_state["goal_status"] == "satisfied"
    assert result.final_state["goal_iteration"] == 2
    assert child_calls == [1, 2]
    assert len(llm_calls) == 2
    assert result.final_state["selected_action"] == child_id
    assert child_lineage_counts == [1, 1]
    assert "wf_mode" not in control_nodes[0].metadata
    child_runs = conversation_engine.read.get_nodes(
        where={
            "$and": [
                {"entity_type": "workflow_run"},
                {"parent_run_id": "run_goal_pattern"},
            ]
        },
        limit=10,
    )
    assert len(child_runs) == 2
    assert {node.metadata["invocation_key"] for node in child_runs} == {
        "goal-iteration-1",
        "goal-iteration-2",
    }


def test_nested_workflow_synthesized_design_is_persisted_and_used(engine_pair):
    """Async mirror: `tests/runtime/test_async_runtime_contract.py::test_async_runtime_nested_workflow_invocation_matches_sync`."""
    workflow_engine, conversation_engine = engine_pair

    parent_id = "wf_parent_spawn_child"
    child_id = "wf_child_on_the_fly"

    parent_nodes = [
        _node(workflow_id=parent_id, node_id=f"wf|{parent_id}|start", op="start", start=True),
        _node(workflow_id=parent_id, node_id=f"wf|{parent_id}|spawn", op="spawn"),
        _node(workflow_id=parent_id, node_id=f"wf|{parent_id}|end", op="end", terminal=True),
    ]
    parent_edges = [
        _edge(
            workflow_id=parent_id,
            edge_id=f"wf|{parent_id}|e|start->spawn",
            src=f"wf|{parent_id}|start",
            dst=f"wf|{parent_id}|spawn",
        ),
        _edge(
            workflow_id=parent_id,
            edge_id=f"wf|{parent_id}|e|spawn->end",
            src=f"wf|{parent_id}|spawn",
            dst=f"wf|{parent_id}|end",
        ),
    ]
    for node in parent_nodes:
        workflow_engine.write.add_node(node)
    for edge in parent_edges:
        workflow_engine.write.add_edge(edge)

    child_nodes = [
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|start", op="start", start=True),
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|body", op="child_body"),
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|end", op="end", terminal=True),
    ]
    child_edges = [
        _edge(
            workflow_id=child_id,
            edge_id=f"wf|{child_id}|e|start->body",
            src=f"wf|{child_id}|start",
            dst=f"wf|{child_id}|body",
        ),
        _edge(
            workflow_id=child_id,
            edge_id=f"wf|{child_id}|e|body->end",
            src=f"wf|{child_id}|body",
            dst=f"wf|{child_id}|end",
        ),
    ]
    child_design = WorkflowDesignArtifact(
        workflow_id=child_id,
        workflow_version="v_test",
        start_node_id=f"wf|{child_id}|start",
        nodes=child_nodes,
        edges=child_edges,
        source_run_id="parent-run",
        source_workflow_id=parent_id,
        source_step_id=f"wf|{parent_id}|spawn",
        notes="synthetic child design",
    )

    resolver = MappingStepResolver()
    child_body_calls: list[bool] = []

    @resolver.register("start")
    def _start(ctx):
        with ctx.state_write as st:
            st["started"] = True
        return RunSuccess(state_update=[], _route_next=["spawn"])

    @resolver.register("spawn")
    def _spawn(ctx):
        with ctx.state_write as st:
            st["spawn_seen"] = True
            st["seed"] = "propagated"
        return RunSuccess(
            state_update=[("u", {"spawned": True})],
            _route_next=["end"],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=child_id,
                    workflow_design=child_design,
                    result_state_key="child_result",
                )
            ],
        )

    @resolver.register("child_body")
    def _child_body(ctx):
        child_body_calls.append(True)
        with ctx.state_write as st:
            st["child_done"] = True
            st["child_seed"] = ctx.state_view.get("seed")
        return RunSuccess(state_update=[])

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as st:
            st["ended"] = True
        return RunSuccess(state_update=[])

    rt = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=4,
    )

    rr = rt.run(
        workflow_id=parent_id,
        conversation_id="conv_nested_design",
        turn_node_id="turn_nested_design",
        initial_state={"_deps": {}, "seed": "present"},
        run_id="run_nested_design",
    )

    assert rr.status == "succeeded"
    assert rr.final_state["ended"] is True
    assert rr.final_state["child_result"]["child_done"] is True
    assert rr.final_state["child_result"]["child_seed"] == "propagated"
    assert rr.final_state["child_result"]["seed"] == "propagated"
    assert rr.final_state["child_result__workflow_id"] == child_id
    assert rr.final_state["child_result__status"] == "succeeded"
    child_run_id = str(rr.final_state["child_result__run_id"])
    child_run_nodes = conversation_engine.read.get_nodes(
        ids=[f"wf_run|{child_run_id}"], limit=1
    )
    assert len(child_run_nodes) == 1
    child_run_metadata = dict(child_run_nodes[0].metadata)
    assert child_run_metadata["parent_run_id"] == "run_nested_design"
    assert child_run_metadata["result_state_key"] == "child_result"
    lineage_edges = conversation_engine.read.get_edges(
        ids=[f"wf_invoked|run_nested_design|{child_run_id}"], limit=1
    )
    assert len(lineage_edges) == 1
    assert lineage_edges[0].relation == "wf_invoked"

    rt._persist_workflow_run(
        conversation_id="conv_nested_design",
        workflow_id=child_id,
        run_id=child_run_id,
        turn_node_id="turn_nested_design",
        status="running",
    )
    resumed_child_nodes = conversation_engine.read.get_nodes(
        ids=[f"wf_run|{child_run_id}"], limit=1
    )
    assert resumed_child_nodes[0].metadata["parent_run_id"] == "run_nested_design"
    assert child_body_calls == [True]
    replayed_child = rt.run_subworkflow(
        workflow_id=child_id,
        parent_state={},
        conversation_id="conv_nested_design",
        turn_node_id="turn_nested_design",
        parent_run_id="run_nested_design",
        result_state_key="child_result",
        run_id=child_run_id,
    )
    assert replayed_child.status == "succeeded"
    assert child_body_calls == [True]

    persisted_child_nodes = workflow_engine.read.get_nodes(
        where={
            "$and": [
                {"entity_type": "workflow_node"},
                {"workflow_id": child_id},
            ]
        },
        limit=100,
    )
    assert len(persisted_child_nodes) == 3


def test_nested_workflow_failure_short_circuits_parent_routing(engine_pair):
    """Async mirror: `tests/runtime/test_async_runtime_contract.py::test_async_runtime_nested_workflow_child_failure_fails_parent`."""
    workflow_engine, conversation_engine = engine_pair

    parent_id = "wf_parent_nested_failure"
    child_id = "wf_child_nested_failure"

    parent_nodes = [
        _node(workflow_id=parent_id, node_id=f"wf|{parent_id}|start", op="start", start=True),
        _node(workflow_id=parent_id, node_id=f"wf|{parent_id}|spawn", op="spawn"),
        _node(workflow_id=parent_id, node_id=f"wf|{parent_id}|end", op="end", terminal=True),
    ]
    parent_edges = [
        _edge(
            workflow_id=parent_id,
            edge_id=f"wf|{parent_id}|e|start->spawn",
            src=f"wf|{parent_id}|start",
            dst=f"wf|{parent_id}|spawn",
        ),
        _edge(
            workflow_id=parent_id,
            edge_id=f"wf|{parent_id}|e|spawn->end",
            src=f"wf|{parent_id}|spawn",
            dst=f"wf|{parent_id}|end",
        ),
    ]
    for node in parent_nodes:
        workflow_engine.write.add_node(node)
    for edge in parent_edges:
        workflow_engine.write.add_edge(edge)

    child_nodes = [
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|start", op="start", start=True),
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|boom", op="boom"),
        _node(workflow_id=child_id, node_id=f"wf|{child_id}|end", op="end", terminal=True),
    ]
    child_edges = [
        _edge(
            workflow_id=child_id,
            edge_id=f"wf|{child_id}|e|start->boom",
            src=f"wf|{child_id}|start",
            dst=f"wf|{child_id}|boom",
        ),
        _edge(
            workflow_id=child_id,
            edge_id=f"wf|{child_id}|e|boom->end",
            src=f"wf|{child_id}|boom",
            dst=f"wf|{child_id}|end",
        ),
    ]
    child_design = WorkflowDesignArtifact(
        workflow_id=child_id,
        workflow_version="v_test",
        start_node_id=f"wf|{child_id}|start",
        nodes=child_nodes,
        edges=child_edges,
    )

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx):
        return RunSuccess(state_update=[], _route_next=["spawn"])

    @resolver.register("spawn")
    def _spawn(ctx):
        with ctx.state_write as st:
            st["spawn_seen"] = True
        return RunSuccess(
            state_update=[("u", {"spawned": True})],
            _route_next=["end"],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id=child_id,
                    workflow_design=child_design,
                    result_state_key="child_result",
                )
            ],
        )

    @resolver.register("boom")
    def _boom(ctx):
        raise ValueError("child exploded")

    @resolver.register("end")
    def _end(ctx):
        with ctx.state_write as st:
            st["ended"] = True
        return RunSuccess(state_update=[])

    rt = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=4,
    )

    rr = rt.run(
        workflow_id=parent_id,
        conversation_id="conv_nested_failure",
        turn_node_id="turn_nested_failure",
        initial_state={"_deps": {}},
        run_id="run_nested_failure",
    )

    assert rr.status == "failure"
    assert rr.final_state["spawn_seen"] is True
    assert "ended" not in rr.final_state
