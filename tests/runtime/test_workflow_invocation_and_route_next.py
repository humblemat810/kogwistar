from __future__ import annotations

import pytest

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

pytestmark = [pytest.mark.workflow]


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


def test_nested_workflow_synthesized_design_is_persisted_and_used(tmp_path):
    workflow_engine, conversation_engine = _build_engine_pair(tmp_path)

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


def test_route_next_alias_can_fan_out_multiple_branches(tmp_path):
    workflow_engine, conversation_engine = _build_engine_pair(tmp_path)

    workflow_id = "wf_route_next_fanout"
    nodes = [
        _node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|start", op="start", start=True),
        _node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|decide", op="decide"),
        _node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|left", op="left"),
        _node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|right", op="right"),
        _node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|end", op="end", terminal=True),
    ]
    edges = [
        _edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|start->decide",
            src=f"wf|{workflow_id}|start",
            dst=f"wf|{workflow_id}|decide",
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|decide->left",
            src=f"wf|{workflow_id}|decide",
            dst=f"wf|{workflow_id}|left",
            is_default=False,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|decide->right",
            src=f"wf|{workflow_id}|decide",
            dst=f"wf|{workflow_id}|right",
            is_default=False,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|left->end",
            src=f"wf|{workflow_id}|left",
            dst=f"wf|{workflow_id}|end",
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|e|right->end",
            src=f"wf|{workflow_id}|right",
            dst=f"wf|{workflow_id}|end",
        ),
    ]
    for node in nodes:
        workflow_engine.write.add_node(node)
    for edge in edges:
        workflow_engine.write.add_edge(edge)

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx):
        return RunSuccess(state_update=[], _route_next=["decide"])

    @resolver.register("decide")
    def _decide(ctx):
        return RunSuccess(state_update=[], _route_next=["left", "right"])

    @resolver.register("left")
    def _left(ctx):
        with ctx.state_write as st:
            st["left_seen"] = True
        return RunSuccess(state_update=[], _route_next=["end"])

    @resolver.register("right")
    def _right(ctx):
        with ctx.state_write as st:
            st["right_seen"] = True
        return RunSuccess(state_update=[], _route_next=["end"])

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
        workflow_id=workflow_id,
        conversation_id="conv_route_next",
        turn_node_id="turn_route_next",
        initial_state={"_deps": {}},
        run_id="run_route_next",
    )

    assert rr.status == "succeeded"
    assert rr.final_state["left_seen"] is True
    assert rr.final_state["right_seen"] is True
    assert rr.final_state["ended"] is True


def test_nested_workflow_failure_short_circuits_parent_routing(tmp_path):
    workflow_engine, conversation_engine = _build_engine_pair(tmp_path)

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
