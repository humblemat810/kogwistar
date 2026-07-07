from __future__ import annotations

import contextlib
import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.policy import DefaultDreamLoopPolicy
from kogwistar.runtime.models import WorkflowDesignArtifact, WorkflowEdge, WorkflowNode
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.wisdom.agentic_dream_design import (
    build_dream_maintenance_workflow_design,
    dream_workflow_expected_ops,
    materialize_dream_workflow_design,
)
from kogwistar.wisdom.resolvers import dream_default_resolver
from tests._helpers.fake_backend import build_fake_backend


pytestmark = [pytest.mark.core, pytest.mark.runtime, pytest.mark.e2e]


def _make_engine(graph_type: str) -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / f".tmp_dream_agent_{graph_type}" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
        kg_graph_type=graph_type,
    )
    return engine, test_db_dir


def _close_engines(*engines: GraphKnowledgeEngine) -> None:
    for engine in engines:
        with contextlib.suppress(Exception):
            engine.close()


def _mentions(workflow_id: str) -> list[Grounding]:
    return [Grounding(spans=[Span.from_dummy_for_workflow(workflow_id)])]


def _signal_node(
    *,
    node_id: str,
    workflow_id: str,
    run_id: str,
    step_op: str,
    completed_at_ms: int,
    failure_count: int,
    error_count: int,
    success_count: int,
    duration_ms_total: int,
    duration_ms_max: int,
    workspace_id: str = "demo",
) -> Node:
    return Node(
        id=node_id,
        label=f"{step_op}:{run_id}",
        type="entity",
        summary="workflow step summary",
        mentions=_mentions(workflow_id),
        metadata={
            "workspace_id": workspace_id,
            "entity_type": "workflow_step_exec",
            "workflow_id": workflow_id,
            "run_id": run_id,
            "step_op": step_op,
            "completed_at_ms": completed_at_ms,
            "failure_count": failure_count,
            "error_count": error_count,
            "success_count": success_count,
            "duration_ms_total": duration_ms_total,
            "duration_ms_max": duration_ms_max,
            "status": "failure" if failure_count or error_count else "success",
        },
    )


def _feedback_node(
    *,
    node_id: str,
    workflow_id: str,
    run_id: str,
    feedback_text: str,
    feedback_score: float,
    candidate_workflow_id: str,
    feedback_sabotage: bool = False,
    workspace_id: str = "demo",
) -> Node:
    return Node(
        id=node_id,
        label=f"feedback:{run_id}",
        type="entity",
        summary=feedback_text,
        mentions=_mentions(workflow_id),
        metadata={
            "workspace_id": workspace_id,
            "entity_type": "conversation_feedback",
            "feedback_kind": "conversation_feedback",
            "workflow_id": workflow_id,
            "candidate_workflow_id": candidate_workflow_id,
            "run_id": run_id,
            "step_op": "conversation_feedback",
            "feedback_text": feedback_text,
            "feedback_score": feedback_score,
            "feedback_trust_score": 1.0,
            "feedback_sabotage": feedback_sabotage,
        },
    )


def _build_candidate_design(proposal, evaluation) -> WorkflowDesignArtifact:
    workflow_id = f"{proposal.workflow_id}.rev.{proposal.step_op}"
    start_id = f"wf:{workflow_id}:start"
    end_id = f"wf:{workflow_id}:end"
    nodes = [
        WorkflowNode(
            id=start_id,
            label="Start",
            type="entity",
            doc_id=start_id,
            summary="Start",
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": "start",
                "wf_start": True,
                "wf_terminal": False,
                "wf_version": "v1",
            },
            mentions=_mentions(workflow_id),
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        ),
        WorkflowNode(
            id=end_id,
            label="End",
            type="entity",
            doc_id=end_id,
            summary="End",
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": "end",
                "wf_start": False,
                "wf_terminal": True,
                "wf_version": "v1",
            },
            mentions=_mentions(workflow_id),
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        ),
    ]
    edges = [
        WorkflowEdge(
            id=f"wf:{workflow_id}:e:start->end",
            source_ids=[start_id],
            target_ids=[end_id],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="wf_next",
            doc_id=workflow_id,
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_predicate": None,
                "wf_priority": 0,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_edge_kind": "wf_next",
                "wf_version": "v1",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            mentions=_mentions(workflow_id),
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    ]
    return WorkflowDesignArtifact(
        workflow_id=workflow_id,
        workflow_version="v1",
        start_node_id=start_id,
        nodes=nodes,
        edges=edges,
        source_run_id=proposal.run_id,
        source_workflow_id=proposal.workflow_id,
        source_step_id=proposal.step_op,
        notes=evaluation.rationale,
    )


def _runtime(
    *,
    workflow_engine: GraphKnowledgeEngine,
    conversation_engine: GraphKnowledgeEngine,
) -> WorkflowRuntime:
    return WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=dream_default_resolver,
        predicate_registry={},
        checkpoint_every_n_steps=9999,
        max_workers=1,
    )


def test_dream_agent_resolver_registers_all_expected_ops():
    assert dream_default_resolver.ops == set(dream_workflow_expected_ops())


def test_dream_workflow_design_matches_expected_ops_and_linear_shape():
    workflow_id = "dream.runtime.shape"
    design = build_dream_maintenance_workflow_design(workflow_id=workflow_id)

    assert design.workflow_id == workflow_id
    assert design.start_node_id == f"wf:{workflow_id}:start"
    assert [node.metadata.get("wf_op") for node in design.nodes] == list(
        dream_workflow_expected_ops()
    )
    assert [edge.source_ids[0] for edge in design.edges] == [node.id for node in design.nodes[:-1]]
    assert [edge.target_ids[0] for edge in design.edges] == [node.id for node in design.nodes[1:]]


def test_dream_agent_workflow_executes_each_step_and_persists_pending_candidate():
    source_engine, source_dir = _make_engine("workflow")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    conversation_engine, conversation_dir = _make_engine("conversation")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        dream_workflow_id = "dream.runtime.pending"
        target_workflow_id = "wf-target"
        source_namespace = "ws:demo:source"
        wisdom_namespace = "ws:demo:wisdom"
        dream_conversation_namespace = "ws:demo:dream-conversation"
        materialized_workflow_namespace = "ws:demo:workflow-materialized"
        materialize_dream_workflow_design(
            workflow_engine,
            build_dream_maintenance_workflow_design(workflow_id=dream_workflow_id),
        )

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="origin-fail",
                    workflow_id=target_workflow_id,
                    run_id="run-origin-fail",
                    step_op="retrieve",
                    completed_at_ms=1000,
                    failure_count=3,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=1200,
                    duration_ms_max=600,
                )
            )

        runtime = _runtime(workflow_engine=workflow_engine, conversation_engine=conversation_engine)
        result = runtime.run(
            workflow_id=dream_workflow_id,
            conversation_id="conv-dream-1",
            turn_node_id="turn-dream-1",
            run_id="run-dream-1",
            initial_state={
                "dream_budget_remaining": 5,
                "_deps": {
                    "source_engine": source_engine,
                    "wisdom_engine": wisdom_engine,
                    "conversation_engine": conversation_engine,
                    "workflow_engine": workflow_engine,
                    "source_namespace": source_namespace,
                    "source_where": {"workspace_id": "demo"},
                    "wisdom_namespace": wisdom_namespace,
                    "dream_conversation_namespace": dream_conversation_namespace,
                    "materialized_workflow_namespace": materialized_workflow_namespace,
                    "target_workflow_id": target_workflow_id,
                    "dream_created_at_ms": 12345,
                    "dream_policy": DefaultDreamLoopPolicy(
                        recent_limit=1,
                        hotspot_limit=0,
                        stale_sample_limit=0,
                        max_proposals_per_tick=1,
                        min_evaluation_runs=2,
                    ),
                    "dream_summary_builder": lambda signal: "Add bounded retry after repeated retrieve failures",
                    "dream_reasoning_builder": lambda signal: (
                        "seeded stats show repeated retrieve failures",
                        "candidate should remain pending until more runs arrive",
                    ),
                    "dream_suggested_change_builder": lambda signal: {
                        "dream_feature": "workflow_revision",
                        "step_op": signal.step_op,
                        "action": "add_bounded_retry",
                    },
                    "dream_confidence_builder": lambda signal: 0.95,
                    "dream_workflow_builder": _build_candidate_design,
                },
            },
        )

        final_state = result.final_state or {}
        with scoped_namespace(conversation_engine, dream_conversation_namespace):
            reasoning_nodes = conversation_engine.read.get_nodes(
                where={"artifact_kind": "dream_reasoning_trace", "workspace_id": "demo"}
            )
            leaked_proposal_nodes = conversation_engine.read.get_nodes(
                where={"artifact_kind": "workflow_revision_proposal", "workspace_id": "demo"}
            )
        with scoped_namespace(wisdom_engine, wisdom_namespace):
            proposal_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "workflow_revision_proposal", "workspace_id": "demo"}
            )
            leaked_reasoning_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "dream_reasoning_trace", "workspace_id": "demo"}
            )
        with scoped_namespace(workflow_engine, materialized_workflow_namespace):
            candidate_nodes = workflow_engine.read.get_nodes(
                where={"workflow_id": "wf-target.rev.retrieve", "entity_type": "workflow_node"}
            )
        step_exec_nodes = conversation_engine.read.get_nodes(
            where={"entity_type": "workflow_step_exec", "workflow_id": dream_workflow_id}
        )

        assert final_state["op_log"] == list(dream_workflow_expected_ops())
        assert final_state["dream_proposals"][0]["status"] == "pending"
        assert final_state["dream_candidate_workflow_node_ids"]
        assert final_state["dream_workflow_lineage_edge_ids"]
        assert reasoning_nodes
        assert proposal_nodes
        assert leaked_proposal_nodes == []
        assert leaked_reasoning_nodes == []
        assert any(node.metadata.get("status") == "pending" for node in proposal_nodes)
        assert candidate_nodes
        assert [node.metadata.get("op") for node in sorted(step_exec_nodes, key=lambda n: int(n.metadata.get("step_seq", -1)))] == list(
            dream_workflow_expected_ops()
        )
    finally:
        _close_engines(
            source_engine,
            wisdom_engine,
            conversation_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_agent_workflow_second_run_approves_pending_candidate_after_eval_rounds():
    source_engine, source_dir = _make_engine("workflow")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    conversation_engine, conversation_dir = _make_engine("conversation")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        dream_workflow_id = "dream.runtime.approve"
        target_workflow_id = "wf-target-approve"
        source_namespace = "ws:demo:source"
        wisdom_namespace = "ws:demo:wisdom"
        dream_conversation_namespace = "ws:demo:dream-conversation"
        materialized_workflow_namespace = "ws:demo:workflow-materialized"
        materialize_dream_workflow_design(
            workflow_engine,
            build_dream_maintenance_workflow_design(workflow_id=dream_workflow_id),
        )
        runtime = _runtime(workflow_engine=workflow_engine, conversation_engine=conversation_engine)

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="origin-fail",
                    workflow_id=target_workflow_id,
                    run_id="run-origin-fail",
                    step_op="retrieve",
                    completed_at_ms=1000,
                    failure_count=3,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=1200,
                    duration_ms_max=600,
                )
            )

        first = runtime.run(
            workflow_id=dream_workflow_id,
            conversation_id="conv-dream-2",
            turn_node_id="turn-dream-2",
            run_id="run-dream-2a",
            initial_state={
                "dream_budget_remaining": 5,
                "_deps": {
                    "source_engine": source_engine,
                    "wisdom_engine": wisdom_engine,
                    "conversation_engine": conversation_engine,
                    "workflow_engine": workflow_engine,
                    "source_namespace": source_namespace,
                    "source_where": {"workspace_id": "demo"},
                    "wisdom_namespace": wisdom_namespace,
                    "dream_conversation_namespace": dream_conversation_namespace,
                    "materialized_workflow_namespace": materialized_workflow_namespace,
                    "target_workflow_id": target_workflow_id,
                    "dream_created_at_ms": 22345,
                    "dream_policy": DefaultDreamLoopPolicy(
                        recent_limit=1,
                        hotspot_limit=0,
                        stale_sample_limit=0,
                        max_proposals_per_tick=1,
                        min_evaluation_runs=2,
                    ),
                    "dream_summary_builder": lambda signal: "Add bounded retry after repeated retrieve failures",
                    "dream_reasoning_builder": lambda signal: (
                        "seeded stats show repeated retrieve failures",
                        "proposal should wait for candidate evidence",
                    ),
                    "dream_suggested_change_builder": lambda signal: {
                        "dream_feature": "workflow_revision",
                        "step_op": signal.step_op,
                        "action": "add_bounded_retry",
                    },
                    "dream_confidence_builder": lambda signal: 0.95,
                    "dream_workflow_builder": _build_candidate_design,
                },
            },
        )
        pending_payload = (first.final_state or {})["dream_proposals"]
        candidate_workflow_id = pending_payload[0]["candidate_workflow_id"]

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-success-1",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-run-1",
                    step_op="retrieve",
                    completed_at_ms=2000,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=300,
                    duration_ms_max=150,
                )
            )
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-success-2",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-run-2",
                    step_op="retrieve",
                    completed_at_ms=2100,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=280,
                    duration_ms_max=140,
                )
            )
            source_engine.write.add_node(
                _feedback_node(
                    node_id="candidate-feedback",
                    workflow_id=candidate_workflow_id,
                    candidate_workflow_id=candidate_workflow_id,
                    run_id="feedback-run-1",
                    feedback_text="user praised faster retrieval and clearer answer",
                    feedback_score=1.0,
                )
            )

        second = runtime.run(
            workflow_id=dream_workflow_id,
            conversation_id="conv-dream-2",
            turn_node_id="turn-dream-2",
            run_id="run-dream-2b",
            initial_state={
                "dream_budget_remaining": 0,
                "dream_pending_proposals": pending_payload,
                "_deps": {
                    "source_engine": source_engine,
                    "wisdom_engine": wisdom_engine,
                    "conversation_engine": conversation_engine,
                    "workflow_engine": workflow_engine,
                    "source_namespace": source_namespace,
                    "source_where": {"workspace_id": "demo"},
                    "wisdom_namespace": wisdom_namespace,
                    "dream_conversation_namespace": dream_conversation_namespace,
                    "materialized_workflow_namespace": materialized_workflow_namespace,
                    "target_workflow_id": target_workflow_id,
                    "dream_created_at_ms": 22346,
                    "dream_policy": DefaultDreamLoopPolicy(
                        recent_limit=0,
                        hotspot_limit=0,
                        stale_sample_limit=0,
                        max_proposals_per_tick=0,
                        min_evaluation_runs=2,
                    ),
                    "dream_workflow_builder": _build_candidate_design,
                },
            },
        )

        final_state = second.final_state or {}
        with scoped_namespace(wisdom_engine, wisdom_namespace):
            approved_nodes = wisdom_engine.read.get_nodes(
                where={
                    "artifact_kind": "workflow_design_artifact",
                    "approved_workflow_id": candidate_workflow_id,
                    "workspace_id": "demo",
                }
            )
        assert final_state["op_log"] == list(dream_workflow_expected_ops())
        assert final_state["dream_proposals"][0]["status"] == "approved"
        assert final_state["dream_evidence"][0]["run_count"] == 2
        assert final_state["dream_approved_workflow_node_ids"] == []
        assert final_state["dream_workflow_lineage_edge_ids"]
        assert approved_nodes
    finally:
        _close_engines(
            source_engine,
            wisdom_engine,
            conversation_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)
