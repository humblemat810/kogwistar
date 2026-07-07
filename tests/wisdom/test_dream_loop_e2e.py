from __future__ import annotations

import contextlib
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.policy import DefaultDreamLoopPolicy
from kogwistar.runtime.models import WorkflowDesignArtifact, WorkflowEdge, WorkflowNode
from kogwistar.wisdom.dream_loop import DreamLoopDecision, run_dream_loop_cycle
from tests._helpers.fake_backend import build_fake_backend


pytestmark = [pytest.mark.core, pytest.mark.runtime, pytest.mark.e2e]


def _make_engine(graph_type: str) -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / f".tmp_dream_loop_{graph_type}" / str(uuid.uuid4())
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
    feedback_trust_score: float = 1.0,
    feedback_sabotage: bool = False,
    candidate_workflow_id: str | None = None,
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
            "feedback_trust_score": feedback_trust_score,
            "feedback_sabotage": feedback_sabotage,
        },
    )


def _build_approved_design(proposal, evaluation) -> WorkflowDesignArtifact:
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
                "wf_version": "v2",
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
                "wf_version": "v2",
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
                "wf_version": "v2",
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
        workflow_version="v2",
        start_node_id=start_id,
        nodes=nodes,
        edges=edges,
        source_run_id=proposal.run_id,
        source_workflow_id=proposal.workflow_id,
        source_step_id=proposal.step_op,
        notes=evaluation.rationale,
    )


def test_dream_loop_happy_path_persists_proposal_and_approved_workflow():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        source_namespace = "ws:demo:workflow"
        target_namespace = "ws:demo:wisdom"
        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="step-recent",
                    workflow_id="wf-main",
                    run_id="run-recent",
                    step_op="review",
                    completed_at_ms=5000,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=12,
                    duration_ms_max=12,
                )
            )
            source_engine.write.add_node(
                _signal_node(
                    node_id="step-hot",
                    workflow_id="wf-main",
                    run_id="run-hot",
                    step_op="distill",
                    completed_at_ms=4000,
                    failure_count=4,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=120,
                    duration_ms_max=80,
                )
            )

        result = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=target_namespace,
            source_where={"entity_type": "workflow_step_exec", "workspace_id": "demo"},
            workflow_id="wf-main",
            created_at_ms=12345,
            conversation_engine=conversation_engine,
            conversation_namespace="ws:demo:conversation",
            workflow_engine=workflow_engine,
            workflow_namespace="ws:demo:workflow",
            budget_remaining=10,
            summary_builder=lambda signal: f"Improve {signal.step_op}",
            reasoning_builder=lambda signal: (
                f"run {signal.run_id} shows score={signal.score:.2f}",
                "propose smaller retry window",
            ),
            suggested_change_builder=lambda signal: {
                "step_op": signal.step_op,
                "change": "reduce_retry" if signal.failure_count else "keep",
            },
            approval_decider=lambda proposal, evidence: (
                True,
                "fake reviewer approves",
                f"{proposal.workflow_id}.rev.{proposal.step_op}",
            ),
            approved_workflow_builder=_build_approved_design,
        )

        with scoped_namespace(wisdom_engine, target_namespace):
            proposal_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "workflow_revision_proposal", "workspace_id": "demo"}
            )
            outcome_nodes = wisdom_engine.read.get_nodes(
                where={"edge_kind": "proposal_evaluation", "workspace_id": "demo"}
            )
            proposal_edges = wisdom_engine.read.get_edges(
                where={"relation": "proposal_evaluation", "workspace_id": "demo"}
            )

        with scoped_namespace(conversation_engine, "ws:demo:conversation"):
            reasoning_nodes = conversation_engine.read.get_nodes(
                where={"artifact_kind": "dream_reasoning_trace", "workspace_id": "demo"}
            )

        with scoped_namespace(workflow_engine, "ws:demo:workflow"):
            approved_nodes = workflow_engine.read.get_nodes(
                where={"workflow_id": "wf-main.rev.review", "entity_type": "workflow_node"}
            )

        assert result.proposals
        assert result.evaluations[0].decision == "approved"
        assert proposal_nodes
        assert outcome_nodes
        assert len(outcome_nodes) == len(result.evaluations)
        assert result.proposal_node_ids
        assert result.outcome_node_ids
        assert result.reasoning_node_ids
        assert reasoning_nodes
        assert len(reasoning_nodes) == len(result.proposals)
        assert all("reasoning_trace" not in node.metadata for node in proposal_nodes)
        assert {node.metadata["reasoning_trace_id"] for node in proposal_nodes} == set(
            result.reasoning_node_ids
        )
        assert result.proposal_evaluation_edge_ids
        assert len(proposal_edges) == len(result.evaluations)
        assert {edge.source_ids[0] for edge in proposal_edges} == set(result.proposal_node_ids)
        assert {edge.target_ids[0] for edge in proposal_edges} == set(result.outcome_node_ids)
        assert result.approved_workflow_node_ids
        assert approved_nodes
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_loop_sad_path_persists_rejection_lesson_only():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        source_namespace = "ws:demo:workflow"
        target_namespace = "ws:demo:wisdom"
        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="step-stale",
                    workflow_id="wf-old",
                    run_id="run-stale",
                    step_op="route",
                    completed_at_ms=100,
                    failure_count=1,
                    error_count=0,
                    success_count=0,
                    duration_ms_total=12,
                    duration_ms_max=12,
                )
            )

        result = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=target_namespace,
            source_where={"entity_type": "workflow_step_exec", "workspace_id": "demo"},
            workflow_id="wf-old",
            created_at_ms=22345,
            conversation_engine=conversation_engine,
            conversation_namespace="ws:demo:conversation",
            workflow_engine=workflow_engine,
            workflow_namespace="ws:demo:workflow",
            budget_remaining=10,
            summary_builder=lambda signal: f"Review {signal.step_op}",
            reasoning_builder=lambda signal: (
                f"run {signal.run_id} shows stale low-signal behavior",
            ),
            suggested_change_builder=lambda signal: {"step_op": signal.step_op, "change": "none"},
            approval_decider=lambda proposal, evidence: (False, "not enough evidence", None),
            approved_workflow_builder=_build_approved_design,
        )

        with scoped_namespace(wisdom_engine, target_namespace):
            proposal_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "workflow_revision_proposal", "workspace_id": "demo"}
            )
            lesson_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "wisdom_lesson", "workspace_id": "demo"}
            )
            proposal_edges = wisdom_engine.read.get_edges(
                where={"relation": "proposal_evaluation", "workspace_id": "demo"}
            )

        with scoped_namespace(conversation_engine, "ws:demo:conversation"):
            reasoning_nodes = conversation_engine.read.get_nodes(
                where={"artifact_kind": "dream_reasoning_trace", "workspace_id": "demo"}
            )

        with scoped_namespace(workflow_engine, "ws:demo:workflow"):
            approved_nodes = workflow_engine.read.get_nodes(
                where={"workflow_id": "wf-old.rev.route", "entity_type": "workflow_node"}
            )

        assert result.proposals
        assert result.evaluations[0].decision == "rejected"
        assert proposal_nodes
        assert lesson_nodes
        assert result.reasoning_node_ids
        assert reasoning_nodes
        assert "reasoning_trace" not in proposal_nodes[0].metadata
        assert proposal_nodes[0].metadata["reasoning_trace_id"] == result.reasoning_node_ids[0]
        assert result.proposal_evaluation_edge_ids
        assert len(proposal_edges) == len(result.evaluations)
        assert proposal_edges[0].source_ids == [result.proposal_node_ids[0]]
        assert proposal_edges[0].target_ids == [result.outcome_node_ids[0]]
        assert not result.approved_workflow_node_ids
        assert not approved_nodes
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_loop_completion_with_seeded_stats_and_fake_llm_payloads():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    fake_llm_payloads = {
        "run-fail": {
            "summary": "Add bounded retry around flaky retrieval",
            "reasoning": (
                "execution stats show retrieve failed 3 times",
                "token budget policy allows one targeted repair proposal",
            ),
            "change": {
                "dream_feature": "workflow_repair",
                "step_op": "retrieve",
                "action": "add_bounded_retry",
                "max_retries": 2,
                "expected_effect": "lower transient failure rate",
            },
            "confidence": 0.91,
            "approved": True,
            "rationale": "high failure signal and bounded blast radius",
        },
        "run-success": {
            "summary": "Do not revise stable summarizer yet",
            "reasoning": (
                "execution stats show summarizer succeeded",
                "dream policy should preserve useful recent workflow behavior",
            ),
            "change": {
                "dream_feature": "workflow_preservation",
                "step_op": "summarize",
                "action": "keep_current_design",
                "expected_effect": "avoid churn",
            },
            "confidence": 0.37,
            "approved": False,
            "rationale": "success path is evidence to keep, not revise",
        },
    }

    try:
        source_namespace = "ws:demo:workflow"
        wisdom_namespace = "ws:demo:wisdom"
        conversation_namespace = "ws:demo:conversation"
        workflow_namespace = "ws:demo:workflow"

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="stats-run-fail",
                    workflow_id="wf-dream",
                    run_id="run-fail",
                    step_op="retrieve",
                    completed_at_ms=9000,
                    failure_count=3,
                    error_count=2,
                    success_count=0,
                    duration_ms_total=1800,
                    duration_ms_max=900,
                )
            )
            source_engine.write.add_node(
                _signal_node(
                    node_id="stats-run-success",
                    workflow_id="wf-dream",
                    run_id="run-success",
                    step_op="summarize",
                    completed_at_ms=8000,
                    failure_count=0,
                    error_count=0,
                    success_count=5,
                    duration_ms_total=210,
                    duration_ms_max=70,
                )
            )

        result = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"entity_type": "workflow_step_exec", "workspace_id": "demo"},
            workflow_id="wf-dream",
            created_at_ms=33333,
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=2,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=2,
                min_budget_remaining=1,
            ),
            budget_remaining=8,
            summary_builder=lambda signal: fake_llm_payloads[signal.run_id]["summary"],
            reasoning_builder=lambda signal: fake_llm_payloads[signal.run_id]["reasoning"],
            suggested_change_builder=lambda signal: fake_llm_payloads[signal.run_id]["change"],
            confidence_builder=lambda signal: fake_llm_payloads[signal.run_id]["confidence"],
            approval_decider=lambda proposal, evidence: (
                bool(fake_llm_payloads[proposal.run_id]["approved"]),
                str(fake_llm_payloads[proposal.run_id]["rationale"]),
                f"{proposal.workflow_id}.rev.{proposal.step_op}"
                if fake_llm_payloads[proposal.run_id]["approved"]
                else None,
            ),
            approved_workflow_builder=_build_approved_design,
        )

        with scoped_namespace(conversation_engine, conversation_namespace):
            reasoning_nodes = conversation_engine.read.get_nodes(
                where={"artifact_kind": "dream_reasoning_trace", "workspace_id": "demo"}
            )

        with scoped_namespace(wisdom_engine, wisdom_namespace):
            proposal_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "workflow_revision_proposal", "workspace_id": "demo"}
            )
            lesson_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "wisdom_lesson", "workspace_id": "demo"}
            )
            outcome_nodes = wisdom_engine.read.get_nodes(
                where={"edge_kind": "proposal_evaluation", "workspace_id": "demo"}
            )
            proposal_edges = wisdom_engine.read.get_edges(
                where={"relation": "proposal_evaluation", "workspace_id": "demo"}
            )

        with scoped_namespace(workflow_engine, workflow_namespace):
            approved_nodes = workflow_engine.read.get_nodes(
                where={"workflow_id": "wf-dream.rev.retrieve", "entity_type": "workflow_node"}
            )
            rejected_nodes = workflow_engine.read.get_nodes(
                where={"workflow_id": "wf-dream.rev.summarize", "entity_type": "workflow_node"}
            )

        decisions = {evaluation.proposal_id: evaluation.decision for evaluation in result.evaluations}
        proposal_by_run = {proposal.run_id: proposal for proposal in result.proposals}
        approved_proposal = proposal_by_run["run-fail"]
        rejected_proposal = proposal_by_run["run-success"]

        assert [signal.run_id for signal in result.selection.selected] == [
            "run-fail",
            "run-success",
        ]
        assert len(result.proposals) == 2
        assert decisions[approved_proposal.proposal_id] == "approved"
        assert decisions[rejected_proposal.proposal_id] == "rejected"
        assert approved_proposal.suggested_change["dream_feature"] == "workflow_repair"
        assert rejected_proposal.suggested_change["dream_feature"] == "workflow_preservation"
        assert len(reasoning_nodes) == 2
        assert len(proposal_nodes) == 2
        assert all("reasoning_trace" not in node.metadata for node in proposal_nodes)
        assert len(outcome_nodes) == 2
        assert len(proposal_edges) == 2
        assert len(lesson_nodes) == 1
        assert result.approved_workflow_node_ids
        assert approved_nodes
        assert not rejected_nodes
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_loop_multicycle_candidate_pending_then_approved_with_seeded_stats_and_feedback():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        source_namespace = "ws:demo:workflow"
        wisdom_namespace = "ws:demo:wisdom"
        conversation_namespace = "ws:demo:conversation"
        workflow_namespace = "ws:demo:workflow"

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="origin-fail",
                    workflow_id="wf-eval",
                    run_id="run-origin-fail",
                    step_op="retrieve",
                    completed_at_ms=1000,
                    failure_count=3,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=1500,
                    duration_ms_max=700,
                )
            )

        cycle1 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-eval",
            created_at_ms=50000,
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=1,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=1,
                min_evaluation_runs=2,
            ),
            budget_remaining=5,
            summary_builder=lambda signal: "Add bounded retry after repeated retrieve failures",
            reasoning_builder=lambda signal: (
                "seeded stats show repeated retrieve failures",
                "proposal should stay pending until candidate has enough runs",
            ),
            suggested_change_builder=lambda signal: {
                "dream_feature": "workflow_revision",
                "step_op": signal.step_op,
                "action": "add_bounded_retry",
                "max_retries": 2,
            },
            confidence_builder=lambda signal: 0.92,
            approved_workflow_builder=_build_approved_design,
        )

        proposal = cycle1.proposals[0]
        candidate_workflow_id = proposal.candidate_workflow_id
        assert proposal.status == "pending"
        assert candidate_workflow_id == "wf-eval.rev.retrieve"
        assert cycle1.candidate_workflow_node_ids
        assert not cycle1.approved_workflow_node_ids
        assert cycle1.evaluations[0].decision == "pending"

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
                    duration_ms_total=400,
                    duration_ms_max=220,
                )
            )
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-success-2",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-run-2",
                    step_op="retrieve",
                    completed_at_ms=3000,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=380,
                    duration_ms_max=210,
                )
            )
            source_engine.write.add_node(
                _feedback_node(
                    node_id="candidate-feedback-1",
                    workflow_id=candidate_workflow_id,
                    candidate_workflow_id=candidate_workflow_id,
                    run_id="feedback-run-1",
                    feedback_text="user praised faster retrieval and clearer answer",
                    feedback_score=1.0,
                )
            )

        cycle2 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-eval",
            created_at_ms=51000,
            pending_proposals=[proposal],
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=0,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=0,
                min_evaluation_runs=2,
            ),
            budget_remaining=5,
            approved_workflow_builder=_build_approved_design,
        )

        with scoped_namespace(wisdom_engine, wisdom_namespace):
            approved_nodes = wisdom_engine.read.get_nodes(
                where={
                    "artifact_kind": "workflow_design_artifact",
                    "workspace_id": "demo",
                    "approved_workflow_id": candidate_workflow_id,
                }
            )
            proposal_nodes = wisdom_engine.read.get_nodes(
                where={"artifact_kind": "workflow_revision_proposal", "workspace_id": "demo"}
            )
        assert cycle2.proposals[0].status == "approved"
        assert cycle2.evaluations[0].decision == "approved"
        assert cycle2.evidence[0].run_count == 2
        assert cycle2.evidence[0].positive_feedback == (
            "user praised faster retrieval and clearer answer",
        )
        assert cycle2.approved_workflow_node_ids == ()
        assert cycle2.workflow_lineage_edge_ids
        assert approved_nodes
        assert any(node.metadata.get("status") == "approved" for node in proposal_nodes)
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_loop_defers_candidate_decision_until_min_evaluation_runs_are_met():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        source_namespace = "ws:demo:workflow"
        wisdom_namespace = "ws:demo:wisdom"
        conversation_namespace = "ws:demo:conversation"
        workflow_namespace = "ws:demo:workflow"

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="origin-fail-delay",
                    workflow_id="wf-delay",
                    run_id="run-origin-delay",
                    step_op="retrieve",
                    completed_at_ms=1000,
                    failure_count=3,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=1500,
                    duration_ms_max=700,
                )
            )

        cycle1 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-delay",
            created_at_ms=70000,
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=1,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=1,
                min_evaluation_runs=3,
            ),
            budget_remaining=5,
            summary_builder=lambda signal: "Add bounded retry after repeated retrieve failures",
            reasoning_builder=lambda signal: (
                "seeded stats show repeated retrieve failures",
                "candidate should stay pending until three runs exist",
            ),
            suggested_change_builder=lambda signal: {
                "dream_feature": "workflow_revision",
                "step_op": signal.step_op,
                "action": "add_bounded_retry",
                "max_retries": 2,
            },
            confidence_builder=lambda signal: 0.92,
            approved_workflow_builder=_build_approved_design,
        )

        proposal = cycle1.proposals[0]
        candidate_workflow_id = proposal.candidate_workflow_id
        assert proposal.status == "pending"
        assert cycle1.evaluations[0].decision == "pending"
        assert cycle1.evidence[0].run_count == 1

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-success-delay-1",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-delay-run-1",
                    step_op="retrieve",
                    completed_at_ms=2000,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=320,
                    duration_ms_max=180,
                )
            )
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-success-delay-2",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-delay-run-2",
                    step_op="retrieve",
                    completed_at_ms=2100,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=300,
                    duration_ms_max=170,
                )
            )
            source_engine.write.add_node(
                _feedback_node(
                    node_id="candidate-feedback-delay-1",
                    workflow_id=candidate_workflow_id,
                    candidate_workflow_id=candidate_workflow_id,
                    run_id="feedback-delay-run-1",
                    feedback_text="user praised faster retrieval and clearer answer",
                    feedback_score=1.0,
                )
            )

        cycle2 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-delay",
            created_at_ms=71000,
            pending_proposals=[proposal],
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=0,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=0,
                min_evaluation_runs=3,
            ),
            budget_remaining=0,
            approved_workflow_builder=_build_approved_design,
        )

        assert cycle2.proposals[0].status == "pending"
        assert cycle2.evaluations[0].decision == "pending"
        assert cycle2.evidence[0].run_count == 2

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-success-delay-3",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-delay-run-3",
                    step_op="retrieve",
                    completed_at_ms=2200,
                    failure_count=0,
                    error_count=0,
                    success_count=1,
                    duration_ms_total=290,
                    duration_ms_max=160,
                )
            )
            source_engine.write.add_node(
                _feedback_node(
                    node_id="candidate-feedback-delay-2",
                    workflow_id=candidate_workflow_id,
                    candidate_workflow_id=candidate_workflow_id,
                    run_id="feedback-delay-run-2",
                    feedback_text="user praised a third successful run",
                    feedback_score=1.0,
                )
            )

        cycle3 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-delay",
            created_at_ms=72000,
            pending_proposals=[cycle2.proposals[0]],
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=0,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=0,
                min_evaluation_runs=3,
            ),
            budget_remaining=0,
            approved_workflow_builder=_build_approved_design,
        )

        assert cycle3.proposals[0].status == "approved"
        assert cycle3.evaluations[0].decision == "approved"
        assert cycle3.evidence[0].run_count == 3
        assert cycle3.workflow_lineage_edge_ids
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_loop_rejects_malformed_synthesized_workflow_before_materialization():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        source_namespace = "ws:demo:workflow"
        wisdom_namespace = "ws:demo:wisdom"
        conversation_namespace = "ws:demo:conversation"
        workflow_namespace = "ws:demo:workflow"

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="origin-bad-design",
                    workflow_id="wf-bad-design",
                    run_id="run-origin-bad-design",
                    step_op="retrieve",
                    completed_at_ms=1000,
                    failure_count=3,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=1500,
                    duration_ms_max=700,
                )
            )

        def _bad_design(proposal, evaluation):
            workflow_id = f"{proposal.workflow_id}.rev.{proposal.step_op}"
            start_id = f"wf:{workflow_id}:start"
            end_id = f"wf:{workflow_id}:end"
            start_node = WorkflowNode(
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
                    "wf_start": False,
                    "wf_terminal": False,
                    "wf_version": "v1",
                },
                mentions=_mentions(workflow_id),
                level_from_root=0,
                domain_id=None,
                canonical_entity_id=None,
                embedding=None,
            )
            end_node = WorkflowNode(
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
            )
            edge = WorkflowEdge(
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
            return SimpleNamespace(
                workflow_id=workflow_id,
                workflow_version="v1",
                start_node_id=start_id,
                nodes=[start_node, end_node],
                edges=[edge],
            )

        with pytest.raises(ValueError, match="exactly one start node"):
            run_dream_loop_cycle(
                source_engine=source_engine,
                source_namespace=source_namespace,
                target_engine=wisdom_engine,
                target_namespace=wisdom_namespace,
                source_where={"workspace_id": "demo"},
                workflow_id="wf-bad-design",
                created_at_ms=80000,
                conversation_engine=conversation_engine,
                conversation_namespace=conversation_namespace,
                workflow_engine=workflow_engine,
                workflow_namespace=workflow_namespace,
                policy=DefaultDreamLoopPolicy(
                    recent_limit=1,
                    hotspot_limit=0,
                    stale_sample_limit=0,
                    max_proposals_per_tick=1,
                    min_evaluation_runs=1,
                ),
                budget_remaining=5,
                summary_builder=lambda signal: "Bad design should be rejected",
                reasoning_builder=lambda signal: ("bad design",),
                suggested_change_builder=lambda signal: {
                    "dream_feature": "workflow_revision",
                    "step_op": signal.step_op,
                    "action": "bad_design",
                },
                confidence_builder=lambda signal: 0.5,
                approval_decider=lambda proposal, evidence: DreamLoopDecision(
                    decision="pending",
                    rationale="force materialization path for validation",
                    candidate_workflow_id=f"{proposal.workflow_id}.rev.{proposal.step_op}",
                ),
                approved_workflow_builder=_bad_design,
            )
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)


def test_dream_loop_multicycle_candidate_deprecated_when_stats_bad_and_feedback_is_sabotage():
    source_engine, source_dir = _make_engine("workflow")
    conversation_engine, conversation_dir = _make_engine("conversation")
    wisdom_engine, wisdom_dir = _make_engine("wisdom")
    workflow_engine, workflow_dir = _make_engine("workflow")

    try:
        source_namespace = "ws:demo:workflow"
        wisdom_namespace = "ws:demo:wisdom"
        conversation_namespace = "ws:demo:conversation"
        workflow_namespace = "ws:demo:workflow"

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="origin-fail-2",
                    workflow_id="wf-risky",
                    run_id="run-origin-risky",
                    step_op="plan",
                    completed_at_ms=1000,
                    failure_count=2,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=900,
                    duration_ms_max=500,
                )
            )

        cycle1 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-risky",
            created_at_ms=60000,
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=1,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=1,
                min_evaluation_runs=2,
                deprecation_score_threshold=-1.0,
                sabotage_feedback_penalty=3.0,
            ),
            budget_remaining=5,
            summary_builder=lambda signal: "Try aggressive shortcut planning revision",
            reasoning_builder=lambda signal: (
                "seeded failure pattern triggered proposal",
                "candidate must still survive later evaluation",
            ),
            suggested_change_builder=lambda signal: {
                "dream_feature": "workflow_revision",
                "step_op": signal.step_op,
                "action": "aggressive_shortcut",
            },
            approved_workflow_builder=_build_approved_design,
        )

        proposal = cycle1.proposals[0]
        candidate_workflow_id = proposal.candidate_workflow_id
        assert proposal.status == "pending"

        with scoped_namespace(source_engine, source_namespace):
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-bad-1",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-bad-run-1",
                    step_op="plan",
                    completed_at_ms=2000,
                    failure_count=1,
                    error_count=1,
                    success_count=0,
                    duration_ms_total=1000,
                    duration_ms_max=700,
                )
            )
            source_engine.write.add_node(
                _signal_node(
                    node_id="candidate-bad-2",
                    workflow_id=candidate_workflow_id,
                    run_id="candidate-bad-run-2",
                    step_op="plan",
                    completed_at_ms=2500,
                    failure_count=1,
                    error_count=0,
                    success_count=0,
                    duration_ms_total=950,
                    duration_ms_max=650,
                )
            )
            source_engine.write.add_node(
                _feedback_node(
                    node_id="candidate-sabotage-1",
                    workflow_id=candidate_workflow_id,
                    candidate_workflow_id=candidate_workflow_id,
                    run_id="feedback-sabotage-1",
                    feedback_text="user insisted unsafe shortcut is genius",
                    feedback_score=1.0,
                    feedback_sabotage=True,
                )
            )

        cycle2 = run_dream_loop_cycle(
            source_engine=source_engine,
            source_namespace=source_namespace,
            target_engine=wisdom_engine,
            target_namespace=wisdom_namespace,
            source_where={"workspace_id": "demo"},
            workflow_id="wf-risky",
            created_at_ms=61000,
            pending_proposals=[proposal],
            conversation_engine=conversation_engine,
            conversation_namespace=conversation_namespace,
            workflow_engine=workflow_engine,
            workflow_namespace=workflow_namespace,
            policy=DefaultDreamLoopPolicy(
                recent_limit=0,
                hotspot_limit=0,
                stale_sample_limit=0,
                max_proposals_per_tick=0,
                min_evaluation_runs=2,
                deprecation_score_threshold=-1.0,
                sabotage_feedback_penalty=3.0,
            ),
            budget_remaining=5,
            approved_workflow_builder=_build_approved_design,
        )

        with scoped_namespace(wisdom_engine, wisdom_namespace):
            deprecated_nodes = wisdom_engine.read.get_nodes(
                where={
                    "artifact_kind": "workflow_candidate_artifact",
                    "workspace_id": "demo",
                    "deprecated": True,
                }
            )
        assert cycle2.proposals[0].status == "deprecated"
        assert cycle2.evaluations[0].decision == "deprecated"
        assert cycle2.evidence[0].run_count == 2
        assert cycle2.evidence[0].sabotage_feedback == (
            "user insisted unsafe shortcut is genius",
        )
        assert deprecated_nodes
        assert cycle2.workflow_lineage_edge_ids
    finally:
        _close_engines(
            source_engine,
            conversation_engine,
            wisdom_engine,
            workflow_engine,
        )
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(conversation_dir, ignore_errors=True)
        shutil.rmtree(wisdom_dir, ignore_errors=True)
        shutil.rmtree(workflow_dir, ignore_errors=True)
