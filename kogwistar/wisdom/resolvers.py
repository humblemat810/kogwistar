from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.wisdom.dream_loop import (
    _evaluation_edge_from_evaluation,
    _evaluation_node_from_evaluation,
    _normalize_dream_loop_decision,
    _persist_edges,
    _persist_nodes,
    _proposal_node_from_proposal,
    _reasoning_node_from_proposal,
    _workflow_artifact_node,
    _workflow_lineage_edge,
    DreamLoopEvidence,
    DreamLoopSignal,
    ProposalEvaluation,
    WisdomRevisionProposal,
    build_wisdom_revision_proposals_for_signals,
    collect_dream_loop_evidence,
    evaluate_wisdom_revision_proposal,
    select_dream_loop_signals,
)
from kogwistar.policy import DefaultDreamLoopPolicy

dream_default_resolver = MappingStepResolver()


def _deps(ctx) -> dict[str, Any]:
    deps = ctx.state_view.get("_deps")
    if not isinstance(deps, dict):
        deps = ctx.state_view.get("dream_deps")
    if not isinstance(deps, dict):
        raise RuntimeError("dream workflow requires state['_deps'] (or legacy state['dream_deps'])")
    return deps


def _policy(ctx) -> DefaultDreamLoopPolicy:
    deps = _deps(ctx)
    value = ctx.state_view.get("dream_policy") or deps.get("dream_policy")
    if isinstance(value, DefaultDreamLoopPolicy):
        return value
    if isinstance(value, dict):
        return DefaultDreamLoopPolicy(**value)
    return DefaultDreamLoopPolicy()


def _read_source_nodes(ctx) -> list[Any]:
    deps = _deps(ctx)
    source_engine = deps["source_engine"]
    source_namespace = str(
        ctx.state_view.get("source_namespace") or deps.get("source_namespace") or ""
    )
    source_where = dict(ctx.state_view.get("source_where") or deps.get("source_where") or {})
    with scoped_namespace(source_engine, source_namespace):
        return list(source_engine.read.get_nodes(where=source_where))


def _conversation_namespace(ctx) -> str:
    deps = _deps(ctx)
    return str(
        ctx.state_view.get("dream_conversation_namespace")
        or deps.get("dream_conversation_namespace")
        or deps.get("conversation_namespace")
        or "dream:conversation"
    )


def _wisdom_namespace(ctx) -> str:
    deps = _deps(ctx)
    return str(ctx.state_view.get("wisdom_namespace") or deps.get("wisdom_namespace") or "dream:wisdom")


def _workflow_namespace(ctx) -> str:
    deps = _deps(ctx)
    return str(
        ctx.state_view.get("materialized_workflow_namespace")
        or deps.get("materialized_workflow_namespace")
        or deps.get("workflow_namespace")
        or "dream:workflow"
    )


def _target_workflow_id(ctx) -> str:
    deps = _deps(ctx)
    return str(ctx.state_view.get("target_workflow_id") or deps.get("target_workflow_id") or ctx.workflow_id)


def _created_at_ms(ctx) -> int:
    deps = _deps(ctx)
    value = ctx.state_view.get("dream_created_at_ms") or deps.get("dream_created_at_ms")
    return int(value or 0)


def _proposal_payloads_to_models(payloads: list[Any]) -> list[WisdomRevisionProposal]:
    return [WisdomRevisionProposal(**dict(item)) for item in payloads]


@dream_default_resolver.register("dream_start")
def _dream_start(ctx):
    return RunSuccess(
        conversation_node_id=None,
        state_update=[("a", {"op_log": "dream_start"})],
    )


@dream_default_resolver.register("dream_select_signals")
def _dream_select_signals(ctx):
    policy = _policy(ctx)
    source_nodes = _read_source_nodes(ctx)
    budget_remaining = int(ctx.state_view.get("dream_budget_remaining") or 0)
    selection = select_dream_loop_signals(
        source_nodes,
        policy=policy,
        budget_remaining=budget_remaining,
    )
    selected_payloads = [asdict(signal) for signal in selection.selected]
    return RunSuccess(
        conversation_node_id=None,
        state_update=[
            (
                "u",
                {
                    "dream_selected_signals": selected_payloads,
                    "dream_selection_counts": {
                        "recent": len(selection.recent),
                        "hotspots": len(selection.hotspots),
                        "stale": len(selection.stale),
                        "selected": len(selection.selected),
                    },
                },
            ),
            ("a", {"op_log": "dream_select_signals"}),
        ],
    )


@dream_default_resolver.register("dream_build_proposals")
def _dream_build_proposals(ctx):
    deps = _deps(ctx)
    selected_payloads = list(ctx.state_view.get("dream_selected_signals") or [])
    pending_payloads = list(ctx.state_view.get("dream_pending_proposals") or [])
    selected_signals = [DreamLoopSignal(**dict(item)) for item in selected_payloads]
    proposals = list(_proposal_payloads_to_models(pending_payloads))
    proposals.extend(
        build_wisdom_revision_proposals_for_signals(
            selected_signals,
            workflow_id=_target_workflow_id(ctx),
            created_at_ms=_created_at_ms(ctx),
            summary_builder=deps.get("dream_summary_builder"),
            reasoning_builder=deps.get("dream_reasoning_builder"),
            suggested_change_builder=deps.get("dream_suggested_change_builder"),
            confidence_builder=deps.get("dream_confidence_builder"),
        )
    )
    return RunSuccess(
        conversation_node_id=None,
        state_update=[
            ("u", {"dream_proposals": [asdict(item) for item in proposals]}),
            ("a", {"op_log": "dream_build_proposals"}),
        ],
    )


@dream_default_resolver.register("dream_persist_reasoning")
def _dream_persist_reasoning(ctx):
    deps = _deps(ctx)
    conversation_engine = deps["conversation_engine"]
    proposals = _proposal_payloads_to_models(list(ctx.state_view.get("dream_proposals") or []))
    reasoning_nodes = [_reasoning_node_from_proposal(item) for item in proposals]
    reasoning_node_ids = tuple(
        _persist_nodes(conversation_engine, _conversation_namespace(ctx), reasoning_nodes)
    )
    return RunSuccess(
        conversation_node_id=(reasoning_node_ids[0] if reasoning_node_ids else None),
        state_update=[
            ("u", {"dream_reasoning_node_ids": list(reasoning_node_ids)}),
            ("a", {"op_log": "dream_persist_reasoning"}),
        ],
    )


@dream_default_resolver.register("dream_persist_proposals")
def _dream_persist_proposals(ctx):
    deps = _deps(ctx)
    wisdom_engine = deps["wisdom_engine"]
    proposals = _proposal_payloads_to_models(list(ctx.state_view.get("dream_proposals") or []))
    reasoning_node_ids = list(ctx.state_view.get("dream_reasoning_node_ids") or [])
    proposal_nodes = [
        _proposal_node_from_proposal(
            item,
            reasoning_node_id=(reasoning_node_ids[index] if index < len(reasoning_node_ids) else None),
        )
        for index, item in enumerate(proposals)
    ]
    proposal_node_ids = tuple(_persist_nodes(wisdom_engine, _wisdom_namespace(ctx), proposal_nodes))
    return RunSuccess(
        conversation_node_id=None,
        state_update=[
            ("u", {"dream_proposal_node_ids": list(proposal_node_ids)}),
            ("a", {"op_log": "dream_persist_proposals"}),
        ],
    )


@dream_default_resolver.register("dream_evaluate_proposals")
def _dream_evaluate_proposals(ctx):
    deps = _deps(ctx)
    policy = _policy(ctx)
    source_nodes = _read_source_nodes(ctx)
    proposals = _proposal_payloads_to_models(list(ctx.state_view.get("dream_proposals") or []))
    reasoning_node_ids = list(ctx.state_view.get("dream_reasoning_node_ids") or [])
    wisdom_engine = deps["wisdom_engine"]
    workflow_engine = deps["workflow_engine"]
    approval_decider = deps.get("dream_approval_decider")
    approved_workflow_builder = deps.get("dream_workflow_builder")

    evaluations: list[ProposalEvaluation] = []
    evidence_items: list[DreamLoopEvidence] = []
    finalized_proposals: list[WisdomRevisionProposal] = []
    outcome_nodes = []
    proposal_edges = []
    candidate_workflow_node_ids: list[str] = []
    candidate_workflow_edge_ids: list[str] = []
    approved_workflow_node_ids: list[str] = []
    approved_workflow_edge_ids: list[str] = []
    workflow_lineage_node_ids: list[str] = []
    workflow_lineage_edge_ids: list[str] = []

    for proposal_index, proposal in enumerate(proposals):
        evidence = collect_dream_loop_evidence(source_nodes, proposal=proposal, policy=policy)
        evidence_items.append(evidence)
        raw_decision = (
            approval_decider(proposal, evidence)
            if callable(approval_decider)
            else None
        )
        decision = _normalize_dream_loop_decision(
            raw_decision,
            proposal=proposal,
            evidence=evidence,
            policy=policy,
        )
        candidate_workflow_id = decision.candidate_workflow_id or proposal.candidate_workflow_id
        should_materialize = (
            decision.decision in {"pending", "approved"}
            and callable(approved_workflow_builder)
            and not proposal.candidate_workflow_id
        )
        if should_materialize:
            preview = evaluate_wisdom_revision_proposal(
                proposal,
                decision=decision.decision,
                rationale=decision.rationale,
                created_at_ms=_created_at_ms(ctx) + proposal_index + 1,
                candidate_workflow_id=candidate_workflow_id,
                lesson_summary=decision.lesson_summary,
                evidence=evidence,
            )
            design = approved_workflow_builder(proposal, preview)
            nodes = list(getattr(design, "nodes", []) or [])
            edges = list(getattr(design, "edges", []) or [])
            candidate_workflow_id = str(getattr(design, "workflow_id", None) or candidate_workflow_id or proposal.workflow_id)
            if nodes:
                persisted_node_ids = _persist_nodes(workflow_engine, _workflow_namespace(ctx), nodes)
                if decision.decision == "pending":
                    candidate_workflow_node_ids.extend(persisted_node_ids)
                else:
                    approved_workflow_node_ids.extend(persisted_node_ids)
            if edges:
                persisted_edge_ids = _persist_edges(workflow_engine, _workflow_namespace(ctx), edges)
                if decision.decision == "pending":
                    candidate_workflow_edge_ids.extend(persisted_edge_ids)
                else:
                    approved_workflow_edge_ids.extend(persisted_edge_ids)

        evaluation = evaluate_wisdom_revision_proposal(
            proposal,
            decision=decision.decision,
            rationale=decision.rationale,
            created_at_ms=_created_at_ms(ctx) + len(evaluations) + 1,
            candidate_workflow_id=candidate_workflow_id,
            lesson_summary=decision.lesson_summary,
            evidence=evidence,
        )
        evaluations.append(evaluation)
        outcome_node = _evaluation_node_from_evaluation(proposal, evaluation)
        outcome_nodes.append(outcome_node)
        proposal_edges.append(_evaluation_edge_from_evaluation(proposal, evaluation, str(outcome_node.id)))
        finalized_proposals.append(
            replace(
                proposal,
                status=decision.decision,
                candidate_workflow_id=candidate_workflow_id,
                evaluation_rounds=max(int(proposal.evaluation_rounds), int(evidence.run_count)),
                conversation_feedback=[
                    *evidence.positive_feedback,
                    *evidence.negative_feedback,
                    *evidence.sabotage_feedback,
                ],
                metadata={
                    **dict(proposal.metadata or {}),
                    "candidate_workflow_id": candidate_workflow_id,
                    "evaluation_rounds": max(int(proposal.evaluation_rounds), int(evidence.run_count)),
                    "evaluation_net_score": float(evidence.net_score),
                },
            )
        )
        if candidate_workflow_id:
            origin_node = _workflow_artifact_node(
                workflow_id=proposal.workflow_id,
                revision_status="origin",
                workspace_id=proposal.metadata.get("workspace_id", "demo"),
                created_at_ms=_created_at_ms(ctx),
            )
            candidate_node = _workflow_artifact_node(
                workflow_id=candidate_workflow_id,
                revision_status=decision.decision,
                workspace_id=proposal.metadata.get("workspace_id", "demo"),
                source_workflow_id=proposal.workflow_id,
                created_at_ms=evaluation.created_at_ms,
            )
            workflow_lineage_node_ids.extend(
                _persist_nodes(
                    workflow_engine,
                    _workflow_namespace(ctx),
                    [origin_node, candidate_node],
                )
            )
            workflow_lineage_edge_ids.extend(
                _persist_edges(
                    workflow_engine,
                    _workflow_namespace(ctx),
                    [
                        _workflow_lineage_edge(
                            proposal=proposal,
                            from_workflow_id=proposal.workflow_id,
                            to_workflow_id=candidate_workflow_id,
                            from_node_id=str(origin_node.id),
                            to_node_id=str(candidate_node.id),
                            revision_status=decision.decision,
                            created_at_ms=evaluation.created_at_ms,
                        )
                    ],
                )
            )

    updated_proposal_nodes = [
        _proposal_node_from_proposal(
            item,
            reasoning_node_id=(reasoning_node_ids[index] if index < len(reasoning_node_ids) else None),
        )
        for index, item in enumerate(finalized_proposals)
    ]
    proposal_node_ids = tuple(_persist_nodes(wisdom_engine, _wisdom_namespace(ctx), updated_proposal_nodes))
    outcome_node_ids = tuple(_persist_nodes(wisdom_engine, _wisdom_namespace(ctx), outcome_nodes))
    proposal_edge_ids = tuple(_persist_edges(wisdom_engine, _wisdom_namespace(ctx), proposal_edges))
    return RunSuccess(
        conversation_node_id=None,
        state_update=[
            (
                "u",
                {
                    "dream_proposals": [asdict(item) for item in finalized_proposals],
                    "dream_evaluations": [asdict(item) for item in evaluations],
                    "dream_evidence": [asdict(item) for item in evidence_items],
                    "dream_proposal_node_ids": list(proposal_node_ids),
                    "dream_outcome_node_ids": list(outcome_node_ids),
                    "dream_proposal_evaluation_edge_ids": list(proposal_edge_ids),
                    "dream_candidate_workflow_node_ids": list(candidate_workflow_node_ids),
                    "dream_candidate_workflow_edge_ids": list(candidate_workflow_edge_ids),
                    "dream_approved_workflow_node_ids": list(approved_workflow_node_ids),
                    "dream_approved_workflow_edge_ids": list(approved_workflow_edge_ids),
                    "dream_workflow_lineage_node_ids": list(workflow_lineage_node_ids),
                    "dream_workflow_lineage_edge_ids": list(workflow_lineage_edge_ids),
                },
            ),
            ("a", {"op_log": "dream_evaluate_proposals"}),
        ],
    )


@dream_default_resolver.register("dream_end")
def _dream_end(ctx):
    return RunSuccess(
        conversation_node_id=None,
        state_update=[
            ("u", {"dream_completed": True}),
            ("a", {"op_log": "dream_end"}),
        ],
    )
