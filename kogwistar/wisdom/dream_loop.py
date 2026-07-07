from __future__ import annotations

"""Dream-loop sampling and proposal helpers for wisdom maintenance."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from kogwistar.id_provider import stable_id
from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.policy import DefaultDreamLoopPolicy
from kogwistar.wisdom.proposals import ProposalEvaluation, WisdomRevisionProposal


@dataclass(frozen=True, slots=True)
class DreamLoopSignal:
    """One workflow-run observation eligible for dream-loop review."""

    workflow_id: str
    run_id: str
    step_op: str
    completed_at_ms: int | None = None
    failure_count: int = 0
    error_count: int = 0
    success_count: int = 0
    duration_ms_total: int = 0
    duration_ms_max: int = 0
    feedback_score: float = 0.0
    feedback_text: str = ""
    feedback_trust_score: float = 1.0
    feedback_kind: str = ""
    feedback_sabotage: bool = False
    protected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return (
            float(self.failure_count) * 5.0
            + float(self.error_count) * 3.0
            + float(self.feedback_score) * 10.0
            + min(float(self.duration_ms_total) / 1000.0, 10.0)
        )


@dataclass(frozen=True, slots=True)
class DreamLoopSelection:
    recent: tuple[DreamLoopSignal, ...]
    hotspots: tuple[DreamLoopSignal, ...]
    stale: tuple[DreamLoopSignal, ...]
    selected: tuple[DreamLoopSignal, ...]


@dataclass(frozen=True, slots=True)
class DreamLoopEvidence:
    proposal_id: str
    observed_run_ids: tuple[str, ...]
    run_count: int
    success_count: int
    failure_count: int
    error_count: int
    duration_ms_total: int
    trusted_feedback_score: float
    sabotage_feedback_score: float
    positive_feedback: tuple[str, ...]
    negative_feedback: tuple[str, ...]
    sabotage_feedback: tuple[str, ...]

    @property
    def net_score(self) -> float:
        return (
            float(self.success_count)
            - float(self.failure_count) * 2.0
            - float(self.error_count)
            + float(self.trusted_feedback_score)
            - float(self.sabotage_feedback_score)
        )


@dataclass(frozen=True, slots=True)
class DreamLoopDecision:
    decision: str
    rationale: str
    candidate_workflow_id: str | None = None
    lesson_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DreamLoopRunResult:
    selection: DreamLoopSelection
    proposals: tuple[WisdomRevisionProposal, ...]
    evaluations: tuple[ProposalEvaluation, ...]
    proposal_node_ids: tuple[str, ...]
    outcome_node_ids: tuple[str, ...]
    evidence: tuple[DreamLoopEvidence, ...] = ()
    reasoning_node_ids: tuple[str, ...] = ()
    proposal_evaluation_edge_ids: tuple[str, ...] = ()
    candidate_workflow_node_ids: tuple[str, ...] = ()
    candidate_workflow_edge_ids: tuple[str, ...] = ()
    approved_workflow_node_ids: tuple[str, ...] = ()
    approved_workflow_edge_ids: tuple[str, ...] = ()
    workflow_lineage_node_ids: tuple[str, ...] = ()
    workflow_lineage_edge_ids: tuple[str, ...] = ()


def _coerce_signal(item: Any) -> DreamLoopSignal:
    if isinstance(item, DreamLoopSignal):
        return item

    if isinstance(item, Mapping):
        meta = dict(item)
        return DreamLoopSignal(
            workflow_id=str(meta.get("workflow_id") or meta.get("workflow") or ""),
            run_id=str(meta.get("run_id") or ""),
            step_op=str(meta.get("step_op") or meta.get("op") or meta.get("wf_op") or ""),
            completed_at_ms=(
                int(meta["completed_at_ms"])
                if meta.get("completed_at_ms") is not None
                else None
            ),
            failure_count=int(meta.get("failure_count") or 0),
            error_count=int(meta.get("error_count") or 0),
            success_count=int(meta.get("success_count") or 0),
            duration_ms_total=int(meta.get("duration_ms_total") or meta.get("duration_ms") or 0),
            duration_ms_max=int(meta.get("duration_ms_max") or meta.get("duration_ms") or 0),
            feedback_score=float(meta.get("feedback_score") or 0.0),
            feedback_text=str(meta.get("feedback_text") or meta.get("user_feedback") or ""),
            feedback_trust_score=float(meta.get("feedback_trust_score") or 1.0),
            feedback_kind=str(meta.get("feedback_kind") or meta.get("entity_type") or ""),
            feedback_sabotage=bool(meta.get("feedback_sabotage") or meta.get("is_sabotage") or False),
            protected=bool(meta.get("protected") or meta.get("is_protected") or False),
            metadata=dict(meta),
        )

    metadata = dict(getattr(item, "metadata", {}) or {})
    return DreamLoopSignal(
        workflow_id=str(
            getattr(item, "workflow_id", None)
            or metadata.get("workflow_id")
            or metadata.get("workflow")
            or ""
        ),
        run_id=str(getattr(item, "run_id", None) or metadata.get("run_id") or ""),
        step_op=str(
            getattr(item, "step_op", None)
            or metadata.get("step_op")
            or metadata.get("op")
            or metadata.get("wf_op")
            or ""
        ),
        completed_at_ms=(
            int(
                getattr(item, "completed_at_ms", None)
                if getattr(item, "completed_at_ms", None) is not None
                else metadata.get("completed_at_ms")
            )
            if (
                getattr(item, "completed_at_ms", None) is not None
                or metadata.get("completed_at_ms") is not None
            )
            else None
        ),
        failure_count=int(getattr(item, "failure_count", None) or metadata.get("failure_count") or 0),
        error_count=int(getattr(item, "error_count", None) or metadata.get("error_count") or 0),
        success_count=int(getattr(item, "success_count", None) or metadata.get("success_count") or 0),
        duration_ms_total=int(
            getattr(item, "duration_ms_total", None)
            or metadata.get("duration_ms_total")
            or metadata.get("duration_ms")
            or 0
        ),
        duration_ms_max=int(
            getattr(item, "duration_ms_max", None)
            or metadata.get("duration_ms_max")
            or metadata.get("duration_ms")
            or 0
        ),
        feedback_score=float(getattr(item, "feedback_score", None) or metadata.get("feedback_score") or 0.0),
        feedback_text=str(
            getattr(item, "feedback_text", None)
            or metadata.get("feedback_text")
            or metadata.get("user_feedback")
            or ""
        ),
        feedback_trust_score=float(
            getattr(item, "feedback_trust_score", None)
            or metadata.get("feedback_trust_score")
            or 1.0
        ),
        feedback_kind=str(
            getattr(item, "feedback_kind", None)
            or metadata.get("feedback_kind")
            or metadata.get("entity_type")
            or ""
        ),
        feedback_sabotage=bool(
            getattr(item, "feedback_sabotage", None)
            or metadata.get("feedback_sabotage")
            or metadata.get("is_sabotage")
            or False
        ),
        protected=bool(getattr(item, "protected", None) or metadata.get("protected") or metadata.get("is_protected") or False),
        metadata=metadata,
    )


def _merge_signals(signals: Iterable[Any]) -> list[DreamLoopSignal]:
    merged: dict[str, DreamLoopSignal] = {}
    for item in signals:
        signal = _coerce_signal(item)
        if not signal.run_id:
            continue
        current = merged.get(signal.run_id)
        if current is None:
            merged[signal.run_id] = signal
            continue
        current_rank = (
            int(current.completed_at_ms or -1),
            float(current.score),
            int(current.duration_ms_total),
        )
        signal_rank = (
            int(signal.completed_at_ms or -1),
            float(signal.score),
            int(signal.duration_ms_total),
        )
        if signal_rank >= current_rank:
            merged[signal.run_id] = signal
    return list(merged.values())


def _stable_sample_key(signal: DreamLoopSignal, *, sample_seed: str) -> str:
    return str(stable_id("dream_loop_sample", sample_seed, signal.workflow_id, signal.run_id, signal.step_op))


def select_dream_loop_signals(
    signals: Iterable[Any],
    *,
    policy: DefaultDreamLoopPolicy | None = None,
    budget_remaining: int | None = None,
) -> DreamLoopSelection:
    """Budgeted selection of recent, hotspot, and stale audit signals."""

    policy = policy or DefaultDreamLoopPolicy()
    if budget_remaining is not None and int(budget_remaining) < int(policy.min_budget_remaining):
        return DreamLoopSelection(recent=(), hotspots=(), stale=(), selected=())

    items = _merge_signals(signals)
    if not items:
        return DreamLoopSelection(recent=(), hotspots=(), stale=(), selected=())

    by_time_desc = sorted(
        items,
        key=lambda signal: (
            int(signal.completed_at_ms or -1),
            signal.workflow_id,
            signal.run_id,
        ),
        reverse=True,
    )
    recent = tuple(by_time_desc[: max(0, int(policy.recent_limit))])
    selected_ids = {signal.run_id for signal in recent}

    hotspot_pool = [signal for signal in items if signal.run_id not in selected_ids]
    hotspots = tuple(
        sorted(
            hotspot_pool,
            key=lambda signal: (
                float(signal.score),
                int(signal.completed_at_ms or -1),
                signal.workflow_id,
                signal.run_id,
            ),
            reverse=True,
        )[: max(0, int(policy.hotspot_limit))]
    )
    selected_ids.update(signal.run_id for signal in hotspots)

    stale_pool = [
        signal
        for signal in items
        if signal.run_id not in selected_ids and not signal.protected
    ]
    stale_sorted = sorted(
        stale_pool,
        key=lambda signal: (
            int(signal.completed_at_ms or 0),
            _stable_sample_key(signal, sample_seed=policy.sample_seed),
        ),
    )
    stale = tuple(stale_sorted[: max(0, int(policy.stale_sample_limit))])

    selected: list[DreamLoopSignal] = []
    seen: set[str] = set()
    for bucket in (recent, hotspots, stale):
        for signal in bucket:
            if signal.run_id in seen:
                continue
            seen.add(signal.run_id)
            selected.append(signal)

    max_selected = int(policy.max_proposals_per_tick)
    if max_selected >= 0:
        selected = selected[:max_selected]

    return DreamLoopSelection(
        recent=recent,
        hotspots=hotspots,
        stale=stale,
        selected=tuple(selected),
    )


def collect_dream_loop_evidence(
    signals: Sequence[Any],
    *,
    proposal: WisdomRevisionProposal,
    policy: DefaultDreamLoopPolicy | None = None,
) -> DreamLoopEvidence:
    """Aggregate workflow-run and feedback evidence for one proposal."""

    policy = policy or DefaultDreamLoopPolicy()
    candidate_workflow_id = str(proposal.candidate_workflow_id or "").strip()
    matched_run_ids: set[str] = set()
    success_count = 0
    failure_count = 0
    error_count = 0
    duration_ms_total = 0
    trusted_feedback_score = 0.0
    sabotage_feedback_score = 0.0
    positive_feedback: list[str] = []
    negative_feedback: list[str] = []
    sabotage_feedback: list[str] = []

    for item in signals:
        signal = _coerce_signal(item)
        metadata = dict(signal.metadata or {})
        signal_workflow_id = str(signal.workflow_id or metadata.get("workflow_id") or "").strip()
        related_candidate_workflow_id = str(
            metadata.get("candidate_workflow_id")
            or metadata.get("evaluated_workflow_id")
            or signal_workflow_id
            or ""
        ).strip()
        is_feedback = str(signal.feedback_kind or "").strip() == "conversation_feedback"

        candidate_match = bool(candidate_workflow_id) and related_candidate_workflow_id == candidate_workflow_id
        origin_match = (
            not candidate_workflow_id
            and
            signal.step_op == proposal.step_op
            and signal.run_id in proposal.evidence_run_ids
            and signal_workflow_id == proposal.workflow_id
        )

        if not is_feedback and (candidate_match or origin_match):
            if signal.run_id:
                matched_run_ids.add(signal.run_id)
            success_count += int(signal.success_count)
            failure_count += int(signal.failure_count)
            error_count += int(signal.error_count)
            duration_ms_total += int(signal.duration_ms_total)

        if not is_feedback:
            continue
        feedback_target = str(metadata.get("candidate_workflow_id") or metadata.get("workflow_id") or "").strip()
        if candidate_workflow_id:
            if feedback_target != candidate_workflow_id:
                continue
        elif feedback_target and feedback_target != proposal.workflow_id:
            continue

        text = str(signal.feedback_text or metadata.get("feedback_text") or "").strip()
        if signal.feedback_sabotage:
            sabotage_feedback_score += max(
                0.0,
                float(signal.feedback_trust_score) * float(policy.sabotage_feedback_penalty),
            )
            if text:
                sabotage_feedback.append(text)
            continue

        weighted_feedback = float(signal.feedback_score) * float(signal.feedback_trust_score) * float(policy.trusted_feedback_weight)
        trusted_feedback_score += weighted_feedback
        if text:
            if weighted_feedback >= 0:
                positive_feedback.append(text)
            else:
                negative_feedback.append(text)

    return DreamLoopEvidence(
        proposal_id=proposal.proposal_id,
        observed_run_ids=tuple(sorted(matched_run_ids)),
        run_count=len(matched_run_ids),
        success_count=success_count,
        failure_count=failure_count,
        error_count=error_count,
        duration_ms_total=duration_ms_total,
        trusted_feedback_score=trusted_feedback_score,
        sabotage_feedback_score=sabotage_feedback_score,
        positive_feedback=tuple(positive_feedback),
        negative_feedback=tuple(negative_feedback),
        sabotage_feedback=tuple(sabotage_feedback),
    )


def default_dream_loop_decider(
    proposal: WisdomRevisionProposal,
    evidence: DreamLoopEvidence,
    *,
    policy: DefaultDreamLoopPolicy | None = None,
) -> DreamLoopDecision:
    """Conservative multi-cycle candidate decision helper."""

    policy = policy or DefaultDreamLoopPolicy()
    if evidence.run_count < int(policy.min_evaluation_runs):
        return DreamLoopDecision(
            decision="pending",
            rationale="insufficient evaluation rounds; keep candidate pending",
            candidate_workflow_id=proposal.candidate_workflow_id,
            metadata={"evidence_runs_missing": int(policy.min_evaluation_runs) - int(evidence.run_count)},
        )
    if evidence.net_score >= float(policy.approval_score_threshold):
        return DreamLoopDecision(
            decision="approved",
            rationale="candidate improved observed runs and trusted feedback",
            candidate_workflow_id=proposal.candidate_workflow_id,
        )
    if evidence.net_score <= float(policy.deprecation_score_threshold):
        return DreamLoopDecision(
            decision="deprecated",
            rationale="candidate degraded outcomes or attracted untrusted sabotage-like feedback",
            candidate_workflow_id=proposal.candidate_workflow_id,
            lesson_summary="candidate revision should be deprecated after evaluation",
        )
    return DreamLoopDecision(
        decision="rejected",
        rationale="candidate did not clear approval threshold after evaluation",
        candidate_workflow_id=proposal.candidate_workflow_id,
        lesson_summary="candidate revision did not show enough benefit",
    )


def build_wisdom_revision_proposals_for_signals(
    signals: Sequence[Any],
    *,
    workflow_id: str,
    created_at_ms: int,
    summary_builder: Callable[[DreamLoopSignal], str] | None = None,
    reasoning_builder: Callable[[DreamLoopSignal], Sequence[str]] | None = None,
    suggested_change_builder: Callable[[DreamLoopSignal], dict[str, Any]] | None = None,
    confidence_builder: Callable[[DreamLoopSignal], float] | None = None,
) -> list[WisdomRevisionProposal]:
    """Turn already-selected dream-loop signals into compact wisdom proposals."""

    selected_signals = [_coerce_signal(item) for item in signals]
    if not selected_signals:
        return []

    proposals: list[WisdomRevisionProposal] = []
    for signal in selected_signals:
        summary = (
            summary_builder(signal)
            if summary_builder is not None
            else f"Revise {signal.step_op} for workflow {signal.workflow_id}"
        )
        reasoning = list(
            reasoning_builder(signal)
            if reasoning_builder is not None
            else [
                f"run {signal.run_id} selected from dream loop",
                f"step {signal.step_op} score={signal.score:.2f}",
            ]
        )
        suggested_change = (
            suggested_change_builder(signal)
            if suggested_change_builder is not None
            else {
                "workflow_id": signal.workflow_id,
                "step_op": signal.step_op,
                "suggestion": "inspect recent failure pattern",
            }
        )
        confidence = (
            float(confidence_builder(signal))
            if confidence_builder is not None
            else min(0.99, max(0.05, signal.score / 10.0))
        )
        proposal_id = str(
            stable_id(
                "wisdom_revision_proposal",
                workflow_id,
                signal.workflow_id,
                signal.run_id,
                signal.step_op,
            )
        )
        proposals.append(
            WisdomRevisionProposal(
                proposal_id=proposal_id,
                workflow_id=workflow_id,
                run_id=signal.run_id,
                step_op=signal.step_op,
                summary=summary,
                reasoning_trace=reasoning,
                evidence_run_ids=[signal.run_id],
                confidence=float(confidence),
                created_at_ms=int(created_at_ms),
                suggested_change=suggested_change,
                metadata={
                    "signal_workflow_id": signal.workflow_id,
                    "signal_completed_at_ms": signal.completed_at_ms,
                    "signal_score": signal.score,
                    "selection_bucket": signal.metadata.get("selection_bucket", "selected"),
                },
            )
        )
    return proposals


def build_wisdom_revision_proposals(
    signals: Sequence[Any],
    *,
    workflow_id: str,
    created_at_ms: int,
    policy: DefaultDreamLoopPolicy | None = None,
    budget_remaining: int | None = None,
    summary_builder: Callable[[DreamLoopSignal], str] | None = None,
    reasoning_builder: Callable[[DreamLoopSignal], Sequence[str]] | None = None,
    suggested_change_builder: Callable[[DreamLoopSignal], dict[str, Any]] | None = None,
    confidence_builder: Callable[[DreamLoopSignal], float] | None = None,
) -> list[WisdomRevisionProposal]:
    """Turn selected dream-loop signals into compact wisdom proposals."""

    selection = select_dream_loop_signals(
        signals,
        policy=policy,
        budget_remaining=budget_remaining,
    )
    selected_signals = []
    for signal in selection.selected:
        bucket = (
            "recent"
            if signal in selection.recent
            else "hotspot"
            if signal in selection.hotspots
            else "stale"
        )
        selected_signals.append(
            replace(
                signal,
                metadata={
                    **dict(signal.metadata or {}),
                    "selection_bucket": bucket,
                },
            )
        )
    return build_wisdom_revision_proposals_for_signals(
        selected_signals,
        workflow_id=workflow_id,
        created_at_ms=created_at_ms,
        summary_builder=summary_builder,
        reasoning_builder=reasoning_builder,
        suggested_change_builder=suggested_change_builder,
        confidence_builder=confidence_builder,
    )


def evaluate_wisdom_revision_proposal(
    proposal: WisdomRevisionProposal,
    *,
    decision: str,
    rationale: str,
    created_at_ms: int,
    candidate_workflow_id: str | None = None,
    lesson_summary: str | None = None,
    evidence: DreamLoopEvidence | None = None,
) -> ProposalEvaluation:
    """Turn proposal outcome into a wisdom-layer evaluation record."""

    if decision == "approved":
        result_kind = "workflow_design_artifact"
        result_id = str(candidate_workflow_id or proposal.candidate_workflow_id or proposal.workflow_id)
        metadata = {
            "lesson_summary": None,
            "approved_workflow_id": result_id,
            "proposal_summary": proposal.summary,
        }
    elif decision == "pending":
        result_kind = "workflow_candidate_artifact"
        result_id = str(candidate_workflow_id or proposal.candidate_workflow_id or proposal.workflow_id)
        metadata = {
            "lesson_summary": None,
            "approved_workflow_id": None,
            "candidate_workflow_id": result_id,
            "proposal_summary": proposal.summary,
        }
    elif decision == "deprecated":
        result_kind = "workflow_candidate_artifact"
        result_id = str(candidate_workflow_id or proposal.candidate_workflow_id or proposal.workflow_id)
        metadata = {
            "lesson_summary": lesson_summary or rationale,
            "approved_workflow_id": None,
            "candidate_workflow_id": result_id,
            "proposal_summary": proposal.summary,
            "deprecated": True,
        }
    else:
        result_kind = "wisdom_lesson"
        result_id = str(
            stable_id("wisdom_lesson", proposal.proposal_id, proposal.workflow_id)
        )
        metadata = {
            "lesson_summary": lesson_summary or rationale,
            "approved_workflow_id": None,
            "proposal_summary": proposal.summary,
        }

    if evidence is not None:
        metadata.update(
            {
                "evaluation_run_count": int(evidence.run_count),
                "evaluation_success_count": int(evidence.success_count),
                "evaluation_failure_count": int(evidence.failure_count),
                "evaluation_error_count": int(evidence.error_count),
                "evaluation_net_score": float(evidence.net_score),
                "positive_feedback": list(evidence.positive_feedback),
                "negative_feedback": list(evidence.negative_feedback),
                "sabotage_feedback": list(evidence.sabotage_feedback),
            }
        )

    return ProposalEvaluation(
        proposal_id=proposal.proposal_id,
        decision=decision,  # type: ignore[arg-type]
        rationale=rationale,
        result_kind=result_kind,
        result_id=result_id,
        created_at_ms=int(created_at_ms),
        metadata=metadata,
    )


def _single_dummy_grounding(workflow_id: str) -> list[Grounding]:
    return [Grounding(spans=[Span.from_dummy_for_workflow(workflow_id)])]


def _reasoning_node_from_proposal(proposal: WisdomRevisionProposal) -> Node:
    reasoning_node_id = str(stable_id("dream_reasoning_trace", proposal.proposal_id))
    return Node(
        id=reasoning_node_id,
        label=f"reasoning:{proposal.workflow_id}:{proposal.step_op}",
        type="entity",
        doc_id=reasoning_node_id,
        summary="\n".join(proposal.reasoning_trace),
        mentions=_single_dummy_grounding(proposal.workflow_id),
        metadata={
            "workspace_id": proposal.metadata.get("workspace_id", "demo"),
            "artifact_kind": "dream_reasoning_trace",
            "conversation_kind": "self_to_self",
            "workflow_id": proposal.workflow_id,
            "run_id": proposal.run_id,
            "step_op": proposal.step_op,
            "proposal_id": proposal.proposal_id,
            "created_at_ms": proposal.created_at_ms,
        },
    )


def _proposal_node_from_proposal(
    proposal: WisdomRevisionProposal, *, reasoning_node_id: str | None = None
) -> Node:
    return Node(
        id=proposal.proposal_id,
        label=f"proposal:{proposal.workflow_id}:{proposal.step_op}",
        type="entity",
        doc_id=proposal.proposal_id,
        summary=proposal.summary,
        mentions=_single_dummy_grounding(proposal.workflow_id),
        metadata={
            "workspace_id": proposal.metadata.get("workspace_id", "demo"),
            "artifact_kind": "workflow_revision_proposal",
            "workflow_id": proposal.workflow_id,
            "run_id": proposal.run_id,
            "step_op": proposal.step_op,
            "proposal_id": proposal.proposal_id,
            "status": proposal.status,
            "confidence": proposal.confidence,
            "evidence_run_ids": list(proposal.evidence_run_ids),
            "candidate_workflow_id": proposal.candidate_workflow_id,
            "evaluation_rounds": proposal.evaluation_rounds,
            "conversation_feedback": list(proposal.conversation_feedback),
            "reasoning_trace_id": reasoning_node_id,
            "suggested_change": proposal.suggested_change,
            "created_at_ms": proposal.created_at_ms,
        },
    )


def _evaluation_node_from_evaluation(
    proposal: WisdomRevisionProposal, evaluation: ProposalEvaluation
) -> Node:
    evaluation_node_id = str(
        stable_id(
            "proposal_evaluation",
            proposal.proposal_id,
            evaluation.decision,
            evaluation.result_kind,
            evaluation.result_id,
        )
    )
    return Node(
        id=evaluation_node_id,
        label=f"{evaluation.edge_kind}:{evaluation.decision}:{proposal.step_op}",
        type="entity",
        doc_id=evaluation_node_id,
        summary=evaluation.rationale,
        mentions=_single_dummy_grounding(proposal.workflow_id),
        metadata={
            "workspace_id": proposal.metadata.get("workspace_id", "demo"),
            "artifact_kind": evaluation.result_kind,
            "workflow_id": proposal.workflow_id,
            "proposal_id": proposal.proposal_id,
            "decision": evaluation.decision,
            "rationale": evaluation.rationale,
            "result_kind": evaluation.result_kind,
            "result_id": evaluation.result_id,
            "edge_kind": evaluation.edge_kind,
            "created_at_ms": evaluation.created_at_ms,
            **evaluation.metadata,
        },
    )


def _evaluation_edge_from_evaluation(
    proposal: WisdomRevisionProposal, evaluation: ProposalEvaluation, outcome_node_id: str
) -> Edge:
    edge_id = str(
        stable_id(
            "proposal_evaluation_edge",
            proposal.proposal_id,
            outcome_node_id,
            evaluation.decision,
        )
    )
    return Edge(
        id=edge_id,
        source_ids=[proposal.proposal_id],
        target_ids=[outcome_node_id],
        relation=evaluation.edge_kind,
        label=f"{proposal.step_op}:{evaluation.decision}",
        type="relationship",
        summary=evaluation.rationale,
        doc_id=proposal.workflow_id,
        properties={},
        metadata={
            "workspace_id": proposal.metadata.get("workspace_id", "demo"),
            "artifact_kind": "proposal_evaluation_edge",
            "workflow_id": proposal.workflow_id,
            "proposal_id": proposal.proposal_id,
            "decision": evaluation.decision,
            "result_kind": evaluation.result_kind,
            "result_id": evaluation.result_id,
            "edge_kind": evaluation.edge_kind,
            "created_at_ms": evaluation.created_at_ms,
        },
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=_single_dummy_grounding(proposal.workflow_id),
    )


def _persist_nodes(engine: Any, namespace: str, nodes: Sequence[Node]) -> list[str]:
    if not nodes:
        return []
    with scoped_namespace(engine, namespace):
        for node in nodes:
            engine.write.add_node(node)
    return [str(node.id) for node in nodes]


def _persist_edges(engine: Any, namespace: str, edges: Sequence[Edge]) -> list[str]:
    if not edges:
        return []
    with scoped_namespace(engine, namespace):
        for edge in edges:
            engine.write.add_edge(edge)
    return [str(edge.id) for edge in edges]


def _validate_candidate_workflow_design(design: Any, *, fallback_workflow_id: str) -> None:
    workflow_id = str(getattr(design, "workflow_id", None) or fallback_workflow_id or "").strip()
    if not workflow_id:
        raise ValueError("synthesized workflow design must have a workflow_id")

    nodes = list(getattr(design, "nodes", []) or [])
    edges = list(getattr(design, "edges", []) or [])
    if not nodes:
        raise ValueError("synthesized workflow design must contain at least one node")

    node_ids = {str(getattr(node, "id", "") or "").strip() for node in nodes}
    if "" in node_ids:
        raise ValueError("synthesized workflow design contains a node without id")

    start_node_id = str(getattr(design, "start_node_id", "") or "").strip()
    if start_node_id not in node_ids:
        raise ValueError("synthesized workflow design start_node_id must reference a node")

    start_nodes = [
        node
        for node in nodes
        if bool(dict(getattr(node, "metadata", {}) or {}).get("wf_start"))
    ]
    if len(start_nodes) != 1:
        raise ValueError("synthesized workflow design must have exactly one start node")

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        src_ids = list(getattr(edge, "source_ids", []) or [])
        dst_ids = list(getattr(edge, "target_ids", []) or [])
        if not src_ids or not dst_ids:
            raise ValueError("synthesized workflow design edge must have endpoints")
        src = str(src_ids[0]).strip()
        dst = str(dst_ids[0]).strip()
        if src not in node_ids or dst not in node_ids:
            raise ValueError(
                f"synthesized workflow design edge endpoints must reference nodes: {src!r} -> {dst!r}"
            )
        adjacency[src].append(dst)

    terminals = [
        node
        for node in nodes
        if bool(dict(getattr(node, "metadata", {}) or {}).get("wf_terminal"))
        or len(adjacency.get(str(getattr(node, "id", "")), [])) == 0
    ]
    if not terminals:
        raise ValueError("synthesized workflow design must contain at least one terminal or sink node")


def _workflow_artifact_node(
    *,
    workflow_id: str,
    revision_status: str,
    workspace_id: str = "demo",
    source_workflow_id: str | None = None,
    created_at_ms: int | None = None,
) -> Node:
    node_id = str(stable_id("workflow_artifact", workflow_id))
    metadata = {
        "workspace_id": workspace_id,
        "artifact_kind": "workflow_artifact",
        "entity_type": "workflow_artifact",
        "workflow_id": workflow_id,
        "revision_status": revision_status,
    }
    if source_workflow_id is not None:
        metadata["source_workflow_id"] = source_workflow_id
    if created_at_ms is not None:
        metadata["created_at_ms"] = int(created_at_ms)
    return Node(
        id=node_id,
        label=f"workflow:{workflow_id}",
        type="entity",
        doc_id=node_id,
        summary=f"workflow artifact for {workflow_id}",
        mentions=_single_dummy_grounding(workflow_id),
        metadata=metadata,
    )


def _workflow_lineage_edge(
    *,
    proposal: WisdomRevisionProposal,
    from_workflow_id: str,
    to_workflow_id: str,
    from_node_id: str,
    to_node_id: str,
    revision_status: str,
    created_at_ms: int,
) -> Edge:
    edge_id = str(
        stable_id(
            "workflow_revision_lineage",
            proposal.proposal_id,
            from_workflow_id,
            to_workflow_id,
        )
    )
    return Edge(
        id=edge_id,
        source_ids=[from_node_id],
        target_ids=[to_node_id],
        relation="workflow_revision_lineage",
        label=f"{proposal.step_op}:{revision_status}",
        type="relationship",
        summary=f"{from_workflow_id} -> {to_workflow_id} ({revision_status})",
        doc_id=proposal.workflow_id,
        properties={},
        metadata={
            "workspace_id": proposal.metadata.get("workspace_id", "demo"),
            "artifact_kind": "workflow_revision_lineage",
            "entity_type": "workflow_revision_lineage",
            "workflow_id": proposal.workflow_id,
            "proposal_id": proposal.proposal_id,
            "from_workflow_id": from_workflow_id,
            "to_workflow_id": to_workflow_id,
            "revision_status": revision_status,
            "created_at_ms": int(created_at_ms),
        },
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=_single_dummy_grounding(proposal.workflow_id),
    )


def _normalize_dream_loop_decision(
    value: Any,
    *,
    proposal: WisdomRevisionProposal,
    evidence: DreamLoopEvidence,
    policy: DefaultDreamLoopPolicy,
) -> DreamLoopDecision:
    if isinstance(value, DreamLoopDecision):
        return value
    if value is None:
        return default_dream_loop_decider(proposal, evidence, policy=policy)
    if isinstance(value, tuple) and len(value) == 3:
        approved, rationale, workflow_id = value
        return DreamLoopDecision(
            decision="approved" if bool(approved) else "rejected",
            rationale=str(rationale),
            candidate_workflow_id=(str(workflow_id) if workflow_id else None),
            lesson_summary=(None if bool(approved) else str(rationale)),
        )
    if isinstance(value, Mapping):
        meta = dict(value)
        return DreamLoopDecision(
            decision=str(meta.get("decision") or "rejected"),
            rationale=str(meta.get("rationale") or ""),
            candidate_workflow_id=(
                str(meta.get("candidate_workflow_id"))
                if meta.get("candidate_workflow_id") is not None
                else None
            ),
            lesson_summary=(
                str(meta.get("lesson_summary"))
                if meta.get("lesson_summary") is not None
                else None
            ),
            metadata={k: v for k, v in meta.items() if k not in {"decision", "rationale", "candidate_workflow_id", "lesson_summary"}},
        )
    raise TypeError(f"Unsupported dream-loop decision payload: {type(value)!r}")


def run_dream_loop_cycle(
    *,
    source_engine: Any,
    source_namespace: str,
    target_engine: Any,
    target_namespace: str,
    source_where: dict[str, Any],
    workflow_id: str,
    created_at_ms: int,
    pending_proposals: Sequence[WisdomRevisionProposal] | None = None,
    conversation_engine: Any | None = None,
    conversation_namespace: str | None = None,
    workflow_engine: Any | None = None,
    workflow_namespace: str | None = None,
    policy: DefaultDreamLoopPolicy | None = None,
    budget_remaining: int | None = None,
    summary_builder: Callable[[DreamLoopSignal], str] | None = None,
    reasoning_builder: Callable[[DreamLoopSignal], Sequence[str]] | None = None,
    suggested_change_builder: Callable[[DreamLoopSignal], dict[str, Any]] | None = None,
    confidence_builder: Callable[[DreamLoopSignal], float] | None = None,
    approval_decider: Callable[[WisdomRevisionProposal, DreamLoopEvidence], Any] | None = None,
    approved_workflow_builder: Callable[[WisdomRevisionProposal, ProposalEvaluation], Any]
    | None = None,
) -> DreamLoopRunResult:
    """End-to-end dream loop: read signals, emit proposals, persist outcomes."""

    policy = policy or DefaultDreamLoopPolicy()
    with scoped_namespace(source_engine, source_namespace):
        source_nodes = list(source_engine.read.get_nodes(where=source_where))

    selection = select_dream_loop_signals(
        source_nodes,
        policy=policy,
        budget_remaining=budget_remaining,
    )
    new_proposals = build_wisdom_revision_proposals(
        source_nodes,
        workflow_id=workflow_id,
        created_at_ms=created_at_ms,
        policy=policy,
        budget_remaining=budget_remaining,
        summary_builder=summary_builder,
        reasoning_builder=reasoning_builder,
        suggested_change_builder=suggested_change_builder,
        confidence_builder=confidence_builder,
    )
    proposals = list(pending_proposals or []) + list(new_proposals)
    if not proposals:
        return DreamLoopRunResult(
            selection=selection,
            proposals=(),
            evaluations=(),
            proposal_node_ids=(),
            outcome_node_ids=(),
        )

    reasoning_nodes = [_reasoning_node_from_proposal(item) for item in proposals]
    reasoning_node_ids: tuple[str, ...] = ()
    if conversation_engine is not None and conversation_namespace is not None:
        reasoning_node_ids = tuple(
            _persist_nodes(conversation_engine, conversation_namespace, reasoning_nodes)
        )

    materialized_workflow_namespace = (
        workflow_namespace
        if workflow_namespace is not None
        else target_namespace.replace(":wisdom", ":workflow")
    )

    evaluations: list[ProposalEvaluation] = []
    outcome_nodes: list[Node] = []
    proposal_evaluation_edges: list[Edge] = []
    evidence_items: list[DreamLoopEvidence] = []
    finalized_proposals: list[WisdomRevisionProposal] = []
    candidate_workflow_node_ids: list[str] = []
    candidate_workflow_edge_ids: list[str] = []
    approved_workflow_node_ids: list[str] = []
    approved_workflow_edge_ids: list[str] = []
    workflow_lineage_node_ids: list[str] = []
    workflow_lineage_edge_ids: list[str] = []
    for proposal_index, proposal in enumerate(proposals):
        evidence = collect_dream_loop_evidence(
            source_nodes,
            proposal=proposal,
            policy=policy,
        )
        evidence_items.append(evidence)
        raw_decision = (
            approval_decider(proposal, evidence)
            if approval_decider is not None
            else None
        )
        decision = _normalize_dream_loop_decision(
            raw_decision,
            proposal=proposal,
            evidence=evidence,
            policy=policy,
        )
        candidate_workflow_id = decision.candidate_workflow_id or proposal.candidate_workflow_id
        should_materialize_candidate = (
            decision.decision in {"pending", "approved"}
            and workflow_engine is not None
            and approved_workflow_builder is not None
            and not proposal.candidate_workflow_id
        )
        if should_materialize_candidate:
            preview_evaluation = evaluate_wisdom_revision_proposal(
                proposal,
                decision=decision.decision,
                rationale=decision.rationale,
                created_at_ms=int(created_at_ms) + proposal_index + 1,
                candidate_workflow_id=candidate_workflow_id,
                lesson_summary=decision.lesson_summary,
                evidence=evidence,
            )
            design = approved_workflow_builder(proposal, preview_evaluation)
            _validate_candidate_workflow_design(
                design,
                fallback_workflow_id=str(
                    candidate_workflow_id or proposal.workflow_id
                ),
            )
            nodes = list(getattr(design, "nodes", []) or [])
            edges = list(getattr(design, "edges", []) or [])
            candidate_workflow_id = str(
                getattr(design, "workflow_id", None)
                or candidate_workflow_id
                or proposal.workflow_id
            )
            if nodes:
                persisted_node_ids = _persist_nodes(
                    workflow_engine,
                    materialized_workflow_namespace,
                    nodes,
                )
                if decision.decision == "pending":
                    candidate_workflow_node_ids.extend(persisted_node_ids)
                else:
                    approved_workflow_node_ids.extend(persisted_node_ids)
            if edges:
                persisted_edge_ids = _persist_edges(
                    workflow_engine,
                    materialized_workflow_namespace,
                    edges,
                )
                if decision.decision == "pending":
                    candidate_workflow_edge_ids.extend(persisted_edge_ids)
                else:
                    approved_workflow_edge_ids.extend(persisted_edge_ids)
        evaluation = evaluate_wisdom_revision_proposal(
            proposal,
            decision=decision.decision,
            rationale=decision.rationale,
            created_at_ms=int(created_at_ms) + len(evaluations) + 1,
            candidate_workflow_id=candidate_workflow_id,
            lesson_summary=decision.lesson_summary,
            evidence=evidence,
        )
        evaluations.append(evaluation)
        outcome_node = _evaluation_node_from_evaluation(proposal, evaluation)
        outcome_nodes.append(outcome_node)
        proposal_evaluation_edges.append(
            _evaluation_edge_from_evaluation(proposal, evaluation, str(outcome_node.id))
        )
        finalized_proposals.append(
            replace(
                proposal,
                status=decision.decision,
                candidate_workflow_id=candidate_workflow_id,
                evaluation_rounds=max(
                    int(proposal.evaluation_rounds),
                    int(evidence.run_count),
                ),
                conversation_feedback=[
                    *evidence.positive_feedback,
                    *evidence.negative_feedback,
                    *evidence.sabotage_feedback,
                ],
                metadata={
                    **dict(proposal.metadata or {}),
                    "candidate_workflow_id": candidate_workflow_id,
                    "evaluation_rounds": max(
                        int(proposal.evaluation_rounds),
                        int(evidence.run_count),
                    ),
                    "evaluation_net_score": float(evidence.net_score),
                    "trusted_feedback_score": float(
                        evidence.trusted_feedback_score
                    ),
                    "sabotage_feedback_score": float(
                        evidence.sabotage_feedback_score
                    ),
                    **dict(decision.metadata or {}),
                },
            )
        )
        if workflow_engine is not None and candidate_workflow_id:
            origin_node = _workflow_artifact_node(
                workflow_id=proposal.workflow_id,
                revision_status="origin",
                workspace_id=proposal.metadata.get("workspace_id", "demo"),
                created_at_ms=created_at_ms,
            )
            candidate_node = _workflow_artifact_node(
                workflow_id=candidate_workflow_id,
                revision_status=decision.decision,
                workspace_id=proposal.metadata.get("workspace_id", "demo"),
                source_workflow_id=proposal.workflow_id,
                created_at_ms=evaluation.created_at_ms,
            )
            persisted_lineage_nodes = _persist_nodes(
                workflow_engine,
                materialized_workflow_namespace,
                [origin_node, candidate_node],
            )
            workflow_lineage_node_ids.extend(persisted_lineage_nodes)
            lineage_edge = _workflow_lineage_edge(
                proposal=proposal,
                from_workflow_id=proposal.workflow_id,
                to_workflow_id=candidate_workflow_id,
                from_node_id=str(origin_node.id),
                to_node_id=str(candidate_node.id),
                revision_status=decision.decision,
                created_at_ms=evaluation.created_at_ms,
            )
            workflow_lineage_edge_ids.extend(
                _persist_edges(
                    workflow_engine,
                    materialized_workflow_namespace,
                    [lineage_edge],
                )
            )

    proposal_nodes = [
        _proposal_node_from_proposal(
            item,
            reasoning_node_id=(
                reasoning_node_ids[index] if index < len(reasoning_node_ids) else None
            ),
        )
        for index, item in enumerate(finalized_proposals)
    ]
    proposal_node_ids = tuple(
        _persist_nodes(target_engine, target_namespace, proposal_nodes)
    )

    outcome_node_ids = tuple(_persist_nodes(target_engine, target_namespace, outcome_nodes))
    proposal_evaluation_edge_ids = tuple(
        _persist_edges(target_engine, target_namespace, proposal_evaluation_edges)
    )
    return DreamLoopRunResult(
        selection=selection,
        proposals=tuple(finalized_proposals),
        evaluations=tuple(evaluations),
        proposal_node_ids=proposal_node_ids,
        outcome_node_ids=outcome_node_ids,
        evidence=tuple(evidence_items),
        reasoning_node_ids=reasoning_node_ids,
        proposal_evaluation_edge_ids=proposal_evaluation_edge_ids,
        candidate_workflow_node_ids=tuple(candidate_workflow_node_ids),
        candidate_workflow_edge_ids=tuple(candidate_workflow_edge_ids),
        approved_workflow_node_ids=tuple(approved_workflow_node_ids),
        approved_workflow_edge_ids=tuple(approved_workflow_edge_ids),
        workflow_lineage_node_ids=tuple(workflow_lineage_node_ids),
        workflow_lineage_edge_ids=tuple(workflow_lineage_edge_ids),
    )
