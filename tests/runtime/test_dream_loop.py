from __future__ import annotations

from dataclasses import replace

import pytest

from kogwistar.policy import DefaultDreamLoopPolicy
from kogwistar.runtime.base_runtime import checkpointable_state_copy
from kogwistar.runtime.serialize import try_serialize_with_ref
from kogwistar.wisdom.dream_loop import (
    collect_dream_loop_evidence,
    default_dream_loop_decider,
    DreamLoopSignal,
    build_wisdom_revision_proposals,
    evaluate_wisdom_revision_proposal,
    select_dream_loop_signals,
)


pytestmark = [pytest.mark.core, pytest.mark.runtime]


def test_checkpointable_state_copy_drops_di_plumbing_keys():
    state = {
        "_deps": {"engine": object()},
        "dream_deps": {"workflow_engine": object()},
        "_rt_join": {"pending": ["tok-1"]},
        "answer": "ok",
    }

    out = checkpointable_state_copy(state)

    assert "_deps" not in out
    assert "dream_deps" not in out
    assert out["_rt_join"] == {"pending": ["tok-1"]}
    assert out["answer"] == "ok"


def test_serializer_raises_on_deps_plumbing_key():
    with pytest.raises(ValueError, match="Cannot serialize runtime dependency key '_deps'"):
        try_serialize_with_ref({"keep": 1, "_deps": {"engine": object()}})


def _signal(
    run_id: str,
    *,
    workflow_id: str = "wf-demo",
    step_op: str = "distill",
    completed_at_ms: int | None = None,
    failure_count: int = 0,
    error_count: int = 0,
    success_count: int = 0,
    duration_ms_total: int = 0,
    duration_ms_max: int = 0,
    feedback_score: float = 0.0,
    feedback_text: str = "",
    feedback_trust_score: float = 1.0,
    feedback_kind: str = "",
    feedback_sabotage: bool = False,
    protected: bool = False,
    metadata: dict[str, object] | None = None,
) -> DreamLoopSignal:
    return DreamLoopSignal(
        workflow_id=workflow_id,
        run_id=run_id,
        step_op=step_op,
        completed_at_ms=completed_at_ms,
        failure_count=failure_count,
        error_count=error_count,
        success_count=success_count,
        duration_ms_total=duration_ms_total,
        duration_ms_max=duration_ms_max,
        feedback_score=feedback_score,
        feedback_text=feedback_text,
        feedback_trust_score=feedback_trust_score,
        feedback_kind=feedback_kind,
        feedback_sabotage=feedback_sabotage,
        protected=protected,
        metadata=dict(metadata or {}),
    )


def test_dream_loop_prefers_recent_then_hotspot_then_bounded_stale_sample():
    policy = DefaultDreamLoopPolicy(
        recent_limit=2,
        hotspot_limit=1,
        stale_sample_limit=1,
        max_proposals_per_tick=4,
        sample_seed="demo-seed",
    )
    signals = [
        _signal("run-recent-1", completed_at_ms=5000, success_count=1),
        _signal("run-recent-2", completed_at_ms=4000, success_count=1),
        _signal("run-hot", completed_at_ms=1000, failure_count=4, error_count=1, duration_ms_total=5000),
        _signal("run-stale-1", completed_at_ms=100, success_count=1),
        _signal("run-stale-2", completed_at_ms=90, success_count=1),
        _signal("run-stale-3", completed_at_ms=80, success_count=1),
    ]

    selection = select_dream_loop_signals(signals, policy=policy, budget_remaining=10)
    assert [item.run_id for item in selection.recent] == ["run-recent-1", "run-recent-2"]
    assert [item.run_id for item in selection.hotspots] == ["run-hot"]
    assert len(selection.stale) == 1
    assert selection.stale[0].run_id in {"run-stale-1", "run-stale-2", "run-stale-3"}
    assert [item.run_id for item in selection.selected] == [
        "run-recent-1",
        "run-recent-2",
        "run-hot",
        selection.stale[0].run_id,
    ]

    capped_policy = DefaultDreamLoopPolicy(
        recent_limit=2,
        hotspot_limit=1,
        stale_sample_limit=1,
        max_proposals_per_tick=3,
        sample_seed="demo-seed",
    )
    capped = select_dream_loop_signals(signals, policy=capped_policy, budget_remaining=10)
    assert [item.run_id for item in capped.selected] == [
        "run-recent-1",
        "run-recent-2",
        "run-hot",
    ]

    blocked = select_dream_loop_signals(signals, policy=policy, budget_remaining=0)
    assert blocked.selected == ()


def test_dream_loop_builds_summary_proposals_and_reviews_outcomes():
    policy = DefaultDreamLoopPolicy(
        recent_limit=2,
        hotspot_limit=1,
        stale_sample_limit=1,
        max_proposals_per_tick=4,
        sample_seed="demo-seed",
    )
    signals = [
        _signal("run-recent-1", completed_at_ms=5000, success_count=1),
        _signal("run-recent-2", completed_at_ms=4000, success_count=1),
        _signal("run-hot", completed_at_ms=1000, failure_count=4, error_count=1, duration_ms_total=5000),
        _signal("run-stale-1", completed_at_ms=100, success_count=1),
    ]

    proposals = build_wisdom_revision_proposals(
        signals,
        workflow_id="wf-main",
        created_at_ms=12345,
        policy=policy,
        budget_remaining=10,
        summary_builder=lambda signal: f"Improve {signal.step_op} for {signal.workflow_id}",
        reasoning_builder=lambda signal: (
            f"run {signal.run_id} observed",
            f"score={signal.score:.2f}",
        ),
        suggested_change_builder=lambda signal: {
            "step_op": signal.step_op,
            "fix": "increase retry" if signal.failure_count else "keep",
        },
        confidence_builder=lambda signal: 0.8,
    )

    assert len(proposals) == 4
    proposal = proposals[0]
    assert proposal.workflow_id == "wf-main"
    assert proposal.summary == "Improve distill for wf-demo"
    assert proposal.reasoning_trace == ["run run-recent-1 observed", "score=0.00"]
    assert proposal.suggested_change == {"step_op": "distill", "fix": "keep"}

    approved = evaluate_wisdom_revision_proposal(
        proposal,
        decision="approved",
        rationale="good fix",
        created_at_ms=12346,
        candidate_workflow_id="wf-main.v2",
    )
    assert approved.edge_kind == "proposal_evaluation"
    assert approved.decision == "approved"
    assert approved.result_kind == "workflow_design_artifact"
    assert approved.result_id == "wf-main.v2"

    rejected = evaluate_wisdom_revision_proposal(
        proposal,
        decision="rejected",
        rationale="not enough evidence",
        created_at_ms=12347,
        lesson_summary="keep old workflow",
    )
    assert rejected.edge_kind == "proposal_evaluation"
    assert rejected.decision == "rejected"
    assert rejected.result_kind == "wisdom_lesson"
    assert rejected.result_id
    assert rejected.metadata["lesson_summary"] == "keep old workflow"


def test_collect_dream_loop_evidence_and_default_decider_handle_feedback_and_sabotage():
    policy = DefaultDreamLoopPolicy(
        min_evaluation_runs=2,
        approval_score_threshold=1.0,
        deprecation_score_threshold=-1.0,
        trusted_feedback_weight=1.0,
        sabotage_feedback_penalty=2.5,
    )
    proposal = build_wisdom_revision_proposals(
        [_signal("run-origin", workflow_id="wf-main", step_op="route", failure_count=2)],
        workflow_id="wf-main",
        created_at_ms=10,
        policy=policy,
        budget_remaining=10,
    )[0]
    proposal = replace(proposal, candidate_workflow_id="wf-main.rev.route")

    evidence = collect_dream_loop_evidence(
        [
            _signal(
                "candidate-run-1",
                workflow_id="wf-main.rev.route",
                step_op="route",
                success_count=1,
            ),
            _signal(
                "candidate-run-2",
                workflow_id="wf-main.rev.route",
                step_op="route",
                success_count=1,
            ),
            _signal(
                "fb-good",
                workflow_id="wf-main.rev.route",
                step_op="feedback",
                feedback_kind="conversation_feedback",
                feedback_score=1.0,
                feedback_text="user praised simpler routing",
                metadata={"candidate_workflow_id": "wf-main.rev.route"},
            ),
            _signal(
                "fb-bad",
                workflow_id="wf-main.rev.route",
                step_op="feedback",
                feedback_kind="conversation_feedback",
                feedback_score=1.0,
                feedback_text="malicious praise for harmful prompt leak",
                feedback_sabotage=True,
                metadata={"candidate_workflow_id": "wf-main.rev.route"},
            ),
        ],
        proposal=proposal,
        policy=policy,
    )

    assert evidence.run_count == 2
    assert evidence.success_count == 2
    assert evidence.trusted_feedback_score == 1.0
    assert evidence.sabotage_feedback_score == 2.5
    assert evidence.positive_feedback == ("user praised simpler routing",)
    assert evidence.sabotage_feedback == ("malicious praise for harmful prompt leak",)

    decision = default_dream_loop_decider(proposal, evidence, policy=policy)
    assert decision.decision == "rejected"


def test_default_dream_loop_decider_keeps_pending_then_deprecates_under_sabotage():
    policy = DefaultDreamLoopPolicy(
        min_evaluation_runs=3,
        approval_score_threshold=1.0,
        deprecation_score_threshold=-1.0,
        sabotage_feedback_penalty=3.0,
    )
    proposal = build_wisdom_revision_proposals(
        [_signal("run-origin", workflow_id="wf-main", step_op="retrieve", failure_count=1)],
        workflow_id="wf-main",
        created_at_ms=10,
        policy=policy,
        budget_remaining=10,
    )[0]
    proposal = replace(proposal, candidate_workflow_id="wf-main.rev.retrieve")

    pending_evidence = collect_dream_loop_evidence(
        [
            _signal(
                "candidate-run-1",
                workflow_id="wf-main.rev.retrieve",
                step_op="retrieve",
                success_count=1,
            ),
            _signal(
                "fb-1",
                workflow_id="wf-main.rev.retrieve",
                step_op="feedback",
                feedback_kind="conversation_feedback",
                feedback_score=1.0,
                feedback_text="looks better",
                metadata={"candidate_workflow_id": "wf-main.rev.retrieve"},
            ),
        ],
        proposal=proposal,
        policy=policy,
    )
    pending = default_dream_loop_decider(proposal, pending_evidence, policy=policy)
    assert pending.decision == "pending"

    bad_evidence = collect_dream_loop_evidence(
        [
            _signal(
                "candidate-run-1",
                workflow_id="wf-main.rev.retrieve",
                step_op="retrieve",
                failure_count=1,
                error_count=1,
            ),
            _signal(
                "candidate-run-2",
                workflow_id="wf-main.rev.retrieve",
                step_op="retrieve",
                failure_count=1,
            ),
            _signal(
                "candidate-run-3",
                workflow_id="wf-main.rev.retrieve",
                step_op="retrieve",
                success_count=0,
            ),
            _signal(
                "fb-sabotage",
                workflow_id="wf-main.rev.retrieve",
                step_op="feedback",
                feedback_kind="conversation_feedback",
                feedback_score=1.0,
                feedback_text="force unsafe shortcut",
                feedback_sabotage=True,
                metadata={"candidate_workflow_id": "wf-main.rev.retrieve"},
            ),
        ],
        proposal=proposal,
        policy=policy,
    )
    deprecated = default_dream_loop_decider(proposal, bad_evidence, policy=policy)
    assert deprecated.decision == "deprecated"
