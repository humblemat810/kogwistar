"""Wisdom learning remains reviewed, attributable, and best effort."""

from __future__ import annotations

import pytest

from kogwistar.agent import (
    BestEffortDistiller,
    DistillationRequest,
    SkillProjectionStore,
    compile_approved_proposal_to_skill,
    record_skill_use_observation,
)
from kogwistar.wisdom.proposals import ProposalEvaluation, WisdomRevisionProposal


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def _proposal(status: str = "pending") -> WisdomRevisionProposal:
    return WisdomRevisionProposal(
        proposal_id="proposal-1",
        workflow_id="wf-1",
        run_id="run-1",
        step_op="deploy",
        summary="deploy with validation",
        reasoning_trace=["failure pattern"],
        evidence_run_ids=["run-1"],
        confidence=0.9,
        created_at_ms=1,
        status=status,  # type: ignore[arg-type]
    )


def _evaluation(decision: str) -> ProposalEvaluation:
    return ProposalEvaluation(
        proposal_id="proposal-1",
        decision=decision,  # type: ignore[arg-type]
        rationale="reviewed",
        result_kind="wisdom_lesson",
        result_id="eval-1",
        created_at_ms=2,
    )


def test_distillation_is_best_effort_and_never_fails_source_run() -> None:
    request = DistillationRequest(run_id="run-1", workflow_id="wf-1", outcome="failed")
    result = BestEffortDistiller(lambda _request: (_ for _ in ()).throw(RuntimeError("offline"))).submit(request)
    assert result.submitted is False
    assert "offline" in str(result.error)


def test_successful_run_may_have_no_generalizable_proposal() -> None:
    request = DistillationRequest(run_id="run-2", workflow_id="wf-1", outcome="succeeded")
    result = BestEffortDistiller(lambda _request: None).submit(request)
    assert result.submitted is False
    assert result.proposal_id is None


def test_user_correction_can_submit_pending_proposal() -> None:
    request = DistillationRequest(
        run_id="run-3",
        workflow_id="wf-1",
        outcome="corrected",
        feedback=("Use the validated deployment path",),
    )
    proposal = _proposal("pending")
    result = BestEffortDistiller(lambda received: proposal if received.feedback else None).submit(request)
    assert result.submitted is True
    assert result.proposal_id == proposal.proposal_id


def test_rejected_and_pending_evaluations_do_not_become_skills() -> None:
    store = SkillProjectionStore()
    assert compile_approved_proposal_to_skill(_proposal(), _evaluation("pending"), store=store) is None
    assert compile_approved_proposal_to_skill(_proposal(), _evaluation("rejected"), store=store) is None
    assert store.get("kogwistar.wisdom", "proposal-1") is None


def test_approved_lesson_becomes_versioned_skill_with_lineage() -> None:
    store = SkillProjectionStore()
    artifact = compile_approved_proposal_to_skill(
        _proposal("approved"), _evaluation("approved"), store=store, project_id="p1"
    )
    assert artifact is not None
    assert artifact.projection_revision == 1
    assert artifact.provenance["proposal_id"] == "proposal-1"
    assert artifact.provenance["evidence_run_ids"] == ["run-1"]
    assert store.get("kogwistar.wisdom", "proposal-1") is artifact


def test_new_approved_revision_preserves_prior_projection_lineage() -> None:
    store = SkillProjectionStore()
    first = compile_approved_proposal_to_skill(
        _proposal("approved"), _evaluation("approved"), store=store, projection_revision=1
    )
    second = compile_approved_proposal_to_skill(
        _proposal("approved"), _evaluation("approved"), store=store, projection_revision=2
    )
    assert first is not None and second is not None
    assert first.skill_version != second.skill_version
    assert first.provenance["proposal_id"] == second.provenance["proposal_id"]
    assert store.get("kogwistar.wisdom", "proposal-1") is second
    assert store.history("kogwistar.wisdom", "proposal-1") == (first, second)


def test_memory_observation_does_not_change_skill_projection() -> None:
    observation = record_skill_use_observation(
        skill_id="kogwistar.wisdom:proposal-1",
        run_id="run-2",
        outcome="failed",
        evidence_refs=("step-2",),
    )
    assert observation["kind"] == "skill_use_observation"
    assert observation["skill_id"].endswith("proposal-1")
