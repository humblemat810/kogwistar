"""Best-effort wisdom distillation and attributable skill projection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from kogwistar.wisdom.proposals import ProposalEvaluation, WisdomRevisionProposal

from .skills import (
    SkillGraphArtifact,
    SkillGraphEdge,
    SkillGraphNode,
    SkillProjectionStore,
    validate_skill_artifact,
)


@dataclass(frozen=True, slots=True)
class DistillationRequest:
    run_id: str
    workflow_id: str
    outcome: str
    evidence_refs: tuple[str, ...] = ()
    feedback: tuple[str, ...] = ()
    tenant_id: str | None = None
    project_id: str | None = None


@dataclass(frozen=True, slots=True)
class DistillationResult:
    submitted: bool
    proposal_id: str | None = None
    error: str | None = None


class BestEffortDistiller:
    """Submit a request without allowing learning failure to affect a run."""

    def __init__(self, submit: Callable[[DistillationRequest], WisdomRevisionProposal | None]) -> None:
        self._submit = submit

    def submit(self, request: DistillationRequest) -> DistillationResult:
        try:
            proposal = self._submit(request)
            return DistillationResult(
                submitted=proposal is not None,
                proposal_id=getattr(proposal, "proposal_id", None),
            )
        except Exception as exc:  # best effort by contract
            return DistillationResult(submitted=False, error=f"{type(exc).__name__}: {exc}")


def compile_approved_proposal_to_skill(
    proposal: WisdomRevisionProposal,
    evaluation: ProposalEvaluation,
    *,
    store: SkillProjectionStore,
    provider_id: str = "kogwistar.wisdom",
    tenant_id: str | None = None,
    project_id: str | None = None,
    projection_revision: int = 1,
) -> SkillGraphArtifact | None:
    """Compile only approved proposal outcomes into a rebuildable skill view."""

    if evaluation.decision != "approved" or proposal.status not in {"approved", "proposed", "pending"}:
        return None
    source = f"wisdom:{proposal.proposal_id}"
    fingerprint = hashlib.sha256(
        f"{proposal.proposal_id}:{evaluation.result_id}:{proposal.summary}".encode("utf-8")
    ).hexdigest()
    root = f"skill:{provider_id}:{proposal.proposal_id}"
    step = f"step:{proposal.proposal_id}:procedure"
    artifact = SkillGraphArtifact(
        provider_id=provider_id,
        provider_local_id=proposal.proposal_id,
        skill_version=f"{evaluation.result_id}:v{projection_revision}",
        source_fingerprint=fingerprint,
        projection_revision=projection_revision,
        tenant_id=tenant_id,
        project_id=project_id,
        namespace="wisdom:skill",
        nodes=[
            SkillGraphNode(
                node_id=root,
                kind="skill",
                name=proposal.step_op,
                summary=proposal.summary,
                source_ref=source,
                metadata={"approval_result_id": evaluation.result_id},
            ),
            SkillGraphNode(
                node_id=step,
                kind="instruction",
                name=proposal.step_op,
                summary=proposal.summary,
                source_ref=source,
                metadata={"evidence_run_ids": list(proposal.evidence_run_ids)},
            ),
        ],
        edges=[
            SkillGraphEdge(
                edge_id=f"edge:{root}:{step}",
                kind="sequence",
                source_ids=[root],
                target_ids=[step],
                source_ref=source,
                metadata={"approval_result_id": evaluation.result_id},
            )
        ],
        provenance={
            "proposal_id": proposal.proposal_id,
            "evaluation_id": evaluation.result_id,
            "source_lesson": source,
            "evidence_run_ids": list(proposal.evidence_run_ids),
        },
    )
    validate_skill_artifact(artifact, tenant_id=tenant_id, project_id=project_id)
    return store.upsert(artifact)


def record_skill_use_observation(
    *,
    skill_id: str,
    run_id: str,
    outcome: str,
    evidence_refs: tuple[str, ...] = (),
    feedback: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return memory-like observation; never mutates skill approval/projection."""

    return {
        "kind": "skill_use_observation",
        "skill_id": str(skill_id),
        "run_id": str(run_id),
        "outcome": str(outcome),
        "evidence_refs": list(evidence_refs),
        "feedback": list(feedback),
    }


__all__ = [
    "BestEffortDistiller",
    "DistillationRequest",
    "DistillationResult",
    "compile_approved_proposal_to_skill",
    "record_skill_use_observation",
]
