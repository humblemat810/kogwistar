from __future__ import annotations

"""Wisdom-layer proposal and evaluation artifacts."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class WisdomRevisionProposal:
    """Compact proposal distilled from internal reasoning."""

    proposal_id: str
    workflow_id: str
    run_id: str
    step_op: str
    summary: str
    reasoning_trace: list[str]
    evidence_run_ids: list[str]
    confidence: float
    created_at_ms: int
    status: Literal["proposed", "pending", "approved", "rejected", "deprecated"] = "proposed"
    candidate_workflow_id: str | None = None
    evaluation_rounds: int = 0
    conversation_feedback: list[str] = field(default_factory=list)
    suggested_change: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProposalEvaluation:
    """Outcome of wisdom proposal review."""

    proposal_id: str
    decision: Literal["approved", "rejected", "pending", "deprecated"]
    rationale: str
    result_kind: str
    result_id: str
    created_at_ms: int
    edge_kind: str = "proposal_evaluation"
    metadata: dict[str, Any] = field(default_factory=dict)
