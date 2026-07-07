"""Wisdom-domain helpers.

This package hosts reusable helpers for distilled, reusable lessons. It should
remain distinct from workflow runtime execution internals.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.wisdom.agentic_dream_design import (
        DREAM_MAINTENANCE_WORKFLOW_ID,
        build_dream_maintenance_workflow_design,
        dream_workflow_expected_ops,
        materialize_dream_workflow_design,
    )
    from kogwistar.wisdom.models import ExecutionWisdomTemplateResult
    from kogwistar.wisdom.proposals import ProposalEvaluation, WisdomRevisionProposal
    from kogwistar.wisdom.resolvers import dream_default_resolver
    from kogwistar.wisdom.dream_loop import (
        DreamLoopDecision,
        DreamLoopEvidence,
        DreamLoopSelection,
        DreamLoopRunResult,
        DreamLoopSignal,
        build_wisdom_revision_proposals,
        build_wisdom_revision_proposals_for_signals,
        collect_dream_loop_evidence,
        default_dream_loop_decider,
        evaluate_wisdom_revision_proposal,
        run_dream_loop_cycle,
        select_dream_loop_signals,
    )
    from kogwistar.wisdom.template import write_execution_wisdom_artifacts

__all__ = [
    "ExecutionWisdomTemplateResult",
    "DREAM_MAINTENANCE_WORKFLOW_ID",
    "DreamLoopDecision",
    "DreamLoopEvidence",
    "DreamLoopSelection",
    "DreamLoopRunResult",
    "DreamLoopSignal",
    "ProposalEvaluation",
    "WisdomRevisionProposal",
    "build_dream_maintenance_workflow_design",
    "build_wisdom_revision_proposals",
    "build_wisdom_revision_proposals_for_signals",
    "collect_dream_loop_evidence",
    "default_dream_loop_decider",
    "dream_default_resolver",
    "dream_workflow_expected_ops",
    "evaluate_wisdom_revision_proposal",
    "materialize_dream_workflow_design",
    "run_dream_loop_cycle",
    "select_dream_loop_signals",
    "write_execution_wisdom_artifacts",
]

_EXPORTS = {
    "ExecutionWisdomTemplateResult": "kogwistar.wisdom.models",
    "DREAM_MAINTENANCE_WORKFLOW_ID": "kogwistar.wisdom.agentic_dream_design",
    "DreamLoopDecision": "kogwistar.wisdom.dream_loop",
    "DreamLoopEvidence": "kogwistar.wisdom.dream_loop",
    "DreamLoopSelection": "kogwistar.wisdom.dream_loop",
    "DreamLoopRunResult": "kogwistar.wisdom.dream_loop",
    "DreamLoopSignal": "kogwistar.wisdom.dream_loop",
    "ProposalEvaluation": "kogwistar.wisdom.proposals",
    "WisdomRevisionProposal": "kogwistar.wisdom.proposals",
    "build_dream_maintenance_workflow_design": "kogwistar.wisdom.agentic_dream_design",
    "build_wisdom_revision_proposals": "kogwistar.wisdom.dream_loop",
    "build_wisdom_revision_proposals_for_signals": "kogwistar.wisdom.dream_loop",
    "collect_dream_loop_evidence": "kogwistar.wisdom.dream_loop",
    "default_dream_loop_decider": "kogwistar.wisdom.dream_loop",
    "dream_default_resolver": "kogwistar.wisdom.resolvers",
    "dream_workflow_expected_ops": "kogwistar.wisdom.agentic_dream_design",
    "evaluate_wisdom_revision_proposal": "kogwistar.wisdom.dream_loop",
    "materialize_dream_workflow_design": "kogwistar.wisdom.agentic_dream_design",
    "run_dream_loop_cycle": "kogwistar.wisdom.dream_loop",
    "select_dream_loop_signals": "kogwistar.wisdom.dream_loop",
    "write_execution_wisdom_artifacts": "kogwistar.wisdom.template",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
