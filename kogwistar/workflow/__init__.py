"""Workflow package entrypoints."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.workflow.analytics import (
        ExecutionFailurePattern,
        WorkflowStepExecutionStats,
        summarize_execution_failure_patterns,
        summarize_workflow_step_execution_stats,
    )

__all__ = [
    "ExecutionFailurePattern",
    "WorkflowStepExecutionStats",
    "summarize_execution_failure_patterns",
    "summarize_workflow_step_execution_stats",
]

_EXPORTS = {
    "ExecutionFailurePattern": "kogwistar.workflow.analytics",
    "WorkflowStepExecutionStats": "kogwistar.workflow.analytics",
    "summarize_execution_failure_patterns": "kogwistar.workflow.analytics",
    "summarize_workflow_step_execution_stats": "kogwistar.workflow.analytics",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
