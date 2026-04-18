"""Runtime package compatibility entrypoints with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.runtime.contract import (
        BasePredicate,
        WorkflowEdgeInfo,
        WorkflowNodeInfo,
        WorkflowSpec,
    )
    from kogwistar.runtime.design import BaseWorkflowDesigner
    from kogwistar.runtime.models import (
        WorkflowDesignArtifact,
        WorkflowInvocationRequest,
    )
    from kogwistar.runtime.replay import load_checkpoint, replay_to
    from kogwistar.runtime.resolvers import (
        BaseResolver,
        MappingStepResolver,
    )
    from kogwistar.workflow.analytics import (
        ExecutionFailurePattern,
        summarize_execution_failure_patterns,
        WorkflowStepExecutionStats,
        summarize_workflow_step_execution_stats,
    )
    from kogwistar.maintenance.models import (
        GroupedArtifactWriteResult,
        MaintenanceTemplateResult,
        VersionedArtifactWriteResult,
    )
    from kogwistar.maintenance.artifacts import (
        write_versioned_artifact,
    )
    from kogwistar.maintenance.grouped_artifacts import (
        write_grouped_versioned_artifacts,
    )
    from kogwistar.maintenance.template import (
        run_grouped_maintenance_template,
    )
    from kogwistar.wisdom.models import (
        ExecutionWisdomTemplateResult,
    )
    from kogwistar.wisdom.template import (
        write_execution_wisdom_artifacts,
    )
    from kogwistar.runtime.runtime import (
        RouteDecision,
        RunResult,
        StepContext,
        WorkflowRuntime,
    )

__all__ = [
    "BasePredicate",
    "WorkflowEdgeInfo",
    "WorkflowNodeInfo",
    "WorkflowSpec",
    "BaseWorkflowDesigner",
    "WorkflowDesignArtifact",
    "WorkflowInvocationRequest",
    "load_checkpoint",
    "replay_to",
    "BaseResolver",
    "MappingStepResolver",
    "ExecutionFailurePattern",
    "summarize_execution_failure_patterns",
    "WorkflowStepExecutionStats",
    "summarize_workflow_step_execution_stats",
    "VersionedArtifactWriteResult",
    "write_versioned_artifact",
    "GroupedArtifactWriteResult",
    "write_grouped_versioned_artifacts",
    "MaintenanceTemplateResult",
    "run_grouped_maintenance_template",
    "ExecutionWisdomTemplateResult",
    "write_execution_wisdom_artifacts",
    "RouteDecision",
    "RunResult",
    "StepContext",
    "WorkflowRuntime",
]

_EXPORTS = {
    "BasePredicate": "kogwistar.runtime.contract",
    "WorkflowEdgeInfo": "kogwistar.runtime.contract",
    "WorkflowNodeInfo": "kogwistar.runtime.contract",
    "WorkflowSpec": "kogwistar.runtime.contract",
    "BaseWorkflowDesigner": "kogwistar.runtime.design",
    "WorkflowDesignArtifact": "kogwistar.runtime.models",
    "WorkflowInvocationRequest": "kogwistar.runtime.models",
    "load_checkpoint": "kogwistar.runtime.replay",
    "replay_to": "kogwistar.runtime.replay",
    "BaseResolver": "kogwistar.runtime.resolvers",
    "MappingStepResolver": "kogwistar.runtime.resolvers",
    "ExecutionFailurePattern": "kogwistar.workflow.analytics",
    "summarize_execution_failure_patterns": "kogwistar.workflow.analytics",
    "WorkflowStepExecutionStats": "kogwistar.workflow.analytics",
    "summarize_workflow_step_execution_stats": "kogwistar.workflow.analytics",
    "VersionedArtifactWriteResult": "kogwistar.maintenance.models",
    "write_versioned_artifact": "kogwistar.maintenance.artifacts",
    "GroupedArtifactWriteResult": "kogwistar.maintenance.models",
    "write_grouped_versioned_artifacts": "kogwistar.maintenance.grouped_artifacts",
    "MaintenanceTemplateResult": "kogwistar.maintenance.models",
    "run_grouped_maintenance_template": "kogwistar.maintenance.template",
    "ExecutionWisdomTemplateResult": "kogwistar.wisdom.models",
    "write_execution_wisdom_artifacts": "kogwistar.wisdom.template",
    "RouteDecision": "kogwistar.runtime.runtime",
    "RunResult": "kogwistar.runtime.runtime",
    "StepContext": "kogwistar.runtime.runtime",
    "WorkflowRuntime": "kogwistar.runtime.runtime",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
