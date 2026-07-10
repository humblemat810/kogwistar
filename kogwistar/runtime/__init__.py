"""Runtime package compatibility entrypoints with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.runtime.budget import BudgetAttribution, BudgetEvent
    from kogwistar.runtime.budget_adapters import summarize_budget_events
    from kogwistar.runtime.checkpointed_projection import (
        CheckpointedProjectionStore,
        ProjectionConflictError,
        ProjectionCheckpoint,
        ProjectionLoadResult,
        refresh_checkpointed_named_projection,
    )
    from kogwistar.runtime.retry import (
        RetryAttemptRecord,
        RetryExhaustedError,
        RetryResult,
        retry_with_context,
    )
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
        AsyncMappingStepResolver,
        BaseResolver,
        MappingStepResolver,
    )
    from kogwistar.runtime.async_runtime import (
        AsyncStepFn,
        AsyncWorkflowRuntime,
        SyncStepFn,
    )
    from kogwistar.runtime.executor import (
        RunRequest,
        TerminalStatus,
        WorkflowExecutor,
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
    from kogwistar.runtime.sinks import JsonlEventSink

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
    "AsyncMappingStepResolver",
    "SyncStepFn",
    "AsyncStepFn",
    "AsyncWorkflowRuntime",
    "WorkflowExecutor",
    "RunRequest",
    "TerminalStatus",
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
    "JsonlEventSink",
    "summarize_budget_events",
    "CheckpointedProjectionStore",
    "ProjectionConflictError",
    "ProjectionCheckpoint",
    "ProjectionLoadResult",
    "refresh_checkpointed_named_projection",
    "BudgetAttribution",
    "BudgetEvent",
    "budget_event_to_dict",
    "budget_event_from_dict",
    "RetryAttemptRecord",
    "RetryExhaustedError",
    "RetryResult",
    "retry_with_context",
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
    "AsyncMappingStepResolver": "kogwistar.runtime.resolvers",
    "SyncStepFn": "kogwistar.runtime.async_runtime",
    "AsyncStepFn": "kogwistar.runtime.async_runtime",
    "AsyncWorkflowRuntime": "kogwistar.runtime.async_runtime",
    "WorkflowExecutor": "kogwistar.runtime.executor",
    "RunRequest": "kogwistar.runtime.executor",
    "TerminalStatus": "kogwistar.runtime.executor",
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
    "JsonlEventSink": "kogwistar.runtime.sinks",
    "summarize_budget_events": "kogwistar.runtime.budget_adapters",
    "CheckpointedProjectionStore": "kogwistar.runtime.checkpointed_projection",
    "ProjectionConflictError": "kogwistar.runtime.checkpointed_projection",
    "ProjectionCheckpoint": "kogwistar.runtime.checkpointed_projection",
    "ProjectionLoadResult": "kogwistar.runtime.checkpointed_projection",
    "refresh_checkpointed_named_projection": "kogwistar.runtime.checkpointed_projection",
    "BudgetAttribution": "kogwistar.runtime.budget",
    "BudgetEvent": "kogwistar.runtime.budget",
    "budget_event_to_dict": "kogwistar.runtime.budget",
    "budget_event_from_dict": "kogwistar.runtime.budget",
    "RetryAttemptRecord": "kogwistar.runtime.retry",
    "RetryExhaustedError": "kogwistar.runtime.retry",
    "RetryResult": "kogwistar.runtime.retry",
    "retry_with_context": "kogwistar.runtime.retry",
}

_SUBMODULE_EXPORTS = {
    "async_runtime": "kogwistar.runtime.async_runtime",
    "runtime": "kogwistar.runtime.runtime",
}


def __getattr__(name: str):
    submodule_name = _SUBMODULE_EXPORTS.get(name)
    if submodule_name is not None:
        return import_module(submodule_name)
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
