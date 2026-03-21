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
    from kogwistar.runtime.replay import load_checkpoint, replay_to
    from kogwistar.runtime.resolvers import (
        BaseResolver,
        MappingStepResolver,
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
    "load_checkpoint",
    "replay_to",
    "BaseResolver",
    "MappingStepResolver",
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
    "load_checkpoint": "kogwistar.runtime.replay",
    "replay_to": "kogwistar.runtime.replay",
    "BaseResolver": "kogwistar.runtime.resolvers",
    "MappingStepResolver": "kogwistar.runtime.resolvers",
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
