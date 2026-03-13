"""Runtime package compatibility entrypoints with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_knowledge_engine.runtime.contract import (
        BasePredicate,
        WorkflowEdgeInfo,
        WorkflowNodeInfo,
        WorkflowSpec,
    )
    from graph_knowledge_engine.runtime.design import BaseWorkflowDesigner
    from graph_knowledge_engine.runtime.replay import load_checkpoint, replay_to
    from graph_knowledge_engine.runtime.resolvers import (
        BaseResolver,
        MappingStepResolver,
    )
    from graph_knowledge_engine.runtime.runtime import (
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
    "BasePredicate": "graph_knowledge_engine.runtime.contract",
    "WorkflowEdgeInfo": "graph_knowledge_engine.runtime.contract",
    "WorkflowNodeInfo": "graph_knowledge_engine.runtime.contract",
    "WorkflowSpec": "graph_knowledge_engine.runtime.contract",
    "BaseWorkflowDesigner": "graph_knowledge_engine.runtime.design",
    "load_checkpoint": "graph_knowledge_engine.runtime.replay",
    "replay_to": "graph_knowledge_engine.runtime.replay",
    "BaseResolver": "graph_knowledge_engine.runtime.resolvers",
    "MappingStepResolver": "graph_knowledge_engine.runtime.resolvers",
    "RouteDecision": "graph_knowledge_engine.runtime.runtime",
    "RunResult": "graph_knowledge_engine.runtime.runtime",
    "StepContext": "graph_knowledge_engine.runtime.runtime",
    "WorkflowRuntime": "graph_knowledge_engine.runtime.runtime",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
