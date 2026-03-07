from __future__ import annotations

from typing import Any, Type

from .graph_kinds import KIND_CHAT


def default_node_type_for_graph_kind(graph_kind: str):
    from .engine_core.models import Node

    if graph_kind == KIND_CHAT:
        from .conversation.models import ConversationNode

        return ConversationNode
    return Node


def default_edge_type_for_graph_kind(graph_kind: str):
    from .engine_core.models import Edge

    if graph_kind == KIND_CHAT:
        from .conversation.models import ConversationEdge

        return ConversationEdge
    return Edge


def _resolve_class_name(class_name: str):
    from .engine_core import models as core_models
    from .conversation import models as chat_models
    from .runtime import models as runtime_models

    return (
        getattr(core_models, class_name, None)
        or getattr(chat_models, class_name, None)
        or getattr(runtime_models, class_name, None)
    )


def pick_node_type(*, graph_kind: str, metadata: dict[str, Any], fallback: Type):
    class_name = metadata.get("_class_name")
    if isinstance(class_name, str) and class_name:
        cls = _resolve_class_name(class_name)
        if cls is not None:
            return cls

    entity_type = metadata.get("entity_type")
    if entity_type == "workflow_checkpoint" and graph_kind == "workflow":
        from .conversation.models import WorkflowCheckpointNode

        return WorkflowCheckpointNode
    return fallback


def pick_edge_type(*, metadata: dict[str, Any], fallback: Type):
    class_name = metadata.get("_class_name")
    if isinstance(class_name, str) and class_name:
        cls = _resolve_class_name(class_name)
        if cls is not None:
            return cls
    return fallback
