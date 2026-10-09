from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar, cast

from .graph_kinds import KIND_CHAT, KIND_FLOW
from .json_types import JsonValue

if TYPE_CHECKING:
    from .engine_core.models import Edge, Node


_TNode = TypeVar("_TNode", bound="Node")
_TEdge = TypeVar("_TEdge", bound="Edge")


def default_node_type_for_graph_kind(graph_kind: str) -> type[Node]:
    from .engine_core.models import Node

    if graph_kind == KIND_CHAT:
        from .conversation.models import ConversationNode

        return ConversationNode
    if graph_kind == KIND_FLOW:
        from .runtime.models import WorkflowNode

        return WorkflowNode
    return Node


def default_edge_type_for_graph_kind(graph_kind: str) -> type[Edge]:
    from .engine_core.models import Edge

    if graph_kind == KIND_CHAT:
        from .conversation.models import ConversationEdge

        return ConversationEdge
    if graph_kind == KIND_FLOW:
        from .runtime.models import WorkflowEdge

        return WorkflowEdge
    return Edge


def _resolve_class_name(class_name: str) -> object | None:
    from .acl import models as acl_models
    from .conversation import models as chat_models
    from .engine_core import models as core_models
    from .runtime import models as runtime_models

    return (
        getattr(core_models, class_name, None)
        or getattr(chat_models, class_name, None)
        or getattr(runtime_models, class_name, None)
        or getattr(acl_models, class_name, None)
    )


def pick_node_type(
    *, graph_kind: str, metadata: Mapping[str, JsonValue], fallback: type[_TNode]
) -> type[_TNode]:
    class_name = metadata.get("_class_name")
    if isinstance(class_name, str) and class_name:
        cls = _resolve_class_name(class_name)
        if cls is not None:
            return cast(type[_TNode], cls)

    entity_type = metadata.get("entity_type")
    if entity_type == "workflow_checkpoint" and graph_kind == "workflow":
        from .runtime.models import WorkflowCheckpointNode

        return cast(type[_TNode], WorkflowCheckpointNode)
    return fallback


def pick_edge_type(
    *, metadata: Mapping[str, JsonValue], fallback: type[_TEdge]
) -> type[_TEdge]:
    class_name = metadata.get("_class_name")
    if isinstance(class_name, str) and class_name:
        cls = _resolve_class_name(class_name)
        if cls is not None:
            return cast(type[_TEdge], cls)

    # Older rows can contain ordinary relationship edges in a workflow graph
    # without a class marker.  Do not validate those rows as WorkflowEdge:
    # WorkflowEdgeMetadata requires workflow_id, while base relationship rows
    # intentionally do not carry workflow fields.
    if fallback.__name__ == "WorkflowEdge" and not _looks_like_workflow_edge(metadata):
        from .engine_core.models import Edge

        return cast(type[_TEdge], Edge)
    return fallback


def _looks_like_workflow_edge(metadata: Mapping[str, JsonValue]) -> bool:
    """Return whether metadata contains the workflow-edge contract."""

    if metadata.get("entity_type") == "workflow_edge":
        return True
    return "workflow_id" in metadata or any(
        key.startswith("wf_") for key in metadata
    )
