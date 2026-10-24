"""Python-side construction of the versioned Rust graph-model envelope.

This module is an opt-in compatibility seam. It does not alter persisted graph
payloads or select a Rust implementation. Unknown model classes fail closed so
that a new Python subclass cannot silently receive the wrong Rust contract.
"""

from __future__ import annotations

from typing import Literal, TypedDict, cast

from pydantic import BaseModel

from kogwistar.engine_core.graph_slice_manifest import build_slice_manifest

GraphModelClassName = Literal[
    "node",
    "edge",
    "acl_node",
    "acl_edge",
    "conversation_node",
    "conversation_edge",
    "workflow_node",
    "workflow_edge",
    "workflow_run_node",
    "workflow_step_exec_node",
    "workflow_checkpoint_node",
    "workflow_completed_node",
    "workflow_cancelled_node",
    "workflow_failed_node",
    "workflow_runtime_edge",
    "pure_chroma_node",
    "pure_chroma_edge",
    "doc_node",
    "llm_node",
    "llm_edge",
    "llm_node_extraction",
    "llm_edge_extraction",
    "flattened_llm_node",
    "flattened_llm_edge",
    "assoc_flattened_llm_node",
    "assoc_flattened_llm_edge",
]

GraphWireFieldMode = Literal["backend", "llm", "llm_in"]
GRAPH_MODEL_ENVELOPE_VERSION = 1


class GraphModelEnvelopeDict(TypedDict):
    schema_version: int
    model_class: GraphModelClassName
    payload: dict[str, object]


class GraphModelSliceEnvelopeDict(GraphModelEnvelopeDict):
    slice: dict[str, object]


class GraphModelAdapterError(ValueError):
    """Raised when a Python model cannot be mapped without guessing."""


_MODEL_CLASS_BY_NAME: dict[str, GraphModelClassName] = {
    "Node": "node",
    "Edge": "edge",
    "ACLNode": "acl_node",
    "ACLEdge": "acl_edge",
    "ConversationNode": "conversation_node",
    "ConversationEdge": "conversation_edge",
    "WorkflowNode": "workflow_node",
    "WorkflowEdge": "workflow_edge",
    "WorkflowRunNode": "workflow_run_node",
    "WorkflowStepExecNode": "workflow_step_exec_node",
    "WorkflowCheckpointNode": "workflow_checkpoint_node",
    "WorkflowCompletedNode": "workflow_completed_node",
    "WorkflowCancelledNode": "workflow_cancelled_node",
    "WorkflowFailedNode": "workflow_failed_node",
    "WorkflowRuntimeEdge": "workflow_runtime_edge",
    "PureChromaNode": "pure_chroma_node",
    "PureChromaEdge": "pure_chroma_edge",
    "DocNode": "doc_node",
    "LLMNode": "llm_node",
    "LLMEdge": "llm_edge",
    "LLMNodeExtraction": "llm_node_extraction",
    "LLMEdgeExtraction": "llm_edge_extraction",
    "FlattenedLLMNode": "flattened_llm_node",
    "FlattenedLLMEdge": "flattened_llm_edge",
    "AssocFlattenedLLMNode": "assoc_flattened_llm_node",
    "AssocFlattenedLLMEdge": "assoc_flattened_llm_edge",
}

# Public for seam attestation and tooling; callers must treat the values as
# contract identifiers, not as authorization capabilities.
SUPPORTED_GRAPH_MODEL_CLASSES: tuple[GraphModelClassName, ...] = tuple(
    _MODEL_CLASS_BY_NAME.values()
)


def graph_model_class_for(model: BaseModel) -> GraphModelClassName:
    """Return the explicit Rust envelope class for a supported Python model."""

    model_class = _MODEL_CLASS_BY_NAME.get(type(model).__name__)
    if model_class is None:
        raise GraphModelAdapterError(
            "unsupported graph model class; add an explicit compatibility mapping: "
            f"{type(model).__module__}.{type(model).__qualname__}"
        )
    return model_class


def build_graph_model_envelope(
    model: BaseModel,
    *,
    field_mode: GraphWireFieldMode = "backend",
) -> GraphModelEnvelopeDict:
    """Build a lossless, versioned envelope without changing the model.

    The caller owns the mode choice. Persistent graph models normally use
    ``backend``; prompt-only models may explicitly request ``llm`` or
    ``llm_in``. The returned dictionary is new and can be handed to a Rust
    bridge without exposing the Python object itself.
    """

    model_class = graph_model_class_for(model)
    try:
        payload = model.model_dump(field_mode=field_mode, dump_format="json")
    except (AttributeError, TypeError, ValueError) as exc:
        raise GraphModelAdapterError(
            "graph model does not support the versioned field-mode wire contract"
        ) from exc
    if not isinstance(payload, dict):
        raise GraphModelAdapterError("graph model payload must serialize as an object")
    return {
        "schema_version": GRAPH_MODEL_ENVELOPE_VERSION,
        "model_class": model_class,
        "payload": cast(dict[str, object], payload),
    }


def build_graph_model_slice_envelope(
    model: BaseModel,
    *,
    field_mode: GraphWireFieldMode,
) -> GraphModelSliceEnvelopeDict:
    """Build a canonical envelope plus an explicit projection declaration."""

    envelope = build_graph_model_envelope(model, field_mode="backend")
    envelope["slice"] = {
        "schema_version": 1,
        "mode": field_mode,
        "manifest": build_slice_manifest(type(model)),
    }
    return cast(GraphModelSliceEnvelopeDict, envelope)

