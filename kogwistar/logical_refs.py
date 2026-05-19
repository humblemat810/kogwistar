from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from kogwistar.id_provider import stable_id


TargetKind = Literal["node", "edge", "artifact"]


@dataclass(frozen=True, slots=True)
class LogicalRef:
    target_namespace: str
    target_kind: TargetKind
    target_id: str


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _collection_to_kind(collection: str) -> TargetKind | None:
    collection = str(collection or "").strip().lower()
    if collection == "nodes":
        return "node"
    if collection == "edges":
        return "edge"
    if collection == "artifacts":
        return "artifact"
    return None


def logical_ref_id(
    *,
    scope: str,
    pointer_kind: str,
    logical_ref: LogicalRef,
) -> str:
    return str(
        stable_id(
            "ptr",
            str(scope or ""),
            str(pointer_kind or ""),
            logical_ref.target_namespace,
            logical_ref.target_kind,
            logical_ref.target_id,
        )
    )


def logical_ref_from_entity(entity: Any) -> LogicalRef | None:
    data = _as_mapping(entity)
    if data is None and hasattr(entity, "model_dump"):
        data = _as_mapping(entity.model_dump(field_mode="backend"))  # type: ignore[attr-defined]
    if data is None and hasattr(entity, "dict"):
        data = _as_mapping(entity.dict())  # type: ignore[attr-defined]
    if data is None:
        return None

    properties = _as_mapping(data.get("properties")) or {}
    metadata = _as_mapping(data.get("metadata")) or {}
    entity_type = str(data.get("type") or "").strip().lower()
    is_explicit_pointer = entity_type == "reference_pointer"
    is_edge_proxy = bool(properties.get("is_pointer") or metadata.get("is_pointer"))

    if not is_explicit_pointer and not is_edge_proxy:
        return None

    target_namespace = _first_text(
        properties.get("target_namespace"),
        metadata.get("target_namespace"),
    )
    target_kind = _first_text(
        properties.get("target_kind"),
        metadata.get("target_kind"),
        _collection_to_kind(
            str(
                properties.get("refers_to_collection")
                or metadata.get("refers_to_collection")
                or ""
            )
        ),
    )
    target_id = _first_text(
        properties.get("target_id"),
        properties.get("refers_to_id"),
        properties.get("refers_to_entity_id"),
        metadata.get("target_id"),
        metadata.get("refers_to_id"),
        metadata.get("refers_to_entity_id"),
    )
    if not target_namespace or not target_kind or not target_id:
        return None
    return LogicalRef(
        target_namespace=target_namespace,
        target_kind=target_kind,  # type: ignore[arg-type]
        target_id=target_id,
    )


def is_reference_artifact(entity: Any) -> bool:
    return logical_ref_from_entity(entity) is not None


def build_reference_node_payload(
    *,
    logical_ref: LogicalRef,
    pointer_kind: str,
    label: str,
    summary: str,
    pointer_id: str | None = None,
    graph_space: str | None = None,
    extra_properties: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "is_pointer": True,
        "pointer_mode": "live",
        "pointer_kind": pointer_kind,
        "target_namespace": logical_ref.target_namespace,
        "target_kind": logical_ref.target_kind,
        "target_id": logical_ref.target_id,
        "refers_to_collection": f"{logical_ref.target_kind}s",
        "refers_to_id": logical_ref.target_id,
        "entity_type": "reference_pointer",
    }
    if extra_properties:
        properties.update(extra_properties)

    metadata: dict[str, Any] = {
        "entity_type": "reference_pointer",
        "is_pointer": True,
        "pointer_mode": "live",
        "pointer_kind": pointer_kind,
        "target_namespace": logical_ref.target_namespace,
        "target_kind": logical_ref.target_kind,
        "target_id": logical_ref.target_id,
        "refers_to_collection": f"{logical_ref.target_kind}s",
        "refers_to_id": logical_ref.target_id,
    }
    if graph_space:
        metadata["graph_space"] = graph_space
    if extra_metadata:
        metadata.update(extra_metadata)

    payload: dict[str, Any] = {
        "label": label,
        "type": "reference_pointer",
        "summary": summary,
        "properties": properties,
        "metadata": metadata,
    }
    if pointer_id is not None:
        payload["id"] = pointer_id
    return payload


def build_reference_edge_payload(
    *,
    logical_ref: LogicalRef,
    pointer_kind: str,
    source_ids: list[str],
    target_ids: list[str],
    relation: str,
    label: str,
    summary: str,
    pointer_id: str | None = None,
    graph_space: str | None = None,
    extra_properties: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "is_pointer": True,
        "pointer_mode": "live",
        "pointer_kind": pointer_kind,
        "target_namespace": logical_ref.target_namespace,
        "target_kind": logical_ref.target_kind,
        "target_id": logical_ref.target_id,
        "refers_to_collection": f"{logical_ref.target_kind}s",
        "refers_to_id": logical_ref.target_id,
        "entity_type": "reference_pointer_edge",
    }
    if extra_properties:
        properties.update(extra_properties)

    metadata: dict[str, Any] = {
        "entity_type": "reference_pointer_edge",
        "is_pointer": True,
        "pointer_mode": "live",
        "pointer_kind": pointer_kind,
        "target_namespace": logical_ref.target_namespace,
        "target_kind": logical_ref.target_kind,
        "target_id": logical_ref.target_id,
        "refers_to_collection": f"{logical_ref.target_kind}s",
        "refers_to_id": logical_ref.target_id,
    }
    if graph_space:
        metadata["graph_space"] = graph_space
    if extra_metadata:
        metadata.update(extra_metadata)

    payload: dict[str, Any] = {
        "source_ids": list(source_ids),
        "target_ids": list(target_ids),
        "relation": relation,
        "label": label,
        "type": "relationship",
        "summary": summary,
        "properties": properties,
        "metadata": metadata,
    }
    if pointer_id is not None:
        payload["id"] = pointer_id
    return payload


__all__ = [
    "LogicalRef",
    "TargetKind",
    "build_reference_edge_payload",
    "build_reference_node_payload",
    "is_reference_artifact",
    "logical_ref_from_entity",
    "logical_ref_id",
]
