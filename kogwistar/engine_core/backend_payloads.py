"""Node and edge payload serialization for storage backend projections."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ..json_types import JsonObject, JsonValue, decode_json_object
from .models import Edge, Node, PureChromaEdge, PureChromaNode


def decode_backend_ids_response(payload: object) -> list[str] | None:
    """Decode the optional string ID list from a backend collection response.

    Backend responses cross an adapter boundary and remain ``object`` until
    their shape is checked here. ``None`` distinguishes malformed/missing IDs
    from a valid empty result.
    """

    if not isinstance(payload, Mapping):
        return None
    raw_ids: object = payload.get("ids")
    if not isinstance(raw_ids, list):
        return None
    return [item for item in raw_ids if isinstance(item, str)]


def strip_none(values: dict[str, JsonValue | None]) -> JsonObject:
    return {key: value for key, value in values.items() if value is not None}


def json_or_none(value: object) -> str | None:
    return None if value is None else json.dumps(value)


def node_doc_and_meta(
    node: Node | PureChromaNode,
) -> tuple[str, JsonObject]:
    """Serialize a node while retaining a separate, unmodified metadata map."""

    document = node.model_dump_json(
        field_mode="backend", exclude=["embedding", "metadata"]
    )
    metadata = dict(node.metadata)
    metadata.update(
        strip_none(
            {
                "doc_id": node.doc_id,
                "label": node.label,
                "type": node.type,
                "summary": node.summary,
                "domain_id": node.domain_id,
                "canonical_entity_id": node.canonical_entity_id,
                "properties": json_or_none(node.properties),
            }
        )
    )
    metadata.update(
        decode_json_object(
            node.get_extra_update(), label="node backend metadata update"
        )
    )
    mentions = node.mentions if isinstance(node, Node) else None
    if mentions is not None:
        metadata["mentions"] = json_or_none(
            [mention.model_dump(field_mode="backend") for mention in mentions]
        )
    return document, metadata


def edge_doc_and_meta(
    edge: Edge | PureChromaEdge,
) -> tuple[str, JsonObject]:
    """Serialize an edge without dropping backend-filterable metadata."""

    document = edge.model_dump_json(field_mode="backend")
    metadata = dict(edge.metadata or {})
    metadata.update(
        strip_none(
            {
                "doc_id": edge.doc_id,
                "relation": edge.relation,
                "source_ids": json_or_none(edge.source_ids),
                "target_ids": json_or_none(edge.target_ids),
                "type": edge.type,
                "summary": edge.summary,
                "domain_id": edge.domain_id,
                "canonical_entity_id": edge.canonical_entity_id,
                "properties": json_or_none(edge.properties),
            }
        )
    )
    mentions = edge.mentions if isinstance(edge, Edge) else None
    if mentions is not None:
        metadata["mentions"] = json_or_none(
            [mention.model_dump(field_mode="backend") for mention in mentions]
        )
    return document, metadata
