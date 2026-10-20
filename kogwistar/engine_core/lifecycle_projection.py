"""Typed metadata-only lifecycle projection updates for storage backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from ..json_types import JsonObject, decode_json_object
from .async_compat import run_awaitable_blocking

LifecycleKind = Literal["node", "edge"]


class LifecycleProjectionBackend(Protocol):
    """Narrow backend methods required for a metadata-only lifecycle patch."""

    def node_get(self, **kwargs: object) -> object: ...

    def node_update(self, **kwargs: object) -> object: ...

    def edge_get(self, **kwargs: object) -> object: ...

    def edge_update(self, **kwargs: object) -> object: ...


def decode_lifecycle_get_result(value: object) -> tuple[list[str], JsonObject]:
    """Validate a backend get response before lifecycle metadata is merged."""

    response = decode_json_object(value, label="lifecycle backend get result")
    raw_ids = response.get("ids")
    if raw_ids is None:
        ids: list[str] = []
    elif isinstance(raw_ids, list) and all(isinstance(item, str) for item in raw_ids):
        ids = [item for item in raw_ids if isinstance(item, str)]
    else:
        raise TypeError("lifecycle backend get result ids must be a list of strings")

    raw_metadatas = response.get("metadatas")
    if raw_metadatas is None:
        return ids, {}
    if not isinstance(raw_metadatas, list):
        raise TypeError("lifecycle backend get result metadatas must be a list")
    if not raw_metadatas or raw_metadatas[0] is None:
        return ids, {}
    return ids, decode_json_object(
        raw_metadatas[0], label="lifecycle backend metadata"
    )


def backend_update_record_lifecycle(
    *,
    backend: LifecycleProjectionBackend,
    kind: LifecycleKind,
    record_id: str,
    lifecycle_patch: Mapping[str, object],
) -> bool:
    """Update lifecycle metadata without reading or rewriting vector data."""

    if kind == "node":
        get_result = backend.node_get(ids=[record_id], include=["metadatas"])
        update = backend.node_update
    elif kind == "edge":
        get_result = backend.edge_get(ids=[record_id], include=["metadatas"])
        update = backend.edge_update
    else:
        raise ValueError(f"unsupported lifecycle record kind: {kind}")

    ids, metadata = decode_lifecycle_get_result(
        run_awaitable_blocking(get_result)
    )
    if not ids:
        return False
    patch = decode_json_object(lifecycle_patch, label="lifecycle patch")
    merged_metadata = {**metadata, **patch}
    run_awaitable_blocking(
        update(ids=[record_id], metadatas=[merged_metadata])
    )
    return True
