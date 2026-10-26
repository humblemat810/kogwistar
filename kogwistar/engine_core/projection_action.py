"""Versioned durable graph-projection action wire contract.

This decoder is deliberately separate from the existing Python index worker:
it provides a cross-language replay boundary without changing legacy job
execution semantics.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
from dataclasses import dataclass
from typing import Literal, Protocol

from typing_extensions import TypedDict


class ProjectionRecordWire(TypedDict):
    id: str
    document: str | None
    metadata: dict[str, object]
    embedding: list[float] | None


class ProjectionActionWire(TypedDict):
    entity_id: str
    entity_kind: Literal["node", "edge"]
    index_kind: str
    job_id: str
    namespace: str
    operation: Literal["DELETE", "UPSERT"]
    record: ProjectionRecordWire | None
    version: Literal[1]


@dataclass(frozen=True)
class ProjectionJobIdentity:
    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    index_kind: str
    op: str


@dataclass(frozen=True)
class ProjectionRecord:
    id: str
    document: str | None
    metadata: dict[str, object]
    embedding: tuple[float, ...] | None


@dataclass(frozen=True)
class ProjectionAction:
    version: Literal[1]
    job_id: str
    namespace: str
    entity_kind: Literal["node", "edge"]
    entity_id: str
    index_kind: str
    operation: Literal["DELETE", "UPSERT"]
    record: ProjectionRecord | None


class ProjectionGraphStore(Protocol):
    """Graph adapter; raise on write errors and treat delete-missing as success."""

    def upsert_graph_record(self, namespace: str, record: ProjectionRecord) -> object: ...

    def delete_graph_record(self, namespace: str, entity_id: str) -> object: ...


class AsyncProjectionGraphStore(Protocol):
    """Nonblocking Python execution seam for async workers."""

    async def upsert_graph_record(
        self, namespace: str, record: ProjectionRecord
    ) -> object: ...

    async def delete_graph_record(self, namespace: str, entity_id: str) -> object: ...


class ProjectionActionError(ValueError):
    """Malformed, mismatched, or unverifiable durable projection action."""

_ACTION_FIELDS = frozenset(
    {
        "version",
        "job_id",
        "namespace",
        "entity_kind",
        "entity_id",
        "index_kind",
        "operation",
        "record",
    }
)
_RECORD_FIELDS = frozenset({"id", "document", "metadata", "embedding"})


def _fail(message: str) -> ProjectionActionError:
    return ProjectionActionError(message)


def _record(value: object, job: ProjectionJobIdentity) -> ProjectionRecord:
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise _fail("projection record has missing or unknown fields")
    record_id = value["id"]
    if not isinstance(record_id, str) or not record_id or record_id != job.entity_id:
        raise _fail("projection record id differs from its queue job")
    document = value["document"]
    if document is not None and not isinstance(document, str):
        raise _fail("projection document must be string or null")
    metadata = value["metadata"]
    if not isinstance(metadata, dict) or "_kogwistar_namespace" in metadata:
        raise _fail("projection metadata is invalid or overrides namespace")

    embedding_value = value["embedding"]
    embedding: tuple[float, ...] | None
    if embedding_value is None:
        embedding = None
    elif isinstance(embedding_value, list):
        components: list[float] = []
        for component in embedding_value:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise _fail("embedding components must be numbers")
            number = float(component)
            try:
                as_f32 = struct.unpack("!f", struct.pack("!f", number))[0]
            except OverflowError as exc:
                raise _fail("embedding components must fit finite f32 values") from exc
            if not math.isfinite(as_f32):
                raise _fail("embedding components must be finite f32 values")
            components.append(as_f32)
        embedding = tuple(components)
    else:
        raise _fail("embedding must be an array or null")
    return ProjectionRecord(record_id, document, dict(metadata), embedding)


def decode_projection_action(
    job: ProjectionJobIdentity,
    result_json: str,
    result_sha256: str | None,
) -> ProjectionAction:
    """Validate a persisted action, its digest, and its binding to a queue job."""
    if not isinstance(result_json, str):
        raise _fail("projection action must be serialized JSON")
    if not isinstance(result_sha256, str) or not result_sha256:
        raise _fail("accepted projection action is missing its SHA-256 digest")
    actual_digest = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual_digest, result_sha256.lower()):
        raise _fail("accepted projection action has an invalid SHA-256 digest")
    try:
        value = json.loads(result_json)
    except (TypeError, ValueError) as exc:
        raise _fail("invalid projection action JSON") from exc
    if not isinstance(value, dict) or set(value) != _ACTION_FIELDS:
        raise _fail("projection action has missing or unknown fields")
    if (
        not job.namespace.strip()
        or job.entity_kind not in {"node", "edge"}
        or not job.entity_id
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["job_id"] != job.job_id
        or value["namespace"] != job.namespace
        or value["entity_kind"] != job.entity_kind
        or value["entity_id"] != job.entity_id
        or value["index_kind"] != job.index_kind
    ):
        raise _fail("projection action identity differs from its claimed queue job")

    operation = value["operation"]
    job_operation = job.op.upper()
    if job_operation == "DELETE" and operation == "DELETE":
        if value["record"] is not None:
            raise _fail("delete action must not contain a record")
        record = None
    elif job_operation in {"UPSERT", "ADD", "REPLACE"} and operation == "UPSERT":
        record = _record(value["record"], job)
    else:
        raise _fail("projection operation is unsupported or differs from its queue job")

    return ProjectionAction(
        version=1,
        job_id=job.job_id,
        namespace=job.namespace,
        entity_kind=job.entity_kind,  # type: ignore[arg-type]
        entity_id=job.entity_id,
        index_kind=job.index_kind,
        operation=operation,  # type: ignore[arg-type]
        record=record,
    )


def is_projection_action_payload(result_json: str) -> bool:
    """Recognize projection actions and reject malformed accepted-result JSON.

    Legacy accepted results are JSON objects too. Invalid JSON must not be
    mistaken for an opaque legacy result and acknowledged without projection.
    """
    try:
        value = json.loads(result_json)
    except (TypeError, ValueError):
        raise _fail("accepted index-job result is not valid JSON") from None
    if not isinstance(value, dict):
        raise _fail("accepted index-job result must be a JSON object")
    return (
        "operation" in value
        or {"job_id", "entity_kind"}.issubset(value)
        or {"entity_id", "index_kind", "record"}.issubset(value)
    )


def apply_projection_action(
    job: ProjectionJobIdentity,
    result_json: str,
    result_sha256: str | None,
    graph: ProjectionGraphStore,
) -> ProjectionAction:
    """Validate and apply a durable action using a Python graph-store adapter."""
    action = decode_projection_action(job, result_json, result_sha256)
    if action.operation == "DELETE":
        graph.delete_graph_record(action.namespace, action.entity_id)
    else:
        assert action.record is not None
        applied = graph.upsert_graph_record(action.namespace, action.record)
        if applied is False:
            raise _fail("graph store rejected projection upsert")
    return action


async def apply_projection_action_async(
    job: ProjectionJobIdentity,
    result_json: str,
    result_sha256: str | None,
    graph: AsyncProjectionGraphStore,
) -> ProjectionAction:
    """Async counterpart that does not hide blocking graph writes."""
    action = decode_projection_action(job, result_json, result_sha256)
    if action.operation == "DELETE":
        await graph.delete_graph_record(action.namespace, action.entity_id)
    else:
        assert action.record is not None
        applied = await graph.upsert_graph_record(action.namespace, action.record)
        if applied is False:
            raise _fail("graph store rejected projection upsert")
    return action
