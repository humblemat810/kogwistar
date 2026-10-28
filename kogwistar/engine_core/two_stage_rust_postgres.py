"""Rust-authority PostgreSQL two-stage projection adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol, cast

from kogwistar.json_types import JsonObject, JsonValue

from .embedding_factory import EmbeddingFunctionLike
from .canonical_events import CanonicalEntityRevision
from .models import Edge, Node
from ..utils.embedding_vectors import normalize_embedding_vector
from .storage_backend import TwoStageProjectionCapability


class _RustProjectionMeta(Protocol):
    def upsert_graph_projection(self, **values: JsonValue) -> None: ...

    def graph_projection_records(self, **values: JsonValue) -> list[JsonObject]: ...


class _RustProjectionCollection(Protocol):
    name: str


class _RustProjectionBackend(Protocol):
    embedding_dim: int
    nodes: _RustProjectionCollection
    edges: _RustProjectionCollection


class _RustProjectionIndexing(Protocol):
    def enqueue_index_job(self, **values: JsonValue) -> str: ...

    def canonical_revision_payload(self, **values: JsonValue) -> str: ...

    def canonical_entity_revision(
        self, **values: JsonValue
    ) -> CanonicalEntityRevision | None: ...


class _RustProjectionWriter(Protocol):
    def node_doc_and_meta(self, node: Node) -> tuple[str, JsonObject]: ...

    def enrich_edge_meta(self, edge: Edge) -> JsonObject: ...


class _RustEmbeddingHelper(Protocol):
    def iterative_defensive_emb(self, document: str) -> Sequence[float]: ...


class _RustProjectionEngine(Protocol):
    namespace: str
    backend: _RustProjectionBackend
    indexing: _RustProjectionIndexing
    write: _RustProjectionWriter
    embed: _RustEmbeddingHelper
    _ef: EmbeddingFunctionLike


def rust_postgres_two_stage_capability() -> TwoStageProjectionCapability:
    return TwoStageProjectionCapability(
        supports_two_stage=True,
        canonical_event_replay=True,
        canonical_read=True,
        stage1_strategy="none",
        stage2_semantic_projection=True,
        revision_gated_promotion=True,
        semantic_readiness_gate=True,
        delete_reconciliation=True,
        atomic_promotion="same_store",
        reason="Rust PostgreSQL authority keeps nullable vector projection and promotion in one native UoW",
    )


class RustPostgresTwoStageProjectionAdapter:
    """Use native graph projection operations; no Python PG writer is involved."""

    def __init__(self, engine: _RustProjectionEngine, meta: _RustProjectionMeta) -> None:
        self.engine = engine
        self.meta = meta

    def _table(self, entity_kind: str) -> str:
        if entity_kind == "node":
            return self.engine.backend.nodes.name
        if entity_kind == "edge":
            return self.engine.backend.edges.name
        raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")

    def _namespace(self) -> str:
        return str(getattr(self.engine, "namespace", "default"))

    def _fingerprint(self, entity_kind: str, entity_id: str) -> str:
        payload = self.engine.indexing.canonical_revision_payload(
            entity_kind=entity_kind, entity_id=entity_id
        )
        return str(json.loads(payload).get("source_fingerprint") or "")

    def _upsert(
        self,
        *,
        entity_kind: str,
        entity: Node | Edge,
        document: str,
        metadata: JsonObject,
    ) -> None:
        metadata = dict(metadata or {})
        # Admission runs inside the parent's native UoW. Do not re-read the
        # event stream here: the current event is already the authority and
        # the job captures its fingerprint after the admission transaction.
        self.meta.upsert_graph_projection(
            namespace=self._namespace(),
            workspace_id=metadata.get("workspace_id"),
            graph_space=metadata.get("graph_space"),
            table=self._table(entity_kind),
            record={
                "id": entity.safe_get_id(),
                "document": str(document),
                "metadata": metadata,
                "embedding": None,
            },
            embedding_dim=int(self.engine.backend.embedding_dim),
        )

    def enqueue_embedding_job(self, *, entity_kind: str, entity_id: str, op: str) -> None:
        self.engine.indexing.enqueue_index_job(
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind="node_embedding",
            op=op,
            payload_json=self.engine.indexing.canonical_revision_payload(
                entity_kind=entity_kind, entity_id=entity_id
            ),
        )

    def _clear_embedding_projection(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        only_if_inactive: bool = False,
    ) -> None:
        """Converge vector state without deleting canonical graph rows."""
        with self.engine.uow():
            records = self.meta.graph_projection_records(
                namespace=self._namespace(),
                workspace_id=None,
                graph_space=None,
                table=self._table(entity_kind),
                ids=[entity_id],
                metadata={},
                limit=1,
                for_update=True,
            )
            if not records:
                return

            current = self.engine.indexing.canonical_entity_revision(
                entity_kind=entity_kind, entity_id=entity_id
            )
            record = records[0]
            if current is not None and current.state == "active":
                if only_if_inactive:
                    return
                raw_metadata = record.get("metadata")
                metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                embedding = record.get("embedding")
                current_fingerprint = self._fingerprint(entity_kind, entity_id)
                projection_is_current = (
                    metadata.get("_kogwistar_stage2_ready") is True
                    and metadata.get("_kogwistar_source_fingerprint") == current_fingerprint
                    and isinstance(embedding, Sequence)
                    and not isinstance(embedding, (str, bytes))
                    and len(embedding) > 0
                )
                if projection_is_current:
                    return
                # DELETE may have absorbed a later UPSERT while retaining its
                # operation. Canonical active state wins; queue current work.
                self.enqueue_embedding_job(
                    entity_kind=entity_kind, entity_id=entity_id, op="UPSERT"
                )
                return

            raw_metadata = record.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata["_kogwistar_stage2_ready"] = False
            metadata.pop("_kogwistar_source_fingerprint", None)
            self.meta.upsert_graph_projection(
                namespace=self._namespace(),
                workspace_id=metadata.get("workspace_id"),
                graph_space=metadata.get("graph_space"),
                table=self._table(entity_kind),
                record={
                    "id": entity_id,
                    "document": str(record.get("document") or ""),
                    "metadata": metadata,
                    "embedding": None,
                },
                embedding_dim=int(self.engine.backend.embedding_dim),
            )

    def add_node(self, node: Node, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        document, metadata = self.engine.write.node_doc_and_meta(node)
        self._upsert(entity_kind="node", entity=node, document=document, metadata=metadata)

    def add_edge(self, edge: Edge, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        document = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        self._upsert(
            entity_kind="edge",
            entity=edge,
            document=document,
            metadata=self.engine.write.enrich_edge_meta(edge),
        )

    def apply_embedding_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str | None,
    ) -> None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-2 entity kind: {entity_kind!r}")
        if op.upper() == "DELETE":
            self._clear_embedding_projection(
                entity_kind=entity_kind,
                entity_id=entity_id,
            )
            return
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        current = self.engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        if current is None or current.state != "active":
            self._clear_embedding_projection(
                entity_kind=entity_kind,
                entity_id=entity_id,
                only_if_inactive=True,
            )
            return
        if expected and expected != self._fingerprint(entity_kind, entity_id):
            return
        records = self.meta.graph_projection_records(
            namespace=self._namespace(),
            workspace_id=None,
            graph_space=None,
            table=self._table(entity_kind),
            ids=[entity_id],
            metadata={},
            limit=1,
        )
        if not records:
            raise RuntimeError("current Rust Stage-1 graph projection is missing")
        record = records[0]
        document = str(record.get("document") or "")
        embedding = cast(
            list[float], self.engine.embed.iterative_defensive_emb(document)
        )
        self._promote_record(
            entity_kind=entity_kind,
            entity_id=entity_id,
            record=record,
            embedding=embedding,
            expected=expected,
        )

    def _promote_record(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        record: JsonObject,
        embedding: list[float],
        expected: str,
    ) -> None:
        # Embedding runs before this point. Lock the projection row first, then
        # revalidate its source revision and payload in the same native UoW as
        # the promotion. This prevents a result that passed an earlier check
        # from overwriting a newer graph mutation.
        with self.engine.uow():
            records = self.meta.graph_projection_records(
                namespace=self._namespace(),
                workspace_id=None,
                graph_space=None,
                table=self._table(entity_kind),
                ids=[entity_id],
                metadata={},
                limit=1,
                for_update=True,
            )
            if not records:
                raise RuntimeError("current Rust graph projection is missing")
            current_record = records[0]
            if (
                current_record.get("document") != record.get("document")
                or current_record.get("metadata") != record.get("metadata")
            ):
                return

            current = self.engine.indexing.canonical_entity_revision(
                entity_kind=entity_kind, entity_id=entity_id
            )
            if current is None or current.state != "active":
                return
            current_fingerprint = self._fingerprint(entity_kind, entity_id)
            if expected and expected != current_fingerprint:
                return

            raw_metadata = current_record.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            stored_fingerprint = metadata.get("_kogwistar_source_fingerprint")
            if expected and stored_fingerprint and stored_fingerprint != expected:
                return
            document = str(current_record.get("document") or "")
            metadata["_kogwistar_stage2_ready"] = True
            metadata["_kogwistar_source_fingerprint"] = current_fingerprint
            self.meta.upsert_graph_projection(
                namespace=self._namespace(),
                workspace_id=metadata.get("workspace_id"),
                graph_space=metadata.get("graph_space"),
                table=self._table(entity_kind),
                record=cast(JsonValue, {
                    "id": entity_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": normalize_embedding_vector(embedding, allow_none=False),
                }),
                embedding_dim=int(self.engine.backend.embedding_dim),
            )

    def apply_embedding_jobs_batch(
        self, jobs: Sequence[object]
    ) -> dict[str, BaseException | None]:
        """Batch provider call with independent native promotion outcomes."""
        prepared: list[tuple[str, str, str, JsonObject]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            def value(name: str) -> object:
                if isinstance(job, dict):
                    return job.get(name)
                return getattr(job, name, None)
            job_id = str(value("job_id") or "")
            entity_kind = str(value("entity_kind") or "")
            entity_id = str(value("entity_id") or "")
            try:
                op = str(value("op") or "UPSERT")
                payload_value = value("payload_json")
                payload_json = (
                    str(payload_value) if payload_value is not None else None
                )
                if op.upper() == "DELETE":
                    self._clear_embedding_projection(
                        entity_kind=entity_kind,
                        entity_id=entity_id,
                    )
                    outcomes[job_id] = None
                    continue
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                current = self.engine.indexing.canonical_entity_revision(
                    entity_kind=entity_kind, entity_id=entity_id
                )
                if current is None or current.state != "active":
                    self._clear_embedding_projection(
                        entity_kind=entity_kind,
                        entity_id=entity_id,
                        only_if_inactive=True,
                    )
                    outcomes[job_id] = None
                    continue
                if expected and expected != self._fingerprint(entity_kind, entity_id):
                    outcomes[job_id] = None
                    continue
                records = self.meta.graph_projection_records(
                    namespace=self._namespace(), workspace_id=None, graph_space=None,
                    table=self._table(entity_kind), ids=[entity_id], metadata={}, limit=1,
                )
                if not records:
                    raise RuntimeError("current Rust Stage-1 graph projection is missing")
                prepared.append((job_id, entity_kind, entity_id, records[0]))
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        provider = getattr(self.engine, "_ef", None)
        try:
            if not callable(provider):
                raise RuntimeError("Rust two-stage batch requires an embedding provider")
            raw = provider([str(item[3].get("document") or "") for item in prepared])
            embeddings = list(cast(Sequence[list[float]], raw))
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException as exc:
            for job_id, *_ in prepared:
                outcomes[job_id] = exc
            return outcomes
        for (job_id, entity_kind, entity_id, record), embedding in zip(prepared, embeddings):
            try:
                expected = str(
                    self.engine.indexing.canonical_revision_payload(
                        entity_kind=entity_kind, entity_id=entity_id
                    )
                )
                expected = str(json.loads(expected).get("source_fingerprint") or "")
                self._promote_record(
                    entity_kind=entity_kind, entity_id=entity_id,
                    record=record, embedding=embedding, expected=expected,
                )
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

__all__ = [
    "RustPostgresTwoStageProjectionAdapter",
    "rust_postgres_two_stage_capability",
]
