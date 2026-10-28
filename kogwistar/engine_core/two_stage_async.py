"""Async two-stage projection adapters for existing backend arrangements."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    nullcontext,
)
from typing import TYPE_CHECKING, Protocol, cast

from ..implementation_selection import (
    transaction_authority_for_selection,
    transaction_authority_scope,
)
from ..json_types import JsonObject
from ..utils.embedding_vectors import normalize_embedding_vector
from .edge_endpoint_rows import edge_endpoint_rows
from .models import Edge, Node
from .storage_backend import TwoStageProjectionCapability
from .two_stage_namespace import active_graph_namespace
from .two_stage_rust_postgres import (
    RustPostgresTwoStageProjectionAdapter,
    _RustProjectionEngine,
    _RustProjectionMeta,
)

if TYPE_CHECKING:
    from ..typing_interfaces import WriteLike


class _RevisionLike(Protocol):
    state: str
    revision: int
    payload: JsonObject


class _IndexingLike(Protocol):
    def canonical_revision_payload(self, *, entity_kind: str, entity_id: str) -> str: ...

    def canonical_revision_payloads(
        self, *, entity_kind: str, entity_ids: Sequence[str]
    ) -> Mapping[str, str]: ...

    def canonical_entity_revision(
        self, *, entity_kind: str, entity_id: str
    ) -> _RevisionLike | None: ...

    def enqueue_index_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        payload_json: str,
    ) -> object: ...


class _AsyncCollectionLike(Protocol):
    async def delete(self, **kwargs: object) -> object: ...


class _AsyncBackendLike(Protocol):
    _is_async_engine: bool
    _edge_endpoints_c: _AsyncCollectionLike
    edge_endpoints: _AsyncCollectionLike
    nodes: _AsyncCollectionLike
    edges: _AsyncCollectionLike

    async def stage1_projection_upsert_async(self, **kwargs: object) -> object: ...
    async def stage1_projection_query_async(
        self, **kwargs: object
    ) -> Sequence[Mapping[str, object]]: ...
    async def stage1_projection_delete_async(self, **kwargs: object) -> object: ...
    async def stage1_projection_get_async(
        self, **kwargs: object
    ) -> Mapping[str, object] | None: ...
    async def async_call(self, *args: object, **kwargs: object) -> object: ...
    async def _upsert_async(self, *args: object, **kwargs: object) -> object: ...
    async def _get_flat_async(self, *args: object, **kwargs: object) -> Mapping[str, object]: ...


class _MetaStoreLike(Protocol):
    def __getattr__(self, name: str) -> Callable[..., object]: ...


EmbeddingProviderResult = Sequence[Sequence[float]]
EmbeddingProvider = Callable[
    [list[str]], EmbeddingProviderResult | Awaitable[EmbeddingProviderResult]
]


class _TwoStageEngineLike(Protocol):
    backend: _AsyncBackendLike
    indexing: _IndexingLike
    write: "WriteLike"
    metadata: _MetaStoreLike
    namespace: str
    _ef: EmbeddingProvider


def _job_value(job: object, name: str) -> object:
    if isinstance(job, Mapping):
        return job.get(name)
    return getattr(job, name, None)


def _row_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("projection row must be a mapping")


def _optional_row_mapping(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    return _row_mapping(value)


def _sequence_number(value: object) -> int:
    return value if isinstance(value, int) else 0


def _payload_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _object_list(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


async def _embed(provider: EmbeddingProvider, documents: list[str]) -> list[Sequence[float]]:
    result = provider(documents)
    if inspect.isawaitable(result):
        result = await cast(Awaitable[EmbeddingProviderResult], result)
    return list(result)


def async_transient_two_stage_capability(reason: str) -> TwoStageProjectionCapability:
    return TwoStageProjectionCapability(
        supports_two_stage=True,
        canonical_event_replay=True,
        canonical_read=True,
        stage1_strategy="transient_projection",
        stage1_metadata_query=True,
        stage1_cleanup=True,
        stage2_semantic_projection=True,
        revision_gated_promotion=True,
        semantic_readiness_gate=True,
        delete_reconciliation=True,
        atomic_promotion="eventual_reconcile",
        reason=reason,
    )


class AsyncPostgresTwoStageProjectionAdapter:
    """Use PostgreSQL async SQL primitives; never enter the sync bridge."""

    def __init__(self, engine: object) -> None:
        if not getattr(getattr(engine, "backend", None), "_is_async_engine", False):
            raise ValueError("async PostgreSQL adapter requires an async engine")
        self.engine = cast(_TwoStageEngineLike, engine)

    def _backend(self) -> _AsyncBackendLike:
        return self.engine.backend

    def _namespace(self) -> str:
        return active_graph_namespace(self.engine)

    def _transaction_authority_scope(self) -> AbstractContextManager[None]:
        snapshot = getattr(self.engine, "_implementation_selection_snapshot", None)
        if snapshot is None:
            return nullcontext()
        return transaction_authority_scope(
            transaction_authority_for_selection(snapshot)
        )

    @asynccontextmanager
    async def _backend_transaction(self) -> AsyncIterator[None]:
        """Join the configured async SQL UOW when one exists."""
        uow = getattr(self.engine, "_async_backend_uow", None)
        transaction = getattr(uow, "transaction", None)
        if callable(transaction):
            context = cast(AbstractAsyncContextManager[None], transaction())
            with self._transaction_authority_scope():
                async with context:
                    yield
        else:
            yield

    async def add_node(self, node: Node, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        document, metadata = self.engine.write.node_doc_and_meta(node)
        await self._add(
            entity_kind="node", entity_id=node.safe_get_id(),
            document=document, metadata=metadata,
        )

    async def add_edge(self, edge: Edge, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        await self._add(
            entity_kind="edge", entity_id=edge.safe_get_id(),
            document=edge.model_dump_json(field_mode="backend", exclude=["embedding"]),
            metadata=self.engine.write.enrich_edge_meta(edge),
        )

    async def _add(
        self, *, entity_kind: str, entity_id: str, document: str,
        metadata: Mapping[str, object],
    ) -> None:
        import asyncio
        await self.remove_stage2_or_invalidate(
            entity_kind=entity_kind, entity_id=entity_id
        )
        revision_payload = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind, entity_id=entity_id,
        )
        payload = json.loads(revision_payload)
        revision = await asyncio.to_thread(
            self.engine.indexing.canonical_entity_revision,
            entity_kind=entity_kind, entity_id=entity_id,
        )
        await self._backend().stage1_projection_upsert_async(
            namespace=self._namespace(), entity_kind=entity_kind,
            entity_id=entity_id, document=str(document), metadata=dict(metadata or {}),
            source_fingerprint=str(payload.get("source_fingerprint") or ""),
            revision=int(getattr(revision, "revision", 0) if revision else 0),
        )
        await self._enqueue(entity_kind=entity_kind, entity_id=entity_id, op="UPSERT")

    async def _enqueue(self, *, entity_kind: str, entity_id: str, op: str) -> None:
        # Existing meta stores are sync facades; queue mutation is not a provider
        # or backend operation, and is isolated from the async SQL projection.
        import asyncio
        payload_json = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind, entity_id=entity_id,
        )
        await asyncio.to_thread(
            self.engine.indexing.enqueue_index_job,
            entity_kind=entity_kind, entity_id=entity_id,
            index_kind="node_embedding", op=op, payload_json=payload_json,
        )

    async def stage1_query(self, **kwargs: object) -> list[dict[str, object]]:
        rows = await self._backend().stage1_projection_query_async(
            namespace=self._namespace(), entity_kind=str(kwargs.get("entity_kind") or "node"),
            ids=kwargs.get("ids"), metadata=kwargs.get("metadata"),
            limit=kwargs.get("limit", 200),
        )
        return [
            {
                "key": f"{row['entity_kind']}:{row['entity_id']}",
                "payload": {
                    "id": row["entity_id"],
                    "entity_kind": row["entity_kind"],
                    "document": row["document"],
                    "metadata": _row_mapping(row.get("metadata")),
                    "source_fingerprint": row["source_fingerprint"],
                },
            }
            for row in rows
        ]

    async def remove_stage1(self, *, entity_kind: str, entity_id: str, **_: object) -> None:
        await self._backend().stage1_projection_delete_async(
            namespace=self._namespace(), entity_kind=entity_kind, entity_id=entity_id
        )

    async def remove_stage2_or_invalidate(self, *, entity_kind: str, entity_id: str, **_: object) -> None:
        await getattr(self._backend(), f"_{entity_kind}s_c").delete(ids=[entity_id])
        if entity_kind == "edge":
            await self._backend()._edge_endpoints_c.delete(where={"edge_id": entity_id})

    async def _promote_edge_endpoints(self, document: str) -> None:
        from .models import Edge

        rows = edge_endpoint_rows(Edge.model_validate_json(document))
        if rows:
            await self._backend()._upsert_async(
                self._backend().edge_endpoints,
                ids=[row["id"] for row in rows],
                documents=[json.dumps(row) for row in rows],
                metadatas=rows,
            )

    async def _current(self, entity_kind: str, entity_id: str) -> _RevisionLike | None:
        # Canonical event scanning remains synchronous today. Keep it off the
        # async event loop until the meta store exposes native async reads.
        import asyncio
        return await asyncio.to_thread(
            self.engine.indexing.canonical_entity_revision,
            entity_kind=entity_kind, entity_id=entity_id,
        )

    async def _fingerprint(self, entity_kind: str, entity_id: str) -> str:
        import asyncio
        payload = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind, entity_id=entity_id,
        )
        return str(json.loads(payload).get("source_fingerprint") or "")

    async def apply_embedding_job(
        self, *, entity_kind: str, entity_id: str, op: str,
        payload_json: str | None,
    ) -> None:
        current = await self._current(entity_kind, entity_id)
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        if op.upper() == "DELETE":
            if (
                expected
                and current is not None
                and current.state == "active"
                and expected != await self._fingerprint(entity_kind, entity_id)
            ):
                return
            await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
            await self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
            return
        if current is None or current.state != "active":
            await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
            await self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
            return
        if expected and expected != await self._fingerprint(entity_kind, entity_id):
            return
        row = await self._backend().stage1_projection_get_async(
            namespace=self._namespace(), entity_kind=entity_kind, entity_id=entity_id
        )
        if row is None:
            raise RuntimeError("current PostgreSQL Stage-1 projection is missing")
        document = str(row["document"])
        embedding = normalize_embedding_vector(
            list((await _embed(self.engine._ef, [document]))[0]), allow_none=False
        )
        async with self._backend_transaction():
            locked_row = await self._backend().stage1_projection_get_async(
                namespace=self._namespace(), entity_kind=entity_kind,
                entity_id=entity_id, for_update=True,
            )
            if locked_row is None or any(
                row.get(field) != locked_row.get(field)
                for field in ("document", "metadata", "source_fingerprint", "revision")
            ):
                return
            staged_fingerprint = str(locked_row.get("source_fingerprint") or "")
            if expected and expected != staged_fingerprint:
                return
            document = str(locked_row["document"])
            metadata = _row_mapping(locked_row.get("metadata"))
            metadata["_kogwistar_stage2_ready"] = True
            metadata["_kogwistar_source_fingerprint"] = staged_fingerprint
            await self._backend()._upsert_async(
                getattr(self._backend(), f"{entity_kind}s"), ids=[entity_id],
                documents=[document], metadatas=[metadata], embeddings=[embedding],
            )
            if entity_kind == "edge":
                await self._promote_edge_endpoints(document)
            await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)

    async def promote_stage2(
        self, *, entity_kind: str, entity_id: str, op: str,
        payload_json: str | None,
    ) -> None:
        await self.apply_embedding_job(
            entity_kind=entity_kind, entity_id=entity_id, op=op,
            payload_json=payload_json,
        )

    async def apply_embedding_jobs_batch(
        self, jobs: list[object]
    ) -> dict[str, BaseException | None]:
        prepared: list[tuple[str, str, str, dict[str, object], str]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            job_id = str(_job_value(job, "job_id") or "")
            kind = str(_job_value(job, "entity_kind") or "")
            entity_id = str(_job_value(job, "entity_id") or "")
            payload_json = _payload_text(_job_value(job, "payload_json"))
            op = str(_job_value(job, "op") or "UPSERT")
            try:
                current = await self._current(kind, entity_id)
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                if op.upper() == "DELETE" or current is None or current.state != "active":
                    await self.apply_embedding_job(
                        entity_kind=kind, entity_id=entity_id, op=op,
                        payload_json=payload_json,
                    )
                    outcomes[job_id] = None
                    continue
                if expected and expected != await self._fingerprint(kind, entity_id):
                    outcomes[job_id] = None
                    continue
                row = await self._backend().stage1_projection_get_async(
                    namespace=self._namespace(), entity_kind=kind, entity_id=entity_id
                )
                if row is None:
                    raise RuntimeError("current PostgreSQL Stage-1 projection is missing")
                prepared.append((job_id, kind, entity_id, _row_mapping(row), expected))
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        try:
            embeddings = await _embed(
                self.engine._ef,
                [str(row.get("document") or "") for _, _, _, row, _ in prepared],
            )
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException as exc:
            for job_id, *_ in prepared:
                outcomes[job_id] = exc
            return outcomes
        for (job_id, kind, entity_id, row, expected), embedding in zip(prepared, embeddings):
            try:
                async with self._backend_transaction():
                    locked_row = await self._backend().stage1_projection_get_async(
                        namespace=self._namespace(), entity_kind=kind,
                        entity_id=entity_id, for_update=True,
                    )
                    if locked_row is None or any(
                        row.get(field) != locked_row.get(field)
                        for field in ("document", "metadata", "source_fingerprint", "revision")
                    ):
                        outcomes[job_id] = None
                        continue
                    source_fingerprint = str(locked_row.get("source_fingerprint") or "")
                    if expected and expected != source_fingerprint:
                        outcomes[job_id] = None
                        continue
                    document = str(locked_row["document"])
                    metadata = _row_mapping(locked_row.get("metadata"))
                    metadata["_kogwistar_stage2_ready"] = True
                    metadata["_kogwistar_source_fingerprint"] = source_fingerprint
                    await self._backend()._upsert_async(
                        getattr(self._backend(), f"{kind}s"), ids=[entity_id],
                        documents=[document], metadatas=[metadata],
                        embeddings=[normalize_embedding_vector(embedding, allow_none=False)],
                    )
                    if kind == "edge":
                        await self._promote_edge_endpoints(document)
                    await self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

    async def remove_stage2_or_invalidate_and_cleanup(
        self, *, entity_kind: str, entity_id: str,
    ) -> None:
        await self.remove_stage2_or_invalidate(
            entity_kind=entity_kind, entity_id=entity_id,
        )

    async def reconcile_projection(self, **_: object) -> int:
        removed = 0
        for row in await self.stage1_query(entity_kind="node") + await self.stage1_query(entity_kind="edge"):
            payload = _row_mapping(row.get("payload"))
            entity_kind = str(payload.get("entity_kind") or "node")
            entity_id = str(payload.get("id") or "")
            if not entity_id:
                continue
            current = await self._current(entity_kind, entity_id)
            if current is None or current.state != "active":
                await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
                await self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
                removed += 1
                continue
            stage2 = await self._backend()._get_flat_async(
                getattr(self._backend(), f"{entity_kind}s"),
                ids=[entity_id], where=None,
                include=["documents", "metadatas"], limit=1,
            )
            metadatas = _object_list(stage2.get("metadatas"))
            stage2_metadata = metadatas[0] if metadatas else None
            if (
                _object_list(stage2.get("ids"))
                and isinstance(stage2_metadata, dict)
                and stage2_metadata.get("_kogwistar_source_fingerprint")
                == payload.get("source_fingerprint")
            ):
                if entity_kind == "edge":
                    documents = _object_list(stage2.get("documents"))
                    document = documents[0] if documents else None
                    if document:
                        async with self._backend_transaction():
                            await self._promote_edge_endpoints(str(document))
                            await self.remove_stage1(
                                entity_kind=entity_kind, entity_id=entity_id
                            )
                    else:
                        await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
                else:
                    await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
                removed += 1
        return removed


class AsyncChromaTwoStageProjectionAdapter:
    """SQLite Stage 1 plus direct async Chroma collection operations."""

    def __init__(self, engine: object) -> None:
        self.engine = cast(_TwoStageEngineLike, engine)

    def _namespace(self) -> str:
        return active_graph_namespace(self.engine)

    def _key(self, entity_kind: str, entity_id: str) -> str:
        return f"{entity_kind}:{entity_id}"

    async def _meta(self, method: str, *args: object, **kwargs: object) -> object:
        import asyncio
        return await asyncio.to_thread(getattr(self.engine.metadata, method), *args, **kwargs)

    async def _current(self, entity_kind: str, entity_id: str) -> _RevisionLike | None:
        import asyncio
        return await asyncio.to_thread(
            self.engine.indexing.canonical_entity_revision,
            entity_kind=entity_kind, entity_id=entity_id,
        )

    async def _fingerprint(self, entity_kind: str, entity_id: str) -> str:
        import asyncio
        payload = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind, entity_id=entity_id,
        )
        return str(json.loads(payload).get("source_fingerprint") or "")

    async def add_node(self, node: Node, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        document, metadata = self.engine.write.node_doc_and_meta(node)
        await self._add("node", node.safe_get_id(), document, metadata)

    async def add_edge(self, edge: Edge, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        await self._add("edge", edge.safe_get_id(), edge.model_dump_json(field_mode="backend", exclude=["embedding"]), self.engine.write.enrich_edge_meta(edge))

    async def _add(
        self, entity_kind: str, entity_id: str, document: str,
        metadata: Mapping[str, object],
    ) -> None:
        import asyncio
        await self.remove_stage2_or_invalidate(
            entity_kind=entity_kind, entity_id=entity_id
        )
        payload = json.loads(await asyncio.to_thread(self.engine.indexing.canonical_revision_payload, entity_kind=entity_kind, entity_id=entity_id))
        revision = await self._current(entity_kind, entity_id)
        await self._meta("replace_stage1_node_projection", self._namespace(), self._key(entity_kind, entity_id), {
            "id": entity_id, "entity_kind": entity_kind, "document": str(document),
            "metadata": dict(metadata or {}), "source_fingerprint": str(payload.get("source_fingerprint") or ""),
        }, last_authoritative_seq=int(getattr(revision, "revision", 0) if revision else 0), last_materialized_seq=int(getattr(revision, "revision", 0) if revision else 0), projection_schema_version=1, materialization_status="pending")
        payload_json = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind, entity_id=entity_id,
        )
        await asyncio.to_thread(
            self.engine.indexing.enqueue_index_job,
            entity_kind=entity_kind, entity_id=entity_id,
            index_kind="node_embedding", op="UPSERT", payload_json=payload_json,
        )

    async def stage1_query(self, **kwargs: object) -> list[dict[str, object]]:
        rows = await self._meta("query_stage1_node_projections", self._namespace(), **kwargs)
        return [dict(row) for row in cast(Sequence[Mapping[str, object]], rows)]

    async def remove_stage1(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        expected_last_authoritative_seq: int | None = None,
        **_: object,
    ) -> bool | None:
        if expected_last_authoritative_seq is not None:
            return bool(
                await self._meta(
                    "clear_stage1_node_projection_if_current",
                    self._namespace(),
                    self._key(entity_kind, entity_id),
                    int(expected_last_authoritative_seq),
                )
            )
        await self._meta(
            "clear_stage1_node_projection",
            self._namespace(),
            self._key(entity_kind, entity_id),
        )
        return None

    async def restore_stage1_from_canonical_event(
        self, *, entity_kind: str, entity_id: str
    ) -> bool:
        """Re-stage the latest event revision after a stale Chroma write."""
        import asyncio

        revision = await self._current(entity_kind, entity_id)
        if revision is None or revision.state != "active":
            return False
        if entity_kind == "node":
            entity: Node | Edge = Node.model_validate(revision.payload)
            document, metadata = self.engine.write.node_doc_and_meta(entity)
        elif entity_kind == "edge":
            entity = Edge.model_validate(revision.payload)
            document = entity.model_dump_json(
                field_mode="backend", exclude=["embedding"]
            )
            metadata = self.engine.write.enrich_edge_meta(entity)
        else:
            raise ValueError(f"unsupported two-stage entity kind: {entity_kind!r}")

        payload_json = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        revision_payload = json.loads(payload_json)
        await self._meta(
            "replace_stage1_node_projection",
            self._namespace(),
            self._key(entity_kind, entity_id),
            {
                "id": entity_id,
                "entity_kind": entity_kind,
                "document": str(document),
                "metadata": dict(metadata or {}),
                "source_fingerprint": str(
                    revision_payload.get("source_fingerprint") or ""
                ),
            },
            last_authoritative_seq=int(revision.revision),
            last_materialized_seq=int(revision.revision),
            projection_schema_version=1,
            materialization_status="pending",
        )
        latest = await self._current(entity_kind, entity_id)
        if latest is None or latest.revision != revision.revision:
            await self.remove_stage1(
                entity_kind=entity_kind,
                entity_id=entity_id,
                expected_last_authoritative_seq=int(revision.revision),
            )
            return False
        await asyncio.to_thread(
            self.engine.indexing.enqueue_index_job,
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind="node_embedding",
            op="UPSERT",
            payload_json=payload_json,
        )
        return True

    async def _restore_current_stage1(
        self, *, entity_kind: str, entity_id: str
    ) -> bool:
        """Make stale async job replay schedule the current canonical projection."""
        current = await self._current(entity_kind, entity_id)
        if current is None or current.state != "active":
            await self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return False
        if not await self.restore_stage1_from_canonical_event(
            entity_kind=entity_kind, entity_id=entity_id
        ):
            latest = await self._current(entity_kind, entity_id)
            if latest is not None and latest.state == "active":
                raise RuntimeError(
                    "current async Chroma revision changed during Stage-1 recovery"
                )
            await self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return False
        return True

    async def remove_stage2_or_invalidate(self, *, entity_kind: str, entity_id: str, **_: object) -> None:
        await self.engine.backend.async_call(entity_kind, "delete", ids=[entity_id])
        if entity_kind == "edge":
            await self.engine.backend.async_call(
                "edge_endpoints", "delete", where={"edge_id": entity_id}
            )

    async def _delete_projection_and_restore_if_active(
        self, *, entity_kind: str, entity_id: str
    ) -> None:
        row = _optional_row_mapping(
            await self._meta(
                "get_stage1_node_projection",
                self._namespace(),
                self._key(entity_kind, entity_id),
            )
        )
        if row is not None:
            raw_sequence = row.get("last_authoritative_seq")
            sequence = raw_sequence if isinstance(raw_sequence, int) else 0
            await self.remove_stage1(
                entity_kind=entity_kind,
                entity_id=entity_id,
                expected_last_authoritative_seq=sequence,
            )
        await self.remove_stage2_or_invalidate(
            entity_kind=entity_kind, entity_id=entity_id
        )
        latest = await self._current(entity_kind, entity_id)
        if latest is not None and latest.state == "active":
            if not await self.restore_stage1_from_canonical_event(
                entity_kind=entity_kind, entity_id=entity_id
            ):
                raise RuntimeError(
                    "async Chroma delete raced with resurrection and could not restore Stage-1"
                )

    async def _promote_edge_endpoints(
        self, document: str, embedding: list[float] | None = None
    ) -> None:
        from .models import Edge

        edge = Edge.model_validate_json(document)
        rows = edge_endpoint_rows(edge)
        if not rows:
            return
        documents = [json.dumps(row) for row in rows]
        if embedding is None:
            raise RuntimeError(
                "edge endpoint promotion requires the current edge embedding"
            )
        await self.engine.backend.async_call(
            "edge_endpoints", "upsert",
            ids=[row["id"] for row in rows],
            documents=documents,
            metadatas=rows,
            # Chroma requires vectors here; structural rows reuse the already
            # computed edge vector and never call the provider.
            embeddings=[list(embedding) for _ in rows],
        )

    async def apply_embedding_job(self, *, entity_kind: str, entity_id: str, op: str, payload_json: str | None) -> None:
        current = await self._current(entity_kind, entity_id)
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        if op.upper() == "DELETE":
            if (
                expected
                and current is not None
                and current.state == "active"
                and expected != await self._fingerprint(entity_kind, entity_id)
            ):
                await self._restore_current_stage1(
                    entity_kind=entity_kind, entity_id=entity_id
                )
                return
            await self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        if current is None or current.state != "active":
            await self._delete_projection_and_restore_if_active(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        current_fingerprint = await self._fingerprint(entity_kind, entity_id)
        if expected and expected != current_fingerprint:
            await self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        row = _optional_row_mapping(
            await self._meta(
                "get_stage1_node_projection", self._namespace(),
                self._key(entity_kind, entity_id),
            )
        )
        if not row:
            await self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        staged = _row_mapping(row.get("payload"))
        staged_seq = _sequence_number(row.get("last_authoritative_seq"))
        staged_fingerprint = str(staged.get("source_fingerprint") or "")
        if not staged_fingerprint:
            raise RuntimeError("Chroma Stage-1 projection has no source fingerprint")
        if (expected and staged_fingerprint != expected) or (
            staged_fingerprint != current_fingerprint
        ):
            await self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        embedding = (await _embed(self.engine._ef, [str(staged.get("document") or "")]))[0]
        current = await self._current(entity_kind, entity_id)
        if current is None or current.state != "active":
            return
        if staged_fingerprint != await self._fingerprint(entity_kind, entity_id):
            await self._restore_current_stage1(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        collection_key = entity_kind
        await self.engine.backend.async_call(
            collection_key, "upsert", ids=[entity_id],
            documents=[str(staged.get("document") or "")],
            metadatas=[{
                **_row_mapping(staged.get("metadata")),
                "_kogwistar_stage2_ready": True,
                "_kogwistar_source_fingerprint": staged_fingerprint,
                "_kogwistar_source_revision": staged_seq,
            }],
            embeddings=[list(embedding)],
        )
        latest = await self._current(entity_kind, entity_id)
        if latest is None or latest.state != "active":
            await self.remove_stage1(
                entity_kind=entity_kind,
                entity_id=entity_id,
                expected_last_authoritative_seq=staged_seq,
            )
            await self.remove_stage2_or_invalidate(
                entity_kind=entity_kind, entity_id=entity_id
            )
            return
        latest_fingerprint = await self._fingerprint(entity_kind, entity_id)
        if latest_fingerprint != staged_fingerprint:
            if not await self.restore_stage1_from_canonical_event(
                entity_kind=entity_kind, entity_id=entity_id
            ):
                raise RuntimeError("async Chroma promotion could not restore Stage-1")
            return
        if entity_kind == "edge":
            await self._promote_edge_endpoints(
                str(staged.get("document") or ""), list(embedding)
            )
        await self.remove_stage1(
            entity_kind=entity_kind,
            entity_id=entity_id,
            expected_last_authoritative_seq=staged_seq,
        )

    async def promote_stage2(
        self, *, entity_kind: str, entity_id: str, op: str,
        payload_json: str | None,
    ) -> None:
        await self.apply_embedding_job(
            entity_kind=entity_kind, entity_id=entity_id, op=op,
            payload_json=payload_json,
        )

    async def apply_embedding_jobs_batch(
        self, jobs: list[object]
    ) -> dict[str, BaseException | None]:
        prepared: list[tuple[str, str, str, dict[str, object]]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            job_id = str(_job_value(job, "job_id") or "")
            kind = str(_job_value(job, "entity_kind") or "")
            entity_id = str(_job_value(job, "entity_id") or "")
            payload_json = _payload_text(_job_value(job, "payload_json"))
            op = str(_job_value(job, "op") or "UPSERT")
            try:
                current = await self._current(kind, entity_id)
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                if op.upper() == "DELETE" or current is None or current.state != "active":
                    await self.apply_embedding_job(entity_kind=kind, entity_id=entity_id, op=op, payload_json=payload_json)
                    outcomes[job_id] = None
                    continue
                if expected and expected != await self._fingerprint(kind, entity_id):
                    await self._restore_current_stage1(
                        entity_kind=kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                row = _optional_row_mapping(
                    await self._meta(
                        "get_stage1_node_projection", self._namespace(),
                        self._key(kind, entity_id),
                    )
                )
                if not row:
                    await self._restore_current_stage1(
                        entity_kind=kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                staged = _row_mapping(row.get("payload"))
                staged_fingerprint = str(staged.get("source_fingerprint") or "")
                if not staged_fingerprint:
                    raise RuntimeError("Chroma Stage-1 projection has no source fingerprint")
                if (expected and staged_fingerprint != expected) or (
                    staged_fingerprint != await self._fingerprint(kind, entity_id)
                ):
                    await self._restore_current_stage1(
                        entity_kind=kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                prepared.append(
                    (
                        job_id,
                        kind,
                        entity_id,
                        {
                            **staged,
                            "_stage1_authoritative_seq": _sequence_number(
                                row.get("last_authoritative_seq")
                            ),
                        },
                    )
                )
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        try:
            embeddings = await _embed(
                self.engine._ef,
                [str(row.get("document") or "") for *_, row in prepared],
            )
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException as exc:
            for job_id, *_ in prepared:
                outcomes[job_id] = exc
            return outcomes
        for (job_id, kind, entity_id, staged), embedding in zip(prepared, embeddings):
            try:
                current = await self._current(kind, entity_id)
                if current is None or current.state != "active":
                    continue
                staged_fingerprint = str(staged.get("source_fingerprint") or "")
                if staged_fingerprint != await self._fingerprint(kind, entity_id):
                    await self._restore_current_stage1(
                        entity_kind=kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                await self.engine.backend.async_call(
                    kind, "upsert", ids=[entity_id],
                    documents=[str(staged.get("document") or "")],
                    metadatas=[{**_row_mapping(staged.get("metadata")), "_kogwistar_stage2_ready": True, "_kogwistar_source_fingerprint": staged_fingerprint, "_kogwistar_source_revision": _sequence_number(staged.get("_stage1_authoritative_seq"))}],
                    embeddings=[embedding],
                )
                latest = await self._current(kind, entity_id)
                if latest is None or latest.state != "active":
                    await self.remove_stage1(
                        entity_kind=kind,
                        entity_id=entity_id,
                        expected_last_authoritative_seq=_sequence_number(
                            staged.get("_stage1_authoritative_seq")
                        ),
                    )
                    await self.remove_stage2_or_invalidate(
                        entity_kind=kind, entity_id=entity_id
                    )
                    outcomes[job_id] = None
                    continue
                if staged_fingerprint != await self._fingerprint(kind, entity_id):
                    if not await self.restore_stage1_from_canonical_event(
                        entity_kind=kind, entity_id=entity_id
                    ):
                        raise RuntimeError("async Chroma batch promotion could not restore Stage-1")
                    outcomes[job_id] = None
                    continue
                if kind == "edge":
                    await self._promote_edge_endpoints(
                        str(staged.get("document") or ""), list(embedding)
                    )
                await self.remove_stage1(
                    entity_kind=kind,
                    entity_id=entity_id,
                    expected_last_authoritative_seq=_sequence_number(
                        staged.get("_stage1_authoritative_seq")
                    ),
                )
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

    async def backfill_stage2_source_revisions(
        self, *, entity_kind: str, limit: int = 100, offset: int = 0
    ) -> tuple[int, int]:
        """Stamp verified legacy rows without changing vectors.

        This operator migration requires the normal Chroma projection writer
        to be quiesced. Unverifiable rows remain untouched and hidden.
        """
        import asyncio

        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported two-stage entity kind: {entity_kind!r}")
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit must be in [1, 1000] and offset must be non-negative")
        page = _row_mapping(
            await self.engine.backend.async_call(
                entity_kind,
                "get",
                limit=limit,
                offset=offset,
                include=["metadatas"],
            )
        )
        ids = [str(value) for value in _object_list(page.get("ids"))]
        if not ids:
            return 0, 0
        metadatas = _object_list(page.get("metadatas"))
        revisions = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payloads,
            entity_kind=entity_kind,
            entity_ids=ids,
        )
        updates: list[tuple[str, dict[str, object]]] = []
        for index, entity_id in enumerate(ids):
            raw_metadata = metadatas[index] if index < len(metadatas) else None
            metadata = _row_mapping(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            if metadata.get("_kogwistar_stage2_ready") is not True:
                continue
            stored_revision = metadata.get("_kogwistar_source_revision")
            try:
                current = json.loads(revisions[entity_id])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            revision = current.get("canonical_revision")
            if (
                current.get("source_state") != "active"
                or current.get("source_fingerprint")
                != metadata.get("_kogwistar_source_fingerprint")
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision <= 0
            ):
                continue
            if stored_revision == revision:
                continue
            metadata["_kogwistar_source_revision"] = revision
            updates.append((entity_id, metadata))
        if updates:
            await self.engine.backend.async_call(
                entity_kind,
                "update",
                ids=[row[0] for row in updates],
                metadatas=[row[1] for row in updates],
            )
        return len(ids), len(updates)

    async def reconcile_projection(self, **_: object) -> int:
        removed = 0
        for row in await self.stage1_query(entity_kind="node") + await self.stage1_query(entity_kind="edge"):
            kind = str(row.get("entity_kind") or "node")
            payload = _row_mapping(row.get("payload"))
            entity_id = str(payload.get("id") or row.get("id") or row.get("key") or "")
            if ":" in entity_id and entity_id.startswith(("node:", "edge:")):
                entity_id = entity_id.split(":", 1)[1]
            staged_seq = _sequence_number(row.get("last_authoritative_seq"))
            current = await self._current(kind, entity_id)
            source_fingerprint = str(
                payload.get("source_fingerprint") or row.get("source_fingerprint") or ""
            )
            if current is None or current.state != "active":
                if await self.remove_stage1(
                    entity_kind=kind,
                    entity_id=entity_id,
                    expected_last_authoritative_seq=staged_seq,
                ):
                    await self.remove_stage2_or_invalidate(
                        entity_kind=kind, entity_id=entity_id
                    )
                    removed += 1
                continue
            if source_fingerprint and source_fingerprint != await self._fingerprint(kind, entity_id):
                await self.restore_stage1_from_canonical_event(
                    entity_kind=kind, entity_id=entity_id
                )
                continue
            ready = _row_mapping(await self.engine.backend.async_call(
                kind, "get", ids=[entity_id],
                include=["documents", "metadatas", "embeddings"]
            ))
            metadatas = _object_list(ready.get("metadatas"))
            metadata = metadatas[0] if metadatas else None
            if (
                _object_list(ready.get("ids"))
                and isinstance(metadata, dict)
                and metadata.get("_kogwistar_source_fingerprint") == source_fingerprint
            ):
                if kind == "edge":
                    documents = _object_list(ready.get("documents"))
                    document = documents[0] if documents else None
                    if document:
                        embeddings = _object_list(ready.get("embeddings"))
                        await self._promote_edge_endpoints(
                            str(document),
                            list(embeddings[0])
                            if embeddings and embeddings[0] is not None
                            and isinstance(embeddings[0], Sequence) else None,
                        )
                if await self.remove_stage1(
                    entity_kind=kind,
                    entity_id=entity_id,
                    expected_last_authoritative_seq=staged_seq,
                ):
                    removed += 1
        return removed


class AsyncRustPostgresTwoStageProjectionAdapter:
    """Async facade for Rust authority until the native async ABI is exposed.

    Calls execute in worker threads, so the async event loop is not blocked and
    Rust remains the sole PostgreSQL writer. This is an async transport seam,
    not a claim that the current Python extension has an async ABI.
    """

    def __init__(self, engine: object, meta: object) -> None:
        self.engine = cast(_TwoStageEngineLike, engine)
        self.meta = cast(_MetaStoreLike, meta)
        self._sync_adapter = RustPostgresTwoStageProjectionAdapter(
            cast(_RustProjectionEngine, engine),
            cast(_RustProjectionMeta, meta),
        )

    def _table(self, entity_kind: str) -> str:
        return self._sync_adapter._table(entity_kind)

    def _namespace(self) -> str:
        return self._sync_adapter._namespace()

    def enqueue_embedding_job(
        self, *, entity_kind: str, entity_id: str, op: str,
    ) -> None:
        self._sync_adapter.enqueue_embedding_job(
            entity_kind=entity_kind, entity_id=entity_id, op=op,
        )

    def _promote_record(
        self, *, entity_kind: str, entity_id: str,
        record: dict[str, object], embedding: Sequence[float], expected: str,
    ) -> None:
        self._sync_adapter._promote_record(
            entity_kind=entity_kind, entity_id=entity_id,
            record=cast(JsonObject, record),
            embedding=list(embedding),
            expected=expected,
        )

    async def add_node(self, node: Node, *, doc_id: str | None = None) -> None:
        import asyncio
        await asyncio.to_thread(self._sync_adapter.add_node, node, doc_id=doc_id)
        await asyncio.to_thread(
            self.enqueue_embedding_job,
            entity_kind="node", entity_id=node.safe_get_id(), op="UPSERT",
        )

    async def add_edge(self, edge: Edge, *, doc_id: str | None = None) -> None:
        import asyncio
        await asyncio.to_thread(self._sync_adapter.add_edge, edge, doc_id=doc_id)
        await asyncio.to_thread(
            self.enqueue_embedding_job,
            entity_kind="edge", entity_id=edge.safe_get_id(), op="UPSERT",
        )

    async def apply_embedding_job(
        self, *, entity_kind: str, entity_id: str, op: str,
        payload_json: str | None,
    ) -> None:
        import asyncio
        provider = getattr(self.engine, "_ef", None)
        provider_is_async = inspect.iscoroutinefunction(provider) or inspect.iscoroutinefunction(
            getattr(provider, "__call__", None)
        )
        if not provider_is_async:
            await asyncio.to_thread(
                self._sync_adapter.apply_embedding_job,
                entity_kind=entity_kind, entity_id=entity_id,
                op=op, payload_json=payload_json,
            )
            return
        if not callable(provider):
            raise RuntimeError("async Rust two-stage projection requires an embedding provider")
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        if op.upper() == "DELETE":
            await asyncio.to_thread(
                self._sync_adapter._clear_embedding_projection,
                entity_kind=entity_kind,
                entity_id=entity_id,
            )
            return
        async_provider = cast(EmbeddingProvider, provider)
        current = await asyncio.to_thread(
            self.engine.indexing.canonical_entity_revision,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        if current is None or current.state != "active":
            await asyncio.to_thread(
                self._sync_adapter._clear_embedding_projection,
                entity_kind=entity_kind,
                entity_id=entity_id,
                only_if_inactive=True,
            )
            return
        actual = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        if expected and expected != str(json.loads(actual).get("source_fingerprint") or ""):
            return
        records = cast(Sequence[Mapping[str, object]], await asyncio.to_thread(
            self.meta.graph_projection_records,
            namespace=self._namespace(), workspace_id=None, graph_space=None,
            table=self._table(entity_kind), ids=[entity_id], metadata={}, limit=1,
        ))
        if not records:
            raise RuntimeError("current Rust Stage-1 graph projection is missing")
        embedding = (await _embed(
            async_provider,
            [str(records[0].get("document") or "")],
        ))[0]
        current = await asyncio.to_thread(
            self.engine.indexing.canonical_entity_revision,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        if current is None or current.state != "active":
            return
        actual = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        if expected and expected != str(json.loads(actual).get("source_fingerprint") or ""):
            return
        await asyncio.to_thread(
            self._promote_record,
            entity_kind=entity_kind, entity_id=entity_id,
            record=dict(records[0]), embedding=embedding, expected=expected,
        )

    async def apply_embedding_jobs_batch(self, jobs: list[object]) -> dict[str, BaseException | None]:
        import asyncio
        provider = getattr(self.engine, "_ef", None)
        provider_is_async = inspect.iscoroutinefunction(provider) or inspect.iscoroutinefunction(
            getattr(provider, "__call__", None)
        )
        if not provider_is_async:
            return await asyncio.to_thread(self._sync_adapter.apply_embedding_jobs_batch, jobs)
        if not callable(provider):
            raise RuntimeError("async Rust two-stage projection requires an embedding provider")
        async_provider = cast(Callable[[list[str]], Awaitable[Sequence[object]]], provider)

        prepared: list[tuple[str, str, str, dict[str, object], str]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            job_id = str(_job_value(job, "job_id") or "")
            kind = str(_job_value(job, "entity_kind") or "")
            entity_id = str(_job_value(job, "entity_id") or "")
            op = str(_job_value(job, "op") or "UPSERT")
            try:
                payload_json = _payload_text(_job_value(job, "payload_json"))
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                if op.upper() == "DELETE":
                    await asyncio.to_thread(
                        self._sync_adapter._clear_embedding_projection,
                        entity_kind=kind,
                        entity_id=entity_id,
                    )
                    outcomes[job_id] = None
                    continue
                current = await asyncio.to_thread(
                    self.engine.indexing.canonical_entity_revision,
                    entity_kind=kind, entity_id=entity_id,
                )
                if current is None or current.state != "active":
                    await asyncio.to_thread(
                        self._sync_adapter._clear_embedding_projection,
                        entity_kind=kind,
                        entity_id=entity_id,
                        only_if_inactive=True,
                    )
                    outcomes[job_id] = None
                    continue
                actual = await asyncio.to_thread(
                    self.engine.indexing.canonical_revision_payload,
                    entity_kind=kind, entity_id=entity_id,
                )
                if expected and expected != str(json.loads(actual).get("source_fingerprint") or ""):
                    outcomes[job_id] = None
                    continue
                records = cast(Sequence[Mapping[str, object]], await asyncio.to_thread(
                    self.meta.graph_projection_records,
                    namespace=self._namespace(), workspace_id=None, graph_space=None,
                    table=self._table(kind), ids=[entity_id], metadata={}, limit=1,
                ))
                if not records:
                    raise RuntimeError("current Rust Stage-1 graph projection is missing")
                prepared.append((job_id, kind, entity_id, dict(records[0]), expected))
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        try:
            embeddings = await _embed(
                cast(EmbeddingProvider, async_provider),
                [str(item[3].get("document") or "") for item in prepared],
            )
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException as exc:
            for job_id, *_ in prepared:
                outcomes[job_id] = exc
            return outcomes
        for (job_id, kind, entity_id, record, expected), embedding in zip(prepared, embeddings):
            try:
                current = await asyncio.to_thread(
                    self.engine.indexing.canonical_entity_revision,
                    entity_kind=kind, entity_id=entity_id,
                )
                if current is None or current.state != "active":
                    outcomes[job_id] = None
                    continue
                actual = await asyncio.to_thread(
                    self.engine.indexing.canonical_revision_payload,
                    entity_kind=kind, entity_id=entity_id,
                )
                if expected and expected != str(json.loads(actual).get("source_fingerprint") or ""):
                    outcomes[job_id] = None
                    continue
                await asyncio.to_thread(
                    self._promote_record,
                    entity_kind=kind, entity_id=entity_id,
                    record=record, embedding=embedding, expected=expected,
                )
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

    async def stage1_query(self, **_: object) -> list[dict[str, object]]:
        return []

    async def remove_stage1(self, **_: object) -> None:
        return None

    async def promote_stage2(
        self, *, entity_kind: str, entity_id: str, op: str,
        payload_json: str | None,
    ) -> None:
        await self.apply_embedding_job(
            entity_kind=entity_kind, entity_id=entity_id, op=op,
            payload_json=payload_json,
        )

    async def remove_stage2_or_invalidate(self, **_: object) -> None:
        return None

    async def reconcile_projection(self, **_: object) -> int:
        return 0


__all__ = [
    "AsyncChromaTwoStageProjectionAdapter",
    "AsyncPostgresTwoStageProjectionAdapter",
    "AsyncRustPostgresTwoStageProjectionAdapter",
    "async_transient_two_stage_capability",
]
