"""Async two-stage projection adapters for existing backend arrangements."""

from __future__ import annotations

import inspect
import json
from contextlib import asynccontextmanager
from typing import Any

from .storage_backend import TwoStageProjectionCapability
from .edge_endpoint_rows import edge_endpoint_rows
from ..utils.embedding_vectors import normalize_embedding_vector
from .two_stage_rust_postgres import RustPostgresTwoStageProjectionAdapter


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

    def __init__(self, engine: Any) -> None:
        if not getattr(engine.backend, "_is_async_engine", False):
            raise ValueError("async PostgreSQL adapter requires an async engine")
        self.engine = engine

    def _backend(self) -> Any:
        return self.engine.backend

    def _namespace(self) -> str:
        return str(getattr(self.engine, "namespace", "default"))

    @asynccontextmanager
    async def _backend_transaction(self):
        """Join the configured async SQL UOW when one exists."""
        uow = getattr(self.engine, "_async_backend_uow", None)
        transaction = getattr(uow, "transaction", None)
        if callable(transaction):
            async with transaction():
                yield
        else:
            yield

    async def add_node(self, node: Any, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        document, metadata = self.engine.write.node_doc_and_meta(node)
        await self._add(
            entity_kind="node", entity_id=node.safe_get_id(),
            document=document, metadata=metadata,
        )

    async def add_edge(self, edge: Any, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        await self._add(
            entity_kind="edge", entity_id=edge.safe_get_id(),
            document=edge.model_dump_json(field_mode="backend", exclude=["embedding"]),
            metadata=self.engine.write.enrich_edge_meta(edge),
        )

    async def _add(
        self, *, entity_kind: str, entity_id: str, document: str,
        metadata: dict[str, Any],
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

    async def stage1_query(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = await self._backend().stage1_projection_query_async(
            namespace=self._namespace(), entity_kind=str(kwargs.get("entity_kind") or "node"),
            ids=kwargs.get("ids"), metadata=kwargs.get("metadata"),
            limit=kwargs.get("limit", 200),
        )
        return [dict(row) for row in rows]

    async def remove_stage1(self, *, entity_kind: str, entity_id: str, **_: Any) -> None:
        await self._backend().stage1_projection_delete_async(
            namespace=self._namespace(), entity_kind=entity_kind, entity_id=entity_id
        )

    async def remove_stage2_or_invalidate(self, *, entity_kind: str, entity_id: str, **_: Any) -> None:
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

    async def _current(self, entity_kind: str, entity_id: str) -> Any:
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
        if op.upper() == "DELETE" or current is None or current.state != "active":
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
        result = self.engine._ef([document])
        if inspect.isawaitable(result):
            result = await result
        embedding = normalize_embedding_vector(list(result)[0], allow_none=False)
        current = await self._current(entity_kind, entity_id)
        if current is None or current.state != "active":
            await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
            await self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
            return
        if expected and expected != await self._fingerprint(entity_kind, entity_id):
            return
        metadata = dict(row.get("metadata") or {})
        metadata["_kogwistar_stage2_ready"] = True
        metadata["_kogwistar_source_fingerprint"] = expected
        async with self._backend_transaction():
            await self._backend()._upsert_async(
                getattr(self._backend(), f"{entity_kind}s"), ids=[entity_id],
                documents=[document], metadatas=[metadata], embeddings=[embedding],
            )
            if entity_kind == "edge":
                await self._promote_edge_endpoints(document)
            await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)

    async def promote_stage2(self, **kwargs: Any) -> None:
        await self.apply_embedding_job(**kwargs)

    async def apply_embedding_jobs_batch(
        self, jobs: list[Any]
    ) -> dict[str, BaseException | None]:
        prepared: list[tuple[str, str, str, dict[str, Any]]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            value = (lambda name: job.get(name)) if isinstance(job, dict) else (
                lambda name: getattr(job, name, None)
            )
            job_id = str(value("job_id") or "")
            kind, entity_id = str(value("entity_kind") or ""), str(value("entity_id") or "")
            payload_json, op = value("payload_json"), str(value("op") or "UPSERT")
            try:
                current = await self._current(kind, entity_id)
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                if op.upper() == "DELETE" or current is None or current.state != "active":
                    await self.apply_embedding_job(entity_kind=kind, entity_id=entity_id, op=op, payload_json=payload_json)
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
                prepared.append((job_id, kind, entity_id, row))
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        try:
            result = self.engine._ef([str(row.get("document") or "") for *_, row in prepared])
            if inspect.isawaitable(result):
                result = await result
            embeddings = list(result)
            if len(embeddings) != len(prepared):
                raise RuntimeError("embedding provider returned wrong batch length")
        except BaseException as exc:
            for job_id, *_ in prepared:
                outcomes[job_id] = exc
            return outcomes
        for (job_id, kind, entity_id, row), embedding in zip(prepared, embeddings):
            try:
                current = await self._current(kind, entity_id)
                if current is None or current.state != "active":
                    continue
                source_fingerprint = str(row.get("source_fingerprint") or "")
                if source_fingerprint and source_fingerprint != await self._fingerprint(kind, entity_id):
                    continue
                metadata = dict(row.get("metadata") or {})
                metadata["_kogwistar_stage2_ready"] = True
                metadata["_kogwistar_source_fingerprint"] = source_fingerprint
                async with self._backend_transaction():
                    await self._backend()._upsert_async(
                        getattr(self._backend(), f"{kind}s"), ids=[entity_id],
                        documents=[str(row["document"])], metadatas=[metadata],
                        embeddings=[normalize_embedding_vector(embedding, allow_none=False)],
                    )
                    if kind == "edge":
                        await self._promote_edge_endpoints(str(row["document"]))
                    await self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

    async def remove_stage2_or_invalidate_and_cleanup(self, **kwargs: Any) -> None:
        await self.remove_stage2_or_invalidate(**kwargs)

    async def reconcile_projection(self, **_: Any) -> int:
        removed = 0
        for row in await self.stage1_query(entity_kind="node") + await self.stage1_query(entity_kind="edge"):
            entity_kind = str(row["entity_kind"])
            entity_id = str(row["entity_id"])
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
            stage2_metadata = (stage2.get("metadatas") or [None])[0]
            if (
                stage2.get("ids")
                and isinstance(stage2_metadata, dict)
                and stage2_metadata.get("_kogwistar_source_fingerprint")
                == row.get("source_fingerprint")
            ):
                if entity_kind == "edge":
                    document = (stage2.get("documents") or [None])[0]
                    if document:
                        async with self._backend_transaction():
                            await self._promote_edge_endpoints(document)
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

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _namespace(self) -> str:
        return str(getattr(self.engine, "namespace", "default"))

    def _key(self, entity_kind: str, entity_id: str) -> str:
        return f"{entity_kind}:{entity_id}"

    async def _meta(self, method: str, *args: Any, **kwargs: Any) -> Any:
        import asyncio
        return await asyncio.to_thread(getattr(self.engine.meta_sqlite, method), *args, **kwargs)

    async def _current(self, entity_kind: str, entity_id: str) -> Any:
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

    async def add_node(self, node: Any, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            node.doc_id = doc_id
        document, metadata = self.engine.write.node_doc_and_meta(node)
        await self._add("node", node.safe_get_id(), document, metadata)

    async def add_edge(self, edge: Any, *, doc_id: str | None = None) -> None:
        if doc_id is not None:
            edge.doc_id = doc_id
        await self._add("edge", edge.safe_get_id(), edge.model_dump_json(field_mode="backend", exclude=["embedding"]), self.engine.write.enrich_edge_meta(edge))

    async def _add(self, entity_kind: str, entity_id: str, document: str, metadata: dict[str, Any]) -> None:
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

    async def stage1_query(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = await self._meta("query_stage1_node_projections", self._namespace(), **kwargs)
        return list(rows)

    async def remove_stage1(self, *, entity_kind: str, entity_id: str, **_: Any) -> None:
        await self._meta("clear_stage1_node_projection", self._namespace(), self._key(entity_kind, entity_id))

    async def remove_stage2_or_invalidate(self, *, entity_kind: str, entity_id: str, **_: Any) -> None:
        await self.engine.backend.async_call(entity_kind, "delete", ids=[entity_id])
        if entity_kind == "edge":
            await self.engine.backend.async_call(
                "edge_endpoints", "delete", where={"edge_id": entity_id}
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
        if op.upper() == "DELETE" or current is None or current.state != "active":
            await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)
            await self.remove_stage2_or_invalidate(entity_kind=entity_kind, entity_id=entity_id)
            return
        row = await self._meta("get_stage1_node_projection", self._namespace(), self._key(entity_kind, entity_id))
        if not row:
            raise RuntimeError("current Chroma Stage-1 projection is missing")
        staged = row.get("payload") or {}
        result = self.engine._ef([str(staged.get("document") or "")])
        if inspect.isawaitable(result):
            result = await result
        embedding = list(result)[0]
        current = await self._current(entity_kind, entity_id)
        if current is None or current.state != "active":
            return
        if expected and expected != await self._fingerprint(entity_kind, entity_id):
            return
        collection_key = entity_kind
        await self.engine.backend.async_call(collection_key, "upsert", ids=[entity_id], documents=[str(staged.get("document") or "")], metadatas=[{**dict(staged.get("metadata") or {}), "_kogwistar_stage2_ready": True, "_kogwistar_source_fingerprint": expected}], embeddings=[embedding])
        if entity_kind == "edge":
            await self._promote_edge_endpoints(
                str(staged.get("document") or ""), embedding
            )
        await self.remove_stage1(entity_kind=entity_kind, entity_id=entity_id)

    async def promote_stage2(self, **kwargs: Any) -> None:
        await self.apply_embedding_job(**kwargs)

    async def apply_embedding_jobs_batch(
        self, jobs: list[Any]
    ) -> dict[str, BaseException | None]:
        prepared: list[tuple[str, str, str, dict[str, Any]]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            value = (lambda name: job.get(name)) if isinstance(job, dict) else (
                lambda name: getattr(job, name, None)
            )
            job_id = str(value("job_id") or "")
            kind, entity_id = str(value("entity_kind") or ""), str(value("entity_id") or "")
            payload_json, op = value("payload_json"), str(value("op") or "UPSERT")
            try:
                current = await self._current(kind, entity_id)
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
                if op.upper() == "DELETE" or current is None or current.state != "active":
                    await self.apply_embedding_job(entity_kind=kind, entity_id=entity_id, op=op, payload_json=payload_json)
                    outcomes[job_id] = None
                    continue
                row = await self._meta("get_stage1_node_projection", self._namespace(), self._key(kind, entity_id))
                if not row:
                    raise RuntimeError("current Chroma Stage-1 projection is missing")
                staged = row.get("payload") or {}
                if expected and str(staged.get("source_fingerprint") or "") != expected:
                    outcomes[job_id] = None
                    continue
                prepared.append((job_id, kind, entity_id, {**staged, "_expected": expected}))
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        try:
            result = self.engine._ef([str(row.get("document") or "") for *_, row in prepared])
            if inspect.isawaitable(result):
                result = await result
            embeddings = list(result)
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
                await self.engine.backend.async_call(
                    kind, "upsert", ids=[entity_id],
                    documents=[str(staged.get("document") or "")],
                    metadatas=[{**dict(staged.get("metadata") or {}), "_kogwistar_stage2_ready": True, "_kogwistar_source_fingerprint": str(staged.get("_expected") or staged.get("source_fingerprint") or "")}],
                    embeddings=[embedding],
                )
                if kind == "edge":
                    await self._promote_edge_endpoints(
                        str(staged.get("document") or ""), embedding
                    )
                await self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                outcomes[job_id] = None
            except BaseException as exc:
                outcomes[job_id] = exc
        return outcomes

    async def reconcile_projection(self, **_: Any) -> int:
        removed = 0
        for row in await self.stage1_query(entity_kind="node") + await self.stage1_query(entity_kind="edge"):
            kind = str(row.get("entity_kind") or "node")
            payload = row.get("payload") or {}
            entity_id = str(payload.get("id") or row.get("id") or row.get("key") or "")
            if ":" in entity_id and entity_id.startswith(("node:", "edge:")):
                entity_id = entity_id.split(":", 1)[1]
            current = await self._current(kind, entity_id)
            source_fingerprint = str(
                payload.get("source_fingerprint") or row.get("source_fingerprint") or ""
            )
            if current is None or current.state != "active":
                await self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                await self.remove_stage2_or_invalidate(entity_kind=kind, entity_id=entity_id)
                removed += 1
                continue
            if source_fingerprint and source_fingerprint != await self._fingerprint(kind, entity_id):
                await self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                removed += 1
                continue
            ready = await self.engine.backend.async_call(
                kind, "get", ids=[entity_id],
                include=["documents", "metadatas", "embeddings"]
            )
            metadata = (ready.get("metadatas") or [None])[0]
            if (
                ready.get("ids")
                and isinstance(metadata, dict)
                and metadata.get("_kogwistar_source_fingerprint") == source_fingerprint
            ):
                if kind == "edge":
                    document = (ready.get("documents") or [None])[0]
                    if document:
                        embeddings = ready.get("embeddings") or []
                        await self._promote_edge_endpoints(
                            document,
                            list(embeddings[0])
                            if embeddings and embeddings[0] is not None else None,
                        )
                await self.remove_stage1(entity_kind=kind, entity_id=entity_id)
                removed += 1
        return removed


class AsyncRustPostgresTwoStageProjectionAdapter(
    RustPostgresTwoStageProjectionAdapter
):
    """Async facade for Rust authority until the native async ABI is exposed.

    Calls execute in worker threads, so the async event loop is not blocked and
    Rust remains the sole PostgreSQL writer. This is an async transport seam,
    not a claim that the current Python extension has an async ABI.
    """

    async def add_node(self, node: Any, *, doc_id: str | None = None) -> None:
        import asyncio
        await asyncio.to_thread(super().add_node, node, doc_id=doc_id)
        await asyncio.to_thread(
            self.enqueue_embedding_job,
            entity_kind="node", entity_id=node.safe_get_id(), op="UPSERT",
        )

    async def add_edge(self, edge: Any, *, doc_id: str | None = None) -> None:
        import asyncio
        await asyncio.to_thread(super().add_edge, edge, doc_id=doc_id)
        await asyncio.to_thread(
            self.enqueue_embedding_job,
            entity_kind="edge", entity_id=edge.safe_get_id(), op="UPSERT",
        )

    async def apply_embedding_job(self, **kwargs: Any) -> None:
        import asyncio
        provider = getattr(self.engine, "_ef", None)
        provider_is_async = inspect.iscoroutinefunction(provider) or inspect.iscoroutinefunction(
            getattr(provider, "__call__", None)
        )
        if not provider_is_async:
            await asyncio.to_thread(super().apply_embedding_job, **kwargs)
            return

        entity_kind = str(kwargs["entity_kind"])
        entity_id = str(kwargs["entity_id"])
        op = str(kwargs.get("op") or "UPSERT")
        payload_json = kwargs.get("payload_json")
        current = await asyncio.to_thread(
            self.engine.indexing.canonical_entity_revision,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
        if op.upper() == "DELETE" or current is None or current.state != "active":
            return
        actual = await asyncio.to_thread(
            self.engine.indexing.canonical_revision_payload,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
        if expected and expected != str(json.loads(actual).get("source_fingerprint") or ""):
            return
        records = await asyncio.to_thread(
            self.meta.graph_projection_records,
            namespace=self._namespace(), workspace_id=None, graph_space=None,
            table=self._table(entity_kind), ids=[entity_id], metadata={}, limit=1,
        )
        if not records:
            raise RuntimeError("current Rust Stage-1 graph projection is missing")
        raw = await provider([str(records[0].get("document") or "")])
        embedding = list(raw)[0]
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
            record=records[0], embedding=embedding, expected=expected,
        )

    async def apply_embedding_jobs_batch(self, jobs: list[Any]) -> dict[str, BaseException | None]:
        import asyncio
        provider = getattr(self.engine, "_ef", None)
        provider_is_async = inspect.iscoroutinefunction(provider) or inspect.iscoroutinefunction(
            getattr(provider, "__call__", None)
        )
        if not provider_is_async:
            return await asyncio.to_thread(super().apply_embedding_jobs_batch, jobs)

        prepared: list[tuple[str, str, str, dict[str, Any], str]] = []
        outcomes: dict[str, BaseException | None] = {}
        for job in jobs:
            value = lambda name: job.get(name) if isinstance(job, dict) else getattr(job, name, None)
            job_id = str(value("job_id") or "")
            kind = str(value("entity_kind") or "")
            entity_id = str(value("entity_id") or "")
            op = str(value("op") or "UPSERT")
            try:
                if op.upper() == "DELETE":
                    outcomes[job_id] = None
                    continue
                payload_json = value("payload_json")
                expected = str(json.loads(payload_json or "{}").get("source_fingerprint") or "")
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
                records = await asyncio.to_thread(
                    self.meta.graph_projection_records,
                    namespace=self._namespace(), workspace_id=None, graph_space=None,
                    table=self._table(kind), ids=[entity_id], metadata={}, limit=1,
                )
                if not records:
                    raise RuntimeError("current Rust Stage-1 graph projection is missing")
                prepared.append((job_id, kind, entity_id, records[0], expected))
            except BaseException as exc:
                outcomes[job_id] = exc
        if not prepared:
            return outcomes
        try:
            raw = await provider([str(item[3].get("document") or "") for item in prepared])
            embeddings = list(raw)
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

    async def stage1_query(self, **_: Any) -> list[dict[str, Any]]:
        return []

    async def remove_stage1(self, **_: Any) -> None:
        return None

    async def promote_stage2(self, **kwargs: Any) -> None:
        await self.apply_embedding_job(**kwargs)

    async def remove_stage2_or_invalidate(self, **_: Any) -> None:
        return None

    async def reconcile_projection(self, **_: Any) -> int:
        return 0


__all__ = [
    "AsyncChromaTwoStageProjectionAdapter",
    "AsyncPostgresTwoStageProjectionAdapter",
    "AsyncRustPostgresTwoStageProjectionAdapter",
    "async_transient_two_stage_capability",
]
