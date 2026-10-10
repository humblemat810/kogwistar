from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ...cdc.change_event import EntityRefModel
from ...json_types import JsonObject
from ...typing_interfaces import ProjectionBackendLike
from ...utils.embedding_vectors import normalize_embedding_vector
from ..async_compat import run_awaitable_blocking
from ..models import Document, Domain, Edge, Node, PureChromaEdge, PureChromaNode
from ..storage_backend import AsyncTwoStageProjectionAdapter
from ..utils.metadata import json_or_none, strip_none
from ..utils.refs import (
    edge_doc_and_meta as edge_doc_and_meta_util,
)
from ..utils.refs import (
    extract_doc_ids_from_refs,
)
from ..utils.refs import (
    node_doc_and_meta as node_doc_and_meta_util,
)
from .base import NamespaceProxy

if TYPE_CHECKING:
    from ..engine import GraphKnowledgeEngine
    from ..rust_postgres_session import RustEnginePostgresMetaStore


_T = TypeVar("_T")


def _required_embedding(value: list[float] | None) -> list[float]:
    if value is None:
        raise RuntimeError("embedding provider returned no vector")
    return list(value)


def _entity_payload(entity: object) -> dict[str, Any]:
    to_jsonable = getattr(entity, "to_jsonable", None)
    if callable(to_jsonable):
        value = to_jsonable()
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    model_dump = getattr(entity, "model_dump", None)
    if callable(model_dump):
        value = model_dump(exclude=["embedding"])
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    return {}


def _backend_object(value: object) -> dict[str, Any]:
    """Narrow an untyped backend response at the adapter boundary."""

    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _refs_fingerprint(refs: Sequence[object] | None) -> str:
    payload = [
        {
            "doc_id": getattr(r, "doc_id", None),
            "method": getattr(getattr(r, "verification", None), "method", None),
            "is_verified": getattr(
                getattr(r, "verification", None), "is_verified", None
            ),
            "score": getattr(getattr(r, "verification", None), "score", None),
            "sp": getattr(r, "start_page", None),
            "ep": getattr(r, "end_page", None),
            "sc": getattr(r, "start_char", None),
            "ec": getattr(r, "end_char", None),
            "snip": (getattr(r, "excerpt", None) or "")[:64],
        }
        for r in (refs or [])
    ]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


class WriteSubsystem(NamespaceProxy["GraphKnowledgeEngine"]):
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        super().__init__(engine)

    def _projection_backend(self) -> ProjectionBackendLike:
        """Narrow optional native projection attributes at their boundary."""

        return cast(ProjectionBackendLike, self._e.backend)

    # Canonical write API
    def add_node(self, node: Node, doc_id: str | None = None) -> None:
        self._add_node_impl(node, doc_id=doc_id)

    def add_edge(self, edge: Edge, doc_id: str | None = None) -> None:
        self._add_edge_impl(edge, doc_id=doc_id)

    async def add_node_async(self, node: Node, doc_id: str | None = None) -> None:
        """Admit a node through the configured async persistence mode."""
        import asyncio

        adapter = getattr(self._e, "async_two_stage_projection_adapter", None)
        if getattr(self._e, "persistence_mode", "single_stage") == "two_stage":
            if adapter is None:
                raise RuntimeError("async two-stage projection adapter is unavailable")
            if doc_id is not None:
                node.doc_id = doc_id
            await asyncio.to_thread(self._run_pre_add_node_hooks, node)
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            await asyncio.to_thread(
                self._e._append_event_for_entity,
                namespace=getattr(self._e, "namespace", "default"),
                entity_kind="node", entity_id=node.safe_get_id(), op="ADD",
                payload=payload if isinstance(payload, dict) else {}, required=True,
            )
            return await adapter.add_node(node, doc_id=doc_id)

        return await self._add_node_async_single_stage(node, doc_id=doc_id)

    async def add_edge_async(self, edge: Edge, doc_id: str | None = None) -> None:
        """Admit an edge through the configured async persistence mode."""
        adapter = getattr(self._e, "async_two_stage_projection_adapter", None)
        if getattr(self._e, "persistence_mode", "single_stage") == "two_stage":
            if adapter is None:
                raise RuntimeError("async two-stage projection adapter is unavailable")
            return await self._add_edge_async_two_stage(edge, doc_id=doc_id, adapter=adapter)

        return await self._add_edge_async_single_stage(edge, doc_id=doc_id)

    async def _add_edge_async_two_stage(
        self,
        edge: Edge,
        *,
        doc_id: str | None,
        adapter: AsyncTwoStageProjectionAdapter,
    ) -> None:
        import asyncio

        if doc_id is not None:
            edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = await asyncio.to_thread(
            self._e.adjudicate.split_endpoints, edge.source_ids, edge.target_ids
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        await self._e.persist.assert_endpoints_exist_async(edge)
        if await asyncio.to_thread(self._run_pre_add_edge_hooks, edge, pure=False):
            return
        payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
        await asyncio.to_thread(
            self._e._append_event_for_entity,
            namespace=getattr(self._e, "namespace", "default"),
            entity_kind="edge", entity_id=edge.safe_get_id(), op="ADD",
            payload=payload if isinstance(payload, dict) else {}, required=True,
        )
        return await adapter.add_edge(edge, doc_id=doc_id)

    async def _embedding_async(self, document: str) -> list[float]:
        """Await async providers; isolate legacy sync providers from the loop."""
        import asyncio
        import inspect

        from ...utils.embedding_vectors import normalize_embedding_vector

        provider = getattr(self._e, "_ef", None)
        if provider is None:
            raw = await asyncio.to_thread(
                self._e.embed.iterative_defensive_emb, str(document)
            )
        else:
            call = provider if inspect.iscoroutinefunction(provider) else getattr(
                provider, "__call__", provider
            )
            if inspect.iscoroutinefunction(call):
                raw = await provider([str(document)])
            else:
                raw = await asyncio.to_thread(provider, [str(document)])
        raw = list(raw)[0] if raw else None
        embedding = normalize_embedding_vector(raw, allow_none=False)
        if embedding is None:
            raise RuntimeError("embedding provider returned no vector")
        return list(embedding)

    async def _async_semantic_upsert(
        self, *, entity_kind: str, entity_id: str, document: str,
        metadata: dict[str, Any], embedding: list[float],
    ) -> None:
        import asyncio

        backend = self._e.backend
        direct = getattr(backend, f"async_{entity_kind}_upsert", None)
        kwargs = {
            "ids": [entity_id], "documents": [document],
            "embeddings": [embedding], "metadatas": [metadata],
        }
        if callable(direct):
            result = direct(**kwargs)
            if inspect.isawaitable(result):
                await result
            return
        if getattr(backend, "_is_async_engine", False):
            table = getattr(backend, f"{entity_kind}s")
            upsert_async = getattr(backend, "_upsert_async", None)
            if not callable(upsert_async):
                raise RuntimeError("async backend lacks _upsert_async")
            result = upsert_async(table, **kwargs)
            if inspect.isawaitable(result):
                await result
            return
        sync_upsert = getattr(backend, f"{entity_kind}_upsert", None)
        if not callable(sync_upsert):
            raise RuntimeError(
                f"async single-stage backend lacks {entity_kind} upsert operation"
            )
        await asyncio.to_thread(sync_upsert, **kwargs)

    async def _async_backend_call(
        self, collection_key: str, method: str, **kwargs: object
    ) -> object:
        """Call native async backend verbs, with a sync compatibility fallback."""
        import asyncio
        import inspect

        backend = self._e.backend
        async_call = getattr(backend, "async_call", None)
        if callable(async_call):
            result = async_call(collection_key, method, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        if getattr(backend, "_is_async_engine", False):
            facade_key = {
                "node": "_nodes_c",
                "edge": "_edges_c",
                "document": "_documents_c",
                "domain": "_domains_c",
                "edge_endpoints": "_edge_endpoints_c",
                "edge_refs": "_edge_refs_c",
                "node_docs": "_node_docs_c",
                "node_refs": "_node_refs_c",
            }.get(collection_key)
            facade = getattr(backend, facade_key, None) if facade_key else None
            if facade is not None:
                result = getattr(facade, method)(**kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

        operation = getattr(backend, f"{collection_key}_{method}", None)
        if not callable(operation):
            raise RuntimeError(
                f"backend lacks {collection_key}_{method} operation"
            )
        return await asyncio.to_thread(operation, **kwargs)

    async def _async_patch_base_projection_metadata(
        self, *, entity_kind: str, entity_id: str, metadata_patch: dict[str, Any]
    ) -> None:
        import asyncio

        if self._rust_postgres_meta() is not None:
            await asyncio.to_thread(
                self.patch_base_projection_metadata,
                entity_kind=entity_kind,
                entity_id=entity_id,
                metadata_patch=metadata_patch,
            )
            return
        await self._async_backend_call(
            entity_kind, "update", ids=[entity_id], metadatas=[metadata_patch]
        )

    async def _async_index_node_docs(self, node: Node) -> list[str]:
        doc_ids = extract_doc_ids_from_refs(node.mentions)
        await self._async_backend_call("node_docs", "delete", where={"node_id": node.id})
        if doc_ids:
            rows = [
                {
                    "id": f"{node.id}::{did}",
                    "node_id": node.id,
                    "doc_id": did,
                    "mention_count": 1,
                }
                for did in doc_ids
            ]
            await self._async_backend_call(
                "node_docs",
                "add",
                ids=[row["id"] for row in rows],
                documents=[json.dumps(row) for row in rows],
                metadatas=rows,
                embeddings=[
                    await self._embedding_async(json.dumps(row)) for row in rows
                ],
            )

        current = await self._async_backend_call(
            "node", "get", ids=[node.safe_get_id()], include=["metadatas"]
        )
        cur_meta = (_backend_object(current).get("metadatas") or [None])[0] or {}
        new_doc_ids_json = json.dumps(doc_ids)
        if cur_meta.get("doc_ids") != new_doc_ids_json:
            await self._async_patch_base_projection_metadata(
                entity_kind="node", entity_id=node.safe_get_id(),
                metadata_patch={"doc_ids": new_doc_ids_json},
            )
        return doc_ids

    async def _async_delete_ref_rows(self, collection_key: str, field: str, entity_id: str) -> None:
        got = await self._async_backend_call(
            collection_key, "get", where={field: entity_id}, include=[]
        )
        ids = _backend_object(got).get("ids") or []
        if ids:
            await self._async_backend_call(collection_key, "delete", ids=ids)

    async def _async_index_node_refs(self, node: Node) -> list[str]:
        await self._async_delete_ref_rows("node_refs", "node_id", node.safe_get_id())
        rows: list[dict[str, Any]] = []
        for i, mention in enumerate(node.mentions or []):
            for j, span in enumerate(mention.spans):
                ver = getattr(span, "verification", None)
                rows.append(strip_none({
                    "id": f"{node.id}::mention::{i}::span::{j}",
                    "node_id": node.id,
                    "doc_id": getattr(span, "doc_id", None) or node.doc_id,
                    "insertion_method": getattr(span, "insertion_method", None),
                    "verification_method": getattr(ver, "method", None),
                    "is_verified": getattr(ver, "is_verified", None),
                    "verificication_score": getattr(ver, "score", None),
                    "page_number": getattr(span, "start_page", None),
                    "excerpt": getattr(span, "excerpt", None),
                    "start_char": getattr(span, "start_char", None),
                    "end_char": getattr(span, "end_char", None),
                }))
        if rows:
            await self._async_backend_call(
                "node_refs", "add",
                ids=[row["id"] for row in rows],
                documents=[json.dumps(row) for row in rows],
                metadatas=rows,
                embeddings=[
                    await self._embedding_async(json.dumps(row)) for row in rows
                ],
            )
        return [row["id"] for row in rows]

    async def _async_index_edge_refs(self, edge: Edge) -> list[str]:
        await self._async_delete_ref_rows("edge_refs", "edge_id", edge.safe_get_id())
        rows: list[dict[str, Any]] = []
        for i, ref in enumerate(edge.mentions or []):
            ver = getattr(ref, "verification", None)
            rows.append(strip_none({
                "id": f"{edge.id}::ref::{i}",
                "edge_id": edge.id,
                "doc_id": getattr(ref, "doc_id", None) or edge.doc_id,
                "insertion_method": getattr(ref, "insertion_method", None),
                "verification_method": getattr(ver, "method", None),
                "is_verified": getattr(ver, "is_verified", None),
                "verificication_score": getattr(ver, "score", None),
                "start_page": getattr(ref, "start_page", None),
                "end_page": getattr(ref, "end_page", None),
                "start_char": getattr(ref, "start_char", None),
                "end_char": getattr(ref, "end_char", None),
            }))
        if rows:
            await self._async_backend_call(
                "edge_refs", "add",
                ids=[row["id"] for row in rows],
                documents=[json.dumps(row) for row in rows],
                metadatas=rows,
                embeddings=[
                    await self._embedding_async(json.dumps(row)) for row in rows
                ],
            )
        return [row["id"] for row in rows]

    async def _async_maybe_reindex_node_refs(self, node: Node) -> None:
        new_fp = _refs_fingerprint(node.mentions or [])
        meta = await self._async_backend_call(
            "node", "get", ids=[node.safe_get_id()], include=["metadatas"]
        )
        metadatas = _backend_object(meta).get("metadatas") or []
        old_fp = metadatas[0].get("node_refs_fp") if metadatas and metadatas[0] else None
        got = await self._async_backend_call(
            "node_refs", "get", where={"node_id": node.id}, include=["documents"]
        )
        documents = _backend_object(got).get("documents") or []
        current_doc_ids = {json.loads(doc).get("doc_id") for doc in documents}
        expected_doc_ids = {getattr(ref, "doc_id", None) for ref in (node.mentions or [])}
        if (
            new_fp != old_fp
            or len(documents) != len(node.mentions or [])
            or current_doc_ids != expected_doc_ids
        ):
            await self._async_patch_base_projection_metadata(
                entity_kind="node", entity_id=node.safe_get_id(),
                metadata_patch={"node_refs_fp": new_fp},
            )
            await self._async_index_node_refs(node)

    async def _async_maybe_reindex_edge_refs(self, edge: Edge) -> None:
        new_fp = _refs_fingerprint(edge.mentions or [])
        meta = await self._async_backend_call(
            "edge", "get", ids=[edge.safe_get_id()], include=["metadatas"]
        )
        metadatas = _backend_object(meta).get("metadatas") or []
        old_fp = metadatas[0].get("edge_refs_fp") if metadatas and metadatas[0] else None
        got = await self._async_backend_call(
            "edge_refs", "get", where={"edge_id": edge.id}, include=["documents"]
        )
        documents = _backend_object(got).get("documents") or []
        current_doc_ids = {json.loads(doc).get("doc_id") for doc in documents}
        expected_doc_ids = {getattr(ref, "doc_id", None) for ref in (edge.mentions or [])}
        if (
            new_fp != old_fp
            or len(documents) != len(edge.mentions or [])
            or current_doc_ids != expected_doc_ids
        ):
            await self._async_patch_base_projection_metadata(
                entity_kind="edge", entity_id=edge.safe_get_id(),
                metadata_patch={"edge_refs_fp": new_fp},
            )
            await self._async_index_edge_refs(edge)

    async def _async_fanout_endpoints_rows(self, edge: Edge, doc_id: str | None) -> list[dict[str, Any]]:
        async def endpoint_doc(endpoint_id: str, endpoint_kind: str) -> str | None:
            if doc_id is not None:
                return doc_id
            result = await self._async_backend_call(
                endpoint_kind, "get", ids=[endpoint_id], include=["metadatas"]
            )
            metadata = (_backend_object(result).get("metadatas") or [None])[0] or {}
            value = metadata.get("doc_id")
            if isinstance(value, str):
                return value
            if not self._allow_missing_doc_id(edge):
                raise Exception("doc_id is not string")
            return None

        rows: list[dict[str, Any]] = []
        for role, endpoint_ids, endpoint_kind in (
            ("src", edge.source_ids or [], "node"),
            ("tgt", edge.target_ids or [], "node"),
            ("src", getattr(edge, "source_edge_ids", []) or [], "edge"),
            ("tgt", getattr(edge, "target_edge_ids", []) or [], "edge"),
        ):
            for endpoint_id in endpoint_ids:
                row = {
                    "id": f"{edge.id}::{role}::{endpoint_kind}::{endpoint_id}",
                    "edge_id": edge.id,
                    "endpoint_id": endpoint_id,
                    "endpoint_type": endpoint_kind,
                    "role": role,
                    "causal_type": (edge.metadata or {}).get("causal_type"),
                    "relation": edge.relation,
                }
                value = await endpoint_doc(endpoint_id, endpoint_kind)
                if value is not None:
                    row["doc_id"] = value
                rows.append({key: value for key, value in row.items() if value is not None})
        return rows

    async def _async_post_write(self, *, entity_kind: str, entity: Node | Edge) -> None:
        """Keep existing derived-index behavior off the event loop."""
        import asyncio

        if self._e._phase1_enable_index_jobs:
            if entity_kind == "node":
                await asyncio.to_thread(
                    self._e.enqueue_index_jobs_for_node,
                    entity.safe_get_id(),
                    op="UPSERT",
                )
            else:
                await asyncio.to_thread(
                    self._e.enqueue_index_jobs_for_edge,
                    entity.safe_get_id(),
                    op="UPSERT",
                )
            await asyncio.to_thread(self._e.reconcile_indexes, max_jobs=50)
            return
        if entity_kind == "node":
            node = cast(Node, entity)
            await self._async_index_node_docs(node)
            await self._async_maybe_reindex_node_refs(node)
        else:
            edge = cast(Edge, entity)
            await self._async_maybe_reindex_edge_refs(edge)
            rows = await self._async_fanout_endpoints_rows(edge, None)
            if rows:
                kwargs = {
                    "ids": [row["id"] for row in rows],
                    "documents": [json.dumps(row) for row in rows],
                    "metadatas": rows,
                    "embeddings": [
                        await self._embedding_async(json.dumps(row)) for row in rows
                    ],
                }
                await self._async_backend_call("edge_endpoints", "upsert", **kwargs)

    async def _add_node_async_single_stage(
        self, node: Node, *, doc_id: str | None
    ) -> None:
        import asyncio

        if doc_id is not None:
            node.doc_id = doc_id
        await asyncio.to_thread(self._run_pre_add_node_hooks, node)
        document, metadata = await asyncio.to_thread(self.node_doc_and_meta, node)
        if node.embedding is None:
            node.embedding = await self._embedding_async(document)
        else:
            from ...utils.embedding_vectors import normalize_embedding_vector

            node.embedding = normalize_embedding_vector(node.embedding, allow_none=False)
        metadata["_class_name"] = type(node).__name__
        payload = node.model_dump(field_mode="backend", exclude=["embedding"])
        await asyncio.to_thread(
            self._e._append_event_for_entity,
            namespace=getattr(self._e, "namespace", "default"),
            entity_kind="node", entity_id=node.safe_get_id(), op="ADD",
            payload=payload if isinstance(payload, dict) else {}, required=True,
        )
        await self._async_semantic_upsert(
            entity_kind="node", entity_id=node.safe_get_id(),
            document=document, metadata=metadata,
            embedding=_required_embedding(node.embedding),
        )
        await self._async_post_write(entity_kind="node", entity=node)
        await asyncio.to_thread(
            self._e._emit_change,
            op="node.upsert", entity=EntityRefModel(
                kind="node", id=node.safe_get_id(),
                kg_graph_type=self._e.kg_graph_type, url=self._e.persist_directory,
            ), payload=_entity_payload(node),
        )
        return None

    async def _add_edge_async_single_stage(
        self, edge: Edge, *, doc_id: str | None
    ) -> None:
        import asyncio

        if doc_id is not None:
            edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = await asyncio.to_thread(
            self._e.adjudicate.split_endpoints, edge.source_ids, edge.target_ids
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        await asyncio.to_thread(self._e.persist.assert_endpoints_exist, edge)
        if await asyncio.to_thread(self._run_pre_add_edge_hooks, edge, pure=False):
            return None
        document = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        if edge.embedding is None:
            edge.embedding = await self._embedding_async(document)
        else:
            from ...utils.embedding_vectors import normalize_embedding_vector

            edge.embedding = normalize_embedding_vector(edge.embedding, allow_none=False)
        metadata = await asyncio.to_thread(self.enrich_edge_meta, edge)
        payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
        await asyncio.to_thread(
            self._e._append_event_for_entity,
            namespace=getattr(self._e, "namespace", "default"),
            entity_kind="edge", entity_id=edge.safe_get_id(), op="ADD",
            payload=payload if isinstance(payload, dict) else {}, required=True,
        )
        await self._async_semantic_upsert(
            entity_kind="edge", entity_id=edge.safe_get_id(),
            document=document, metadata=metadata,
            embedding=_required_embedding(edge.embedding),
        )
        await self._async_post_write(entity_kind="edge", entity=edge)
        await asyncio.to_thread(
            self._e._emit_change,
            op="edge.upsert", entity=EntityRefModel(
                kind="edge", id=edge.safe_get_id(),
                kg_graph_type=self._e.kg_graph_type, url=self._e.persist_directory,
            ), payload=_entity_payload(edge),
        )
        return None

    def _run_pre_add_node_hooks(self, node: Node) -> None:
        for hook in list(getattr(self._e, "pre_add_node_hooks", []) or []):
            hook(node)

    def _run_pre_add_edge_hooks(self, edge: Edge, *, pure: bool) -> bool:
        hook_name = "pre_add_pure_edge_hooks" if pure else "pre_add_edge_hooks"
        for hook in list(getattr(self._e, hook_name, []) or []):
            if bool(hook(edge)):
                return True
        return False

    def _allow_missing_doc_id(self, edge: Edge) -> bool:
        for hook in list(
            getattr(self._e, "allow_missing_doc_id_on_endpoint_rows_hooks", []) or []
        ):
            if bool(hook(edge)):
                return True
        return False

    def _rust_postgres_meta(self) -> RustEnginePostgresMetaStore | None:
        from ..rust_postgres_session import RustEnginePostgresMetaStore

        meta = getattr(self._e, "meta_sqlite", None)
        return (
            cast(RustEnginePostgresMetaStore, meta)
            if isinstance(meta, RustEnginePostgresMetaStore)
            else None
        )

    def uses_rust_postgres_authority(self) -> bool:
        return self._rust_postgres_meta() is not None

    def patch_base_projection_metadata(
        self, *, entity_kind: str, entity_id: str, metadata_patch: dict[str, Any]
    ) -> None:
        meta = self._rust_postgres_meta()
        if meta is not None:
            table = (
                self._projection_backend().nodes.name
                if entity_kind == "node"
                else self._projection_backend().edges.name
            )
            meta.patch_graph_projection_metadata(
                namespace=getattr(self._e, "namespace", "default"),
                workspace_id=None,
                graph_space=None,
                table=table,
                entity_id=entity_id,
                document=None,
                metadata_patch=metadata_patch,
                patch_document_metadata=False,
            )
            return
        update = getattr(self._e.backend, f"{entity_kind}_update")
        run_awaitable_blocking(
            update(ids=[entity_id], metadatas=[metadata_patch])
        )

    def _rust_postgres_add(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata: dict[str, Any],
        embedding: Sequence[float],
        payload: dict[str, Any],
        enqueue_index_jobs: bool = True,
    ) -> bool:
        meta = self._rust_postgres_meta()
        if meta is None:
            return False
        table = (
            self._projection_backend().nodes.name
            if entity_kind == "node"
            else self._projection_backend().edges.name
            if entity_kind == "edge"
            else self._projection_backend().documents.name
            if entity_kind == "document"
            else self._projection_backend().domains.name
        )
        with meta.transaction():
            meta.apply_graph_mutation(
                namespace=getattr(self._e, "namespace", "default"),
                workspace_id=metadata.get("workspace_id"),
                graph_space=metadata.get("graph_space"),
                table=table,
                entity_kind=entity_kind,
                event_id=str(uuid.uuid4()),
                op="ADD",
                record={
                    "id": entity_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": list(embedding),
                },
                payload=payload,
                embedding_dim=self._projection_backend().embedding_dim,
            )
            if enqueue_index_jobs and entity_kind == "node":
                self._e.enqueue_index_jobs_for_node(entity_id, op="UPSERT")
            elif enqueue_index_jobs and entity_kind == "edge":
                self._e.enqueue_index_jobs_for_edge(entity_id, op="UPSERT")
        return True

    def _rust_postgres_projection_upsert(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata: dict[str, Any],
        embedding: Sequence[float],
        enqueue_index_jobs: bool = True,
    ) -> bool:
        meta = self._rust_postgres_meta()
        if meta is None or not getattr(self._e, "_disable_event_log", False):
            return False
        table = (
            self._projection_backend().nodes.name
            if entity_kind == "node"
            else self._projection_backend().edges.name
        )
        with meta.transaction():
            meta.upsert_graph_projection(
                namespace=getattr(self._e, "namespace", "default"),
                workspace_id=metadata.get("workspace_id"),
                graph_space=metadata.get("graph_space"),
                table=table,
                record={
                    "id": entity_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": list(embedding),
                },
                embedding_dim=self._projection_backend().embedding_dim,
            )
            if enqueue_index_jobs and getattr(
                self._e, "_phase1_enable_index_jobs", False
            ):
                if entity_kind == "node":
                    self._e.enqueue_index_jobs_for_node(entity_id, op="UPSERT")
                else:
                    self._e.enqueue_index_jobs_for_edge(entity_id, op="UPSERT")
        return True

    def _rust_postgres_delete_edges(self, edge_ids: list[str]) -> bool:
        meta = self._rust_postgres_meta()
        if meta is None:
            return False
        with meta.transaction():
            for edge_id in edge_ids:
                result = meta.apply_graph_delete_mutation(
                    namespace=getattr(self._e, "namespace", "default"),
                    workspace_id=None,
                    graph_space=None,
                    table=self._projection_backend().edges.name,
                    entity_kind="edge",
                    event_id=str(uuid.uuid4()),
                    entity_id=edge_id,
                    payload={"entity_id": edge_id},
                )
                if result is not None:
                    self._e.enqueue_index_jobs_for_edge(edge_id, op="DELETE")
        try:
            self._e.reconcile_indexes(max_jobs=50)
        except Exception:
            pass
        return True

    def rust_postgres_delete_existing(
        self, *, entity_kind: str, entity_ids: list[str]
    ) -> bool:
        meta = self._rust_postgres_meta()
        if meta is None:
            return False
        table = {
            "node": self._projection_backend().nodes.name,
            "edge": self._projection_backend().edges.name,
            "document": self._projection_backend().documents.name,
            "domain": self._projection_backend().domains.name,
        }[entity_kind]
        with meta.transaction():
            for entity_id in entity_ids:
                result = meta.apply_graph_delete_mutation(
                    namespace=getattr(self._e, "namespace", "default"),
                    workspace_id=None,
                    graph_space=None,
                    table=table,
                    entity_kind=entity_kind,
                    event_id=str(uuid.uuid4()),
                    entity_id=entity_id,
                    payload={"entity_id": entity_id},
                )
                if result is None:
                    continue
                if entity_kind == "node":
                    self._e.enqueue_index_jobs_for_node(entity_id, op="DELETE")
                elif entity_kind == "edge":
                    self._e.enqueue_index_jobs_for_edge(entity_id, op="DELETE")
        if entity_kind in {"node", "edge"}:
            try:
                self._e.reconcile_indexes(max_jobs=50)
            except Exception:
                pass
        return True

    def rust_postgres_replace_existing(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata_patch: dict[str, Any],
        payload: dict[str, Any],
    ) -> bool:
        meta = self._rust_postgres_meta()
        if meta is None:
            return False
        table = (
            self._projection_backend().nodes.name
            if entity_kind == "node"
            else self._projection_backend().edges.name
        )
        with meta.transaction():
            result = meta.apply_graph_metadata_patch_mutation(
                namespace=getattr(self._e, "namespace", "default"),
                workspace_id=None,
                graph_space=None,
                table=table,
                entity_kind=entity_kind,
                event_id=str(uuid.uuid4()),
                op="REPLACE",
                entity_id=entity_id,
                document=document,
                metadata_patch=metadata_patch,
                payload=payload,
            )
            if result is None:
                raise RuntimeError(
                    f"{entity_kind} {entity_id!r} disappeared during native replace"
                )
            if entity_kind == "node":
                self._e.enqueue_index_jobs_for_node(entity_id, op="UPSERT")
            else:
                self._e.enqueue_index_jobs_for_edge(entity_id, op="UPSERT")
        try:
            self._e.reconcile_indexes(max_jobs=50)
        except Exception:
            pass
        return True

    def enrich_edge_meta(self, edge: Edge) -> JsonObject:
        node_endpoint_count = len(edge.source_ids or []) + len(edge.target_ids or [])
        edge_endpoint_count = len(getattr(edge, "source_edge_ids", []) or []) + len(
            getattr(edge, "target_edge_ids", []) or []
        )
        total_endpoint_count = node_endpoint_count + edge_endpoint_count
        md = dict(getattr(edge, "metadata", None) or {})
        base_metadata: dict[str, Any] = dict(md)
        base_metadata.update(
            strip_none(
                {
                    "doc_id": edge.doc_id,
                    "relation": edge.relation,
                    "source_ids": json_or_none(edge.source_ids),
                    "target_ids": json_or_none(edge.target_ids),
                    "source_edge_ids": json_or_none(getattr(edge, "source_edge_ids", None)),
                    "target_edge_ids": json_or_none(getattr(edge, "target_edge_ids", None)),
                    "type": edge.type,
                    "summary": edge.summary,
                    "domain_id": edge.domain_id,
                    "canonical_entity_id": edge.canonical_entity_id,
                    "properties": json_or_none(edge.properties),
                    "references": json_or_none(
                        [
                            r.model_dump(field_mode="backend")
                            for r in (getattr(edge, "mentions", None) or [])
                        ]
                    ),
                    "node_endpoint_count": node_endpoint_count,
                    "edge_endpoint_count": edge_endpoint_count,
                    "total_endpoint_count": total_endpoint_count,
                }
            )
        )
        if self._e.kg_graph_type == "workflow":
            from ...runtime.models import WorkflowEdge

            edge = cast(WorkflowEdge, edge)
            edge_metadata = edge.metadata
            base_metadata.update(
                strip_none(
                    {
                        "entity_type": edge_metadata.get("entity_type"),
                        "workflow_id": edge_metadata.get("workflow_id"),
                        "wf_priority": edge_metadata.get("wf_priority")
                        or edge_metadata.get("priority"),
                        "wf_is_default": edge_metadata.get("wf_is_default")
                        or edge_metadata.get("is_default"),
                        "wf_predicate": edge_metadata.get("wf_predicate")
                        or edge_metadata.get("predicate"),
                        "wf_multiplicity": edge_metadata.get("wf_multiplicity")
                        or edge_metadata.get("multiplicity"),
                    }
                )
            )
        return base_metadata

    def _add_node_impl(self, node: Node, doc_id: str | None = None) -> None:
        """Commit canonical node event, then converge projections around it.

        For non-native backends, event append is required before the backend write.
        A projection failure may therefore leave a recoverable event without a row;
        an event-store failure leaves no backend projection. Derived rows remain
        asynchronous/best-effort according to existing index-job settings.
        """
        if self._e.persistence_mode == "two_stage":
            adapter = self._e.two_stage_projection_adapter
            if adapter is None:
                raise RuntimeError("two-stage projection adapter is unavailable")
            if doc_id is not None:
                node.doc_id = doc_id
            self._run_pre_add_node_hooks(node)
            doc, meta = self.node_doc_and_meta(node)
            meta["_class_name"] = type(node).__name__
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            with self._e.uow():
                self._e._append_event_for_entity(
                    namespace=getattr(self._e, "namespace", "default"),
                    entity_kind="node",
                    entity_id=node.safe_get_id(),
                    op="ADD",
                    payload=payload if isinstance(payload, dict) else {},
                    required=True,
                )
                result = adapter.add_node(node, doc_id=doc_id)
            enqueue = getattr(adapter, "enqueue_embedding_job", None)
            if callable(enqueue):
                enqueue(entity_kind="node", entity_id=node.safe_get_id(), op="UPSERT")
            return result
        if doc_id is not None:
            node.doc_id = doc_id
        self._run_pre_add_node_hooks(node)

        doc, meta = self.node_doc_and_meta(node)
        if node.embedding is None:
            node.embedding = self._e.embed.iterative_defensive_emb(doc)
        node.embedding = normalize_embedding_vector(node.embedding, allow_none=False)
        meta["_class_name"] = type(node).__name__

        payload = node.model_dump(field_mode="backend", exclude=["embedding"])
        native_added = self._rust_postgres_projection_upsert(
            entity_kind="node",
            entity_id=node.safe_get_id(),
            document=doc,
            metadata=meta,
            embedding=_required_embedding(node.embedding),
        ) or self._rust_postgres_add(
            entity_kind="node",
            entity_id=node.safe_get_id(),
            document=doc,
            metadata=meta,
            embedding=_required_embedding(node.embedding),
            payload=payload if isinstance(payload, dict) else {},
        )

        if not native_added:
            self._e._append_event_for_entity(
                namespace=getattr(self._e, "namespace", "default"),
                entity_kind="node",
                entity_id=node.safe_get_id(),
                op="ADD",
                payload=payload if isinstance(payload, dict) else {},
                required=True,
            )
            run_awaitable_blocking(self._e.backend.node_add(
                ids=[node.safe_get_id()],
                documents=[doc],
                embeddings=[_required_embedding(node.embedding)]
                if node.embedding is not None
                else [self._e.embed.iterative_defensive_emb(str(doc))],
                metadatas=[meta],
            ))

        if self._e._phase1_enable_index_jobs:
            if not native_added:
                self._e.enqueue_index_jobs_for_node(node.safe_get_id(), op="UPSERT")
            self._e.reconcile_indexes(max_jobs=50)
        else:
            self.index_node_docs(node)
            self.maybe_reindex_node_refs(node)
        self._e._emit_change(
            op="node.upsert",
            entity=EntityRefModel(
                kind="node",
                id=node.safe_get_id(),
                kg_graph_type=self._e.kg_graph_type,
                url=self._e.persist_directory,
            ),
            payload=_entity_payload(node),
        )

    def _add_edge_impl(self, edge: Edge, doc_id: str | None = None) -> None:
        """Persist an edge only after all referenced endpoints already exist.

        Edge ingest is structurally strict: missing node or edge endpoints are
        rejected before the base edge row is written. After persistence, the method
        follows the same fast-path-versus-index-job split as nodes for refs and
        edge_endpoints fanout, so derived projections may converge after the base
        edge is durable.
        """
        if self._e.persistence_mode == "two_stage":
            adapter = self._e.two_stage_projection_adapter
            if adapter is None:
                raise RuntimeError("two-stage projection adapter is unavailable")
            if doc_id is not None:
                edge.doc_id = doc_id
            s_nodes, s_edges, t_nodes, t_edges = self._e.adjudicate.split_endpoints(
                edge.source_ids, edge.target_ids
            )
            edge.source_ids = s_nodes
            edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
            edge.target_ids = t_nodes
            edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
            self._e.persist.assert_endpoints_exist(edge)
            if self._run_pre_add_edge_hooks(edge, pure=False):
                return
            payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
            with self._e.uow():
                self._e._append_event_for_entity(
                    namespace=getattr(self._e, "namespace", "default"),
                    entity_kind="edge",
                    entity_id=edge.safe_get_id(),
                    op="ADD",
                    payload=payload if isinstance(payload, dict) else {},
                    required=True,
                )
                result = adapter.add_edge(edge, doc_id=doc_id)
            enqueue = getattr(adapter, "enqueue_embedding_job", None)
            if callable(enqueue):
                enqueue(entity_kind="edge", entity_id=edge.safe_get_id(), op="UPSERT")
            return result
        if doc_id is not None:
            edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = self._e.adjudicate.split_endpoints(
            edge.source_ids, edge.target_ids
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        self._e.persist.assert_endpoints_exist(edge)
        if self._run_pre_add_edge_hooks(edge, pure=False):
            return

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        if edge.embedding is None:
            edge.embedding = self._e.embed.iterative_defensive_emb(str(doc))
        edge.embedding = normalize_embedding_vector(edge.embedding, allow_none=False)

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        base_metadata = [self.enrich_edge_meta(edge)]
        payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
        native_added = self._rust_postgres_projection_upsert(
            entity_kind="edge",
            entity_id=edge.safe_get_id(),
            document=str(doc),
            metadata=base_metadata[0],
            embedding=_required_embedding(edge.embedding),
        ) or self._rust_postgres_add(
            entity_kind="edge",
            entity_id=edge.safe_get_id(),
            document=str(doc),
            metadata=base_metadata[0],
            embedding=_required_embedding(edge.embedding),
            payload=payload if isinstance(payload, dict) else {},
        )
        if not native_added:
            self._e._append_event_for_entity(
                namespace=getattr(self._e, "namespace", "default"),
                entity_kind="edge",
                entity_id=edge.safe_get_id(),
                op="ADD",
                payload=payload if isinstance(payload, dict) else {},
                required=True,
            )
            run_awaitable_blocking(self._e.backend.edge_add(
                ids=[edge.safe_get_id()],
                documents=[str(doc)],
                embeddings=[_required_embedding(edge.embedding)]
                if edge.embedding is not None
                else [self._e.embed.iterative_defensive_emb(str(doc))],
                metadatas=base_metadata,
            ))

        if self._e._phase1_enable_index_jobs:
            if not native_added:
                self._e.enqueue_index_jobs_for_edge(edge.safe_get_id(), op="UPSERT")
            self._e.reconcile_indexes(max_jobs=50)
        else:
            self.maybe_reindex_edge_refs(edge)
            rows = self.fanout_endpoints_rows(edge, doc_id)
            if rows:
                ep_ids = [r["id"] for r in rows]
                ep_docs = [json.dumps(r) for r in rows]
                ep_metas: list[dict] = rows
                run_awaitable_blocking(self._e.backend.edge_endpoints_add(
                    ids=ep_ids,
                    documents=ep_docs,
                    metadatas=ep_metas,
                    embeddings=[
                        self._e.embed.iterative_defensive_emb(str(d)) for d in ep_docs
                    ],
                ))

        self._e._emit_change(
            op="edge.upsert",
            entity=EntityRefModel(
                kind="edge",
                id=edge.safe_get_id(),
                kg_graph_type=self._e.kg_graph_type,
                url=self._e.persist_directory,
            ),
            payload=_entity_payload(edge),
        )

    def add_pure_node(self, node: PureChromaNode) -> None:
        if node.id is None:
            raise ValueError("pure node id must not be None")
        doc, meta = node_doc_and_meta_util(node)
        if meta.get("doc_id"):
            meta.pop("doc_id")
        embedding = normalize_embedding_vector(
            node.embedding
            if node.embedding is not None
            else self._e.embed.iterative_defensive_emb(str(doc)),
            allow_none=False,
        )
        assert embedding is not None
        payload = node.model_dump(field_mode="backend", exclude=["embedding"])
        native_added = self._rust_postgres_projection_upsert(
            entity_kind="node",
            entity_id=cast(str, node.id),
            document=doc,
            metadata=meta,
            embedding=_required_embedding(embedding),
            enqueue_index_jobs=False,
        ) or self._rust_postgres_add(
            entity_kind="node",
            entity_id=cast(str, node.id),
            document=doc,
            metadata=meta,
            embedding=embedding,
            payload=payload if isinstance(payload, dict) else {},
            enqueue_index_jobs=False,
        )
        if not native_added:
            run_awaitable_blocking(
                self._e.backend.node_add(
                    ids=[cast(str, node.id)],
                    documents=[doc],
                    embeddings=[embedding],
                    metadatas=[meta],
                )
            )

    def add_pure_edge(self, edge: PureChromaEdge) -> None:
        """Low-level edge add without endpoint fanout or duplicate checks."""
        if edge.id is None:
            raise ValueError("pure edge id must not be None")
        s_nodes, s_edges, t_nodes, t_edges = self._e.adjudicate.split_endpoints(
            edge.source_ids,
            edge.target_ids,
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (getattr(edge, "source_edge_ids", []) or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (getattr(edge, "target_edge_ids", []) or []) + t_edges
        self._e.persist.assert_endpoints_exist(edge)
        if self._run_pre_add_edge_hooks(cast(Edge, edge), pure=True):
            return

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        metadata = self.enrich_edge_meta(cast(Edge, edge))
        embedding = normalize_embedding_vector(
            edge.embedding
            if edge.embedding is not None
            else self._e.embed.iterative_defensive_emb(str(doc)),
            allow_none=False,
        )
        assert embedding is not None
        payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
        native_added = self._rust_postgres_projection_upsert(
            entity_kind="edge",
            entity_id=cast(str, edge.id),
            document=str(doc),
            metadata=metadata,
            embedding=embedding,
            enqueue_index_jobs=False,
        ) or self._rust_postgres_add(
            entity_kind="edge",
            entity_id=cast(str, edge.id),
            document=str(doc),
            metadata=metadata,
            embedding=embedding,
            payload=payload if isinstance(payload, dict) else {},
            enqueue_index_jobs=False,
        )
        if not native_added:
            run_awaitable_blocking(
                self._e.backend.edge_add(
                    ids=[cast(str, edge.id)],
                    documents=[str(doc)],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )
            )

    def add_document(self, document: Document) -> None:
        if document.embeddings is None:
            document.embeddings = self._e.embed.iterative_defensive_emb(
                str(document.content)
            )
        document.embeddings = normalize_embedding_vector(
            document.embeddings, allow_none=False
        )
        metadata = strip_none(
            {
                "doc_id": document.id,
                "type": document.type,
                "metadata": json_or_none(document.metadata),
                "domain_id": document.domain_id,
                "processed": document.processed,
            }
        )
        payload = document.model_dump(field_mode="backend", exclude=["embeddings"])
        native_added = self._rust_postgres_add(
            entity_kind="document",
            entity_id=document.id,
            document=str(document.content),
            metadata=metadata,
            embedding=cast(Sequence[float], document.embeddings),
            payload=payload if isinstance(payload, dict) else {},
        )
        if not native_added:
            run_awaitable_blocking(self._e.backend.document_add(
                ids=[document.id],
                documents=[str(document.content)],
                embeddings=[cast(Sequence[float], document.embeddings)]
                if document.embeddings is not None
                else [
                    normalize_embedding_vector(
                        self._e.embed.iterative_defensive_emb(str(document.content)),
                        allow_none=False,
                    )
                ],
                metadatas=[metadata],
            ))
        self._e._emit_change(
            op="doc.upsert",
            entity=EntityRefModel(
                kind="doc_node",
                id=document.id,
                kg_graph_type=self._e.kg_graph_type,
                url=self._e.persist_directory,
            ),
            payload=_entity_payload(document),
        )

    def add_domain(self, domain: Domain) -> None:
        if domain.id is None:
            raise ValueError("domain id is required for persistence")
        document = domain.model_dump_json()
        metadata = self._e.chroma_sanitize_metadata(
            {"name": domain.name, "description": domain.description}
        )
        embedding = normalize_embedding_vector(
            self._e.embed.iterative_defensive_emb(str(document)), allow_none=False
        )
        payload = domain.model_dump()
        native_added = self._rust_postgres_add(
            entity_kind="domain",
            entity_id=domain.id,
            document=document,
            metadata=metadata,
            embedding=_required_embedding(embedding),
            payload=payload if isinstance(payload, dict) else {},
        )
        if not native_added:
            run_awaitable_blocking(self._e.backend.domain_add(
                ids=[domain.id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding],
            ))

    # Index/metadata helpers
    def index_node_docs(self, node: Node) -> list[str]:
        doc_ids = extract_doc_ids_from_refs(node.mentions)

        run_awaitable_blocking(self._e.backend.node_docs_delete(where={"node_id": node.id}))
        if doc_ids:
            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict] = []
            for did in doc_ids:
                rid = f"{node.id}::{did}"
                row = {"id": rid, "node_id": node.id, "doc_id": did, "mention_count": 1}
                ids.append(rid)
                docs.append(json.dumps(row))
                metas.append(row)
            run_awaitable_blocking(self._e.backend.node_docs_add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=[self._e.embed.iterative_defensive_emb(d) for d in docs],
            ))

        current = run_awaitable_blocking(self._e.backend.node_get(ids=[node.safe_get_id()], include=["metadatas"]))
        cur_meta = (_backend_object(current).get("metadatas") or [None])[0] or {}
        new_doc_ids_json = json.dumps(doc_ids)
        if cur_meta.get("doc_ids") != new_doc_ids_json:
            self.patch_base_projection_metadata(
                entity_kind="node",
                entity_id=node.safe_get_id(),
                metadata_patch={"doc_ids": new_doc_ids_json},
            )

        return doc_ids

    def index_node_refs(
        self, node: Node | None = None, **kwargs: object
    ) -> list[str]:
        if node is None:
            candidate = kwargs.get("node")
            if not isinstance(candidate, Node):
                raise TypeError("index_node_refs requires a Node")
            node = candidate
        self.delete_node_ref_rows(node.id)

        ids, docs, metas = [], [], []
        for i, mention in enumerate(node.mentions or []):
            for j, span in enumerate(mention.spans):
                rid = f"{node.id}::mention::{i}::span::{j}"
                did = getattr(span, "doc_id", None) or node.doc_id
                ver = getattr(span, "verification", None)
                row = strip_none(
                    {
                        "id": rid,
                        "node_id": node.id,
                        "doc_id": did,
                        "insertion_method": getattr(span, "insertion_method", None),
                        "verification_method": getattr(ver, "method", None),
                        "is_verified": getattr(ver, "is_verified", None),
                        "verificication_score": getattr(ver, "score", None),
                        "page_number": getattr(span, "start_page", None),
                        "excerpt": getattr(span, "excerpt", None),
                        "start_char": getattr(span, "start_char", None),
                        "end_char": getattr(span, "end_char", None),
                    }
                )
                ids.append(rid)
                docs.append(json.dumps(row))
                metas.append(row)

        if ids:
            run_awaitable_blocking(self._e.backend.node_refs_add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=[self._e._iterative_defensive_emb(str(d)) for d in docs],
            ))
        return ids

    def index_edge_refs(
        self, edge: Edge | None = None, **kwargs: object
    ) -> list[str]:
        if edge is None:
            candidate = kwargs.get("edge")
            if not isinstance(candidate, Edge):
                raise TypeError("index_edge_refs requires an Edge")
            edge = candidate
        self.delete_edge_ref_rows(edge.id)

        ids, docs, metas = [], [], []
        for i, ref in enumerate(edge.mentions or []):
            rid = f"{edge.id}::ref::{i}"
            did = getattr(ref, "doc_id", None) or edge.doc_id
            ver = getattr(ref, "verification", None)
            row = strip_none(
                {
                    "id": rid,
                    "edge_id": edge.id,
                    "doc_id": did,
                    "insertion_method": getattr(ref, "insertion_method", None),
                    "verification_method": getattr(ver, "method", None),
                    "is_verified": getattr(ver, "is_verified", None),
                    "verificication_score": getattr(ver, "score", None),
                    "start_page": getattr(ref, "start_page", None),
                    "end_page": getattr(ref, "end_page", None),
                    "start_char": getattr(ref, "start_char", None),
                    "end_char": getattr(ref, "end_char", None),
                }
            )
            ids.append(rid)
            docs.append(json.dumps(row))
            metas.append(row)

        if ids:
            run_awaitable_blocking(self._e.backend.edge_refs_add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=[self._e._iterative_defensive_emb(str(d)) for d in docs],
            ))
        return ids

    def fanout_endpoints_rows(
        self, edge: Edge, doc_id: str | None
    ) -> list[dict[str, object]]:
        """Build derived edge_endpoints rows from the current edge payload.

        Rows inherit doc_id from the explicit argument when available; otherwise the
        helper reads doc_id from the referenced node or edge metadata. Missing
        doc_ids are allowed only when _allow_missing_doc_id permits them. These rows
        are rebuildable projections and are safe to regenerate from the base edge.
        """

        def _maybe_doc_for_edge(eid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = run_awaitable_blocking(self._e.backend.edge_get(ids=[eid], include=["metadatas"]))
            metadata = _backend_object(meta).get("metadatas")
            if metadata and metadata[0]:
                if isinstance(metadata[0].get("doc_id"), str):
                    return str(metadata[0].get("doc_id"))
                if not self._allow_missing_doc_id(edge):
                    raise Exception("doc_id is not string")
            return None

        def _per_node_doc(nid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = run_awaitable_blocking(self._e.backend.node_get(ids=[nid], include=["metadatas"]))
            metadata = _backend_object(meta).get("metadatas")
            if metadata and metadata[0]:
                if isinstance(metadata[0].get("doc_id"), str):
                    return str(metadata[0].get("doc_id"))
                if not self._allow_missing_doc_id(edge):
                    raise Exception("doc_id is not string")
            return None

        rows: list[dict] = []
        for role, node_ids in (
            ("src", edge.source_ids or []),
            ("tgt", edge.target_ids or []),
        ):
            for nid in node_ids:
                r = {
                    "id": f"{edge.id}::{role}::node::{nid}",
                    "edge_id": edge.id,
                    "endpoint_id": nid,
                    "endpoint_type": "node",
                    "role": role,
                    "causal_type": (edge.metadata or {}).get("causal_type"),
                    "relation": edge.relation,
                }
                did = _per_node_doc(nid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})

        for role, eids in (
            ("src", getattr(edge, "source_edge_ids", []) or []),
            ("tgt", getattr(edge, "target_edge_ids", []) or []),
        ):
            for mid in eids:
                r = {
                    "id": f"{edge.id}::{role}::edge::{mid}",
                    "edge_id": edge.id,
                    "endpoint_id": mid,
                    "endpoint_type": "edge",
                    "role": role,
                    "causal_type": (edge.metadata or {}).get("causal_type"),
                    "relation": edge.relation,
                }
                did = _maybe_doc_for_edge(mid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})

        return [{k: v for k, v in r.items() if v is not None} for r in rows]

    def node_doc_and_meta(self, node: Node) -> tuple[str, JsonObject]:
        return cast(tuple[str, JsonObject], node_doc_and_meta_util(node))

    def edge_doc_and_meta(self, edge: Edge) -> tuple[str, JsonObject]:
        return cast(tuple[str, JsonObject], edge_doc_and_meta_util(edge))

    def strip_none(self, d: Mapping[str, _T]) -> dict[str, _T]:
        return strip_none(d)

    def json_or_none(self, value: object) -> str | None:
        return json_or_none(value)

    def delete_edge_ref_rows(
        self, edge_id: str | None = None, **kwargs: object
    ) -> None:
        if edge_id is None:
            value = kwargs.get("edge_id")
            if not isinstance(value, str):
                raise TypeError("delete_edge_ref_rows requires an edge_id")
            edge_id = value
        got = run_awaitable_blocking(self._e.backend.edge_refs_get(where={"edge_id": edge_id}, include=[]))
        ids = _backend_object(got).get("ids") or []
        if ids:
            run_awaitable_blocking(self._e.backend.edge_refs_delete(ids=ids))
        return None

    def delete_node_ref_rows(
        self, node_id: str | None = None, **kwargs: object
    ) -> None:
        if node_id is None:
            value = kwargs.get("node_id")
            if not isinstance(value, str):
                raise TypeError("delete_node_ref_rows requires a node_id")
            node_id = value
        got = run_awaitable_blocking(self._e.backend.node_refs_get(where={"node_id": node_id}, include=[]))
        ids = _backend_object(got).get("ids") or []
        if ids:
            run_awaitable_blocking(self._e.backend.node_refs_delete(ids=ids))
        return None

    def maybe_reindex_edge_refs(self, edge: Edge, *, force: bool = False) -> None:
        """Repair edge_refs when mention fingerprints or observed rows drift.

        The fingerprint is stored on the base edge metadata, but the method also
        checks row count and doc_id membership so partial deletes or stale rows are
        repaired even when the stored fingerprint still matches. force=True bypasses
        the drift short-circuit.
        """
        new_fp = _refs_fingerprint(edge.mentions or [])
        meta = run_awaitable_blocking(self._e.backend.edge_get(ids=[edge.safe_get_id()], include=["metadatas"]))
        old_fp = None
        metadatas = _backend_object(meta).get("metadatas")
        if metadatas and metadatas[0]:
            old_fp = metadatas[0].get("edge_refs_fp")

        got = run_awaitable_blocking(self._e.backend.edge_refs_get(
            where={"edge_id": edge.id}, include=["documents"]
        ))
        current_rows = _backend_object(got).get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (edge.mentions or [])}
        count_ok = len(current_rows) == len(edge.mentions or [])
        docset_ok = current_doc_ids == expect_doc_ids

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self.patch_base_projection_metadata(
                entity_kind="edge",
                entity_id=edge.safe_get_id(),
                metadata_patch={"edge_refs_fp": new_fp},
            )
            self.index_edge_refs(edge)

    def maybe_reindex_node_refs(self, node: Node, *, force: bool = False) -> None:
        """Repair node_refs when mention fingerprints or observed rows drift.

        The base node metadata stores the last reference fingerprint, but count and
        doc_id-set checks are also used so derived rows are rebuilt after partial
        corruption or manual deletes. force=True skips the fingerprint equality
        optimization and always reindexes.
        """
        new_fp = _refs_fingerprint(node.mentions or [])
        meta = run_awaitable_blocking(self._e.backend.node_get(ids=[node.safe_get_id()], include=["metadatas"]))
        old_fp = None
        metadatas = _backend_object(meta).get("metadatas")
        if metadatas and metadatas[0]:
            old_fp = metadatas[0].get("node_refs_fp")

        got = run_awaitable_blocking(self._e.backend.node_refs_get(
            where={"node_id": node.id}, include=["documents"]
        ))
        current_rows = _backend_object(got).get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (node.mentions or [])}
        count_ok = len(current_rows) == len(node.mentions or [])
        docset_ok = current_doc_ids == expect_doc_ids

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self.patch_base_projection_metadata(
                entity_kind="node",
                entity_id=node.safe_get_id(),
                metadata_patch={"node_refs_fp": new_fp},
            )
            self.index_node_refs(node)

    def prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        """Remove references to doc_id from node; delete node_docs link; refresh denormalized meta."""
        got = run_awaitable_blocking(self._e.backend.node_get(
            ids=[node_id], include=["documents", "metadatas"]
        ))
        docs = _backend_object(got).get("documents")
        if not (docs and docs[0]):
            return False
        node = Node.model_validate_json(docs[0])
        before = sum(len(grounding.spans) for grounding in (node.mentions or []))
        for groundings in node.mentions or []:
            filtered_spans = [
                span for span in groundings.spans if span.doc_id != doc_id
            ]
            groundings.spans = filtered_spans
        node.mentions = [
            grounding for grounding in (node.mentions or []) if grounding.spans
        ]

        after = sum(len(grounding.spans) for grounding in (node.mentions or []))
        changed = after != before
        if changed:
            if not node.mentions:
                raise ValueError(
                    f"cannot prune final reference from node {node_id!r}; "
                    "tombstone or replace the node instead"
                )
            document = node.model_dump_json(field_mode="backend")
            _, metadata = self.node_doc_and_meta(node)
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            native_replaced = self.rust_postgres_replace_existing(
                entity_kind="node",
                entity_id=node_id,
                document=document,
                metadata_patch=metadata,
                payload=payload if isinstance(payload, dict) else {},
            )
            if not native_replaced:
                run_awaitable_blocking(
                    self._e.backend.node_update(ids=[node_id], documents=[document])
                )
            run_awaitable_blocking(self._e.backend.node_docs_delete(
                where={"$and": [{"node_id": node_id}, {"doc_id": doc_id}]}
            ))
            self.index_node_docs(node)
        return changed

    def rebuild_edge_refs_for_doc(self, doc_id: str) -> int:
        eps = run_awaitable_blocking(self._e.backend.edge_endpoints_get(
            where={"doc_id": doc_id}, include=["documents"]
        ))
        edge_ids = list(
            {json.loads(d)["edge_id"] for d in (_backend_object(eps).get("documents") or [])}
        )
        if not edge_ids:
            return 0
        got = run_awaitable_blocking(self._e.backend.edge_get(ids=edge_ids, include=["documents"]))
        cnt = 0
        for js in _backend_object(got).get("documents") or []:
            e = Edge.model_validate_json(js)
            self.index_edge_refs(e)
            cnt += 1
        return cnt

    def rebuild_all_edge_refs(self) -> int:
        got = run_awaitable_blocking(self._e.backend.edge_get())
        total = 0
        for eid in _backend_object(got).get("ids") or []:
            edges = run_awaitable_blocking(self._e.backend.edge_get(ids=[eid], include=["documents"]))
            if edge_docs := _backend_object(edges).get("documents"):
                e = Edge.model_validate_json(edge_docs[0])
                self.index_edge_refs(e)
                total += 1
        return total

    def rebuild_node_refs_for_doc(self, doc_id: str) -> int:
        node_ids = []
        if hasattr(self._e, "node_docs_collection"):
            rows = run_awaitable_blocking(self._e.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["documents"]
            ))
            node_ids = list(
                {json.loads(d)["node_id"] for d in (_backend_object(rows).get("documents") or [])}
            )
        else:
            got = run_awaitable_blocking(self._e.backend.node_get(
                where={"doc_id": doc_id}, include=["documents"]
            ))
            node_ids = list(_backend_object(got).get("ids") or [])

        if not node_ids:
            return 0

        got = run_awaitable_blocking(self._e.backend.node_get(ids=node_ids, include=["documents"]))
        cnt = 0
        for js in _backend_object(got).get("documents") or []:
            n = Node.model_validate_json(js)
            self.index_node_refs(n)
            cnt += 1
        return cnt

    def rebuild_all_node_refs(self) -> int:
        got = run_awaitable_blocking(self._e.backend.node_get())
        total = 0
        for nid in _backend_object(got).get("ids") or []:
            doc = run_awaitable_blocking(self._e.backend.node_get(ids=[nid], include=["documents"]))
            if nod_docs := _backend_object(doc).get("documents"):
                n = Node.model_validate_json(nod_docs[0])
                self.index_node_refs(n)
                total += 1
        return total

    def delete_edges_by_ids(self, edge_ids: list[str]) -> None:
        if not edge_ids:
            return
        if self._rust_postgres_delete_edges(edge_ids):
            return
        run_awaitable_blocking(self._e.backend.edge_delete(ids=edge_ids))
        run_awaitable_blocking(
            self._e.backend.edge_endpoints_delete(
                where=cast(dict[str, object], {"edge_id": {"$in": edge_ids}})
            )
        )
