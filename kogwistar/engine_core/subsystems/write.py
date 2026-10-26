from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)
from typing_extensions import TypedDict
from pydantic import TypeAdapter

from ...cdc.change_event import EntityRefModel
from ...json_types import JsonObject, decode_json_object
from ...typing_interfaces import EmbeddingFunctionLike, ProjectionBackendLike
from ...utils.embedding_vectors import normalize_embedding_vector
from ..async_compat import run_awaitable_blocking
from ..models import (
    Document,
    Domain,
    Edge,
    Grounding,
    Node,
    PureChromaEdge,
    PureChromaNode,
    Span,
)
from ..storage_backend import AsyncTwoStageProjectionAdapter, StorageBackend
from ...implementation_selection import implementation_selection_bound
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
_DEFERRED_INDEX_RECONCILIATION: ContextVar[frozenset[int]] = ContextVar(
    "kogwistar_deferred_index_reconciliation", default=frozenset()
)
_ProjectionEntityKind: TypeAlias = Literal["node", "edge"]
_GraphEntityKind: TypeAlias = Literal["node", "edge", "document", "domain"]
_AsyncBackendCollectionKey: TypeAlias = Literal[
    "node",
    "edge",
    "document",
    "domain",
    "edge_endpoints",
    "edge_refs",
    "node_docs",
    "node_refs",
]
_AsyncBackendMethod: TypeAlias = Literal["add", "delete", "get", "update", "upsert"]
_BackendOperation: TypeAlias = Callable[..., object]
_ReferenceOwnerField: TypeAlias = Literal["node_id", "edge_id"]
_EndpointRole: TypeAlias = Literal["src", "tgt"]


def _engine_identity(engine: object) -> int:
    return id(getattr(engine, "_real", engine))


def _validate_graph_model_contract(
    entity: Node | Edge | PureChromaNode | PureChromaEdge,
    *,
    exclude: set[str] | list[str] | None = None,
    dump_format: Literal["python", "json"] = "json",
) -> JsonObject | None:
    """Run optional Rust model validation before any write-side effect."""

    from ..._rust_bridge import contract_implementation_mode, validate_graph_model_family

    if contract_implementation_mode() == "python":
        return None
    from ..graph_contracts import build_graph_model_envelope

    envelope = build_graph_model_envelope(
        entity,
        exclude=exclude,
        dump_format=dump_format,
    )
    payload = decode_json_object(envelope, label="graph model envelope")
    validate_graph_model_family(payload)
    return payload


@runtime_checkable
class _AsyncSemanticUpsertBackend(Protocol):
    def async_node_upsert(self, **kwargs: object) -> object: ...

    def async_edge_upsert(self, **kwargs: object) -> object: ...


@runtime_checkable
class _AsyncCallBackend(Protocol):
    def async_call(
        self,
        collection_key: _AsyncBackendCollectionKey,
        method: _AsyncBackendMethod,
        **kwargs: object,
    ) -> object: ...


class _AsyncCollectionFacade(Protocol):
    def add(self, **kwargs: object) -> object: ...

    def delete(self, **kwargs: object) -> object: ...

    def get(self, **kwargs: object) -> object: ...

    def update(self, **kwargs: object) -> object: ...

    def upsert(self, **kwargs: object) -> object: ...


@runtime_checkable
class _AsyncFacadeBackend(Protocol):
    _is_async_engine: bool
    _nodes_c: _AsyncCollectionFacade
    _edges_c: _AsyncCollectionFacade
    _documents_c: _AsyncCollectionFacade
    _domains_c: _AsyncCollectionFacade
    _edge_endpoints_c: _AsyncCollectionFacade
    _edge_refs_c: _AsyncCollectionFacade
    _node_docs_c: _AsyncCollectionFacade
    _node_refs_c: _AsyncCollectionFacade


def _sync_backend_operation(
    backend: StorageBackend,
    collection_key: _AsyncBackendCollectionKey,
    method: _AsyncBackendMethod,
) -> _BackendOperation:
    """Resolve the declared sync backend operation without reflection."""
    operations: dict[tuple[_AsyncBackendCollectionKey, _AsyncBackendMethod], _BackendOperation] = {
        ("node", "add"): backend.node_add,
        ("node", "delete"): backend.node_delete,
        ("node", "get"): backend.node_get,
        ("node", "update"): backend.node_update,
        ("node", "upsert"): backend.node_upsert,
        ("edge", "add"): backend.edge_add,
        ("edge", "delete"): backend.edge_delete,
        ("edge", "get"): backend.edge_get,
        ("edge", "update"): backend.edge_update,
        ("edge", "upsert"): backend.edge_upsert,
        ("document", "add"): backend.document_add,
        ("document", "delete"): backend.document_delete,
        ("document", "get"): backend.document_get,
        ("document", "update"): backend.document_update,
        ("document", "upsert"): backend.document_upsert,
        ("domain", "add"): backend.domain_add,
        ("domain", "delete"): backend.domain_delete,
        ("domain", "get"): backend.domain_get,
        ("domain", "update"): backend.domain_update,
        ("domain", "upsert"): backend.domain_upsert,
        ("edge_endpoints", "add"): backend.edge_endpoints_add,
        ("edge_endpoints", "delete"): backend.edge_endpoints_delete,
        ("edge_endpoints", "get"): backend.edge_endpoints_get,
        ("edge_endpoints", "update"): backend.edge_endpoints_update,
        ("edge_endpoints", "upsert"): backend.edge_endpoints_upsert,
        ("edge_refs", "add"): backend.edge_refs_add,
        ("edge_refs", "delete"): backend.edge_refs_delete,
        ("edge_refs", "get"): backend.edge_refs_get,
        ("edge_refs", "update"): backend.edge_refs_update,
        ("edge_refs", "upsert"): backend.edge_refs_upsert,
        ("node_docs", "add"): backend.node_docs_add,
        ("node_docs", "delete"): backend.node_docs_delete,
        ("node_docs", "get"): backend.node_docs_get,
        ("node_docs", "update"): backend.node_docs_update,
        ("node_docs", "upsert"): backend.node_docs_upsert,
        ("node_refs", "add"): backend.node_refs_add,
        ("node_refs", "delete"): backend.node_refs_delete,
        ("node_refs", "get"): backend.node_refs_get,
        ("node_refs", "update"): backend.node_refs_update,
        ("node_refs", "upsert"): backend.node_refs_upsert,
    }
    return operations[(collection_key, method)]


class _BackendResponse(TypedDict, total=False):
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, object] | None]


_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


def _validated_json_object(value: object) -> JsonObject:
    """Decode a Python mapping into the recursive JSON contract."""

    return _JSON_OBJECT_ADAPTER.validate_python(value)


class _BackendPayloadEntity(Protocol):
    def model_dump(
        self,
        *,
        field_mode: Iterable[str] | str | None = None,
        dump_format: Literal["python", "json"] = "python",
        exclude: list[str] | None = None,
    ) -> dict[str, object]: ...


class _AsyncEmbeddingProvider(Protocol):
    def __call__(
        self, documents: list[str], /
    ) -> Awaitable[list[list[float]]]: ...


@runtime_checkable
class _EmbeddingJobEnqueuer(Protocol):
    """Optional sync adapter capability for post-commit embedding work."""

    def enqueue_embedding_job(
        self, *, entity_kind: str, entity_id: str, op: str
    ) -> None: ...


@runtime_checkable
class _IdentityLockStore(Protocol):
    def lock_entity_identity(
        self, *, namespace: str, entity_kind: str, entity_id: str
    ) -> None: ...


@dataclass(frozen=True)
class CreateNodeResult:
    """Outcome of canonical create-once admission for one node identity."""

    status: Literal["created", "already_exists", "conflict"]
    entity_id: str
    revision: int | None = None
    message: str | None = None

    @property
    def created(self) -> bool:
        return self.status == "created"


def _required_embedding(value: list[float] | None) -> list[float]:
    if value is None:
        raise RuntimeError("embedding provider returned no vector")
    return list(value)


def _enqueue_embedding_job_if_supported(
    adapter: object, *, entity_kind: _ProjectionEntityKind, entity_id: str
) -> None:
    if isinstance(adapter, _EmbeddingJobEnqueuer):
        adapter.enqueue_embedding_job(
            entity_kind=entity_kind, entity_id=entity_id, op="UPSERT"
        )


def _entity_payload(entity: _BackendPayloadEntity) -> JsonObject:
    try:
        payload = entity.model_dump(
            field_mode="backend",
            dump_format="json",
            exclude=["embedding"],
        )
    except TypeError:
        # Keep compatibility with ordinary Pydantic models that do not
        # expose Kogwistar's mode-aware dump parameters.
        payload = entity.model_dump(exclude=["embedding"])
    return cast(JsonObject, payload)


def _backend_object(value: object) -> _BackendResponse:
    """Validate the result fields consumed from vector/backend adapters."""

    if not isinstance(value, dict):
        return {}
    response: _BackendResponse = {}
    ids = value.get("ids")
    if ids is not None:
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise TypeError("backend response field 'ids' must be a list of strings")
        response["ids"] = ids
    documents = value.get("documents")
    if documents is not None:
        if not isinstance(documents, list) or not all(
            isinstance(item, str) for item in documents
        ):
            raise TypeError("backend response field 'documents' must be a list of strings")
        response["documents"] = documents
    metadatas = value.get("metadatas")
    if metadatas is not None:
        if not isinstance(metadatas, list) or not all(
            item is None
            or (
                isinstance(item, dict)
                and all(isinstance(key, str) for key in item)
            )
            for item in metadatas
        ):
            raise TypeError(
                "backend response field 'metadatas' must be a list of string-keyed objects"
            )
        response["metadatas"] = cast(
            list[dict[str, object] | None], metadatas
        )
    return response


def _refs_fingerprint(refs: Sequence[Grounding] | None) -> str:
    payload = [reference.model_dump(dump_format="json") for reference in refs or ()]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _node_reference_spans(node: Node) -> list[tuple[int, int, Span]]:
    return [
        (mention_index, span_index, span)
        for mention_index, mention in enumerate(node.mentions or [])
        for span_index, span in enumerate(mention.spans)
    ]


def _node_reference_doc_ids(node: Node) -> set[str | None]:
    return {span.doc_id or node.doc_id for _, _, span in _node_reference_spans(node)}


def _edge_reference_spans(edge: Edge) -> list[tuple[int, int, Span]]:
    return [
        (mention_index, span_index, span)
        for mention_index, mention in enumerate(edge.mentions or [])
        for span_index, span in enumerate(mention.spans)
    ]


class WriteSubsystem(NamespaceProxy["GraphKnowledgeEngine"]):
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        super().__init__(engine)

    @contextmanager
    def defer_index_reconciliation(self, *, max_jobs: int = 50) -> Iterator[None]:
        """Defer queued index projection work until a related write batch ends.

        Canonical graph writes remain durable as they happen. Only the
        best-effort/queued derived-index drain is coalesced at the outermost
        context for this engine.
        """
        if max_jobs < 0:
            raise ValueError("max_jobs must be non-negative")
        engine_key = _engine_identity(self._e)
        previous = _DEFERRED_INDEX_RECONCILIATION.get()
        token = _DEFERRED_INDEX_RECONCILIATION.set(previous | {engine_key})
        try:
            yield
        finally:
            _DEFERRED_INDEX_RECONCILIATION.reset(token)
        if (
            engine_key not in previous
            and self._e._phase1_enable_index_jobs
            and max_jobs > 0
        ):
            self._e.reconcile_indexes(max_jobs=max_jobs)

    def _index_reconciliation_is_deferred(self) -> bool:
        return _engine_identity(self._e) in _DEFERRED_INDEX_RECONCILIATION.get()

    def _projection_backend(self) -> ProjectionBackendLike:
        """Narrow optional native projection attributes at their boundary."""

        return cast(ProjectionBackendLike, self._e.backend)

    def _attach_graph_namespace(self, metadata: JsonObject) -> JsonObject:
        """Persist the engine's authoritative graph namespace on projections."""
        declared_namespace = metadata.get("namespace")
        if declared_namespace is not None and declared_namespace != self._e.namespace:
            raise ValueError(
                "graph metadata namespace conflicts with the engine namespace"
            )
        metadata["namespace"] = self._e.namespace
        return metadata

    # Canonical write API
    @implementation_selection_bound
    def add_node(self, node: Node, doc_id: str | None = None) -> None:
        self._add_node_impl(node, doc_id=doc_id)

    @implementation_selection_bound
    def create_node_if_absent(
        self, node: Node, doc_id: str | None = None
    ) -> CreateNodeResult:
        """Atomically admit a never-before-seen canonical node identity.

        This is intentionally separate from ``add_node``. Existing ingestion
        callers may rely on projection upsert behavior, while maintenance
        patches need create-once semantics at the canonical event boundary.
        A Chroma or other eventual projection is updated only after the
        identity admission has succeeded.
        """

        if doc_id is not None:
            node.doc_id = doc_id
        entity_id = node.safe_get_id()
        metadata_store = self._e.metadata

        payload = node.model_dump(field_mode="backend", exclude=["embedding"])
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        namespace = self._e.namespace

        with self._e.uow():
            if isinstance(metadata_store, _IdentityLockStore):
                metadata_store.lock_entity_identity(
                    namespace=namespace,
                    entity_kind="node",
                    entity_id=entity_id,
                )

            current = self._e.indexing.canonical_entity_revision(
                entity_kind="node", entity_id=entity_id, namespace=namespace
            )
            if current is not None:
                current_json = json.dumps(
                    current.payload, sort_keys=True, separators=(",", ":")
                )
                if current.state == "active" and current_json == payload_json:
                    return CreateNodeResult(
                        status="already_exists",
                        entity_id=entity_id,
                        revision=current.revision,
                    )
                return CreateNodeResult(
                    status="conflict",
                    entity_id=entity_id,
                    revision=current.revision,
                    message=(
                        "canonical node identity already has a different payload "
                        "or has been tombstoned"
                    ),
                )

            self._e._append_event_for_entity(
                namespace=namespace,
                entity_kind="node",
                entity_id=entity_id,
                op="ADD",
                payload=cast(dict[str, object], payload),
                required=True,
            )
            previous_disable_event_log = self._e._disable_event_log
            self._e._disable_event_log = True
            try:
                self._add_node_impl(node, doc_id=doc_id)
            finally:
                self._e._disable_event_log = previous_disable_event_log

            revision = self._e.indexing.canonical_entity_revision(
                entity_kind="node", entity_id=entity_id, namespace=namespace
            )
            return CreateNodeResult(
                status="created",
                entity_id=entity_id,
                revision=revision.revision if revision is not None else None,
            )

    @implementation_selection_bound
    def add_edge(self, edge: Edge, doc_id: str | None = None) -> None:
        self._add_edge_impl(edge, doc_id=doc_id)

    @implementation_selection_bound
    async def add_node_async(self, node: Node, doc_id: str | None = None) -> None:
        """Admit a node through the configured async persistence mode."""
        import asyncio

        adapter = self._e.async_two_stage_projection_adapter
        if self._e.persistence_mode == "two_stage":
            if adapter is None:
                raise RuntimeError("async two-stage projection adapter is unavailable")
            if doc_id is not None:
                node.doc_id = doc_id
            await asyncio.to_thread(self._run_pre_add_node_hooks, node)
            await asyncio.to_thread(_validate_graph_model_contract, node)
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            await asyncio.to_thread(
                self._e._append_event_for_entity,
                namespace=self._e.namespace,
                entity_kind="node", entity_id=node.safe_get_id(), op="ADD",
                payload=payload if isinstance(payload, dict) else {}, required=True,
            )
            return await adapter.add_node(node, doc_id=doc_id)

        return await self._add_node_async_single_stage(node, doc_id=doc_id)

    @implementation_selection_bound
    async def add_edge_async(self, edge: Edge, doc_id: str | None = None) -> None:
        """Admit an edge through the configured async persistence mode."""
        adapter = self._e.async_two_stage_projection_adapter
        if self._e.persistence_mode == "two_stage":
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
        edge.source_edge_ids = (edge.source_edge_ids or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (edge.target_edge_ids or []) + t_edges
        await self._e.persist.assert_endpoints_exist_async(edge)
        if await asyncio.to_thread(self._run_pre_add_edge_hooks, edge, pure=False):
            return
        await asyncio.to_thread(_validate_graph_model_contract, edge)
        payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
        await asyncio.to_thread(
            self._e._append_event_for_entity,
            namespace=self._e.namespace,
            entity_kind="edge", entity_id=edge.safe_get_id(), op="ADD",
            payload=payload if isinstance(payload, dict) else {}, required=True,
        )
        return await adapter.add_edge(edge, doc_id=doc_id)

    async def _embedding_async(self, document: str) -> list[float]:
        """Await async providers; isolate legacy sync providers from the loop."""
        import asyncio
        import inspect

        from ...utils.embedding_vectors import normalize_embedding_vector

        provider = self._e._ef
        is_async = inspect.iscoroutinefunction(provider) or inspect.iscoroutinefunction(
            provider.__call__
        )
        if is_async:
            vectors: list[list[float]] = await cast(
                _AsyncEmbeddingProvider, provider
            )([str(document)])
        else:
            vectors = await asyncio.to_thread(
                cast(EmbeddingFunctionLike, provider), [str(document)]
            )
        raw = vectors[0] if vectors else None
        embedding = normalize_embedding_vector(raw, allow_none=False)
        if embedding is None:
            raise RuntimeError("embedding provider returned no vector")
        return list(embedding)

    async def _async_semantic_upsert(
        self, *, entity_kind: _ProjectionEntityKind, entity_id: str, document: str,
        metadata: JsonObject, embedding: list[float],
    ) -> None:
        import asyncio

        backend = self._e.backend
        kwargs = {
            "ids": [entity_id], "documents": [document],
            "embeddings": [embedding], "metadatas": [metadata],
        }
        if isinstance(backend, _AsyncSemanticUpsertBackend):
            direct = (
                backend.async_node_upsert
                if entity_kind == "node"
                else backend.async_edge_upsert
            )
            result = direct(**kwargs)
            if inspect.isawaitable(result):
                await result
            return
        sync_upsert = (
            backend.node_upsert
            if entity_kind == "node"
            else backend.edge_upsert
        )
        await asyncio.to_thread(sync_upsert, **kwargs)

    async def _async_backend_call(
        self,
        collection_key: _AsyncBackendCollectionKey,
        method: _AsyncBackendMethod,
        **kwargs: object,
    ) -> object:
        """Call native async backend verbs, with a sync compatibility fallback."""
        import asyncio
        import inspect

        backend = self._e.backend
        if isinstance(backend, _AsyncCallBackend):
            result = backend.async_call(collection_key, method, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        if isinstance(backend, _AsyncFacadeBackend) and backend._is_async_engine:
            facades: dict[_AsyncBackendCollectionKey, _AsyncCollectionFacade] = {
                "node": backend._nodes_c,
                "edge": backend._edges_c,
                "document": backend._documents_c,
                "domain": backend._domains_c,
                "edge_endpoints": backend._edge_endpoints_c,
                "edge_refs": backend._edge_refs_c,
                "node_docs": backend._node_docs_c,
                "node_refs": backend._node_refs_c,
            }
            facade = facades[collection_key]
            operations: dict[_AsyncBackendMethod, Callable[..., object]] = {
                "add": facade.add,
                "delete": facade.delete,
                "get": facade.get,
                "update": facade.update,
                "upsert": facade.upsert,
            }
            result = operations[method](**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        operation = _sync_backend_operation(backend, collection_key, method)
        return await asyncio.to_thread(operation, **kwargs)

    async def _async_patch_base_projection_metadata(
        self, *, entity_kind: _ProjectionEntityKind, entity_id: str, metadata_patch: JsonObject
    ) -> None:
        import asyncio

        if self._rust_postgres_meta_or_python_fallback(
            operation="async projection metadata patch"
        ) is not None:
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

    async def _async_delete_ref_rows(
        self,
        collection_key: Literal["node_refs", "edge_refs"],
        field: _ReferenceOwnerField,
        entity_id: str,
    ) -> None:
        got = await self._async_backend_call(
            collection_key, "get", where={field: entity_id}, include=[]
        )
        ids = _backend_object(got).get("ids") or []
        if ids:
            await self._async_backend_call(collection_key, "delete", ids=ids)

    async def _async_index_node_refs(self, node: Node) -> list[str]:
        await self._async_delete_ref_rows("node_refs", "node_id", node.safe_get_id())
        rows: list[JsonObject] = []
        reference_ids: list[str] = []
        for mention_index, span_index, span in _node_reference_spans(node):
            verification = span.verification
            reference_id = f"{node.id}::mention::{mention_index}::span::{span_index}"
            reference_ids.append(reference_id)
            rows.append(strip_none({
                "id": reference_id,
                "node_id": node.id,
                "doc_id": span.doc_id or node.doc_id,
                "insertion_method": span.insertion_method,
                "verification_method": verification.method if verification else None,
                "is_verified": verification.is_verified if verification else None,
                "verificication_score": verification.score if verification else None,
                "page_number": span.page_number,
                "excerpt": span.excerpt,
                "start_char": span.start_char,
                "end_char": span.end_char,
            }))
        if rows:
            await self._async_backend_call(
                "node_refs", "add",
                ids=reference_ids,
                documents=[json.dumps(row) for row in rows],
                metadatas=rows,
                embeddings=[
                    await self._embedding_async(json.dumps(row)) for row in rows
                ],
            )
        return reference_ids

    async def _async_index_edge_refs(self, edge: Edge) -> list[str]:
        await self._async_delete_ref_rows("edge_refs", "edge_id", edge.safe_get_id())
        rows: list[JsonObject] = []
        reference_ids: list[str] = []
        for mention_index, span_index, span in _edge_reference_spans(edge):
            verification = span.verification
            reference_id = f"{edge.id}::mention::{mention_index}::span::{span_index}"
            reference_ids.append(reference_id)
            rows.append(strip_none({
                "id": reference_id,
                "edge_id": edge.id,
                "doc_id": span.doc_id or edge.doc_id,
                "insertion_method": span.insertion_method,
                "verification_method": verification.method if verification else None,
                "is_verified": verification.is_verified if verification else None,
                "verificication_score": verification.score if verification else None,
                "page_number": span.page_number,
                "excerpt": span.excerpt,
                "start_char": span.start_char,
                "end_char": span.end_char,
            }))
        if rows:
            await self._async_backend_call(
                "edge_refs", "add",
                ids=reference_ids,
                documents=[json.dumps(row) for row in rows],
                metadatas=rows,
                embeddings=[
                    await self._embedding_async(json.dumps(row)) for row in rows
                ],
            )
        return reference_ids

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
        expected_doc_ids = _node_reference_doc_ids(node)
        if (
            new_fp != old_fp
            or len(documents) != len(_node_reference_spans(node))
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
        edge_spans = _edge_reference_spans(edge)
        expected_doc_ids = {span.doc_id or edge.doc_id for _, _, span in edge_spans}
        if (
            new_fp != old_fp
            or len(documents) != len(edge_spans)
            or current_doc_ids != expected_doc_ids
        ):
            await self._async_patch_base_projection_metadata(
                entity_kind="edge", entity_id=edge.safe_get_id(),
                metadata_patch={"edge_refs_fp": new_fp},
            )
            await self._async_index_edge_refs(edge)

    async def _async_fanout_endpoints_rows(self, edge: Edge, doc_id: str | None) -> list[JsonObject]:
        async def endpoint_doc(
            endpoint_id: str, endpoint_kind: _ProjectionEntityKind
        ) -> str | None:
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

        rows: list[JsonObject] = []
        endpoint_specs: tuple[
            tuple[_EndpointRole, list[str], _ProjectionEntityKind], ...
        ] = (
            ("src", edge.source_ids or [], "node"),
            ("tgt", edge.target_ids or [], "node"),
            ("src", edge.source_edge_ids or [], "edge"),
            ("tgt", edge.target_edge_ids or [], "edge"),
        )
        for role, endpoint_ids, endpoint_kind in endpoint_specs:
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

    async def _async_post_write(
        self, *, entity_kind: _ProjectionEntityKind, entity: Node | Edge
    ) -> None:
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
        model_envelope = await asyncio.to_thread(_validate_graph_model_contract, node)
        document, metadata = await asyncio.to_thread(self.node_doc_and_meta, node)
        if node.embedding is None:
            node.embedding = await self._embedding_async(document)
        else:
            from ...utils.embedding_vectors import normalize_embedding_vector

            node.embedding = normalize_embedding_vector(node.embedding, allow_none=False)
        metadata["_class_name"] = type(node).__name__
        payload = (
            model_envelope["payload"]
            if model_envelope is not None
            else node.model_dump(field_mode="backend", exclude=["embedding"])
        )
        native_added = False
        if self._rust_postgres_meta_or_python_fallback(
            operation="async node add"
        ) is not None:
            native_added = await asyncio.to_thread(
                self._rust_postgres_projection_upsert,
                entity_kind="node",
                entity_id=node.safe_get_id(),
                document=document,
                metadata=metadata,
                embedding=_required_embedding(node.embedding),
                enqueue_index_jobs=False,
            ) or await asyncio.to_thread(
                self._rust_postgres_add,
                entity_kind="node",
                entity_id=node.safe_get_id(),
                document=document,
                metadata=metadata,
                embedding=_required_embedding(node.embedding),
                payload=payload if isinstance(payload, dict) else {},
                model_envelope=model_envelope,
                enqueue_index_jobs=False,
            )
        if not native_added:
            async with self._e.async_uow():
                await self._e._append_event_for_entity_async(
                    namespace=self._e.namespace,
                    entity_kind="node",
                    entity_id=node.safe_get_id(),
                    op="ADD",
                    payload=payload if isinstance(payload, dict) else {},
                    required=True,
                )
                await self._async_semantic_upsert(
                    entity_kind="node",
                    entity_id=node.safe_get_id(),
                    document=document,
                    metadata=metadata,
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
        edge.source_edge_ids = (edge.source_edge_ids or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (edge.target_edge_ids or []) + t_edges
        await asyncio.to_thread(self._e.persist.assert_endpoints_exist, edge)
        if await asyncio.to_thread(self._run_pre_add_edge_hooks, edge, pure=False):
            return None
        model_envelope = await asyncio.to_thread(_validate_graph_model_contract, edge)
        document = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        if edge.embedding is None:
            edge.embedding = await self._embedding_async(document)
        else:
            from ...utils.embedding_vectors import normalize_embedding_vector

            edge.embedding = normalize_embedding_vector(edge.embedding, allow_none=False)
        metadata = await asyncio.to_thread(self.enrich_edge_meta, edge)
        payload = (
            model_envelope["payload"]
            if model_envelope is not None
            else edge.model_dump(field_mode="backend", exclude=["embedding"])
        )
        native_added = False
        if self._rust_postgres_meta_or_python_fallback(
            operation="async edge add"
        ) is not None:
            native_added = await asyncio.to_thread(
                self._rust_postgres_projection_upsert,
                entity_kind="edge",
                entity_id=edge.safe_get_id(),
                document=document,
                metadata=metadata,
                embedding=_required_embedding(edge.embedding),
                enqueue_index_jobs=False,
            ) or await asyncio.to_thread(
                self._rust_postgres_add,
                entity_kind="edge",
                entity_id=edge.safe_get_id(),
                document=document,
                metadata=metadata,
                embedding=_required_embedding(edge.embedding),
                payload=payload if isinstance(payload, dict) else {},
                model_envelope=model_envelope,
                enqueue_index_jobs=False,
            )
        if not native_added:
            async with self._e.async_uow():
                await self._e._append_event_for_entity_async(
                    namespace=self._e.namespace,
                    entity_kind="edge",
                    entity_id=edge.safe_get_id(),
                    op="ADD",
                    payload=payload if isinstance(payload, dict) else {},
                    required=True,
                )
                await self._async_semantic_upsert(
                    entity_kind="edge",
                    entity_id=edge.safe_get_id(),
                    document=document,
                    metadata=metadata,
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
        for hook in self._e.pre_add_node_hooks:
            hook(node)

    def _run_pre_add_edge_hooks(self, edge: Edge, *, pure: bool) -> bool:
        hooks = (
            self._e.pre_add_pure_edge_hooks
            if pure
            else self._e.pre_add_edge_hooks
        )
        for hook in hooks:
            if bool(hook(edge)):
                return True
        return False

    def _allow_missing_doc_id(self, edge: Edge) -> bool:
        for hook in self._e.allow_missing_doc_id_on_endpoint_rows_hooks:
            if bool(hook(edge)):
                return True
        return False

    def _rust_postgres_meta(self) -> RustEnginePostgresMetaStore | None:
        from ..rust_postgres_session import RustEnginePostgresMetaStore

        meta = self._e.metadata
        return (
            cast(RustEnginePostgresMetaStore, meta)
            if isinstance(meta, RustEnginePostgresMetaStore)
            else None
        )

    def _rust_postgres_meta_or_python_fallback(
        self, *, operation: str
    ) -> RustEnginePostgresMetaStore | None:
        from ..._rust_bridge import meta_store_implementation_mode
        from ..._rust_bridge import postgres_authority_implementation_mode

        selected_mode = meta_store_implementation_mode()
        authority_mode = postgres_authority_implementation_mode()
        meta = self._rust_postgres_meta()
        if selected_mode != "rust":
            if authority_mode == "rust":
                raise RuntimeError(
                    "Rust PostgreSQL authority is selected without the Rust "
                    f"metadata selector for {operation}; refusing split authority"
                )
            if meta is not None:
                return None
            return None
        if meta is None:
            if authority_mode == "rust":
                raise RuntimeError(
                    "Rust PostgreSQL authority is selected but the Rust PostgreSQL "
                    f"meta-store is unavailable for {operation}; refusing Python "
                    "persistence fallback"
                )
            return None
        return meta

    def uses_rust_postgres_authority(self) -> bool:
        return (
            self._rust_postgres_meta_or_python_fallback(
                operation="authority capability check"
            )
            is not None
        )

    def patch_base_projection_metadata(
        self, *, entity_kind: _ProjectionEntityKind, entity_id: str, metadata_patch: JsonObject
    ) -> None:
        meta = self._rust_postgres_meta_or_python_fallback(
            operation="projection metadata patch"
        )
        if meta is not None:
            table = (
                self._projection_backend().nodes.name
                if entity_kind == "node"
                else self._projection_backend().edges.name
            )
            meta.patch_graph_projection_metadata(
                namespace=self._e.namespace,
                workspace_id=None,
                graph_space=None,
                table=table,
                entity_id=entity_id,
                document=None,
                metadata_patch=metadata_patch,
                patch_document_metadata=False,
            )
            return
        update = (
            self._e.backend.node_update
            if entity_kind == "node"
            else self._e.backend.edge_update
        )
        run_awaitable_blocking(
            update(ids=[entity_id], metadatas=[metadata_patch])
        )

    def _rust_postgres_add(
        self,
        *,
        entity_kind: _GraphEntityKind,
        entity_id: str,
        document: str,
        metadata: JsonObject,
        embedding: Sequence[float],
        payload: JsonObject,
        model_envelope: JsonObject | None = None,
        enqueue_index_jobs: bool = True,
    ) -> bool:
        meta = self._rust_postgres_meta_or_python_fallback(operation="graph add")
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
            mutation_values: JsonObject = {
                "namespace": self._e.namespace,
                "workspace_id": metadata.get("workspace_id"),
                "graph_space": metadata.get("graph_space"),
                "table": table,
                "entity_kind": entity_kind,
                "event_id": str(uuid.uuid4()),
                "op": "ADD",
                "record": {
                    "id": entity_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": list(embedding),
                },
                "payload": payload,
                "embedding_dim": self._projection_backend().embedding_dim,
            }
            if model_envelope is not None:
                mutation_values["model_envelope"] = model_envelope
            meta.apply_graph_mutation(**mutation_values)
            if enqueue_index_jobs and entity_kind == "node":
                self._e.enqueue_index_jobs_for_node(entity_id, op="UPSERT")
            elif enqueue_index_jobs and entity_kind == "edge":
                self._e.enqueue_index_jobs_for_edge(entity_id, op="UPSERT")
        return True

    def _rust_postgres_projection_upsert(
        self,
        *,
        entity_kind: _ProjectionEntityKind,
        entity_id: str,
        document: str,
        metadata: JsonObject,
        embedding: Sequence[float],
        enqueue_index_jobs: bool = True,
    ) -> bool:
        meta = self._rust_postgres_meta()
        if meta is None or not self._e._disable_event_log:
            return False
        table = (
            self._projection_backend().nodes.name
            if entity_kind == "node"
            else self._projection_backend().edges.name
        )
        with meta.transaction():
            meta.upsert_graph_projection(
            namespace=self._e.namespace,
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
            if enqueue_index_jobs and self._e._phase1_enable_index_jobs:
                if entity_kind == "node":
                    self._e.enqueue_index_jobs_for_node(entity_id, op="UPSERT")
                else:
                    self._e.enqueue_index_jobs_for_edge(entity_id, op="UPSERT")
        return True

    def _ensure_rust_postgres_canonical_tombstone(
        self, *, entity_kind: Literal["node", "edge"], entity_id: str
    ) -> bool:
        """Make physical native purge safe after canonical lifecycle admission."""
        getter = (
            self._e.backend.node_get
            if entity_kind == "node"
            else self._e.backend.edge_get
        )
        current = _backend_object(
            run_awaitable_blocking(getter(ids=[entity_id], include=["metadatas"]))
        )
        if not current.get("ids"):
            return False
        metadata_rows = current.get("metadatas") or []
        metadata = metadata_rows[0] if metadata_rows else None
        if isinstance(metadata, Mapping) and str(
            metadata.get("lifecycle_status") or "active"
        ) in {"tombstoned", "deleted"}:
            return True
        tombstoned = (
            self._e.lifecycle.tombstone_node(entity_id, reason="native_graph_purge")
            if entity_kind == "node"
            else self._e.lifecycle.tombstone_edge(entity_id, reason="native_graph_purge")
        )
        if not tombstoned:
            raise RuntimeError(
                f"cannot physically purge active {entity_kind} {entity_id!r}"
                " without a canonical tombstone"
            )
        return True

    def _rust_postgres_delete_edges(self, edge_ids: list[str]) -> bool:
        meta = self._rust_postgres_meta_or_python_fallback(operation="edge delete")
        if meta is None:
            return False
        for edge_id in edge_ids:
            self._ensure_rust_postgres_canonical_tombstone(
                entity_kind="edge", entity_id=edge_id
            )
        with meta.transaction():
            for edge_id in edge_ids:
                result = meta.apply_graph_delete_mutation(
                    namespace=self._e.namespace,
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
        self, *, entity_kind: _GraphEntityKind, entity_ids: list[str]
    ) -> bool:
        meta = self._rust_postgres_meta_or_python_fallback(
            operation=f"{entity_kind} rollback delete"
        )
        if meta is None:
            return False
        table = {
            "node": self._projection_backend().nodes.name,
            "edge": self._projection_backend().edges.name,
            "document": self._projection_backend().documents.name,
            "domain": self._projection_backend().domains.name,
        }[entity_kind]
        if entity_kind == "node" or entity_kind == "edge":
            for entity_id in entity_ids:
                self._ensure_rust_postgres_canonical_tombstone(
                    entity_kind=entity_kind, entity_id=entity_id
                )
        with meta.transaction():
            for entity_id in entity_ids:
                result = meta.apply_graph_delete_mutation(
                    namespace=self._e.namespace,
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
        entity_kind: _ProjectionEntityKind,
        entity_id: str,
        document: str,
        metadata_patch: JsonObject,
        payload: JsonObject,
    ) -> bool:
        meta = self._rust_postgres_meta_or_python_fallback(
            operation=f"{entity_kind} replacement"
        )
        if meta is None:
            return False
        table = (
            self._projection_backend().nodes.name
            if entity_kind == "node"
            else self._projection_backend().edges.name
        )
        with meta.transaction():
            result = meta.apply_graph_metadata_patch_mutation(
                namespace=self._e.namespace,
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
        edge_endpoint_count = len(edge.source_edge_ids or []) + len(
            edge.target_edge_ids or []
        )
        total_endpoint_count = node_endpoint_count + edge_endpoint_count
        md = dict(edge.metadata or {})
        base_metadata: JsonObject = dict(md)
        base_metadata.update(
            strip_none(
                {
                    "doc_id": edge.doc_id,
                    "relation": edge.relation,
                    "source_ids": json_or_none(edge.source_ids),
                    "target_ids": json_or_none(edge.target_ids),
                    "source_edge_ids": json_or_none(edge.source_edge_ids),
                    "target_edge_ids": json_or_none(edge.target_edge_ids),
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
        return self._attach_graph_namespace(base_metadata)

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
            _validate_graph_model_contract(node)
            doc, meta = self.node_doc_and_meta(node)
            meta["_class_name"] = type(node).__name__
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            with self._e.uow():
                self._e._append_event_for_entity(
                    namespace=self._e.namespace,
                    entity_kind="node",
                    entity_id=node.safe_get_id(),
                    op="ADD",
                    payload=payload if isinstance(payload, dict) else {},
                    required=True,
                )
                result = adapter.add_node(node, doc_id=doc_id)
            _enqueue_embedding_job_if_supported(
                adapter, entity_kind="node", entity_id=node.safe_get_id()
            )
            return result
        if doc_id is not None:
            node.doc_id = doc_id
        self._run_pre_add_node_hooks(node)
        model_envelope = _validate_graph_model_contract(node)

        doc, meta = self.node_doc_and_meta(node)
        if node.embedding is None:
            node.embedding = self._e.embed.iterative_defensive_emb(doc)
        node.embedding = normalize_embedding_vector(node.embedding, allow_none=False)
        meta["_class_name"] = type(node).__name__

        payload = (
            model_envelope["payload"]
            if model_envelope is not None
            else node.model_dump(field_mode="backend", exclude=["embedding"])
        )
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
            model_envelope=model_envelope,
        )

        if not native_added:
            with self._e.uow():
                self._e._append_event_for_entity(
                    namespace=self._e.namespace,
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
            if not self._index_reconciliation_is_deferred():
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
            edge.source_edge_ids = (edge.source_edge_ids or []) + s_edges
            edge.target_ids = t_nodes
            edge.target_edge_ids = (edge.target_edge_ids or []) + t_edges
            self._e.persist.assert_endpoints_exist(edge)
            if self._run_pre_add_edge_hooks(edge, pure=False):
                return
            _validate_graph_model_contract(edge)
            payload = edge.model_dump(field_mode="backend", exclude=["embedding"])
            with self._e.uow():
                self._e._append_event_for_entity(
                    namespace=self._e.namespace,
                    entity_kind="edge",
                    entity_id=edge.safe_get_id(),
                    op="ADD",
                    payload=payload if isinstance(payload, dict) else {},
                    required=True,
                )
                result = adapter.add_edge(edge, doc_id=doc_id)
            _enqueue_embedding_job_if_supported(
                adapter, entity_kind="edge", entity_id=edge.safe_get_id()
            )
            return result
        if doc_id is not None:
            edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = self._e.adjudicate.split_endpoints(
            edge.source_ids, edge.target_ids
        )
        edge.source_ids = s_nodes
        edge.source_edge_ids = (edge.source_edge_ids or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (edge.target_edge_ids or []) + t_edges
        self._e.persist.assert_endpoints_exist(edge)
        if self._run_pre_add_edge_hooks(edge, pure=False):
            return
        model_envelope = _validate_graph_model_contract(edge)

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        if edge.embedding is None:
            edge.embedding = self._e.embed.iterative_defensive_emb(str(doc))
        edge.embedding = normalize_embedding_vector(edge.embedding, allow_none=False)

        base_metadata = [self.enrich_edge_meta(edge)]
        payload = (
            model_envelope["payload"]
            if model_envelope is not None
            else edge.model_dump(field_mode="backend", exclude=["embedding"])
        )
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
            model_envelope=model_envelope,
        )
        if not native_added:
            with self._e.uow():
                self._e._append_event_for_entity(
                    namespace=self._e.namespace,
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
            if not self._index_reconciliation_is_deferred():
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
        model_envelope = _validate_graph_model_contract(
            node,
            exclude=["embedding"],
            dump_format="python",
        )
        embedding = normalize_embedding_vector(
            node.embedding
            if node.embedding is not None
            else self._e.embed.iterative_defensive_emb(str(doc)),
            allow_none=False,
        )
        assert embedding is not None
        payload = (
            model_envelope["payload"]
            if model_envelope is not None
            else node.model_dump(field_mode="backend", exclude=["embedding"])
        )
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
            model_envelope=model_envelope,
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
        edge.source_edge_ids = (edge.source_edge_ids or []) + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = (edge.target_edge_ids or []) + t_edges
        self._e.persist.assert_endpoints_exist(edge)
        if self._run_pre_add_edge_hooks(cast(Edge, edge), pure=True):
            return

        doc = edge.model_dump_json(field_mode="backend", exclude=["embedding"])
        metadata = self.enrich_edge_meta(cast(Edge, edge))
        model_envelope = _validate_graph_model_contract(
            edge,
            exclude=["embedding"],
            dump_format="python",
        )
        embedding = normalize_embedding_vector(
            edge.embedding
            if edge.embedding is not None
            else self._e.embed.iterative_defensive_emb(str(doc)),
            allow_none=False,
        )
        assert embedding is not None
        payload = (
            model_envelope["payload"]
            if model_envelope is not None
            else edge.model_dump(field_mode="backend", exclude=["embedding"])
        )
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
            model_envelope=model_envelope,
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
        metadata = _validated_json_object(
            strip_none(
                {
                "doc_id": document.id,
                "type": document.type,
                "metadata": json_or_none(document.metadata),
                "domain_id": document.domain_id,
                "processed": document.processed,
                }
            )
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
        for mention_index, span_index, span in _node_reference_spans(node):
            rid = f"{node.id}::mention::{mention_index}::span::{span_index}"
            verification = span.verification
            row = strip_none(
                {
                    "id": rid,
                    "node_id": node.id,
                    "doc_id": span.doc_id or node.doc_id,
                    "insertion_method": span.insertion_method,
                    "verification_method": verification.method if verification else None,
                    "is_verified": verification.is_verified if verification else None,
                    "verificication_score": verification.score if verification else None,
                    "page_number": span.page_number,
                    "excerpt": span.excerpt,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
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
        for mention_index, span_index, span in _edge_reference_spans(edge):
            rid = f"{edge.id}::mention::{mention_index}::span::{span_index}"
            verification = span.verification
            row = strip_none(
                {
                    "id": rid,
                    "edge_id": edge.id,
                    "doc_id": span.doc_id or edge.doc_id,
                    "insertion_method": span.insertion_method,
                    "verification_method": verification.method if verification else None,
                    "is_verified": verification.is_verified if verification else None,
                    "verificication_score": verification.score if verification else None,
                    "page_number": span.page_number,
                    "excerpt": span.excerpt,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
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
            ("src", edge.source_edge_ids or []),
            ("tgt", edge.target_edge_ids or []),
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
        document, metadata = node_doc_and_meta_util(node)
        return document, self._attach_graph_namespace(metadata)

    def edge_doc_and_meta(self, edge: Edge) -> tuple[str, JsonObject]:
        document, metadata = edge_doc_and_meta_util(edge)
        return document, self._attach_graph_namespace(metadata)

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
        edge_spans = _edge_reference_spans(edge)
        expect_doc_ids = {span.doc_id or edge.doc_id for _, _, span in edge_spans}
        count_ok = len(current_rows) == len(edge_spans)
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
        expect_doc_ids = _node_reference_doc_ids(node)
        count_ok = len(current_rows) == len(_node_reference_spans(node))
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
