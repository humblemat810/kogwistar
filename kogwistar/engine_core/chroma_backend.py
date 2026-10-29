import hashlib
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .async_compat import run_awaitable_blocking
from .embedding_profile import EmbeddingStorageState
from ..implementation_selection import assert_transaction_entry_allowed
from .storage_backend import (
    AsyncTwoStageProjectionAdapter,
    AtomicMutationCapability,
    TwoStageProjectionAdapter,
    TwoStageProjectionCapability,
)

_VECTOR_COLLECTION_NAMES = (
    "nodes_index",
    "nodes",
    "edges",
    "edge_endpoints",
    "documents",
    "domains",
    "node_docs",
)
_CHROMA_JSON_METADATA_PREFIX = "__kogwistar_json_v1__:"
_CHROMA_MUTATION_METHODS = frozenset({"add", "upsert", "update", "delete"})


class ChromaCollection(Protocol):
    """Minimal collection surface used by the backend adapter."""

    def count(self) -> int: ...

    def get(self, **kwargs: object) -> object: ...

    def query(self, **kwargs: object) -> object: ...

    def add(self, **kwargs: object) -> object: ...

    def upsert(self, **kwargs: object) -> object: ...

    def update(self, **kwargs: object) -> object: ...

    def delete(self, **kwargs: object) -> object: ...


class ChromaClient(Protocol):
    """Minimal client surface needed to inspect persistent collections."""

    def list_collections(self) -> Sequence[object]: ...

    def get_collection(self, *, name: str) -> ChromaCollection: ...

    def get_or_create_collection(
        self, name: str, **kwargs: object
    ) -> ChromaCollection: ...


def _chroma_scope(persist_directory: str | None) -> str:
    # The profile binding is stored in this directory's metadata database.
    # A portable identity must therefore survive copying the complete bundle
    # to a new host/path; the metadata database provides the physical scope.
    if not persist_directory:
        return "chroma:ephemeral"
    root = Path(persist_directory)
    marker = root / ".kogwistar-storage-identity"
    try:
        root.mkdir(parents=True, exist_ok=True)
        identity = marker.read_text(encoding="ascii").strip()
        if not identity:
            raise ValueError("empty storage identity")
    except (FileNotFoundError, ValueError):
        identity = uuid.uuid4().hex
        try:
            with marker.open("x", encoding="ascii") as handle:
                handle.write(identity + "\n")
        except FileExistsError:
            identity = marker.read_text(encoding="ascii").strip()
    return f"chroma:bundle:{identity}"


def _legacy_chroma_scope(persist_directory: str | None) -> str:
    value = persist_directory or "chroma:ephemeral"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"chroma:{digest}"


class ChromaStorageInspector:
    """Inspect existing collections without passing an embedder to Chroma."""

    def __init__(self, client: ChromaClient, persist_directory: str | None) -> None:
        self._client = client
        self._persist_directory = (
            str(Path(persist_directory).expanduser().resolve())
            if persist_directory
            else None
        )

    def embedding_storage_scope(self) -> str:
        return _chroma_scope(self._persist_directory)

    def embedding_storage_scope_aliases(self) -> tuple[str, ...]:
        legacy = _legacy_chroma_scope(self._persist_directory)
        current = self.embedding_storage_scope()
        return (legacy,) if legacy != current else ()

    def inspect_embedding_storage(self) -> EmbeddingStorageState:
        existing = {
            str(getattr(item, "name", item))
            for item in self._client.list_collections()
        }
        counts: list[str] = []
        total = 0
        for name in _VECTOR_COLLECTION_NAMES:
            count = (
                int(self._client.get_collection(name=name).count())
                if name in existing
                else 0
            )
            total += count
            counts.append(f"{name}={count}")
        return EmbeddingStorageState(
            backend_kind="chroma",
            storage_scope=self.embedding_storage_scope(),
            persistent=self._persist_directory is not None,
            vector_count=total,
            details=tuple(counts),
        )


def _chroma_safe_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Adapt unsupported compound, null, and empty-list metadata for Chroma.

    Chroma accepts only scalar metadata values and flat lists. Compound JSON
    values use a tagged string representation and are restored on reads.
    Null values are omitted, preserving their effective ``metadata.get(key)``
    behavior. Inputs are never mutated.
    """
    metadatas = kwargs.get("metadatas")
    if not isinstance(metadatas, (list, tuple)):
        return kwargs
    sanitized: list[object] = []
    changed = False
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            sanitized.append(metadata)
            continue
        safe_metadata: dict[object, object] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if _is_compound_chroma_metadata(value):
                safe_metadata[key] = _CHROMA_JSON_METADATA_PREFIX + json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            elif isinstance(value, list) and not value:
                safe_metadata[key] = "[]"
            else:
                safe_metadata[key] = value
        sanitized.append(safe_metadata)
        changed = changed or safe_metadata != metadata
    if not changed:
        return kwargs
    return {**kwargs, "metadatas": sanitized}


def _is_compound_chroma_metadata(value: object) -> bool:
    if isinstance(value, Mapping):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, (Mapping, list, tuple)) for item in value)
    return False


def _restore_chroma_metadata_map(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    restored: dict[object, object] = {}
    for key, item in value.items():
        if (
            isinstance(item, str)
            and item.startswith(_CHROMA_JSON_METADATA_PREFIX)
        ):
            restored[key] = json.loads(item[len(_CHROMA_JSON_METADATA_PREFIX) :])
        else:
            restored[key] = item
    return restored


def _restore_chroma_result(value: object) -> object:
    if not isinstance(value, dict):
        return value
    restored = dict(value)
    if "metadatas" in restored:
        def restore_rows(rows: object) -> object:
            if isinstance(rows, list):
                return [restore_rows(row) for row in rows]
            return _restore_chroma_metadata_map(rows)

        restored["metadatas"] = restore_rows(restored["metadatas"])
    if "metadata" in restored:
        restored["metadata"] = _restore_chroma_metadata_map(restored["metadata"])
    return restored


class _AwaitableValue:
    def __init__(self, value: object) -> None:
        self._value = value

    def __await__(self) -> Iterator[object]:
        async def _done() -> object:
            return self._value

        return _done().__await__()

    def __getattr__(self, name: str) -> object:
        return getattr(self._value, name)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return repr(self._value)


class _AwaitableDict(dict):
    def __await__(self) -> Iterator[object]:
        async def _done() -> object:
            return self

        return _done().__await__()


def _awaitable_result(value: object) -> object:
    if isinstance(value, dict):
        return _AwaitableDict(value)
    return _AwaitableValue(value)


class ChromaBackend:
    """
    Thin wrapper around Chroma collections.

    Important: this class does NOT try to implement transactions; callers should
    treat vector writes as best-effort unless they use an outbox pattern.
    """

    atomic_mutation_capability = AtomicMutationCapability(
        mode="eventual",
        reason="Chroma writes are durable but do not provide multi-operation rollback",
    )
    two_stage_projection_capability: TwoStageProjectionCapability
    two_stage_projection_adapter: TwoStageProjectionAdapter | None
    async_two_stage_projection_adapter: AsyncTwoStageProjectionAdapter | None
    def __init__(
        self,
        *,
        node_index_collection: ChromaCollection,
        node_collection: ChromaCollection,
        edge_collection: ChromaCollection,
        edge_endpoints_collection: ChromaCollection,
        document_collection: ChromaCollection,
        domain_collection: ChromaCollection,
        node_docs_collection: ChromaCollection,
        node_refs_collection: ChromaCollection,
        edge_refs_collection: ChromaCollection,
        persist_directory: str | None = None,
        distance_metric: str = "cosine",
    ) -> None:
        self._collections: dict[str, ChromaCollection] = {
            "node_index": node_index_collection,
            "node": node_collection,
            "edge": edge_collection,
            "edge_endpoints": edge_endpoints_collection,
            "document": document_collection,
            "domain": domain_collection,
            "node_docs": node_docs_collection,
            "node_refs": node_refs_collection,
            "edge_refs": edge_refs_collection,
        }
        self._persist_directory = (
            str(Path(persist_directory).expanduser().resolve())
            if persist_directory
            else None
        )
        metric = str(distance_metric).strip().lower()
        if metric not in {"cosine", "ip", "l2"}:
            raise ValueError("distance_metric must be one of: cosine, ip, l2")
        self._vector_distance_kind = "squared_l2" if metric == "l2" else "distance"

    @property
    def vector_distance_kind(self) -> str:
        return self._vector_distance_kind

    def embedding_storage_scope(self) -> str:
        """Return a stable, secret-free identity for this Chroma directory."""

        return _chroma_scope(self._persist_directory)

    def inspect_embedding_storage(self) -> EmbeddingStorageState:
        """Count all collections that participate in semantic vector storage."""

        vector_keys = (
            "node_index",
            "node",
            "edge",
            "edge_endpoints",
            "document",
            "domain",
            "node_docs",
        )
        counts: list[str] = []
        total = 0
        for key in vector_keys:
            count = int(self._c(key).count())
            total += count
            counts.append(f"{key}={count}")
        return EmbeddingStorageState(
            backend_kind="chroma",
            storage_scope=self.embedding_storage_scope(),
            persistent=self._persist_directory is not None,
            vector_count=total,
            details=tuple(counts),
        )

    def _c(self, key: str) -> ChromaCollection:
        try:
            return self._collections[key]
        except KeyError as e:
            raise KeyError(f"Unknown collection_key={key!r}") from e

    def call(self, collection_key: str, method: str, **kwargs: object) -> object:
        if method in _CHROMA_MUTATION_METHODS:
            assert_transaction_entry_allowed()
        coll = self._c(collection_key)
        fn = getattr(coll, method)
        result = run_awaitable_blocking(fn(**_chroma_safe_kwargs(kwargs)))
        return _awaitable_result(_restore_chroma_result(result))

    # --- node_index ---
    def node_index_get(self, **kwargs: object) -> object:
        return self.call("node_index", "get", **kwargs)

    def node_index_query(self, **kwargs: object) -> object:
        return self.call("node_index", "query", **kwargs)

    def node_index_add(self, **kwargs: object) -> object:
        return self.call("node_index", "add", **kwargs)

    def node_index_upsert(self, **kwargs: object) -> object:
        return self.call("node_index", "upsert", **kwargs)

    def node_index_update(self, **kwargs: object) -> object:
        return self.call("node_index", "update", **kwargs)

    def node_index_delete(self, **kwargs: object) -> object:
        return self.call("node_index", "delete", **kwargs)

    # --- nodes ---
    def node_get(self, **kwargs: object) -> object:
        return self.call("node", "get", **kwargs)

    def node_query(self, **kwargs: object) -> object:
        return self.call("node", "query", **kwargs)

    def node_add(self, **kwargs: object) -> object:
        return self.call("node", "add", **kwargs)

    def node_upsert(self, **kwargs: object) -> object:
        return self.call("node", "upsert", **kwargs)

    def node_update(self, **kwargs: object) -> object:
        return self.call("node", "update", **kwargs)

    def node_delete(self, **kwargs: object) -> object:
        return self.call("node", "delete", **kwargs)

    # --- edges ---
    def edge_get(self, **kwargs: object) -> object:
        return self.call("edge", "get", **kwargs)

    def edge_query(self, **kwargs: object) -> object:
        return self.call("edge", "query", **kwargs)

    def edge_add(self, **kwargs: object) -> object:
        return self.call("edge", "add", **kwargs)

    def edge_upsert(self, **kwargs: object) -> object:
        return self.call("edge", "upsert", **kwargs)

    def edge_update(self, **kwargs: object) -> object:
        return self.call("edge", "update", **kwargs)

    def edge_delete(self, **kwargs: object) -> object:
        return self.call("edge", "delete", **kwargs)

    # --- edge_endpoints ---
    def edge_endpoints_get(self, **kwargs: object) -> object:
        return self.call("edge_endpoints", "get", **kwargs)

    def edge_endpoints_query(self, **kwargs: object) -> object:
        raise ValueError(
            "edge_endpoints is structural; semantic query is unsupported; "
            "use edge_endpoints_get with metadata filters"
        )

    def edge_endpoints_add(self, **kwargs: object) -> object:
        return self.call("edge_endpoints", "add", **kwargs)

    def edge_endpoints_upsert(self, **kwargs: object) -> object:
        return self.call("edge_endpoints", "upsert", **kwargs)

    def edge_endpoints_update(self, **kwargs: object) -> object:
        return self.call("edge_endpoints", "update", **kwargs)

    def edge_endpoints_delete(self, **kwargs: object) -> object:
        return self.call("edge_endpoints", "delete", **kwargs)

    # --- documents ---
    def document_get(self, **kwargs: object) -> object:
        return self.call("document", "get", **kwargs)

    def document_query(self, **kwargs: object) -> object:
        return self.call("document", "query", **kwargs)

    def document_add(self, **kwargs: object) -> object:
        return self.call("document", "add", **kwargs)

    def document_upsert(self, **kwargs: object) -> object:
        return self.call("document", "upsert", **kwargs)

    def document_update(self, **kwargs: object) -> object:
        return self.call("document", "update", **kwargs)

    def document_delete(self, **kwargs: object) -> object:
        return self.call("document", "delete", **kwargs)

    # --- domains ---
    def domain_get(self, **kwargs: object) -> object:
        return self.call("domain", "get", **kwargs)

    def domain_query(self, **kwargs: object) -> object:
        return self.call("domain", "query", **kwargs)

    def domain_add(self, **kwargs: object) -> object:
        return self.call("domain", "add", **kwargs)

    def domain_upsert(self, **kwargs: object) -> object:
        return self.call("domain", "upsert", **kwargs)

    def domain_update(self, **kwargs: object) -> object:
        return self.call("domain", "update", **kwargs)

    def domain_delete(self, **kwargs: object) -> object:
        return self.call("domain", "delete", **kwargs)

    # --- node_docs ---
    def node_docs_get(self, **kwargs: object) -> object:
        return self.call("node_docs", "get", **kwargs)

    def node_docs_query(self, **kwargs: object) -> object:
        return self.call("node_docs", "query", **kwargs)

    def node_docs_add(self, **kwargs: object) -> object:
        return self.call("node_docs", "add", **kwargs)

    def node_docs_upsert(self, **kwargs: object) -> object:
        return self.call("node_docs", "upsert", **kwargs)

    def node_docs_update(self, **kwargs: object) -> object:
        return self.call("node_docs", "update", **kwargs)

    def node_docs_delete(self, **kwargs: object) -> object:
        return self.call("node_docs", "delete", **kwargs)

    # --- node_refs ---
    def node_refs_get(self, **kwargs: object) -> object:
        return self.call("node_refs", "get", **kwargs)

    def node_refs_query(self, **kwargs: object) -> object:
        return self.call("node_refs", "query", **kwargs)

    def node_refs_add(self, **kwargs: object) -> object:
        return self.call("node_refs", "add", **kwargs)

    def node_refs_upsert(self, **kwargs: object) -> object:
        return self.call("node_refs", "upsert", **kwargs)

    def node_refs_update(self, **kwargs: object) -> object:
        return self.call("node_refs", "update", **kwargs)

    def node_refs_delete(self, **kwargs: object) -> object:
        return self.call("node_refs", "delete", **kwargs)

    # --- edge_refs ---
    def edge_refs_get(self, **kwargs: object) -> object:
        return self.call("edge_refs", "get", **kwargs)

    def edge_refs_query(self, **kwargs: object) -> object:
        return self.call("edge_refs", "query", **kwargs)

    def edge_refs_add(self, **kwargs: object) -> object:
        return self.call("edge_refs", "add", **kwargs)

    def edge_refs_upsert(self, **kwargs: object) -> object:
        return self.call("edge_refs", "upsert", **kwargs)

    def edge_refs_update(self, **kwargs: object) -> object:
        return self.call("edge_refs", "update", **kwargs)

    def edge_refs_delete(self, **kwargs: object) -> object:
        return self.call("edge_refs", "delete", **kwargs)


class AsyncChromaBackend(ChromaBackend):
    """Chroma facade retaining sync API while exposing non-blocking verbs."""

    is_async_backend = True

    async def async_call(self, collection_key: str, method: str, **kwargs: object) -> object:
        if method in _CHROMA_MUTATION_METHODS:
            assert_transaction_entry_allowed()
        coll = self._c(collection_key)
        result = getattr(coll, method)(**_chroma_safe_kwargs(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return _restore_chroma_result(result)

    async def async_node_get(self, **kwargs: object) -> object:
        return await self.async_call("node", "get", **kwargs)

    async def async_node_upsert(self, **kwargs: object) -> object:
        return await self.async_call("node", "upsert", **kwargs)

    async def async_node_delete(self, **kwargs: object) -> object:
        return await self.async_call("node", "delete", **kwargs)

    async def async_edge_get(self, **kwargs: object) -> object:
        return await self.async_call("edge", "get", **kwargs)

    async def async_edge_upsert(self, **kwargs: object) -> object:
        return await self.async_call("edge", "upsert", **kwargs)

    async def async_edge_delete(self, **kwargs: object) -> object:
        return await self.async_call("edge", "delete", **kwargs)
