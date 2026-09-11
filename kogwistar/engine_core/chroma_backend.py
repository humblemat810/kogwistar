import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict

from .async_compat import run_awaitable_blocking
from .embedding_profile import EmbeddingStorageState


_VECTOR_COLLECTION_NAMES = (
    "nodes_index",
    "nodes",
    "edges",
    "edge_endpoints",
    "documents",
    "domains",
    "node_docs",
)


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

    def __init__(self, client: Any, persist_directory: str | None) -> None:
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


def _chroma_safe_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Encode empty metadata lists rejected by Chroma without changing nodes."""
    metadatas = kwargs.get("metadatas")
    if not isinstance(metadatas, (list, tuple)):
        return kwargs
    sanitized: list[Any] = []
    changed = False
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            sanitized.append(metadata)
            continue
        safe_metadata = {
            key: ("[]" if isinstance(value, list) and not value else value)
            for key, value in metadata.items()
        }
        sanitized.append(safe_metadata)
        changed = changed or safe_metadata != metadata
    if not changed:
        return kwargs
    return {**kwargs, "metadatas": sanitized}


class _AwaitableValue:
    def __init__(self, value: Any):
        self._value = value

    def __await__(self):
        async def _done():
            return self._value

        return _done().__await__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return repr(self._value)


class _AwaitableDict(dict):
    def __await__(self):
        async def _done():
            return self

        return _done().__await__()


def _awaitable_result(value: Any) -> Any:
    if isinstance(value, dict):
        return _AwaitableDict(value)
    return _AwaitableValue(value)


class ChromaBackend:
    """
    Thin wrapper around Chroma collections.

    Important: this class does NOT try to implement transactions; callers should
    treat vector writes as best-effort unless they use an outbox pattern.
    """

    def __init__(
        self,
        *,
        node_index_collection: Any,
        node_collection: Any,
        edge_collection: Any,
        edge_endpoints_collection: Any,
        document_collection: Any,
        domain_collection: Any,
        node_docs_collection: Any,
        node_refs_collection: Any,
        edge_refs_collection: Any,
        persist_directory: str | None = None,
    ):
        self._collections: Dict[str, Any] = {
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

    def _c(self, key: str) -> Any:
        try:
            return self._collections[key]
        except KeyError as e:
            raise KeyError(f"Unknown collection_key={key!r}") from e

    def call(self, collection_key: str, method: str, **kwargs) -> Any:
        coll = self._c(collection_key)
        fn = getattr(coll, method)
        return _awaitable_result(run_awaitable_blocking(fn(**_chroma_safe_kwargs(kwargs))))

    # --- node_index ---
    def node_index_get(self, **kwargs) -> Any:
        return self.call("node_index", "get", **kwargs)

    def node_index_query(self, **kwargs) -> Any:
        return self.call("node_index", "query", **kwargs)

    def node_index_add(self, **kwargs) -> Any:
        return self.call("node_index", "add", **kwargs)

    def node_index_upsert(self, **kwargs) -> Any:
        return self.call("node_index", "upsert", **kwargs)

    def node_index_update(self, **kwargs) -> Any:
        return self.call("node_index", "update", **kwargs)

    def node_index_delete(self, **kwargs) -> Any:
        return self.call("node_index", "delete", **kwargs)

    # --- nodes ---
    def node_get(self, **kwargs) -> Any:
        return self.call("node", "get", **kwargs)

    def node_query(self, **kwargs) -> Any:
        return self.call("node", "query", **kwargs)

    def node_add(self, **kwargs) -> Any:
        return self.call("node", "add", **kwargs)

    def node_upsert(self, **kwargs) -> Any:
        return self.call("node", "upsert", **kwargs)

    def node_update(self, **kwargs) -> Any:
        return self.call("node", "update", **kwargs)

    def node_delete(self, **kwargs) -> Any:
        return self.call("node", "delete", **kwargs)

    # --- edges ---
    def edge_get(self, **kwargs) -> Any:
        return self.call("edge", "get", **kwargs)

    def edge_query(self, **kwargs) -> Any:
        return self.call("edge", "query", **kwargs)

    def edge_add(self, **kwargs) -> Any:
        return self.call("edge", "add", **kwargs)

    def edge_upsert(self, **kwargs) -> Any:
        return self.call("edge", "upsert", **kwargs)

    def edge_update(self, **kwargs) -> Any:
        return self.call("edge", "update", **kwargs)

    def edge_delete(self, **kwargs) -> Any:
        return self.call("edge", "delete", **kwargs)

    # --- edge_endpoints ---
    def edge_endpoints_get(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "get", **kwargs)

    def edge_endpoints_query(self, **kwargs) -> Any:
        raise ValueError(
            "edge_endpoints is structural; semantic query is unsupported; "
            "use edge_endpoints_get with metadata filters"
        )

    def edge_endpoints_add(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "add", **kwargs)

    def edge_endpoints_upsert(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "upsert", **kwargs)

    def edge_endpoints_update(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "update", **kwargs)

    def edge_endpoints_delete(self, **kwargs) -> Any:
        return self.call("edge_endpoints", "delete", **kwargs)

    # --- documents ---
    def document_get(self, **kwargs) -> Any:
        return self.call("document", "get", **kwargs)

    def document_query(self, **kwargs) -> Any:
        return self.call("document", "query", **kwargs)

    def document_add(self, **kwargs) -> Any:
        return self.call("document", "add", **kwargs)

    def document_upsert(self, **kwargs) -> Any:
        return self.call("document", "upsert", **kwargs)

    def document_update(self, **kwargs) -> Any:
        return self.call("document", "update", **kwargs)

    def document_delete(self, **kwargs) -> Any:
        return self.call("document", "delete", **kwargs)

    # --- domains ---
    def domain_get(self, **kwargs) -> Any:
        return self.call("domain", "get", **kwargs)

    def domain_query(self, **kwargs) -> Any:
        return self.call("domain", "query", **kwargs)

    def domain_add(self, **kwargs) -> Any:
        return self.call("domain", "add", **kwargs)

    def domain_upsert(self, **kwargs) -> Any:
        return self.call("domain", "upsert", **kwargs)

    def domain_update(self, **kwargs) -> Any:
        return self.call("domain", "update", **kwargs)

    def domain_delete(self, **kwargs) -> Any:
        return self.call("domain", "delete", **kwargs)

    # --- node_docs ---
    def node_docs_get(self, **kwargs) -> Any:
        return self.call("node_docs", "get", **kwargs)

    def node_docs_query(self, **kwargs) -> Any:
        return self.call("node_docs", "query", **kwargs)

    def node_docs_add(self, **kwargs) -> Any:
        return self.call("node_docs", "add", **kwargs)

    def node_docs_upsert(self, **kwargs) -> Any:
        return self.call("node_docs", "upsert", **kwargs)

    def node_docs_update(self, **kwargs) -> Any:
        return self.call("node_docs", "update", **kwargs)

    def node_docs_delete(self, **kwargs) -> Any:
        return self.call("node_docs", "delete", **kwargs)

    # --- node_refs ---
    def node_refs_get(self, **kwargs) -> Any:
        return self.call("node_refs", "get", **kwargs)

    def node_refs_query(self, **kwargs) -> Any:
        return self.call("node_refs", "query", **kwargs)

    def node_refs_add(self, **kwargs) -> Any:
        return self.call("node_refs", "add", **kwargs)

    def node_refs_upsert(self, **kwargs) -> Any:
        return self.call("node_refs", "upsert", **kwargs)

    def node_refs_update(self, **kwargs) -> Any:
        return self.call("node_refs", "update", **kwargs)

    def node_refs_delete(self, **kwargs) -> Any:
        return self.call("node_refs", "delete", **kwargs)

    # --- edge_refs ---
    def edge_refs_get(self, **kwargs) -> Any:
        return self.call("edge_refs", "get", **kwargs)

    def edge_refs_query(self, **kwargs) -> Any:
        return self.call("edge_refs", "query", **kwargs)

    def edge_refs_add(self, **kwargs) -> Any:
        return self.call("edge_refs", "add", **kwargs)

    def edge_refs_upsert(self, **kwargs) -> Any:
        return self.call("edge_refs", "upsert", **kwargs)

    def edge_refs_update(self, **kwargs) -> Any:
        return self.call("edge_refs", "update", **kwargs)

    def edge_refs_delete(self, **kwargs) -> Any:
        return self.call("edge_refs", "delete", **kwargs)


class AsyncChromaBackend(ChromaBackend):
    """Chroma facade retaining sync API while exposing non-blocking verbs."""

    is_async_backend = True

    async def async_call(self, collection_key: str, method: str, **kwargs) -> Any:
        coll = self._c(collection_key)
        result = getattr(coll, method)(**_chroma_safe_kwargs(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def async_node_get(self, **kwargs) -> Any:
        return await self.async_call("node", "get", **kwargs)

    async def async_node_upsert(self, **kwargs) -> Any:
        return await self.async_call("node", "upsert", **kwargs)

    async def async_node_delete(self, **kwargs) -> Any:
        return await self.async_call("node", "delete", **kwargs)

    async def async_edge_get(self, **kwargs) -> Any:
        return await self.async_call("edge", "get", **kwargs)

    async def async_edge_upsert(self, **kwargs) -> Any:
        return await self.async_call("edge", "upsert", **kwargs)

    async def async_edge_delete(self, **kwargs) -> Any:
        return await self.async_call("edge", "delete", **kwargs)
