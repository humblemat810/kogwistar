from typing import Any, Dict

from .async_compat import run_awaitable_blocking


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

    def _c(self, key: str) -> Any:
        try:
            return self._collections[key]
        except KeyError as e:
            raise KeyError(f"Unknown collection_key={key!r}") from e

    def call(self, collection_key: str, method: str, **kwargs) -> Any:
        coll = self._c(collection_key)
        fn = getattr(coll, method)
        return _awaitable_result(run_awaitable_blocking(fn(**kwargs)))

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
        return self.call("edge_endpoints", "query", **kwargs)

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
    pass
