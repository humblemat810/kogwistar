from typing import Any, Dict

from .async_compat import run_awaitable_blocking


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
        return run_awaitable_blocking(fn(**kwargs))

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
    async def call(self, collection_key: str, method: str, **kwargs) -> Any:
        coll = self._c(collection_key)
        fn = getattr(coll, method)
        return await fn(**kwargs)

    # --- node_index ---
    async def node_index_get(self, **kwargs) -> Any:
        return await self.call("node_index", "get", **kwargs)

    async def node_index_query(self, **kwargs) -> Any:
        return await self.call("node_index", "query", **kwargs)

    async def node_index_add(self, **kwargs) -> Any:
        return await self.call("node_index", "add", **kwargs)

    async def node_index_upsert(self, **kwargs) -> Any:
        return await self.call("node_index", "upsert", **kwargs)

    async def node_index_update(self, **kwargs) -> Any:
        return await self.call("node_index", "update", **kwargs)

    async def node_index_delete(self, **kwargs) -> Any:
        return await self.call("node_index", "delete", **kwargs)

    # --- nodes ---
    async def node_get(self, **kwargs) -> Any:
        return await self.call("node", "get", **kwargs)

    async def node_query(self, **kwargs) -> Any:
        return await self.call("node", "query", **kwargs)

    async def node_add(self, **kwargs) -> Any:
        return await self.call("node", "add", **kwargs)

    async def node_upsert(self, **kwargs) -> Any:
        return await self.call("node", "upsert", **kwargs)

    async def node_update(self, **kwargs) -> Any:
        return await self.call("node", "update", **kwargs)

    async def node_delete(self, **kwargs) -> Any:
        return await self.call("node", "delete", **kwargs)

    # --- edges ---
    async def edge_get(self, **kwargs) -> Any:
        return await self.call("edge", "get", **kwargs)

    async def edge_query(self, **kwargs) -> Any:
        return await self.call("edge", "query", **kwargs)

    async def edge_add(self, **kwargs) -> Any:
        return await self.call("edge", "add", **kwargs)

    async def edge_upsert(self, **kwargs) -> Any:
        return await self.call("edge", "upsert", **kwargs)

    async def edge_update(self, **kwargs) -> Any:
        return await self.call("edge", "update", **kwargs)

    async def edge_delete(self, **kwargs) -> Any:
        return await self.call("edge", "delete", **kwargs)

    # --- edge_endpoints ---
    async def edge_endpoints_get(self, **kwargs) -> Any:
        return await self.call("edge_endpoints", "get", **kwargs)

    async def edge_endpoints_query(self, **kwargs) -> Any:
        return await self.call("edge_endpoints", "query", **kwargs)

    async def edge_endpoints_add(self, **kwargs) -> Any:
        return await self.call("edge_endpoints", "add", **kwargs)

    async def edge_endpoints_upsert(self, **kwargs) -> Any:
        return await self.call("edge_endpoints", "upsert", **kwargs)

    async def edge_endpoints_update(self, **kwargs) -> Any:
        return await self.call("edge_endpoints", "update", **kwargs)

    async def edge_endpoints_delete(self, **kwargs) -> Any:
        return await self.call("edge_endpoints", "delete", **kwargs)

    # --- documents ---
    async def document_get(self, **kwargs) -> Any:
        return await self.call("document", "get", **kwargs)

    async def document_query(self, **kwargs) -> Any:
        return await self.call("document", "query", **kwargs)

    async def document_add(self, **kwargs) -> Any:
        return await self.call("document", "add", **kwargs)

    async def document_upsert(self, **kwargs) -> Any:
        return await self.call("document", "upsert", **kwargs)

    async def document_update(self, **kwargs) -> Any:
        return await self.call("document", "update", **kwargs)

    async def document_delete(self, **kwargs) -> Any:
        return await self.call("document", "delete", **kwargs)

    # --- domains ---
    async def domain_get(self, **kwargs) -> Any:
        return await self.call("domain", "get", **kwargs)

    async def domain_query(self, **kwargs) -> Any:
        return await self.call("domain", "query", **kwargs)

    async def domain_add(self, **kwargs) -> Any:
        return await self.call("domain", "add", **kwargs)

    async def domain_upsert(self, **kwargs) -> Any:
        return await self.call("domain", "upsert", **kwargs)

    async def domain_update(self, **kwargs) -> Any:
        return await self.call("domain", "update", **kwargs)

    async def domain_delete(self, **kwargs) -> Any:
        return await self.call("domain", "delete", **kwargs)

    # --- node_docs ---
    async def node_docs_get(self, **kwargs) -> Any:
        return await self.call("node_docs", "get", **kwargs)

    async def node_docs_query(self, **kwargs) -> Any:
        return await self.call("node_docs", "query", **kwargs)

    async def node_docs_add(self, **kwargs) -> Any:
        return await self.call("node_docs", "add", **kwargs)

    async def node_docs_upsert(self, **kwargs) -> Any:
        return await self.call("node_docs", "upsert", **kwargs)

    async def node_docs_update(self, **kwargs) -> Any:
        return await self.call("node_docs", "update", **kwargs)

    async def node_docs_delete(self, **kwargs) -> Any:
        return await self.call("node_docs", "delete", **kwargs)

    # --- node_refs ---
    async def node_refs_get(self, **kwargs) -> Any:
        return await self.call("node_refs", "get", **kwargs)

    async def node_refs_query(self, **kwargs) -> Any:
        return await self.call("node_refs", "query", **kwargs)

    async def node_refs_add(self, **kwargs) -> Any:
        return await self.call("node_refs", "add", **kwargs)

    async def node_refs_upsert(self, **kwargs) -> Any:
        return await self.call("node_refs", "upsert", **kwargs)

    async def node_refs_update(self, **kwargs) -> Any:
        return await self.call("node_refs", "update", **kwargs)

    async def node_refs_delete(self, **kwargs) -> Any:
        return await self.call("node_refs", "delete", **kwargs)

    # --- edge_refs ---
    async def edge_refs_get(self, **kwargs) -> Any:
        return await self.call("edge_refs", "get", **kwargs)

    async def edge_refs_query(self, **kwargs) -> Any:
        return await self.call("edge_refs", "query", **kwargs)

    async def edge_refs_add(self, **kwargs) -> Any:
        return await self.call("edge_refs", "add", **kwargs)

    async def edge_refs_upsert(self, **kwargs) -> Any:
        return await self.call("edge_refs", "upsert", **kwargs)

    async def edge_refs_update(self, **kwargs) -> Any:
        return await self.call("edge_refs", "update", **kwargs)

    async def edge_refs_delete(self, **kwargs) -> Any:
        return await self.call("edge_refs", "delete", **kwargs)
