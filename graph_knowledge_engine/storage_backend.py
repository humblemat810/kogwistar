from __future__ import annotations

"""Backend abstraction for persistence + vector index.

This module introduces a small interface layer so `engine.py` can be refactored
incrementally:

Phase 1 goals (what this file provides now)
-------------------------------------------
* A `StorageBackend` protocol capturing the Chroma operations currently used
  directly inside `GraphKnowledgeEngine`.
* A `ChromaBackend` implementation that simply forwards to existing Chroma
  collections (no behavior change).
* A `UnitOfWork` context manager surface so callers can write
  `with engine.uow(): ...` even when the backend is non-transactional.

Phase 2 goals (future)
----------------------
* Add `PostgresBackend` implementing the same protocol using SQLAlchemy
  + pgvector + JSONB.
* In Postgres mode, `UnitOfWork` will open a real SQL transaction.
* In Chroma mode, UoW remains a no-op for vector operations, but can still wrap
  the engine's SQLite meta store (outbox/checkpoints/etc.).
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Protocol, Sequence


JSONDict = Dict[str, Any]


class UnitOfWork(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        ...


class StorageBackend(Protocol):
    """Minimal set of operations the engine needs.

    This is intentionally not a full Chroma API clone; it's the subset used by
    `engine.py` today. We can grow this interface as we refactor.
    """

    # ---- nodes ----
    def node_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def node_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        ...

    def node_query(
        self,
        *,
        query_texts: Optional[Sequence[str]] = None,
        query_embeddings: Optional[Any] = None,
        n_results: int = 10,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
    ) -> JSONDict:
        ...

    def node_delete(self, *, ids: Sequence[str]) -> None:
        ...

    # ---- edges ----
    def edge_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def edge_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        ...

    def edge_query(
        self,
        *,
        query_texts: Optional[Sequence[str]] = None,
        query_embeddings: Optional[Any] = None,
        n_results: int = 10,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
    ) -> JSONDict:
        ...

    def edge_delete(self, *, ids: Sequence[str]) -> None:
        ...

    # ---- node docs / refs / documents / domains ----
    def node_docs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def node_docs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        ...

    def node_docs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        ...

    def node_refs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def node_refs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        ...

    def node_refs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        ...

    def edge_refs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def edge_refs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        ...

    def edge_refs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        ...

    def document_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def document_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        ...

    def document_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        ...

    def domain_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def domain_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        ...

    def domain_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        ...

    # ---- edge endpoints (hypergraph incidence materialization) ----
    def edge_endpoints_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        ...

    def edge_endpoints_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        ...

    def edge_endpoints_delete(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
    ) -> None:
        ...


@dataclass
class NoopUnitOfWork(UnitOfWork):
    """UoW that does nothing (for Chroma-only paths)."""

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


@dataclass
class ChromaBackend(StorageBackend):
    """Thin wrapper around existing Chroma collections.

    NOTE: we deliberately keep method names distinct from Chroma's to avoid the
    temptation to grow a full Chroma compatibility layer.
    """

    node_collection: Any
    edge_collection: Any
    edge_endpoints_collection: Any
    document_collection: Optional[Any] = None
    domain_collection: Optional[Any] = None
    node_docs_collection: Optional[Any] = None
    node_refs_collection: Optional[Any] = None
    edge_refs_collection: Optional[Any] = None

    def node_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        kwargs: JSONDict = {
            "ids": list(ids),
            "documents": list(documents),
            "metadatas": list(metadatas),
        }
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        self.node_collection.add(**kwargs)

    def node_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.node_collection.get(**kwargs)

    def node_query(
        self,
        *,
        query_texts: Optional[Sequence[str]] = None,
        query_embeddings: Optional[Any] = None,
        n_results: int = 10,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
    ) -> JSONDict:
        kwargs: JSONDict = {"n_results": int(n_results)}
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        elif query_texts is not None:
            kwargs["query_texts"] = list(query_texts)
        else:
            raise ValueError("node_query requires query_texts or query_embeddings")
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.node_collection.query(**kwargs)

    def node_delete(self, *, ids: Sequence[str]) -> None:
        self.node_collection.delete(ids=list(ids))

    def edge_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        kwargs: JSONDict = {
            "ids": list(ids),
            "documents": list(documents),
            "metadatas": list(metadatas),
        }
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        self.edge_collection.add(**kwargs)

    def edge_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.edge_collection.get(**kwargs)

    def edge_query(
        self,
        *,
        query_texts: Optional[Sequence[str]] = None,
        query_embeddings: Optional[Any] = None,
        n_results: int = 10,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
    ) -> JSONDict:
        kwargs: JSONDict = {"n_results": int(n_results)}
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        elif query_texts is not None:
            kwargs["query_texts"] = list(query_texts)
        else:
            raise ValueError("edge_query requires query_texts or query_embeddings")
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.edge_collection.query(**kwargs)

    def edge_delete(self, *, ids: Sequence[str]) -> None:
        self.edge_collection.delete(ids=list(ids))

    def edge_endpoints_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[JSONDict],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        kwargs: JSONDict = {
            "ids": list(ids),
            "documents": list(documents),
            "metadatas": list(metadatas),
        }
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        self.edge_endpoints_collection.add(**kwargs)

    def edge_endpoints_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
        include: Optional[list[str]] = None,
        limit: int = 200,
    ) -> JSONDict:
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return self.edge_endpoints_collection.get(**kwargs)

    def edge_endpoints_delete(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[JSONDict] = None,
    ) -> None:
        kwargs: JSONDict = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        self.edge_endpoints_collection.delete(**kwargs)

    # ---- node docs / refs / documents / domains ----
    def _req(self, col: Optional[Any], name: str) -> Any:
        if col is None:
            raise RuntimeError(f"ChromaBackend missing collection: {name}")
        return col

    def node_docs_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[JSONDict], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        col = self._req(self.node_docs_collection, "node_docs_collection")
        kwargs: JSONDict = {"ids": list(ids), "documents": list(documents), "metadatas": list(metadatas)}
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        col.add(**kwargs)

    def node_docs_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        col = self._req(self.node_docs_collection, "node_docs_collection")
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return col.get(**kwargs)

    def node_docs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        col = self._req(self.node_docs_collection, "node_docs_collection")
        kwargs: JSONDict = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        col.delete(**kwargs)

    def node_refs_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[JSONDict], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        col = self._req(self.node_refs_collection, "node_refs_collection")
        kwargs: JSONDict = {"ids": list(ids), "documents": list(documents), "metadatas": list(metadatas)}
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        col.add(**kwargs)

    def node_refs_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        col = self._req(self.node_refs_collection, "node_refs_collection")
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return col.get(**kwargs)

    def node_refs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        col = self._req(self.node_refs_collection, "node_refs_collection")
        kwargs: JSONDict = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        col.delete(**kwargs)

    def edge_refs_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[JSONDict], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        col = self._req(self.edge_refs_collection, "edge_refs_collection")
        kwargs: JSONDict = {"ids": list(ids), "documents": list(documents), "metadatas": list(metadatas)}
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        col.add(**kwargs)

    def edge_refs_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        col = self._req(self.edge_refs_collection, "edge_refs_collection")
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return col.get(**kwargs)

    def edge_refs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        col = self._req(self.edge_refs_collection, "edge_refs_collection")
        kwargs: JSONDict = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        col.delete(**kwargs)

    def document_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[JSONDict], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        col = self._req(self.document_collection, "document_collection")
        kwargs: JSONDict = {"ids": list(ids), "documents": list(documents), "metadatas": list(metadatas)}
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        col.add(**kwargs)

    def document_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        col = self._req(self.document_collection, "document_collection")
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return col.get(**kwargs)

    def document_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        col = self._req(self.document_collection, "document_collection")
        kwargs: JSONDict = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        col.delete(**kwargs)

    def domain_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[JSONDict], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        col = self._req(self.domain_collection, "domain_collection")
        kwargs: JSONDict = {"ids": list(ids), "documents": list(documents), "metadatas": list(metadatas)}
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        col.add(**kwargs)

    def domain_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None, include: Optional[list[str]] = None, limit: int = 200) -> JSONDict:
        col = self._req(self.domain_collection, "domain_collection")
        kwargs: JSONDict = {"limit": int(limit)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        if include is not None:
            kwargs["include"] = list(include)
        return col.get(**kwargs)

    def domain_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[JSONDict] = None) -> None:
        col = self._req(self.domain_collection, "domain_collection")
        kwargs: JSONDict = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        col.delete(**kwargs)
