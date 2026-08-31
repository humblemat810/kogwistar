from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from ...utils.embedding_vectors import (
    normalize_embedding_rows,
    normalize_embedding_vector,
)

from ...entity_registry import (
    default_edge_type_for_graph_kind,
    default_node_type_for_graph_kind,
    pick_edge_type,
    pick_node_type,
)
from ..async_compat import run_awaitable_blocking
from ..models import Document, Edge, Node
from ..utils.refs import ref_doc_id
from .base import NamespaceProxy
from ...typing_interfaces import ReadLike

TNode = TypeVar("TNode", bound=Node)


class ReadSubsystem(NamespaceProxy, ReadLike):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    @staticmethod
    def _native_equality_filter(where: Any) -> dict[str, Any] | None:
        if where is None:
            return {}
        if not isinstance(where, dict):
            return None
        if any(
            not isinstance(key, str)
            or key.startswith("$")
            or isinstance(value, dict)
            for key, value in where.items()
        ):
            return None
        return dict(where)

    def _rust_postgres_projection_get(
        self,
        *,
        entity_kind: str,
        ids: Sequence[str] | None,
        where: Any,
        limit: int | None,
        include: list[str],
    ) -> dict[str, Any] | None:
        from ..rust_postgres_session import RustEnginePostgresMetaStore

        meta = getattr(self._e, "meta_sqlite", None)
        metadata = self._native_equality_filter(where)
        if (
            not isinstance(meta, RustEnginePostgresMetaStore)
            or metadata is None
            or limit is None
            # Native projection reads are id-sorted while the legacy backend
            # leaves multi-row get order unspecified. Cut over only the exact
            # singleton case until a public ordering contract is approved.
            or ids is None
            or len(ids) != 1
        ):
            return None
        table = (
            self._e.backend.nodes.name
            if entity_kind == "node"
            else self._e.backend.edges.name
            if entity_kind == "edge"
            else self._e.backend.documents.name
        )
        effective_include = include or ["documents", "metadatas"]
        records = meta.graph_projection_records(
            namespace=getattr(self._e, "namespace", "default"),
            workspace_id=None,
            graph_space=None,
            table=table,
            ids=None if ids is None else list(ids),
            metadata=metadata,
            limit=int(limit),
        )
        result: dict[str, Any] = {
            "ids": [str(record.get("id") or "") for record in records]
        }
        if "documents" in effective_include:
            result["documents"] = [record.get("document") for record in records]
        if "metadatas" in effective_include:
            result["metadatas"] = [
                dict(record.get("metadata") or {}) for record in records
            ]
        if "embeddings" in effective_include:
            result["embeddings"] = [
                normalize_embedding_vector(record.get("embedding"))
                for record in records
            ]
        return result

    def _rust_postgres_projection_query(
        self,
        *,
        entity_kind: str,
        query_embeddings: Sequence[Sequence[float]],
        where: Any,
        n_results: int,
        include: list[str],
    ) -> dict[str, Any] | None:
        from ..rust_postgres_session import RustEnginePostgresMetaStore

        meta = getattr(self._e, "meta_sqlite", None)
        metadata = self._native_equality_filter(where)
        if (
            not isinstance(meta, RustEnginePostgresMetaStore)
            or metadata is None
            or len(query_embeddings) != 1
        ):
            return None
        table = (
            self._e.backend.nodes.name
            if entity_kind == "node"
            else self._e.backend.edges.name
        )
        matches = meta.graph_projection_vector_query(
            namespace=getattr(self._e, "namespace", "default"),
            workspace_id=None,
            graph_space=None,
            table=table,
            embedding=list(query_embeddings[0]),
            embedding_dim=int(self._e.backend.embedding_dim),
            metadata=metadata,
            metric=str(self._e.backend.distance),
            limit=int(n_results),
        )
        records = [dict(match.get("record") or {}) for match in matches]
        effective_include = include or ["documents", "metadatas", "distances"]
        result: dict[str, Any] = {
            "ids": [[str(record.get("id") or "") for record in records]]
        }
        if "documents" in effective_include:
            result["documents"] = [[record.get("document") for record in records]]
        if "metadatas" in effective_include:
            result["metadatas"] = [
                [dict(record.get("metadata") or {}) for record in records]
            ]
        if "embeddings" in effective_include:
            result["embeddings"] = [
                [
                    normalize_embedding_vector(record.get("embedding"))
                    for record in records
                ]
            ]
        if "distances" in effective_include:
            result["distances"] = [
                [float(match.get("distance", 0.0)) for match in matches]
            ]
        return result

    def _stage1_fallback_get(
        self,
        *,
        entity_kind: str,
        ids: Sequence[str] | None,
        where: Any,
        limit: int | None,
        include: list[str],
    ) -> dict[str, Any] | None:
        """Read pending Chroma entities from the transient SQLite projection."""
        if getattr(self._e, "persistence_mode", "single_stage") != "two_stage":
            return None
        adapter = getattr(self._e, "two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if not callable(query):
            return None
        if where is not None and self._native_equality_filter(where) is None:
            return None
        rows = query(
            ids=list(ids) if ids is not None else None,
            entity_kind=entity_kind,
            metadata=dict(where or {}),
            limit=limit,
        )
        if not rows:
            return None
        payloads = [dict(row.get("payload") or {}) for row in rows]
        result: dict[str, Any] = {
            "ids": [str(payload.get("id") or row.get("key") or "") for payload, row in zip(payloads, rows)]
        }
        if "documents" in include:
            result["documents"] = [payload.get("document") for payload in payloads]
        if "metadatas" in include:
            result["metadatas"] = [dict(payload.get("metadata") or {}) for payload in payloads]
        if "embeddings" in include:
            result["embeddings"] = [None for _ in payloads]
        return result

    def _node_get_raw(
        self,
        *,
        ids: Sequence[str] | None = None,
        where=None,
        limit: int | None = 200,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        native = self._rust_postgres_projection_get(
            entity_kind="node",
            ids=ids,
            where=where,
            limit=limit,
            include=include,
        )
        if native is not None:
            return native
        result = run_awaitable_blocking(self._e.backend.node_get(
            ids=ids,
            include=include,
            where=where,
            limit=limit,
        ))
        if not result.get("ids"):
            return self._stage1_fallback_get(
                entity_kind="node", ids=ids, where=where, limit=limit, include=include
            ) or result
        return result

    def _edge_get_raw(
        self,
        *,
        ids: Sequence[str] | None = None,
        where=None,
        limit: int | None = 400,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        native = self._rust_postgres_projection_get(
            entity_kind="edge",
            ids=ids,
            where=where,
            limit=limit,
            include=include,
        )
        if native is not None:
            return native
        result = run_awaitable_blocking(self._e.backend.edge_get(
            ids=ids,
            include=include,
            where=where,
            limit=limit,
        ))
        if not result.get("ids"):
            return self._stage1_fallback_get(
                entity_kind="edge", ids=ids, where=where, limit=limit, include=include
            ) or result
        return result

    def node_exists(
        self,
        ids: Sequence[str] | None = None,
        where=None,
    ) -> bool:
        limit = 1 if ids is None else max(1, len(ids))
        got = self._node_get_raw(
            ids=ids,
            where=where,
            limit=limit,
            include=["metadatas"],
        )
        return bool(got.get("ids"))

    def edge_exists(
        self,
        ids: Sequence[str] | None = None,
        where=None,
    ) -> bool:
        limit = 1 if ids is None else max(1, len(ids))
        got = self._edge_get_raw(
            ids=ids,
            where=where,
            limit=limit,
            include=["metadatas"],
        )
        return bool(got.get("ids"))

    def get_node_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where=None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        got = self._node_get_raw(
            ids=ids,
            where=where,
            limit=limit,
            include=["metadatas"],
        )
        metadatas = got.get("metadatas") or []
        return [dict(meta) for meta in metadatas if isinstance(meta, dict)]

    def get_edge_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where=None,
        limit: int | None = 400,
    ) -> list[dict[str, Any]]:
        got = self._edge_get_raw(
            ids=ids,
            where=where,
            limit=limit,
            include=["metadatas"],
        )
        metadatas = got.get("metadatas") or []
        return [dict(meta) for meta in metadatas if isinstance(meta, dict)]

    # Canonical read API
    def get_nodes(
        self,
        ids: Sequence[str] | None = None,
        node_type: Type[Node] | None = None,
        include: list[str] | None = None,
        where=None,
        limit: int | None = 200,
        resolve_mode: Literal[
            "active_only", "redirect", "include_tombstones"
        ] = "active_only",
    ) -> list[Node]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        if not node_type:
            node_type = default_node_type_for_graph_kind(self._e.kg_graph_type)

        try:
            got = self._node_get_raw(
                ids=ids,
                include=include,
                where=where,
                limit=limit,
            )
        except Exception:
            if "embeddings" not in include:
                raise
            fallback_include = [item for item in include if item != "embeddings"]
            got = self._node_get_raw(
                ids=ids,
                include=fallback_include,
                where=where,
                limit=limit,
            )
        nodes = self.nodes_from_single_or_id_query_result(got, node_type=node_type)
        nodes = self._e._resolve_redirect_chain(
            initial_items=nodes,
            resolve_mode=resolve_mode,
            fetch_by_ids=lambda redirect_ids: self.get_nodes(
                redirect_ids,
                node_type=node_type,
                resolve_mode=resolve_mode,
            ),
        )
        return self._e._filter_items_by_resolve_mode(nodes, resolve_mode)

    def get_edges(
        self,
        ids: Sequence[str] | None = None,
        edge_type: Type[Edge] | None = None,
        where=None,
        limit: int | None = 400,
        include: list[str] | None = None,
        resolve_mode: Literal[
            "active_only", "redirect", "include_tombstones"
        ] = "active_only",
    ) -> list[Edge]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        if not edge_type:
            edge_type = default_edge_type_for_graph_kind(self._e.kg_graph_type)

        try:
            got = self._edge_get_raw(
                ids=ids,
                include=include,
                where=where,
                limit=limit,
            )
        except Exception:
            if "embeddings" not in include:
                raise
            fallback_include = [item for item in include if item != "embeddings"]
            got = self._edge_get_raw(
                ids=ids,
                include=fallback_include,
                where=where,
                limit=limit,
            )
        edges = self.edges_from_single_or_id_query_result(
            got, edge_type=edge_type, include=include
        )
        edges = self._e._resolve_redirect_chain(
            initial_items=edges,
            resolve_mode=resolve_mode,
            fetch_by_ids=lambda redirect_ids: self.get_edges(
                redirect_ids,
                edge_type=edge_type,
                resolve_mode=resolve_mode,
            ),
        )
        return self._e._filter_items_by_resolve_mode(edges, resolve_mode)

    def get_document(self, doc_id: str) -> Document:
        doc_get_result = self._rust_postgres_projection_get(
            entity_kind="document",
            ids=[doc_id],
            where=None,
            limit=1,
            include=["documents", "metadatas"],
        )
        if doc_get_result is None:
            doc_get_result = run_awaitable_blocking(
                self._e.backend.document_get(ids=[doc_id])
            )
        if len(doc_get_result["ids"]) == 0:
            raise ValueError(f"no document found for doc id = {doc_id}")
        metadatas = doc_get_result["metadatas"]
        docs = doc_get_result["documents"]

        if docs is None or metadatas is None:
            raise ValueError("Invalid documnet metadata")
        metadata: dict = cast(dict, metadatas[0])

        doc = Document(
            id=doc_get_result["ids"][0],
            content=docs[0],
            metadata=metadata,
            domain_id=(
                None
                if metadata.get("domain_id") is None
                else str(metadata.get("domain_id"))
            ),
            type=metadata["type"],
            processed=metadata["processed"],
            embeddings=None,
            source_map=None,
        )
        return doc

    def query_nodes(
        self,
        *args,
        query=None,
        query_embeddings=None,
        include=["documents", "embeddings", "metadatas"],
        node_type: Type[Node] = Node,
        **kwargs,
    ):
        if query_embeddings is not None:
            if query is not None:
                raise Exception(
                    "either query or query embedding but not both specified."
                )
        else:
            if query is not None:
                query_embeddings = self._e._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")
        query_embeddings = cast(
            list[list[float]],
            normalize_embedding_rows(query_embeddings, allow_empty=False),
        )

        native = (
            None
            if args or set(kwargs) - {"where", "n_results"}
            else self._rust_postgres_projection_query(
            entity_kind="node",
            query_embeddings=query_embeddings,
            where=kwargs.get("where"),
            n_results=int(kwargs.get("n_results", 10)),
            include=include,
            )
        )
        got = native or run_awaitable_blocking(self._e.backend.node_query(
            query_embeddings=query_embeddings,
            *args,
            include=include,
            **kwargs,
        ))
        return self.nodes_from_query_result(got, node_type=node_type)

    def _coerce_ts_utc(self, raw: Any) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            dt = raw
        elif isinstance(raw, (int, float)):
            dt = datetime.fromtimestamp(float(raw), tz=timezone.utc)
        elif isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                return None
        else:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _is_node_visible_as_of(self, node: Node, as_of: datetime) -> bool:
        meta = getattr(node, "metadata", {}) or {}
        effective_from = self._coerce_ts_utc(meta.get("effective_from"))
        if effective_from is not None and effective_from > as_of:
            return False

        status = str(meta.get("lifecycle_status") or "active")
        if status != "tombstoned":
            return True

        deleted_at = self._coerce_ts_utc(meta.get("deleted_at"))
        if deleted_at is None:
            return False
        return deleted_at > as_of

    def _redirect_applies_as_of(self, node: Node, as_of: datetime) -> bool:
        meta = getattr(node, "metadata", {}) or {}
        redirect_to_id = meta.get("redirect_to_id")
        if not redirect_to_id:
            return False
        deleted_at = self._coerce_ts_utc(meta.get("deleted_at"))
        if deleted_at is None:
            return False
        return deleted_at <= as_of

    def _resolve_node_as_of(
        self,
        node: Node,
        *,
        as_of: datetime,
        node_type: Type[Node],
        cache: dict[str, Node],
        follow_redirects: bool,
        max_redirect_hops: int,
    ) -> Node | None:
        visited: set[str] = set()
        current = node
        hops = 0

        while True:
            current_id = str(current.safe_get_id())
            if current_id in visited:
                return None
            visited.add(current_id)

            if self._is_node_visible_as_of(current, as_of):
                return current

            if (not follow_redirects) or (
                not self._redirect_applies_as_of(current, as_of)
            ):
                return None

            next_id = str(
                ((getattr(current, "metadata", {}) or {}).get("redirect_to_id") or "")
            ).strip()
            if not next_id:
                return None

            hops += 1
            if hops > max_redirect_hops:
                return None

            nxt = cache.get(next_id)
            if nxt is None:
                fetched = self.get_nodes(
                    ids=[next_id],
                    node_type=node_type,
                    include=["documents", "embeddings", "metadatas"],
                    resolve_mode="include_tombstones",
                )
                if not fetched:
                    return None
                nxt = fetched[0]
                cache[next_id] = nxt
            current = nxt

    def search_nodes_as_of(
        self,
        *,
        query: str | None = None,
        query_embeddings: list[list[float]] | None = None,
        as_of_ts: datetime | str,
        where: dict[str, Any] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: Type[Node] = Node,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        **kwargs,
    ) -> list[Node]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        if query is not None and query_embeddings is not None:
            raise ValueError("Specify only one of query or query_embeddings.")
        if query_embeddings is None:
            if query is None:
                raise ValueError("Either query or query_embeddings must be provided.")
            query_embeddings = self._e._iterative_defensive_emb(query)
        query_embeddings = cast(
            list[list[float]],
            normalize_embedding_rows(query_embeddings, allow_empty=False),
        )
        if not query_embeddings:
            raise ValueError("query_embeddings resolved to an empty list.")

        as_of = self._coerce_ts_utc(as_of_ts)
        if as_of is None:
            raise ValueError(f"Invalid as_of_ts: {as_of_ts!r}")

        query_kwargs = {
            "query_embeddings": query_embeddings,
            "include": include,
            "n_results": n_results,
            "where": where,
            **kwargs,
        }
        # Historical search needs tombstone candidates for redirect/time
        # resolution; ordinary semantic search must continue excluding them.
        if getattr(self._e.backend, "supports_historical_tombstone_query", False):
            query_kwargs["include_tombstoned"] = True
        got = run_awaitable_blocking(self._e.backend.node_query(**query_kwargs))
        batches = self.nodes_from_query_result(got, node_type=node_type)
        candidates = [node for batch in batches for node in batch] if batches else []
        cache = {str(node.safe_get_id()): node for node in candidates}

        out: list[Node] = []
        seen: set[str] = set()
        for node in candidates:
            resolved = self._resolve_node_as_of(
                node,
                as_of=as_of,
                node_type=node_type,
                cache=cache,
                follow_redirects=follow_redirects,
                max_redirect_hops=max_redirect_hops,
            )
            if resolved is None:
                continue
            node_id = str(resolved.safe_get_id())
            if node_id in seen:
                continue
            seen.add(node_id)
            out.append(resolved)
        return out

    def query_edges(
        self,
        *args,
        query=None,
        query_embeddings=None,
        include=["documents", "embeddings", "metadatas"],
        edge_type: Type[Edge] = Edge,
        **kwargs,
    ):
        if query_embeddings is None:
            if query is not None:
                query_embeddings = self._e._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")
        query_embeddings = cast(
            list[list[float]],
            normalize_embedding_rows(query_embeddings, allow_empty=False),
        )

        native = (
            None
            if args or set(kwargs) - {"where", "n_results"}
            else self._rust_postgres_projection_query(
            entity_kind="edge",
            query_embeddings=query_embeddings,
            where=kwargs.get("where"),
            n_results=int(kwargs.get("n_results", 10)),
            include=include,
            )
        )
        got = native or run_awaitable_blocking(self._e.backend.edge_query(
            query_embeddings=query_embeddings,
            *args,
            include=include,
            **kwargs,
        ))
        return self.edges_from_query_result(got, edge_type=edge_type)

    def nodes_from_single_or_id_query_result(
        self,
        got,
        node_type: Type[TNode] = Node,
    ) -> list[TNode]:
        docs: list[str] = cast(list[str], got.get("documents"))
        if docs is None:
            raise Exception("Missing docs")
        if len(docs) == 0:
            return []

        embs = got.get("embeddings")
        if embs is None:
            embs = [None] * len(docs)
        embs = cast(list[list[float] | None], embs)

        metadatas = cast(list[dict[str, Any]], got.get("metadatas"))
        if metadatas is None:
            raise Exception("Missing Metadatas")

        res: list[TNode] = []
        for d, emb, metadata in zip(docs, embs, metadatas):
            json_d = json.loads(d)
            json_d.update(
                {
                    "embedding": normalize_embedding_vector(emb),
                    "metadata": metadata,
                }
            )
            selected_type = pick_node_type(
                graph_kind=self._e.kg_graph_type,
                metadata=metadata,
                fallback=node_type,
            )
            res.append(selected_type.model_validate(json_d))
        return res

    def edges_from_single_or_id_query_result(
        self, got, edge_type: Type[Edge] = Edge, include=None
    ):
        if include is None:
            include = ["documents", "metadatas", "embeddings"]
        docs: list[str] = cast(list[str], got.get("documents"))
        if docs is None and "documents" in include:
            raise Exception("Missing docs")
        if docs is not None and len(docs) == 0:
            return []

        embs = got.get("embeddings")
        if embs is None:
            embs = [None] * len(docs or [])
        embs = cast(list[list[float] | None], embs)

        metadatas = cast(list[dict[str, Any]], got.get("metadatas"))
        if metadatas is None:
            raise Exception("Missing Metadatas")

        res = []
        for d, emb, metadata in zip(docs, embs, metadatas):
            json_d = json.loads(d)
            json_d.update(
                {
                    "embedding": normalize_embedding_vector(emb),
                    "metadata": metadata,
                }
            )
            selected_type = pick_edge_type(metadata=metadata, fallback=edge_type)
            res.append(selected_type.model_validate(json_d))
        return res

    def nodes_from_query_result(self, gots, node_type: Type[Node] = Node):
        res = []
        for i_q in range(len(gots["ids"])):
            n_doc = len(gots["ids"][i_q])
            if n_doc == 0:
                continue
            for _ids, docs, embs, metadatas in zip(
                gots.get("ids"),
                gots.get("documents")
                if gots.get("documents") is not None
                else [[]] * n_doc,
                gots.get("embeddings")
                if gots.get("embeddings") is not None
                else [[]] * n_doc,
                gots.get("metadatas")
                if gots.get("metadatas") is not None
                else [[]] * n_doc,
            ):
                docs = cast(list[str], docs)
                got = {"documents": docs, "embeddings": embs, "metadatas": metadatas}
                res.append(
                    self.nodes_from_single_or_id_query_result(got, node_type=node_type)
                )
        return res

    def edges_from_query_result(self, gots, edge_type: Type[Edge] = Edge):
        res = []
        for i_q in range(len(gots["ids"])):
            n_doc = len(gots["ids"][i_q])
            if n_doc == 0:
                continue
            for ids, docs, embs, metadatas in zip(
                gots.get("ids"),
                gots.get("documents")
                if gots.get("documents") is not None
                else [[]] * n_doc,
                gots.get("embeddings")
                if gots.get("embeddings") is not None
                else [[]] * n_doc,
                gots.get("metadatas")
                if gots.get("metadatas") is not None
                else [[]] * n_doc,
            ):
                docs = cast(list[str], docs)
                got = {
                    "ids": ids,
                    "documents": docs,
                    "embeddings": embs,
                    "metadatas": metadatas,
                }
                res.append(
                    self.edges_from_single_or_id_query_result(got, edge_type=edge_type)
                )
        return res

    def where_update_from_resolve_mode(
        self,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ):
        if resolve_mode == "active_only":
            return {"lifecycle_status": "active"}
        return {}

    def _infer_doc_id_from_ref(self, ref) -> Optional[str]:
        did = getattr(ref, "doc_id", None)
        if did:
            return did
        url = getattr(ref, "document_page_url", None) or ""
        try:
            tail = url.strip("/").split("/")[-1]
            return tail or None
        except Exception:
            return None

    def extract_reference_contexts(
        self,
        node_or_id: Union[Node | Edge, str],
        *,
        window_chars: int = 120,
        max_contexts: Optional[int] = None,
        prefer_label_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        from ..models import GraphEntityRefBase

        if isinstance(node_or_id, GraphEntityRefBase):
            obj = node_or_id
        else:
            got = run_awaitable_blocking(self._e.backend.node_get(ids=[node_or_id], include=["documents"]))
            doc_list = got.get("documents") or []
            if doc_list:
                obj = Node.model_validate_json(doc_list[0])
            else:
                got = run_awaitable_blocking(self._e.backend.edge_get(ids=[node_or_id], include=["documents"]))
                edoc_list = got.get("documents") or []
                if not edoc_list:
                    raise ValueError(f"Unknown node/edge id: {node_or_id}")
                obj = Edge.model_validate_json(edoc_list[0])

        label = getattr(obj, "label", None)
        out: List[Dict[str, Any]] = []
        doc_cache = {}

        def _coerce_to_referencable_text(text_or_ast_str):
            try:
                return "\n".join(
                    (
                        i["text"]
                        for i in ast.literal_eval(text_or_ast_str)["OCR_text_clusters"]
                    )
                )
            except Exception:
                return text_or_ast_str

        mentions = getattr(obj, "mentions", None) or []
        for mention in mentions:
            for span in mention.spans:
                doc_id = self._infer_doc_id_from_ref(span)

                pages = doc_cache.get(doc_id)
                full_doc = None
                if not pages:
                    full_doc = (
                        self._e.extract.fetch_document_text(doc_id) if doc_id else None
                    )
                    pages = self._e.extract.coerce_pages(full_doc)
                    doc_cache[doc_id] = pages
                excerpt = getattr(span, "excerpt", None)
                ctx_text = excerpt or (label or "")
                span_start = None
                span_end = None

                if full_doc:
                    page_relevant = {
                        p[0]: p[1]
                        for p in pages
                        if (p[0] >= span.page_number and p[0] <= span.page_number)
                    }
                    if span.page_number and span.page_number:
                        if span.page_number == span.page_number:
                            ctx_text = ""
                            if span.start_char and span.end_char:
                                try:
                                    _coerce_to_referencable_text(
                                        page_relevant[span.page_number]
                                    )
                                except Exception:
                                    raise
                                ctx_text = _coerce_to_referencable_text(
                                    page_relevant[span.page_number]
                                )[
                                    max(
                                        span.start_char - window_chars, 0
                                    ) : span.end_char + window_chars
                                ]

                    if ctx_text is None:
                        idx = full_doc.find(excerpt) if excerpt else -1
                        if idx < 0 and label and prefer_label_fallback:
                            idx = full_doc.find(label)

                        if idx >= 0:
                            length = (
                                len(excerpt)
                                if excerpt
                                else (len(label) if label else 0)
                            )
                            span_start = idx
                            span_end = idx + length
                            left = max(0, span_start - window_chars)
                            right = min(len(full_doc), span_end + window_chars)
                            ctx_text = full_doc[left:right]
                out.append(
                    {
                        "doc_id": doc_id,
                        "collection_page_url": getattr(
                            span, "collection_page_url", None
                        ),
                        "document_page_url": getattr(span, "document_page_url", None),
                        "start_page": getattr(span, "start_page", None),
                        "end_page": getattr(span, "end_page", None),
                        "start_char": getattr(span, "start_char", None),
                        "end_char": getattr(span, "end_char", None),
                        "insertion_method": getattr(span, "insertion_method", None),
                        "verification": (
                            span.verification.model_dump()
                            if getattr(span, "verification", None)
                            else None
                        ),
                        "context": ctx_text,
                        "mention": mention,
                        "loc_found": (span_start is not None),
                        "loc_span": [span_start, span_end]
                        if span_start is not None
                        else None,
                        "ref": span.model_dump(field_mode="backend"),
                    }
                )

                if max_contexts and len(out) >= max_contexts:
                    break

        return out

    # Doc-index helpers
    def node_ids_by_doc(
        self, doc_id: str, insertion_method: str | None = None
    ) -> list[str]:
        if insertion_method:
            return self.ids_with_insertion_method(
                kind="node",
                insertion_method=insertion_method,
                doc_id=doc_id,
            )
        if hasattr(self._e, "node_docs_collection"):
            rows = run_awaitable_blocking(self._e.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            ))
            result = set()
            for m in rows.get("metadatas") or []:
                if m and m.get("node_id"):
                    result.add(m.get("node_id"))
            if result:
                return sorted(result)
        got = run_awaitable_blocking(self._e.backend.node_get(where={"doc_id": doc_id}))
        result = set(got.get("ids") or [])
        adapter = getattr(self._e, "two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if callable(query):
            for row in query(entity_kind="node", limit=10000) or []:
                payload = dict(row.get("payload") or {})
                if (payload.get("metadata") or {}).get("doc_id") == doc_id:
                    result.add(str(payload.get("id") or row.get("key")))
        return sorted(result)

    def edge_ids_by_doc(
        self, doc_id: str, insertion_method: str | None = None
    ) -> list[str]:
        if insertion_method:
            return self.ids_with_insertion_method(
                kind="edge",
                insertion_method=insertion_method,
                doc_id=doc_id,
            )
        eps = run_awaitable_blocking(self._e.backend.edge_endpoints_get(
            where={"doc_id": doc_id}, include=["metadatas"]
        ))
        result = set()
        for m in eps.get("metadatas") or []:
            if m and m.get("edge_id"):
                result.add(m.get("edge_id"))
        adapter = getattr(self._e, "two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if callable(query):
            for row in query(entity_kind="edge", limit=10000) or []:
                payload = dict(row.get("payload") or {})
                if (payload.get("metadata") or {}).get("doc_id") == doc_id:
                    result.add(str(payload.get("id") or row.get("key")))
        return sorted(result)

    def edges_by_doc(self, doc_id: str, where: dict | None = None) -> list[str]:
        where = (
            {"doc_id": doc_id}
            if not where
            else {"$and": [{"doc_id": doc_id}] + [{k: v} for k, v in where.items()]}
        )
        rows = run_awaitable_blocking(self._e.backend.edge_refs_get(where=where, include=["documents"]))
        return list({json.loads(d)["edge_id"] for d in (rows.get("documents") or [])})

    def list_edges_with_ref_filter(
        self, doc_id: str, where: dict | None = None
    ) -> list[Edge]:
        ids = self.edges_by_doc(doc_id, where)
        if not ids:
            return []
        got = run_awaitable_blocking(self._e.backend.edge_get(ids=ids, include=["documents"]))
        return [Edge.model_validate_json(js) for js in (got.get("documents") or [])]

    def nodes_by_doc(self, doc_id: str, *, where: dict | None = None) -> list[str]:
        where = (
            {"doc_id": doc_id}
            if not where
            else {"$and": [{"doc_id": doc_id}] + [{k: v} for k, v in where.items()]}
        )
        rows = run_awaitable_blocking(self._e.backend.node_refs_get(where=where, include=["documents"]))
        return list({json.loads(d)["node_id"] for d in (rows.get("documents") or [])})

    def list_nodes_with_ref_filter(
        self, doc_id: str, *, where: dict | None = None
    ) -> list[Node]:
        ids = self.nodes_by_doc(doc_id, where=where)
        if not ids:
            return []
        got = run_awaitable_blocking(self._e.backend.node_get(ids=ids, include=["documents"]))
        return [Node.model_validate_json(js) for js in (got.get("documents") or [])]

    def ids_with_insertion_method(
        self,
        *,
        kind: str,
        insertion_method: str,
        ids: Optional[Sequence[str]] = None,
        doc_id: Optional[str] = None,
    ) -> list[str]:
        """
        Return distinct node_ids/edge_ids that have at least one reference row with the
        requested insertion_method. Falls back to scanning primary records if needed.
        """
        assert kind in ("node", "edge"), f"kind must be 'node' or 'edge', got {kind!r}"
        if kind == "node":
            key = "node_id"
            model_cls = Node
        else:
            key = "edge_id"
            model_cls = Edge

        where: dict[str, Any] = {"insertion_method": insertion_method}
        if doc_id:
            where["doc_id"] = doc_id
        if ids:
            where[key] = {"$in": list(ids)}

        get_refs = self._e.backend.node_refs_get if kind == "node" else self._e.backend.edge_refs_get
        rows = run_awaitable_blocking(get_refs(where=where, include=["metadatas"]))
        picked = {
            str(m.get(key)) for m in (rows.get("metadatas") or []) if m and m.get(key)
        }
        if picked:
            return sorted(picked)

        get_primary = self._e.backend.node_get if kind == "node" else self._e.backend.edge_get
        if ids:
            got = run_awaitable_blocking(get_primary(ids=list(ids), include=["documents"]))
            documents = got.get("documents") or []
            entity_ids = got.get("ids") or []
        else:
            got = run_awaitable_blocking(get_primary(include=["documents"]))
            documents = got.get("documents") or []
            entity_ids = got.get("ids") or []

        keep: set[str] = set()
        for entity_id, blob in zip(entity_ids, documents):
            ent = model_cls.model_validate_json(blob)
            for ref in ent.mentions or []:
                im = getattr(ref, "insertion_method", None)
                if im == insertion_method and (not doc_id or ref_doc_id(ref) == doc_id):
                    keep.add(entity_id)
                    break
        return sorted(keep)

    # Legacy names retained during migration
    def nodes_by_doc_index(
        self, doc_id: str, insertion_method: str | None = None
    ) -> list[str]:
        return self.node_ids_by_doc(doc_id, insertion_method=insertion_method)

    def edge_ids_by_doc_index(
        self, doc_id: str, insertion_method: str | None = None
    ) -> list[str]:
        return self.edge_ids_by_doc(doc_id, insertion_method=insertion_method)

    def load_node_map(self, *args, **kwargs):
        ids = kwargs.pop("ids", None)
        if ids is None and args:
            ids = args[0]
            args = args[1:]
        node_type = kwargs.pop("node_type", None)
        include = kwargs.pop("include", None)
        if args or kwargs:
            raise TypeError("load_node_map accepts only ids, node_type, and include")
        if ids is None:
            return {}
        nodes = self.get_nodes(
            ids=list(ids), node_type=node_type, include=include or ["documents"]
        )
        return {n.safe_get_id(): n for n in nodes}

    def load_edge_map(self, *args, **kwargs):
        ids = kwargs.pop("ids", None)
        if ids is None and args:
            ids = args[0]
            args = args[1:]
        edge_type = kwargs.pop("edge_type", None)
        include = kwargs.pop("include", None)
        if args or kwargs:
            raise TypeError("load_edge_map accepts only ids, edge_type, and include")
        if ids is None:
            return {}
        edges = self.get_edges(
            ids=list(ids), edge_type=edge_type, include=include or ["documents"]
        )
        return {e.safe_get_id(): e for e in edges}
