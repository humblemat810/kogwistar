from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
    cast,
)

from ...entity_registry import (
    default_edge_type_for_graph_kind,
    default_node_type_for_graph_kind,
    pick_edge_type,
    pick_node_type,
)
from ...json_types import JsonObject, JsonValue
from ...typing_interfaces import ProjectionBackendLike
from ...utils.embedding_vectors import (
    normalize_embedding_rows,
    normalize_embedding_vector,
)
from ..async_compat import run_awaitable_blocking
from ..models import Document, Edge, Node
from ..utils.refs import ref_doc_id
from ..vector_search import VectorSearchHit, similarity_from_distance
from .base import NamespaceProxy

if TYPE_CHECKING:
    from ..engine import GraphKnowledgeEngine

TNode = TypeVar("TNode", bound=Node)
TEdge = TypeVar("TEdge", bound=Edge)


def _json_object(value: object) -> JsonObject:
    """Narrow backend payloads at the graph read boundary."""

    if not isinstance(value, Mapping):
        raise TypeError("graph backend returned a non-object payload")
    return cast(JsonObject, {str(key): item for key, item in value.items()})


def _json_objects(value: object) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [_json_object(item) for item in value if isinstance(item, Mapping)]


def _json_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _json_string_rows(value: object) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    return [_json_strings(row) for row in value]


def _json_float_rows(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for row in value:
        if isinstance(row, list):
            rows.append([float(item) for item in row if isinstance(item, (int, float))])
    return rows


def _string_sequence(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("expected a sequence of identifiers")
    return [str(item) for item in value]


def _json_embeddings(value: object) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    result: list[JsonValue] = []
    for row in value:
        if row is None:
            result.append(None)
        elif isinstance(row, (list, tuple)):
            result.append([float(item) for item in row])
    return result


def _json_float(value: JsonValue, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


class ReadSubsystem(NamespaceProxy["GraphKnowledgeEngine"]):
    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        super().__init__(engine)

    @staticmethod
    def _native_equality_filter(where: object) -> dict[str, JsonValue] | None:
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
        return cast(dict[str, JsonValue], dict(where))

    def _rust_postgres_projection_get(
        self,
        *,
        entity_kind: str,
        ids: Sequence[str] | None,
        where: object,
        limit: int | None,
        include: list[str],
    ) -> dict[str, JsonValue] | None:
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
            cast(ProjectionBackendLike, self._e.backend).nodes.name
            if entity_kind == "node"
            else cast(ProjectionBackendLike, self._e.backend).edges.name
            if entity_kind == "edge"
            else cast(ProjectionBackendLike, self._e.backend).documents.name
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
        result: JsonObject = {
            "ids": [str(record.get("id") or "") for record in records]
        }
        if "documents" in effective_include:
            result["documents"] = [record.get("document") for record in records]
        if "metadatas" in effective_include:
            result["metadatas"] = [_json_object(record.get("metadata") or {}) for record in records]
        if "embeddings" in effective_include:
            result["embeddings"] = _json_embeddings(
                [normalize_embedding_vector(record.get("embedding")) for record in records]
            )
        return result

    def _rust_postgres_projection_query(
        self,
        *,
        entity_kind: str,
        query_embeddings: Sequence[Sequence[float]],
        where: object,
        n_results: int,
        include: list[str],
    ) -> dict[str, JsonValue] | None:
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
            cast(ProjectionBackendLike, self._e.backend).nodes.name
            if entity_kind == "node"
            else cast(ProjectionBackendLike, self._e.backend).edges.name
        )
        matches = meta.graph_projection_vector_query(
            namespace=getattr(self._e, "namespace", "default"),
            workspace_id=None,
            graph_space=None,
            table=table,
            embedding=list(query_embeddings[0]),
            embedding_dim=int(cast(ProjectionBackendLike, self._e.backend).embedding_dim),
            metadata=metadata,
            metric=str(cast(ProjectionBackendLike, self._e.backend).distance),
            limit=int(n_results),
        )
        records = [_json_object(match.get("record") or {}) for match in matches]
        effective_include = include or ["documents", "metadatas", "distances"]
        result: JsonObject = {
            "ids": [[str(record.get("id") or "") for record in records]]
        }
        if "documents" in effective_include:
            result["documents"] = [[record.get("document") for record in records]]
        if "metadatas" in effective_include:
            result["metadatas"] = [[_json_object(record.get("metadata") or {}) for record in records]]
        if "embeddings" in effective_include:
            result["embeddings"] = [_json_embeddings(
                [normalize_embedding_vector(record.get("embedding")) for record in records]
            )]
        if "distances" in effective_include:
            result["distances"] = cast(
                JsonValue,
                [[_json_float(match.get("distance", 0.0)) for match in matches]],
            )
        return result

    def _stage1_fallback_get(
        self,
        *,
        entity_kind: str,
        ids: Sequence[str] | None,
        where: object,
        limit: int | None,
        include: list[str],
    ) -> dict[str, JsonValue] | None:
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
            metadata=(
                dict(cast(Mapping[str, object], where))
                if isinstance(where, Mapping)
                else {}
            ),
            limit=limit,
        )
        if not isinstance(rows, Iterable):
            return None
        rows = [cast(Mapping[str, object], row) for row in rows if isinstance(row, Mapping)]
        if not rows:
            return None
        payloads = [_json_object(row.get("payload") or {}) for row in rows]
        result: JsonObject = {
            "ids": [str(payload.get("id") or row.get("key") or "") for payload, row in zip(payloads, rows)]
        }
        if "documents" in include:
            result["documents"] = [payload.get("document") for payload in payloads]
        if "metadatas" in include:
            result["metadatas"] = [_json_object(payload.get("metadata") or {}) for payload in payloads]
        if "embeddings" in include:
            result["embeddings"] = [None for _ in payloads]
        return result

    def _node_get_raw(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 200,
        include: list[str] | None = None,
    ) -> dict[str, JsonValue]:
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
        result = _json_object(run_awaitable_blocking(self._e.backend.node_get(
            ids=ids,
            include=include,
            where=where,
            limit=limit,
        )))
        if not result.get("ids"):
            return self._stage1_fallback_get(
                entity_kind="node", ids=ids, where=where, limit=limit, include=include
            ) or result
        return result

    def _edge_get_raw(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 400,
        include: list[str] | None = None,
    ) -> dict[str, JsonValue]:
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
        result = _json_object(run_awaitable_blocking(self._e.backend.edge_get(
            ids=ids,
            include=include,
            where=where,
            limit=limit,
        )))
        if not result.get("ids"):
            return self._stage1_fallback_get(
                entity_kind="edge", ids=ids, where=where, limit=limit, include=include
            ) or result
        return result

    def node_exists(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
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
        where: dict[str, JsonValue] | None = None,
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
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, JsonValue]]:
        got = self._node_get_raw(
            ids=ids,
            where=where,
            limit=limit,
            include=["metadatas"],
        )
        return _json_objects(got.get("metadatas"))

    def get_edge_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 400,
    ) -> list[dict[str, JsonValue]]:
        got = self._edge_get_raw(
            ids=ids,
            where=where,
            limit=limit,
            include=["metadatas"],
        )
        return _json_objects(got.get("metadatas"))

    def get_edge_endpoints(
        self,
        *,
        where: dict[str, JsonValue] | None = None,
        include: list[str] | None = None,
        limit: int | None = 10000,
    ) -> dict[str, JsonValue]:
        """Read structural endpoint rows through the engine read boundary."""
        return _json_object(run_awaitable_blocking(
            self._e.backend.edge_endpoints_get(
                where=where,
                include=include or ["documents", "metadatas"],
                limit=limit,
            )
            ))

    # Canonical read API
    def get_nodes(
        self,
        ids: Sequence[str] | None = None,
        node_type: type[Node] | None = None,
        include: list[str] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 200,
        resolve_mode: Literal[
            "active_only", "redirect", "include_tombstones"
        ] = "active_only",
    ) -> list[Node]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        else:
            # Model reconstruction requires both payload and metadata, even
            # when compatibility callers pass a narrow or empty include list.
            include = [*include]
            if "documents" not in include:
                include.append("documents")
            if "metadatas" not in include:
                include.append("metadatas")
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
        edge_type: type[Edge] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 400,
        include: list[str] | None = None,
        resolve_mode: Literal[
            "active_only", "redirect", "include_tombstones"
        ] = "active_only",
    ) -> list[Edge]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        else:
            # Edge reconstruction and structural traversal require both
            # payload and metadata for every compatibility include shape.
            include = [*include]
            if "documents" not in include:
                include.append("documents")
            if "metadatas" not in include:
                include.append("metadatas")
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
        raw_doc_get_result = self._rust_postgres_projection_get(
            entity_kind="document",
            ids=[doc_id],
            where=None,
            limit=1,
            include=["documents", "metadatas"],
        )
        if raw_doc_get_result is None:
            raw_doc_get_result = run_awaitable_blocking(
                self._e.backend.document_get(ids=[doc_id])
            )
        doc_get_result = _json_object(raw_doc_get_result)
        ids = _json_strings(doc_get_result.get("ids"))
        if len(ids) == 0:
            raise ValueError(f"no document found for doc id = {doc_id}")
        metadatas = _json_objects(doc_get_result.get("metadatas"))
        docs = _json_strings(doc_get_result.get("documents"))

        if not docs or not metadatas:
            raise ValueError("Invalid documnet metadata")
        metadata = metadatas[0]

        doc = Document(
            id=ids[0],
            content=docs[0],
            metadata=metadata,
            domain_id=(
                None
                if metadata.get("domain_id") is None
                else str(metadata.get("domain_id"))
            ),
            type=str(metadata.get("type") or "text"),
            processed=bool(metadata.get("processed", False)),
            embeddings=None,
            source_map=None,
        )
        return doc

    def query_nodes(
        self,
        *args: object,
        query: str | None = None,
        query_embeddings: Sequence[float] | Sequence[Sequence[float]] | None = None,
        include: list[str] | None = None,
        node_type: type[TNode] | None = None,
        **kwargs: object,
    ) -> list[list[TNode]]:
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
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
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
            n_results=int(cast(int, kwargs.get("n_results", 10))),
            include=include,
            )
        )
        got = cast(
            Mapping[str, JsonValue],
            native
            or run_awaitable_blocking(
                self._e.backend.node_query(
                    *args,
                    query_embeddings=query_embeddings,
                    include=include,
                    **kwargs,
                )
            ),
        )
        return self.nodes_from_query_result(
            got, node_type=cast(type[TNode], node_type or Node)
        )

    def _coerce_ts_utc(self, raw: object) -> datetime | None:
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
        node_type: type[Node],
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
        query_embeddings: Sequence[float] | Sequence[Sequence[float]] | None = None,
        as_of_ts: datetime | str,
        where: dict[str, JsonValue] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: type[Node] = Node,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        similarity_threshold: float | None = None,
        **kwargs: object,
    ) -> list[Node]:
        return [
            node
            for hit in self.search_nodes_as_of_scored(
                query=query,
                query_embeddings=query_embeddings,
                as_of_ts=as_of_ts,
                where=where,
                n_results=n_results,
                follow_redirects=follow_redirects,
                node_type=node_type,
                include=include,
                max_redirect_hops=max_redirect_hops,
                similarity_threshold=similarity_threshold,
                **kwargs,
            )
            for node in (hit.node,)
        ]

    def search_nodes_as_of_scored(
        self,
        *,
        query: str | None = None,
        query_embeddings: Sequence[float] | Sequence[Sequence[float]] | None = None,
        as_of_ts: datetime | str,
        where: dict[str, JsonValue] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: type[Node] = Node,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        similarity_threshold: float | None = None,
        **kwargs: object,
    ) -> list[VectorSearchHit[Node]]:
        """Return as-of nodes together with backend order scores.

        Results retain raw backend distance and expose one higher-is-better
        similarity value.  This is a read-only projection, never graph truth.
        """
        if similarity_threshold is not None:
            threshold = float(similarity_threshold)
            if threshold != threshold:
                raise ValueError("similarity_threshold must not be NaN")
        else:
            threshold = None
        if include is None:
            include = ["documents", "embeddings", "metadatas", "distances"]
        elif "distances" not in include:
            include = [*include, "distances"]
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
        try:
            got = _json_object(run_awaitable_blocking(self._e.backend.node_query(**query_kwargs)))
        except Exception as exc:
            # Chroma's embedded Rust reader can briefly lose an HNSW segment
            # after a local persistent update. Historical filtering is still
            # correct over non-semantic node records, so fall back to a
            # metadata/payload candidate read without retrying the vector path.
            message = str(exc).lower()
            is_chroma_hnsw_gap = (
                self._e.backend.__class__.__name__ in {"ChromaBackend", "AsyncChromaBackend"}
                and "nothing found on disk" in message
                and "hnsw segment reader" in message
            )
            if not is_chroma_hnsw_gap:
                raise
            got = _json_object(run_awaitable_blocking(
                self._e.backend.node_get(
                    where=where,
                    limit=max(int(n_results), 10_000),
                    include=["documents", "metadatas"],
                )
            ))
        batches = self.nodes_from_query_result(got, node_type=node_type)
        candidates = [node for batch in batches for node in batch] if batches else []
        raw_ids = [item for batch in _json_string_rows(got.get("ids")) for item in batch]
        raw_distances = [item for batch in _json_float_rows(got.get("distances")) for item in batch]
        metric = str(
            getattr(getattr(self._e, "embedding_profile", None), "similarity_metric", None)
            or "cosine"
        ).lower()
        distance_kind = str(
            getattr(self._e.backend, "vector_distance_kind", "distance")
        )
        scores_by_id = {
            str(node_id): (
                float(raw_distances[index])
                if index < len(raw_distances) and raw_distances[index] is not None
                else None
            )
            for index, node_id in enumerate(raw_ids)
        }
        cache = {str(node.safe_get_id()): node for node in candidates}

        out: list[VectorSearchHit[Node]] = []
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
            raw_distance = scores_by_id.get(str(node.safe_get_id()))
            similarity = similarity_from_distance(
                raw_distance,
                metric=metric,
                distance_kind=distance_kind,
            )
            if threshold is not None:
                if similarity is None:
                    raise ValueError(
                        "similarity_threshold requires backend distance results"
                    )
                if similarity < threshold:
                    continue
            out.append(
                VectorSearchHit(
                    node=resolved,
                    raw_distance=raw_distance,
                    similarity=similarity,
                    metric=metric,
                    distance_kind=distance_kind,
                )
            )
        return out

    def query_edges(
        self,
        *args: object,
        query: str | None = None,
        query_embeddings: Sequence[float] | Sequence[Sequence[float]] | None = None,
        include: list[str] | None = None,
        edge_type: type[TEdge] | None = None,
        **kwargs: object,
    ) -> list[list[TEdge]]:
        if query_embeddings is None:
            if query is not None:
                query_embeddings = self._e._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
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
            n_results=int(cast(int, kwargs.get("n_results", 10))),
            include=include,
            )
        )
        got = cast(
            Mapping[str, JsonValue],
            native
            or run_awaitable_blocking(
                self._e.backend.edge_query(
                    query_embeddings=query_embeddings,
                    *args,
                    include=include,
                    **kwargs,
                )
            ),
        )
        return self.edges_from_query_result(
            got, edge_type=cast(type[TEdge], edge_type or Edge)
        )

    def nodes_from_single_or_id_query_result(
        self,
        got: Mapping[str, JsonValue],
        node_type: type[TNode] = Node,
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

        metadatas = cast(list[dict[str, JsonValue]], got.get("metadatas"))
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
            res.append(cast(TNode, selected_type.model_validate(json_d)))
        return res

    def edges_from_single_or_id_query_result(
        self,
        got: Mapping[str, JsonValue],
        edge_type: type[TEdge] = Edge,
        include: list[str] | None = None,
    ) -> list[TEdge]:
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

        metadatas = cast(list[dict[str, JsonValue]], got.get("metadatas"))
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
            res.append(cast(TEdge, selected_type.model_validate(json_d)))
        return res

    def nodes_from_query_result(
        self, gots: Mapping[str, JsonValue], node_type: type[TNode] = Node
    ) -> list[list[TNode]]:
        res: list[list[TNode]] = []
        id_rows = cast(list[list[str]], gots.get("ids") or [])
        document_rows = cast(list[list[str]], gots.get("documents") or [[] for _ in id_rows])
        embedding_rows = cast(list[list[list[float] | None]], gots.get("embeddings") or [[] for _ in id_rows])
        metadata_rows = cast(list[list[dict[str, JsonValue]]], gots.get("metadatas") or [[] for _ in id_rows])
        for index, ids in enumerate(id_rows):
            if not ids:
                continue
            got = {
                "documents": document_rows[index] if index < len(document_rows) else [],
                "embeddings": embedding_rows[index] if index < len(embedding_rows) else [],
                "metadatas": metadata_rows[index] if index < len(metadata_rows) else [],
            }
            res.append(self.nodes_from_single_or_id_query_result(got, node_type=node_type))
        return res

    def edges_from_query_result(
        self, gots: Mapping[str, JsonValue], edge_type: type[TEdge] = Edge
    ) -> list[list[TEdge]]:
        res: list[list[TEdge]] = []
        id_rows = cast(list[list[str]], gots.get("ids") or [])
        document_rows = cast(list[list[str]], gots.get("documents") or [[] for _ in id_rows])
        embedding_rows = cast(list[list[list[float] | None]], gots.get("embeddings") or [[] for _ in id_rows])
        metadata_rows = cast(list[list[dict[str, JsonValue]]], gots.get("metadatas") or [[] for _ in id_rows])
        for index, ids in enumerate(id_rows):
            if not ids:
                continue
            got = {
                "ids": ids,
                "documents": document_rows[index] if index < len(document_rows) else [],
                "embeddings": embedding_rows[index] if index < len(embedding_rows) else [],
                "metadatas": metadata_rows[index] if index < len(metadata_rows) else [],
            }
            res.append(self.edges_from_single_or_id_query_result(got, edge_type=edge_type))
        return res

    def where_update_from_resolve_mode(
        self,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ) -> dict[str, str]:
        if resolve_mode == "active_only":
            return {"lifecycle_status": "active"}
        return {}

    def _infer_doc_id_from_ref(self, ref: object) -> str | None:
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
        node_or_id: Node | Edge | str,
        *,
        window_chars: int = 120,
        max_contexts: int | None = None,
        prefer_label_fallback: bool = True,
    ) -> list[dict[str, object]]:
        from ..models import GraphEntityRefBase

        if isinstance(node_or_id, GraphEntityRefBase):
            obj = node_or_id
        else:
            got = _json_object(
                run_awaitable_blocking(self._e.backend.node_get(ids=[node_or_id], include=["documents"]))
            )
            doc_list = _json_strings(got.get("documents"))
            if doc_list:
                obj = Node.model_validate_json(doc_list[0])
            else:
                got = _json_object(
                    run_awaitable_blocking(self._e.backend.edge_get(ids=[node_or_id], include=["documents"]))
                )
                edoc_list = _json_strings(got.get("documents"))
                if not edoc_list:
                    raise ValueError(f"Unknown node/edge id: {node_or_id}")
                obj = Edge.model_validate_json(edoc_list[0])

        label = getattr(obj, "label", None)
        out: list[dict[str, object]] = []
        doc_cache = {}

        def _coerce_to_referencable_text(text_or_ast_str: str) -> str:
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
            rows = _json_object(run_awaitable_blocking(self._e.backend.node_docs_get(
                where={"doc_id": doc_id}, include=["metadatas"]
            )))
            result = set()
            for m in _json_objects(rows.get("metadatas")):
                if m.get("node_id"):
                    result.add(str(m.get("node_id")))
            if result:
                return sorted(result)
        got = _json_object(run_awaitable_blocking(self._e.backend.node_get(where={"doc_id": doc_id})))
        result = set(_json_strings(got.get("ids")))
        adapter = getattr(self._e, "two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if callable(query):
            rows = query(entity_kind="node", limit=10000) or []
            if not isinstance(rows, Iterable):
                rows = []
            for raw_row in rows:
                if not isinstance(raw_row, Mapping):
                    continue
                row = cast(Mapping[str, JsonValue], raw_row)
                payload = _json_object(row.get("payload"))
                if _json_object(payload.get("metadata")).get("doc_id") == doc_id:
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
        eps = _json_object(run_awaitable_blocking(self._e.backend.edge_endpoints_get(
            where={"doc_id": doc_id}, include=["metadatas"]
        )))
        result = set()
        for m in _json_objects(eps.get("metadatas")):
            if m.get("edge_id"):
                result.add(str(m.get("edge_id")))
        adapter = getattr(self._e, "two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if callable(query):
            rows = query(entity_kind="edge", limit=10000) or []
            if not isinstance(rows, Iterable):
                rows = []
            for raw_row in rows:
                if not isinstance(raw_row, Mapping):
                    continue
                row = cast(Mapping[str, JsonValue], raw_row)
                payload = _json_object(row.get("payload"))
                if _json_object(payload.get("metadata")).get("doc_id") == doc_id:
                    result.add(str(payload.get("id") or row.get("key")))
        return sorted(result)

    def edges_by_doc(
        self, doc_id: str, where: dict[str, JsonValue] | None = None
    ) -> list[str]:
        query_where: dict[str, object] = (
            {"doc_id": doc_id}
            if not where
            else {"$and": [{"doc_id": doc_id}] + [{k: v} for k, v in where.items()]}
        )
        rows = _json_object(run_awaitable_blocking(self._e.backend.edge_refs_get(where=query_where, include=["documents"])))
        return list({json.loads(d)["edge_id"] for d in _json_strings(rows.get("documents"))})

    def list_edges_with_ref_filter(
        self, doc_id: str, where: dict | None = None
    ) -> list[Edge]:
        ids = self.edges_by_doc(doc_id, where)
        if not ids:
            return []
        got = _json_object(run_awaitable_blocking(self._e.backend.edge_get(ids=ids, include=["documents"])))
        return [Edge.model_validate_json(js) for js in _json_strings(got.get("documents"))]

    def nodes_by_doc(self, doc_id: str, *, where: dict | None = None) -> list[str]:
        where = (
            {"doc_id": doc_id}
            if not where
            else {"$and": [{"doc_id": doc_id}] + [{k: v} for k, v in where.items()]}
        )
        rows = _json_object(run_awaitable_blocking(self._e.backend.node_refs_get(where=where, include=["documents"])))
        return list({json.loads(d)["node_id"] for d in _json_strings(rows.get("documents"))})

    def list_nodes_with_ref_filter(
        self, doc_id: str, *, where: dict | None = None
    ) -> list[Node]:
        ids = self.nodes_by_doc(doc_id, where=where)
        if not ids:
            return []
        got = _json_object(run_awaitable_blocking(self._e.backend.node_get(ids=ids, include=["documents"])))
        return [Node.model_validate_json(js) for js in _json_strings(got.get("documents"))]

    def ids_with_insertion_method(
        self,
        *,
        kind: str,
        insertion_method: str,
        ids: Sequence[str] | None = None,
        doc_id: str | None = None,
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

        where: dict[str, JsonValue] = {"insertion_method": insertion_method}
        if doc_id:
            where["doc_id"] = doc_id
        if ids:
            where[key] = {"$in": list(ids)}

        get_refs = self._e.backend.node_refs_get if kind == "node" else self._e.backend.edge_refs_get
        rows = _json_object(run_awaitable_blocking(get_refs(where=where, include=["metadatas"])))
        picked = {
            str(m.get(key)) for m in _json_objects(rows.get("metadatas")) if m.get(key)
        }
        if picked:
            return sorted(picked)

        get_primary = self._e.backend.node_get if kind == "node" else self._e.backend.edge_get
        if ids:
            got = _json_object(run_awaitable_blocking(get_primary(ids=list(ids), include=["documents"])))
            documents = _json_strings(got.get("documents"))
            entity_ids = _json_strings(got.get("ids"))
        else:
            got = _json_object(run_awaitable_blocking(get_primary(include=["documents"])))
            documents = _json_strings(got.get("documents"))
            entity_ids = _json_strings(got.get("ids"))

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

    def load_node_map(self, *args: object, **kwargs: object) -> dict[str, Node]:
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
            ids=_string_sequence(ids),
            node_type=cast(type[Node] | None, node_type if isinstance(node_type, type) else None),
            include=cast(list[str] | None, include if isinstance(include, list) else None)
            or ["documents"],
        )
        return {n.safe_get_id(): n for n in nodes}

    def load_edge_map(self, *args: object, **kwargs: object) -> dict[str, Edge]:
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
            ids=_string_sequence(ids),
            edge_type=cast(type[Edge] | None, edge_type if isinstance(edge_type, type) else None),
            include=cast(list[str] | None, include if isinstance(include, list) else None)
            or ["documents"],
        )
        return {e.safe_get_id(): e for e in edges}
