from __future__ import annotations

"""PostgreSQL + pgvector backend (Phase 3 rebased on Phase 2).

Scope (Phase 3)
---------------
Add Postgres support for the edge-related collections your engine already uses,
WITHOUT changing the engine/chroma-side contract:

Collections implemented:
- node            (vector store for nodes)
- edge            (vector store for edges)
- edge_endpoints  (hypergraph incidence materialization)
- edge_refs       (doc -> edge ref index)

Contracts preserved:
- Chroma-style where DSL (portable):
    * {"k": v}
    * {"k": {"$in": [...]}}
    * {"k": {"$gt/$gte/$lt/$lte/$ne": v}}
    * {"$and": [..]}, {"$or": [..]}
- get(): returns FLAT lists (Chroma Collection.get()):
    {"ids":[...], "documents":[...], "metadatas":[...], "embeddings":[...]}
- query(): returns NESTED lists (Chroma Collection.query()):
    {"ids":[[...]], "documents":[[...]], "metadatas":[[...]], "distances":[[...]]}

Notes:
- metadata is stored as JSONB (single column)
- document is stored as TEXT
- embedding is stored as pgvector Vector(dim) for node/edge only
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql

try:
    # pip install pgvector
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception as e:  # pragma: no cover
    Vector = None  # type: ignore
    _pgvector_import_error = e
else:
    _pgvector_import_error = None


Json = Dict[str, Any]
JSONB = psql.JSONB


class PostgresUnitOfWork:
    """Backend unit-of-work: wraps a SQL transaction."""

    def __init__(self, *, engine: sa.Engine):
        self._engine = engine

    def transaction(self):
        return self._engine.begin()


@dataclass(frozen=True)
class PgVectorConfig:
    dsn: str
    embedding_dim: int
    schema: str = "public"
    nodes_table: str = "gke_nodes"
    edges_table: str = "gke_edges"
    edge_endpoints_table: str = "gke_edge_endpoints"
    edge_refs_table: str = "gke_edge_refs"


# ----------------------------
# where DSL → SQLAlchemy
# ----------------------------

_NUMERIC_KEYS_DEFAULT: Set[str] = {"seq"}


def _json_text(col: sa.ColumnElement, key: str) -> sa.ColumnElement:
    # (metadata ->> 'key') yields text
    return col.op("->>")(key)


def where_jsonb(metadata_col: sa.ColumnElement, where: Json, *, numeric_keys: Optional[Set[str]] = None) -> sa.ColumnElement:
    """Translate Chroma-like `where` dict into SQLAlchemy boolean expression over JSONB."""
    numeric_keys = numeric_keys or _NUMERIC_KEYS_DEFAULT

    if not where:
        return sa.true()

    if "$and" in where:
        parts = where.get("$and") or []
        if not isinstance(parts, list):
            raise TypeError("$and must be a list")
        return sa.and_(*[where_jsonb(metadata_col, p, numeric_keys=numeric_keys) for p in parts]) if parts else sa.true()

    if "$or" in where:
        parts = where.get("$or") or []
        if not isinstance(parts, list):
            raise TypeError("$or must be a list")
        return sa.or_(*[where_jsonb(metadata_col, p, numeric_keys=numeric_keys) for p in parts]) if parts else sa.true()

    clauses: List[sa.ColumnElement] = []
    for k, v in where.items():
        if k in ("$and", "$or"):
            continue

        lhs_text = _json_text(metadata_col, k)

        # Numeric keys: cast for comparisons. Equality uses text semantics for cross-backend consistency.
        lhs_num = sa.cast(lhs_text, sa.BigInteger) if k in numeric_keys else None

        if isinstance(v, dict):
            if "$in" in v:
                vals = v["$in"]
                if not isinstance(vals, list):
                    raise TypeError(f"$in for {k} must be a list")
                clauses.append(lhs_text.in_([str(x) for x in vals]))
                continue

            # comparisons
            for op, rhs in v.items():
                if op == "$gt":
                    clauses.append((lhs_num if lhs_num is not None else lhs_text) > rhs)
                elif op == "$gte":
                    clauses.append((lhs_num if lhs_num is not None else lhs_text) >= rhs)
                elif op == "$lt":
                    clauses.append((lhs_num if lhs_num is not None else lhs_text) < rhs)
                elif op == "$lte":
                    clauses.append((lhs_num if lhs_num is not None else lhs_text) <= rhs)
                elif op == "$ne":
                    clauses.append(lhs_text != str(rhs))
                else:
                    raise NotImplementedError(f"Unsupported where operator: {op} (key={k})")
        else:
            # equality (text semantics to match Chroma's string-like metadata comparisons)
            clauses.append(lhs_text == str(v))

    return sa.and_(*clauses) if clauses else sa.true()


# ----------------------------
# Backend
# ----------------------------

class PgVectorBackend:
    """pgvector backend implementing node/edge + edge_endpoints/edge_refs operations."""
    
    def __init__(
        self,
        *,
        engine: sa.Engine,
        embedding_dim: int,
        schema: str = "public",
        nodes_table: str = "gke_nodes",
        edges_table: str = "gke_edges",
        edge_endpoints_table: str = "gke_edge_endpoints",
        edge_refs_table: str = "gke_edge_refs",
        numeric_keys: Optional[Set[str]] = None,
        node_docs_table: str = "gke_node_docs",
        node_refs_table: str = "gke_node_refs",
    ):
        if Vector is None:  # pragma: no cover
            raise RuntimeError(
                "pgvector is not installed. Install with `pip install pgvector` to use PgVectorBackend."
            ) from _pgvector_import_error

        self.engine = engine
        self.embedding_dim = int(embedding_dim)
        self.schema = schema
        self.numeric_keys = numeric_keys or set(_NUMERIC_KEYS_DEFAULT)

        md = sa.MetaData(schema=self.schema)

        self.nodes = sa.Table(
            nodes_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        )

        self.edges = sa.Table(
            edges_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        )

        self.edge_endpoints = sa.Table(
            edge_endpoints_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
            sa.Index(f"ix_{edge_endpoints_table}_endpoint", sa.text("(metadata->>'endpoint_node_id')")),
            sa.Index(f"ix_{edge_endpoints_table}_edge", sa.text("(metadata->>'edge_id')")),
        )

        self.edge_refs = sa.Table(
            edge_refs_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
            sa.Index(f"ix_{edge_refs_table}_doc", sa.text("(metadata->>'doc_id')")),
            sa.Index(f"ix_{edge_refs_table}_edge", sa.text("(metadata->>'edge_id')")),
        )
        self.node_docs = sa.Table(
            node_docs_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
            sa.Index(f"ix_{node_docs_table}_node", sa.text("(metadata->>'node_id')")),
            sa.Index(f"ix_{node_docs_table}_doc", sa.text("(metadata->>'doc_id')")),
        )
        self.node_refs = sa.Table(
            node_refs_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
            sa.Index(f"ix_{node_refs_table}_doc", sa.text("(metadata->>'doc_id')")),
            sa.Index(f"ix_{node_refs_table}_node", sa.text("(metadata->>'node_id')")),
        )
        self._md = md

    def ensure_schema(self) -> None:
        """Dev convenience: create extension/schema/tables if missing."""
        with self.engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            if self.schema and self.schema != "public":
                conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
            self._md.create_all(conn)

    # ----------------------------
    # Shared helpers
    # ----------------------------

    def _get_flat(self, table: sa.Table, *, ids: Optional[Sequence[str]], where: Optional[Json], include: List[str], limit: int) -> Dict[str, Any]:
        has_embedding = "embedding" in table.c
        cols = [table.c.id, table.c.document, table.c.metadata]
        if has_embedding:
            cols.append(table.c.embedding)

        q = sa.select(*cols).limit(int(limit))
        if ids is not None:
            q = q.where(table.c.id.in_(list(ids)))
        if where:
            q = q.where(where_jsonb(table.c.metadata, where, numeric_keys=self.numeric_keys))

        with self.engine.begin() as conn:
            rows = conn.execute(q).fetchall()

        out: Dict[str, Any] = {"ids": [r.id for r in rows]}
        if "documents" in include:
            out["documents"] = [r.document for r in rows]
        if "metadatas" in include:
            out["metadatas"] = [dict(r.metadata or {}) for r in rows]
        if "embeddings" in include and has_embedding:
            out["embeddings"] = [list(r.embedding) if r.embedding is not None else None for r in rows]
        return out

    def _delete(self, table: sa.Table, *, ids: Optional[Sequence[str]], where: Optional[Json]) -> None:
        stmt = sa.delete(table)
        if ids is not None:
            stmt = stmt.where(table.c.id.in_(list(ids)))
        if where:
            stmt = stmt.where(where_jsonb(table.c.metadata, where, numeric_keys=self.numeric_keys))
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def _upsert(self, table: sa.Table, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError("embeddings length must match ids length")

        rows: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            row: Dict[str, Any] = {
                "id": _id,
                "document": documents[i] if i < len(documents) else None,
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            if embeddings is not None and "embedding" in table.c:
                row["embedding"] = list(embeddings[i])
            rows.append(row)

        stmt = psql.insert(table).values(rows)
        set_map: Dict[str, Any] = {
            "document": stmt.excluded.document,
            "metadata": stmt.excluded.metadata,
            "updated_at": sa.func.now(),
        }
        if "embedding" in table.c:
            set_map["embedding"] = stmt.excluded.embedding
        stmt = stmt.on_conflict_do_update(index_elements=[table.c.id], set_=set_map)

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def _query_vector(self, table: sa.Table, *, query_embeddings: Sequence[Sequence[float]], n_results: int, where: Optional[Json], include: List[str]) -> Dict[str, Any]:
        if not query_embeddings:
            raise ValueError("query_embeddings is required")
        if "embedding" not in table.c:
            raise TypeError("vector query requested for a table without embedding")

        ids_out: List[List[str]] = []
        docs_out: List[List[Optional[str]]] = []
        metas_out: List[List[Json]] = []
        dists_out: List[List[float]] = []

        with self.engine.begin() as conn:
            for qv in query_embeddings:
                q = sa.select(
                    table.c.id,
                    table.c.document,
                    table.c.metadata,
                    (table.c.embedding.op("<=>")(sa.bindparam("qv"))).label("distance"),
                ).where(table.c.embedding.is_not(None))

                if where:
                    q = q.where(where_jsonb(table.c.metadata, where, numeric_keys=self.numeric_keys))

                q = q.order_by(sa.text("distance asc")).limit(int(n_results))
                rows = conn.execute(q, {"qv": list(qv)}).fetchall()

                ids_out.append([r.id for r in rows])
                docs_out.append([r.document for r in rows])
                metas_out.append([dict(r.metadata or {}) for r in rows])
                dists_out.append([float(r.distance) for r in rows])

        out: Dict[str, Any] = {"ids": ids_out}
        if "documents" in include:
            out["documents"] = docs_out
        if "metadatas" in include:
            out["metadatas"] = metas_out
        if "distances" in include:
            out["distances"] = dists_out
        return out


    def edge_update(
        self,
        *,
        ids: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if len(ids) != len(metadatas):
            raise ValueError("metadatas length must match ids length")

        patch_param = sa.bindparam("patch", type_=JSONB)

        with self.engine.begin() as conn:
            for _id, patch in zip(ids, metadatas):
                stmt = (
                    sa.update(self.edges)
                    .where(self.edges.c.id == _id)
                    .values(
                        metadata=self.edges.c.metadata.op("||")(patch_param),
                        updated_at=sa.func.now(),
                    )
                )
                conn.execute(stmt, {"patch": patch})

    # ----------------------------
    # Nodes refs
    # ----------------------------

    def node_refs_add(self, *, ids, documents, metadatas, embeddings=None) -> None:
        self._upsert(self.node_refs, ids=ids, documents=documents, metadatas=metadatas, embeddings=None)

    def node_refs_get(self, *, ids=None, where=None, include=None, limit=200) -> dict:
        include = include or ["documents", "metadatas"]
        return self._get_flat(self.node_refs, ids=ids, where=where, include=include, limit=limit)

    def node_refs_delete(self, *, ids=None, where=None) -> None:
        self._delete(self.node_refs, ids=ids, where=where)
    # ----------------------------
    # Nodes
    # ----------------------------

    def node_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        self._upsert(self.nodes, ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def node_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None, include: Optional[List[str]] = None, limit: int = 200) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        return self._get_flat(self.nodes, ids=ids, where=where, include=include, limit=limit)

    def node_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None) -> None:
        self._delete(self.nodes, ids=ids, where=where)

    def node_query(self, *, query_embeddings: Sequence[Sequence[float]], n_results: int = 10, where: Optional[Json] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "distances"]
        return self._query_vector(self.nodes, query_embeddings=query_embeddings, n_results=n_results, where=where, include=include)

    # ----------------------------
    # Edges
    # ----------------------------

    def edge_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json], embeddings: Optional[Sequence[Sequence[float]]] = None) -> None:
        self._upsert(self.edges, ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def edge_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None, include: Optional[List[str]] = None, limit: int = 200) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        return self._get_flat(self.edges, ids=ids, where=where, include=include, limit=limit)

    def edge_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None) -> None:
        self._delete(self.edges, ids=ids, where=where)

    def edge_query(self, *, query_embeddings: Sequence[Sequence[float]], n_results: int = 10, where: Optional[Json] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "distances"]
        return self._query_vector(self.edges, query_embeddings=query_embeddings, n_results=n_results, where=where, include=include)

    # ----------------------------
    # Edge endpoints
    # ----------------------------

    def edge_endpoints_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json], embeddings: Any = None) -> None:
        # embeddings ignored for this collection (kept for callsite compatibility)
        self._upsert(self.edge_endpoints, ids=ids, documents=documents, metadatas=metadatas, embeddings=None)

    def edge_endpoints_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None, include: Optional[List[str]] = None, limit: int = 200) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        return self._get_flat(self.edge_endpoints, ids=ids, where=where, include=include, limit=limit)

    def edge_endpoints_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None) -> None:
        self._delete(self.edge_endpoints, ids=ids, where=where)

    # ----------------------------
    # Edge refs
    # ----------------------------

    def edge_refs_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json], embeddings: Any = None) -> None:
        # embeddings ignored for this collection (kept for callsite/test compatibility)
        self._upsert(self.edge_refs, ids=ids, documents=documents, metadatas=metadatas, embeddings=None)

    def edge_refs_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None, include: Optional[List[str]] = None, limit: int = 200) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        return self._get_flat(self.edge_refs, ids=ids, where=where, include=include, limit=limit)

    def edge_refs_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None) -> None:
        self._delete(self.edge_refs, ids=ids, where=where)
    # ----------------------------
    # Node docs
    # ----------------------------
    def node_docs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,   # accepted, ignored
    ) -> None:
        self._upsert(self.node_docs, ids=ids, documents=documents, metadatas=metadatas, embeddings=None)

    def node_docs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        return self._get_flat(self.node_docs, ids=ids, where=where, include=include, limit=limit)

    def node_docs_delete(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
    ) -> None:
        self._delete(self.node_docs, ids=ids, where=where)

    def node_update(self, *, ids: Sequence[str], metadatas: Sequence[dict[str, Any]]) -> None:
        if len(ids) != len(metadatas):
            raise ValueError("metadatas length must match ids length")

        patch_param = sa.bindparam("patch", type_=JSONB)

        with self.engine.begin() as conn:
            for _id, patch in zip(ids, metadatas):
                stmt = (
                    sa.update(self.nodes)
                    .where(self.nodes.c.id == _id)
                    .values(
                        metadata=self.nodes.c.metadata.op("||")(patch_param),
                        updated_at=sa.func.now(),
                    )
                )
                conn.execute(stmt, {"patch": patch})    
        
def build_postgres_backend(cfg: PgVectorConfig) -> Tuple[PgVectorBackend, PostgresUnitOfWork]:
    """Convenience helper for engine wiring."""
    engine = sa.create_engine(cfg.dsn, future=True)
    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edges_table=cfg.edges_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
        edge_refs_table=cfg.edge_refs_table,
    )
    backend.ensure_schema()
    return backend, PostgresUnitOfWork(engine=engine)
