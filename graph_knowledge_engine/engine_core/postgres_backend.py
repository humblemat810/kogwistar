from __future__ import annotations

"""PostgreSQL + pgvector backend.

This module provides a *Chroma-shaped* backend surface so `engine.py` can swap
between ChromaDB and Postgres with minimal friction.

Key compatibility rules
-----------------------
* where-dsl: Chroma-like dicts (portable):
    - {"k": v}
    - {"k": {"$in": [...]}}
    - {"k": {"$gt/$gte/$lt/$lte/$ne": v}}
    - {"$and": [..]}, {"$or": [..]}

* get(): returns FLAT lists (mirrors Chroma Collection.get() shape)
    {"ids": [...], "documents": [...], "metadatas": [...], "embeddings": [...]}.

* query(): returns NESTED lists (mirrors Chroma Collection.query() shape)
    {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}.

Notes
-----
* metadata is stored as JSONB
* document is stored as TEXT
* embedding is stored as pgvector Vector(dim) for vector tables only

Collections implemented (current scope)
--------------------------------------
Vector collections:
* nodes
* edges

Index/materialization collections (non-vector):
* edge_endpoints  (hypergraph incidence materialization)
* edge_refs       (doc -> edge ref index)
* node_docs       (node -> doc index)
* node_refs       (doc -> node ref index)

"""

from dataclasses import dataclass
from contextlib import contextmanager
import contextvars
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql

from ..utils.embedding_vectors import normalize_embedding_rows, normalize_embedding_vector

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


_pg_uow_conn: contextvars.ContextVar[sa.Connection | None] = contextvars.ContextVar(
    "gke_pg_uow_conn", default=None
)


@contextmanager
def _set_active_conn(conn: sa.Connection):
    token = _pg_uow_conn.set(conn)
    try:
        yield
    finally:
        _pg_uow_conn.reset(token)


def get_active_conn() -> sa.Connection | None:
    return _pg_uow_conn.get()


class PostgresUnitOfWork:
    """Backend unit-of-work: wraps a SQL transaction and exposes it to the backend.

    PgVectorBackend methods will *join* the active connection if one is set.
    """

    def __init__(self, *, engine: sa.Engine):
        self._engine = engine

    @contextmanager
    def transaction(self):
        existing = get_active_conn()
        if existing is not None:
            # Join outer transaction
            yield
            return

        with self._engine.begin() as conn:
            with _set_active_conn(conn):
                yield


@dataclass(frozen=True)
class PgVectorConfig:
    dsn: str
    embedding_dim: int
    # Distance metric used for similarity search.
    # Supported: "cosine" (default), "l2", "ip" (inner product).
    distance: str = "cosine"
    schema: str = "public"
    nodes_table: str = "gke_nodes"
    edges_table: str = "gke_edges"
    documents_table: str = "gke_documents"
    domains_table: str = "gke_domains"
    edge_endpoints_table: str = "gke_edge_endpoints"
    edge_refs_table: str = "gke_edge_refs"
    node_docs_table: str = "gke_node_docs"
    node_refs_table: str = "gke_node_refs"


@dataclass(frozen=True)
class CollectionSpec:
    """Configuration for a collection-like table.

    We keep the public backend API explicit (node_add/edge_add/etc) but route
    the common behavior through a small facade to avoid duplication.
    """

    vector: bool
    ignore_embeddings: bool = False


class PgCollectionFacade:
    """Small, precise adapter that implements the repeated Chroma-shaped verbs."""

    def __init__(
        self, backend: "PgVectorBackend", table: sa.Table, spec: CollectionSpec
    ):
        self._b = backend
        self._t = table
        self._s = spec

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if self._s.ignore_embeddings:
            embeddings = None
        self._b._upsert(
            self._t,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        include = include or ["documents", "metadatas"]
        return self._b._get_flat(
            self._t, ids=ids, where=where, include=include, limit=limit
        )

    def delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._b._delete(self._t, ids=ids, where=where)

    def query(
        self,
        *,
        query_embeddings: Optional[Sequence[Sequence[float]]] = None,
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self._s.vector:
            include = include or ["documents", "metadatas", "distances"]
            if query_embeddings is None:
                raise ValueError("query_embeddings is required for vector collections")
            return self._b._query_vector(
                self._t,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include,
            )

        include = include or ["documents", "metadatas"]
        return self._b._query_nonvector(
            self._t, where=where, n_results=n_results, include=include
        )

    def update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Chroma-shaped update.

        Parity with Chroma: update existing ids' documents/metadatas.
        For vector collections we also allow updating embeddings (atomic in Postgres).
        For non-vector collections embeddings are ignored.
        """
        if self._s.ignore_embeddings:
            embeddings = None
        self._b._update_doc_meta_embedding_merge(
            self._t,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


# ----------------------------
# where DSL → SQLAlchemy
# ----------------------------

_NUMERIC_KEYS_DEFAULT: Set[str] = {"seq"}


def _json_text(metadata_col: sa.ColumnElement, key: str) -> sa.ColumnElement:
    # (metadata ->> 'key') yields text
    return metadata_col.op("->>")(key)


def _json_typed(
    metadata_col: sa.ColumnElement,
    key: str,
    rhs: Any,
    *,
    numeric_keys: Set[str],
) -> sa.ColumnElement:
    """
    Return a SQLAlchemy expression for metadata[key] with an appropriate type.

    Postgres JSONB `->>` returns TEXT, which breaks numeric/boolean comparisons.
    We instead use JSONB index access + typed accessors for comparisons.
    """
    try:
        node = metadata_col[key]  # type: ignore[index]
    except Exception:
        node = None

    # Force numeric by key (override)
    if key in numeric_keys:
        if node is not None:
            return node.as_integer()
        return sa.cast(_json_text(metadata_col, key), sa.BigInteger)

    # Infer from rhs
    if isinstance(rhs, bool):
        if node is not None:
            return node.as_boolean()
        return sa.cast(_json_text(metadata_col, key), sa.Boolean)

    # bool is a subclass of int; handled above
    if isinstance(rhs, int):
        if node is not None:
            return node.as_integer()
        return sa.cast(_json_text(metadata_col, key), sa.BigInteger)

    if isinstance(rhs, float):
        if node is not None:
            return node.as_float()
        return sa.cast(_json_text(metadata_col, key), sa.Float)

    # Default: treat as text
    if node is not None:
        return node.astext
    return _json_text(metadata_col, key)


def where_jsonb(
    metadata_col: sa.ColumnElement,
    where: Json,
    *,
    numeric_keys: Optional[Set[str]] = None,
) -> sa.ColumnElement:
    """Translate a Chroma-like `where` dict into a SQLAlchemy boolean expression over JSONB.

    Supports:
      - {"k": v}
      - {"k": {"$in": [...]}}
      - {"k": {"$gt/$gte/$lt/$lte/$ne": v}}
      - {"$and": [..]}, {"$or": [..]}

    Important:
      Postgres JSONB `->>` returns TEXT. For numeric/boolean comparisons we must cast.
    """

    numeric_keys_set = set(numeric_keys or _NUMERIC_KEYS_DEFAULT)
    if not where:
        return sa.true()

    if "$and" in where:
        parts = where.get("$and") or []
        if not isinstance(parts, list):
            raise TypeError("$and must be a list")
        return (
            sa.and_(
                *[
                    where_jsonb(metadata_col, p, numeric_keys=numeric_keys_set)
                    for p in parts
                ]
            )
            if parts
            else sa.true()
        )

    if "$or" in where:
        parts = where.get("$or") or []
        if not isinstance(parts, list):
            raise TypeError("$or must be a list")
        return (
            sa.or_(
                *[
                    where_jsonb(metadata_col, p, numeric_keys=numeric_keys_set)
                    for p in parts
                ]
            )
            if parts
            else sa.true()
        )

    clauses: List[sa.ColumnElement] = []
    for k, v in where.items():
        if k in ("$and", "$or"):
            continue

        if isinstance(v, dict):
            if "$in" in v:
                vals = v["$in"]
                if not isinstance(vals, list):
                    raise TypeError(f"$in for {k} must be a list")

                sample = next((x for x in vals if x is not None), None)
                if sample is None:
                    clauses.append(sa.false())
                    continue

                lhs = _json_typed(
                    metadata_col, k, sample, numeric_keys=numeric_keys_set
                )

                if isinstance(sample, bool):
                    rhs_list = [bool(x) for x in vals]
                elif isinstance(sample, int) and not isinstance(sample, bool):
                    rhs_list = [int(x) for x in vals]
                elif isinstance(sample, float):
                    rhs_list = [float(x) for x in vals]
                else:
                    rhs_list = [str(x) for x in vals]

                clauses.append(lhs.in_(rhs_list))
                continue

            for op, rhs in v.items():
                lhs = _json_typed(metadata_col, k, rhs, numeric_keys=numeric_keys_set)

                if op == "$gt":
                    clauses.append(lhs > rhs)
                elif op == "$gte":
                    clauses.append(lhs >= rhs)
                elif op == "$lt":
                    clauses.append(lhs < rhs)
                elif op == "$lte":
                    clauses.append(lhs <= rhs)
                elif op == "$ne":
                    clauses.append(lhs != rhs)
                else:
                    raise NotImplementedError(
                        f"Unsupported where operator: {op} (key={k})"
                    )
        else:
            lhs = _json_typed(metadata_col, k, v, numeric_keys=numeric_keys_set)
            clauses.append(lhs == v)

    return sa.and_(*clauses) if clauses else sa.true()


# ----------------------------
# Backend
# ----------------------------


class PgVectorBackend:
    """pgvector backend implementing a Chroma-shaped interface for engine usage."""

    @staticmethod
    def _normalize_distance(distance: str) -> str:
        d = (distance or "").strip().lower()
        if d in ("cos", "cosine"):
            return "cosine"
        if d in ("l2", "euclid", "euclidean"):
            return "l2"
        if d in ("ip", "inner", "inner_product", "innerproduct"):
            return "ip"
        raise ValueError(
            f"Unsupported distance metric: {distance!r}. Use one of: cosine, l2, ip"
        )

    def _distance_operator(self) -> str:
        # pgvector operators:
        #   <->  L2 distance
        #   <#>  negative inner product (lower is closer)
        #   <=>  cosine distance
        return {"cosine": "<=>", "l2": "<->", "ip": "<#>"}[self.distance]

    def _hnsw_ops_class(self) -> str:
        # ops class depends on metric
        return {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }[self.distance]

    def __init__(
        self,
        *,
        engine: sa.Engine,
        embedding_dim: int,
        distance: str = "cosine",
        schema: str = "public",
        nodes_table: str = "gke_nodes",
        edges_table: str = "gke_edges",
        documents_table: str = "gke_documents",
        domains_table: str = "gke_domains",
        edge_endpoints_table: str = "gke_edge_endpoints",
        edge_refs_table: str = "gke_edge_refs",
        node_docs_table: str = "gke_node_docs",
        node_refs_table: str = "gke_node_refs",
        numeric_keys: Optional[Set[str]] = None,
    ):
        if Vector is None:  # pragma: no cover
            raise RuntimeError(
                "pgvector is not installed. Install with `pip install pgvector` to use PgVectorBackend."
            ) from _pgvector_import_error

        self.engine = engine
        self.embedding_dim = int(embedding_dim)
        self.distance = str(self._normalize_distance(distance)).lower()
        self.schema = schema
        self.numeric_keys = numeric_keys or set(_NUMERIC_KEYS_DEFAULT)

        if self.distance not in {"cosine", "l2", "ip"}:
            raise ValueError("distance must be one of: 'cosine', 'l2', 'ip'")

        md = sa.MetaData(schema=self.schema)

        # Vector tables
        self.nodes = sa.Table(
            nodes_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
        )

        self.edges = sa.Table(
            edges_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
        )

        self.documents = sa.Table(
            documents_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
        )

        self.domains = sa.Table(
            domains_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
        )

        # Non-vector collections
        self.edge_endpoints = sa.Table(
            edge_endpoints_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
            sa.Index(
                f"ix_{edge_endpoints_table}_endpoint",
                sa.text("(metadata->>'endpoint_node_id')"),
            ),
            sa.Index(
                f"ix_{edge_endpoints_table}_edge", sa.text("(metadata->>'edge_id')")
            ),
        )

        self.edge_refs = sa.Table(
            edge_refs_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
            sa.Index(f"ix_{edge_refs_table}_doc", sa.text("(metadata->>'doc_id')")),
            sa.Index(f"ix_{edge_refs_table}_edge", sa.text("(metadata->>'edge_id')")),
        )

        self.node_docs = sa.Table(
            node_docs_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
            sa.Index(f"ix_{node_docs_table}_node", sa.text("(metadata->>'node_id')")),
            sa.Index(f"ix_{node_docs_table}_doc", sa.text("(metadata->>'doc_id')")),
        )

        self.node_refs = sa.Table(
            node_refs_table,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column(
                "metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
            sa.Index(f"ix_{node_refs_table}_doc", sa.text("(metadata->>'doc_id')")),
            sa.Index(f"ix_{node_refs_table}_node", sa.text("(metadata->>'node_id')")),
        )

        self._init_facades()

        self._md = md
        self.ensure_schema()

    # ----------------------------
    # DDL / bootstrap
    # ----------------------------

    def ensure_schema(self) -> None:
        """Dev convenience: create extension/schema/tables if missing."""
        with self.engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            if self.schema and self.schema != "public":
                conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
            self._md.create_all(conn)

            # Optional-but-useful vector indexes. We default to HNSW because it's
            # generally strong out of the box and doesn't require ANALYZE/training.
            #
            # NOTE: older pgvector versions may not support HNSW; if so, users can
            # drop these statements or switch to ivfflat.
            ops = self._hnsw_ops_class()
            for tbl, name in (
                (self.nodes.name, "nodes"),
                (self.edges.name, "edges"),
                (self.documents.name, "documents"),
                (self.domains.name, "domains"),
            ):
                idx = f"idx_{name}_embedding_hnsw"
                conn.execute(
                    sa.text(
                        f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{self.schema}"."{tbl}" '
                        f"USING hnsw (embedding {ops})"
                    )
                )

    # ----------------------------
    # Shared helpers
    # ----------------------------

    @contextmanager
    def _conn(self):
        """Yield an active SQLAlchemy connection.

        If the runtime/engine opened a PostgresUnitOfWork transaction, backend
        methods will join it. Otherwise we open an implicit transaction here.
        """
        active = get_active_conn()
        if active is not None:
            yield active
            return
        with self.engine.begin() as conn:
            yield conn

    def _get_flat(
        self,
        table: sa.Table,
        *,
        ids: Optional[Sequence[str]],
        where: Optional[Json],
        include: List[str],
        limit: int,
    ) -> Dict[str, Any]:
        has_embedding = "embedding" in table.c
        cols = [table.c.id, table.c.document, table.c.metadata]
        if has_embedding:
            cols.append(table.c.embedding)

        q = sa.select(*cols).limit(int(limit))
        if ids is not None:
            q = q.where(table.c.id.in_(list(ids)))
        if where:
            q = q.where(
                where_jsonb(table.c.metadata, where, numeric_keys=self.numeric_keys)
            )

        with self._conn() as conn:
            rows = conn.execute(q).fetchall()

        out: Dict[str, Any] = {"ids": [r.id for r in rows]}
        if "documents" in include:
            out["documents"] = [r.document for r in rows]
        if "metadatas" in include:
            out["metadatas"] = [dict(r.metadata or {}) for r in rows]
        if "embeddings" in include and has_embedding:
            out["embeddings"] = [
                normalize_embedding_vector(r.embedding) for r in rows
            ]
        return out

    def _delete(
        self, table: sa.Table, *, ids: Optional[Sequence[str]], where: Optional[Json]
    ) -> None:
        stmt = sa.delete(table)
        if ids is not None:
            stmt = stmt.where(table.c.id.in_(list(ids)))
        if where:
            stmt = stmt.where(
                where_jsonb(table.c.metadata, where, numeric_keys=self.numeric_keys)
            )
        with self._conn() as conn:
            conn.execute(stmt)

    def _upsert(
        self,
        table: sa.Table,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError("embeddings length must match ids length")

        if embeddings is not None:
            for i, e in enumerate(embeddings):
                if e is None:
                    continue
                e = normalize_embedding_vector(e, allow_none=False) or []
                if len(e) != self.embedding_dim:
                    raise ValueError(
                        f"embedding dim mismatch at index {i}: got {len(e)}, expected {self.embedding_dim}"
                    )

        rows: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            row: Dict[str, Any] = {
                "id": _id,
                "document": documents[i] if i < len(documents) else None,
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            if embeddings is not None and "embedding" in table.c:
                row["embedding"] = normalize_embedding_vector(
                    embeddings[i], allow_none=False
                )
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

        with self._conn() as conn:
            conn.execute(stmt)

    def _query_vector(
        self,
        table: sa.Table,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        where: Optional[Json],
        include: List[str],
    ) -> Dict[str, Any]:
        query_embeddings = cast(
            Sequence[Sequence[float]],
            normalize_embedding_rows(query_embeddings, allow_empty=False),
        )
        if not query_embeddings:
            raise ValueError("query_embeddings is required")
        if "embedding" not in table.c:
            raise TypeError("vector query requested for a table without embedding")

        ids_out: List[List[str]] = []
        docs_out: List[List[Optional[str]]] = []
        metas_out: List[List[Json]] = []
        dists_out: List[List[float]] = []

        # Operator mapping per pgvector docs:
        #   <->  : L2 distance
        #   <#>  : negative inner product
        #   <=>  : cosine distance
        op_map = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
        op = op_map[self.distance]

        # Bind the RHS as a real pgvector type to avoid adapter / text-cast issues.
        qv_param = sa.bindparam("qv", type_=Vector(self.embedding_dim))

        # IMPORTANT: cast to Float so the pgvector result processor doesn't try
        # to parse this column as a Vector.
        distance_expr = sa.cast(table.c.embedding.op(op)(qv_param), sa.Float).label(
            "distance"
        )
        want_embeddings = "embeddings" in include
        if want_embeddings:
            embs_out: List[List[float]] = []
        with self._conn() as conn:
            for qv in query_embeddings:
                cols = [table.c.id, table.c.document, table.c.metadata, distance_expr]
                if want_embeddings:
                    cols.append(table.c.embedding)
                q = sa.select(*cols).where(table.c.embedding.is_not(None))

                if where:
                    q = q.where(
                        where_jsonb(
                            table.c.metadata, where, numeric_keys=self.numeric_keys
                        )
                    )

                q = q.order_by(distance_expr.asc(), table.c.id.asc()).limit(
                    int(n_results)
                )
                rows = conn.execute(q, {"qv": list(qv)}).fetchall()

                ids_out.append([r.id for r in rows])
                docs_out.append([r.document for r in rows])
                metas_out.append([dict(r.metadata or {}) for r in rows])
                dists_out.append([float(r.distance) for r in rows])
                if want_embeddings:
                    embs_out.append(
                        [
                            normalize_embedding_vector(r.embedding, allow_none=False) or []
                            for r in rows
                        ]
                    )

        out: Dict[str, Any] = {"ids": ids_out}
        if "documents" in include:
            out["documents"] = docs_out
        if "metadatas" in include:
            out["metadatas"] = metas_out
        if "distances" in include:
            out["distances"] = dists_out
        if want_embeddings:
            out["embeddings"] = embs_out
        return out

    def _update_doc_meta_embedding_merge(
        self,
        table: sa.Table,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Update document and/or merge metadata patch and/or update embedding for each id.

        Parity requirement: engine-level `update()` and `upsert()` share a compatible
        shape. Postgres can apply document+metadata+embedding atomically within a
        transaction; if a UnitOfWork is active, we join it.

        Semantics:
        * `documents` replaces the document value (can be None).
        * `metadatas` is merged (JSONB || patch).
        * `embeddings` replaces the embedding value (vector tables only). For
          non-vector tables, it is ignored.
        """

        if documents is None and metadatas is None and embeddings is None:
            return

        if documents is not None and len(documents) != len(ids):
            raise ValueError("documents length must match ids length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids length")
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError("embeddings length must match ids length")

        # Validate embedding dimensions early.
        if embeddings is not None and "embedding" in table.c:
            for i, e in enumerate(embeddings):
                if e is None:
                    continue
                e = normalize_embedding_vector(e, allow_none=False) or []
                if len(e) != self.embedding_dim:
                    raise ValueError(
                        f"embedding dim mismatch at index {i}: got {len(e)}, expected {self.embedding_dim}"
                    )

        patch_text = sa.bindparam("patch_text", type_=sa.Text)
        merged_expr = table.c.metadata
        if metadatas is not None:
            merged_expr = table.c.metadata.op("||")(sa.cast(patch_text, JSONB))

        with self._conn() as conn:
            for i, _id in enumerate(ids):
                values: Dict[str, Any] = {"updated_at": sa.func.now()}
                params: Dict[str, Any] = {}

                if documents is not None:
                    values["document"] = documents[i]

                if metadatas is not None:
                    values["metadata"] = merged_expr
                    params["patch_text"] = json.dumps(metadatas[i])

                if embeddings is not None and "embedding" in table.c:
                    # If caller passes None, we clear the embedding.
                    e = embeddings[i]
                    values["embedding"] = (
                        normalize_embedding_vector(e, allow_none=False)
                        if e is not None
                        else None
                    )

                stmt = sa.update(table).where(table.c.id == _id).values(**values)
                conn.execute(stmt, params)

    def _update_doc_and_metadata_merge(
        self,
        table: sa.Table,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
    ) -> None:
        """Backward compatible wrapper: update doc + merge metadata (no embedding)."""
        self._update_doc_meta_embedding_merge(
            table, ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def _update_metadata_merge(
        self, table: sa.Table, *, ids: Sequence[str], metadatas: Sequence[Json]
    ) -> None:
        """Backward-compatible metadata-only merge."""
        self._update_doc_and_metadata_merge(
            table, ids=ids, documents=None, metadatas=metadatas
        )

    def _query_nonvector(
        self,
        table: sa.Table,
        *,
        where: Optional[Json],
        n_results: int,
        include: List[str],
    ) -> Dict[str, Any]:
        """Best-effort query for non-vector tables.

        Chroma's `.query()` is fundamentally vector-similarity driven.
        For materialized/index tables we treat `query()` as a filtered read and
        return a Chroma-shaped nested payload.
        """

        flat = self._get_flat(
            table,
            ids=None,
            where=where,
            include=["documents", "metadatas"],
            limit=int(n_results),
        )
        out: Dict[str, Any] = {"ids": [flat.get("ids", [])]}
        if "documents" in include:
            out["documents"] = [flat.get("documents", [])]
        if "metadatas" in include:
            out["metadatas"] = [flat.get("metadatas", [])]
        if "distances" in include:
            out["distances"] = [[0.0 for _ in out["ids"][0]]]
        return out

    # ----------------------------
    # Collections (facades)
    # ----------------------------

    def _init_facades(self) -> None:
        # Vector collections
        self._nodes_c = PgCollectionFacade(
            self, self.nodes, CollectionSpec(vector=True)
        )
        self._edges_c = PgCollectionFacade(
            self, self.edges, CollectionSpec(vector=True)
        )
        self._documents_c = PgCollectionFacade(
            self, self.documents, CollectionSpec(vector=True)
        )
        self._domains_c = PgCollectionFacade(
            self, self.domains, CollectionSpec(vector=True)
        )

        # Non-vector collections (materialized/index tables)
        nv = CollectionSpec(vector=False, ignore_embeddings=True)
        self._edge_endpoints_c = PgCollectionFacade(self, self.edge_endpoints, nv)
        self._edge_refs_c = PgCollectionFacade(self, self.edge_refs, nv)
        self._node_docs_c = PgCollectionFacade(self, self.node_docs, nv)
        self._node_refs_c = PgCollectionFacade(self, self.node_refs, nv)

    # ----------------------------
    # Nodes
    # ----------------------------

    def node_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._nodes_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def node_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._nodes_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def node_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._nodes_c.get(ids=ids, where=where, include=include, limit=limit)

    def node_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._nodes_c.delete(ids=ids, where=where)

    def node_query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._nodes_c.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
        )

    def node_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._nodes_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    # ----------------------------
    # Edges
    # ----------------------------

    def edge_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._edges_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def edge_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._edges_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def edge_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._edges_c.get(ids=ids, where=where, include=include, limit=limit)

    def edge_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._edges_c.delete(ids=ids, where=where)

    def edge_query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._edges_c.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
        )

    def edge_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._edges_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    # ----------------------------
    # Documents
    # ----------------------------

    def document_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._documents_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def document_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._documents_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def document_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._documents_c.get(ids=ids, where=where, include=include, limit=limit)

    def document_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._documents_c.delete(ids=ids, where=where)

    def document_query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._documents_c.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
        )

    def document_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._documents_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    # ----------------------------
    # Domains
    # ----------------------------

    def domain_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._domains_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def domain_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._domains_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def domain_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._domains_c.get(ids=ids, where=where, include=include, limit=limit)

    def domain_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._domains_c.delete(ids=ids, where=where)

    def domain_query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._domains_c.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include,
        )

    def domain_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._domains_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    # ----------------------------
    # Edge endpoints
    # ----------------------------

    def edge_endpoints_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._edge_endpoints_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def edge_endpoints_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._edge_endpoints_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def edge_endpoints_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._edge_endpoints_c.get(
            ids=ids, where=where, include=include, limit=limit
        )

    def edge_endpoints_query(
        self,
        *,
        query_embeddings: Any = None,
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._edge_endpoints_c.query(
            query_embeddings=None, n_results=n_results, where=where, include=include
        )

    def edge_endpoints_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._edge_endpoints_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def edge_endpoints_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._edge_endpoints_c.delete(ids=ids, where=where)

    # ----------------------------
    # Edge refs
    # ----------------------------

    def edge_refs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._edge_refs_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def edge_refs_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._edge_refs_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def edge_refs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._edge_refs_c.get(ids=ids, where=where, include=include, limit=limit)

    def edge_refs_query(
        self,
        *,
        query_embeddings: Any = None,
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._edge_refs_c.query(
            query_embeddings=None, n_results=n_results, where=where, include=include
        )

    def edge_refs_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._edge_refs_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def edge_refs_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._edge_refs_c.delete(ids=ids, where=where)

    # ----------------------------
    # Node docs
    # ----------------------------

    def node_docs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._node_docs_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def node_docs_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._node_docs_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def node_docs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._node_docs_c.get(ids=ids, where=where, include=include, limit=limit)

    def node_docs_query(
        self,
        *,
        query_embeddings: Any = None,
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._node_docs_c.query(
            query_embeddings=None, n_results=n_results, where=where, include=include
        )

    def node_docs_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._node_docs_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def node_docs_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._node_docs_c.delete(ids=ids, where=where)

    # ----------------------------
    # Node refs
    # ----------------------------

    def node_refs_add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._node_refs_c.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def node_refs_upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Any = None,
    ) -> None:
        self._node_refs_c.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=None
        )

    def node_refs_get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return self._node_refs_c.get(ids=ids, where=where, include=include, limit=limit)

    def node_refs_query(
        self,
        *,
        query_embeddings: Any = None,
        n_results: int = 10,
        where: Optional[Json] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._node_refs_c.query(
            query_embeddings=None, n_results=n_results, where=where, include=include
        )

    def node_refs_update(
        self,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._node_refs_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def node_refs_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        self._node_refs_c.delete(ids=ids, where=where)


def build_postgres_backend(
    cfg: PgVectorConfig,
) -> Tuple[PgVectorBackend, PostgresUnitOfWork]:
    """Convenience helper for engine wiring."""

    engine = sa.create_engine(cfg.dsn, future=True)
    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edges_table=cfg.edges_table,
        documents_table=cfg.documents_table,
        domains_table=cfg.domains_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
        edge_refs_table=cfg.edge_refs_table,
        node_docs_table=cfg.node_docs_table,
        node_refs_table=cfg.node_refs_table,
    )
    backend.ensure_schema()
    return backend, PostgresUnitOfWork(engine=engine)
