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

import asyncio
import hashlib
import inspect
import os
import re
import threading
import sys
import time
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
import contextvars
import json
from typing import Any, AsyncIterator, cast, Dict, List, Optional, Sequence, Set, Tuple

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.dialects import postgresql as psql

from ..utils.embedding_vectors import normalize_embedding_rows, normalize_embedding_vector
from .async_compat import run_awaitable_blocking
from .embedding_profile import EmbeddingProfileError, EmbeddingStorageState

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


_VECTOR_TYPE_RE = re.compile(r"^vector\((?P<dimension>\d+)\)$")


@dataclass(frozen=True)
class PgVectorColumnDimension:
    """Observed physical type for one pgvector embedding column."""

    table_name: str
    column_name: str
    type_name: str
    dimension: int | None


class PgVectorSchemaMismatchError(EmbeddingProfileError):
    """Raised before writes when an existing pgvector column has the wrong shape."""

    def __init__(
        self,
        *,
        schema: str,
        expected_dimension: int,
        mismatches: Sequence[PgVectorColumnDimension],
    ) -> None:
        self.schema = schema
        self.expected_dimension = expected_dimension
        self.mismatches = tuple(mismatches)
        observed = "; ".join(
            f'{schema}.{item.table_name}.{item.column_name} is {item.type_name}'
            for item in self.mismatches
        )
        super().__init__(
            "PostgreSQL pgvector schema mismatch: "
            f"configured embedding dimension is {expected_dimension}, but {observed}. "
            "No data was written. Do not alter the live vector column in place: "
            "existing embeddings and HNSW indexes must be rebuilt. Stop writers, "
            "create an isolated target schema or database configured for the new "
            "dimension, replay canonical state, re-embed, validate, then cut over."
        )


def _parse_vector_dimension(type_name: object) -> int | None:
    """Extract a dimension from PostgreSQL's stable ``format_type`` output."""

    match = _VECTOR_TYPE_RE.fullmatch(str(type_name or "").strip())
    return int(match.group("dimension")) if match is not None else None


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


_pg_uow_conn: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "gke_pg_uow_conn", default=None
)


@contextmanager
def _set_active_conn(conn: Any):
    token = _pg_uow_conn.set(conn)
    try:
        yield
    finally:
        _pg_uow_conn.reset(token)


def get_active_conn() -> Any | None:
    return _pg_uow_conn.get()


def _install_connection_observability(engine: sa.Engine | AsyncEngine, *, component: str) -> None:
    """Tag checked-out PostgreSQL connections with their Python owner."""

    sync_engine = engine.sync_engine if isinstance(engine, AsyncEngine) else engine

    @sa.event.listens_for(sync_engine, "checkout")
    def _tag_connection(dbapi_connection, connection_record, connection_proxy) -> None:
        del connection_proxy
        label = (
            f"kogwistar:{component}:p{os.getpid()}:t{threading.get_ident()}"
        )[:63]
        connection_record.info["kogwistar_application_name"] = label
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET application_name = %s", (label,))
        finally:
            cursor.close()


def _run_coro_sync(coro):
    if sys.platform == "win32":
        runner = asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return runner.run(coro)
            box: dict[str, Any] = {}

            def _worker() -> None:
                try:
                    box["result"] = _run_coro_sync(coro)
                except BaseException as exc:  # pragma: no cover - thread ferry
                    box["error"] = exc

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            thread.join()
            if "error" in box:
                raise box["error"]
            return box.get("result")
        finally:
            runner.close()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["result"] = _run_coro_sync(coro)
        except BaseException as exc:  # pragma: no cover - thread ferry
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


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


class AsyncPostgresUnitOfWork:
    """Async transaction wrapper for async SQLAlchemy Postgres engines."""

    def __init__(self, *, engine: AsyncEngine):
        self._engine = engine

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        existing = get_active_conn()
        if existing is not None:
            yield
            return

        async with self._engine.begin() as conn:
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
    stage1_table: str = "gke_stage1_projections"
    documents_table: str = "gke_documents"
    domains_table: str = "gke_domains"
    edge_endpoints_table: str = "gke_edge_endpoints"
    edge_refs_table: str = "gke_edge_refs"
    node_docs_table: str = "gke_node_docs"
    node_refs_table: str = "gke_node_refs"
    # Database-side safeguards apply only while SQL is executing. They do not
    # limit time spent inside an LLM provider call.
    statement_timeout_ms: int | None = 300_000
    idle_transaction_timeout_ms: int | None = 300_000
    pool_timeout_s: float = 10.0
    application_name: str = "kogwistar"


def postgres_connect_args(cfg: PgVectorConfig) -> dict[str, str]:
    """Build safe PostgreSQL connection settings without imposing lock_timeout.

    ``statement_timeout`` and ``idle_in_transaction_session_timeout`` protect
    database work and abandoned transactions. LLM calls happen outside these
    SQL statements, so a slow provider response is not terminated by them.
    Lock timeout is intentionally omitted: callers may need to wait for a
    short-lived writer and should rely on the operation/runtime watchdog when
    deciding whether to abort.
    """
    options: list[str] = []
    if cfg.statement_timeout_ms is not None:
        options.extend(["-c", f"statement_timeout={int(cfg.statement_timeout_ms)}"])
    if cfg.idle_transaction_timeout_ms is not None:
        options.extend(
            ["-c", f"idle_in_transaction_session_timeout={int(cfg.idle_transaction_timeout_ms)}"]
        )
    args: dict[str, str] = {"application_name": cfg.application_name}
    if options:
        args["options"] = " ".join(options)
    return args


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

    def _call_async(self, fn):
        return fn()

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
        if self._b._is_async_engine:
            return self._call_async(
                lambda: self._b._upsert_async(
                    self._t,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            )
        return _awaitable_result(self._b._upsert(
            self._t,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        ))

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Json],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        return self.add(
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
        if self._b._is_async_engine:
            return self._call_async(
                lambda: self._b._get_flat_async(
                    self._t, ids=ids, where=where, include=include, limit=limit
                )
            )
        return _awaitable_result(self._b._get_flat(
            self._t, ids=ids, where=where, include=include, limit=limit
        ))

    def delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        if self._b._is_async_engine:
            return self._call_async(
                lambda: self._b._delete_async(self._t, ids=ids, where=where)
            )
        return _awaitable_result(self._b._delete(self._t, ids=ids, where=where))

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
            if self._b._is_async_engine:
                return self._call_async(
                    lambda: self._b._query_vector_async(
                        self._t,
                        query_embeddings=query_embeddings,
                        n_results=n_results,
                        where=where,
                        include=include,
                    )
                )
            return _awaitable_result(self._b._query_vector(
                self._t,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include,
            ))

        include = include or ["documents", "metadatas"]
        if self._b._is_async_engine:
            return self._call_async(
                lambda: self._b._query_nonvector_async(
                    self._t, where=where, n_results=n_results, include=include
                )
            )
        return _awaitable_result(self._b._query_nonvector(
            self._t, where=where, n_results=n_results, include=include
        ))

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
        if self._b._is_async_engine:
            return self._call_async(
                lambda: self._b._update_doc_meta_embedding_merge_async(
                    self._t,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            )
        return _awaitable_result(self._b._update_doc_meta_embedding_merge(
            self._t,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        ))


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
        engine: sa.Engine | AsyncEngine,
        embedding_dim: int,
        distance: str = "cosine",
        schema: str = "public",
        nodes_table: str = "gke_nodes",
        edges_table: str = "gke_edges",
        stage1_table: str = "gke_stage1_projections",
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
        self._is_async_engine = isinstance(engine, AsyncEngine)
        self.embedding_dim = int(embedding_dim)
        self.distance = str(self._normalize_distance(distance)).lower()
        self.schema = schema
        self.stage1_table_name = stage1_table
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

        # High-churn, non-semantic staging projection.  It is deliberately
        # separate from named projections and vector serving tables.
        self.stage1_projections = sa.Table(
            stage1_table,
            md,
            sa.Column("namespace", sa.String, nullable=False, server_default="default"),
            sa.Column("entity_kind", sa.String, nullable=False),
            sa.Column("entity_id", sa.String, nullable=False),
            sa.Column("document", sa.Text, nullable=False),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("source_fingerprint", sa.String, nullable=False),
            sa.Column("revision", sa.BigInteger, nullable=False, server_default="0"),
            sa.Column("materialization_status", sa.String, nullable=False, server_default="'pending'"),
            sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
            sa.PrimaryKeyConstraint("namespace", "entity_kind", "entity_id"),
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
        if self._is_async_engine:
            self._install_async_passthroughs()
        self.ensure_schema()

    def embedding_storage_scope(self) -> str:
        """Return a stable identity for the shared PostgreSQL vector bundle."""

        url = self.engine.url
        # Credentials, dialect drivers, and query parameters describe an
        # access path, not the physical vector bundle.  Excluding them keeps
        # profile compatibility stable when operators rotate credentials or
        # switch between sync and async SQLAlchemy drivers.
        host = str(getattr(url, "host", None) or "localhost").lower()
        port = int(getattr(url, "port", None) or 5432)
        database = str(getattr(url, "database", None) or "").strip()
        tables = ":".join(
            table.name
            for table in (self.nodes, self.edges, self.documents, self.domains)
        )
        value = f"host={host}|port={port}|database={database}|schema={self.schema}|tables={tables}"
        return "pgvector:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

    def embedding_storage_scope_aliases(self) -> tuple[str, ...]:
        """Return the pre-profile-guard URL-derived scope for migration."""

        rendered = self.engine.url.render_as_string(hide_password=True)
        tables = ":".join(
            table.name
            for table in (self.nodes, self.edges, self.documents, self.domains)
        )
        value = f"{rendered}|schema={self.schema}|tables={tables}"
        legacy = "pgvector:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
        current = self.embedding_storage_scope()
        return (legacy,) if legacy != current else ()

    def inspect_embedding_storage(self) -> EmbeddingStorageState:
        """Report whether the physical vector tables already contain rows."""

        if self._is_async_engine:
            return run_awaitable_blocking(self.inspect_embedding_storage_async())
        tables = (self.nodes, self.edges, self.documents, self.domains)
        with self.engine.connect() as conn:
            counts = tuple(
                (
                    table.name,
                    int(conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()),
                )
                for table in tables
            )
        return EmbeddingStorageState(
            backend_kind="pgvector",
            storage_scope=self.embedding_storage_scope(),
            persistent=True,
            vector_count=sum(count for _name, count in counts),
            details=tuple(f"{name}={count}" for name, count in counts),
        )

    async def inspect_embedding_storage_async(self) -> EmbeddingStorageState:
        """Async counterpart used by async engine bootstrap paths."""

        tables = (self.nodes, self.edges, self.documents, self.domains)
        async with self.engine.connect() as conn:
            counts_list: list[tuple[str, int]] = []
            for table in tables:
                result = await conn.execute(sa.select(sa.func.count()).select_from(table))
                counts_list.append((table.name, int(result.scalar_one())))
            counts = tuple(counts_list)
        return EmbeddingStorageState(
            backend_kind="pgvector",
            storage_scope=self.embedding_storage_scope(),
            persistent=True,
            vector_count=sum(count for _name, count in counts),
            details=tuple(f"{name}={count}" for name, count in counts),
        )
    # ----------------------------
    # DDL / bootstrap
    # ----------------------------

    def ensure_schema(self) -> None:
        """Dev convenience: create extension/schema/tables if missing."""
        if self._is_async_engine:
            _run_coro_sync(self._ensure_schema_async())
            return
        with self.engine.begin() as conn:
            self._ensure_schema_sync(conn)

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

    @asynccontextmanager
    async def _async_conn(self):
        """Yield active async connection without falling back to sync bridge."""
        active = get_active_conn()
        if isinstance(active, AsyncConnection):
            yield active
            return
        async with self.engine.begin() as conn:
            yield conn

    # ----------------------------
    # ADR-018 PostgreSQL Stage-1 projection
    # ----------------------------

    def stage1_projection_upsert(
        self,
        *,
        namespace: str,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata: dict[str, Any],
        source_fingerprint: str,
        revision: int = 0,
    ) -> None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        table = self.stage1_projections
        stmt = psql.insert(table).values(
            namespace=str(namespace),
            entity_kind=entity_kind,
            entity_id=str(entity_id),
            document=str(document),
            metadata=dict(metadata or {}),
            source_fingerprint=str(source_fingerprint or ""),
            revision=int(revision),
            materialization_status="pending",
            updated_at_ms=int(time.time() * 1000),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.namespace, table.c.entity_kind, table.c.entity_id],
            set_={
                "document": stmt.excluded.document,
                "metadata": stmt.excluded.metadata,
                "source_fingerprint": stmt.excluded.source_fingerprint,
                "revision": stmt.excluded.revision,
                "materialization_status": "pending",
                "updated_at_ms": stmt.excluded.updated_at_ms,
            },
        )
        with self._conn() as conn:
            conn.execute(stmt)

    def stage1_projection_get(
        self, *, namespace: str, entity_kind: str, entity_id: str
    ) -> dict[str, Any] | None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        with self._conn() as conn:
            row = conn.execute(
                sa.select(self.stage1_projections).where(
                    self.stage1_projections.c.namespace == str(namespace),
                    self.stage1_projections.c.entity_kind == entity_kind,
                    self.stage1_projections.c.entity_id == str(entity_id),
                )
            ).mappings().first()
        return dict(row) if row is not None else None

    def stage1_projection_query(
        self,
        *,
        namespace: str,
        entity_kind: str,
        ids: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        table = self.stage1_projections
        q = sa.select(table).where(
            table.c.namespace == str(namespace), table.c.entity_kind == entity_kind
        )
        if ids is not None:
            q = q.where(table.c.entity_id.in_([str(item) for item in ids]))
        # Keep Stage-1 query semantics aligned with the existing narrow adapter.
        for key, value in (metadata or {}).items():
            if not isinstance(key, str) or isinstance(value, (dict, list, tuple, set)):
                raise ValueError("PostgreSQL Stage-1 supports flat metadata equality only")
            q = q.where(table.c.metadata[key].astext == str(value))
        q = q.order_by(table.c.updated_at_ms, table.c.entity_id)
        if limit is not None:
            q = q.limit(int(limit))
        with self._conn() as conn:
            rows = conn.execute(q).mappings().all()
        return [dict(row) for row in rows]

    def stage1_projection_delete(
        self, *, namespace: str, entity_kind: str, entity_id: str
    ) -> None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        with self._conn() as conn:
            conn.execute(
                sa.delete(self.stage1_projections).where(
                    self.stage1_projections.c.namespace == str(namespace),
                    self.stage1_projections.c.entity_kind == entity_kind,
                    self.stage1_projections.c.entity_id == str(entity_id),
                )
            )

    async def stage1_projection_upsert_async(
        self,
        *,
        namespace: str,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata: dict[str, Any],
        source_fingerprint: str,
        revision: int = 0,
    ) -> None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        table = self.stage1_projections
        stmt = psql.insert(table).values(
            namespace=str(namespace), entity_kind=entity_kind,
            entity_id=str(entity_id), document=str(document),
            metadata=dict(metadata or {}), source_fingerprint=str(source_fingerprint or ""),
            revision=int(revision), materialization_status="pending",
            updated_at_ms=int(time.time() * 1000),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.namespace, table.c.entity_kind, table.c.entity_id],
            set_={
                "document": stmt.excluded.document,
                "metadata": stmt.excluded.metadata,
                "source_fingerprint": stmt.excluded.source_fingerprint,
                "revision": stmt.excluded.revision,
                "materialization_status": "pending",
                "updated_at_ms": stmt.excluded.updated_at_ms,
            },
        )
        async with self._async_conn() as conn:
            await conn.execute(stmt)

    async def stage1_projection_get_async(
        self, *, namespace: str, entity_kind: str, entity_id: str
    ) -> dict[str, Any] | None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        async with self._async_conn() as conn:
            row = (await conn.execute(
                sa.select(self.stage1_projections).where(
                    self.stage1_projections.c.namespace == str(namespace),
                    self.stage1_projections.c.entity_kind == entity_kind,
                    self.stage1_projections.c.entity_id == str(entity_id),
                )
            )).mappings().first()
        return dict(row) if row is not None else None

    async def stage1_projection_query_async(
        self,
        *,
        namespace: str,
        entity_kind: str,
        ids: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        table = self.stage1_projections
        query = sa.select(table).where(
            table.c.namespace == str(namespace),
            table.c.entity_kind == entity_kind,
        )
        if ids is not None:
            query = query.where(table.c.entity_id.in_([str(item) for item in ids]))
        for key, value in (metadata or {}).items():
            if not isinstance(key, str) or isinstance(value, (dict, list, tuple, set)):
                raise ValueError("PostgreSQL Stage-1 supports flat metadata equality only")
            query = query.where(table.c.metadata[key].astext == str(value))
        query = query.order_by(table.c.updated_at_ms, table.c.entity_id)
        if limit is not None:
            query = query.limit(int(limit))
        async with self._async_conn() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [dict(row) for row in rows]

    async def stage1_projection_delete_async(
        self, *, namespace: str, entity_kind: str, entity_id: str
    ) -> None:
        if entity_kind not in {"node", "edge"}:
            raise ValueError(f"unsupported Stage-1 entity kind: {entity_kind!r}")
        async with self._async_conn() as conn:
            await conn.execute(sa.delete(self.stage1_projections).where(
                self.stage1_projections.c.namespace == str(namespace),
                self.stage1_projections.c.entity_kind == entity_kind,
                self.stage1_projections.c.entity_id == str(entity_id),
            ))

    def _ensure_schema_sync(self, conn: sa.Connection) -> None:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        if self.schema and self.schema != "public":
            conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
        self._md.create_all(conn)
        self._validate_vector_column_dimensions_sync(conn)
        conn.execute(
            sa.text(
                f'CREATE INDEX IF NOT EXISTS "idx_{self.stage1_table_name}_namespace_status" '
                f'ON "{self.schema}"."{self.stage1_table_name}" '
                "(namespace, materialization_status, updated_at_ms)"
            )
        )

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

    def _validate_vector_column_dimensions_sync(self, conn: sa.Connection) -> None:
        """Reject stale ``vector(N)`` columns before any graph write can occur.

        SQLAlchemy's ``create_all`` is intentionally additive. It cannot alter an
        existing pgvector typmod, so reopening a database with a new embedding
        model would otherwise fail later during an opaque provider/graph write.
        ``format_type`` keeps this independent of PostgreSQL's internal typmod
        representation.
        """
        table_names = (
            self.nodes.name,
            self.edges.name,
            self.documents.name,
            self.domains.name,
        )
        rows = conn.execute(
            sa.text(
                """
                SELECT c.relname AS table_name,
                       a.attname AS column_name,
                       format_type(a.atttypid, a.atttypmod) AS type_name
                  FROM pg_catalog.pg_attribute AS a
                  JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
                  JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                 WHERE n.nspname = :schema
                   AND c.relname IN :table_names
                   AND c.relkind IN ('r', 'p')
                   AND a.attname = 'embedding'
                   AND a.attnum > 0
                   AND NOT a.attisdropped
                """
            ).bindparams(sa.bindparam("table_names", expanding=True)),
            {"schema": self.schema, "table_names": list(table_names)},
        ).mappings().all()
        mismatches: list[PgVectorColumnDimension] = []
        for row in rows:
            type_name = str(row["type_name"])
            dimension = _parse_vector_dimension(type_name)
            if dimension != self.embedding_dim:
                mismatches.append(
                    PgVectorColumnDimension(
                        table_name=str(row["table_name"]),
                        column_name=str(row["column_name"]),
                        type_name=type_name,
                        dimension=dimension,
                    )
                )
        if mismatches:
            raise PgVectorSchemaMismatchError(
                schema=self.schema,
                expected_dimension=self.embedding_dim,
                mismatches=mismatches,
            )

    async def _ensure_schema_async(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(self._ensure_schema_sync)

    async def _run_in_async_txn(self, fn):
        active = get_active_conn()
        invoke_async = getattr(active, "invoke_async", None)
        if callable(invoke_async):
            def _call(sync_conn):
                token = _pg_uow_conn.set(sync_conn)
                try:
                    return fn()
                finally:
                    _pg_uow_conn.reset(token)

            return await invoke_async(_call)
        invoke_sync = getattr(active, "invoke_sync", None)
        if callable(invoke_sync):
            def _call(sync_conn):
                token = _pg_uow_conn.set(sync_conn)
                try:
                    return fn()
                finally:
                    _pg_uow_conn.reset(token)

            return invoke_sync(_call)
        if isinstance(active, AsyncConnection):
            def _call(sync_conn):
                token = _pg_uow_conn.set(sync_conn)
                try:
                    return fn()
                finally:
                    _pg_uow_conn.reset(token)

            return await active.run_sync(_call)

        async with self.engine.begin() as conn:
            def _call(sync_conn):
                token = _pg_uow_conn.set(sync_conn)
                try:
                    return fn()
                finally:
                    _pg_uow_conn.reset(token)

            return await conn.run_sync(_call)

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

    async def _get_flat_async(
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

        async with self._async_conn() as conn:
            rows = (await conn.execute(q)).fetchall()

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

    async def _delete_async(
        self, table: sa.Table, *, ids: Optional[Sequence[str]], where: Optional[Json]
    ) -> None:
        stmt = sa.delete(table)
        if ids is not None:
            stmt = stmt.where(table.c.id.in_(list(ids)))
        if where:
            stmt = stmt.where(
                where_jsonb(table.c.metadata, where, numeric_keys=self.numeric_keys)
            )
        async with self._async_conn() as conn:
            await conn.execute(stmt)

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

    async def _upsert_async(
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

        async with self._async_conn() as conn:
            await conn.execute(stmt)

    async def async_node_upsert(self, **kwargs: Any) -> None:
        """Public async semantic write used by single-stage admission."""
        await self._upsert_async(self.nodes, **kwargs)

    async def async_edge_upsert(self, **kwargs: Any) -> None:
        """Public async semantic write used by single-stage admission."""
        await self._upsert_async(self.edges, **kwargs)

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

    async def _query_vector_async(
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

        op_map = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
        op = op_map[self.distance]
        qv_param = sa.bindparam("qv", type_=Vector(self.embedding_dim))
        distance_expr = sa.cast(table.c.embedding.op(op)(qv_param), sa.Float).label(
            "distance"
        )
        want_embeddings = "embeddings" in include
        if want_embeddings:
            embs_out: List[List[float]] = []

        async with self._async_conn() as conn:
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
                rows = (await conn.execute(q, {"qv": list(qv)})).fetchall()
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

    async def _update_doc_meta_embedding_merge_async(
        self,
        table: sa.Table,
        *,
        ids: Sequence[str],
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Json]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if documents is None and metadatas is None and embeddings is None:
            return

        if documents is not None and len(documents) != len(ids):
            raise ValueError("documents length must match ids length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids length")
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError("embeddings length must match ids length")

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

        async with self._async_conn() as conn:
            for i, _id in enumerate(ids):
                values: Dict[str, Any] = {"updated_at": sa.func.now()}
                params: Dict[str, Any] = {}

                if documents is not None:
                    values["document"] = documents[i]

                if metadatas is not None:
                    values["metadata"] = merged_expr
                    params["patch_text"] = json.dumps(metadatas[i])

                if embeddings is not None and "embedding" in table.c:
                    e = embeddings[i]
                    values["embedding"] = (
                        normalize_embedding_vector(e, allow_none=False)
                        if e is not None
                        else None
                    )

                stmt = sa.update(table).where(table.c.id == _id).values(**values)
                await conn.execute(stmt, params)

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

    async def _query_nonvector_async(
        self,
        table: sa.Table,
        *,
        where: Optional[Json],
        n_results: int,
        include: List[str],
    ) -> Dict[str, Any]:
        flat = await self._get_flat_async(
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

    def close(self) -> None:
        """Dispose pooled connections and roll back any checked-in work."""

        dispose = getattr(self.engine, "dispose", None)
        if not callable(dispose):
            return
        result = dispose()
        if inspect.isawaitable(result):
            _run_coro_sync(result)

    def _install_async_passthroughs(self) -> None:
        """Expose async-engine collection methods as eager awaitable results.

        Public backend methods must behave like Chroma collections: sync call
        sites should receive immediate dict/None-shaped results, while async
        call sites can still `await` the same returned object. We execute the
        async SQL work to completion here and wrap the value so both styles work.
        """

        def _bind(name: str):
            original = getattr(self, name)

            def _sync_wrapper(*args, **kwargs):
                if get_active_conn() is not None:
                    return original(*args, **kwargs)
                return _awaitable_result(_run_coro_sync(original(*args, **kwargs)))

            setattr(self, name, _sync_wrapper)

        for name in (
            "node_add",
            "node_upsert",
            "node_get",
            "node_delete",
            "node_query",
            "node_update",
            "edge_add",
            "edge_upsert",
            "edge_get",
            "edge_delete",
            "edge_query",
            "edge_update",
            "document_add",
            "document_upsert",
            "document_get",
            "document_delete",
            "document_query",
            "document_update",
            "domain_add",
            "domain_upsert",
            "domain_get",
            "domain_delete",
            "domain_query",
            "domain_update",
            "edge_endpoints_add",
            "edge_endpoints_upsert",
            "edge_endpoints_get",
            "edge_endpoints_query",
            "edge_endpoints_update",
            "edge_endpoints_delete",
            "edge_refs_add",
            "edge_refs_upsert",
            "edge_refs_get",
            "edge_refs_query",
            "edge_refs_update",
            "edge_refs_delete",
            "node_docs_add",
            "node_docs_upsert",
            "node_docs_get",
            "node_docs_query",
            "node_docs_update",
            "node_docs_delete",
            "node_refs_add",
            "node_refs_upsert",
            "node_refs_get",
            "node_refs_query",
            "node_refs_update",
            "node_refs_delete",
        ):
            _bind(name)

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
        return self._nodes_c.add(
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
        return self._nodes_c.upsert(
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
        return self._nodes_c.delete(ids=ids, where=where)

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
        return self._nodes_c.update(
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
        return self._edges_c.add(
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
        return self._edges_c.upsert(
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
        return self._edges_c.delete(ids=ids, where=where)

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
        return self._edges_c.update(
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
        return self._documents_c.add(
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
        return self._documents_c.upsert(
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
        return self._documents_c.delete(ids=ids, where=where)

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
        return self._documents_c.update(
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
        return self._domains_c.add(
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
        return self._domains_c.upsert(
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
        return self._domains_c.delete(ids=ids, where=where)

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
        return self._domains_c.update(
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
        return self._edge_endpoints_c.add(
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
        return self._edge_endpoints_c.upsert(
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
        return self._edge_endpoints_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def edge_endpoints_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        return self._edge_endpoints_c.delete(ids=ids, where=where)

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
        return self._edge_refs_c.add(
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
        return self._edge_refs_c.upsert(
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
        return self._edge_refs_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def edge_refs_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        return self._edge_refs_c.delete(ids=ids, where=where)

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
        return self._node_docs_c.add(
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
        return self._node_docs_c.upsert(
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
        return self._node_docs_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def node_docs_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        return self._node_docs_c.delete(ids=ids, where=where)

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
        return self._node_refs_c.add(
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
        return self._node_refs_c.upsert(
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
        return self._node_refs_c.update(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def node_refs_delete(
        self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None
    ) -> None:
        return self._node_refs_c.delete(ids=ids, where=where)


def build_postgres_backend(
    cfg: PgVectorConfig,
) -> Tuple[PgVectorBackend, PostgresUnitOfWork]:
    """Convenience helper for engine wiring."""

    engine = sa.create_engine(
        cfg.dsn,
        future=True,
        pool_pre_ping=True,
        pool_reset_on_return="rollback",
        pool_timeout=float(cfg.pool_timeout_s),
        connect_args=postgres_connect_args(cfg),
    )
    _install_connection_observability(engine, component="pgvector")
    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edges_table=cfg.edges_table,
        stage1_table=cfg.stage1_table,
        documents_table=cfg.documents_table,
        domains_table=cfg.domains_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
        edge_refs_table=cfg.edge_refs_table,
        node_docs_table=cfg.node_docs_table,
        node_refs_table=cfg.node_refs_table,
    )
    backend.ensure_schema()
    return backend, PostgresUnitOfWork(engine=engine)


def build_async_postgres_backend(
    cfg: PgVectorConfig,
) -> Tuple[PgVectorBackend, AsyncPostgresUnitOfWork]:
    """Async SQLAlchemy variant of `build_postgres_backend`."""

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        cfg.dsn,
        future=True,
        pool_pre_ping=True,
        pool_reset_on_return="rollback",
        pool_timeout=float(cfg.pool_timeout_s),
        connect_args=postgres_connect_args(cfg),
    )
    _install_connection_observability(engine, component="pgvector-async")
    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edges_table=cfg.edges_table,
        stage1_table=cfg.stage1_table,
        documents_table=cfg.documents_table,
        domains_table=cfg.domains_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
        edge_refs_table=cfg.edge_refs_table,
        node_docs_table=cfg.node_docs_table,
        node_refs_table=cfg.node_refs_table,
    )
    return backend, AsyncPostgresUnitOfWork(engine=engine)
