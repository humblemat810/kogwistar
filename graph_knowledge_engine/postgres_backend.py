from __future__ import annotations

"""PostgreSQL + pgvector backend (minimal).

This module intentionally implements only the first slice needed for refactor:
- nodes: add/get/delete/query (vector similarity)
- edge_endpoints: add/get/delete (incidence materialization)

Other collections should be added in later slices once tests pass.

Design notes:
- Metadata is stored as JSONB.
- Embedding storage uses pgvector (Vector column).
- Upserts are implemented via INSERT .. ON CONFLICT .. DO UPDATE.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
try:
    # pip install pgvector
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception as e:  # pragma: no cover
    Vector = None  # type: ignore
    _pgvector_import_error = e
else:
    _pgvector_import_error = None


Json = Dict[str, Any]


class PostgresUnitOfWork:
    """A simple UoW wrapper around SQLAlchemy Engine.begin()."""

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
    edge_endpoints_table: str = "gke_edge_endpoints"


class PgVectorBackend:
    """Minimal pgvector backend implementing node + edge_endpoints operations."""

    def __init__(self, *, engine: sa.Engine, embedding_dim: int, schema: str, nodes_table: str, edge_endpoints_table: str):
        if Vector is None:  # pragma: no cover
            raise RuntimeError(
                "pgvector is not installed. Install with `pip install pgvector` to use PgVectorBackend."
            ) from _pgvector_import_error

        self.engine = engine
        self.embedding_dim = int(embedding_dim)
        self.schema = schema
        self.nodes_table_name = nodes_table
        self.edge_endpoints_table_name = edge_endpoints_table

        md = sa.MetaData(schema=self.schema)

        self.nodes = sa.Table(
            self.nodes_table_name,
            md,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("embedding", Vector(self.embedding_dim), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        )

        # Hyperedge incidence materialization:
        # one row per (edge_id, endpoint_node_id, role, ord)
        self.edge_endpoints = sa.Table(
            self.edge_endpoints_table_name,
            md,
            sa.Column("id", sa.String, primary_key=True),  # stable synthetic id from engine (edge_id|role|ord|endpoint)
            sa.Column("document", sa.Text, nullable=True),
            sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
            sa.Index(f"ix_{self.edge_endpoints_table_name}_endpoint", sa.text("(metadata->>'endpoint_node_id')")),
            sa.Index(f"ix_{self.edge_endpoints_table_name}_edge", sa.text("(metadata->>'edge_id')")),
        )

        self._md = md

    def ensure_schema(self) -> None:
        """Dev convenience: create schema/tables if missing. Prefer migrations in production."""
        with self.engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            if self.schema and self.schema != "public":
                conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
            self._md.create_all(conn)

    # -------------------------
    # Nodes
    # -------------------------
    def node_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json], embeddings: Optional[Sequence[List[float]]] = None) -> None:
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError("embeddings length must match ids length")

        rows = []
        for i, _id in enumerate(ids):
            rows.append(
                {
                    "id": _id,
                    "document": documents[i] if i < len(documents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "embedding": embeddings[i] if embeddings is not None else None,
                }
            )

        stmt = pg_insert(self.nodes).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.nodes.c.id],
            set_={
                "document": stmt.excluded.document,
                "metadata": stmt.excluded.metadata,
                "embedding": stmt.excluded.embedding,
                "updated_at": sa.func.now(),
            },
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def node_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None, include: Optional[List[str]] = None, limit: int = 200) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "ids"]
        q = sa.select(self.nodes.c.id, self.nodes.c.document, self.nodes.c.metadata).limit(int(limit))
        if ids is not None:
            q = q.where(self.nodes.c.id.in_(list(ids)))
        if where:
            q = q.where(self._where_jsonb(self.nodes.c.metadata, where))

        with self.engine.begin() as conn:
            rows = conn.execute(q).fetchall()

        out: Dict[str, Any] = {"ids": [[r.id for r in rows]]}
        if "documents" in include:
            out["documents"] = [[r.document for r in rows]]
        if "metadatas" in include:
            out["metadatas"] = [[r.metadata for r in rows]]
        return out

    def node_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None) -> None:
        stmt = sa.delete(self.nodes)
        if ids is not None:
            stmt = stmt.where(self.nodes.c.id.in_(list(ids)))
        if where:
            stmt = stmt.where(self._where_jsonb(self.nodes.c.metadata, where))
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def node_query(self, *, query_embeddings: Sequence[List[float]], n_results: int = 10, where: Optional[Json] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "ids", "distances"]
        if not query_embeddings:
            raise ValueError("query_embeddings is required for PgVectorBackend")

        # Use cosine distance (<=>) by default (pgvector). If you prefer L2, use <->.
        # We return "distances" matching Chroma semantics (smaller is closer).
        results_ids: List[List[str]] = []
        results_docs: List[List[Optional[str]]] = []
        results_metas: List[List[Json]] = []
        results_dists: List[List[float]] = []

        with self.engine.begin() as conn:
            for qv in query_embeddings:
                q = sa.select(
                    self.nodes.c.id,
                    self.nodes.c.document,
                    self.nodes.c.metadata,
                    (self.nodes.c.embedding.op("<=>")(sa.bindparam("qv"))).label("distance"),
                ).where(self.nodes.c.embedding.is_not(None))
                if where:
                    q = q.where(self._where_jsonb(self.nodes.c.metadata, where))
                q = q.order_by(sa.text("distance asc")).limit(int(n_results))

                rows = conn.execute(q, {"qv": qv}).fetchall()
                results_ids.append([r.id for r in rows])
                results_docs.append([r.document for r in rows])
                results_metas.append([r.metadata for r in rows])
                results_dists.append([float(r.distance) for r in rows])

        out: Dict[str, Any] = {"ids": results_ids}
        if "documents" in include:
            out["documents"] = results_docs
        if "metadatas" in include:
            out["metadatas"] = results_metas
        if "distances" in include:
            out["distances"] = results_dists
        return out

    # -------------------------
    # Edge endpoints
    # -------------------------
    def edge_endpoints_add(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Json]) -> None:
        rows = []
        for i, _id in enumerate(ids):
            rows.append(
                {
                    "id": _id,
                    "document": documents[i] if i < len(documents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        stmt = pg_insert(self.edge_endpoints).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.edge_endpoints.c.id],
            set_={
                "document": stmt.excluded.document,
                "metadata": stmt.excluded.metadata,
                "updated_at": sa.func.now(),
            },
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def edge_endpoints_get(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None, include: Optional[List[str]] = None, limit: int = 200) -> Dict[str, Any]:
        include = include or ["documents", "metadatas", "ids"]
        q = sa.select(self.edge_endpoints.c.id, self.edge_endpoints.c.document, self.edge_endpoints.c.metadata).limit(int(limit))
        if ids is not None:
            q = q.where(self.edge_endpoints.c.id.in_(list(ids)))
        if where:
            q = q.where(self._where_jsonb(self.edge_endpoints.c.metadata, where))

        with self.engine.begin() as conn:
            rows = conn.execute(q).fetchall()

        out: Dict[str, Any] = {"ids": [[r.id for r in rows]]}
        if "documents" in include:
            out["documents"] = [[r.document for r in rows]]
        if "metadatas" in include:
            out["metadatas"] = [[r.metadata for r in rows]]
        return out

    def edge_endpoints_delete(self, *, ids: Optional[Sequence[str]] = None, where: Optional[Json] = None) -> None:
        stmt = sa.delete(self.edge_endpoints)
        if ids is not None:
            stmt = stmt.where(self.edge_endpoints.c.id.in_(list(ids)))
        if where:
            stmt = stmt.where(self._where_jsonb(self.edge_endpoints.c.metadata, where))
        with self.engine.begin() as conn:
            conn.execute(stmt)

    # -------------------------
    # Helpers
    # -------------------------
    def _where_jsonb(self, col: sa.ColumnElement, where: Json) -> sa.ColumnElement:
        """Minimal where translation: only supports equality on scalar keys."""
        clauses = []
        for k, v in where.items():
            # Chroma where may include operators; minimal impl supports direct equality only.
            if isinstance(v, dict):
                raise NotImplementedError(f"Operator where not implemented for key={k}: {v}")
            clauses.append(col.op("->>")(k) == str(v))
        return sa.and_(*clauses) if clauses else sa.true()
