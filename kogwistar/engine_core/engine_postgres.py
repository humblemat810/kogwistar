from __future__ import annotations

"""EnginePostgres wiring.

This module is intentionally thin: it wires SQLAlchemy Engine + PgVectorBackend
and returns a UnitOfWork you can plug into the engine.

Vector/index collections supported by PgVectorBackend:
  - nodes, edges
  - edge_endpoints, edge_refs
  - node_docs, node_refs

Note: Your meta store is still EngineSQLite today. A later step is to migrate
meta tables to Postgres (either fully, or via a dual-write/outbox strategy).
"""

from dataclasses import dataclass
from typing import Tuple

import sqlalchemy as sa

from .postgres_backend import (
    AsyncPostgresUnitOfWork,
    PgVectorBackend,
    PgVectorConfig,
    PostgresUnitOfWork,
)


@dataclass(frozen=True)
class EnginePostgresConfig(PgVectorConfig):
    """Alias for now; keeps the config name stable at call sites."""


def build_postgres_backend(
    cfg: EnginePostgresConfig,
) -> Tuple[PgVectorBackend, PostgresUnitOfWork]:
    max_workers = 4
    engine = sa.create_engine(
        cfg.dsn,
        future=True,
        pool_size=max_workers + 2,
        pool_timeout=10.0,
    )

    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edges_table=cfg.edges_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
        edge_refs_table=cfg.edge_refs_table,
        node_docs_table=cfg.node_docs_table,
        node_refs_table=cfg.node_refs_table,
    )
    backend.ensure_schema()

    return backend, PostgresUnitOfWork(engine=engine)


def build_async_postgres_backend(
    cfg: EnginePostgresConfig,
) -> Tuple[PgVectorBackend, AsyncPostgresUnitOfWork]:
    from sqlalchemy.ext.asyncio import create_async_engine

    max_workers = 4
    engine = create_async_engine(
        cfg.dsn,
        future=True,
        pool_size=max_workers + 2,
        pool_timeout=10.0,
    )

    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edges_table=cfg.edges_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
        edge_refs_table=cfg.edge_refs_table,
        node_docs_table=cfg.node_docs_table,
        node_refs_table=cfg.node_refs_table,
    )
    return backend, AsyncPostgresUnitOfWork(engine=engine)
