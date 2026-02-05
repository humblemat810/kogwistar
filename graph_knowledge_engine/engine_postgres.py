from __future__ import annotations

"""EnginePostgres wiring (minimal).

This is a minimal, practical starting point for Postgres + pgvector:

- Vector/index backend: PgVectorBackend (nodes + edge_endpoints only).
- Unit-of-work: PostgresUnitOfWork (wraps SQLAlchemy Engine.begin()).

Note: Your meta store is still EngineSQLite today. A later step is to migrate
meta tables to Postgres (either fully, or via a dual-write/outbox strategy).
"""

from dataclasses import dataclass
from typing import Tuple

import sqlalchemy as sa

from .postgres_backend import PgVectorBackend, PgVectorConfig, PostgresUnitOfWork


@dataclass(frozen=True)
class EnginePostgresConfig(PgVectorConfig):
    """Alias for now; keeps the config name stable at call sites."""


def build_postgres_backend(cfg: EnginePostgresConfig) -> Tuple[PgVectorBackend, PostgresUnitOfWork]:
    engine = sa.create_engine(cfg.dsn, future=True)

    backend = PgVectorBackend(
        engine=engine,
        embedding_dim=cfg.embedding_dim,
        schema=cfg.schema,
        nodes_table=cfg.nodes_table,
        edge_endpoints_table=cfg.edge_endpoints_table,
    )
    backend.ensure_schema()

    return backend, PostgresUnitOfWork(engine=engine)
