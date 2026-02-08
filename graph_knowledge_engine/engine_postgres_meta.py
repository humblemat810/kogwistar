from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Iterator, Optional

import sqlalchemy as sa

from .postgres_backend import get_active_conn, _set_active_conn


_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class EnginePostgresMetaStore:
    """
    Postgres-backed replacement for EngineSQLite.

    Purpose:
      - allocate monotonic sequence numbers (global + per-user)
      - provide a transaction context that *joins* the active Postgres UoW connection
        when present, so seq allocation and graph writes are in the SAME transaction.

    Tables (in `schema`):
      - global_seq(value BIGINT NOT NULL) with a single row
      - user_seq(user_id TEXT PRIMARY KEY, value BIGINT NOT NULL)

    This is intentionally small and mirrors EngineSQLite semantics closely.
    """

    engine: sa.Engine
    schema: str = "public"
    global_table: str = "global_seq"
    user_table: str = "user_seq"

    def __post_init__(self) -> None:
        if not _SCHEMA_RE.match(self.schema):
            raise ValueError(f"invalid schema: {self.schema!r}")

    def ensure_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            conn.execute(sa.text(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.user_seq (
                    user_id TEXT PRIMARY KEY,
                    value BIGINT NOT NULL
                )
            """))
            conn.execute(sa.text(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.global_seq (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE,
                    value BIGINT NOT NULL
                )
            """))


    # ----------------------------
    # Initialization
    # ----------------------------
    def ensure_initialized(self) -> None:
        schema = self.schema
        gt = f'{schema}."{self.global_table}"' if self.global_table != "global_seq" else f"{schema}.global_seq"
        ut = f'{schema}."{self.user_table}"' if self.user_table != "user_seq" else f"{schema}.user_seq"

        with self.transaction() as conn:
            # Ensure schema exists
            conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            # Tables
            conn.execute(sa.text(f"CREATE TABLE IF NOT EXISTS {gt} (value BIGINT NOT NULL)"))            # The above ON CONFLICT requires a constraint; for single-row table, we keep it simple:
            # if table is empty, insert 0; otherwise do nothing.
            # Implement as: INSERT ... SELECT ... WHERE NOT EXISTS ...
            conn.execute(sa.text(
                f"INSERT INTO {gt}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {gt})"
            ))
            conn.execute(sa.text(
                f"CREATE TABLE IF NOT EXISTS {ut} (user_id TEXT PRIMARY KEY, value BIGINT NOT NULL)"
            ))


    # ----------------------------
    # Transaction helpers
    # ----------------------------
    @contextmanager
    def transaction(self) -> Iterator[sa.Connection]:
        """
        Start a transaction if none exists; otherwise join the active UoW connection.

        This matches PostgresUnitOfWork semantics but yields the connection so callers can
        execute SQL within the same transactional boundary.
        """
        existing = get_active_conn()
        if existing is not None:
            yield existing
            return

        with self.engine.begin() as conn:
            with _set_active_conn(conn):
                yield conn

    # ----------------------------
    # Global sequence
    # ----------------------------
    def next_global_seq(self) -> int:
        with self.transaction() as conn:
            return self.next_global_seq_conn(conn)

    def next_global_seq_conn(self, conn: sa.Connection) -> int:
        gt = f"{self.schema}.global_seq"
        row = conn.execute(sa.text(f"UPDATE {gt} SET value = value + 1 RETURNING value")).fetchone()
        if not row:
            # Defensive: if table is empty, initialize and retry
            conn.execute(sa.text(f"INSERT INTO {gt}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {gt})"))
            row = conn.execute(sa.text(f"UPDATE {gt} SET value = value + 1 RETURNING value")).fetchone()
        return int(row[0])

    def current_global_seq(self) -> int:
        gt = f"{self.schema}.global_seq"
        with self.transaction() as conn:
            row = conn.execute(sa.text(f"SELECT value FROM {gt} LIMIT 1")).fetchone()
            return int(row[0]) if row else 0

    # ----------------------------
    # Per-user sequence
    # ----------------------------
    def next_user_seq(self, user_id: str) -> int:
        with self.transaction() as conn:
            return self.next_user_seq_conn(conn, user_id)

    def next_user_seq_conn(self, conn: sa.Connection, user_id: str) -> int:
        ut = f"{self.schema}.user_seq"
        row = conn.execute(sa.text(
            f"""
            INSERT INTO {ut}(user_id, value)
            VALUES (:user_id, 1)
            ON CONFLICT(user_id)
            DO UPDATE SET value = {ut}.value + 1
            RETURNING value
            """
        ), {"user_id": user_id}).fetchone()
        return int(row[0])

    def current_user_seq(self, user_id: str) -> int:
        ut = f"{self.schema}.user_seq"
        with self.transaction() as conn:
            row = conn.execute(sa.text(f"SELECT value FROM {ut} WHERE user_id = :user_id"), {"user_id": user_id}).fetchone()
            return int(row[0]) if row else 0

    def set_user_seq(self, user_id: str, value: int) -> None:
        if value < 0:
            raise ValueError("value must be >= 0")
        with self.transaction() as conn:
            self.set_user_seq_conn(conn, user_id, value)

    def set_user_seq_conn(self, conn: sa.Connection, user_id: str, value: int) -> None:
        ut = f"{self.schema}.user_seq"
        conn.execute(sa.text(
            f"""
            INSERT INTO {ut}(user_id, value)
            VALUES (:user_id, :value)
            ON CONFLICT(user_id)
            DO UPDATE SET value = EXCLUDED.value
            """
        ), {"user_id": user_id, "value": int(value)})
