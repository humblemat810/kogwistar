from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional


@dataclass(frozen=True)
class IndexJobRow:
    job_id: str
    entity_kind: str
    entity_id: str
    index_kind: str
    op: str
    status: str
    lease_until: Optional[int]
    retry_count: int
    last_error: Optional[str]
    payload_json: Optional[str]
    created_at: int
    updated_at: int


class EngineSQLite:
    """
    Lightweight SQLite helper for engine persistence.

    Responsibilities:
    - ensure persistent DB exists
    - allocate monotonic sequence numbers:
        * global counter (single row, no growth)
        * per-user counter (one row per user, no growth per increment)
    - provide safe transactional context

    Seq semantics:
    - Global counter is stored in table: global_seq(value)
        * current_global_seq() returns the last issued value (0 if never issued).
    - Per-user counters are stored in table: user_seq(user_id, value)
        * current_user_seq(user_id) returns last issued value for that user (0 if none).
    """

    def __init__(
        self,
        persistent_directory: Path,
        filename: str = "engine.db",
    ) -> None:
        self.persistent_directory = persistent_directory
        self.db_path = persistent_directory / filename

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """
        Create persistent directory and required tables if they do not exist.
        Safe to call multiple times.
        """
        self.persistent_directory.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_seq (
                    value INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO global_seq(rowid, value) VALUES (1, 0)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_seq (
                    user_id TEXT PRIMARY KEY,
                    value   INTEGER NOT NULL
                )
                """
            )

            # Durable outbox-style queue for derived index convergence.
            # Lease-based claiming prevents "halt forever" if a worker stalls.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id       TEXT PRIMARY KEY,
                    entity_kind  TEXT NOT NULL,
                    entity_id    TEXT NOT NULL,
                    index_kind   TEXT NOT NULL,
                    op           TEXT NOT NULL,
                    status       TEXT NOT NULL,
                    lease_until  INTEGER,
                    retry_count  INTEGER NOT NULL DEFAULT 0,
                    last_error   TEXT,
                    payload_json TEXT,
                    created_at   INTEGER NOT NULL,
                    updated_at   INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_index_jobs_status_lease ON index_jobs(status, lease_until)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_index_jobs_entity ON index_jobs(entity_kind, entity_id, index_kind)"
            )

    # ------------------------------------------------------------------
    # Connection / transaction helpers
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """
        Create a SQLite connection with sane defaults.
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """
        Context manager for a short, safe transaction.

        Default: BEGIN IMMEDIATE (prevents writer starvation).

        Example:
            with db.transaction() as conn:
                ...
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Global sequence
    # ------------------------------------------------------------------

    def next_global_seq(self) -> int:
        """
        Allocate the next global monotonic sequence number.
        """
        with self.transaction() as conn:
            return self.next_global_seq_conn(conn)

    def next_global_seq_conn(self, conn: sqlite3.Connection) -> int:
        """
        Allocate the next global sequence using an existing transaction.
        """
        (value,) = conn.execute(
            "UPDATE global_seq SET value = value + 1 RETURNING value"
        ).fetchone()
        return int(value)

    def current_global_seq(self) -> int:
        """
        Return the current global sequence value (last issued).
        """
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM global_seq WHERE rowid = 1").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Per-user sequence
    # ------------------------------------------------------------------

    def next_user_seq(self, user_id: str) -> int:
        """
        Allocate the next sequence number scoped to user_id.
        """
        with self.transaction() as conn:
            return self.next_user_seq_conn(conn, user_id)

    def next_user_seq_conn(self, conn: sqlite3.Connection, user_id: str) -> int:
        """
        Allocate the next user sequence using an existing transaction.
        """
        (value,) = conn.execute(
            """
            INSERT INTO user_seq(user_id, value)
            VALUES (?, 1)
            ON CONFLICT(user_id)
            DO UPDATE SET value = user_seq.value + 1
            RETURNING value
            """,
            (user_id,),
        ).fetchone()
        return int(value)

    def current_user_seq(self, user_id: str) -> int:
        """
        Return the current sequence value for user_id.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM user_seq WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    def set_user_seq(self, user_id: str, value: int) -> None:
        """
        Hard reset: set the counter for user_id to an exact value.

        WARNING: this allows seq reuse, which can collide with previously-issued seq
        that may already exist in persisted logs / Chroma / nodes tables.
        """
        if value < 0:
            raise ValueError("value must be >= 0")

        with self.transaction() as conn:
            self.set_user_seq_conn(conn, user_id, value)

    def set_user_seq_conn(self, conn: sqlite3.Connection, user_id: str, value: int) -> None:
        conn.execute(
            """
            INSERT INTO user_seq(user_id, value)
            VALUES (?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET value = excluded.value
            """,
            (user_id, value),
        )

    # ------------------------------------------------------------------
    # Index jobs (outbox-style derived-index convergence)
    # ------------------------------------------------------------------

    @staticmethod
    def _now_epoch() -> int:
        return int(datetime.now(timezone.utc).timestamp())

    def enqueue_index_job(
        self,
        *,
        job_id: str,
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        payload_json: Optional[str] = None,
    ) -> None:
        """Insert a new job (PENDING). If job_id already exists, do not overwrite."""
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO index_jobs(
                    job_id, entity_kind, entity_id, index_kind, op,
                    status, lease_until, retry_count, last_error, payload_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'PENDING', NULL, 0, NULL, ?, ?, ?)
                """,
                (job_id, entity_kind, entity_id, index_kind, op, payload_json, now, now),
            )

    def claim_index_jobs(self, *, limit: int = 50, lease_seconds: int = 60) -> List[IndexJobRow]:
        """Claim up to `limit` jobs with an expiring lease.

        Steals jobs whose lease has expired (covers "halt forever" stalls).
        """
        if limit <= 0:
            return []
        now = self._now_epoch()
        lease_until = now + int(lease_seconds)
        with self.transaction() as conn:
            rows = conn.execute(
                """
                WITH candidates AS (
                    SELECT job_id
                    FROM index_jobs
                    WHERE (
                        status IN ('PENDING', 'FAILED')
                        OR (status = 'DOING' AND lease_until IS NOT NULL AND lease_until < ?)
                    )
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                UPDATE index_jobs
                SET status = 'DOING', lease_until = ?, updated_at = ?
                WHERE job_id IN (SELECT job_id FROM candidates)
                RETURNING job_id, entity_kind, entity_id, index_kind, op, status,
                          lease_until, retry_count, last_error, payload_json, created_at, updated_at
                """,
                (now, limit, lease_until, now),
            ).fetchall()

        out: List[IndexJobRow] = []
        for r in rows:
            out.append(
                IndexJobRow(
                    job_id=str(r[0]),
                    entity_kind=str(r[1]),
                    entity_id=str(r[2]),
                    index_kind=str(r[3]),
                    op=str(r[4]),
                    status=str(r[5]),
                    lease_until=int(r[6]) if r[6] is not None else None,
                    retry_count=int(r[7] or 0),
                    last_error=str(r[8]) if r[8] is not None else None,
                    payload_json=str(r[9]) if r[9] is not None else None,
                    created_at=int(r[10]),
                    updated_at=int(r[11]),
                )
            )
        return out

    def mark_index_job_done(self, job_id: str) -> None:
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status='DONE', lease_until=NULL, updated_at=?
                WHERE job_id=?
                """,
                (now, job_id),
            )

    def mark_index_job_failed(self, job_id: str, error: str) -> None:
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status='FAILED', lease_until=NULL, retry_count=retry_count+1,
                    last_error=?, updated_at=?
                WHERE job_id=?
                """,
                (error[:2000], now, job_id),
            )

    def list_index_jobs(self, *, status: Optional[str] = None, entity_id: Optional[str] = None) -> List[IndexJobRow]:
        where = []
        params: List[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if entity_id is not None:
            where.append("entity_id = ?")
            params.append(entity_id)
        sql = "SELECT job_id, entity_kind, entity_id, index_kind, op, status, lease_until, retry_count, last_error, payload_json, created_at, updated_at FROM index_jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            IndexJobRow(
                job_id=str(r[0]),
                entity_kind=str(r[1]),
                entity_id=str(r[2]),
                index_kind=str(r[3]),
                op=str(r[4]),
                status=str(r[5]),
                lease_until=int(r[6]) if r[6] is not None else None,
                retry_count=int(r[7] or 0),
                last_error=str(r[8]) if r[8] is not None else None,
                payload_json=str(r[9]) if r[9] is not None else None,
                created_at=int(r[10]),
                updated_at=int(r[11]),
            )
            for r in rows
        ]
    # ------------------------------------------------------------------
    # Usage examples
    # ------------------------------------------------------------------

    """
    Usage examples
    ==============

    Engine startup
    --------------
    >>> db = EngineSQLite(Path("/var/lib/my_engine"))
    >>> db.ensure_initialized()

    Global sequence (engine-wide ordering)
    -------------------------------------
    >>> seq = db.next_global_seq()
    >>> seq
    1

    >>> db.current_global_seq()
    1

    Per-user sequence
    -----------------
    >>> u1 = db.next_user_seq("alice")
    >>> u2 = db.next_user_seq("alice")
    >>> u1, u2
    (1, 2)

    >>> db.current_user_seq("alice")
    2

    Atomic allocation + insert
    --------------------------
    >>> with db.transaction() as conn:
    ...     seq = db.next_global_seq_conn(conn)
    ...     conn.execute(
    ...         "INSERT INTO nodes (node_id, seq) VALUES (?, ?)",
    ...         ("n1", seq),
    ...     )

    Notes
    -----
    - Global sequence uses a single-row counter (constant storage).
    - Per-user sequence uses one row per user (constant storage per user).
    - All increments are atomic and safe under concurrency.
    - BEGIN IMMEDIATE prevents writer starvation under read load.
    db.set_user_seq("alice", 17)
    """
