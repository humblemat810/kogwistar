from __future__ import annotations

import contextvars
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

from ..messaging.models import ProjectedLaneMessageRow
from .meta_lane_messages import LaneMessageMetaStoreMixin

_active_sqlite_conn: contextvars.ContextVar[sqlite3.Connection | None] = (
    contextvars.ContextVar("gke_sqlite_active_conn", default=None)
)


def get_active_sqlite_conn() -> sqlite3.Connection | None:
    return _active_sqlite_conn.get()


@contextmanager
def _set_active_sqlite_conn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    token = _active_sqlite_conn.set(conn)
    try:
        yield conn
    finally:
        _active_sqlite_conn.reset(token)


@dataclass(frozen=True)
class IndexJobRow:
    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    index_kind: str
    coalesce_key: str
    op: str
    status: str
    lease_until: Optional[int]
    next_run_at: Optional[int]
    max_retries: int
    retry_count: int
    last_error: Optional[str]
    payload_json: Optional[str]
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class ProjectedLaneMessageSqlRow:
    message_id: str
    namespace: str
    purpose: str
    inbox_id: str
    conversation_id: str
    recipient_id: str
    sender_id: str
    msg_type: str
    status: str
    seq: int
    conversation_seq: int
    claimed_by: str | None
    lease_until: int | None
    retry_count: int
    created_at: int
    available_at: int
    run_id: str | None
    step_id: str | None
    correlation_id: str | None
    payload_json: str | None
    error_json: str | None
    prev_message_id: str | None = None
    next_message_id: str | None = None
    inbox_tail_message_id: str | None = None
    conversation_tail_message_id: str | None = None

    def as_row(self) -> ProjectedLaneMessageRow:
        return ProjectedLaneMessageRow(
            message_id=self.message_id,
            namespace=self.namespace,
            purpose=self.purpose,
            inbox_id=self.inbox_id,
            conversation_id=self.conversation_id,
            recipient_id=self.recipient_id,
            sender_id=self.sender_id,
            msg_type=self.msg_type,
            status=self.status,
            seq=self.seq,
            conversation_seq=self.conversation_seq,
            claimed_by=self.claimed_by,
            lease_until=self.lease_until,
            retry_count=self.retry_count,
            created_at=self.created_at,
            available_at=self.available_at,
            run_id=self.run_id,
            step_id=self.step_id,
            correlation_id=self.correlation_id,
            payload_json=self.payload_json,
            error_json=self.error_json,
            prev_message_id=self.prev_message_id,
            next_message_id=self.next_message_id,
            inbox_tail_message_id=self.inbox_tail_message_id,
            conversation_tail_message_id=self.conversation_tail_message_id,
        )


class EngineSQLite(LaneMessageMetaStoreMixin):
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
            conn.execute("INSERT OR IGNORE INTO global_seq(rowid, value) VALUES (1, 0)")

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
                    namespace    TEXT NOT NULL DEFAULT 'default',
                    entity_kind  TEXT NOT NULL,
                    entity_id    TEXT NOT NULL,
                    index_kind   TEXT NOT NULL,
                    coalesce_key TEXT NOT NULL,
                    op           TEXT NOT NULL,
                    status       TEXT NOT NULL,
                    lease_until  INTEGER,
                    next_run_at INTEGER,
                    max_retries INTEGER NOT NULL DEFAULT 10,
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_index_jobs_namespace ON index_jobs(namespace)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projected_lane_messages (
                    message_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    purpose TEXT NOT NULL DEFAULT 'user_visible',
                    inbox_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    msg_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    conversation_seq INTEGER NOT NULL,
                    claimed_by TEXT,
                    lease_until INTEGER,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    available_at INTEGER NOT NULL,
                    run_id TEXT,
                    step_id TEXT,
                    correlation_id TEXT,
                    payload_json TEXT,
                    error_json TEXT,
                    prev_message_id TEXT,
                    next_message_id TEXT,
                    inbox_tail_message_id TEXT,
                    conversation_tail_message_id TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lane_messages_namespace_inbox_seq ON projected_lane_messages(namespace, inbox_id, seq)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lane_messages_claim ON projected_lane_messages(namespace, inbox_id, status, available_at, lease_until)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lane_messages_conversation_seq ON projected_lane_messages(namespace, conversation_id, conversation_seq)"
            )

            # Phase 2: coalescing + fingerprints
            # Ensure legacy DBs have the new column(s).
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()
            ]
            if "namespace" not in cols:
                conn.execute(
                    "ALTER TABLE index_jobs ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
                )
            # Phase 2: coalescing + fingerprints
            # Ensure legacy DBs have the new column(s).
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()
            ]
            if "coalesce_key" not in cols:
                conn.execute(
                    "ALTER TABLE index_jobs ADD COLUMN coalesce_key TEXT NOT NULL DEFAULT ''"
                )

            # Phase 5: scheduling / DLQ controls
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()
            ]
            if "next_run_at" not in cols:
                conn.execute("ALTER TABLE index_jobs ADD COLUMN next_run_at INTEGER")
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()
            ]
            if "max_retries" not in cols:
                conn.execute(
                    "ALTER TABLE index_jobs ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 10"
                )

            cols = [
                r[1]
                for r in conn.execute(
                    "PRAGMA table_info(projected_lane_messages)"
                ).fetchall()
            ]
            if "purpose" not in cols:
                conn.execute(
                    "ALTER TABLE projected_lane_messages ADD COLUMN purpose TEXT NOT NULL DEFAULT 'user_visible'"
                )

            # Unique pending job per (namespace, coalesce_key) (partial unique index)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_index_jobs_pending_ns_coalesce ON index_jobs(namespace, coalesce_key) WHERE status='PENDING'"
            )

            # Applied fingerprints for derived indexes
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_applied_state (
                    namespace           TEXT NOT NULL DEFAULT 'default',
                    coalesce_key        TEXT NOT NULL,
                    applied_fingerprint TEXT,
                    applied_at          INTEGER NOT NULL,
                    last_job_id         TEXT,
                    PRIMARY KEY(namespace, coalesce_key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_index_applied_state_key ON index_applied_state(coalesce_key)"
            )

            # -------------------------------
            # Phase 2b: event log foundation
            # -------------------------------

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS namespace_seq (
                    namespace TEXT PRIMARY KEY,
                    next_seq   INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO namespace_seq(namespace, next_seq) VALUES ('default', 1)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_events (
                    namespace    TEXT NOT NULL DEFAULT 'default',
                    seq          INTEGER NOT NULL,
                    event_id     TEXT NOT NULL,
                    entity_kind  TEXT NOT NULL,     -- 'node' | 'edge'
                    entity_id    TEXT NOT NULL,
                    op           TEXT NOT NULL,     -- 'ADD' | 'TOMBSTONE' | 'DELETE' | 'REPLACE'
                    payload_json TEXT NOT NULL,
                    created_at   INTEGER NOT NULL,
                    PRIMARY KEY(namespace, seq),
                    UNIQUE(event_id)
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_entity_events_aggregate
                ON entity_events(namespace, entity_kind, entity_id, seq)
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_cursors (
                    namespace  TEXT NOT NULL DEFAULT 'default',
                    consumer   TEXT NOT NULL,
                    last_seq   INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(namespace, consumer)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS named_projections (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    last_authoritative_seq INTEGER NOT NULL,
                    last_materialized_seq INTEGER NOT NULL,
                    projection_schema_version INTEGER NOT NULL,
                    materialization_status TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_named_projections_namespace
                ON named_projections(namespace, updated_at_ms)
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_design_snapshots (
                    workflow_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (workflow_id, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_design_version_deltas (
                    workflow_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    prev_version INTEGER NOT NULL,
                    target_seq INTEGER NOT NULL,
                    forward_json TEXT NOT NULL,
                    inverse_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (workflow_id, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS server_runs (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    user_id TEXT,
                    user_turn_node_id TEXT,
                    assistant_turn_node_id TEXT,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error_json TEXT,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    started_at_ms INTEGER,
                    finished_at_ms INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS server_run_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_server_runs_status ON server_runs(status, updated_at_ms)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_server_run_events_run_seq ON server_run_events(run_id, seq)"
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
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
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
        existing = get_active_sqlite_conn()
        if existing is not None:
            # Nested transaction scope: join outer UoW/transaction.
            # Do NOT BEGIN/COMMIT/ROLLBACK here.
            yield existing
            return

        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            with _set_active_sqlite_conn(conn):
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
            row = conn.execute(
                "SELECT value FROM global_seq WHERE rowid = 1"
            ).fetchone()
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

    def next_scoped_seq(self, scope_id: str) -> int:
        """
        Allocate the next sequence number scoped to an arbitrary scope id.
        """
        return self.next_user_seq(scope_id)

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

    def current_scoped_seq(self, scope_id: str) -> int:
        """
        Return the current sequence value for an arbitrary scope id.
        """
        return self.current_user_seq(scope_id)

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

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        """
        Hard reset: set the counter for an arbitrary scope id to an exact value.
        """
        self.set_user_seq(scope_id, value)

    def set_user_seq_conn(
        self, conn: sqlite3.Connection, user_id: str, value: int
    ) -> None:
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
        max_retries: int = 10,
        namespace: str = "default",
    ) -> str:
        """Enqueue durable derived-index work in the SQLite metastore.

        This queue is DB-backed, not an in-memory deque. Pending jobs coalesce by
        (namespace, coalesce_key), so repeated UPSERTs reuse one row while DELETE
        wins over UPSERT. The returned job_id may therefore identify an existing
        pending job instead of the caller-provided id.
        """
        now = self._now_epoch()
        coalesce_key = f"{entity_kind}:{entity_id}:{index_kind}"
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT job_id, op
                FROM index_jobs
                WHERE namespace = ? AND coalesce_key = ? AND status = 'PENDING'
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (namespace, coalesce_key),
            ).fetchone()
            if row:
                existing_job_id = str(row[0])
                existing_op = str(row[1] or "")
                next_op = (
                    "DELETE" if (op == "DELETE" or existing_op == "DELETE") else op
                )
                conn.execute(
                    """
                    UPDATE index_jobs
                    SET op = ?, payload_json = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (next_op, payload_json, now, existing_job_id),
                )
                return existing_job_id

            conn.execute(
                """
    INSERT OR IGNORE INTO index_jobs(
        job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op,
        status, lease_until, next_run_at, max_retries, retry_count, last_error, payload_json,
        created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', NULL, NULL, ?, 0, NULL, ?, ?, ?)
    """,
                (
                    job_id,
                    namespace,
                    entity_kind,
                    entity_id,
                    index_kind,
                    coalesce_key,
                    op,
                    max_retries,
                    payload_json,
                    now,
                    now,
                ),
            )
            return job_id

    def claim_index_jobs(
        self,
        *,
        limit: int = 50,
        lease_seconds: int = 60,
        namespace: Optional[str] = "default",
    ) -> List[IndexJobRow]:
        """Lease runnable jobs from the SQLite-backed queue.

        Eligibility is decided in SQL: pending jobs whose delay has elapsed plus
        doing jobs whose lease expired. Ordering is created_at ASC, namespace
        filtering happens at claim time, and FAILED rows are terminal. Claiming
        flips rows to DOING so concurrent workers do not process the same job.
        """
        if limit <= 0:
            return []
        now = self._now_epoch()
        lease_until = now + int(lease_seconds)
        with self.transaction() as conn:
            ns_filter = "" if namespace is None else "AND namespace = ?"
            ns_param: List[Any] = [] if namespace is None else [namespace]
            rows = conn.execute(
                f"""
                WITH candidates AS (
                    SELECT job_id
                    FROM index_jobs
                    WHERE (
                        (status='PENDING' AND (next_run_at IS NULL OR next_run_at <= ?))
                        OR (status='DOING' AND lease_until IS NOT NULL AND lease_until < ?)
                    )
                    {ns_filter}
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                UPDATE index_jobs
                SET status = 'DOING', lease_until = ?, updated_at = ?
                WHERE job_id IN (SELECT job_id FROM candidates)
                RETURNING job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op, status,
                        lease_until, next_run_at, max_retries, retry_count, last_error, payload_json, created_at, updated_at
                """,
                (now, now, *ns_param, limit, lease_until, now),
            ).fetchall()

        out: List[IndexJobRow] = []
        for r in rows:
            out.append(
                IndexJobRow(
                    job_id=str(r[0]),
                    namespace=str(r[1]),
                    entity_kind=str(r[2]),
                    entity_id=str(r[3]),
                    index_kind=str(r[4]),
                    coalesce_key=str(r[5]),
                    op=str(r[6]),
                    status=str(r[7]),
                    lease_until=int(r[8]) if r[8] is not None else None,
                    next_run_at=int(r[9]) if r[9] is not None else None,
                    max_retries=int(r[10] or 10),
                    retry_count=int(r[11] or 0),
                    last_error=str(r[12]) if r[12] is not None else None,
                    payload_json=str(r[13]) if r[13] is not None else None,
                    created_at=int(r[14]),
                    updated_at=int(r[15]),
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

    def mark_index_job_failed(
        self, job_id: str, error: str, *, final: bool = True
    ) -> None:
        """Mark a job failed.

        If final=True, job becomes terminal FAILED (DLQ) and will never be reclaimed.
        """
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status = CASE WHEN ? THEN 'FAILED' ELSE status END,
                    lease_until = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (1 if final else 0, (error or "")[:2000], now, job_id),
            )

    def bump_retry_and_requeue(
        self, job_id: str, error: str, *, next_run_at_seconds: int
    ) -> None:
        """Advance retry state after a failed apply attempt.

        The row stays in the durable queue: retry_count increments, last_error is
        updated, status returns to PENDING with next_run_at in the future, and rows
        that exhaust max_retries are promoted to terminal FAILED. The caller chooses
        the delay and backoff policy.
        """
        now = self._now_epoch()
        delay = max(0, int(next_run_at_seconds))
        next_run_at = now + delay
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET retry_count = retry_count + 1,
                    last_error = ?,
                    status = CASE WHEN (retry_count + 1) >= max_retries THEN 'FAILED' ELSE 'PENDING' END,
                    lease_until = NULL,
                    next_run_at = CASE WHEN (retry_count + 1) >= max_retries THEN NULL ELSE ? END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                ((error or "")[:2000], next_run_at, now, job_id),
            )

    def list_index_jobs(
        self,
        *,
        status: Optional[str] = None,
        entity_kind: Optional[str] = None,
        entity_id: Optional[str] = None,
        index_kind: Optional[str] = None,
        namespace: Optional[str] = "default",
        limit: int = 1000,
    ) -> List[IndexJobRow]:
        where: List[str] = []
        params: List[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if entity_kind is not None:
            where.append("entity_kind = ?")
            params.append(entity_kind)
        if entity_id is not None:
            where.append("entity_id = ?")
            params.append(entity_id)
        if namespace is not None:
            where.append("namespace = ?")
            params.append(namespace)
        if index_kind is not None:
            where.append("index_kind = ?")
            params.append(index_kind)
        sql = "SELECT job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op, status, lease_until, next_run_at, max_retries, retry_count, last_error, payload_json, created_at, updated_at FROM index_jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            IndexJobRow(
                job_id=str(r[0]),
                namespace=str(r[1]),
                entity_kind=str(r[2]),
                entity_id=str(r[3]),
                index_kind=str(r[4]),
                coalesce_key=str(r[5]),
                op=str(r[6]),
                status=str(r[7]),
                lease_until=int(r[8]) if r[8] is not None else None,
                next_run_at=int(r[9]) if r[9] is not None else None,
                max_retries=int(r[10] or 10),
                retry_count=int(r[11] or 0),
                last_error=str(r[12]) if r[12] is not None else None,
                payload_json=str(r[13]) if r[13] is not None else None,
                created_at=int(r[14]),
                updated_at=int(r[15]),
            )
            for r in rows
        ]

    def project_lane_message(
        self,
        *,
        message_id: str,
        namespace: str,
        purpose: str = "user_visible",
        inbox_id: str,
        conversation_id: str,
        recipient_id: str,
        sender_id: str,
        msg_type: str,
        status: str,
        created_at: int,
        available_at: int,
        run_id: str | None,
        step_id: str | None,
        correlation_id: str | None,
        payload_json: str | None = None,
        error_json: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            (next_seq,) = conn.execute(
                """
                SELECT COALESCE(MAX(seq), 0) + 1
                FROM projected_lane_messages
                WHERE namespace = ? AND inbox_id = ?
                """,
                (str(namespace), str(inbox_id)),
            ).fetchone()
            (next_conversation_seq,) = conn.execute(
                """
                SELECT COALESCE(MAX(conversation_seq), 0) + 1
                FROM projected_lane_messages
                WHERE namespace = ? AND conversation_id = ?
                """,
                (str(namespace), str(conversation_id)),
            ).fetchone()
            conn.execute(
                """
                INSERT OR IGNORE INTO projected_lane_messages(
                    message_id, namespace, purpose, inbox_id, conversation_id,
                    recipient_id, sender_id, msg_type, status,
                    seq, conversation_seq, claimed_by, lease_until,
                    retry_count, created_at, available_at, run_id,
                    step_id, correlation_id, payload_json, error_json,
                    prev_message_id, next_message_id,
                    inbox_tail_message_id, conversation_tail_message_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(message_id),
                    str(namespace),
                    str(purpose or "user_visible"),
                    str(inbox_id),
                    str(conversation_id),
                    str(recipient_id),
                    str(sender_id),
                    str(msg_type),
                    str(status),
                    int(next_seq),
                    int(next_conversation_seq),
                    int(created_at),
                    int(available_at),
                    None if run_id is None else str(run_id),
                    None if step_id is None else str(step_id),
                    None if correlation_id is None else str(correlation_id),
                    payload_json,
                    error_json,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )

    def update_projected_lane_message_status(
        self,
        *,
        message_id: str,
        status: str,
        error_json: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE projected_lane_messages
                SET status = ?,
                    error_json = COALESCE(?, error_json),
                    claimed_by = CASE WHEN ? IN ('completed', 'failed', 'cancelled') THEN NULL ELSE claimed_by END,
                    lease_until = CASE WHEN ? IN ('completed', 'failed', 'cancelled') THEN NULL ELSE lease_until END
                WHERE message_id = ?
                """,
                (str(status), error_json, str(status), str(status), str(message_id)),
            )

    def update_projected_lane_message_links(
        self,
        *,
        message_id: str,
        prev_message_id: str | None = None,
        next_message_id: str | None = None,
        inbox_tail_message_id: str | None = None,
        conversation_tail_message_id: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE projected_lane_messages
                SET prev_message_id = ?,
                    next_message_id = ?,
                    inbox_tail_message_id = ?,
                    conversation_tail_message_id = ?
                WHERE message_id = ?
                """,
                (
                    prev_message_id,
                    next_message_id,
                    inbox_tail_message_id,
                    conversation_tail_message_id,
                    str(message_id),
                ),
            )

    def claim_projected_lane_messages(
        self,
        *,
        namespace: str = "default",
        inbox_id: str,
        claimed_by: str,
        limit: int = 50,
        lease_seconds: int = 60,
    ) -> list[ProjectedLaneMessageRow]:
        if int(limit) <= 0:
            return []
        now = self._now_epoch()
        lease_until = now + int(lease_seconds)
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT message_id
                FROM projected_lane_messages
                WHERE namespace = ?
                  AND inbox_id = ?
                  AND (
                    (status = 'pending' AND available_at <= ?)
                    OR
                    (status = 'claimed' AND lease_until IS NOT NULL AND lease_until < ?)
                  )
                ORDER BY seq ASC, created_at ASC
                LIMIT ?
                """,
                (str(namespace), str(inbox_id), int(now), int(now), int(limit)),
            ).fetchall()
            ids = [str(row[0]) for row in rows]
            if not ids:
                return []
            for message_id in ids:
                conn.execute(
                    """
                    UPDATE projected_lane_messages
                    SET status = 'claimed', claimed_by = ?, lease_until = ?
                    WHERE message_id = ?
                    """,
                    (str(claimed_by), int(lease_until), message_id),
                )
            placeholders = ",".join("?" for _ in ids)
            got = conn.execute(
                f"""
                SELECT message_id, namespace, purpose, inbox_id, conversation_id, recipient_id, sender_id,
                       msg_type, status, seq, conversation_seq, claimed_by, lease_until,
                       retry_count, created_at, available_at, run_id, step_id, correlation_id,
                       payload_json, error_json, prev_message_id, next_message_id,
                       inbox_tail_message_id, conversation_tail_message_id
                FROM projected_lane_messages
                WHERE message_id IN ({placeholders})
                ORDER BY seq ASC
                """,
                tuple(ids),
            ).fetchall()
        return [
            ProjectedLaneMessageSqlRow(
                message_id=str(row[0]),
                namespace=str(row[1]),
                purpose=str(row[2] or "user_visible"),
                inbox_id=str(row[3]),
                conversation_id=str(row[4]),
                recipient_id=str(row[5]),
                sender_id=str(row[6]),
                msg_type=str(row[7]),
                status=str(row[8]),
                seq=int(row[9]),
                conversation_seq=int(row[10]),
                claimed_by=None if row[11] is None else str(row[11]),
                lease_until=None if row[12] is None else int(row[12]),
                retry_count=int(row[13]),
                created_at=int(row[14]),
                available_at=int(row[15]),
                run_id=None if row[16] is None else str(row[16]),
                step_id=None if row[17] is None else str(row[17]),
                correlation_id=None if row[18] is None else str(row[18]),
                payload_json=None if row[19] is None else str(row[19]),
                error_json=None if row[20] is None else str(row[20]),
                prev_message_id=None if row[21] is None else str(row[21]),
                next_message_id=None if row[22] is None else str(row[22]),
                inbox_tail_message_id=None if row[23] is None else str(row[23]),
                conversation_tail_message_id=None if row[24] is None else str(row[24]),
            ).as_row()
            for row in got
        ]

    def ack_projected_lane_message(self, *, message_id: str, claimed_by: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE projected_lane_messages
                SET status = 'completed', claimed_by = NULL, lease_until = NULL
                WHERE message_id = ? AND (claimed_by IS NULL OR claimed_by = ?)
                """,
                (str(message_id), str(claimed_by)),
            )

    def requeue_projected_lane_message(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error_json: str | None = None,
        delay_seconds: int = 0,
    ) -> None:
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE projected_lane_messages
                SET status = 'pending',
                    claimed_by = NULL,
                    lease_until = NULL,
                    retry_count = retry_count + 1,
                    available_at = ?,
                    error_json = COALESCE(?, error_json)
                WHERE message_id = ? AND (claimed_by IS NULL OR claimed_by = ?)
                """,
                (
                    int(now + max(0, int(delay_seconds))),
                    error_json,
                    str(message_id),
                    str(claimed_by),
                ),
            )

    def list_projected_lane_messages(
        self,
        *,
        namespace: str = "default",
        purpose: str | None = None,
        inbox_id: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[ProjectedLaneMessageRow]:
        where: list[str] = ["namespace = ?"]
        params: list[Any] = [str(namespace)]
        if purpose is not None:
            where.append("purpose = ?")
            params.append(str(purpose))
        if inbox_id is not None:
            where.append("inbox_id = ?")
            params.append(str(inbox_id))
        if status is not None:
            where.append("status = ?")
            params.append(str(status))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT message_id, namespace, purpose, inbox_id, conversation_id, recipient_id, sender_id,
                       msg_type, status, seq, conversation_seq, claimed_by, lease_until,
                       retry_count, created_at, available_at, run_id, step_id, correlation_id,
                       payload_json, error_json, prev_message_id, next_message_id,
                       inbox_tail_message_id, conversation_tail_message_id
                FROM projected_lane_messages
                WHERE {' AND '.join(where)}
                ORDER BY inbox_id ASC, seq ASC, created_at ASC
                LIMIT ?
                """,
                tuple(params + [int(limit)]),
            ).fetchall()
        return [
            ProjectedLaneMessageSqlRow(
                message_id=str(row[0]),
                namespace=str(row[1]),
                purpose=str(row[2] or "user_visible"),
                inbox_id=str(row[3]),
                conversation_id=str(row[4]),
                recipient_id=str(row[5]),
                sender_id=str(row[6]),
                msg_type=str(row[7]),
                status=str(row[8]),
                seq=int(row[9]),
                conversation_seq=int(row[10]),
                claimed_by=None if row[11] is None else str(row[11]),
                lease_until=None if row[12] is None else int(row[12]),
                retry_count=int(row[13]),
                created_at=int(row[14]),
                available_at=int(row[15]),
                run_id=None if row[16] is None else str(row[16]),
                step_id=None if row[17] is None else str(row[17]),
                correlation_id=None if row[18] is None else str(row[18]),
                payload_json=None if row[19] is None else str(row[19]),
                error_json=None if row[20] is None else str(row[20]),
                prev_message_id=None if row[21] is None else str(row[21]),
                next_message_id=None if row[22] is None else str(row[22]),
                inbox_tail_message_id=None if row[23] is None else str(row[23]),
                conversation_tail_message_id=None if row[24] is None else str(row[24]),
            ).as_row()
            for row in rows
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

    # ------------------------------------------------------------------
    # Phase 2: applied fingerprints (derived index status)
    # ------------------------------------------------------------------

    def get_index_applied_fingerprint(
        self, *, namespace: str = "default", coalesce_key: str
    ) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT applied_fingerprint FROM index_applied_state WHERE namespace = ? AND coalesce_key = ?",
                (namespace, coalesce_key),
            ).fetchone()
            return str(row[0]) if row and row[0] is not None else None

    def set_index_applied_fingerprint(
        self,
        *,
        namespace: str = "default",
        coalesce_key: str,
        applied_fingerprint: Optional[str],
        last_job_id: Optional[str] = None,
    ) -> None:
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO index_applied_state(namespace, coalesce_key, applied_fingerprint, applied_at, last_job_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, coalesce_key)
                DO UPDATE SET applied_fingerprint = excluded.applied_fingerprint,
                              applied_at = excluded.applied_at,
                              last_job_id = excluded.last_job_id
                """,
                (namespace, coalesce_key, applied_fingerprint, now, last_job_id),
            )

    # ------------------------------------------------------------------
    # Phase 2b: event log foundation
    # ------------------------------------------------------------------

    def alloc_event_seq(self, namespace: str = "default") -> int:
        with self.transaction() as conn:
            row = conn.execute(
                """
                UPDATE namespace_seq
                SET next_seq = next_seq + 1
                WHERE namespace = ?
                RETURNING next_seq - 1
                """,
                (namespace,),
            ).fetchone()

            if row is not None:
                return int(row[0])

            conn.execute(
                "INSERT INTO namespace_seq(namespace, next_seq) VALUES (?, 2)",
                (namespace,),
            )
            return 1

    def append_entity_event(
        self,
        *,
        namespace: str = "default",
        event_id: str,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str,
    ) -> int:
        seq = self.alloc_event_seq(namespace)
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO entity_events(
                    namespace, seq, event_id,
                    entity_kind, entity_id, op,
                    payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    seq,
                    event_id,
                    entity_kind,
                    entity_id,
                    op,
                    payload_json,
                    now,
                ),
            )
        return seq

    def iter_entity_events(  # mini batched
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
        batch_size: int = 500,
    ):
        next_seq = int(from_seq)
        while True:
            with self.connect() as conn:
                if to_seq is None:
                    rows = list(
                        conn.execute(
                            """
                            SELECT seq, entity_kind, entity_id, op, payload_json
                            FROM entity_events
                            WHERE namespace = ? AND seq >= ?
                            ORDER BY seq ASC
                            LIMIT ?
                            """,
                            (namespace, next_seq, int(batch_size)),
                        )
                    )
                else:
                    rows = list(
                        conn.execute(
                            """
                            SELECT seq, entity_kind, entity_id, op, payload_json
                            FROM entity_events
                            WHERE namespace = ? AND seq >= ? AND seq <= ?
                            ORDER BY seq ASC
                            LIMIT ?
                            """,
                            (namespace, next_seq, int(to_seq), int(batch_size)),
                        )
                    )

            if not rows:
                break

            for r in rows:
                yield r

            # advance cursor: next batch starts after the last seq we just yielded
            next_seq = int(rows[-1][0]) + 1

    def prune_entity_events_after(
        self, *, namespace: str = "default", to_seq: int
    ) -> int:
        """Delete namespace events with seq > to_seq.

        Used by workflow-design history branching to discard superseded redo events.
        """
        with self.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM entity_events WHERE namespace = ? AND seq > ?",
                (namespace, int(to_seq)),
            )
            return int(cur.rowcount or 0)

    # def iter_entity_events(
    #     self,
    #     *,
    #     namespace: str = "default",
    #     from_seq: int = 1,
    #     to_seq: int | None = None,
    # ):
    #     with self.connect() as conn:
    #         if to_seq is None:
    #             rows = conn.execute(
    #                 """
    #                 SELECT seq, entity_kind, entity_id, op, payload_json
    #                 FROM entity_events
    #                 WHERE namespace = ? AND seq >= ?
    #                 ORDER BY seq ASC
    #                 """,
    #                 (namespace, from_seq),
    #             )
    #         else:
    #             rows = conn.execute(
    #                 """
    #                 SELECT seq, entity_kind, entity_id, op, payload_json
    #                 FROM entity_events
    #                 WHERE namespace = ? AND seq BETWEEN ? AND ?
    #                 ORDER BY seq ASC
    #                 """,
    #                 (namespace, from_seq, to_seq),
    #             )
    #         yield from rows

    def cursor_get(self, *, namespace: str, consumer: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT last_seq FROM replay_cursors WHERE namespace=? AND consumer=?",
                (namespace, consumer),
            ).fetchone()
        return int(row[0]) if row else 0

    def cursor_set(self, *, namespace: str, consumer: str, last_seq: int) -> None:
        now = self._now_epoch()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO replay_cursors(namespace, consumer, last_seq, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, consumer) DO UPDATE
                SET last_seq=excluded.last_seq, updated_at=excluded.updated_at
                """,
                (namespace, consumer, int(last_seq), now),
            )

    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM entity_events WHERE namespace = ?",
                (namespace,),
            ).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _decode_named_projection_payload(raw_payload: Any) -> dict[str, Any]:
        payload = json.loads(str(raw_payload)) if raw_payload is not None else {}
        if not isinstance(payload, dict):
            raise ValueError("named projection payload must deserialize to a dict")
        return payload

    def get_named_projection(self, namespace: str, key: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT namespace, key, payload_json,
                       last_authoritative_seq, last_materialized_seq,
                       projection_schema_version, materialization_status, updated_at_ms
                FROM named_projections
                WHERE namespace = ? AND key = ?
                """,
                (str(namespace), str(key)),
            ).fetchone()
        if row is None:
            return None
        return {
            "namespace": str(row[0]),
            "key": str(row[1]),
            "payload": self._decode_named_projection_payload(row[2]),
            "last_authoritative_seq": int(row[3]),
            "last_materialized_seq": int(row[4]),
            "projection_schema_version": int(row[5]),
            "materialization_status": str(row[6]),
            "updated_at_ms": int(row[7]),
        }

    def replace_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        *,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        updated_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO named_projections(
                    namespace, key, payload_json,
                    last_authoritative_seq, last_materialized_seq,
                    projection_schema_version, materialization_status, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    last_authoritative_seq = excluded.last_authoritative_seq,
                    last_materialized_seq = excluded.last_materialized_seq,
                    projection_schema_version = excluded.projection_schema_version,
                    materialization_status = excluded.materialization_status,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    str(namespace),
                    str(key),
                    payload_json,
                    int(last_authoritative_seq),
                    int(last_materialized_seq),
                    int(projection_schema_version),
                    str(materialization_status),
                    updated_at_ms,
                ),
            )

    def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT namespace, key, payload_json,
                       last_authoritative_seq, last_materialized_seq,
                       projection_schema_version, materialization_status, updated_at_ms
                FROM named_projections
                WHERE namespace = ?
                ORDER BY key ASC
                """,
                (str(namespace),),
            ).fetchall()
        return [
            {
                "namespace": str(row[0]),
                "key": str(row[1]),
                "payload": self._decode_named_projection_payload(row[2]),
                "last_authoritative_seq": int(row[3]),
                "last_materialized_seq": int(row[4]),
                "projection_schema_version": int(row[5]),
                "materialization_status": str(row[6]),
                "updated_at_ms": int(row[7]),
            }
            for row in rows
        ]

    def clear_named_projection(self, namespace: str, key: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM named_projections WHERE namespace = ? AND key = ?",
                (str(namespace), str(key)),
            )

    def clear_projection_namespace(self, namespace: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM named_projections WHERE namespace = ?",
                (str(namespace),),
            )

    def get_workflow_design_projection(
        self, *, workflow_id: str
    ) -> Optional[dict[str, Any]]:
        projection = self.get_named_projection("workflow_design", str(workflow_id))
        if projection is None:
            return None
        payload = projection.get("payload") or {}
        versions = payload.get("versions") or []
        dropped_ranges = payload.get("dropped_ranges") or []
        return {
            "workflow_id": str(workflow_id),
            "current_version": int(payload.get("current_version") or 0),
            "active_tip_version": int(payload.get("active_tip_version") or 0),
            "last_authoritative_seq": int(
                projection.get("last_authoritative_seq") or 0
            ),
            "last_materialized_seq": int(projection.get("last_materialized_seq") or 0),
            "projection_schema_version": int(
                projection.get("projection_schema_version") or 1
            ),
            "snapshot_schema_version": int(payload.get("snapshot_schema_version") or 1),
            "materialization_status": str(
                projection.get("materialization_status") or "ready"
            ),
            "updated_at_ms": int(projection.get("updated_at_ms") or 0),
            "versions": [
                {
                    "version": int(item.get("version") or 0),
                    "prev_version": int(item.get("prev_version") or 0),
                    "target_seq": int(item.get("target_seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in versions
                if isinstance(item, dict)
            ],
            "dropped_ranges": [
                {
                    "start_seq": int(item.get("start_seq") or 0),
                    "end_seq": int(item.get("end_seq") or 0),
                    "start_version": int(item.get("start_version") or 0),
                    "end_version": int(item.get("end_version") or 0),
                }
                for item in dropped_ranges
                if isinstance(item, dict)
            ],
        }

    def replace_workflow_design_projection(
        self,
        *,
        workflow_id: str,
        head: dict[str, Any],
        versions: list[dict[str, Any]],
        dropped_ranges: list[dict[str, Any]],
    ) -> None:
        payload = {
            "current_version": int(head.get("current_version") or 0),
            "active_tip_version": int(head.get("active_tip_version") or 0),
            "snapshot_schema_version": int(head.get("snapshot_schema_version") or 1),
            "versions": [
                {
                    "version": int(item.get("version") or 0),
                    "prev_version": int(item.get("prev_version") or 0),
                    "target_seq": int(item.get("target_seq") or 0),
                    "created_at_ms": int(item.get("created_at_ms") or 0),
                }
                for item in versions
            ],
            "dropped_ranges": [
                {
                    "start_seq": int(item.get("start_seq") or 0),
                    "end_seq": int(item.get("end_seq") or 0),
                    "start_version": int(item.get("start_version") or 0),
                    "end_version": int(item.get("end_version") or 0),
                }
                for item in dropped_ranges
            ],
        }
        self.replace_named_projection(
            "workflow_design",
            str(workflow_id),
            payload,
            last_authoritative_seq=int(head.get("last_authoritative_seq") or 0),
            last_materialized_seq=int(head.get("last_materialized_seq") or 0),
            projection_schema_version=int(head.get("projection_schema_version") or 1),
            materialization_status=str(head.get("materialization_status") or "ready"),
        )

    def clear_workflow_design_projection(self, *, workflow_id: str) -> None:
        self.clear_named_projection("workflow_design", str(workflow_id))

    def put_workflow_design_snapshot(
        self,
        *,
        workflow_id: str,
        version: int,
        seq: int,
        payload_json: str,
        schema_version: int,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO workflow_design_snapshots(
                    workflow_id, version, seq, payload_json, schema_version, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id, version) DO UPDATE SET
                    seq = excluded.seq,
                    payload_json = excluded.payload_json,
                    schema_version = excluded.schema_version,
                    created_at_ms = excluded.created_at_ms
                """,
                (
                    workflow_id,
                    int(version),
                    int(seq),
                    payload_json,
                    int(schema_version),
                    int(self._now_epoch() * 1000),
                ),
            )

    def get_workflow_design_snapshot(
        self,
        *,
        workflow_id: str,
        max_version: int,
        schema_version: int,
    ) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT workflow_id, version, seq, payload_json, schema_version, created_at_ms
                FROM workflow_design_snapshots
                WHERE workflow_id = ? AND version <= ? AND schema_version = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (workflow_id, int(max_version), int(schema_version)),
            ).fetchone()
        if row is None:
            return None
        return {
            "workflow_id": str(row[0]),
            "version": int(row[1]),
            "seq": int(row[2]),
            "payload_json": str(row[3]),
            "schema_version": int(row[4]),
            "created_at_ms": int(row[5]),
        }

    def clear_workflow_design_snapshots(self, *, workflow_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM workflow_design_snapshots WHERE workflow_id = ?",
                (workflow_id,),
            )

    def put_workflow_design_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        prev_version: int,
        target_seq: int,
        forward_json: str,
        inverse_json: str,
        schema_version: int,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO workflow_design_version_deltas(
                    workflow_id, version, prev_version, target_seq,
                    forward_json, inverse_json, schema_version, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id, version) DO UPDATE SET
                    prev_version = excluded.prev_version,
                    target_seq = excluded.target_seq,
                    forward_json = excluded.forward_json,
                    inverse_json = excluded.inverse_json,
                    schema_version = excluded.schema_version,
                    created_at_ms = excluded.created_at_ms
                """,
                (
                    workflow_id,
                    int(version),
                    int(prev_version),
                    int(target_seq),
                    forward_json,
                    inverse_json,
                    int(schema_version),
                    int(self._now_epoch() * 1000),
                ),
            )

    def get_workflow_design_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        schema_version: int,
    ) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT workflow_id, version, prev_version, target_seq,
                       forward_json, inverse_json, schema_version, created_at_ms
                FROM workflow_design_version_deltas
                WHERE workflow_id = ? AND version = ? AND schema_version = ?
                """,
                (workflow_id, int(version), int(schema_version)),
            ).fetchone()
        if row is None:
            return None
        return {
            "workflow_id": str(row[0]),
            "version": int(row[1]),
            "prev_version": int(row[2]),
            "target_seq": int(row[3]),
            "forward_json": str(row[4]),
            "inverse_json": str(row[5]),
            "schema_version": int(row[6]),
            "created_at_ms": int(row[7]),
        }

    def clear_workflow_design_deltas(self, *, workflow_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM workflow_design_version_deltas WHERE workflow_id = ?",
                (workflow_id,),
            )

    @staticmethod
    def _decode_run_json(raw: Any) -> Any:
        if raw in (None, ""):
            return None
        return json.loads(str(raw))

    def create_server_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        workflow_id: str,
        user_id: str | None,
        user_turn_node_id: str,
        status: str = "queued",
    ) -> None:
        now = int(self._now_epoch() * 1000)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO server_runs(
                    run_id, conversation_id, workflow_id, user_id,
                    user_turn_node_id, assistant_turn_node_id, status,
                    cancel_requested, result_json, error_json,
                    created_at_ms, updated_at_ms, started_at_ms, finished_at_ms
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, 0, NULL, NULL, ?, ?, NULL, NULL)
                """,
                (
                    run_id,
                    conversation_id,
                    workflow_id,
                    user_id,
                    user_turn_node_id,
                    status,
                    now,
                    now,
                ),
            )

    def get_server_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id,
                       assistant_turn_node_id, status, cancel_requested, result_json,
                       error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms
                FROM server_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        status = str(row[6])
        return {
            "run_id": str(row[0]),
            "conversation_id": str(row[1]),
            "workflow_id": str(row[2]),
            "user_id": None if row[3] is None else str(row[3]),
            "user_turn_node_id": None if row[4] is None else str(row[4]),
            "assistant_turn_node_id": None if row[5] is None else str(row[5]),
            "status": status,
            "cancel_requested": bool(int(row[7] or 0)),
            "result": self._decode_run_json(row[8]),
            "error": self._decode_run_json(row[9]),
            "created_at_ms": int(row[10]),
            "updated_at_ms": int(row[11]),
            "started_at_ms": None if row[12] is None else int(row[12]),
            "finished_at_ms": None if row[13] is None else int(row[13]),
            "terminal": status in {"succeeded", "failed", "cancelled"},
        }

    def list_server_runs(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = [
            "SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id,",
            "       assistant_turn_node_id, status, cancel_requested, result_json,",
            "       error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms",
            "FROM server_runs",
        ]
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(str(workflow_id))
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(str(conversation_id))
        if clauses:
            sql.append("WHERE " + " AND ".join(clauses))
        sql.append("ORDER BY created_at_ms DESC, run_id DESC")
        sql.append("LIMIT ?")
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute("\n".join(sql), tuple(params)).fetchall()
        out = []
        for row in rows:
            status_val = str(row[6])
            out.append(
                {
                    "run_id": str(row[0]),
                    "conversation_id": str(row[1]),
                    "workflow_id": str(row[2]),
                    "user_id": None if row[3] is None else str(row[3]),
                    "user_turn_node_id": None if row[4] is None else str(row[4]),
                    "assistant_turn_node_id": None if row[5] is None else str(row[5]),
                    "status": status_val,
                    "cancel_requested": bool(int(row[7] or 0)),
                    "result": self._decode_run_json(row[8]),
                    "error": self._decode_run_json(row[9]),
                    "created_at_ms": int(row[10]),
                    "updated_at_ms": int(row[11]),
                    "started_at_ms": None if row[12] is None else int(row[12]),
                    "finished_at_ms": None if row[13] is None else int(row[13]),
                    "terminal": status_val in {"succeeded", "failed", "cancelled"},
                }
            )
        return out

    def list_server_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, run_id, event_type, payload_json, created_at_ms
                FROM server_run_events
                WHERE run_id = ? AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (run_id, int(after_seq), int(limit)),
            ).fetchall()
        return [
            {
                "seq": int(row[0]),
                "run_id": str(row[1]),
                "event_type": str(row[2]),
                "payload": self._decode_run_json(row[3]) or {},
                "created_at_ms": int(row[4]),
            }
            for row in rows
        ]

    def append_server_run_event(
        self,
        run_id: str,
        event_type: str,
        payload_json: str,
    ) -> dict[str, Any]:
        now = int(self._now_epoch() * 1000)
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO server_run_events(run_id, event_type, payload_json, created_at_ms)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, event_type, payload_json, now),
            )
            seq = int(cur.lastrowid)
        return {
            "seq": seq,
            "run_id": run_id,
            "event_type": event_type,
            "payload": self._decode_run_json(payload_json) or {},
            "created_at_ms": now,
        }

    def update_server_run(
        self,
        *,
        run_id: str,
        status: str,
        assistant_turn_node_id: str | None,
        result_json: str | None,
        error_json: str | None,
        started_at_ms: int | None,
        finished_at_ms: int | None,
        cancel_requested: bool | None = None,
    ) -> None:
        now = int(self._now_epoch() * 1000)
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE server_runs
                SET status = ?,
                    assistant_turn_node_id = ?,
                    result_json = ?,
                    error_json = ?,
                    started_at_ms = ?,
                    finished_at_ms = ?,
                    cancel_requested = COALESCE(?, cancel_requested),
                    updated_at_ms = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    assistant_turn_node_id,
                    result_json,
                    error_json,
                    started_at_ms,
                    finished_at_ms,
                    (None if cancel_requested is None else int(bool(cancel_requested))),
                    now,
                    run_id,
                ),
            )

    def request_server_run_cancel(self, *, run_id: str) -> None:
        now = int(self._now_epoch() * 1000)
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE server_runs
                SET cancel_requested = 1,
                    status = CASE
                        WHEN status IN ('cancelled', 'failed', 'succeeded') THEN status
                        ELSE 'cancelling'
                    END,
                    updated_at_ms = ?
                WHERE run_id = ?
                """,
                (now, run_id),
            )
