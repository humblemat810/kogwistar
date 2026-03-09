from __future__ import annotations

import contextvars
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional



_active_sqlite_conn: contextvars.ContextVar[sqlite3.Connection | None] = contextvars.ContextVar(
    "gke_sqlite_active_conn", default=None
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

            # Phase 2: coalescing + fingerprints
            # Ensure legacy DBs have the new column(s).
            cols = [r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()]
            if "namespace" not in cols:
                conn.execute("ALTER TABLE index_jobs ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'")
            # Phase 2: coalescing + fingerprints
            # Ensure legacy DBs have the new column(s).
            cols = [r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()]
            if "coalesce_key" not in cols:
                conn.execute("ALTER TABLE index_jobs ADD COLUMN coalesce_key TEXT NOT NULL DEFAULT ''")


            # Phase 5: scheduling / DLQ controls
            cols = [r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()]
            if "next_run_at" not in cols:
                conn.execute("ALTER TABLE index_jobs ADD COLUMN next_run_at INTEGER")
            cols = [r[1] for r in conn.execute("PRAGMA table_info(index_jobs)").fetchall()]
            if "max_retries" not in cols:
                conn.execute("ALTER TABLE index_jobs ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 10")

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
        max_retries: int = 10,
        namespace: str = 'default',
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
                next_op = "DELETE" if (op == "DELETE" or existing_op == "DELETE") else op
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
    (job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op,
     max_retries, payload_json, now, now),
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

        
    def mark_index_job_failed(self, job_id: str, error: str, *, final: bool = True) -> None:
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

    def bump_retry_and_requeue(self, job_id: str, error: str, *, next_run_at_seconds: int) -> None:
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
        namespace: Optional[str] = 'default',
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

    def get_index_applied_fingerprint(self, *, namespace: str = 'default', coalesce_key: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT applied_fingerprint FROM index_applied_state WHERE namespace = ? AND coalesce_key = ?",
                (namespace, coalesce_key),
            ).fetchone()
            return str(row[0]) if row and row[0] is not None else None

    def set_index_applied_fingerprint(
        self,
        *,
        namespace: str = 'default',
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
                (namespace, seq, event_id, entity_kind, entity_id, op, payload_json, now),
            )
        return seq

    def iter_entity_events( # mini batched
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

    def prune_entity_events_after(self, *, namespace: str = "default", to_seq: int) -> int:
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
