from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from .postgres_backend import get_active_conn, _set_active_conn
from ..messaging.models import ProjectedLaneMessageRow
from .meta_lane_messages import LaneMessageMetaStoreMixin


_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _run_coro_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if sys.platform == "win32":
            runner = asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
            try:
                return runner.run(coro)
            finally:
                runner.close()
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["result"] = _run_coro_blocking(coro)
        except BaseException as exc:  # pragma: no cover - thread ferry
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


class _BufferedMappings:
    def __init__(self, rows: list[Any]):
        self._rows = rows

    def all(self) -> list[dict[str, Any]]:
        return [dict(getattr(row, "_mapping", row)) for row in self._rows]


class _BufferedResult:
    def __init__(self, rows: list[Any], rowcount: int | None = None):
        self._rows = rows
        self.rowcount = rowcount if rowcount is not None else len(rows)

    def __iter__(self):
        return iter(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def mappings(self) -> _BufferedMappings:
        return _BufferedMappings(self._rows)


class _AsyncConnectionAdapter:
    def __init__(self, conn: AsyncConnection, runner: asyncio.Runner):
        self._conn = conn
        self._runner = runner

    def invoke_sync(self, fn):
        return _run_coro_blocking(self._conn.run_sync(fn))

    async def invoke_async(self, fn):
        return await self._conn.run_sync(fn)

    def execute(self, statement, params=None):
        def _execute(sync_conn):
            result = (
                sync_conn.execute(statement)
                if params is None
                else sync_conn.execute(statement, params)
            )
            rows = result.fetchall() if result.returns_rows else []
            return _BufferedResult(rows, getattr(result, "rowcount", None))

        return self.invoke_sync(_execute)


def _make_runner() -> asyncio.Runner:
    if sys.platform == "win32":
        return asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
    return asyncio.Runner()


@dataclass
class IndexJob:
    """Typed view of an index job (matches EngineSQLite.IndexJobRow)."""

    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    index_kind: str
    coalesce_key: str
    op: str
    status: str
    lease_until: Optional[str] = None
    next_run_at: Optional[str] = None
    max_retries: int = 10
    retry_count: int = 0
    last_error: Optional[str] = None
    payload_json: Optional[str] = None


@dataclass
class ProjectedLaneMessage:
    message_id: str
    namespace: str
    inbox_id: str
    conversation_id: str
    recipient_id: str
    sender_id: str
    msg_type: str
    status: str
    seq: int
    conversation_seq: int
    claimed_by: Optional[str] = None
    lease_until: Optional[str] = None
    retry_count: int = 0
    created_at: int = 0
    available_at: int = 0
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    correlation_id: Optional[str] = None
    payload_json: Optional[str] = None
    error_json: Optional[str] = None
    prev_message_id: Optional[str] = None
    next_message_id: Optional[str] = None
    inbox_tail_message_id: Optional[str] = None
    conversation_tail_message_id: Optional[str] = None


@dataclass
class EnginePostgresMetaStore(LaneMessageMetaStoreMixin):
    """Postgres-backed replacement for EngineSQLite.

    Responsibilities:
      - allocate monotonic sequence numbers (global + per-user)
      - provide a transaction context that joins the active Postgres UoW
      - persist outbox-style index jobs with leasing for derived-index convergence

    Tables (in `schema`):
      - global_seq(value BIGINT NOT NULL) single-row
      - user_seq(user_id TEXT PRIMARY KEY, value BIGINT NOT NULL)
      - index_jobs(...) durable queue
    """

    engine: sa.Engine | AsyncEngine
    schema: str = "public"
    global_table: str = "global_seq"
    user_table: str = "user_seq"
    index_jobs_table: str = "index_jobs"

    def __post_init__(self) -> None:
        if not _SCHEMA_RE.match(self.schema):
            raise ValueError(f"invalid schema: {self.schema!r}")
        self._is_async_engine = isinstance(self.engine, AsyncEngine)

    # ----------------------------
    # Initialization
    # ----------------------------
    def _bootstrap_identifiers(self) -> tuple[str, str, str, str, str]:
        schema = self.schema
        gt = (
            f'{schema}."{self.global_table}"'
            if self.global_table != "global_seq"
            else f"{schema}.global_seq"
        )
        ut = (
            f'{schema}."{self.user_table}"'
            if self.user_table != "user_seq"
            else f"{schema}.user_seq"
        )
        ij = (
            f"{schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f'{schema}."{self.index_jobs_table}"'
        )
        ias = f"{schema}.index_applied_state"
        return schema, gt, ut, ij, ias

    def _bootstrap_statements(self) -> list[str]:
        schema, gt, ut, ij, ias = self._bootstrap_identifiers()
        plm = f"{schema}.projected_lane_messages"
        return [
            f"CREATE SCHEMA IF NOT EXISTS {schema}",
            f"CREATE TABLE IF NOT EXISTS {gt} (value BIGINT NOT NULL)",
            f"INSERT INTO {gt}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {gt})",
            f"CREATE TABLE IF NOT EXISTS {ut} (user_id TEXT PRIMARY KEY, value BIGINT NOT NULL)",
            f"""
                CREATE TABLE IF NOT EXISTS {ij} (
                    job_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    entity_kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    index_kind TEXT NOT NULL,
                    coalesce_key TEXT NOT NULL,
                    op TEXT NOT NULL,
                    status TEXT NOT NULL,
                    lease_until TIMESTAMPTZ NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NULL,
                    payload_json TEXT NULL,
                    next_run_at TIMESTAMPTZ NULL,
                    max_retries INTEGER NOT NULL DEFAULT 10,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """,
            f"CREATE INDEX IF NOT EXISTS idx_index_jobs_status_lease ON {ij}(status, lease_until)",
            f"CREATE INDEX IF NOT EXISTS idx_index_jobs_status_next_run ON {ij}(status, next_run_at)",
            f"CREATE INDEX IF NOT EXISTS idx_index_jobs_entity ON {ij}(entity_kind, entity_id, index_kind)",
            f"CREATE INDEX IF NOT EXISTS idx_index_jobs_namespace ON {ij}(namespace)",
            f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default'",
            f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS coalesce_key TEXT NOT NULL DEFAULT ''",
            f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS next_run_at TIMESTAMPTZ NULL",
            f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS max_retries INTEGER NOT NULL DEFAULT 10",
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_index_jobs_pending_ns_ck ON {ij}(namespace, coalesce_key) WHERE status='PENDING'",
            f"""
                CREATE TABLE IF NOT EXISTS {ias} (
                    namespace TEXT NOT NULL DEFAULT 'default',
                    coalesce_key TEXT NOT NULL,
                    applied_fingerprint TEXT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_job_id TEXT NULL,
                    PRIMARY KEY(namespace, coalesce_key)
                )
            """,
            f"CREATE INDEX IF NOT EXISTS idx_index_applied_state_key ON {ias}(coalesce_key)",
            f"""
                CREATE TABLE IF NOT EXISTS {plm} (
                    message_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    purpose TEXT NOT NULL DEFAULT 'user_visible',
                    inbox_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    msg_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    seq BIGINT NOT NULL,
                    conversation_seq BIGINT NOT NULL,
                    claimed_by TEXT NULL,
                    lease_until TIMESTAMPTZ NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at BIGINT NOT NULL,
                    available_at BIGINT NOT NULL,
                    run_id TEXT NULL,
                    step_id TEXT NULL,
                    correlation_id TEXT NULL,
                    payload_json TEXT NULL,
                    error_json TEXT NULL,
                    prev_message_id TEXT NULL,
                    next_message_id TEXT NULL,
                    inbox_tail_message_id TEXT NULL,
                    conversation_tail_message_id TEXT NULL
                )
            """,
            f"ALTER TABLE {plm} ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'user_visible'",
            f"CREATE INDEX IF NOT EXISTS idx_lane_messages_namespace_inbox_seq ON {plm}(namespace, inbox_id, seq)",
            f"CREATE INDEX IF NOT EXISTS idx_lane_messages_claim ON {plm}(namespace, inbox_id, status, available_at, lease_until)",
            f"CREATE INDEX IF NOT EXISTS idx_lane_messages_conversation_seq ON {plm}(namespace, conversation_id, conversation_seq)",
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.namespace_seq (
                namespace TEXT PRIMARY KEY,
                next_seq  BIGINT NOT NULL
            )
            """,
            f"""
            INSERT INTO {schema}.namespace_seq(namespace, next_seq)
            VALUES ('default', 1)
            ON CONFLICT(namespace) DO NOTHING
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.entity_events (
                namespace    TEXT NOT NULL DEFAULT 'default',
                seq          BIGINT NOT NULL,
                event_id     TEXT NOT NULL,
                entity_kind  TEXT NOT NULL,
                entity_id    TEXT NOT NULL,
                op           TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(namespace, seq),
                UNIQUE(event_id)
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_entity_events_aggregate
            ON {schema}.entity_events(namespace, entity_kind, entity_id, seq)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.replay_cursors (
                namespace  TEXT NOT NULL DEFAULT 'default',
                consumer   TEXT NOT NULL,
                last_seq   BIGINT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(namespace, consumer)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.named_projections (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                last_authoritative_seq BIGINT NOT NULL,
                last_materialized_seq BIGINT NOT NULL,
                projection_schema_version INTEGER NOT NULL,
                materialization_status TEXT NOT NULL,
                updated_at_ms BIGINT NOT NULL,
                PRIMARY KEY(namespace, key)
            )
            """,
            f"""
            CREATE INDEX IF NOT EXISTS idx_named_projections_namespace
            ON {schema}.named_projections(namespace, updated_at_ms)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.workflow_design_snapshots (
                workflow_id TEXT NOT NULL,
                version BIGINT NOT NULL,
                seq BIGINT NOT NULL,
                payload_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at_ms BIGINT NOT NULL,
                PRIMARY KEY(workflow_id, version)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.workflow_design_version_deltas (
                workflow_id TEXT NOT NULL,
                version BIGINT NOT NULL,
                prev_version BIGINT NOT NULL,
                target_seq BIGINT NOT NULL,
                forward_json TEXT NOT NULL,
                inverse_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at_ms BIGINT NOT NULL,
                PRIMARY KEY(workflow_id, version)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.server_runs (
                run_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                user_id TEXT NULL,
                user_turn_node_id TEXT NULL,
                assistant_turn_node_id TEXT NULL,
                status TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                result_json TEXT NULL,
                error_json TEXT NULL,
                created_at_ms BIGINT NOT NULL,
                updated_at_ms BIGINT NOT NULL,
                started_at_ms BIGINT NULL,
                finished_at_ms BIGINT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.server_run_events (
                seq BIGSERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_ms BIGINT NOT NULL
            )
            """,
            f"CREATE INDEX IF NOT EXISTS idx_server_runs_status ON {schema}.server_runs(status, updated_at_ms)",
            f"CREATE INDEX IF NOT EXISTS idx_server_run_events_run_seq ON {schema}.server_run_events(run_id, seq)",
        ]

    def _run_bootstrap(self, conn: Any) -> None:
        for stmt in self._bootstrap_statements():
            conn.execute(sa.text(stmt))

    def ensure_initialized(self) -> None:
        if self._is_async_engine:
            _run_coro_blocking(self._ensure_initialized_async())
            return
        with self.transaction() as conn:
            self._run_bootstrap(conn)

    async def _ensure_initialized_async(self) -> None:
        async with self.engine.begin() as conn:
            for stmt in self._bootstrap_statements():
                await conn.execute(sa.text(stmt))

    # ----------------------------
    # Transaction helpers
    # ----------------------------
    @contextmanager
    def transaction(self) -> Iterator[sa.Connection | _AsyncConnectionAdapter]:
        existing = get_active_conn()
        if existing is not None:
            if isinstance(existing, _AsyncConnectionAdapter):
                yield existing
            elif isinstance(existing, AsyncConnection):
                runner = _make_runner()
                try:
                    yield _AsyncConnectionAdapter(existing, runner)
                finally:
                    runner.close()
            else:
                yield existing
            return

        if self._is_async_engine:
            runner = _make_runner()
            conn_ctx = self.engine.connect()
            conn = _run_coro_blocking(conn_ctx.__aenter__())
            txn = conn.begin()
            _run_coro_blocking(txn.start())
            adapter = _AsyncConnectionAdapter(conn, runner)
            try:
                with _set_active_conn(adapter):
                    yield adapter
            except BaseException as exc:
                _run_coro_blocking(txn.rollback())
                raise
            else:
                _run_coro_blocking(txn.commit())
            finally:
                _run_coro_blocking(conn.close())
                runner.close()
            return

        with self.engine.begin() as conn:
            with _set_active_conn(conn):
                yield conn

    # ----------------------------
    # Global sequence
    # ----------------------------
    def next_global_seq(self) -> int:
        with self.transaction() as conn:
            gt = f"{self.schema}.global_seq"
            row = conn.execute(
                sa.text(f"UPDATE {gt} SET value = value + 1 RETURNING value")
            ).fetchone()
            if not row:
                conn.execute(
                    sa.text(
                        f"INSERT INTO {gt}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {gt})"
                    )
                )
                row = conn.execute(
                    sa.text(f"UPDATE {gt} SET value = value + 1 RETURNING value")
                ).fetchone()
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
            ut = f"{self.schema}.user_seq"
            row = conn.execute(
                sa.text(
                    f"""
                INSERT INTO {ut}(user_id, value)
                VALUES (:user_id, 1)
                ON CONFLICT(user_id)
                DO UPDATE SET value = {ut}.value + 1
                RETURNING value
                """
                ),
                {"user_id": user_id},
            ).fetchone()
            return int(row[0])

    def next_scoped_seq(self, scope_id: str) -> int:
        return self.next_user_seq(scope_id)

    def current_user_seq(self, user_id: str) -> int:
        ut = f"{self.schema}.user_seq"
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(f"SELECT value FROM {ut} WHERE user_id = :user_id"),
                {"user_id": user_id},
            ).fetchone()
            return int(row[0]) if row else 0

    def current_scoped_seq(self, scope_id: str) -> int:
        return self.current_user_seq(scope_id)

    def set_user_seq(self, user_id: str, value: int) -> None:
        if value < 0:
            raise ValueError("value must be >= 0")
        ut = f"{self.schema}.user_seq"
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                INSERT INTO {ut}(user_id, value)
                VALUES (:user_id, :value)
                ON CONFLICT(user_id)
                DO UPDATE SET value = EXCLUDED.value
                """
                ),
                {"user_id": user_id, "value": int(value)},
            )

    def set_scoped_seq(self, scope_id: str, value: int) -> None:
        self.set_user_seq(scope_id, value)

    # ----------------------------
    # Index jobs
    # ----------------------------

    def enqueue_index_job(
        self,
        *,
        job_id: str,
        namespace: str = "default",
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        payload_json: Optional[str] = None,
        max_retries: int = 10,
    ) -> str:
        """Enqueue durable derived-index work in the Postgres metastore.

        This queue is DB-backed, not an in-memory deque. Pending jobs coalesce by
        (namespace, coalesce_key), so repeated UPSERTs reuse one row while DELETE
        wins over UPSERT. The returned job_id may therefore be an existing pending
        row rather than the caller-provided id.
        """
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f'{self.schema}."{self.index_jobs_table}"'
        )
        coalesce_key = f"{entity_kind}:{entity_id}:{index_kind}"

        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"""
                    SELECT job_id, op
                    FROM {ij}
                    WHERE namespace = :ns AND coalesce_key = :ck AND status='PENDING'
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE
                    """
                ),
                {"ns": namespace, "ck": coalesce_key},
            ).fetchone()

            if row:
                existing_job_id = str(row[0])
                existing_op = str(row[1] or "")
                next_op = (
                    "DELETE" if (op == "DELETE" or existing_op == "DELETE") else op
                )
                conn.execute(
                    sa.text(
                        f"""
                        UPDATE {ij}
                        SET op=:op, payload_json=:payload_json, updated_at=NOW()
                        WHERE job_id=:job_id
                        """
                    ),
                    {
                        "op": next_op,
                        "payload_json": payload_json,
                        "job_id": existing_job_id,
                    },
                )
                return existing_job_id

            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {ij}(
                        job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op,
                        status, lease_until, next_run_at, max_retries, retry_count, last_error, payload_json, created_at, updated_at
                    )
                    VALUES (:job_id, :ns, :entity_kind, :entity_id, :index_kind, :ck, :op,
                            'PENDING', NULL, NULL, :max_retries, 0, NULL, :payload_json, NOW(), NOW())
                    ON CONFLICT (job_id) DO NOTHING
                    """
                ),
                {
                    "job_id": job_id,
                    "ns": namespace,
                    "entity_kind": entity_kind,
                    "entity_id": entity_id,
                    "index_kind": index_kind,
                    "ck": coalesce_key,
                    "op": op,
                    "payload_json": payload_json,
                    "max_retries": int(max_retries),
                },
            )
            return job_id

    def claim_index_jobs(
        self,
        *,
        limit: int = 50,
        lease_seconds: int = 60,
        namespace: Optional[str] = "default",
    ) -> List[IndexJob]:
        """Lease runnable jobs from the Postgres-backed queue.

        Eligibility is decided in SQL: pending jobs whose delay has elapsed plus
        doing jobs whose lease expired. Ordering is created_at ASC, namespace
        scoping happens at claim time, and FOR UPDATE SKIP LOCKED prevents workers
        from claiming the same row concurrently. FAILED rows are terminal.
        """
        if limit <= 0:
            return []
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f'{self.schema}."{self.index_jobs_table}"'
        )
        namespace_sql = ""
        params: Dict[str, Any] = {
            "limit": int(limit),
            "lease_seconds": int(lease_seconds),
        }
        if namespace is not None:
            namespace_sql = "AND namespace = :namespace"
            params["namespace"] = namespace
        with self.transaction() as conn:
            res = conn.execute(
                sa.text(
                    f"""
                    WITH candidates AS (
                        SELECT job_id
                        FROM {ij}
                        WHERE (
                            status = 'PENDING' AND (next_run_at IS NULL OR next_run_at <= NOW())
                            OR (status='DOING' AND lease_until IS NOT NULL AND lease_until < NOW())
                        )
                        {namespace_sql}
                        ORDER BY created_at ASC
                        LIMIT :limit
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE {ij} j
                    SET status='DOING',
                        lease_until = NOW() + (:lease_seconds || ' seconds')::interval,
                        updated_at = NOW()
                    FROM candidates c
                    WHERE j.job_id = c.job_id
                    RETURNING j.job_id, j.namespace, j.entity_kind, j.entity_id, j.index_kind, j.coalesce_key, j.op, j.status,
                              j.lease_until, j.next_run_at, j.max_retries, j.retry_count, j.last_error, j.payload_json
                    """
                ),
                params,
            )
            rows = res.mappings().all()

        out: List[IndexJob] = []
        for r in rows:
            out.append(
                IndexJob(
                    job_id=str(r.get("job_id")),
                    namespace=str(r.get("namespace")),
                    entity_kind=str(r.get("entity_kind")),
                    entity_id=str(r.get("entity_id")),
                    index_kind=str(r.get("index_kind")),
                    coalesce_key=str(r.get("coalesce_key")),
                    op=str(r.get("op")),
                    status=str(r.get("status")),
                    lease_until=(
                        str(r.get("lease_until"))
                        if r.get("lease_until") is not None
                        else None
                    ),
                    next_run_at=(
                        str(r.get("next_run_at"))
                        if r.get("next_run_at") is not None
                        else None
                    ),
                    max_retries=int(r.get("max_retries") or 10),
                    retry_count=int(r.get("retry_count") or 0),
                    last_error=(
                        str(r.get("last_error"))
                        if r.get("last_error") is not None
                        else None
                    ),
                    payload_json=(
                        str(r.get("payload_json"))
                        if r.get("payload_json") is not None
                        else None
                    ),
                )
            )
        return out

    def mark_index_job_done(self, job_id: str) -> None:
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f'{self.schema}."{self.index_jobs_table}"'
        )
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"UPDATE {ij} SET status='DONE', lease_until=NULL, updated_at=NOW() WHERE job_id=:job_id"
                ),
                {"job_id": job_id},
            )

    def mark_index_job_failed(
        self, job_id: str, error: str, *, final: bool = True
    ) -> None:
        """Mark a job failed.

        If final=True, job becomes terminal FAILED (DLQ) and will never be reclaimed.
        If final=False, caller should use bump_retry_and_requeue(...).
        """
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f"{self.schema}.{self.index_jobs_table}"
        )
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {ij}
                    SET status = CASE WHEN :final THEN 'FAILED' ELSE status END,
                        lease_until=NULL,
                        last_error=:err,
                        updated_at=NOW()
                    WHERE job_id=:job_id
                    """
                ),
                {"job_id": job_id, "err": (error or "")[:2000], "final": bool(final)},
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
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f"{self.schema}.{self.index_jobs_table}"
        )
        delay = max(0, int(next_run_at_seconds))
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {ij}
                    SET retry_count = retry_count + 1,
                        last_error = :err,
                        status = CASE
                            WHEN (retry_count + 1) >= max_retries THEN 'FAILED'
                            ELSE 'PENDING'
                        END,
                        lease_until = NULL,
                        next_run_at = CASE
                            WHEN (retry_count + 1) >= max_retries THEN NULL
                            ELSE NOW() + (:delay || ' seconds')::interval
                        END,
                        updated_at = NOW()
                    WHERE job_id=:job_id
                    """
                ),
                {"job_id": job_id, "err": (error or "")[:2000], "delay": delay},
            )

    def list_index_jobs(
        self,
        *,
        namespace: Optional[str] = "default",
        status: Optional[str] = None,
        entity_kind: Optional[str] = None,
        entity_id: Optional[str] = None,
        index_kind: Optional[str] = None,
        limit: int = 1000,
    ) -> List[IndexJob]:
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f'{self.schema}."{self.index_jobs_table}"'
        )
        where: List[str] = []
        params: Dict[str, Any] = {"limit": int(limit)}
        if namespace is not None:
            where.append("namespace = :namespace")
            params["namespace"] = namespace
        if status:
            where.append("status = :status")
            params["status"] = status
        if entity_kind:
            where.append("entity_kind = :entity_kind")
            params["entity_kind"] = entity_kind
        if entity_id:
            where.append("entity_id = :entity_id")
            params["entity_id"] = entity_id
        if index_kind:
            where.append("index_kind = :index_kind")
            params["index_kind"] = index_kind
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = sa.text(
            f"""
            SELECT job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op, status,
                   lease_until, next_run_at, max_retries, retry_count, last_error, payload_json
            FROM {ij}
            {where_sql}
            ORDER BY created_at ASC
            LIMIT :limit
            """
        )
        with self.transaction() as conn:
            rows = conn.execute(sql, params).mappings().all()
        out: List[IndexJob] = []
        for r in rows:
            out.append(
                IndexJob(
                    job_id=str(r.get("job_id")),
                    namespace=str(r.get("namespace")),
                    entity_kind=str(r.get("entity_kind")),
                    entity_id=str(r.get("entity_id")),
                    index_kind=str(r.get("index_kind")),
                    coalesce_key=str(r.get("coalesce_key")),
                    op=str(r.get("op")),
                    status=str(r.get("status")),
                    lease_until=(
                        str(r.get("lease_until"))
                        if r.get("lease_until") is not None
                        else None
                    ),
                    next_run_at=(
                        str(r.get("next_run_at"))
                        if r.get("next_run_at") is not None
                        else None
                    ),
                    max_retries=int(r.get("max_retries") or 10),
                    retry_count=int(r.get("retry_count") or 0),
                    last_error=(
                        str(r.get("last_error"))
                        if r.get("last_error") is not None
                        else None
                    ),
                    payload_json=(
                        str(r.get("payload_json"))
                        if r.get("payload_json") is not None
                        else None
                    ),
                )
            )
        return out

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
        table = f"{self.schema}.projected_lane_messages"
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {table}(
                        message_id, namespace, purpose, inbox_id, conversation_id,
                        recipient_id, sender_id, msg_type, status,
                        seq, conversation_seq, claimed_by, lease_until,
                        retry_count, created_at, available_at, run_id,
                        step_id, correlation_id, payload_json, error_json,
                        prev_message_id, next_message_id,
                        inbox_tail_message_id, conversation_tail_message_id
                    )
                    SELECT
                        :message_id, :namespace, :purpose, :inbox_id, :conversation_id,
                        :recipient_id, :sender_id, :msg_type, :status,
                        COALESCE((
                            SELECT MAX(seq) + 1 FROM {table}
                            WHERE namespace = :namespace AND inbox_id = :inbox_id
                        ), 1),
                        COALESCE((
                            SELECT MAX(conversation_seq) + 1 FROM {table}
                            WHERE namespace = :namespace AND conversation_id = :conversation_id
                        ), 1),
                        NULL, NULL, 0, :created_at, :available_at, :run_id,
                        :step_id, :correlation_id, :payload_json, :error_json,
                        (
                            SELECT message_id FROM {table}
                            WHERE namespace = :namespace AND inbox_id = :inbox_id
                            ORDER BY seq DESC, created_at DESC
                            LIMIT 1
                        ),
                        NULL, :message_id, :message_id
                    ON CONFLICT (message_id) DO NOTHING
                    """
                ),
                {
                        "message_id": str(message_id),
                        "namespace": str(namespace),
                        "purpose": str(purpose or "user_visible"),
                        "inbox_id": str(inbox_id),
                    "conversation_id": str(conversation_id),
                    "recipient_id": str(recipient_id),
                    "sender_id": str(sender_id),
                    "msg_type": str(msg_type),
                    "status": str(status),
                    "created_at": int(created_at),
                    "available_at": int(available_at),
                    "run_id": run_id,
                    "step_id": step_id,
                    "correlation_id": correlation_id,
                    "payload_json": payload_json,
                    "error_json": error_json,
                },
            )

    def update_projected_lane_message_status(
        self,
        *,
        message_id: str,
        status: str,
        error_json: str | None = None,
    ) -> None:
        table = f"{self.schema}.projected_lane_messages"
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {table}
                    SET status = :status,
                        error_json = COALESCE(:error_json, error_json),
                        claimed_by = CASE WHEN :status IN ('completed','failed','cancelled') THEN NULL ELSE claimed_by END,
                        lease_until = CASE WHEN :status IN ('completed','failed','cancelled') THEN NULL ELSE lease_until END
                    WHERE message_id = :message_id
                    """
                ),
                {
                    "message_id": str(message_id),
                    "status": str(status),
                    "error_json": error_json,
                },
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
        table = f"{self.schema}.projected_lane_messages"
        with self.transaction() as conn:
            rows = conn.execute(
                sa.text(
                    f"""
                    WITH picked AS (
                        SELECT message_id
                        FROM {table}
                        WHERE namespace = :namespace
                          AND inbox_id = :inbox_id
                          AND (
                            (status = 'pending' AND available_at <= EXTRACT(EPOCH FROM NOW())::BIGINT)
                            OR
                            (status = 'claimed' AND lease_until IS NOT NULL AND lease_until < NOW())
                          )
                        ORDER BY seq ASC, created_at ASC
                        LIMIT :limit
                        FOR UPDATE
                    )
                    UPDATE {table} t
                    SET status = 'claimed',
                        claimed_by = :claimed_by,
                        lease_until = NOW() + (:lease_seconds || ' seconds')::interval
                    FROM picked
                    WHERE t.message_id = picked.message_id
                    RETURNING t.message_id, t.namespace, t.purpose, t.inbox_id, t.conversation_id, t.recipient_id, t.sender_id,
                              t.msg_type, t.status, t.seq, t.conversation_seq, t.claimed_by, t.lease_until,
                              t.retry_count, t.created_at, t.available_at, t.run_id, t.step_id, t.correlation_id,
                              t.payload_json, t.error_json,
                              t.prev_message_id, t.next_message_id,
                              t.inbox_tail_message_id, t.conversation_tail_message_id
                    """
                ),
                {
                    "namespace": str(namespace),
                    "inbox_id": str(inbox_id),
                    "claimed_by": str(claimed_by),
                    "lease_seconds": int(lease_seconds),
                    "limit": int(limit),
                },
            ).mappings().all()
        return [
            ProjectedLaneMessageRow(
                message_id=str(r.get("message_id")),
                namespace=str(r.get("namespace")),
                purpose=str(r.get("purpose") or "user_visible"),
                inbox_id=str(r.get("inbox_id")),
                conversation_id=str(r.get("conversation_id")),
                recipient_id=str(r.get("recipient_id")),
                sender_id=str(r.get("sender_id")),
                msg_type=str(r.get("msg_type")),
                status=str(r.get("status")),
                seq=int(r.get("seq") or 0),
                conversation_seq=int(r.get("conversation_seq") or 0),
                claimed_by=(str(r.get("claimed_by")) if r.get("claimed_by") is not None else None),
                lease_until=None,
                retry_count=int(r.get("retry_count") or 0),
                created_at=int(r.get("created_at") or 0),
                available_at=int(r.get("available_at") or 0),
                run_id=(str(r.get("run_id")) if r.get("run_id") is not None else None),
                step_id=(str(r.get("step_id")) if r.get("step_id") is not None else None),
                correlation_id=(str(r.get("correlation_id")) if r.get("correlation_id") is not None else None),
                payload_json=(str(r.get("payload_json")) if r.get("payload_json") is not None else None),
                error_json=(str(r.get("error_json")) if r.get("error_json") is not None else None),
                prev_message_id=(str(r.get("prev_message_id")) if r.get("prev_message_id") is not None else None),
                next_message_id=(str(r.get("next_message_id")) if r.get("next_message_id") is not None else None),
                inbox_tail_message_id=(str(r.get("inbox_tail_message_id")) if r.get("inbox_tail_message_id") is not None else None),
                conversation_tail_message_id=(str(r.get("conversation_tail_message_id")) if r.get("conversation_tail_message_id") is not None else None),
            )
            for r in rows
        ]

    def ack_projected_lane_message(self, *, message_id: str, claimed_by: str) -> None:
        table = f"{self.schema}.projected_lane_messages"
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {table}
                    SET status = 'completed', claimed_by = NULL, lease_until = NULL
                    WHERE message_id = :message_id
                      AND (claimed_by IS NULL OR claimed_by = :claimed_by)
                    """
                ),
                {"message_id": str(message_id), "claimed_by": str(claimed_by)},
            )

    def requeue_projected_lane_message(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error_json: str | None = None,
        delay_seconds: int = 0,
    ) -> None:
        table = f"{self.schema}.projected_lane_messages"
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {table}
                    SET status = 'pending',
                        claimed_by = NULL,
                        lease_until = NULL,
                        retry_count = retry_count + 1,
                        available_at = EXTRACT(EPOCH FROM (NOW() + (:delay_seconds || ' seconds')::interval))::BIGINT,
                        error_json = COALESCE(:error_json, error_json)
                    WHERE message_id = :message_id
                      AND (claimed_by IS NULL OR claimed_by = :claimed_by)
                    """
                ),
                {
                    "message_id": str(message_id),
                    "claimed_by": str(claimed_by),
                    "delay_seconds": int(delay_seconds),
                    "error_json": error_json,
                },
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
        table = f"{self.schema}.projected_lane_messages"
        where = ["namespace = :namespace"]
        params: Dict[str, Any] = {"namespace": str(namespace), "limit": int(limit)}
        if purpose is not None:
            where.append("purpose = :purpose")
            params["purpose"] = str(purpose)
        if inbox_id is not None:
            where.append("inbox_id = :inbox_id")
            params["inbox_id"] = str(inbox_id)
        if status is not None:
            where.append("status = :status")
            params["status"] = str(status)
        with self.transaction() as conn:
            rows = conn.execute(
                sa.text(
                    f"""
                    SELECT message_id, namespace, purpose, inbox_id, conversation_id, recipient_id, sender_id,
                           msg_type, status, seq, conversation_seq, claimed_by, lease_until,
                           retry_count, created_at, available_at, run_id, step_id, correlation_id,
                           payload_json, error_json,
                           prev_message_id, next_message_id,
                           inbox_tail_message_id, conversation_tail_message_id
                    FROM {table}
                    WHERE {' AND '.join(where)}
                    ORDER BY inbox_id ASC, seq ASC, created_at ASC
                    LIMIT :limit
                    """
                ),
                params,
            ).mappings().all()
        return [
            ProjectedLaneMessageRow(
                message_id=str(r.get("message_id")),
                namespace=str(r.get("namespace")),
                purpose=str(r.get("purpose") or "user_visible"),
                inbox_id=str(r.get("inbox_id")),
                conversation_id=str(r.get("conversation_id")),
                recipient_id=str(r.get("recipient_id")),
                sender_id=str(r.get("sender_id")),
                msg_type=str(r.get("msg_type")),
                status=str(r.get("status")),
                seq=int(r.get("seq") or 0),
                conversation_seq=int(r.get("conversation_seq") or 0),
                claimed_by=(str(r.get("claimed_by")) if r.get("claimed_by") is not None else None),
                lease_until=None,
                retry_count=int(r.get("retry_count") or 0),
                created_at=int(r.get("created_at") or 0),
                available_at=int(r.get("available_at") or 0),
                run_id=(str(r.get("run_id")) if r.get("run_id") is not None else None),
                step_id=(str(r.get("step_id")) if r.get("step_id") is not None else None),
                correlation_id=(str(r.get("correlation_id")) if r.get("correlation_id") is not None else None),
                payload_json=(str(r.get("payload_json")) if r.get("payload_json") is not None else None),
                error_json=(str(r.get("error_json")) if r.get("error_json") is not None else None),
                prev_message_id=(str(r.get("prev_message_id")) if r.get("prev_message_id") is not None else None),
                next_message_id=(str(r.get("next_message_id")) if r.get("next_message_id") is not None else None),
                inbox_tail_message_id=(str(r.get("inbox_tail_message_id")) if r.get("inbox_tail_message_id") is not None else None),
                conversation_tail_message_id=(str(r.get("conversation_tail_message_id")) if r.get("conversation_tail_message_id") is not None else None),
            )
            for r in rows
        ]

    # ----------------------------
    # Phase 2: applied fingerprints (derived index status)
    # ----------------------------

    def get_index_applied_fingerprint(
        self, *, namespace: str = "default", coalesce_key: str
    ) -> Optional[str]:
        ias = f"{self.schema}.index_applied_state"
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"SELECT applied_fingerprint FROM {ias} WHERE namespace=:ns AND coalesce_key=:ck"
                ),
                {"ns": namespace, "ck": coalesce_key},
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
        ias = f"{self.schema}.index_applied_state"
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {ias}(namespace, coalesce_key, applied_fingerprint, applied_at, last_job_id)
                    VALUES (:ns, :ck, :fp, NOW(), :jid)
                    ON CONFLICT(namespace, coalesce_key)
                    DO UPDATE SET applied_fingerprint = EXCLUDED.applied_fingerprint,
                                  applied_at = NOW(),
                                  last_job_id = EXCLUDED.last_job_id
                    """
                ),
                {
                    "ns": namespace,
                    "ck": coalesce_key,
                    "fp": applied_fingerprint,
                    "jid": last_job_id,
                },
            )

    # ------------------------------------------------------------------
    # Phase 2b: event log foundation
    # ------------------------------------------------------------------

    def alloc_event_seq(self, namespace: str = "default") -> int:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(f"""
                    UPDATE {schema}.namespace_seq
                    SET next_seq = next_seq + 1
                    WHERE namespace = :ns
                    RETURNING next_seq - 1
                """),
                {"ns": namespace},
            ).fetchone()
            if row is not None:
                return int(row[0])

            conn.execute(
                sa.text(
                    f"INSERT INTO {schema}.namespace_seq(namespace, next_seq) VALUES (:ns, 2)"
                ),
                {"ns": namespace},
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
        schema = self.schema
        seq = self.alloc_event_seq(namespace)
        with self.transaction() as conn:
            conn.execute(
                sa.text(f"""
                    INSERT INTO {schema}.entity_events(
                        namespace, seq, event_id, entity_kind, entity_id, op, payload_json
                    )
                    VALUES (:ns, :seq, :eid, :ek, :id, :op, :payload)
                """),
                {
                    "ns": namespace,
                    "seq": seq,
                    "eid": event_id,
                    "ek": entity_kind,
                    "id": entity_id,
                    "op": op,
                    "payload": payload_json,
                },
            )
        return seq

    def iter_entity_events(
        self,
        *,
        namespace: str = "default",
        from_seq: int = 1,
        to_seq: int | None = None,
    ):
        schema = self.schema
        with self.transaction() as conn:
            if to_seq is None:
                rows = conn.execute(
                    sa.text(f"""
                        SELECT seq, entity_kind, entity_id, op, payload_json
                        FROM {schema}.entity_events
                        WHERE namespace=:ns AND seq >= :from_seq
                        ORDER BY seq ASC
                    """),
                    {"ns": namespace, "from_seq": int(from_seq)},
                )
            else:
                rows = conn.execute(
                    sa.text(f"""
                        SELECT seq, entity_kind, entity_id, op, payload_json
                        FROM {schema}.entity_events
                        WHERE namespace=:ns AND seq BETWEEN :from_seq AND :to_seq
                        ORDER BY seq ASC
                    """),
                    {"ns": namespace, "from_seq": int(from_seq), "to_seq": int(to_seq)},
                )
            yield from rows

    def prune_entity_events_after(
        self, *, namespace: str = "default", to_seq: int
    ) -> int:
        schema = self.schema
        with self.transaction() as conn:
            res = conn.execute(
                sa.text(
                    f"""
                    DELETE FROM {schema}.entity_events
                    WHERE namespace=:ns AND seq > :to_seq
                    """
                ),
                {"ns": namespace, "to_seq": int(to_seq)},
            )
            return int(res.rowcount or 0)

    def cursor_get(self, *, namespace: str, consumer: str) -> int:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(f"""
                    SELECT last_seq FROM {schema}.replay_cursors
                    WHERE namespace=:ns AND consumer=:c
                """),
                {"ns": namespace, "c": consumer},
            ).fetchone()
        return int(row[0]) if row else 0

    def cursor_set(self, *, namespace: str, consumer: str, last_seq: int) -> None:
        schema = self.schema
        with self.transaction() as conn:
            conn.execute(
                sa.text(f"""
                    INSERT INTO {schema}.replay_cursors(namespace, consumer, last_seq, updated_at)
                    VALUES (:ns, :c, :s, NOW())
                    ON CONFLICT(namespace, consumer)
                    DO UPDATE SET last_seq=EXCLUDED.last_seq, updated_at=NOW()
                """),
                {"ns": namespace, "c": consumer, "s": int(last_seq)},
            )

    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"SELECT COALESCE(MAX(seq), 0) FROM {schema}.entity_events WHERE namespace = :ns"
                ),
                {"ns": namespace},
            ).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _decode_named_projection_payload(raw_payload: Any) -> dict[str, Any]:
        payload = json.loads(str(raw_payload)) if raw_payload is not None else {}
        if not isinstance(payload, dict):
            raise ValueError("named projection payload must deserialize to a dict")
        return payload

    def get_named_projection(self, namespace: str, key: str) -> Optional[dict[str, Any]]:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"""
                    SELECT namespace, key, payload_json,
                           last_authoritative_seq, last_materialized_seq,
                           projection_schema_version, materialization_status, updated_at_ms
                    FROM {schema}.named_projections
                    WHERE namespace = :namespace AND key = :key
                    """
                ),
                {"namespace": str(namespace), "key": str(key)},
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
        schema = self.schema
        updated_at_ms = int(time.time() * 1000)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {schema}.named_projections(
                        namespace, key, payload_json,
                        last_authoritative_seq, last_materialized_seq,
                        projection_schema_version, materialization_status, updated_at_ms
                    ) VALUES (
                        :namespace, :key, :payload_json,
                        :last_authoritative_seq, :last_materialized_seq,
                        :projection_schema_version, :materialization_status, :updated_at_ms
                    )
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        payload_json = EXCLUDED.payload_json,
                        last_authoritative_seq = EXCLUDED.last_authoritative_seq,
                        last_materialized_seq = EXCLUDED.last_materialized_seq,
                        projection_schema_version = EXCLUDED.projection_schema_version,
                        materialization_status = EXCLUDED.materialization_status,
                        updated_at_ms = EXCLUDED.updated_at_ms
                    """
                ),
                {
                    "namespace": str(namespace),
                    "key": str(key),
                    "payload_json": payload_json,
                    "last_authoritative_seq": int(last_authoritative_seq),
                    "last_materialized_seq": int(last_materialized_seq),
                    "projection_schema_version": int(projection_schema_version),
                    "materialization_status": str(materialization_status),
                    "updated_at_ms": updated_at_ms,
                },
            )

    def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        schema = self.schema
        with self.transaction() as conn:
            rows = conn.execute(
                sa.text(
                    f"""
                    SELECT namespace, key, payload_json,
                           last_authoritative_seq, last_materialized_seq,
                           projection_schema_version, materialization_status, updated_at_ms
                    FROM {schema}.named_projections
                    WHERE namespace = :namespace
                    ORDER BY key ASC
                    """
                ),
                {"namespace": str(namespace)},
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
        schema = self.schema
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    DELETE FROM {schema}.named_projections
                    WHERE namespace = :namespace AND key = :key
                    """
                ),
                {"namespace": str(namespace), "key": str(key)},
            )

    def clear_projection_namespace(self, namespace: str) -> None:
        schema = self.schema
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    DELETE FROM {schema}.named_projections
                    WHERE namespace = :namespace
                    """
                ),
                {"namespace": str(namespace)},
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
        schema = self.schema
        now_ms = int(time.time() * 1000)
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {schema}.workflow_design_snapshots(
                        workflow_id, version, seq, payload_json, schema_version, created_at_ms
                    ) VALUES (
                        :workflow_id, :version, :seq, :payload_json, :schema_version, :created_at_ms
                    )
                    ON CONFLICT(workflow_id, version) DO UPDATE SET
                        seq = EXCLUDED.seq,
                        payload_json = EXCLUDED.payload_json,
                        schema_version = EXCLUDED.schema_version,
                        created_at_ms = EXCLUDED.created_at_ms
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "version": int(version),
                    "seq": int(seq),
                    "payload_json": payload_json,
                    "schema_version": int(schema_version),
                    "created_at_ms": now_ms,
                },
            )

    def get_workflow_design_snapshot(
        self,
        *,
        workflow_id: str,
        max_version: int,
        schema_version: int,
    ) -> Optional[dict[str, Any]]:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"""
                    SELECT workflow_id, version, seq, payload_json, schema_version, created_at_ms
                    FROM {schema}.workflow_design_snapshots
                    WHERE workflow_id = :workflow_id
                      AND version <= :max_version
                      AND schema_version = :schema_version
                    ORDER BY version DESC
                    LIMIT 1
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "max_version": int(max_version),
                    "schema_version": int(schema_version),
                },
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
        schema = self.schema
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_snapshots WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
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
        schema = self.schema
        now_ms = int(time.time() * 1000)
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {schema}.workflow_design_version_deltas(
                        workflow_id, version, prev_version, target_seq,
                        forward_json, inverse_json, schema_version, created_at_ms
                    ) VALUES (
                        :workflow_id, :version, :prev_version, :target_seq,
                        :forward_json, :inverse_json, :schema_version, :created_at_ms
                    )
                    ON CONFLICT(workflow_id, version) DO UPDATE SET
                        prev_version = EXCLUDED.prev_version,
                        target_seq = EXCLUDED.target_seq,
                        forward_json = EXCLUDED.forward_json,
                        inverse_json = EXCLUDED.inverse_json,
                        schema_version = EXCLUDED.schema_version,
                        created_at_ms = EXCLUDED.created_at_ms
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "version": int(version),
                    "prev_version": int(prev_version),
                    "target_seq": int(target_seq),
                    "forward_json": forward_json,
                    "inverse_json": inverse_json,
                    "schema_version": int(schema_version),
                    "created_at_ms": now_ms,
                },
            )

    def get_workflow_design_delta(
        self,
        *,
        workflow_id: str,
        version: int,
        schema_version: int,
    ) -> Optional[dict[str, Any]]:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"""
                    SELECT workflow_id, version, prev_version, target_seq,
                           forward_json, inverse_json, schema_version, created_at_ms
                    FROM {schema}.workflow_design_version_deltas
                    WHERE workflow_id = :workflow_id
                      AND version = :version
                      AND schema_version = :schema_version
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "version": int(version),
                    "schema_version": int(schema_version),
                },
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
        schema = self.schema
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_version_deltas WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
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
        schema = self.schema
        now_ms = int(time.time() * 1000)
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {schema}.server_runs(
                        run_id, conversation_id, workflow_id, user_id,
                        user_turn_node_id, assistant_turn_node_id, status,
                        cancel_requested, result_json, error_json,
                        created_at_ms, updated_at_ms, started_at_ms, finished_at_ms
                    ) VALUES (
                        :run_id, :conversation_id, :workflow_id, :user_id,
                        :user_turn_node_id, NULL, :status,
                        0, NULL, NULL,
                        :created_at_ms, :updated_at_ms, NULL, NULL
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "workflow_id": workflow_id,
                    "user_id": user_id,
                    "user_turn_node_id": user_turn_node_id,
                    "status": status,
                    "created_at_ms": now_ms,
                    "updated_at_ms": now_ms,
                },
            )

    def get_server_run(self, run_id: str) -> Optional[dict[str, Any]]:
        schema = self.schema
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"""
                    SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id,
                           assistant_turn_node_id, status, cancel_requested, result_json,
                           error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms
                    FROM {schema}.server_runs
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
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
        schema = self.schema
        clauses = []
        params: dict[str, Any] = {"limit": int(limit)}
        if status is not None:
            clauses.append("status = :status")
            params["status"] = str(status)
        if workflow_id is not None:
            clauses.append("workflow_id = :workflow_id")
            params["workflow_id"] = str(workflow_id)
        if conversation_id is not None:
            clauses.append("conversation_id = :conversation_id")
            params["conversation_id"] = str(conversation_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.transaction() as conn:
            rows = conn.execute(
                sa.text(
                    f"""
                    SELECT run_id, conversation_id, workflow_id, user_id, user_turn_node_id,
                           assistant_turn_node_id, status, cancel_requested, result_json,
                           error_json, created_at_ms, updated_at_ms, started_at_ms, finished_at_ms
                    FROM {schema}.server_runs
                    {where}
                    ORDER BY created_at_ms DESC, run_id DESC
                    LIMIT :limit
                    """
                ),
                params,
            ).fetchall()
        out: list[dict[str, Any]] = []
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
        schema = self.schema
        with self.transaction() as conn:
            rows = conn.execute(
                sa.text(
                    f"""
                    SELECT seq, run_id, event_type, payload_json, created_at_ms
                    FROM {schema}.server_run_events
                    WHERE run_id = :run_id AND seq > :after_seq
                    ORDER BY seq ASC
                    LIMIT :limit
                    """
                ),
                {"run_id": run_id, "after_seq": int(after_seq), "limit": int(limit)},
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
        self, run_id: str, event_type: str, payload_json: str
    ) -> dict[str, Any]:
        schema = self.schema
        now_ms = int(time.time() * 1000)
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {schema}.server_run_events(run_id, event_type, payload_json, created_at_ms)
                    VALUES (:run_id, :event_type, :payload_json, :created_at_ms)
                    RETURNING seq
                    """
                ),
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "payload_json": payload_json,
                    "created_at_ms": now_ms,
                },
            ).fetchone()
            seq = int(row[0])
        return {
            "seq": seq,
            "run_id": run_id,
            "event_type": event_type,
            "payload": self._decode_run_json(payload_json) or {},
            "created_at_ms": now_ms,
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
        schema = self.schema
        now_ms = int(time.time() * 1000)
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {schema}.server_runs
                    SET status = :status,
                        assistant_turn_node_id = :assistant_turn_node_id,
                        result_json = :result_json,
                        error_json = :error_json,
                        started_at_ms = :started_at_ms,
                        finished_at_ms = :finished_at_ms,
                        cancel_requested = COALESCE(:cancel_requested, cancel_requested),
                        updated_at_ms = :updated_at_ms
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "status": status,
                    "assistant_turn_node_id": assistant_turn_node_id,
                    "result_json": result_json,
                    "error_json": error_json,
                    "started_at_ms": started_at_ms,
                    "finished_at_ms": finished_at_ms,
                    "cancel_requested": (
                        None
                        if cancel_requested is None
                        else int(bool(cancel_requested))
                    ),
                    "updated_at_ms": now_ms,
                },
            )

    def request_server_run_cancel(self, *, run_id: str) -> None:
        schema = self.schema
        now_ms = int(time.time() * 1000)
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"""
                    UPDATE {schema}.server_runs
                    SET cancel_requested = 1,
                        status = CASE
                            WHEN status IN ('cancelled', 'failed', 'succeeded') THEN status
                            ELSE 'cancelling'
                        END,
                        updated_at_ms = :updated_at_ms
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "updated_at_ms": now_ms},
            )
