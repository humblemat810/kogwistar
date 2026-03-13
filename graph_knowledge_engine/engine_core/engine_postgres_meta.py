from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
import time
from typing import Any, Dict, Iterator, List, Optional

import sqlalchemy as sa

from .postgres_backend import get_active_conn, _set_active_conn


_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
class EnginePostgresMetaStore:
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

    engine: sa.Engine
    schema: str = "public"
    global_table: str = "global_seq"
    user_table: str = "user_seq"
    index_jobs_table: str = "index_jobs"

    def __post_init__(self) -> None:
        if not _SCHEMA_RE.match(self.schema):
            raise ValueError(f"invalid schema: {self.schema!r}")

    # ----------------------------
    # Initialization
    # ----------------------------
    def ensure_initialized(self) -> None:
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

        with self.transaction() as conn:
            conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            conn.execute(
                sa.text(f"CREATE TABLE IF NOT EXISTS {gt} (value BIGINT NOT NULL)")
            )
            conn.execute(
                sa.text(
                    f"INSERT INTO {gt}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {gt})"
                )
            )
            conn.execute(
                sa.text(
                    f"CREATE TABLE IF NOT EXISTS {ut} (user_id TEXT PRIMARY KEY, value BIGINT NOT NULL)"
                )
            )

            ij = (
                f"{schema}.{self.index_jobs_table}"
                if self.index_jobs_table == "index_jobs"
                else f'{schema}."{self.index_jobs_table}"'
            )

            conn.execute(
                sa.text(f"""
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
            """)
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_index_jobs_status_lease ON {ij}(status, lease_until)"
                )
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_index_jobs_status_next_run ON {ij}(status, next_run_at)"
                )
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_index_jobs_entity ON {ij}(entity_kind, entity_id, index_kind)"
                )
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_index_jobs_namespace ON {ij}(namespace)"
                )
            )

            # Legacy safety: ensure columns exist (no-op on fresh schemas)
            conn.execute(
                sa.text(
                    f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default'"
                )
            )
            conn.execute(
                sa.text(
                    f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS coalesce_key TEXT NOT NULL DEFAULT ''"
                )
            )

            # Phase 5: scheduling / DLQ controls
            conn.execute(
                sa.text(
                    f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS next_run_at TIMESTAMPTZ NULL"
                )
            )
            conn.execute(
                sa.text(
                    f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS max_retries INTEGER NOT NULL DEFAULT 10"
                )
            )

            # Phase 2: coalescing constraint (namespaced)
            conn.execute(
                sa.text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS uq_index_jobs_pending_ns_ck ON {ij}(namespace, coalesce_key) WHERE status='PENDING'"
                )
            )

            # Phase 2: applied fingerprints (namespaced)
            ias = f"{schema}.index_applied_state"
            conn.execute(
                sa.text(f"""
                CREATE TABLE IF NOT EXISTS {ias} (
                    namespace TEXT NOT NULL DEFAULT 'default',
                    coalesce_key TEXT NOT NULL,
                    applied_fingerprint TEXT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_job_id TEXT NULL,
                    PRIMARY KEY(namespace, coalesce_key)
                )
            """)
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_index_applied_state_key ON {ias}(coalesce_key)"
                )
            )

            # -------------------------------
            # Phase 2b: event log foundation
            # -------------------------------

            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.namespace_seq (
                namespace TEXT PRIMARY KEY,
                next_seq  BIGINT NOT NULL
            )
            """)
            )
            conn.execute(
                sa.text(f"""
            INSERT INTO {schema}.namespace_seq(namespace, next_seq)
            VALUES ('default', 1)
            ON CONFLICT(namespace) DO NOTHING
            """)
            )

            conn.execute(
                sa.text(f"""
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
            """)
            )
            conn.execute(
                sa.text(f"""
            CREATE INDEX IF NOT EXISTS idx_entity_events_aggregate
            ON {schema}.entity_events(namespace, entity_kind, entity_id, seq)
            """)
            )

            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.replay_cursors (
                namespace  TEXT NOT NULL DEFAULT 'default',
                consumer   TEXT NOT NULL,
                last_seq   BIGINT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(namespace, consumer)
            )
            """)
            )

            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.workflow_design_projection_head (
                workflow_id TEXT PRIMARY KEY,
                current_version BIGINT NOT NULL,
                active_tip_version BIGINT NOT NULL,
                last_authoritative_seq BIGINT NOT NULL,
                last_materialized_seq BIGINT NOT NULL,
                projection_schema_version INTEGER NOT NULL,
                snapshot_schema_version INTEGER NOT NULL,
                materialization_status TEXT NOT NULL,
                updated_at_ms BIGINT NOT NULL
            )
            """)
            )
            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.workflow_design_projection_versions (
                workflow_id TEXT NOT NULL,
                version BIGINT NOT NULL,
                prev_version BIGINT NOT NULL,
                target_seq BIGINT NOT NULL,
                created_at_ms BIGINT NOT NULL,
                PRIMARY KEY(workflow_id, version)
            )
            """)
            )
            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.workflow_design_projection_dropped_ranges (
                workflow_id TEXT NOT NULL,
                start_seq BIGINT NOT NULL,
                end_seq BIGINT NOT NULL,
                start_version BIGINT NOT NULL,
                end_version BIGINT NOT NULL,
                PRIMARY KEY(workflow_id, start_seq, end_seq)
            )
            """)
            )
            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.workflow_design_snapshots (
                workflow_id TEXT NOT NULL,
                version BIGINT NOT NULL,
                seq BIGINT NOT NULL,
                payload_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at_ms BIGINT NOT NULL,
                PRIMARY KEY(workflow_id, version)
            )
            """)
            )
            conn.execute(
                sa.text(f"""
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
            """)
            )
            conn.execute(
                sa.text(f"""
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
            """)
            )
            conn.execute(
                sa.text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.server_run_events (
                seq BIGSERIAL PRIMARY KEY,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_ms BIGINT NOT NULL
            )
            """)
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_server_runs_status ON {schema}.server_runs(status, updated_at_ms)"
                )
            )
            conn.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS idx_server_run_events_run_seq ON {schema}.server_run_events(run_id, seq)"
                )
            )

    # ----------------------------
    # Transaction helpers
    # ----------------------------
    @contextmanager
    def transaction(self) -> Iterator[sa.Connection]:
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

    def current_user_seq(self, user_id: str) -> int:
        ut = f"{self.schema}.user_seq"
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(f"SELECT value FROM {ut} WHERE user_id = :user_id"),
                {"user_id": user_id},
            ).fetchone()
            return int(row[0]) if row else 0

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
                        AND (:namespace IS NULL OR namespace = :namespace)
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
                {
                    "limit": int(limit),
                    "lease_seconds": int(lease_seconds),
                    "namespace": namespace,
                },
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

    def get_workflow_design_projection(
        self, *, workflow_id: str
    ) -> Optional[dict[str, Any]]:
        schema = self.schema
        with self.transaction() as conn:
            head = conn.execute(
                sa.text(
                    f"""
                    SELECT workflow_id, current_version, active_tip_version,
                           last_authoritative_seq, last_materialized_seq,
                           projection_schema_version, snapshot_schema_version,
                           materialization_status, updated_at_ms
                    FROM {schema}.workflow_design_projection_head
                    WHERE workflow_id = :workflow_id
                    """
                ),
                {"workflow_id": workflow_id},
            ).fetchone()
            if head is None:
                return None
            versions = conn.execute(
                sa.text(
                    f"""
                    SELECT version, prev_version, target_seq, created_at_ms
                    FROM {schema}.workflow_design_projection_versions
                    WHERE workflow_id = :workflow_id
                    ORDER BY version ASC
                    """
                ),
                {"workflow_id": workflow_id},
            ).fetchall()
            dropped = conn.execute(
                sa.text(
                    f"""
                    SELECT start_seq, end_seq, start_version, end_version
                    FROM {schema}.workflow_design_projection_dropped_ranges
                    WHERE workflow_id = :workflow_id
                    ORDER BY start_seq ASC, end_seq ASC
                    """
                ),
                {"workflow_id": workflow_id},
            ).fetchall()
        return {
            "workflow_id": str(head[0]),
            "current_version": int(head[1]),
            "active_tip_version": int(head[2]),
            "last_authoritative_seq": int(head[3]),
            "last_materialized_seq": int(head[4]),
            "projection_schema_version": int(head[5]),
            "snapshot_schema_version": int(head[6]),
            "materialization_status": str(head[7]),
            "updated_at_ms": int(head[8]),
            "versions": [
                {
                    "version": int(row[0]),
                    "prev_version": int(row[1]),
                    "target_seq": int(row[2]),
                    "created_at_ms": int(row[3]),
                }
                for row in versions
            ],
            "dropped_ranges": [
                {
                    "start_seq": int(row[0]),
                    "end_seq": int(row[1]),
                    "start_version": int(row[2]),
                    "end_version": int(row[3]),
                }
                for row in dropped
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
        schema = self.schema
        updated_at_ms = int(head.get("updated_at_ms") or 0)
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_projection_versions WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
            )
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_projection_dropped_ranges WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
            )
            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {schema}.workflow_design_projection_head(
                        workflow_id, current_version, active_tip_version,
                        last_authoritative_seq, last_materialized_seq,
                        projection_schema_version, snapshot_schema_version,
                        materialization_status, updated_at_ms
                    ) VALUES (
                        :workflow_id, :current_version, :active_tip_version,
                        :last_authoritative_seq, :last_materialized_seq,
                        :projection_schema_version, :snapshot_schema_version,
                        :materialization_status, :updated_at_ms
                    )
                    ON CONFLICT(workflow_id) DO UPDATE SET
                        current_version = EXCLUDED.current_version,
                        active_tip_version = EXCLUDED.active_tip_version,
                        last_authoritative_seq = EXCLUDED.last_authoritative_seq,
                        last_materialized_seq = EXCLUDED.last_materialized_seq,
                        projection_schema_version = EXCLUDED.projection_schema_version,
                        snapshot_schema_version = EXCLUDED.snapshot_schema_version,
                        materialization_status = EXCLUDED.materialization_status,
                        updated_at_ms = EXCLUDED.updated_at_ms
                    """
                ),
                {
                    "workflow_id": workflow_id,
                    "current_version": int(head.get("current_version") or 0),
                    "active_tip_version": int(head.get("active_tip_version") or 0),
                    "last_authoritative_seq": int(
                        head.get("last_authoritative_seq") or 0
                    ),
                    "last_materialized_seq": int(
                        head.get("last_materialized_seq") or 0
                    ),
                    "projection_schema_version": int(
                        head.get("projection_schema_version") or 1
                    ),
                    "snapshot_schema_version": int(
                        head.get("snapshot_schema_version") or 1
                    ),
                    "materialization_status": str(
                        head.get("materialization_status") or "ready"
                    ),
                    "updated_at_ms": updated_at_ms,
                },
            )
            for item in versions:
                conn.execute(
                    sa.text(
                        f"""
                        INSERT INTO {schema}.workflow_design_projection_versions(
                            workflow_id, version, prev_version, target_seq, created_at_ms
                        ) VALUES (
                            :workflow_id, :version, :prev_version, :target_seq, :created_at_ms
                        )
                        """
                    ),
                    {
                        "workflow_id": workflow_id,
                        "version": int(item.get("version") or 0),
                        "prev_version": int(item.get("prev_version") or 0),
                        "target_seq": int(item.get("target_seq") or 0),
                        "created_at_ms": int(item.get("created_at_ms") or 0),
                    },
                )
            for item in dropped_ranges:
                conn.execute(
                    sa.text(
                        f"""
                        INSERT INTO {schema}.workflow_design_projection_dropped_ranges(
                            workflow_id, start_seq, end_seq, start_version, end_version
                        ) VALUES (
                            :workflow_id, :start_seq, :end_seq, :start_version, :end_version
                        )
                        """
                    ),
                    {
                        "workflow_id": workflow_id,
                        "start_seq": int(item.get("start_seq") or 0),
                        "end_seq": int(item.get("end_seq") or 0),
                        "start_version": int(item.get("start_version") or 0),
                        "end_version": int(item.get("end_version") or 0),
                    },
                )

    def clear_workflow_design_projection(self, *, workflow_id: str) -> None:
        schema = self.schema
        with self.transaction() as conn:
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_projection_versions WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
            )
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_projection_dropped_ranges WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
            )
            conn.execute(
                sa.text(
                    f"DELETE FROM {schema}.workflow_design_projection_head WHERE workflow_id = :workflow_id"
                ),
                {"workflow_id": workflow_id},
            )

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
