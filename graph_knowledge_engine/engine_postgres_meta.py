from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import re
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
        gt = f'{schema}."{self.global_table}"' if self.global_table != "global_seq" else f"{schema}.global_seq"
        ut = f'{schema}."{self.user_table}"' if self.user_table != "user_seq" else f"{schema}.user_seq"

        with self.transaction() as conn:
            conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            conn.execute(sa.text(f"CREATE TABLE IF NOT EXISTS {gt} (value BIGINT NOT NULL)"))
            conn.execute(sa.text(f"INSERT INTO {gt}(value) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {gt})"))
            conn.execute(sa.text(f"CREATE TABLE IF NOT EXISTS {ut} (user_id TEXT PRIMARY KEY, value BIGINT NOT NULL)"))

            ij = (
                f"{schema}.{self.index_jobs_table}"
                if self.index_jobs_table == "index_jobs"
                else f'{schema}."{self.index_jobs_table}"'
            )

            conn.execute(sa.text(f"""
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
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            conn.execute(sa.text(f"CREATE INDEX IF NOT EXISTS idx_index_jobs_status_lease ON {ij}(status, lease_until)"))
            conn.execute(sa.text(f"CREATE INDEX IF NOT EXISTS idx_index_jobs_entity ON {ij}(entity_kind, entity_id, index_kind)"))
            conn.execute(sa.text(f"CREATE INDEX IF NOT EXISTS idx_index_jobs_namespace ON {ij}(namespace)"))

            # Legacy safety: ensure columns exist (no-op on fresh schemas)
            conn.execute(sa.text(f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default'"))
            conn.execute(sa.text(f"ALTER TABLE {ij} ADD COLUMN IF NOT EXISTS coalesce_key TEXT NOT NULL DEFAULT ''"))

            # Phase 2: coalescing constraint (namespaced)
            conn.execute(sa.text(
                f"CREATE UNIQUE INDEX IF NOT EXISTS uq_index_jobs_pending_ns_ck ON {ij}(namespace, coalesce_key) WHERE status='PENDING'"
            ))

            # Phase 2: applied fingerprints (namespaced)
            ias = f"{schema}.index_applied_state"
            conn.execute(sa.text(f"""
                CREATE TABLE IF NOT EXISTS {ias} (
                    namespace TEXT NOT NULL DEFAULT 'default',
                    coalesce_key TEXT NOT NULL,
                    applied_fingerprint TEXT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_job_id TEXT NULL,
                    PRIMARY KEY(namespace, coalesce_key)
                )
            """))
            conn.execute(sa.text(f"CREATE INDEX IF NOT EXISTS idx_index_applied_state_key ON {ias}(coalesce_key)"))


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
            row = conn.execute(sa.text(f"UPDATE {gt} SET value = value + 1 RETURNING value")).fetchone()
            if not row:
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
        ut = f"{self.schema}.user_seq"
        with self.transaction() as conn:
            conn.execute(sa.text(
                f"""
                INSERT INTO {ut}(user_id, value)
                VALUES (:user_id, :value)
                ON CONFLICT(user_id)
                DO UPDATE SET value = EXCLUDED.value
                """
            ), {"user_id": user_id, "value": int(value)})

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
    ) -> str:
        """Enqueue a job, coalescing repeated PENDING work by (namespace, coalesce_key)."""
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
                next_op = "DELETE" if (op == "DELETE" or existing_op == "DELETE") else op
                conn.execute(
                    sa.text(
                        f"""
                        UPDATE {ij}
                        SET op=:op, payload_json=:payload_json, updated_at=NOW()
                        WHERE job_id=:job_id
                        """
                    ),
                    {"op": next_op, "payload_json": payload_json, "job_id": existing_job_id},
                )
                return existing_job_id

            conn.execute(
                sa.text(
                    f"""
                    INSERT INTO {ij}(
                        job_id, namespace, entity_kind, entity_id, index_kind, coalesce_key, op,
                        status, lease_until, retry_count, last_error, payload_json, created_at, updated_at
                    )
                    VALUES (:job_id, :ns, :entity_kind, :entity_id, :index_kind, :ck, :op,
                            'PENDING', NULL, 0, NULL, :payload_json, NOW(), NOW())
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
                            status IN ('PENDING','FAILED')
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
                              j.lease_until, j.retry_count, j.last_error, j.payload_json
                    """
                ),
                {"limit": int(limit), "lease_seconds": int(lease_seconds), "namespace": namespace},
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
                    lease_until=(str(r.get("lease_until")) if r.get("lease_until") is not None else None),
                    retry_count=int(r.get("retry_count") or 0),
                    last_error=(str(r.get("last_error")) if r.get("last_error") is not None else None),
                    payload_json=(str(r.get("payload_json")) if r.get("payload_json") is not None else None),
                )
            )
        return out

    def mark_index_job_done(self, job_id: str) -> None:
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f"{self.schema}.\"{self.index_jobs_table}\""
        )
        with self.transaction() as conn:
            conn.execute(sa.text(
                f"UPDATE {ij} SET status='DONE', lease_until=NULL, updated_at=NOW() WHERE job_id=:job_id"
            ), {"job_id": job_id})

    def mark_index_job_failed(self, job_id: str, error: str) -> None:
        ij = (
            f"{self.schema}.{self.index_jobs_table}"
            if self.index_jobs_table == "index_jobs"
            else f"{self.schema}.\"{self.index_jobs_table}\""
        )
        with self.transaction() as conn:
            conn.execute(sa.text(
                f"""
                UPDATE {ij}
                SET status='FAILED', lease_until=NULL, retry_count=retry_count+1,
                    last_error=:err, updated_at=NOW()
                WHERE job_id=:job_id
                """
            ), {"job_id": job_id, "err": (error or "")[:2000]})

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
            else f"{self.schema}.\"{self.index_jobs_table}\""
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
            SELECT job_id, namespace, entity_kind, entity_id, index_kind, op, status,
                   lease_until, retry_count, last_error, payload_json
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
                    lease_until=(str(r.get("lease_until")) if r.get("lease_until") is not None else None),
                    retry_count=int(r.get("retry_count") or 0),
                    last_error=(str(r.get("last_error")) if r.get("last_error") is not None else None),
                    payload_json=(str(r.get("payload_json")) if r.get("payload_json") is not None else None),
                )
            )
        return out


    # ----------------------------
    # Phase 2: applied fingerprints (derived index status)
    # ----------------------------

    def get_index_applied_fingerprint(self, *, namespace: str = 'default', coalesce_key: str) -> Optional[str]:
        ias = f"{self.schema}.index_applied_state"
        with self.transaction() as conn:
            row = conn.execute(
                sa.text(f"SELECT applied_fingerprint FROM {ias} WHERE namespace=:ns AND coalesce_key=:ck"),
                {"ns": namespace, "ck": coalesce_key},
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
                {"ns": namespace, "ck": coalesce_key, "fp": applied_fingerprint, "jid": last_job_id},
            )
