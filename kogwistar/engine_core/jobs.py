from __future__ import annotations

"""Typed facade over the durable metastore job queue."""

import json
import uuid
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import GraphKnowledgeEngine


class DurableQueueUnavailableError(RuntimeError):
    """Raised when an app-critical durable queue capability is not available."""


@dataclass(frozen=True, slots=True)
class JobQueueItem:
    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    job_kind: str
    op: str
    payload: dict[str, Any]
    retry_count: int
    max_retries: int
    last_error: str | None = None
    claim_token: str | None = None
    claim_attempts: int = 0


class JobQueueSubsystem:
    """Generic durable job facade backed by the existing ``index_jobs`` table."""

    def __init__(self, engine: "GraphKnowledgeEngine") -> None:
        self.engine = engine

    def require_available(self, *, enqueue: bool = False, claim: bool = False) -> None:
        required: list[str] = []
        if enqueue:
            required.append("enqueue_index_job")
        if claim:
            required.append("claim_index_jobs")
        if not required:
            raise ValueError("require_available() needs at least one capability check")

        missing = [
            name
            for name in required
            if getattr(self.engine.meta_sqlite, name, None) is None
        ]
        if missing:
            raise DurableQueueUnavailableError(
                "Durable queue support is required but unavailable: "
                + ", ".join(sorted(missing))
            )

    def enqueue(
        self,
        *,
        job_id: str | None = None,
        namespace: str,
        entity_kind: str,
        entity_id: str,
        job_kind: str,
        op: str = "UPSERT",
        payload: dict[str, Any] | None = None,
        max_retries: int = 10,
    ) -> str:
        enqueue = getattr(self.engine.meta_sqlite, "enqueue_index_job", None)
        if enqueue is None:
            return ""

        chosen_job_id = str(job_id or uuid.uuid4())
        stored_job_id = enqueue(
            job_id=chosen_job_id,
            namespace=namespace,
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind=job_kind,
            op=op,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
            max_retries=max_retries,
        )
        return str(stored_job_id or chosen_job_id)

    def claim(
        self,
        *,
        namespace: str,
        limit: int = 50,
        lease_seconds: int = 60,
    ) -> list[JobQueueItem]:
        claim = getattr(self.engine.meta_sqlite, "claim_index_jobs", None)
        if claim is None:
            return []
        return [
            self._from_row(row)
            for row in claim(limit=limit, lease_seconds=lease_seconds, namespace=namespace)
        ]

    def mark_done(self, job_id: str, *, claim_token: str | None = None) -> bool:
        mark_done = getattr(self.engine.meta_sqlite, "mark_index_job_done", None)
        if mark_done is not None:
            if claim_token is None:
                return bool(mark_done(str(job_id)))
            return bool(mark_done(str(job_id), claim_token=claim_token))
        return False

    def renew_lease(self, job: JobQueueItem, *, lease_seconds: int) -> bool:
        renew = getattr(self.engine.meta_sqlite, "renew_index_job_lease", None)
        if renew is None or job.claim_token is None:
            return False
        return bool(renew(job.job_id, claim_token=job.claim_token, lease_seconds=lease_seconds))

    def mark_failed(self, job_id: str, error: str, *, final: bool = True, claim_token: str | None = None) -> None:
        mark_failed = getattr(self.engine.meta_sqlite, "mark_index_job_failed", None)
        if mark_failed is not None:
            if claim_token is None:
                mark_failed(str(job_id), str(error), final=final)
            else:
                mark_failed(str(job_id), str(error), final=final, claim_token=claim_token)

    def retry_or_fail(
        self,
        job: JobQueueItem,
        error: Exception | str,
        *,
        max_delay_seconds: int = 300,
    ) -> None:
        err = f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
        next_retry = int(job.retry_count) + 1
        if next_retry < int(job.max_retries):
            bump = getattr(self.engine.meta_sqlite, "bump_retry_and_requeue", None)
            if bump is not None:
                delay = min(int(max_delay_seconds), 2 ** max(int(job.retry_count), 0))
                if job.claim_token is None:
                    bump(job.job_id, err, next_run_at_seconds=delay)
                else:
                    bump(job.job_id, err, next_run_at_seconds=delay, claim_token=job.claim_token)
                return
        self.mark_failed(job.job_id, err, final=True, claim_token=job.claim_token)

    def requeue_at_tail(
        self,
        job: JobQueueItem,
        *,
        payload: dict[str, Any] | None = None,
        delay_seconds: int = 0,
    ) -> None:
        """Return a claimed job to the runnable queue tail.

        This is for cooperative continuation, not failure retry. The payload
        may carry a durable workflow checkpoint so the next claim resumes the
        same job instead of restarting its work.
        """
        requeue = getattr(self.engine.meta_sqlite, "requeue_index_job_at_tail", None)
        if requeue is None:
            raise RuntimeError("meta store does not support fair job requeue")
        kwargs = {
            "payload_json": json.dumps(payload if payload is not None else job.payload),
            "delay_seconds": max(0, int(delay_seconds)),
        }
        if job.claim_token is not None:
            kwargs["claim_token"] = job.claim_token
        requeue(str(job.job_id), **kwargs)

    def list(
        self,
        *,
        namespace: str,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[JobQueueItem]:
        list_jobs = getattr(self.engine.meta_sqlite, "list_index_jobs", None)
        if list_jobs is None:
            return []
        return [
            self._from_row(row)
            for row in list_jobs(namespace=namespace, status=status, limit=limit)
        ]

    def coerce(self, row: Any) -> JobQueueItem:
        """Normalize a raw metastore row or return an existing job item."""
        if isinstance(row, JobQueueItem):
            return row
        return self._from_row(row)

    def _from_row(self, row: Any) -> JobQueueItem:
        def field(name: str, default: Any = None) -> Any:
            if isinstance(row, dict):
                return row.get(name, default)
            return getattr(row, name, default)

        payload = self._decode_payload(field("payload_json"))
        return JobQueueItem(
            job_id=str(field("job_id", "")),
            namespace=str(field("namespace", "")),
            entity_kind=str(field("entity_kind", "")),
            entity_id=str(field("entity_id", "")),
            job_kind=str(field("index_kind", "")),
            op=str(field("op", "")),
            payload=payload,
            retry_count=int(field("retry_count", 0) or 0),
            max_retries=int(field("max_retries", 10) or 10),
            last_error=(None if field("last_error") is None else str(field("last_error"))),
            claim_token=(None if field("claim_token") is None else str(field("claim_token"))),
            claim_attempts=int(field("claim_attempts", 0) or 0),
        )

    @staticmethod
    def _decode_payload(payload_json: Any) -> dict[str, Any]:
        if isinstance(payload_json, dict):
            return dict(payload_json)
        if isinstance(payload_json, str) and payload_json:
            try:
                decoded = json.loads(payload_json)
            except Exception:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}
