from __future__ import annotations

"""Typed facade over the durable metastore job queue."""

import json
import uuid
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import GraphKnowledgeEngine


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


class JobQueueSubsystem:
    """Generic durable job facade backed by the existing ``index_jobs`` table."""

    def __init__(self, engine: "GraphKnowledgeEngine") -> None:
        self.engine = engine

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

    def mark_done(self, job_id: str) -> None:
        mark_done = getattr(self.engine.meta_sqlite, "mark_index_job_done", None)
        if mark_done is not None:
            mark_done(str(job_id))

    def mark_failed(self, job_id: str, error: str, *, final: bool = True) -> None:
        mark_failed = getattr(self.engine.meta_sqlite, "mark_index_job_failed", None)
        if mark_failed is not None:
            mark_failed(str(job_id), str(error), final=final)

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
                bump(job.job_id, err, next_run_at_seconds=delay)
                return
        self.mark_failed(job.job_id, err, final=True)

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
