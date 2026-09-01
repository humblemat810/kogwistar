from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AsyncWorkerTickMetrics:
    claimed: int = 0
    done: int = 0
    retried: int = 0
    failed: int = 0
    avg_job_duration_s: Optional[float] = None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AsyncIndexJobWorker:
    """Async index worker using the existing queue and lease semantics."""

    def __init__(
        self,
        *,
        engine: Any,
        max_inflight: int = 1,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_jobs_per_tick: int = 200,
        namespace: str | None = None,
    ) -> None:
        self.engine = engine
        self.max_inflight = int(max_inflight)
        self.batch_size = int(batch_size)
        self.lease_seconds = int(lease_seconds)
        self.max_jobs_per_tick = int(max_jobs_per_tick)
        self.namespace = namespace

    async def tick(self) -> AsyncWorkerTickMetrics:
        metrics = AsyncWorkerTickMetrics()
        meta = getattr(self.engine, "meta_sqlite", None)
        claim = getattr(meta, "claim_index_jobs", None)
        if not callable(claim):
            return metrics
        ns = self.namespace or getattr(self.engine, "namespace", "default")
        jobs = await _maybe_await(
            claim(
                limit=max(0, self.max_jobs_per_tick),
                lease_seconds=self.lease_seconds,
                namespace=ns,
            )
        )
        jobs = list(jobs or [])
        metrics.claimed = len(jobs)
        durations: list[float] = []
        adapter = getattr(self.engine, "async_two_stage_projection_adapter", None)
        embedding_groups: dict[tuple[str, str, str], list[Any]] = {}
        ordinary_jobs: list[Any] = []
        for job in jobs:
            if str(self._value(job, "index_kind") or "") != "node_embedding":
                ordinary_jobs.append(job)
                continue
            payload = self._value(job, "payload_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (TypeError, ValueError):
                    payload = {}
            payload = payload if isinstance(payload, dict) else {}
            key = (
                str(payload.get("embedding_model") or ""),
                str(payload.get("embedding_version") or ""),
                str(payload.get("embedding_provider") or ""),
            )
            embedding_groups.setdefault(key, []).append(job)

        work_units: list[tuple[list[Any], Any | None]] = []
        batch_apply = getattr(adapter, "apply_embedding_jobs_batch", None)
        for group in embedding_groups.values():
            width = max(1, self.batch_size)
            for start in range(0, len(group), width):
                work_units.append((group[start : start + width], batch_apply))
        work_units.extend(([job], None) for job in ordinary_jobs)

        for unit, batch_apply in work_units:
            started = time.perf_counter()
            batch_results: dict[str, BaseException | None] | None = None
            if batch_apply is not None and len(unit) > 1:
                try:
                    if not inspect.iscoroutinefunction(batch_apply):
                        raise RuntimeError("async embedding batch handler must be awaitable")
                    batch_results = await batch_apply(unit)
                except BaseException as exc:
                    batch_results = {
                        str(self._value(job, "job_id")): exc for job in unit
                    }
            for job in unit:
                job_id = self._value(job, "job_id")
                try:
                    if batch_results is not None:
                        error = batch_results.get(str(job_id))
                        if error is not None:
                            raise error
                    elif str(self._value(job, "index_kind") or "") == "node_embedding":
                        apply = getattr(adapter, "apply_embedding_job", None)
                        if not callable(apply) or not inspect.iscoroutinefunction(apply):
                            raise RuntimeError(
                                "node_embedding requires an executable async two-stage adapter"
                            )
                        await apply(
                            entity_kind=str(self._value(job, "entity_kind")),
                            entity_id=str(self._value(job, "entity_id")),
                            op=str(self._value(job, "op") or "UPSERT"),
                            payload_json=self._value(job, "payload_json"),
                        )
                    else:
                        # Legacy projection handlers remain isolated from the
                        # event loop while retaining one-job semantics.
                        await asyncio.to_thread(
                            self.engine.indexing.apply_index_job,
                            job_id=str(job_id),
                            entity_kind=str(self._value(job, "entity_kind")),
                            entity_id=str(self._value(job, "entity_id")),
                            index_kind=str(self._value(job, "index_kind") or ""),
                            op=str(self._value(job, "op") or "UPSERT"),
                            namespace=ns,
                            payload_json=self._value(job, "payload_json"),
                        )
                    done = getattr(meta, "mark_index_job_done", None)
                    if callable(done):
                        await _maybe_await(done(str(job_id)))
                    metrics.done += 1
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    retry = int(self._value(job, "retry_count") or 0) + 1
                    max_retries = int(self._value(job, "max_retries") or 10)
                    bump = getattr(meta, "bump_retry_and_requeue", None)
                    failed = getattr(meta, "mark_index_job_failed", None)
                    if retry < max_retries and callable(bump):
                        await _maybe_await(
                            bump(
                                str(job_id),
                                error,
                                next_run_at_seconds=min(300, 2 ** min(retry - 1, 8)),
                            )
                        )
                        metrics.retried += 1
                    elif callable(failed):
                        await _maybe_await(failed(str(job_id), error, final=True))
                        metrics.failed += 1
                finally:
                    durations.append(time.perf_counter() - started)
        if durations:
            metrics.avg_job_duration_s = sum(durations) / len(durations)
        return metrics

    @staticmethod
    def _value(job: Any, name: str) -> Any:
        return job.get(name) if isinstance(job, dict) else getattr(job, name, None)


__all__ = ["AsyncIndexJobWorker", "AsyncWorkerTickMetrics"]
