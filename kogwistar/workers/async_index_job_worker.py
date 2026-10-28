from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from ..engine_core.projection_action import (
    AsyncProjectionGraphStore,
    ProjectionJobIdentity,
    apply_projection_action_async,
    is_projection_action_payload,
)
from ..engine_core.index_job_provider import embedding_job_provider_input
from ..implementation_selection import implementation_selection_bound
from .lease_heartbeat import AsyncLeaseHeartbeat

if TYPE_CHECKING:
    from ..engine_core.engine import GraphKnowledgeEngine


T = TypeVar("T")


@dataclass
class AsyncWorkerTickMetrics:
    claimed: int = 0
    done: int = 0
    retried: int = 0
    failed: int = 0
    claim_lost: int = 0
    avg_job_duration_s: float | None = None


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


class AsyncIndexJobWorker:
    """Async index worker using the existing queue and lease semantics."""

    def __init__(
        self,
        *,
        engine: GraphKnowledgeEngine,
        max_inflight: int = 1,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_jobs_per_tick: int = 200,
        namespace: str | None = None,
        projection_store: AsyncProjectionGraphStore | None = None,
    ) -> None:
        self.engine = engine
        self.max_inflight = int(max_inflight)
        self.batch_size = int(batch_size)
        self.lease_seconds = int(lease_seconds)
        self.max_jobs_per_tick = int(max_jobs_per_tick)
        self.namespace = namespace
        self.projection_store = projection_store

    @property
    def _implementation_selection_snapshot(self) -> object:
        return getattr(self.engine, "_implementation_selection_snapshot", None)

    @implementation_selection_bound
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
        jobs = self._jobs(jobs)
        metrics.claimed = len(jobs)
        durations: list[float] = []
        adapter = getattr(self.engine, "async_two_stage_projection_adapter", None)
        embedding_groups: dict[tuple[str, str, str], list[object]] = {}
        ordinary_jobs: list[object] = []
        for job in jobs:
            if self._value(job, "accepted_result_json") is not None:
                ordinary_jobs.append(job)
                continue
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

        work_units: list[tuple[list[object], object | None]] = []
        batch_apply = getattr(adapter, "apply_embedding_jobs_batch", None)
        for group in embedding_groups.values():
            width = max(1, self.batch_size)
            for start in range(0, len(group), width):
                work_units.append((group[start : start + width], batch_apply))
        work_units.extend(([job], None) for job in ordinary_jobs)

        claims = [
            (
                str(self._value(job, "job_id")),
                self._optional_text(self._value(job, "claim_token")),
            )
            for job in jobs
        ]
        for unit, batch_apply in work_units:
            started = time.perf_counter()
            async with AsyncLeaseHeartbeat(meta, claims, self.lease_seconds) as heartbeat:
                batch_results: dict[str, BaseException | None] | None = None
                if batch_apply is not None and len(unit) > 1:
                    try:
                        if not inspect.iscoroutinefunction(batch_apply):
                            raise RuntimeError("async embedding batch handler must be awaitable")
                        provider_inputs = [embedding_job_provider_input(job) for job in unit]
                        batch_results = await batch_apply(provider_inputs)
                    except Exception as exc:
                        batch_results = {
                            str(self._value(job, "job_id")): exc for job in unit
                        }
                for job in unit:
                    job_id = self._value(job, "job_id")
                    claim_token = self._optional_text(self._value(job, "claim_token"))
                    if (
                        claim_token is not None
                        and job_id
                        and not heartbeat.owns(str(job_id))
                    ):
                        metrics.claim_lost += 1
                        continue
                    try:
                        accepted_result = self._value(job, "accepted_result_json")
                        is_projection_action = (
                            accepted_result is not None
                            and is_projection_action_payload(accepted_result)
                        )
                        if is_projection_action:
                            if self.projection_store is None:
                                raise RuntimeError(
                                    "accepted graph projection action requires projection_store"
                                )
                            if not isinstance(accepted_result, str):
                                raise RuntimeError("accepted projection action must be JSON text")
                            identity_fields = (
                                "job_id",
                                "namespace",
                                "entity_kind",
                                "entity_id",
                                "index_kind",
                                "op",
                            )
                            identity_values = {
                                name: self._value(job, name) for name in identity_fields
                            }
                            if not all(
                                isinstance(value, str)
                                for value in identity_values.values()
                            ):
                                raise RuntimeError(
                                    "claimed projection action lacks a complete queue identity"
                                )
                            identity = ProjectionJobIdentity(
                                job_id=cast(str, identity_values["job_id"]),
                                namespace=cast(str, identity_values["namespace"]),
                                entity_kind=cast(str, identity_values["entity_kind"]),
                                entity_id=cast(str, identity_values["entity_id"]),
                                index_kind=cast(str, identity_values["index_kind"]),
                                op=cast(str, identity_values["op"]),
                            )
                            if identity.namespace != ns:
                                raise RuntimeError(
                                    "claimed projection action is outside the worker namespace"
                                )
                            digest = self._value(job, "accepted_result_sha256")
                            await apply_projection_action_async(
                                identity,
                                accepted_result,
                                digest if isinstance(digest, str) else None,
                                self.projection_store,
                            )
                        elif accepted_result is not None:
                            pass
                        elif batch_results is not None:
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
                                payload_json=self._optional_text(
                                    self._value(job, "payload_json")
                                ),
                            )
                        else:
                            await asyncio.to_thread(
                                self.engine.indexing.apply_index_job,
                                job_id=str(job_id),
                                entity_kind=str(self._value(job, "entity_kind")),
                                entity_id=str(self._value(job, "entity_id")),
                                index_kind=str(self._value(job, "index_kind") or ""),
                                op=str(self._value(job, "op") or "UPSERT"),
                                namespace=ns,
                                payload_json=self._optional_text(
                                    self._value(job, "payload_json")
                                ),
                            )
                        if (
                            claim_token is not None
                            and job_id
                            and not heartbeat.owns(str(job_id))
                        ):
                            metrics.claim_lost += 1
                            continue
                        done = getattr(meta, "mark_index_job_done", None)
                        acknowledged: object = None
                        if callable(done):
                            if claim_token is None:
                                acknowledged = await _maybe_await(done(str(job_id)))
                            else:
                                acknowledged = await _maybe_await(
                                    done(str(job_id), claim_token=claim_token)
                                )
                        if acknowledged is False:
                            metrics.claim_lost += 1
                            continue
                        metrics.done += 1
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        retry = self._as_int(self._value(job, "retry_count"), 0) + 1
                        max_retries = self._as_int(self._value(job, "max_retries"), 10)
                        bump = getattr(meta, "bump_retry_and_requeue", None)
                        failed = getattr(meta, "mark_index_job_failed", None)
                        if retry < max_retries and callable(bump):
                            delay = min(300, 2 ** min(retry - 1, 8))
                            if claim_token is not None:
                                await _maybe_await(
                                    bump(
                                        str(job_id),
                                        error,
                                        next_run_at_seconds=delay,
                                        claim_token=claim_token,
                                    )
                                )
                            else:
                                await _maybe_await(
                                    bump(
                                        str(job_id), error, next_run_at_seconds=delay
                                    )
                                )
                            metrics.retried += 1
                        elif callable(failed):
                            if claim_token is not None:
                                await _maybe_await(
                                    failed(
                                        str(job_id),
                                        error,
                                        final=True,
                                        claim_token=claim_token,
                                    )
                                )
                            else:
                                await _maybe_await(
                                    failed(str(job_id), error, final=True)
                                )
                            metrics.failed += 1
                    finally:
                        durations.append(time.perf_counter() - started)
        if durations:
            metrics.avg_job_duration_s = sum(durations) / len(durations)
        return metrics

    @staticmethod
    def _value(job: object, name: str) -> object:
        return job.get(name) if isinstance(job, dict) else getattr(job, name, None)

    @staticmethod
    def _jobs(value: object) -> list[object]:
        if value is None or isinstance(value, (str, bytes, bytearray)):
            return []
        if isinstance(value, Iterable):
            return list(value)
        return []

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default


__all__ = ["AsyncIndexJobWorker", "AsyncWorkerTickMetrics"]
