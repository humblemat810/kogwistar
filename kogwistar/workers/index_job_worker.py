from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from ..engine_core.projection_action import (
    ProjectionGraphStore,
    ProjectionJobIdentity,
    apply_projection_action,
    is_projection_action_payload,
)
from ..implementation_selection import implementation_selection_bound

if TYPE_CHECKING:
    from ..engine_core.engine import GraphKnowledgeEngine


BatchApply = Callable[[list[object]], dict[str, BaseException | None]]


class IndexJobDone(Protocol):
    """Acknowledge one successfully applied index job."""

    def __call__(self, job_id: str, /, *, claim_token: str | None = None) -> object: ...


class IndexJobRetry(Protocol):
    """Requeue one failed index job with a bounded delay."""

    def __call__(
        self,
        job_id: str,
        error: str,
        /,
        *,
        next_run_at_seconds: int,
        claim_token: str | None = None,
    ) -> object: ...


class IndexJobFailed(Protocol):
    """Move one index job to terminal failure or a non-terminal failure state."""

    def __call__(
        self,
        job_id: str,
        error: str,
        /,
        *,
        final: bool,
        claim_token: str | None = None,
    ) -> object: ...


@dataclass
class WorkerTickMetrics:
    claimed: int = 0
    done: int = 0
    retried: int = 0
    failed: int = 0
    claim_lost: int = 0
    avg_job_duration_s: float | None = None


class IndexJobWorker:
    """Operational worker for index_jobs.

    - Uses meta store's claim/ack/requeue APIs.
    - Applies jobs via engine.indexing.apply_index_job(...).
    - ``batch_size`` controls provider batch width; ``max_inflight`` is reserved
      for future concurrent batch scheduling and does not alter batch width.

    This is intentionally decoupled from the Engine hot path.
    """

    def __init__(
        self,
        *,
        engine: GraphKnowledgeEngine,
        max_inflight: int = 1,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_jobs_per_tick: int = 200,
        namespace: str | None = None,
        projection_store: ProjectionGraphStore | None = None,
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
    def tick(self) -> WorkerTickMetrics:
        """Drain at most one worker tick of index jobs for a namespace.

        Each tick claims leased batches from the metastore, applies them through the
        shared indexing logic, marks successes DONE, and requeues failures with
        delayed retry until max_retries is exhausted. Queue ordering and lease
        semantics remain owned by the metastore implementation.
        """
        metrics = WorkerTickMetrics()
        meta = getattr(self.engine, "meta_sqlite", None)
        if meta is None:
            return metrics

        claim = getattr(meta, "claim_index_jobs", None)
        if claim is None:
            return metrics

        mark_done = getattr(meta, "mark_index_job_done", None)
        mark_failed = getattr(meta, "mark_index_job_failed", None)
        bump = getattr(meta, "bump_retry_and_requeue", None)

        remaining = self.max_jobs_per_tick
        durations: list[float] = []

        ns = self.namespace or getattr(self.engine, "namespace", "default")
        entity_cache: dict[tuple[str, str, str], object] = {}

        while remaining > 0:
            # IMPORTANT: don't over-claim beyond what we can process this tick.
            # ``batch_size`` is the provider request size. ``max_inflight`` is
            # reserved for concurrent batch scheduling and must not shrink a
            # batch when this worker is operating serially.
            batch_n = min(self.batch_size, remaining)
            if batch_n <= 0:
                break

            jobs = claim(limit=batch_n, lease_seconds=self.lease_seconds, namespace=ns)
            if not jobs:
                break

            metrics.claimed += len(jobs)

            adapter = getattr(self.engine, "two_stage_projection_adapter", None)
            batch_apply = getattr(adapter, "apply_embedding_jobs_batch", None)

            # Keep embedding provider calls homogeneous, while allowing a single
            # claim to contain ordinary projection work as well. Embeddings run
            # first; non-embedding jobs retain their existing one-job handler.
            embedding_groups: dict[tuple[str, str, str], list[object]] = {}
            ordinary_jobs: list[object] = []
            for job in jobs:
                if self._job_value(job, "accepted_result_json") is not None:
                    ordinary_jobs.append(job)
                    continue
                index_kind = self._job_value(job, "index_kind")
                if index_kind == "node_embedding":
                    payload = self._decode_payload(self._job_value(job, "payload_json"))
                    key = (
                        str(payload.get("embedding_model") or ""),
                        str(payload.get("embedding_version") or ""),
                        str(payload.get("embedding_provider") or ""),
                    )
                    embedding_groups.setdefault(key, []).append(job)
                else:
                    ordinary_jobs.append(job)

            batch_fn: BatchApply | None = (
                cast(BatchApply, batch_apply) if callable(batch_apply) else None
            )
            work_units: list[tuple[list[object], BatchApply | None]] = []
            for group in embedding_groups.values():
                for start in range(0, len(group), max(1, self.batch_size)):
                    work_units.append((group[start : start + max(1, self.batch_size)], batch_fn))
            work_units.extend(([job], None) for job in ordinary_jobs)

            for unit, batch_fn in work_units:
                batch_results: dict[str, BaseException | None] | None = None
                if batch_fn is not None and len(unit) > 1:
                    try:
                        batch_results = batch_fn(unit)
                    except Exception as exc:
                        # A provider/adapter batch failure is per-job failure,
                        # not a worker-process failure or batch transaction.
                        batch_results = {
                            str(self._job_value(job, "job_id")): exc for job in unit
                        }

                for job in unit:
                    self._process_job(
                        job=job,
                        batch_results=batch_results,
                        namespace=ns,
                        entity_cache=entity_cache,
                        mark_done=mark_done,
                        bump=bump,
                        mark_failed=mark_failed,
                        metrics=metrics,
                        durations=durations,
                    )
                    remaining -= 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

        if durations:
            metrics.avg_job_duration_s = sum(durations) / len(durations)
        return metrics

    @staticmethod
    def _job_value(job: object, name: str) -> object:
        if isinstance(job, dict):
            return job.get(name)
        return getattr(job, name, None)

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

    @staticmethod
    def _decode_payload(payload_json: object) -> dict[str, object]:
        if isinstance(payload_json, dict):
            return payload_json
        if isinstance(payload_json, str) and payload_json:
            try:
                value = json.loads(payload_json)
            except (TypeError, ValueError):
                return {}
            return value if isinstance(value, dict) else {}
        return {}

    def _process_job(
        self,
        *,
        job: object,
        batch_results: dict[str, BaseException | None] | None,
        namespace: str,
        entity_cache: dict[tuple[str, str, str], object],
        mark_done: IndexJobDone | None,
        bump: IndexJobRetry | None,
        mark_failed: IndexJobFailed | None,
        metrics: WorkerTickMetrics,
        durations: list[float],
    ) -> None:
                start = time.time()

                # Support both dict-like and dataclass rows.
                job_id = self._job_value(job, "job_id")
                entity_kind = self._job_value(job, "entity_kind")
                entity_id = self._job_value(job, "entity_id")
                index_kind = self._job_value(job, "index_kind")
                op = self._job_value(job, "op")
                payload_json = self._job_value(job, "payload_json")
                claim_token = self._optional_text(self._job_value(job, "claim_token"))
                retry_count = (
                    self._job_value(job, "retry_count")
                )
                max_retries = (
                    self._job_value(job, "max_retries")
                )

                try_rc = self._as_int(retry_count, 0)
                try_mr = self._as_int(max_retries, 10)

                try:
                    # Rust-owned workers may have admitted the provider result
                    # before losing the queue acknowledgement.  Do not repeat
                    # projection/provider work when the durable result exists.
                    accepted_result = self._job_value(job, "accepted_result_json")
                    if isinstance(accepted_result, str) and is_projection_action_payload(
                        accepted_result
                    ):
                        if self.projection_store is None:
                            raise RuntimeError(
                                "accepted graph projection action requires projection_store"
                            )
                        identity_fields = (
                            "job_id",
                            "namespace",
                            "entity_kind",
                            "entity_id",
                            "index_kind",
                            "op",
                        )
                        identity_values = {
                            name: self._job_value(job, name) for name in identity_fields
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
                        if identity.namespace != namespace:
                            raise RuntimeError(
                                "claimed projection action is outside the worker namespace"
                            )
                        digest = self._job_value(job, "accepted_result_sha256")
                        apply_projection_action(
                            identity,
                            accepted_result,
                            digest if isinstance(digest, str) else None,
                            self.projection_store,
                        )
                    elif accepted_result is not None:
                        # Existing provider-result records were already applied
                        # by their producer; preserve their legacy replay rule.
                        pass
                    elif batch_results is not None:
                        if str(job_id) not in batch_results:
                            raise RuntimeError("batch embedding result missing job")
                        batch_error = batch_results[str(job_id)]
                        if batch_error is not None:
                            raise batch_error
                    else:
                        self.engine.indexing.apply_index_job(
                            job_id=str(job_id),
                            entity_kind=str(entity_kind),
                            entity_id=str(entity_id),
                            index_kind=str(index_kind),
                            op=str(op),
                            namespace=namespace,
                            payload_json=self._optional_text(payload_json),
                            validated_entity_cache=entity_cache,
                        )
                    if mark_done is not None and job_id:
                        if claim_token is None:
                            # Preserve compatibility with legacy queue rows that
                            # predate leases; current claimed rows always carry a
                            # token and therefore take the ownership-checked path.
                            acknowledged = mark_done(str(job_id))
                        else:
                            acknowledged = mark_done(
                                str(job_id), claim_token=claim_token
                            )
                        if acknowledged is False:
                            metrics.claim_lost += 1
                            return
                    metrics.done += 1
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    next_retry = try_rc + 1
                    if bump is not None and job_id and next_retry < try_mr:
                        delay = min(300, 2 ** min(next_retry - 1, 8))
                        if claim_token is None:
                            bump(str(job_id), err, next_run_at_seconds=int(delay))
                        else:
                            bump(
                                str(job_id),
                                err,
                                next_run_at_seconds=int(delay),
                                claim_token=claim_token,
                            )
                        metrics.retried += 1
                    elif mark_failed is not None and job_id:
                        if claim_token is None:
                            mark_failed(str(job_id), err, final=True)
                        else:
                            mark_failed(
                                str(job_id),
                                err,
                                final=True,
                                claim_token=claim_token,
                            )
                        metrics.failed += 1
                finally:
                    durations.append(time.time() - start)


def run_forever(
    *,
    worker: IndexJobWorker,
    tick_interval_s: float = 0.5,
    stop_flag: Callable[[], bool] | None = None,
    on_tick: Callable[[WorkerTickMetrics], None] | None = None,
) -> None:
    """Runnable loop for a worker process."""
    while True:
        if stop_flag is not None and stop_flag():
            return
        m = worker.tick()
        if on_tick is not None:
            on_tick(m)
        time.sleep(float(tick_interval_s))


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="GraphKnowledgeEngine index job worker runner"
    )
    parser.add_argument(
        "--persist-directory",
        required=True,
        help="Chroma/engine persist directory (shared with producer)",
    )
    parser.add_argument("--namespace", default="default", help="Namespace to process")
    parser.add_argument(
        "--tick-interval-ms", type=int, default=200, help="Sleep between ticks"
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-jobs-per-tick", type=int, default=200)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=50,
        help="Reserved concurrent-batch limit; serial worker currently processes one batch at a time",
    )
    parser.add_argument("--lease-seconds", type=int, default=60)
    parser.add_argument(
        "--phase1-enable-index-jobs",
        action="store_true",
        help="Enable index_jobs feature flag",
    )
    args = parser.parse_args(argv)

    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    eng = GraphKnowledgeEngine(persist_directory=args.persist_directory)
    eng._phase1_enable_index_jobs = bool(args.phase1_enable_index_jobs)  # noqa: SLF001

    # Ensure the namespace matches what the producer uses.
    try:
        eng.namespace = args.namespace  # type: ignore[attr-defined]
    except Exception:
        pass

    worker = IndexJobWorker(
        engine=eng,
        namespace=args.namespace,
        batch_size=args.batch_size,
        max_jobs_per_tick=args.max_jobs_per_tick,
        max_inflight=args.max_inflight,
        lease_seconds=args.lease_seconds,
    )

    stop = {"flag": False}

    def _handle(_signum: int, _frame: object) -> None:
        stop["flag"] = True

    # Cross-platform-ish: SIGTERM works on POSIX; on Windows terminate() is hard-kill, but handler still helps for Ctrl+C.
    try:
        signal.signal(signal.SIGTERM, _handle)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGINT, _handle)
    except Exception:
        pass

    run_forever(
        worker=worker,
        tick_interval_s=max(0.01, args.tick_interval_ms / 1000.0),
        stop_flag=lambda: stop["flag"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
