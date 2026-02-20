from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class WorkerTickMetrics:
    claimed: int = 0
    done: int = 0
    retried: int = 0
    failed: int = 0
    avg_job_duration_s: Optional[float] = None


class IndexJobWorker:
    """Operational worker for index_jobs.

    - Uses meta store's claim/ack/requeue APIs.
    - Applies jobs via engine._apply_index_job(...).

    This is intentionally decoupled from the Engine hot path.
    """

    def __init__(
        self,
        *,
        engine: Any,
        max_inflight: int = 1,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_jobs_per_tick: int = 200,
        namespace: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.max_inflight = int(max_inflight)
        self.batch_size = int(batch_size)
        self.lease_seconds = int(lease_seconds)
        self.max_jobs_per_tick = int(max_jobs_per_tick)
        self.namespace = namespace

    def tick(self) -> WorkerTickMetrics:
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

        while remaining > 0:
            # IMPORTANT: don't over-claim beyond what we can process this tick.
            batch_n = min(self.batch_size, remaining, self.max_inflight)
            if batch_n <= 0:
                break

            jobs = claim(limit=batch_n, lease_seconds=self.lease_seconds, namespace=ns)
            if not jobs:
                break

            metrics.claimed += len(jobs)

            for job in jobs:
                start = time.time()

                # Support both dict-like and dataclass rows.
                job_id = getattr(job, "job_id", None) or (job.get("job_id") if isinstance(job, dict) else None)
                entity_kind = getattr(job, "entity_kind", None) or (job.get("entity_kind") if isinstance(job, dict) else None)
                entity_id = getattr(job, "entity_id", None) or (job.get("entity_id") if isinstance(job, dict) else None)
                index_kind = getattr(job, "index_kind", None) or (job.get("index_kind") if isinstance(job, dict) else None)
                op = getattr(job, "op", None) or (job.get("op") if isinstance(job, dict) else None)
                retry_count = getattr(job, "retry_count", None) if not isinstance(job, dict) else job.get("retry_count")
                max_retries = getattr(job, "max_retries", None) if not isinstance(job, dict) else job.get("max_retries")

                try_rc = int(retry_count or 0)
                try_mr = int(max_retries or 10)

                try:
                    self.engine._apply_index_job(
                        job_id=str(job_id),
                        entity_kind=str(entity_kind),
                        entity_id=str(entity_id),
                        index_kind=str(index_kind),
                        op=str(op),
                        namespace=ns,
                    )
                    if mark_done is not None and job_id:
                        mark_done(str(job_id))
                    metrics.done += 1
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    next_retry = try_rc + 1
                    if bump is not None and job_id and next_retry < try_mr:
                        delay = min(300, 2 ** min(next_retry - 1, 8))
                        bump(str(job_id), err, next_run_at_seconds=int(delay))
                        metrics.retried += 1
                    elif mark_failed is not None and job_id:
                        mark_failed(str(job_id), err, final=True)
                        metrics.failed += 1
                finally:
                    durations.append(time.time() - start)

                remaining -= 1
                if remaining <= 0:
                    break

        if durations:
            metrics.avg_job_duration_s = sum(durations) / len(durations)
        return metrics


def run_forever(
    *,
    worker: IndexJobWorker,
    tick_interval_s: float = 0.5,
    stop_flag: Optional[Callable[[], bool]] = None,
    on_tick: Optional[Callable[[WorkerTickMetrics], None]] = None,
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
    parser = argparse.ArgumentParser(description="GraphKnowledgeEngine index job worker runner")
    parser.add_argument("--persist-directory", required=True, help="Chroma/engine persist directory (shared with producer)")
    parser.add_argument("--namespace", default="default", help="Namespace to process")
    parser.add_argument("--tick-interval-ms", type=int, default=200, help="Sleep between ticks")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-jobs-per-tick", type=int, default=200)
    parser.add_argument("--max-inflight", type=int, default=50)
    parser.add_argument("--lease-seconds", type=int, default=60)
    parser.add_argument("--phase1-enable-index-jobs", action="store_true", help="Enable index_jobs feature flag")
    args = parser.parse_args(argv)

    from graph_knowledge_engine.engine import GraphKnowledgeEngine

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

    def _handle(_signum, _frame):
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
