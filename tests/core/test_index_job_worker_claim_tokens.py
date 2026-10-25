from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.workers.async_index_job_worker import AsyncIndexJobWorker
from kogwistar.workers.index_job_worker import IndexJobWorker


class _Meta:
    def __init__(self, job: SimpleNamespace) -> None:
        self.job = job
        self.claimed = False
        self.done: list[tuple[str, str | None]] = []
        self.retried: list[tuple[str, str | None]] = []
        self.failed: list[tuple[str, str | None]] = []

    def claim_index_jobs(self, **_: object) -> list[SimpleNamespace]:
        if self.claimed:
            return []
        self.claimed = True
        return [self.job]

    def mark_index_job_done(self, job_id: str, *, claim_token: str | None = None) -> None:
        self.done.append((job_id, claim_token))

    def bump_retry_and_requeue(
        self,
        job_id: str,
        error: str,
        *,
        next_run_at_seconds: int,
        claim_token: str | None = None,
    ) -> None:
        del error, next_run_at_seconds
        self.retried.append((job_id, claim_token))

    def mark_index_job_failed(
        self,
        job_id: str,
        error: str,
        *,
        final: bool,
        claim_token: str | None = None,
    ) -> None:
        del error, final
        self.failed.append((job_id, claim_token))


class _Indexing:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def apply_index_job(self, **_: object) -> None:
        if self.fail:
            raise RuntimeError("projection failed")


def _engine(meta: _Meta, *, fail: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        meta_sqlite=meta,
        indexing=_Indexing(fail=fail),
        namespace="test",
        _implementation_selection_snapshot=None,
    )


def _reconciler(meta: _Meta, *, fail: bool = False) -> IndexingSubsystem:
    engine = SimpleNamespace(
        meta_sqlite=meta,
        namespace="test",
        _phase1_enable_index_jobs=True,
        _phase1_enable_validation_cache=False,
        _implementation_selection_snapshot=None,
    )
    subsystem = IndexingSubsystem(engine)

    def apply_index_job(**_: object) -> None:
        if fail:
            raise RuntimeError("projection failed")

    subsystem.apply_index_job = apply_index_job  # type: ignore[method-assign]
    return subsystem


def _job(*, claim_token: str | None, max_retries: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-1",
        entity_kind="node",
        entity_id="node-1",
        index_kind="node_docs",
        op="UPSERT",
        payload_json="{}",
        retry_count=0,
        max_retries=max_retries,
        claim_token=claim_token,
    )


def test_sync_worker_propagates_claim_token_to_success_ack() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    worker = IndexJobWorker(engine=_engine(meta))

    metrics = worker.tick()

    assert metrics.done == 1
    assert meta.done == [("job-1", "claim-1")]


def test_sync_worker_propagates_claim_token_to_retry() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    worker = IndexJobWorker(engine=_engine(meta, fail=True))

    metrics = worker.tick()

    assert metrics.retried == 1
    assert meta.retried == [("job-1", "claim-1")]


def test_sync_worker_propagates_claim_token_to_terminal_failure() -> None:
    meta = _Meta(_job(claim_token="claim-1", max_retries=1))
    worker = IndexJobWorker(engine=_engine(meta, fail=True))

    metrics = worker.tick()

    assert metrics.failed == 1
    assert meta.failed == [("job-1", "claim-1")]


def test_async_worker_propagates_claim_token_to_success_ack() -> None:
    async def run() -> None:
        meta = _Meta(_job(claim_token="claim-1"))
        worker = AsyncIndexJobWorker(engine=_engine(meta))

        metrics = await worker.tick()

        assert metrics.done == 1
        assert meta.done == [("job-1", "claim-1")]

    asyncio.run(run())


def test_async_worker_propagates_claim_token_to_retry() -> None:
    async def run() -> None:
        meta = _Meta(_job(claim_token="claim-1"))
        worker = AsyncIndexJobWorker(engine=_engine(meta, fail=True))

        metrics = await worker.tick()

        assert metrics.retried == 1
        assert meta.retried == [("job-1", "claim-1")]

    asyncio.run(run())


def test_async_worker_propagates_claim_token_to_terminal_failure() -> None:
    async def run() -> None:
        meta = _Meta(_job(claim_token="claim-1", max_retries=1))
        worker = AsyncIndexJobWorker(engine=_engine(meta, fail=True))

        metrics = await worker.tick()

        assert metrics.failed == 1
        assert meta.failed == [("job-1", "claim-1")]

    asyncio.run(run())


def test_legacy_claim_without_token_keeps_compatibility_call_shape() -> None:
    meta = _Meta(_job(claim_token=None))
    worker = IndexJobWorker(engine=_engine(meta))

    assert worker.tick().done == 1
    assert meta.done == [("job-1", None)]


def test_reconciler_propagates_claim_token_to_success_ack() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    subsystem = _reconciler(meta)

    assert subsystem.reconcile_indexes(max_jobs=1) == 1
    assert meta.done == [("job-1", "claim-1")]


def test_reconciler_propagates_claim_token_to_retry() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    subsystem = _reconciler(meta, fail=True)

    assert subsystem.reconcile_indexes(max_jobs=1) == 0
    assert meta.retried == [("job-1", "claim-1")]
