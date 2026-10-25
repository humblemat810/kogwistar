from __future__ import annotations

import asyncio
from types import SimpleNamespace

from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.workers.async_index_job_worker import AsyncIndexJobWorker
from kogwistar.workers.index_job_worker import IndexJobWorker


class _Meta:
    def __init__(self, job: SimpleNamespace, *, reject_ack: bool = False) -> None:
        self.job = job
        self.reject_ack = reject_ack
        self.claimed = False
        self.done: list[tuple[str, str | None]] = []
        self.retried: list[tuple[str, str | None]] = []
        self.failed: list[tuple[str, str | None]] = []

    def claim_index_jobs(self, **_: object) -> list[SimpleNamespace]:
        if self.claimed:
            return []
        self.claimed = True
        return [self.job]

    def mark_index_job_done(
        self, job_id: str, *, claim_token: str | None = None
    ) -> object:
        self.done.append((job_id, claim_token))
        if self.reject_ack:
            return False  # type: ignore[return-value]
        return None

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
        self.calls = 0

    def apply_index_job(self, **_: object) -> None:
        self.calls += 1
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
        accepted_result_json=None,
    )


def _admitted_job(*, claim_token: str | None) -> SimpleNamespace:
    job = _job(claim_token=claim_token)
    job.accepted_result_json = '{"embedding":[1,0]}'
    return job


def test_sync_worker_propagates_claim_token_to_success_ack() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    worker = IndexJobWorker(engine=_engine(meta))

    metrics = worker.tick()

    assert metrics.done == 1
    assert meta.done == [("job-1", "claim-1")]


def test_sync_worker_does_not_count_lost_claim_as_done() -> None:
    meta = _Meta(_job(claim_token="claim-1"), reject_ack=True)
    worker = IndexJobWorker(engine=_engine(meta))

    metrics = worker.tick()

    assert metrics.done == 0
    assert metrics.claim_lost == 1


def test_sync_worker_replays_admitted_result_without_projection() -> None:
    meta = _Meta(_admitted_job(claim_token="claim-1"))
    engine = _engine(meta)
    worker = IndexJobWorker(engine=engine)

    metrics = worker.tick()

    assert metrics.done == 1
    assert engine.indexing.calls == 0


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


def test_async_worker_does_not_count_lost_claim_as_done() -> None:
    async def run() -> None:
        meta = _Meta(_job(claim_token="claim-1"), reject_ack=True)
        worker = AsyncIndexJobWorker(engine=_engine(meta))

        metrics = await worker.tick()

        assert metrics.done == 0
        assert metrics.claim_lost == 1

    asyncio.run(run())


def test_async_worker_replays_admitted_result_without_projection() -> None:
    async def run() -> None:
        meta = _Meta(_admitted_job(claim_token="claim-1"))
        engine = _engine(meta)
        worker = AsyncIndexJobWorker(engine=engine)

        metrics = await worker.tick()

        assert metrics.done == 1
        assert engine.indexing.calls == 0

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


def test_reconciler_does_not_count_lost_claim_as_applied() -> None:
    meta = _Meta(_job(claim_token="claim-1"), reject_ack=True)
    subsystem = _reconciler(meta)

    assert subsystem.reconcile_indexes(max_jobs=1) == 0


def test_reconciler_replays_admitted_result_without_projection() -> None:
    meta = _Meta(_admitted_job(claim_token="claim-1"))
    engine = SimpleNamespace(
        meta_sqlite=meta,
        namespace="test",
        _phase1_enable_index_jobs=True,
        _phase1_enable_validation_cache=False,
        _implementation_selection_snapshot=None,
    )
    subsystem = IndexingSubsystem(engine)
    calls = 0

    def apply_index_job(**_: object) -> None:
        nonlocal calls
        calls += 1

    subsystem.apply_index_job = apply_index_job  # type: ignore[method-assign]

    assert subsystem.reconcile_indexes(max_jobs=1) == 1
    assert calls == 0


def test_reconciler_propagates_claim_token_to_retry() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    subsystem = _reconciler(meta, fail=True)

    assert subsystem.reconcile_indexes(max_jobs=1) == 0
    assert meta.retried == [("job-1", "claim-1")]
