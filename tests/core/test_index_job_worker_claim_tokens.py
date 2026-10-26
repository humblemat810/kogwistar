from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.workers.async_index_job_worker import AsyncIndexJobWorker
from kogwistar.workers.index_job_worker import IndexJobWorker

_REPLAY_FIXTURE = json.loads(
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath("contracts/golden/index-job-opaque-result-replay-v1.json")
    .read_text(encoding="utf-8")
)
_OPAQUE_RESULT_CASES = _REPLAY_FIXTURE["cases"]


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


class _ProjectionStore:
    def __init__(
        self, *, reject_upsert: bool = False, delete_missing: bool = False
    ) -> None:
        self.reject_upsert = reject_upsert
        self.delete_missing = delete_missing
        self.upserts: list[tuple[str, object]] = []
        self.deletes: list[tuple[str, str]] = []

    def upsert_graph_record(self, namespace: str, record: object) -> bool | None:
        self.upserts.append((namespace, record))
        return False if self.reject_upsert else None

    def delete_graph_record(self, namespace: str, entity_id: str) -> bool:
        self.deletes.append((namespace, entity_id))
        return not self.delete_missing


class _AsyncProjectionStore:
    def __init__(
        self, *, reject_upsert: bool = False, delete_missing: bool = False
    ) -> None:
        self.reject_upsert = reject_upsert
        self.delete_missing = delete_missing
        self.upserts: list[tuple[str, object]] = []
        self.deletes: list[tuple[str, str]] = []

    async def upsert_graph_record(self, namespace: str, record: object) -> bool | None:
        self.upserts.append((namespace, record))
        return False if self.reject_upsert else None

    async def delete_graph_record(self, namespace: str, entity_id: str) -> bool:
        self.deletes.append((namespace, entity_id))
        return not self.delete_missing


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
        metadata=meta,
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


def _admitted_projection_job(*, claim_token: str | None) -> SimpleNamespace:
    action_json = (
        Path(__file__).resolve().parents[2]
        .joinpath("contracts/golden/graph-projection-action-v1.json")
        .read_text(encoding="utf-8")
        .strip()
    )
    return SimpleNamespace(
        job_id="projection-job",
        namespace="workspace",
        entity_kind="node",
        entity_id="node-1",
        index_kind="node_embedding",
        op="UPSERT",
        payload_json=None,
        retry_count=0,
        max_retries=3,
        claim_token=claim_token,
        accepted_result_json=action_json,
        accepted_result_sha256=hashlib.sha256(action_json.encode("utf-8")).hexdigest(),
    )


def test_sync_worker_propagates_claim_token_to_success_ack() -> None:
    meta = _Meta(_job(claim_token="claim-1"))
    worker = IndexJobWorker(engine=_engine(meta))

    metrics = worker.tick()

    assert metrics.done == 1
    assert meta.done == [("job-1", "claim-1")]


def test_sync_worker_applies_rust_projection_action_before_ack() -> None:
    meta = _Meta(_admitted_projection_job(claim_token="claim-1"))
    graph = _ProjectionStore()
    worker = IndexJobWorker(
        engine=_engine(meta), namespace="workspace", projection_store=graph
    )

    metrics = worker.tick()

    assert metrics.done == 1
    assert meta.done == [("projection-job", "claim-1")]
    assert len(graph.upserts) == 1
    assert graph.upserts[0][0] == "workspace"
    assert getattr(graph.upserts[0][1], "id") == "node-1"


def test_sync_worker_retries_when_graph_store_rejects_projection_upsert() -> None:
    meta = _Meta(_admitted_projection_job(claim_token="claim-1"))
    graph = _ProjectionStore(reject_upsert=True)
    worker = IndexJobWorker(
        engine=_engine(meta), namespace="workspace", projection_store=graph
    )

    metrics = worker.tick()

    assert metrics.done == 0
    assert metrics.retried == 1
    assert meta.done == []
    assert meta.retried == [("projection-job", "claim-1")]


def test_sync_worker_acknowledges_delete_when_record_is_already_absent() -> None:
    job = _admitted_projection_job(claim_token="claim-1")
    job.op = "DELETE"
    action = json.loads(job.accepted_result_json)
    action.update(operation="DELETE", record=None)
    job.accepted_result_json = json.dumps(action, separators=(",", ":"))
    job.accepted_result_sha256 = hashlib.sha256(
        job.accepted_result_json.encode("utf-8")
    ).hexdigest()
    meta = _Meta(job)
    graph = _ProjectionStore(delete_missing=True)
    worker = IndexJobWorker(
        engine=_engine(meta), namespace="workspace", projection_store=graph
    )

    metrics = worker.tick()

    assert metrics.done == 1
    assert metrics.retried == 0
    assert meta.done == [("projection-job", "claim-1")]
    assert graph.deletes == [("workspace", "node-1")]


def test_sync_worker_does_not_ack_projection_action_without_store() -> None:
    meta = _Meta(_admitted_projection_job(claim_token="claim-1"))
    worker = IndexJobWorker(engine=_engine(meta), namespace="workspace")

    metrics = worker.tick()

    assert metrics.done == 0
    assert metrics.retried == 1
    assert meta.done == []
    assert meta.retried == [("projection-job", "claim-1")]


def test_sync_worker_does_not_misclassify_malformed_action_as_legacy_result() -> None:
    job = _admitted_projection_job(claim_token="claim-1")
    action = json.loads(job.accepted_result_json)
    action.pop("job_id")
    job.accepted_result_json = json.dumps(action, separators=(",", ":"))
    job.accepted_result_sha256 = hashlib.sha256(
        job.accepted_result_json.encode("utf-8")
    ).hexdigest()
    meta = _Meta(job)
    worker = IndexJobWorker(engine=_engine(meta), namespace="workspace")

    metrics = worker.tick()

    assert metrics.done == 0
    assert metrics.retried == 1
    assert meta.done == []


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


@pytest.mark.parametrize(
    "case", _OPAQUE_RESULT_CASES, ids=lambda case: str(case["name"])
)
def test_sync_worker_preserves_opaque_legacy_accepted_result_string(
    case: dict[str, str],
) -> None:
    job = _admitted_job(claim_token="claim-1")
    job.accepted_result_json = case["result_json"]
    meta = _Meta(job)
    worker = IndexJobWorker(engine=_engine(meta))

    metrics = worker.tick()

    assert metrics.done == 1
    assert metrics.retried == 0
    assert meta.done == [("job-1", "claim-1")]
    assert meta.retried == []


def test_sync_worker_does_not_ack_non_string_accepted_result() -> None:
    job = _admitted_job(claim_token="claim-1")
    job.accepted_result_json = {"embedding": [1, 0]}
    meta = _Meta(job)
    worker = IndexJobWorker(engine=_engine(meta))

    metrics = worker.tick()

    assert metrics.done == 0
    assert metrics.retried == 1
    assert meta.done == []
    assert meta.retried == [("job-1", "claim-1")]


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


def test_async_worker_applies_rust_projection_action_before_ack() -> None:
    async def run() -> None:
        meta = _Meta(_admitted_projection_job(claim_token="claim-1"))
        graph = _AsyncProjectionStore()
        worker = AsyncIndexJobWorker(
            engine=_engine(meta), namespace="workspace", projection_store=graph
        )

        metrics = await worker.tick()

        assert metrics.done == 1
        assert meta.done == [("projection-job", "claim-1")]
        assert len(graph.upserts) == 1
        assert graph.upserts[0][0] == "workspace"
        assert getattr(graph.upserts[0][1], "id") == "node-1"

    asyncio.run(run())


def test_async_worker_retries_when_graph_store_rejects_projection_upsert() -> None:
    async def run() -> None:
        meta = _Meta(_admitted_projection_job(claim_token="claim-1"))
        graph = _AsyncProjectionStore(reject_upsert=True)
        worker = AsyncIndexJobWorker(
            engine=_engine(meta), namespace="workspace", projection_store=graph
        )

        metrics = await worker.tick()

        assert metrics.done == 0
        assert metrics.retried == 1
        assert meta.done == []
        assert meta.retried == [("projection-job", "claim-1")]

    asyncio.run(run())


def test_async_worker_acknowledges_delete_when_record_is_already_absent() -> None:
    async def run() -> None:
        job = _admitted_projection_job(claim_token="claim-1")
        job.op = "DELETE"
        action = json.loads(job.accepted_result_json)
        action.update(operation="DELETE", record=None)
        job.accepted_result_json = json.dumps(action, separators=(",", ":"))
        job.accepted_result_sha256 = hashlib.sha256(
            job.accepted_result_json.encode("utf-8")
        ).hexdigest()
        meta = _Meta(job)
        graph = _AsyncProjectionStore(delete_missing=True)
        worker = AsyncIndexJobWorker(
            engine=_engine(meta), namespace="workspace", projection_store=graph
        )

        metrics = await worker.tick()

        assert metrics.done == 1
        assert metrics.retried == 0
        assert meta.done == [("projection-job", "claim-1")]
        assert graph.deletes == [("workspace", "node-1")]

    asyncio.run(run())


def test_async_worker_does_not_ack_projection_action_without_store() -> None:
    async def run() -> None:
        meta = _Meta(_admitted_projection_job(claim_token="claim-1"))
        worker = AsyncIndexJobWorker(engine=_engine(meta), namespace="workspace")

        metrics = await worker.tick()

        assert metrics.done == 0
        assert metrics.retried == 1
        assert meta.done == []
        assert meta.retried == [("projection-job", "claim-1")]

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


@pytest.mark.parametrize(
    "case", _OPAQUE_RESULT_CASES, ids=lambda case: str(case["name"])
)
def test_async_worker_preserves_opaque_legacy_accepted_result_string(
    case: dict[str, str],
) -> None:
    async def run() -> None:
        job = _admitted_job(claim_token="claim-1")
        job.accepted_result_json = case["result_json"]
        meta = _Meta(job)
        worker = AsyncIndexJobWorker(engine=_engine(meta))

        metrics = await worker.tick()

        assert metrics.done == 1
        assert metrics.retried == 0
        assert meta.done == [("job-1", "claim-1")]
        assert meta.retried == []

    asyncio.run(run())


def test_async_worker_does_not_ack_non_string_accepted_result() -> None:
    async def run() -> None:
        job = _admitted_job(claim_token="claim-1")
        job.accepted_result_json = {"embedding": [1, 0]}
        meta = _Meta(job)
        worker = AsyncIndexJobWorker(engine=_engine(meta))

        metrics = await worker.tick()

        assert metrics.done == 0
        assert metrics.retried == 1
        assert meta.done == []
        assert meta.retried == [("job-1", "claim-1")]

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


def test_async_batch_worker_propagates_cancellation_without_queue_mutation() -> None:
    first = _job(claim_token="claim-1")
    second = _job(claim_token="claim-2")
    for job, job_id in ((first, "job-1"), (second, "job-2")):
        job.job_id = job_id
        job.index_kind = "node_embedding"
        job.payload_json = '{"embedding_model":"test"}'

    class BatchMeta(_Meta):
        def claim_index_jobs(self, **_: object) -> list[SimpleNamespace]:
            self.claimed = True
            return [first, second]

    class BatchAdapter:
        async def apply_embedding_jobs_batch(
            self, _jobs: list[object]
        ) -> dict[str, BaseException | None]:
            raise asyncio.CancelledError

    meta = BatchMeta(first)
    engine = _engine(meta)
    engine.async_two_stage_projection_adapter = BatchAdapter()
    worker = AsyncIndexJobWorker(engine=engine)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.tick())

    assert meta.done == []
    assert meta.retried == []
    assert meta.failed == []


def test_sync_batch_adapter_cannot_mutate_claim_identity_or_ack_token() -> None:
    first = vars(_job(claim_token="claim-1"))
    second = vars(_job(claim_token="claim-2"))
    for job, job_id in ((first, "job-1"), (second, "job-2")):
        job["job_id"] = job_id
        job["index_kind"] = "node_embedding"
        job["payload_json"] = '{"embedding_model":"test"}'

    class BatchMeta(_Meta):
        def claim_index_jobs(self, **_: object) -> list[dict[str, object]]:
            if self.claimed:
                return []
            self.claimed = True
            return [first, second]

    class MutatingBatchAdapter:
        def apply_embedding_jobs_batch(
            self, jobs: list[object]
        ) -> dict[str, BaseException | None]:
            for job in jobs:
                assert isinstance(job, dict)
                job["job_id"] = "attacker-job"
                job["claim_token"] = "attacker-token"
            return {"job-1": None, "job-2": None}

    meta = BatchMeta(first)
    engine = _engine(meta)
    engine.two_stage_projection_adapter = MutatingBatchAdapter()

    metrics = IndexJobWorker(engine=engine).tick()

    assert metrics.done == 2
    assert meta.done == [("job-1", "claim-1"), ("job-2", "claim-2")]
    assert first["job_id"] == "job-1" and first["claim_token"] == "claim-1"
    assert second["job_id"] == "job-2" and second["claim_token"] == "claim-2"


def test_async_batch_adapter_cannot_mutate_claim_identity_or_ack_token() -> None:
    first = vars(_job(claim_token="claim-1"))
    second = vars(_job(claim_token="claim-2"))
    for job, job_id in ((first, "job-1"), (second, "job-2")):
        job["job_id"] = job_id
        job["index_kind"] = "node_embedding"
        job["payload_json"] = '{"embedding_model":"test"}'

    class BatchMeta(_Meta):
        def claim_index_jobs(self, **_: object) -> list[dict[str, object]]:
            if self.claimed:
                return []
            self.claimed = True
            return [first, second]

    class MutatingBatchAdapter:
        async def apply_embedding_jobs_batch(
            self, jobs: list[object]
        ) -> dict[str, BaseException | None]:
            for job in jobs:
                assert isinstance(job, dict)
                job["job_id"] = "attacker-job"
                job["claim_token"] = "attacker-token"
            return {"job-1": None, "job-2": None}

    meta = BatchMeta(first)
    engine = _engine(meta)
    engine.async_two_stage_projection_adapter = MutatingBatchAdapter()

    metrics = asyncio.run(AsyncIndexJobWorker(engine=engine).tick())

    assert metrics.done == 2
    assert meta.done == [("job-1", "claim-1"), ("job-2", "claim-2")]
    assert first["job_id"] == "job-1" and first["claim_token"] == "claim-1"
    assert second["job_id"] == "job-2" and second["claim_token"] == "claim-2"


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
        metadata=meta,
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
