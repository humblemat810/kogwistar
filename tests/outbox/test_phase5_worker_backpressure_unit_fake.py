import pytest
import json
from dataclasses import dataclass
from typing import List, Literal, Optional

from kogwistar.workers.index_job_worker import IndexJobWorker


@dataclass
class _Job:
    job_id: str
    entity_kind: str = "node"
    entity_id: str = "n1"
    index_kind: str = "node_docs"
    op: str = "upsert"
    retry_count: int = 0
    max_retries: int = 10
    lease_until: Optional[int] = None
    payload_json: Optional[str] = None


class _FakeMeta:
    def __init__(self, jobs: List[_Job]):
        self._jobs = jobs[:]  # queue
        self.claim_calls: list[int] = []
        self.done: set[str] = set()

    def claim_index_jobs(self, *, limit: int, lease_seconds: int, namespace: str):
        # record backpressure behavior
        self.claim_calls.append(limit)
        out = []
        while self._jobs and len(out) < limit:
            out.append(self._jobs.pop(0))
        return out

    def mark_index_job_done(self, job_id: str):
        self.done.add(job_id)

    def bump_retry_and_requeue(
        self, job_id: str, err: str, next_run_at_seconds: int = 1
    ):
        raise AssertionError("not used in backpressure unit tests")

    def mark_index_job_failed(self, job_id: str, err: str, final: bool = True):
        raise AssertionError("not used in backpressure unit tests")


class FakeIndexing:
    @property
    def applied(self):
        return self.engine.applied

    def __init__(self, engine):
        self.engine = engine

    def apply_index_job(
        self,
        *,
        job_id: str,
        entity_kind: str,
        entity_id: str,
        index_kind: str,
        op: str,
        namespace: str,
        payload_json=None,
        validated_entity_cache=None,
    ):
        # record that we processed this job
        self.applied.append(job_id)


class _FakeEngine:
    def __init__(self, jobs: List[_Job], namespace: str = "default"):
        self.meta_sqlite = _FakeMeta(jobs)
        self.namespace = namespace
        self.indexing = FakeIndexing(self)
        self.applied: list[str] = []


@pytest.mark.ci
@pytest.mark.parametrize(
    "batch_size,max_jobs_per_tick,max_inflight,n_jobs,expected_processed,expected_claim_limits",
    [
        # single claim, limited by max_jobs_per_tick
        (50, 3, 50, 10, 3, [3]),
        # batch size smaller than max_jobs_per_tick => multiple claims
        (2, 5, 50, 10, 5, [2, 2, 1]),
        # max_inflight must not cap the provider batch size
        (50, 10, 3, 10, 10, [10]),
        # max_jobs_per_tick caps total, independent of max_inflight
        (2, 3, 1, 10, 3, [2, 1]),
    ],
)
def test_phase5_worker_backpressure_respected_unit_fake(
    batch_size: Literal[50] | Literal[2],
    max_jobs_per_tick,
    max_inflight,
    n_jobs,
    expected_processed,
    expected_claim_limits,
):
    jobs = [_Job(job_id=f"j{i}") for i in range(n_jobs)]
    eng = _FakeEngine(jobs)
    worker = IndexJobWorker(
        engine=eng,
        batch_size=batch_size,
        max_jobs_per_tick=max_jobs_per_tick,
        max_inflight=max_inflight,
        lease_seconds=60,
        namespace="ns",
    )

    m = worker.tick()

    assert m.claimed == expected_processed
    assert m.done == expected_processed
    assert len(eng.applied) == expected_processed
    assert eng.meta_sqlite.claim_calls == expected_claim_limits
    # done set matches processed
    assert eng.meta_sqlite.done == set(eng.applied)


def test_batch_size_is_not_capped_by_max_inflight():
    jobs = [_Job(job_id=f"embed-{i}", index_kind="node_embedding") for i in range(3)]
    eng = _FakeEngine(jobs)

    class _BatchAdapter:
        def __init__(self):
            self.calls: list[list[str]] = []

        def apply_embedding_jobs_batch(self, batch):
            self.calls.append([job.job_id for job in batch])
            return {job.job_id: None for job in batch}

    adapter = _BatchAdapter()
    eng.two_stage_projection_adapter = adapter
    worker = IndexJobWorker(
        engine=eng,
        batch_size=3,
        max_inflight=1,
        max_jobs_per_tick=3,
        lease_seconds=60,
        namespace="ns",
    )

    metrics = worker.tick()

    assert metrics.done == 3
    assert adapter.calls == [["embed-0", "embed-1", "embed-2"]]


@pytest.mark.ci
def test_mixed_claim_batches_embeddings_then_applies_other_jobs_individually():
    jobs = [
        _Job(job_id="projection-1", index_kind="node_docs"),
        _Job(
            job_id="embed-1",
            index_kind="node_embedding",
            payload_json=json.dumps({"embedding_model": "fake-v1"}),
        ),
        _Job(
            job_id="embed-2",
            index_kind="node_embedding",
            payload_json=json.dumps({"embedding_model": "fake-v1"}),
        ),
        _Job(job_id="projection-2", index_kind="edge_endpoints"),
    ]
    eng = _FakeEngine(jobs)

    class _BatchAdapter:
        def __init__(self):
            self.calls: list[list[str]] = []

        def apply_embedding_jobs_batch(self, batch):
            self.calls.append([job.job_id for job in batch])
            return {job.job_id: None for job in batch}

    adapter = _BatchAdapter()
    eng.two_stage_projection_adapter = adapter
    worker = IndexJobWorker(
        engine=eng,
        batch_size=10,
        max_jobs_per_tick=10,
        lease_seconds=60,
        namespace="ns",
    )

    metrics = worker.tick()

    assert metrics.done == 4
    assert adapter.calls == [["embed-1", "embed-2"]]
    # Non-embedding work uses the ordinary one-job path, after embedding.
    assert eng.applied == ["projection-1", "projection-2"]


@pytest.mark.ci
def test_batch_adapter_failure_isolated_to_jobs_and_does_not_escape_tick():
    jobs = [
        _Job(
            job_id="embed-1",
            index_kind="node_embedding",
            max_retries=3,
            payload_json=json.dumps({"embedding_model": "fake-v1"}),
        ),
        _Job(
            job_id="embed-2",
            index_kind="node_embedding",
            max_retries=3,
            payload_json=json.dumps({"embedding_model": "fake-v1"}),
        ),
    ]
    eng = _FakeEngine(jobs)

    class _RetryMeta(_FakeMeta):
        def bump_retry_and_requeue(self, job_id, err, next_run_at_seconds=1):
            self.done.add(f"retry:{job_id}")

    eng.meta_sqlite = _RetryMeta(jobs)

    class _FailingBatchAdapter:
        def apply_embedding_jobs_batch(self, _batch):
            raise RuntimeError("provider unavailable")

    eng.two_stage_projection_adapter = _FailingBatchAdapter()
    worker = IndexJobWorker(
        engine=eng,
        batch_size=10,
        max_jobs_per_tick=10,
        lease_seconds=60,
        namespace="ns",
    )

    metrics = worker.tick()

    assert metrics.claimed == 2
    assert metrics.done == 0
    assert metrics.retried == 2
    assert metrics.failed == 0


@pytest.mark.ci
def test_partial_batch_outcomes_acknowledge_members_independently():
    jobs = [
        _Job(
            job_id="embed-ok",
            index_kind="node_embedding",
            max_retries=3,
            payload_json=json.dumps({"embedding_model": "fake-v1"}),
        ),
        _Job(
            job_id="embed-fail",
            index_kind="node_embedding",
            max_retries=3,
            payload_json=json.dumps({"embedding_model": "fake-v1"}),
        ),
    ]
    eng = _FakeEngine(jobs)

    class _RetryMeta(_FakeMeta):
        def bump_retry_and_requeue(self, job_id, err, next_run_at_seconds=1):
            self.done.add(f"retry:{job_id}")

    eng.meta_sqlite = _RetryMeta(jobs)

    class _PartialBatchAdapter:
        def apply_embedding_jobs_batch(self, batch):
            return {
                batch[0].job_id: None,
                batch[1].job_id: RuntimeError("one document failed"),
            }

    eng.two_stage_projection_adapter = _PartialBatchAdapter()
    worker = IndexJobWorker(
        engine=eng,
        batch_size=10,
        max_jobs_per_tick=10,
        lease_seconds=60,
        namespace="ns",
    )

    metrics = worker.tick()

    assert metrics.done == 1
    assert metrics.retried == 1
    assert eng.meta_sqlite.done == {"embed-ok", "retry:embed-fail"}
