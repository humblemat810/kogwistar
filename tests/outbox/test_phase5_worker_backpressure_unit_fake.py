import pytest
from dataclasses import dataclass
from typing import List, Literal, Optional

from graph_knowledge_engine.workers.index_job_worker import IndexJobWorker


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

    def bump_retry_and_requeue(self, job_id: str, err: str, next_run_at_seconds: int = 1):
        raise AssertionError("not used in backpressure unit tests")

    def mark_index_job_failed(self, job_id: str, err: str, final: bool = True):
        raise AssertionError("not used in backpressure unit tests")


class FakeIndexing:
    @property
    def applied(self):
        return self.engine.applied
    def __init__(self, engine):
        self.engine = engine
    def apply_index_job(self, *, job_id: str, entity_kind: str, entity_id: str, index_kind: str, op: str, namespace: str):
        # record that we processed this job
        self.applied.append(job_id)
    
class _FakeEngine:
    def __init__(self, jobs: List[_Job], namespace: str = "default"):
        self.meta_sqlite = _FakeMeta(jobs)
        self.namespace = namespace
        self.indexing = FakeIndexing(self)
        self.applied: list[str] = []

@pytest.mark.unit
@pytest.mark.parametrize(
    "batch_size,max_jobs_per_tick,max_inflight,n_jobs,expected_processed,expected_claim_limits",
    [
        # single claim, limited by max_jobs_per_tick
        (50, 3, 50, 10, 3, [3]),
        # batch size smaller than max_jobs_per_tick => multiple claims
        (2, 5, 50, 10, 5, [2, 2, 1]),
        # max_inflight caps claim size even if batch_size larger
        (50, 10, 3, 10, 10, [3, 3, 3, 1]),
        # max_jobs_per_tick caps total even if lots available + inflight=1 forces limit 1
        (2, 3, 1, 10, 3, [1, 1, 1]),
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
