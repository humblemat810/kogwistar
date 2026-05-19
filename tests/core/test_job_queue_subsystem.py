from __future__ import annotations

from kogwistar.engine_core import GraphKnowledgeEngine, JobQueueItem
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend


def _engine(tmp_path):
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
    )


def test_jobs_enqueue_uses_existing_index_jobs_store(tmp_path):
    engine = _engine(tmp_path)

    job_id = engine.jobs.enqueue(
        job_id="job-1",
        namespace="jobs:demo",
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
        payload={"workspace_id": "demo"},
    )

    rows = engine.meta_sqlite.list_index_jobs(namespace="jobs:demo")
    assert job_id == "job-1"
    assert len(rows) == 1
    assert rows[0].job_id == "job-1"
    assert rows[0].index_kind == "maintenance_job"


def test_jobs_claim_returns_typed_item_with_decoded_payload(tmp_path):
    engine = _engine(tmp_path)
    engine.jobs.enqueue(
        job_id="job-1",
        namespace="jobs:demo",
        entity_kind="projection_request",
        entity_id="entity-1",
        job_kind="projection_request",
        payload={"promoted_entity_id": "entity-1"},
    )

    claimed = engine.jobs.claim(namespace="jobs:demo", limit=1, lease_seconds=60)

    assert claimed == [
        JobQueueItem(
            job_id="job-1",
            namespace="jobs:demo",
            entity_kind="projection_request",
            entity_id="entity-1",
            job_kind="projection_request",
            op="UPSERT",
            payload={"promoted_entity_id": "entity-1"},
            retry_count=0,
            max_retries=10,
            last_error=None,
        )
    ]


def test_jobs_claim_is_namespace_isolated(tmp_path):
    engine = _engine(tmp_path)
    engine.jobs.enqueue(
        job_id="job-a",
        namespace="jobs:a",
        entity_kind="maintenance_job",
        entity_id="entity-a",
        job_kind="maintenance_job",
    )
    engine.jobs.enqueue(
        job_id="job-b",
        namespace="jobs:b",
        entity_kind="maintenance_job",
        entity_id="entity-b",
        job_kind="maintenance_job",
    )

    claimed = engine.jobs.claim(namespace="jobs:a")

    assert [job.job_id for job in claimed] == ["job-a"]
    assert [job.job_id for job in engine.jobs.list(namespace="jobs:b")] == ["job-b"]


def test_jobs_mark_done_transitions_row(tmp_path):
    engine = _engine(tmp_path)
    engine.jobs.enqueue(
        job_id="job-1",
        namespace="jobs:demo",
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
    )
    job = engine.jobs.claim(namespace="jobs:demo")[0]

    engine.jobs.mark_done(job.job_id)

    assert [row.job_id for row in engine.jobs.list(namespace="jobs:demo", status="DONE")] == ["job-1"]


def test_jobs_retry_or_fail_requeues_before_max_retries(tmp_path):
    engine = _engine(tmp_path)
    engine.jobs.enqueue(
        job_id="job-1",
        namespace="jobs:demo",
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
        max_retries=3,
    )
    job = engine.jobs.claim(namespace="jobs:demo")[0]

    engine.jobs.retry_or_fail(job, RuntimeError("boom"), max_delay_seconds=0)

    pending = engine.jobs.list(namespace="jobs:demo", status="PENDING")
    assert len(pending) == 1
    assert pending[0].retry_count == 1
    assert "RuntimeError: boom" in (pending[0].last_error or "")


def test_jobs_retry_or_fail_marks_failed_when_exhausted(tmp_path):
    engine = _engine(tmp_path)
    engine.jobs.enqueue(
        job_id="job-1",
        namespace="jobs:demo",
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
        max_retries=1,
    )
    job = engine.jobs.claim(namespace="jobs:demo")[0]

    engine.jobs.retry_or_fail(job, "terminal")

    failed = engine.jobs.list(namespace="jobs:demo", status="FAILED")
    assert [item.job_id for item in failed] == ["job-1"]
    assert failed[0].last_error == "terminal"


def test_jobs_enqueue_preserves_existing_coalescing_semantics(tmp_path):
    engine = _engine(tmp_path)

    first = engine.jobs.enqueue(
        job_id="job-1",
        namespace="jobs:demo",
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
        payload={"version": 1},
    )
    second = engine.jobs.enqueue(
        job_id="job-2",
        namespace="jobs:demo",
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
        payload={"version": 2},
    )

    rows = engine.jobs.list(namespace="jobs:demo", status="PENDING")
    assert first == second == "job-1"
    assert [row.job_id for row in rows] == ["job-1"]
    assert rows[0].payload == {"version": 2}
