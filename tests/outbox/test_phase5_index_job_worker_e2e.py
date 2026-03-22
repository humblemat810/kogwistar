import uuid
import pytest
pytestmark = pytest.mark.ci_full

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.workers.index_job_worker import IndexJobWorker
from tests.conftest import FakeEmbeddingFunction

TEST_EMBEDDING = FakeEmbeddingFunction(dim=3)


@pytest.fixture
def eng(tmp_path) -> GraphKnowledgeEngine:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    e = GraphKnowledgeEngine(
        persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING
    )
    e._phase1_enable_index_jobs = True  # ensure reconcile pipeline enabled
    return e


def _get_job(meta, job_id: str, ns: str):
    rows = meta.list_index_jobs(namespace=ns, limit=1000)
    for r in rows:
        if r.job_id == job_id:
            return r
    raise AssertionError(f"job {job_id} not found")


def test_phase5_job_claim_sets_lease_until(eng):
    ns = f"phase5_claim_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"

    eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
    )

    jobs = eng.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace=ns)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.job_id == job_id
    assert j.status == "DOING"
    assert j.lease_until is not None


def test_phase5_job_fail_increments_retry_and_requeues(eng, monkeypatch):
    ns = f"phase5_retry_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"

    eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
        max_retries=10,
    )

    def _boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(eng.indexing, "apply_index_job", lambda **kw: _boom(**kw))

    worker = IndexJobWorker(
        engine=eng, batch_size=1, lease_seconds=60, max_jobs_per_tick=1, namespace=ns
    )
    m = worker.tick()
    assert m.claimed == 1
    assert m.retried == 1
    assert m.failed == 0

    j = _get_job(eng.meta_sqlite, job_id, ns)
    assert j.status == "PENDING"
    assert j.retry_count == 1
    assert j.next_run_at is not None
    assert j.next_run_at > eng.meta_sqlite._now_epoch()  # should be delayed


def test_phase5_job_exceeds_max_retry_becomes_dlq_terminal(eng, monkeypatch):
    ns = f"phase5_dlq_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"

    eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
        max_retries=2,
    )

    monkeypatch.setattr(
        eng.indexing,
        "apply_index_job",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    worker = IndexJobWorker(
        engine=eng, batch_size=1, lease_seconds=60, max_jobs_per_tick=1, namespace=ns
    )

    # 1st failure => retry
    m1 = worker.tick()
    assert m1.retried == 1

    # Make eligible immediately (simulate time passing)
    with eng.meta_sqlite.transaction() as conn:
        now = eng.meta_sqlite._now_epoch()
        conn.execute(
            "UPDATE index_jobs SET next_run_at=? WHERE job_id=?", (now, job_id)
        )

    # 2nd failure => DLQ (FAILED terminal)
    m2 = worker.tick()
    assert m2.failed == 1

    j = _get_job(eng.meta_sqlite, job_id, ns)
    assert j.status == "FAILED"

    # Ensure it's never reclaimed
    jobs = eng.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace=ns)
    assert jobs == []


def test_phase5_coalesce_uniqueness_holds_no_duplicate_pending(eng):
    ns = f"phase5_coalesce_{uuid.uuid4().hex}"
    j1 = f"job_{uuid.uuid4().hex}"
    j2 = f"job_{uuid.uuid4().hex}"

    jid1 = eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=j1,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
    )
    jid2 = eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=j2,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
    )

    assert jid1 == j1
    assert jid2 == j1  # coalesced into first pending job

    rows = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=1000)
    assert len([r for r in rows if r.status == "PENDING"]) == 1


def test_phase5_idempotent_apply_under_at_least_once(eng, monkeypatch):
    ns = f"phase5_idem_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"

    applied = set()

    def _apply(**kw):
        # idempotent write to a set (projection side-effect)
        applied.add((kw["entity_kind"], kw["entity_id"], kw["index_kind"], kw["op"]))

    monkeypatch.setattr(eng.indexing, "apply_index_job", lambda **kw: _apply(**kw))

    eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
    )

    worker = IndexJobWorker(
        engine=eng, batch_size=1, lease_seconds=60, max_jobs_per_tick=1, namespace=ns
    )

    m1 = worker.tick()
    assert m1.done == 1
    assert len(applied) == 1

    # Simulate at-least-once: same job gets re-delivered (e.g., lease stolen after crash)
    with eng.meta_sqlite.transaction() as conn:
        now = eng.meta_sqlite._now_epoch()
        conn.execute(
            "UPDATE index_jobs SET status='PENDING', lease_until=NULL, next_run_at=NULL, updated_at=? WHERE job_id=?",
            (now, job_id),
        )

    m2 = worker.tick()
    assert m2.done == 1
    assert len(applied) == 1  # idempotent apply
