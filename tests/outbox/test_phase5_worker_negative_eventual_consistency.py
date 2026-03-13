import uuid
import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.workers.index_job_worker import IndexJobWorker
from tests.conftest import FakeEmbeddingFunction

TEST_EMBEDDING = FakeEmbeddingFunction(dim=3)


@pytest.fixture
def eng(tmp_path) -> GraphKnowledgeEngine:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    e = GraphKnowledgeEngine(
        persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING
    )
    e._phase1_enable_index_jobs = True
    return e


def _enqueue(meta, *, ns: str, job_id: str, max_retries: int = 10) -> None:
    meta.enqueue_index_job(
        namespace=ns,
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
        max_retries=max_retries,
    )


def test_phase5_leased_job_not_claimable_until_expiry(eng):
    ns = f"phase5_leaseblock_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"
    _enqueue(eng.meta_sqlite, ns=ns, job_id=job_id)

    got1 = eng.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace=ns)
    assert len(got1) == 1
    assert got1[0].job_id == job_id
    assert got1[0].lease_until is not None

    got2 = eng.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace=ns)
    assert got2 == []


def test_phase5_lease_expiry_allows_steal(eng):
    ns = f"phase5_leasesteal_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"
    _enqueue(eng.meta_sqlite, ns=ns, job_id=job_id)

    got1 = eng.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace=ns)
    assert len(got1) == 1

    # Force lease expiry + make runnable now
    with eng.meta_sqlite.transaction() as conn:
        now = eng.meta_sqlite._now_epoch()
        conn.execute(
            "UPDATE index_jobs SET lease_until=?, next_run_at=? WHERE job_id=?",
            (now - 10, now, job_id),
        )

    got2 = eng.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace=ns)
    assert len(got2) == 1
    assert got2[0].job_id == job_id
    assert got2[0].lease_until is not None


def test_phase5_crash_after_apply_before_ack_eventually_converges(eng, monkeypatch):
    ns = f"phase5_crash_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"
    _enqueue(eng.meta_sqlite, ns=ns, job_id=job_id, max_retries=3)

    applied = set()

    def _apply(**kw):
        # projection effect should be idempotent
        applied.add((kw["entity_kind"], kw["entity_id"], kw["index_kind"], kw["op"]))

    monkeypatch.setattr(eng.indexing, "apply_index_job", lambda **kw: _apply(**kw))

    # Simulate crash after apply but before ack: make meta mark-done raise once.
    # IMPORTANT: the worker is expected to *catch* this and requeue/fail, not crash the whole process.
    real_done = eng.meta_sqlite.mark_index_job_done
    crashed = {"once": False}

    def _done_crash(*args, **kwargs):
        if not crashed["once"]:
            crashed["once"] = True
            raise RuntimeError("crash before ack")
        return real_done(*args, **kwargs)

    monkeypatch.setattr(eng.meta_sqlite, "mark_index_job_done", _done_crash)

    worker = IndexJobWorker(
        engine=eng, batch_size=1, lease_seconds=60, max_jobs_per_tick=1, namespace=ns
    )

    # First tick: apply happens, but ack fails => job should NOT be DONE.
    m1 = worker.tick()
    assert m1.claimed == 1
    # Depending on policy, this is retried or failed; for max_retries=3 we expect retried.
    assert (m1.retried + m1.failed) == 1
    assert len(applied) == 1

    rows1 = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=50)
    row1 = [r for r in rows1 if r.job_id == job_id][0]
    assert row1.status in (
        "PENDING",
        "FAILED",
    )  # SQLite meta may immediately requeue to PENDING with next_run_at
    assert row1.status != "DONE"

    # Make eligible again (skip delay/lease) and run again; should converge to DONE without duplicating effect
    with eng.meta_sqlite.transaction() as conn:
        now = eng.meta_sqlite._now_epoch()
        conn.execute(
            "UPDATE index_jobs SET status='PENDING', lease_until=NULL, next_run_at=? WHERE job_id=?",
            (now, job_id),
        )

    m2 = worker.tick()
    assert m2.done == 1
    assert len(applied) == 1  # idempotent effect (same tuple)
    rows2 = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=50)
    row2 = [r for r in rows2 if r.job_id == job_id][0]
    assert row2.status == "DONE"
