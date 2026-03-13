import uuid
import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.workers.index_job_worker import IndexJobWorker
from tests.conftest import FakeEmbeddingFunction

TEST_EMBEDDING = FakeEmbeddingFunction(dim=3)


def _enqueue_jobs(meta, *, ns: str, n: int) -> list[str]:
    """Enqueue N distinct jobs.

    IMPORTANT: EngineSQLite/PG meta coalesce by (entity_kind, entity_id, index_kind) while status=PENDING.
    So we must vary entity_id (or index_kind) per job to avoid collapsing into a single pending job.
    """
    ids = []
    for i in range(n):
        job_id = f"job_{uuid.uuid4().hex}"
        meta.enqueue_index_job(
            namespace=ns,
            job_id=job_id,
            entity_kind="node",
            entity_id=f"n{i}",          # <-- avoid coalescing
            index_kind="node_docs",
            op="upsert",
            max_retries=10,
        )
        ids.append(job_id)
    return ids


@pytest.fixture(params=["chroma", "pg"], ids=["chroma", "pg"])
def eng(request, tmp_path, sa_engine, pg_schema) -> GraphKnowledgeEngine:
    """Backend-parametrized engine fixture for Phase 5 worker semantics."""
    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        e = GraphKnowledgeEngine(persist_directory=str(persist_dir), embedding_function=TEST_EMBEDDING)
        e._phase1_enable_index_jobs = True
        return e

    pytest.importorskip("pgvector")
    from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend

    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    backend.ensure_schema()
    e = GraphKnowledgeEngine(backend=backend, embedding_function=TEST_EMBEDDING)
    e._phase1_enable_index_jobs = True
    return e


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size,max_jobs_per_tick,max_inflight,n_jobs",
    [
        pytest.param(50, 3, 50, 10, id="cap_by_max_jobs"),
        pytest.param(2, 5, 50, 10, id="small_batch_multiple_claims"),
        pytest.param(50, 10, 3, 10, id="small_inflight_multiple_claims"),
        pytest.param(2, 3, 1, 10, id="tight_inflight_and_tick"),
    ],
)
def test_phase5_worker_backpressure_integration_respects_caps(
    eng, monkeypatch, batch_size, max_jobs_per_tick, max_inflight, n_jobs
):
    ns = f"phase5_bp_int_{uuid.uuid4().hex}"
    _enqueue_jobs(eng.meta_sqlite, ns=ns, n=n_jobs)

    applied = set()
    monkeypatch.setattr(eng.indexing, "apply_index_job", lambda **kw: applied.add(kw["job_id"]))

    worker = IndexJobWorker(
        engine=eng,
        batch_size=batch_size,
        max_jobs_per_tick=max_jobs_per_tick,
        max_inflight=max_inflight,
        lease_seconds=60,
        namespace=ns,
    )

    m = worker.tick()
    expected = min(n_jobs, max_jobs_per_tick)

    assert m.claimed == expected
    assert m.done == expected
    assert len(applied) == expected

    rows = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=2000)
    done = [r for r in rows if r.status == "DONE"]
    pending = [r for r in rows if r.status == "PENDING"]
    doing = [r for r in rows if r.status == "DOING"]

    assert len(done) == expected
    assert len(doing) == 0
    assert len(pending) == n_jobs - expected
