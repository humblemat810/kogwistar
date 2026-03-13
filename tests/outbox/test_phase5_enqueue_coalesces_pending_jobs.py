import uuid
import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
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


@pytest.mark.integration
def test_phase5_enqueue_coalesces_pending_jobs(eng):
    ns = f"phase5_coalesce_{uuid.uuid4().hex}"

    # Enqueue many times with the same coalesce_key = entity_kind:entity_id:index_kind
    job_ids = []
    for _ in range(10):
        job_id = f"job_{uuid.uuid4().hex}"
        job_ids.append(job_id)
        eng.meta_sqlite.enqueue_index_job(
            namespace=ns,
            job_id=job_id,
            entity_kind="node",
            entity_id="n1",
            index_kind="node_docs",
            op="upsert",
            max_retries=10,
        )

    rows = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=2000)
    pending = [r for r in rows if r.status == "PENDING"]
    assert len(pending) == 1, (
        "coalescing should collapse multiple PENDING enqueues into one row"
    )

    # It should be one of the submitted job ids (implementation may keep the first or last depending on coalesce strategy)
    assert pending[0].job_id in set(job_ids)

    # Claim should return exactly one job
    claimed = eng.meta_sqlite.claim_index_jobs(limit=10, lease_seconds=60, namespace=ns)
    assert len(claimed) == 1
    assert claimed[0].job_id == pending[0].job_id

    # Mark done; no other pending jobs should exist
    eng.meta_sqlite.mark_index_job_done(claimed[0].job_id)
    rows2 = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=2000)
    pending2 = [r for r in rows2 if r.status == "PENDING"]
    assert pending2 == []
