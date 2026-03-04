import uuid
import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.storage_backend import ChromaBackend
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend

EMBEDDING_DIM = 3

def _emb(*args, **kwargs):
    return [0.1] * EMBEDDING_DIM

@pytest.fixture(params=["chroma", "pgvector"], ids=["chroma", "pgvector"])
def e2e_engine(request, tmp_path, sa_engine, pg_schema) -> GraphKnowledgeEngine:
    if request.param == "chroma":
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
    else:
        pytest.importorskip("pgvector")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=EMBEDDING_DIM, schema=pg_schema)
        eng = GraphKnowledgeEngine(backend=backend)

    eng._test_backend_kind = request.param  # type: ignore[attr-defined]
    return eng


def _count_events(eng: GraphKnowledgeEngine, ns: str) -> int:
    return sum(1 for _ in eng.meta_sqlite.iter_entity_events(namespace=ns, from_seq=1))


def _count_jobs(eng: GraphKnowledgeEngine, ns: str) -> int:
    return len(eng.meta_sqlite.list_index_jobs(namespace=ns, limit=1000))


def test_phase3_meta_atomicity_event_and_job_rollback(e2e_engine):
    """
    Inside engine.uow():
      - append entity_events
      - enqueue index_jobs
    then raise => BOTH tables must have 0 rows for that namespace.
    """
    eng = e2e_engine
    eng._ef._emb = _emb  # keep consistent with other E2E tests

    ns = f"phase3_atomic_{uuid.uuid4().hex}"

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0

    with pytest.raises(RuntimeError):
        with eng.uow():
            eng.meta_sqlite.append_entity_event(
                namespace=ns,
                event_id=f"ev_{uuid.uuid4().hex}",
                entity_kind="node",
                entity_id="n1",
                op="upsert",
                payload_json='{"id":"n1","dummy":true}',
            )
            eng.meta_sqlite.enqueue_index_job(
                namespace=ns,
                job_id=f"job_{uuid.uuid4().hex}",
                entity_kind="node",
                entity_id="n1",
                index_kind="node_index",
                op="upsert",
                payload_json='{"id":"n1"}',
            )
            raise RuntimeError("boom")

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0


def test_phase3_nested_uow_inner_does_not_commit_if_outer_rolls_back(e2e_engine):
    eng = e2e_engine
    eng._ef._emb = _emb

    ns = f"phase3_nested_{uuid.uuid4().hex}"

    with pytest.raises(RuntimeError):
        with eng.uow():
            with eng.uow():
                eng.meta_sqlite.append_entity_event(
                    namespace=ns,
                    event_id=f"ev_{uuid.uuid4().hex}",
                    entity_kind="edge",
                    entity_id="e1",
                    op="upsert",
                    payload_json='{"id":"e1","dummy":true}',
                )
                eng.meta_sqlite.enqueue_index_job(
                    namespace=ns,
                    job_id=f"job_{uuid.uuid4().hex}",
                    entity_kind="edge",
                    entity_id="e1",
                    index_kind="edge_index",
                    op="upsert",
                    payload_json='{"id":"e1"}',
                )
            # Outer fails after inner block returns
            raise RuntimeError("boom")

    assert _count_events(eng, ns) == 0
    assert _count_jobs(eng, ns) == 0
