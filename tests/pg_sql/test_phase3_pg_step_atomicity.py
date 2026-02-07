
import os
import pytest

# NOTE: this test uses the v2 patched engine/backend copies living in /mnt/data.
from graph_knowledge_engine.engine_postgres import PgVectorBackend
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.postgres_backend import PgVectorConfig, build_postgres_backend

@pytest.mark.integration
def test_pg_uow_rolls_back_backend_writes(sa_engine, pg_schema):
    # pg_url = os.getenv("TEST_PG_URL")
    # if not pg_url:
    #     pytest.skip("Set TEST_PG_URL to run Postgres integration test")
    be = PgVectorBackend(engine=sa_engine, embedding_dim=8, schema=pg_schema)
    # backend = build_postgres_backend(PgVectorConfig(url=pg_url))
    engine = GraphKnowledgeEngine(backend=be)
    be.ensure_schema()
    nid = "n_test_atomicity"
    engine.backend.node_delete(ids=[nid])

    with pytest.raises(RuntimeError):
        with engine.uow():
            engine.backend.node_add(ids=[nid], documents=["doc"], metadatas=[{"a": 1}], embeddings=None)
            raise RuntimeError("boom")

    got = engine.backend.node_get(ids=[nid], include=["documents", "metadatas"])
    assert got.get("ids") in ([], None) or len(got.get("ids", [])) == 0
