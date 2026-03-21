import argparse
import uuid

import pytest

import kogwistar.workers.run_index_job_worker as worker_runner
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.workers.run_index_job_worker import main
from tests.conftest import FakeEmbeddingFunction

TEST_EMBEDDING = FakeEmbeddingFunction(dim=3)


def test_phase5_worker_runner_once_smoke(tmp_path, capsys):
    persist = tmp_path / "chroma"
    persist.mkdir(parents=True, exist_ok=True)

    # Build an engine directly to enqueue a job into its sqlite meta.
    eng = GraphKnowledgeEngine(
        persist_directory=str(persist), embedding_function=TEST_EMBEDDING
    )
    ns = f"phase5_runner_{uuid.uuid4().hex}"
    eng.meta_sqlite.enqueue_index_job(
        namespace=ns,
        job_id=f"job_{uuid.uuid4().hex}",
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="upsert",
    )

    # Run worker once via CLI entrypoint. It will build a NEW engine pointing to same persist dir.
    # We don't assert job completion here (engine.apply may be a no-op depending on configuration),
    # just that it runs without crashing and prints metrics.
    real_kogwistar = worker_runner.GraphKnowledgeEngine

    def _fake_kogwistar(*args, **kwargs):
        kwargs.setdefault("embedding_function", TEST_EMBEDDING)
        return real_kogwistar(*args, **kwargs)

    worker_runner.GraphKnowledgeEngine = _fake_kogwistar
    try:
        rc = main(
            [
                "--backend",
                "chroma",
                "--persist-directory",
                str(persist),
                "--namespace",
                ns,
                "--once",
                "--batch-size",
                "1",
                "--max-jobs-per-tick",
                "1",
            ]
        )
    finally:
        worker_runner.GraphKnowledgeEngine = real_kogwistar
    assert rc == 0
    out = capsys.readouterr().out
    assert "claimed=" in out


@pytest.mark.parametrize("backend_alias", ["pg", "pgvector", "postgres"])
def test_phase5_worker_runner_pg_aliases_build_pg_backend(
    monkeypatch, backend_alias: str
):
    sqlalchemy = pytest.importorskip("sqlalchemy")

    class FakePgVectorBackend:
        def __init__(self, *, engine, embedding_dim, schema):
            self.engine = engine
            self.embedding_dim = embedding_dim
            self.schema = schema

    class FakeGraphKnowledgeEngine:
        def __init__(self, persist_directory=None, backend=None):
            self.persist_directory = persist_directory
            self.backend = backend
            self.namespace = None

    monkeypatch.setattr(worker_runner, "PgVectorBackend", FakePgVectorBackend)
    monkeypatch.setattr(worker_runner, "GraphKnowledgeEngine", FakeGraphKnowledgeEngine)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda dsn: f"sa:{dsn}")

    args = argparse.Namespace(
        backend=backend_alias,
        namespace="ns_alias",
        persist_directory=None,
        pg_url="postgresql://example/test",
        pg_schema="test_schema",
        embedding_dim="8",
    )

    eng = worker_runner._build_engine(args)
    assert isinstance(eng.backend, FakePgVectorBackend)
    assert eng.namespace == "ns_alias"
    assert eng.backend.engine == "sa:postgresql://example/test"
    assert eng.backend.embedding_dim == 8
    assert eng.backend.schema == "test_schema"
