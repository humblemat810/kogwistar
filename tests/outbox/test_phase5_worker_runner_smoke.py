import uuid

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.workers.run_index_job_worker import main


def test_phase5_worker_runner_once_smoke(tmp_path, capsys):
    persist = tmp_path / "chroma"
    persist.mkdir(parents=True, exist_ok=True)

    # Build an engine directly to enqueue a job into its sqlite meta.
    eng = GraphKnowledgeEngine(persist_directory=str(persist))
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
    rc = main([
        "--backend", "chroma",
        "--persist-directory", str(persist),
        "--namespace", ns,
        "--once",
        "--batch-size", "1",
        "--max-jobs-per-tick", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "claimed=" in out
