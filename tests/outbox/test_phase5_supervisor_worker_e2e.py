import os
import sys
import time
import uuid
import subprocess
from pathlib import Path

import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine


def _wait_until_done(eng: GraphKnowledgeEngine, *, ns: str, job_id: str, timeout_s: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        rows = eng.meta_sqlite.list_index_jobs(namespace=ns, limit=2000)
        for r in rows:
            if getattr(r, "job_id", None) == job_id and getattr(r, "status", None) == "DONE":
                return
        time.sleep(0.05)
    raise AssertionError(f"job did not complete within {timeout_s}s: {job_id}")


@pytest.mark.integration
def test_phase5_supervisor_runs_worker_processes_job_then_graceful_shutdown(tmp_path):
    ns = f"phase5_sup_{uuid.uuid4().hex}"
    persist_dir = tmp_path / "persist"
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Start supervisor (spawns worker)
    pidfile = tmp_path / "worker.pid"
    sup_py = Path(__file__).resolve().parent.parent.parent / "graph_knowledge_engine" / "workers" / "worker_supervisor.py"
    # In case repo path differs, allow module execution too:
    cmd = [sys.executable, str(sup_py), "--persist-directory", str(persist_dir), "--namespace", ns, "--pidfile", str(pidfile), "--phase1-enable-index-jobs"]
    proc = subprocess.Popen(cmd)

    try:
        eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
        eng._phase1_enable_index_jobs = True
        eng.namespace = ns  # type: ignore[attr-defined]

        job_id = f"job_{uuid.uuid4().hex}"
        # enqueue a NOOP job: engine.indexing.apply_index_job will just return; worker still marks DONE
        eng.meta_sqlite.enqueue_index_job(
            namespace=ns,
            job_id=job_id,
            entity_kind="node",
            entity_id="n1",
            index_kind="noop",
            op="upsert",
            max_retries=3,
        )

        _wait_until_done(eng, ns=ns, job_id=job_id, timeout_s=10.0)

    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.integration
def test_phase5_supervisor_restarts_worker_after_kill_and_processes_job(tmp_path):
    ns = f"phase5_sup_restart_{uuid.uuid4().hex}"
    persist_dir = tmp_path / "persist"
    persist_dir.mkdir(parents=True, exist_ok=True)

    pidfile = tmp_path / "worker.pid"
    sup_py = Path(__file__).resolve().parent.parent.parent / "graph_knowledge_engine" / "workers" / "worker_supervisor.py"
    cmd = [sys.executable, str(sup_py), "--persist-directory", str(persist_dir), "--namespace", ns, "--pidfile", str(pidfile), "--phase1-enable-index-jobs"]
    proc = subprocess.Popen(cmd)

    try:
        # Wait for pidfile
        start = time.time()
        while time.time() - start < 5:
            if pidfile.exists():
                break
            time.sleep(0.05)
        assert pidfile.exists(), "worker pidfile not created"

        worker_pid = int(pidfile.read_text().strip())
        # Kill the worker process; supervisor should restart it.
        if os.name == "nt":
            subprocess.run([sys.executable, "-c", f"import os; os.kill({worker_pid}, 9)"], check=False)
        else:
            os.kill(worker_pid, 9)

        # Enqueue a job and ensure it is completed after restart.
        eng = GraphKnowledgeEngine(persist_directory=str(persist_dir))
        eng._phase1_enable_index_jobs = True
        eng.namespace = ns  # type: ignore[attr-defined]

        job_id = f"job_{uuid.uuid4().hex}"
        eng.meta_sqlite.enqueue_index_job(
            namespace=ns,
            job_id=job_id,
            entity_kind="node",
            entity_id="n1",
            index_kind="noop",
            op="upsert",
            max_retries=3,
        )

        _wait_until_done(eng, ns=ns, job_id=job_id, timeout_s=10.0)

    finally:
        proc.terminate()
        proc.wait(timeout=5)
