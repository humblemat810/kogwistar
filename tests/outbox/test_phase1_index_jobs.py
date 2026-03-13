from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set


# Make local modules importable when tests are run from repo root.
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_knowledge_engine.engine_core.engine_sqlite import EngineSQLite


@dataclass
class FakeDerivedStore:
    """In-memory representation of a derived join index.

    This stands in for join tables/collections like node_docs.
    """

    node_docs: Dict[str, Set[str]]

    def __init__(self) -> None:
        self.node_docs = {}

    def node_docs_delete(self, node_id: str) -> None:
        self.node_docs.pop(node_id, None)

    def node_docs_add(self, node_id: str, doc_ids: List[str]) -> None:
        self.node_docs[node_id] = set(doc_ids)


class SimpleIndexJobDrainer:
    """A minimal drainer used only for Phase 1 tests.

    It exercises the outbox-style invariant:
    - jobs can be enqueued without any fast-path application
    - drainer can later converge derived state

    NOTE: This deliberately avoids monkeypatching; it uses a real EngineSQLite
    metastore and a real (in-memory) derived store.
    """

    def __init__(self, meta: EngineSQLite, derived: FakeDerivedStore):
        self.meta = meta
        self.derived = derived
        # failure injection: job_id set triggers exactly one injected failure
        self.fail_after_delete_once: Set[str] = set()

    def drain(self, *, limit: int = 50, lease_seconds: int = 1) -> int:
        jobs = self.meta.claim_index_jobs(limit=limit, lease_seconds=lease_seconds)
        for job in jobs:
            try:
                self._apply(job.index_kind, job.op, job.entity_id, job.payload_json, job.job_id)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                next_retry = int(job.retry_count or 0) + 1
                max_retries = int(job.max_retries or 10)
                if next_retry < max_retries:
                    self.meta.bump_retry_and_requeue(job.job_id, err, next_run_at_seconds=0)
                else:
                    self.meta.mark_index_job_failed(job.job_id, err)
            else:
                self.meta.mark_index_job_done(job.job_id)
        return len(jobs)

    def _apply(self, index_kind: str, op: str, entity_id: str, payload_json: str | None, job_id: str) -> None:
        if index_kind != "node_docs":
            raise NotImplementedError("Phase 1 tests focus on node_docs join index")

        if op == "DELETE":
            self.derived.node_docs_delete(entity_id)
            return

        if op != "UPSERT":
            raise ValueError(f"Unknown op: {op}")

        payload = json.loads(payload_json or "{}")
        doc_ids = payload.get("doc_ids")
        if not isinstance(doc_ids, list) or not all(isinstance(x, str) for x in doc_ids):
            raise ValueError("payload_json must contain {'doc_ids': [..]} for UPSERT")

        # delete -> add rebuild style (this is the dangerous window Phase 1 addresses)
        self.derived.node_docs_delete(entity_id)

        # Fault injection: simulate crash/freeze after delete.
        if job_id in self.fail_after_delete_once:
            self.fail_after_delete_once.remove(job_id)
            raise RuntimeError("Injected failure after delete")

        self.derived.node_docs_add(entity_id, doc_ids)


def _mk_sqlite(tmp_path: str) -> EngineSQLite:
    # EngineSQLite expects a directory + filename.
    meta = EngineSQLite(tmp_path, filename="meta.sqlite")
    meta.ensure_initialized()
    return meta


def _force_expire_job_lease(meta: EngineSQLite, job_id: str) -> None:
    # Intentionally bypass the metastore API to simulate a stalled worker lease.
    with meta.connect() as conn:
        conn.execute("UPDATE index_jobs SET lease_until = 0 WHERE job_id = ?", (job_id,))
        conn.commit()


def test_sqlite_index_jobs_lease_steal(tmp_path: str) -> None:
    """If a worker halts forever while holding a job, another worker can steal it."""
    meta = _mk_sqlite(tmp_path)
    job_id = "job-lease-steal"
    meta.enqueue_index_job(
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="UPSERT",
        payload_json=json.dumps({"doc_ids": ["d1"]}),
    )

    claimed1 = meta.claim_index_jobs(limit=10, lease_seconds=60)
    assert [j.job_id for j in claimed1] == [job_id]

    # Simulate halted worker by expiring lease.
    _force_expire_job_lease(meta, job_id)

    claimed2 = meta.claim_index_jobs(limit=10, lease_seconds=60)
    assert [j.job_id for j in claimed2] == [job_id]


def test_enqueue_only_then_drain_converges(tmp_path: str) -> None:
    """Outbox-style invariant: enqueue without fast path; drainer still converges."""
    meta = _mk_sqlite(tmp_path)
    derived = FakeDerivedStore()
    drainer = SimpleIndexJobDrainer(meta, derived)

    meta.enqueue_index_job(
        job_id="job-enqueue-only",
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="UPSERT",
        payload_json=json.dumps({"doc_ids": ["d1", "d2"]}),
    )

    assert derived.node_docs == {}
    drainer.drain(limit=10)
    assert derived.node_docs["n1"] == {"d1", "d2"}
    jobs = meta.list_index_jobs(status="DONE")
    assert [j.job_id for j in jobs] == ["job-enqueue-only"]


def test_crash_after_delete_then_retry_recovers(tmp_path: str) -> None:
    """Simulate delete->crash before add; ensure later drain repairs."""
    meta = _mk_sqlite(tmp_path)
    derived = FakeDerivedStore()
    drainer = SimpleIndexJobDrainer(meta, derived)

    # Existing index rows (pre-state)
    derived.node_docs_add("n1", ["d_old"])

    job_id = "job-crash-after-delete"
    meta.enqueue_index_job(
        job_id=job_id,
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="UPSERT",
        payload_json=json.dumps({"doc_ids": ["d1", "d2"]}),
    )

    # Inject failure once right after delete
    drainer.fail_after_delete_once.add(job_id)

    drainer.drain(limit=10)

    # After failure: derived state should be missing (deleted), and job requeued.
    assert "n1" not in derived.node_docs
    pending = meta.list_index_jobs(status="PENDING")
    assert [j.job_id for j in pending] == [job_id]
    assert pending[0].retry_count == 1

    # Next drain should retry and complete
    drainer.drain(limit=10)
    assert derived.node_docs["n1"] == {"d1", "d2"}
    done = meta.list_index_jobs(status="DONE")
    assert [j.job_id for j in done] == [job_id]


def test_delete_job_removes_join_rows(tmp_path: str) -> None:
    """Tombstone-style cleanup: DELETE op removes join rows."""
    meta = _mk_sqlite(tmp_path)
    derived = FakeDerivedStore()
    drainer = SimpleIndexJobDrainer(meta, derived)

    derived.node_docs_add("n1", ["d1", "d2"])
    meta.enqueue_index_job(
        job_id="job-delete",
        entity_kind="node",
        entity_id="n1",
        index_kind="node_docs",
        op="DELETE",
        payload_json=None,
    )
    drainer.drain(limit=10)
    assert "n1" not in derived.node_docs
