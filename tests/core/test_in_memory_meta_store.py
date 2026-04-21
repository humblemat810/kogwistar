from __future__ import annotations

import shutil
from pathlib import Path
import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore

from tests.conftest import FakeEmbeddingFunction


pytestmark = pytest.mark.core


def test_in_memory_meta_transaction_joins_and_rolls_back() -> None:
    meta = InMemoryMetaStore()

    with pytest.raises(RuntimeError):
        with meta.transaction() as outer:
            meta.next_global_seq()
            with meta.transaction() as inner:
                assert inner is outer
                meta.next_user_seq("u1")
            raise RuntimeError("boom")

    assert meta.current_global_seq() == 0
    assert meta.current_user_seq("u1") == 0


def test_in_memory_meta_index_jobs_and_event_log_contract() -> None:
    meta = InMemoryMetaStore()

    assert meta.next_user_seq("user-a") == 1
    assert meta.next_scoped_seq("scope-a") == 1
    assert meta.next_global_seq() == 1

    jid = meta.enqueue_index_job(
        job_id="job-1",
        namespace="ns-a",
        entity_kind="node",
        entity_id="n-1",
        index_kind="node_docs",
        op="UPSERT",
    )
    assert jid == "job-1"

    claimed = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="ns-a")
    assert claimed and claimed[0].job_id == "job-1"
    meta.mark_index_job_done("job-1")
    assert meta.list_index_jobs(namespace="ns-a", status="DONE", limit=10)[0].job_id == "job-1"

    jid2 = meta.enqueue_index_job(
        job_id="job-2",
        namespace="ns-a",
        entity_kind="edge",
        entity_id="e-1",
        index_kind="edge_endpoints",
        op="UPSERT",
    )
    with meta.connect() as conn:
        conn.execute("UPDATE index_jobs SET lease_until = 0 WHERE job_id = ?", (jid2,))
        conn.commit()
    reclaimed = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="ns-a")
    assert reclaimed and reclaimed[0].job_id == "job-2"

    seq = meta.append_entity_event(
        namespace="ns-a",
        event_id="evt-1",
        entity_kind="node",
        entity_id="n-1",
        op="ADD",
        payload_json='{"hello":"world"}',
    )
    assert seq == 1
    assert list(meta.iter_entity_events(namespace="ns-a", from_seq=1)) == [
        (1, "node", "n-1", "ADD", '{"hello":"world"}')
    ]
    meta.cursor_set(namespace="ns-a", consumer="replay", last_seq=seq)
    assert meta.cursor_get(namespace="ns-a", consumer="replay") == 1

    meta.set_index_applied_fingerprint(
        namespace="ns-a",
        coalesce_key="node:n-1:node_docs",
        applied_fingerprint="fp-1",
        last_job_id="job-1",
    )
    assert (
        meta.get_index_applied_fingerprint(
            namespace="ns-a", coalesce_key="node:n-1:node_docs"
        )
        == "fp-1"
    )


def test_in_memory_meta_named_projection_and_workflow_design_helpers() -> None:
    meta = InMemoryMetaStore()

    meta.replace_named_projection(
        "bridge_governance",
        "interaction-1",
        {"active_agents": ["agent-a"]},
        last_authoritative_seq=7,
        last_materialized_seq=6,
        projection_schema_version=2,
        materialization_status="rebuilding",
    )
    assert meta.get_named_projection("bridge_governance", "interaction-1") is not None
    assert [item["key"] for item in meta.list_named_projections("bridge_governance")] == [
        "interaction-1"
    ]

    meta.replace_workflow_design_projection(
        workflow_id="wf-1",
        head={
            "current_version": 2,
            "active_tip_version": 3,
            "last_authoritative_seq": 11,
            "last_materialized_seq": 10,
            "projection_schema_version": 1,
            "snapshot_schema_version": 4,
            "materialization_status": "ready",
        },
        versions=[
            {"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0},
            {"version": 2, "prev_version": 1, "target_seq": 8, "created_at_ms": 20},
        ],
        dropped_ranges=[
            {"start_seq": 9, "end_seq": 10, "start_version": 2, "end_version": 3}
        ],
    )
    projection = meta.get_workflow_design_projection(workflow_id="wf-1")
    assert projection is not None
    assert projection["current_version"] == 2

    meta.put_workflow_design_snapshot(
        workflow_id="wf-1",
        version=2,
        seq=8,
        payload_json='{"nodes":[]}',
        schema_version=4,
    )
    snapshot = meta.get_workflow_design_snapshot(
        workflow_id="wf-1", max_version=2, schema_version=4
    )
    assert snapshot is not None
    assert snapshot["version"] == 2

    meta.put_workflow_design_delta(
        workflow_id="wf-1",
        version=3,
        prev_version=2,
        target_seq=10,
        forward_json='{"upsert_nodes":[]}',
        inverse_json='{"delete_node_ids":[]}',
        schema_version=1,
    )
    delta = meta.get_workflow_design_delta(
        workflow_id="wf-1", version=3, schema_version=1
    )
    assert delta is not None
    assert delta["prev_version"] == 2

    meta.clear_workflow_design_snapshots(workflow_id="wf-1")
    meta.clear_workflow_design_deltas(workflow_id="wf-1")
    meta.clear_workflow_design_projection(workflow_id="wf-1")
    assert (
        meta.get_workflow_design_snapshot(
            workflow_id="wf-1", max_version=2, schema_version=4
        )
        is None
    )
    assert meta.get_workflow_design_delta(
        workflow_id="wf-1", version=3, schema_version=1
    ) is None
    assert meta.get_workflow_design_projection(workflow_id="wf-1") is None


def test_in_memory_meta_workflow_snapshot_selection_prefers_version_over_timestamp(
    monkeypatch,
) -> None:
    meta = InMemoryMetaStore()
    timestamps = iter([9_999, 1])
    monkeypatch.setattr("kogwistar.engine_core.in_memory_meta._now_ms", lambda: next(timestamps))

    meta.put_workflow_design_snapshot(
        workflow_id="wf-1",
        version=1,
        seq=10,
        payload_json='{"nodes":[{"id":"v1"}]}',
        schema_version=4,
    )
    meta.put_workflow_design_snapshot(
        workflow_id="wf-1",
        version=2,
        seq=20,
        payload_json='{"nodes":[{"id":"v2"}]}',
        schema_version=4,
    )

    snapshot = meta.get_workflow_design_snapshot(
        workflow_id="wf-1",
        max_version=2,
        schema_version=4,
    )
    assert snapshot is not None
    assert snapshot["version"] == 2
    assert snapshot["seq"] == 20
    assert snapshot["created_at_ms"] == 1


def test_in_memory_meta_retry_and_run_cancel_contract() -> None:
    meta = InMemoryMetaStore()

    meta.enqueue_index_job(
        job_id="job-retry",
        namespace="ns-a",
        entity_kind="node",
        entity_id="n-1",
        index_kind="node_docs",
        op="UPSERT",
        max_retries=2,
    )
    claimed = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="ns-a")
    assert claimed and claimed[0].job_id == "job-retry"

    meta.bump_retry_and_requeue(
        "job-retry", "transient failure", next_run_at_seconds=0
    )
    pending = meta.list_index_jobs(namespace="ns-a", status="PENDING", limit=10)
    assert pending and pending[0].retry_count == 1

    claimed_again = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="ns-a")
    assert claimed_again and claimed_again[0].job_id == "job-retry"
    meta.mark_index_job_failed("job-retry", "final failure", final=True)
    failed = meta.list_index_jobs(namespace="ns-a", status="FAILED", limit=10)
    assert failed and failed[0].last_error == "final failure"

    meta.create_server_run(
        run_id="run-1",
        conversation_id="conv-1",
        workflow_id="wf-1",
        user_id="user-a",
        user_turn_node_id="turn-1",
    )
    meta.request_server_run_cancel(run_id="run-1")
    run = meta.get_server_run("run-1")
    assert run is not None
    assert run["cancel_requested"] is True
    assert run["status"] == "cancelling"


def test_fake_backend_uses_diskless_in_memory_meta_store() -> None:
    root = Path(".tmp_in_memory_meta_store_test")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(root / "conv"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=build_in_memory_backend,
    )

    assert isinstance(engine.meta_sqlite, InMemoryMetaStore)
    assert not (root / "conv" / "meta.sqlite").exists()
    assert not (root / "conv" / "_memory_meta").exists()
