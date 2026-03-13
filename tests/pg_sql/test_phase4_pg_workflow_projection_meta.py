from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.engine_postgres_meta import (
    EnginePostgresMetaStore,
)
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend
from graph_knowledge_engine.server.run_registry import RunRegistry


class FakeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 3):
        self._dim = dim
        self.is_legacy = False

    def __call__(self, inputs):
        return [[0.01] * self._dim for _ in inputs]


def test_pg_meta_store_workflow_projection_and_server_run_tables(sa_engine, pg_schema):
    if sa_engine is None or pg_schema is None:
        pytest.skip(
            "pg backend requested but sa_engine/pg_schema fixtures not available"
        )

    root = Path(".tmp_pg_workflow_projection_meta") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=3, schema=f"{pg_schema}_wf"
        )
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(root / "pg_workflow_meta"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=3),
            backend=backend,
        )
        meta = workflow_engine.meta_sqlite
        assert isinstance(meta, EnginePostgresMetaStore)

        workflow_id = "wf.pg.meta"
        meta.replace_workflow_design_projection(
            workflow_id=workflow_id,
            head={
                "current_version": 4,
                "active_tip_version": 4,
                "last_authoritative_seq": 12,
                "last_materialized_seq": 12,
                "projection_schema_version": 1,
                "snapshot_schema_version": 1,
                "materialization_status": "ready",
                "updated_at_ms": 1234,
            },
            versions=[
                {"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0},
                {"version": 1, "prev_version": 0, "target_seq": 3, "created_at_ms": 10},
                {
                    "version": 4,
                    "prev_version": 1,
                    "target_seq": 12,
                    "created_at_ms": 40,
                },
            ],
            dropped_ranges=[
                {"start_seq": 4, "end_seq": 9, "start_version": 2, "end_version": 3}
            ],
        )
        projection = meta.get_workflow_design_projection(workflow_id=workflow_id)
        assert projection is not None
        assert projection["current_version"] == 4
        assert projection["active_tip_version"] == 4
        assert [int(item["version"]) for item in projection["versions"]] == [0, 1, 4]
        assert projection["dropped_ranges"] == [
            {"start_seq": 4, "end_seq": 9, "start_version": 2, "end_version": 3}
        ]

        meta.put_workflow_design_snapshot(
            workflow_id=workflow_id,
            version=4,
            seq=12,
            payload_json=json.dumps(
                {"nodes": [{"id": "n1"}], "edges": []},
                sort_keys=True,
                separators=(",", ":"),
            ),
            schema_version=1,
        )
        snapshot = meta.get_workflow_design_snapshot(
            workflow_id=workflow_id, max_version=10, schema_version=1
        )
        assert snapshot is not None
        assert snapshot["version"] == 4
        assert json.loads(str(snapshot["payload_json"])) == {
            "nodes": [{"id": "n1"}],
            "edges": [],
        }

        meta.put_workflow_design_delta(
            workflow_id=workflow_id,
            version=4,
            prev_version=1,
            target_seq=12,
            forward_json=json.dumps(
                {
                    "upsert_nodes": [
                        {
                            "id": "n1",
                            "metadata": {
                                "entity_type": "workflow_node",
                                "workflow_id": workflow_id,
                                "wf_op": "start",
                                "wf_start": True,
                            },
                        }
                    ],
                    "delete_node_ids": [],
                    "upsert_edges": [],
                    "delete_edge_ids": [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            inverse_json=json.dumps(
                {
                    "upsert_nodes": [],
                    "delete_node_ids": ["n1"],
                    "upsert_edges": [],
                    "delete_edge_ids": [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            schema_version=1,
        )
        delta = meta.get_workflow_design_delta(
            workflow_id=workflow_id, version=4, schema_version=1
        )
        assert delta is not None
        assert delta["version"] == 4
        assert delta["prev_version"] == 1
        assert delta["target_seq"] == 12
        assert json.loads(str(delta["forward_json"]))["upsert_nodes"][0]["id"] == "n1"
        assert json.loads(str(delta["inverse_json"]))["delete_node_ids"] == ["n1"]

        registry = RunRegistry(meta)
        run_id = "run-pg-meta"
        registry.create_run(
            run_id=run_id,
            conversation_id="conv-pg-meta",
            workflow_id=workflow_id,
            user_id="alice",
            user_turn_node_id="turn-1",
            status="queued",
        )
        created = registry.get_run(run_id)
        assert created is not None
        assert created["status"] == "queued"
        assert created["cancel_requested"] is False

        evt = registry.append_event(run_id, "run.created", {"run_id": run_id})
        assert evt["event_type"] == "run.created"
        assert evt["payload"] == {"run_id": run_id}

        updated = registry.update_status(
            run_id,
            status="running",
            assistant_turn_node_id=None,
            result=None,
            error=None,
            started=True,
        )
        assert updated["status"] == "running"
        assert updated["started_at_ms"] is not None

        cancelled = registry.request_cancel(run_id)
        assert cancelled["cancel_requested"] is True

        events = registry.list_events(run_id)
        assert [str(item["event_type"]) for item in events] == ["run.created"]

        meta.clear_workflow_design_snapshots(workflow_id=workflow_id)
        meta.clear_workflow_design_deltas(workflow_id=workflow_id)
        meta.clear_workflow_design_projection(workflow_id=workflow_id)
        assert (
            meta.get_workflow_design_snapshot(
                workflow_id=workflow_id, max_version=10, schema_version=1
            )
            is None
        )
        assert (
            meta.get_workflow_design_delta(
                workflow_id=workflow_id, version=4, schema_version=1
            )
            is None
        )
        assert meta.get_workflow_design_projection(workflow_id=workflow_id) is None
    finally:
        shutil.rmtree(root, ignore_errors=True)
