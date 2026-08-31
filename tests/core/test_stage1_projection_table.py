from __future__ import annotations

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite


def _replace_stage1(store: EngineSQLite, node_id: str, payload: dict[str, object]) -> None:
    store.replace_stage1_node_projection(
        "tenant-a",
        node_id,
        payload,
        last_authoritative_seq=7,
        last_materialized_seq=7,
        projection_schema_version=1,
        materialization_status="pending",
    )


def test_stage1_projection_is_physically_isolated_from_named_projections(tmp_path) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()

    _replace_stage1(store, "node-1", {"source": "stage1"})
    store.replace_named_projection(
        "tenant-a",
        "node-1",
        {"source": "named"},
        last_authoritative_seq=7,
        last_materialized_seq=7,
        projection_schema_version=1,
        materialization_status="ready",
    )

    assert store.get_stage1_node_projection("tenant-a", "node-1")["payload"] == {
        "source": "stage1"
    }
    assert store.get_named_projection("tenant-a", "node-1")["payload"] == {
        "source": "named"
    }


def test_stage1_clear_does_not_delete_named_projection_history(tmp_path) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()
    _replace_stage1(store, "node-1", {"state": "pending"})
    _replace_stage1(store, "node-2", {"state": "pending"})
    store.replace_named_projection(
        "tenant-a",
        "workflow-1",
        {"state": "durable"},
        last_authoritative_seq=9,
        last_materialized_seq=9,
        projection_schema_version=1,
        materialization_status="ready",
    )

    store.clear_stage1_node_projection("tenant-a", "node-1")
    assert store.get_stage1_node_projection("tenant-a", "node-1") is None
    assert store.get_stage1_node_projection("tenant-a", "node-2") is not None

    store.clear_stage1_node_namespace("tenant-a")
    assert store.list_stage1_node_projections("tenant-a") == []
    assert store.get_named_projection("tenant-a", "workflow-1")["payload"] == {
        "state": "durable"
    }


def test_stage1_projection_can_use_a_dedicated_sqlite_file(tmp_path) -> None:
    primary = EngineSQLite(tmp_path, filename="engine.db")
    staging = EngineSQLite(tmp_path, filename="stage1_projection.sqlite")
    primary.ensure_initialized()
    staging.ensure_initialized()

    _replace_stage1(staging, "node-1", {"state": "pending"})

    assert staging.get_stage1_node_projection("tenant-a", "node-1") is not None
    assert primary.get_stage1_node_projection("tenant-a", "node-1") is None


def test_stage1_projection_stats_support_explicit_high_churn_cleanup(tmp_path) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()
    _replace_stage1(store, "node-1", {"state": "pending"})
    _replace_stage1(store, "node-2", {"state": "pending"})
    store.replace_named_projection(
        "tenant-a",
        "workflow-1",
        {"state": "durable"},
        last_authoritative_seq=9,
        last_materialized_seq=9,
        projection_schema_version=1,
        materialization_status="ready",
    )

    stats = store.stage1_node_projection_stats("tenant-a")
    assert stats["row_count"] == 2
    assert stats["oldest_updated_at_ms"] > 0
    assert stats["newest_updated_at_ms"] >= stats["oldest_updated_at_ms"]

    store.clear_stage1_node_namespace("tenant-a")
    assert store.stage1_node_projection_stats("tenant-a")["row_count"] == 0
    assert store.get_named_projection("tenant-a", "workflow-1") is not None


def test_stage1_query_has_portable_metadata_and_id_contract(tmp_path) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()
    _replace_stage1(
        store,
        "node-1",
        {"metadata": {"kind": "person", "active": True}, "value": 1},
    )
    _replace_stage1(
        store,
        "node-2",
        {"metadata": {"kind": "place", "active": True}, "value": 2},
    )

    rows = store.query_stage1_node_projections(
        "tenant-a", metadata={"kind": "person"}, limit=1
    )
    assert [row["key"] for row in rows] == ["node-1"]
    assert store.query_stage1_node_projections("tenant-a", ids=["node-2"])[0]["key"] == "node-2"
    assert store.query_stage1_node_projections("tenant-a", metadata={"kind": "missing"}) == []

    with pytest.raises(ValueError, match="flat equality"):
        store.query_stage1_node_projections("tenant-a", metadata={"kind": {"$eq": "person"}})
    assert store.query_stage1_node_projections("tenant-a", limit=0) == []
