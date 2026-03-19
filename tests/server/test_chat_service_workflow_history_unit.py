from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from graph_knowledge_engine.server.chat_service_workflow_history import (
    _WorkflowDesignHistoryMixin,
)


class MockOwner:
    def __init__(self):
        self._DESIGN_CONTROL_KIND = "design_control"
        self._CTRL_UNDO_APPLIED = "UNDO_APPLIED"
        self._CTRL_REDO_APPLIED = "REDO_APPLIED"
        self._CTRL_BRANCH_DROPPED = "BRANCH_DROPPED"
        self._CTRL_MUTATION_COMMITTED = "MUTATION_COMMITTED"
        self._PROJECTION_SCHEMA_VERSION = 1
        self._SNAPSHOT_SCHEMA_VERSION = 1
        self._DELTA_SCHEMA_VERSION = 1
        self._SNAPSHOT_INTERVAL = 50
        self._workflow_history_lock = MagicMock()

        self.mock_workflow_engine = MagicMock()
        self.mock_meta_store = MagicMock()
        self.mock_workflow_engine.meta_sqlite = self.mock_meta_store

    def _workflow_engine(self):
        return self.mock_workflow_engine

    def _workflow_namespace(self, workflow_id):
        return f"wf_design:{workflow_id}"

    def _now_ms(self):
        return 1000


class HistoryHelper(_WorkflowDesignHistoryMixin):
    def __init__(self, owner):
        super().__init__(owner)


@pytest.fixture
def history_context():
    owner = MockOwner()
    helper = HistoryHelper(owner)
    return owner, helper


def test_workflow_latest_seq_shortcut(history_context):
    owner, helper = history_context
    owner.mock_meta_store.get_latest_entity_event_seq.return_value = 42

    seq = helper._workflow_latest_seq(namespace="test_ns")

    assert seq == 42
    owner.mock_meta_store.get_latest_entity_event_seq.assert_called_once_with(
        namespace="test_ns"
    )


def test_workflow_latest_seq_fallback(history_context):
    owner, helper = history_context
    # Disable shortcut
    owner.mock_meta_store.get_latest_entity_event_seq = None

    # Mock iter_entity_events
    owner.mock_meta_store.iter_entity_events.return_value = [
        (10, "kind", "id", "op", "{}"),
        (20, "kind", "id", "op", "{}"),
        (30, "kind", "id", "op", "{}"),
    ]

    seq = helper._workflow_latest_seq(namespace="test_ns")

    assert seq == 30
    owner.mock_meta_store.iter_entity_events.assert_called_once_with(
        namespace="test_ns", from_seq=1
    )


def test_workflow_projection_stale_missing(history_context):
    _, helper = history_context
    state = {"latest_seq": 10}
    assert helper._workflow_projection_stale(state=state, projection=None) is True


def test_workflow_projection_stale_schema_mismatch(history_context):
    _, helper = history_context
    state = {"latest_seq": 10}

    # Projection schema mismatch
    proj = {
        "projection_schema_version": 0,  # Expected 1
        "snapshot_schema_version": 1,
        "last_authoritative_seq": 10,
        "materialization_status": "ready",
    }
    assert helper._workflow_projection_stale(state=state, projection=proj) is True

    # Snapshot schema mismatch
    proj = {
        "projection_schema_version": 1,
        "snapshot_schema_version": 0,  # Expected 1
        "last_authoritative_seq": 10,
        "materialization_status": "ready",
    }
    assert helper._workflow_projection_stale(state=state, projection=proj) is True


def test_workflow_projection_stale_sequence_lag(history_context):
    _, helper = history_context
    state = {"latest_seq": 15}
    proj = {
        "projection_schema_version": 1,
        "snapshot_schema_version": 1,
        "last_authoritative_seq": 10,  # Lags behind 15
        "materialization_status": "ready",
    }
    assert helper._workflow_projection_stale(state=state, projection=proj) is True


def test_workflow_projection_stale_not_ready(history_context):
    _, helper = history_context
    state = {"latest_seq": 10}
    proj = {
        "projection_schema_version": 1,
        "snapshot_schema_version": 1,
        "last_authoritative_seq": 10,
        "materialization_status": "rebuilding",  # Not ready
    }
    assert helper._workflow_projection_stale(state=state, projection=proj) is True


def test_workflow_projection_not_stale(history_context):
    _, helper = history_context
    state = {"latest_seq": 10}
    proj = {
        "projection_schema_version": 1,
        "snapshot_schema_version": 1,
        "last_authoritative_seq": 10,
        "materialization_status": "ready",
    }
    assert helper._workflow_projection_stale(state=state, projection=proj) is False


def test_workflow_sync_projection_locked_rebuild_needed(history_context):
    owner, helper = history_context
    workflow_id = "wf1"

    # Mock staleness
    helper._workflow_fold_history = MagicMock(return_value={"latest_seq": 20})
    helper._workflow_projection = MagicMock(
        return_value={"last_authoritative_seq": 10, "projection_schema_version": 1}
    )

    # Mock rebuild methods
    helper._workflow_rebuild_namespace_for_state = MagicMock()
    helper._workflow_store_snapshot_if_needed = MagicMock()
    helper._store_workflow_projection = MagicMock()

    helper._workflow_sync_projection_locked(workflow_id=workflow_id)

    # Rebuild should be called
    helper._workflow_rebuild_namespace_for_state.assert_called_once()
    helper._workflow_store_snapshot_if_needed.assert_called_once()
    helper._store_workflow_projection.assert_called_once()


def test_workflow_sync_projection_locked_no_rebuild_needed(history_context):
    owner, helper = history_context
    workflow_id = "wf1"

    # Mock up-to-date
    helper._workflow_fold_history = MagicMock(return_value={"latest_seq": 10})
    helper._workflow_projection = MagicMock(
        return_value={
            "last_authoritative_seq": 10,
            "projection_schema_version": 1,
            "snapshot_schema_version": 1,
            "materialization_status": "ready",
        }
    )

    # Mock rebuild methods
    helper._workflow_rebuild_namespace_for_state = MagicMock()
    helper._workflow_store_snapshot_if_needed = MagicMock()
    helper._store_workflow_projection = MagicMock()

    helper._workflow_sync_projection_locked(workflow_id=workflow_id)

    # Rebuild should NOT be called
    helper._workflow_rebuild_namespace_for_state.assert_not_called()
    helper._store_workflow_projection.assert_called_once()
