from __future__ import annotations

import contextlib
import threading

from graph_knowledge_engine.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
    WorkflowProjectionRebuildingError,
)
from graph_knowledge_engine.server.chat_service_workflow_design import _WorkflowDesignService
from graph_knowledge_engine.server.resources import _LazyResource


class _Owner:
    _DESIGN_CONTROL_KIND = "design_control"
    _CTRL_UNDO_APPLIED = "UNDO_APPLIED"
    _CTRL_REDO_APPLIED = "REDO_APPLIED"
    _CTRL_BRANCH_DROPPED = "BRANCH_DROPPED"
    _CTRL_MUTATION_COMMITTED = "MUTATION_COMMITTED"
    _PROJECTION_SCHEMA_VERSION = 1
    _SNAPSHOT_SCHEMA_VERSION = 1
    _DELTA_SCHEMA_VERSION = 1
    _SNAPSHOT_INTERVAL = 50

    def __init__(self) -> None:
        self.run_registry = object()
        self.answer_runner = lambda req: {}
        self.runtime_runner = lambda req: {}
        self._workflow_history_lock = threading.Lock()

    def _knowledge_engine(self):
        raise AssertionError("not used in smoke test")

    def _conversation_engine(self):
        raise AssertionError("not used in smoke test")

    def _workflow_engine(self):
        raise AssertionError("not used in smoke test")

    def _conversation_service(self):
        raise AssertionError("not used in smoke test")

    def _publish(self, run_id: str, event_type: str, payload=None):
        return {"run_id": run_id, "event_type": event_type, "payload": payload}

    @contextlib.contextmanager
    def _workflow_namespace_scope(self, workflow_id: str):
        yield None


def test_chat_service_reexports_public_symbols() -> None:
    assert ChatRunService is not None
    assert AnswerRunRequest is not None
    assert RuntimeRunRequest is not None
    assert RunCancelledError is not None
    assert WorkflowProjectionRebuildingError is not None


def test_workflow_design_visible_delta_round_trip() -> None:
    service = _WorkflowDesignService(_Owner())
    before = {
        "nodes": [{"id": "n1", "label": "alpha"}],
        "edges": [{"id": "e1", "label": "old"}],
    }
    after = {
        "nodes": [{"id": "n1", "label": "beta"}, {"id": "n2", "label": "gamma"}],
        "edges": [{"id": "e2", "label": "new"}],
    }

    delta = service._workflow_compute_visible_delta(before=before, after=after)

    assert delta["upsert_nodes"] == [{"id": "n1", "label": "beta"}, {"id": "n2", "label": "gamma"}]
    assert delta["delete_node_ids"] == []
    assert delta["upsert_edges"] == [{"id": "e2", "label": "new"}]
    assert delta["delete_edge_ids"] == ["e1"]


def test_workflow_design_projection_head_and_stale_detection() -> None:
    service = _WorkflowDesignService(_Owner())
    state = {
        "current_version": 2,
        "active_tip_version": 3,
        "latest_seq": 40,
        "current_seq": 35,
    }

    head = service._workflow_projection_head(state=state, materialization_status="ready")

    assert head["current_version"] == 2
    assert head["active_tip_version"] == 3
    assert head["last_authoritative_seq"] == 40
    assert head["last_materialized_seq"] == 35
    assert service._workflow_projection_stale(state=state, projection=head) is False

    stale_head = dict(head)
    stale_head["last_authoritative_seq"] = 39
    assert service._workflow_projection_stale(state=state, projection=stale_head) is True


def test_workflow_design_branch_drop_emits_only_when_redo_branch_exists(monkeypatch) -> None:
    service = _WorkflowDesignService(_Owner())
    recorded: dict[str, object] = {}

    def _record(**kwargs):
        recorded.update(kwargs)
        return 1

    monkeypatch.setattr(service, "_append_design_control_event", _record)

    dropped = service._workflow_append_branch_drop_if_needed_locked(
        workflow_id="wf.demo",
        state={
            "current_version": 1,
            "versions": [
                {"version": 1, "seq": 10},
                {"version": 2, "seq": 20},
                {"version": 3, "seq": 30},
            ],
        },
        designer_id="designer-1",
        source="test",
    )

    assert dropped is True
    assert recorded["op"] == "BRANCH_DROPPED"
    assert recorded["payload"] == {
        "drop_from_version": 2,
        "drop_to_version": 3,
        "drop_from_seq": 20,
        "drop_to_seq": 30,
        "reason": "new_edit_after_undo",
    }


def test_lazy_resource_get_initializes_once_and_returns_cached_instance() -> None:
    calls = 0

    class _Thing:
        value = 7

    def _factory() -> _Thing:
        nonlocal calls
        calls += 1
        return _Thing()

    resource = _LazyResource(_factory, "thing")

    first = resource.get()
    second = resource.get()

    assert first is second
    assert first.value == 7
    assert calls == 1


def test_lazy_resource_keeps_proxy_access_for_compatibility() -> None:
    class _Thing:
        def __init__(self) -> None:
            self.value = 1

    resource = _LazyResource(_Thing, "thing")

    assert resource.value == 1
    resource.value = 3
    assert resource.get().value == 3
