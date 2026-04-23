from __future__ import annotations

import json

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowCheckpointNode, WorkflowStepExecNode
from kogwistar.runtime.replay import _apply_state_update
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.runtime.replay import load_checkpoint, replay_to

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class _DummyRead:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self, *args, **kwargs):
        nodes = list(self._nodes)
        ids = kwargs.get("ids")
        if ids is not None:
            ids = set(ids)
            nodes = [n for n in nodes if getattr(n, "id", None) in ids]
        where = kwargs.get("where") or {}
        if "$and" in where:
            for clause in where["$and"]:
                if not isinstance(clause, dict):
                    continue
                for key, value in clause.items():
                    nodes = [
                        n
                        for n in nodes
                        if (getattr(n, "metadata", {}) or {}).get(key) == value
                    ]
        return nodes


class _DummyEngine:
    def __init__(self, nodes):
        self.read = _DummyRead(nodes)


def _ckpt(step_seq: int, *, schema_version: int | None = 1):
    md = {
        "entity_type": "workflow_checkpoint",
        "run_id": "run-1",
        "workflow_id": "wf-1",
        "step_seq": step_seq,
        "state_json": json.dumps({"answer": "ok"}),
    }
    if schema_version is not None:
        md["checkpoint_schema_version"] = schema_version
    return WorkflowCheckpointNode(
        id=f"wf_ckpt|run-1|{step_seq}",
        label="checkpoint",
        type="entity",
        doc_id=f"wf_ckpt|run-1|{step_seq}",
        summary="checkpoint",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata=md,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def test_load_checkpoint_accepts_current_schema():
    eng = _DummyEngine([_ckpt(0)])
    state = load_checkpoint(conversation_engine=eng, run_id="run-1", step_seq=0)
    assert state["answer"] == "ok"


def test_load_checkpoint_rejects_future_schema():
    eng = _DummyEngine([_ckpt(0, schema_version=99)])
    with pytest.raises(ValueError, match="incompatible checkpoint_schema_version"):
        load_checkpoint(conversation_engine=eng, run_id="run-1", step_seq=0)


def test_replay_rejects_future_schema():
    eng = _DummyEngine([_ckpt(0, schema_version=99)])
    with pytest.raises(ValueError, match="incompatible checkpoint_schema_version"):
        replay_to(conversation_engine=eng, run_id="run-1", target_step_seq=0)


def test_replay_state_reducer_matches_sync_runtime_merge_semantics():
    """Async mirror: `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_uses_shared_state_merge_semantics`."""
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = type("_Resolver", (), {"_state_schema": {}})()

    sync_state = {}
    runtime.apply_state_update(
        sync_state,
        state_update=[
            ("u", {"answer": "ok"}),
            ("a", {"ops": "first"}),
            ("e", {"nums": [1, 2]}),
        ],
    )

    replay_state = {}
    _apply_state_update(
        replay_state,
        [
            ("u", {"answer": "ok"}),
            ("a", {"ops": "first"}),
            ("e", {"nums": [1, 2]}),
        ],
    )

    assert replay_state == sync_state


def test_replay_ignores_created_at_ms_and_orders_by_step_seq():
    """Async mirror: `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_persists_rt_join_frontier_shape`."""
    checkpoint = WorkflowCheckpointNode(
        id="wf_ckpt|run-1|0",
        label="checkpoint",
        type="entity",
        doc_id="wf_ckpt|run-1|0",
        summary="checkpoint",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_checkpoint",
            "run_id": "run-1",
            "workflow_id": "wf-1",
            "step_seq": 0,
            "state_json": json.dumps({"answer": "ok"}),
            "checkpoint_schema_version": 1,
            "created_at_ms": 999999,
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    step_late = WorkflowStepExecNode(
        id="wf_step|run-1|2",
        label="step",
        type="entity",
        doc_id="wf_step|run-1|2",
        summary="step",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_step_exec",
            "run_id": "run-1",
            "workflow_id": "wf-1",
            "workflow_node_id": "node-late",
            "step_seq": 2,
            "op": "late",
            "status": "ok",
            "duration_ms": 1,
            "result_json": json.dumps(
                {"state_update": [("a", {"ops": "late"})], "conversation_node_id": None}
            ),
            "created_at_ms": 1,
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    step_early = WorkflowStepExecNode(
        id="wf_step|run-1|1",
        label="step",
        type="entity",
        doc_id="wf_step|run-1|1",
        summary="step",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_step_exec",
            "run_id": "run-1",
            "workflow_id": "wf-1",
            "workflow_node_id": "node-early",
            "step_seq": 1,
            "op": "early",
            "status": "ok",
            "duration_ms": 1,
            "result_json": json.dumps(
                {"state_update": [("a", {"ops": "early"})], "conversation_node_id": None}
            ),
            "created_at_ms": 999999,
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )

    eng = _DummyEngine([checkpoint, step_late, step_early])
    state = replay_to(conversation_engine=eng, run_id="run-1", target_step_seq=2)
    assert state["ops"] == ["early", "late"]


def test_replay_to_is_read_only_and_does_not_append_new_history():
    """Async mirror: `tests/runtime/test_async_runtime_contract.py::test_async_runtime_resume_run_delegates_to_sync_resume`."""
    checkpoint = _ckpt(0)
    step = WorkflowStepExecNode(
        id="wf_step|run-1|1",
        label="step",
        type="entity",
        doc_id="wf_step|run-1|1",
        summary="step",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_step_exec",
            "run_id": "run-1",
            "workflow_id": "wf-1",
            "workflow_node_id": "node-1",
            "step_seq": 1,
            "op": "append",
            "status": "ok",
            "duration_ms": 1,
            "result_json": json.dumps(
                {"state_update": [("a", {"ops": "append"})], "conversation_node_id": None}
            ),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )

    class _Write:
        def __getattr__(self, name):
            raise AssertionError(f"replay_to must not call write.{name}")

    eng = type(
        "_Eng",
        (),
        {
            "read": _DummyEngine([checkpoint, step]).read,
            "write": _Write(),
        },
    )()
    state = replay_to(conversation_engine=eng, run_id="run-1", target_step_seq=1)
    assert state["ops"] == ["append"]
