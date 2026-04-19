from __future__ import annotations

import json

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowCheckpointNode
from kogwistar.runtime.replay import load_checkpoint, replay_to


class _DummyRead:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self, *args, **kwargs):
        return list(self._nodes)


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
