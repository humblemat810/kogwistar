from __future__ import annotations

from dataclasses import dataclass

import pytest

from kogwistar.runtime import AsyncWorkflowRuntime, WorkflowRuntime
from kogwistar.runtime.models import RunSuccess

pytestmark = [pytest.mark.ci]


@dataclass
class _Node:
    id: str
    label: str
    op: str
    metadata: dict


@dataclass
class _Edge:
    id: str
    label: str
    source_ids: list[str]
    target_ids: list[str]
    metadata: dict

    def safe_get_id(self):
        return self.id

    @property
    def priority(self):
        return int(self.metadata.get("wf_priority", 100))

    @property
    def multiplicity(self):
        return str(self.metadata.get("wf_multiplicity", "one"))

    @property
    def is_default(self):
        return bool(self.metadata.get("wf_is_default", False))

    @property
    def predicate(self):
        return self.metadata.get("wf_predicate")


def _mk_nodes():
    return {
        "n|left": _Node(
            id="n|left",
            label="left_label",
            op="left_op",
            metadata={"wf_terminal": False, "wf_fanout": False},
        ),
        "n|right": _Node(
            id="n|right",
            label="right_label",
            op="right_op",
            metadata={"wf_terminal": False, "wf_fanout": False},
        ),
        "n|fallback": _Node(
            id="n|fallback",
            label="fallback_label",
            op="fallback_op",
            metadata={"wf_terminal": True, "wf_fanout": False},
        ),
    }


def _mk_fanout_edges():
    return [
        _Edge(
            id="e-left",
            label="go_left",
            source_ids=["start"],
            target_ids=["n|left"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "many",
                "wf_is_default": False,
                "wf_predicate": None,
            },
        ),
        _Edge(
            id="e-right",
            label="go_right",
            source_ids=["start"],
            target_ids=["n|right"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "many",
                "wf_is_default": False,
                "wf_predicate": None,
            },
        ),
    ]


def _mk_default_edges():
    return [
        _Edge(
            id="e-pred",
            label="pred_path",
            source_ids=["start"],
            target_ids=["n|left"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "one",
                "wf_is_default": False,
                "wf_predicate": "if_true",
            },
        ),
        _Edge(
            id="e-default",
            label="default_path",
            source_ids=["start"],
            target_ids=["n|fallback"],
            metadata={
                "wf_priority": 100,
                "wf_multiplicity": "one",
                "wf_is_default": True,
                "wf_predicate": "if_false",
            },
        ),
    ]


def test_runtime_parity_bridge_route_next_shared_semantics():
    """Bridge parity: sync `_route_next` and async native edge selection share explicit alias, fanout, and default fallback semantics."""
    nodes = _mk_nodes()

    sync_rt = WorkflowRuntime.__new__(WorkflowRuntime)
    sync_rt.predicate_registry = {}

    fanout_result = RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=["go_left", "right_op"],
    )
    sync_next, sync_decision = sync_rt._route_next(
        edges=_mk_fanout_edges(),
        state={},
        last_result=fanout_result,
        fanout=True,
        nodes=nodes,
    )

    async_node = _Node(
        id="start",
        label="start",
        op="start",
        metadata={"wf_fanout": True},
    )
    async_edges = AsyncWorkflowRuntime._select_next_edges(
        async_node,
        _mk_fanout_edges(),
        {},
        fanout_result,
        {},
        nodes=nodes,
    )
    async_next = [str(edge.target_ids[0]) for edge in async_edges]

    assert sync_next == ["n|left", "n|right"]
    assert async_next == sync_next
    assert sync_decision.selected == [
        ("e-left", "n|left", "explicit"),
        ("e-right", "n|right", "explicit"),
    ]

    default_result = RunSuccess(conversation_node_id=None, state_update=[])
    sync_default_next, sync_default_decision = sync_rt._route_next(
        edges=_mk_default_edges(),
        state={},
        last_result=default_result,
        fanout=False,
        nodes=nodes,
    )
    async_default_node = _Node(
        id="start",
        label="start",
        op="start",
        metadata={"wf_fanout": False},
    )
    async_default_edges = AsyncWorkflowRuntime._select_next_edges(
        async_default_node,
        _mk_default_edges(),
        {},
        default_result,
        {},
        nodes=nodes,
    )
    async_default_next = [str(edge.target_ids[0]) for edge in async_default_edges]

    assert sync_default_next == ["n|fallback"]
    assert async_default_next == sync_default_next
    assert sync_default_decision.selected == [
        ("e-default", "n|fallback", "default")
    ]
