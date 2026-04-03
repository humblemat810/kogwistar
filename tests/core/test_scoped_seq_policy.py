from __future__ import annotations

import pytest

from kogwistar.conversation.policy import install_engine_hooks
from kogwistar.engine_core.scoped_seq import (
    ScopedSeqHookConfig,
    install_scoped_seq_hooks,
)

pytestmark = pytest.mark.ci


class _FakeMeta:
    def __init__(self) -> None:
        self._seq_by_scope: dict[str, int] = {}

    def next_scoped_seq(self, scope_id: str) -> int:
        next_value = self._seq_by_scope.get(scope_id, 0) + 1
        self._seq_by_scope[scope_id] = next_value
        return next_value

    def current_scoped_seq(self, scope_id: str) -> int:
        return self._seq_by_scope.get(scope_id, 0)


class _FakeEngine:
    def __init__(self, kg_graph_type: str) -> None:
        self.kg_graph_type = kg_graph_type
        self.meta_sqlite = _FakeMeta()
        self.pre_add_node_hooks: list = []
        self.pre_add_edge_hooks: list = []
        self.pre_add_pure_edge_hooks: list = []
        self.allow_missing_doc_id_on_endpoint_rows_hooks: list = []


class _Subject:
    def __init__(self, **kwargs) -> None:
        metadata = kwargs.pop("metadata", None)
        self.metadata = dict(metadata or {})
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_scoped_seq_hooks_stamp_custom_node_and_edge_scopes():
    engine = _FakeEngine("governance")
    install_scoped_seq_hooks(
        engine,
        ScopedSeqHookConfig(
            should_stamp_node=lambda _e, node: getattr(node, "entity_type", None)
            == "governance_turn",
            scope_id_for_node=lambda _e, node: getattr(
                node, "agent_interaction_id", None
            ),
            should_stamp_edge=lambda _e, edge: (getattr(edge, "metadata", {}) or {}).get(
                "entity_type"
            )
            == "governance_edge",
            scope_id_for_edge=lambda _e, edge: (getattr(edge, "metadata", {}) or {}).get(
                "agent_interaction_id"
            ),
        ),
    )

    node = _Subject(
        agent_interaction_id="interaction-1",
        entity_type="governance_turn",
    )
    engine.pre_add_node_hooks[0](node)
    assert node.metadata["seq"] == 1

    # Existing seq values are preserved, so repeated pre-add hook runs stay idempotent.
    engine.pre_add_node_hooks[0](node)
    assert node.metadata["seq"] == 1

    ignored = _Subject(
        agent_interaction_id="interaction-1",
        entity_type="not_governance_turn",
    )
    engine.pre_add_node_hooks[0](ignored)
    assert "seq" not in ignored.metadata

    edge = _Subject(
        metadata={
            "entity_type": "governance_edge",
            "agent_interaction_id": "interaction-1",
        }
    )
    assert engine.pre_add_edge_hooks[0](edge) is False
    assert edge.metadata["seq"] == 2
    assert len(engine.pre_add_pure_edge_hooks) == 1


def test_conversation_policy_shim_remains_conversation_only():
    conv_engine = _FakeEngine("conversation")
    install_engine_hooks(conv_engine)
    conv_node = _Subject(conversation_id="conv-1")
    conv_engine.pre_add_node_hooks[0](conv_node)
    assert conv_node.metadata["seq"] == 1
    assert conv_engine.meta_sqlite.current_scoped_seq("conv-1") == 1

    kg_engine = _FakeEngine("knowledge")
    install_engine_hooks(kg_engine)
    kg_node = _Subject(conversation_id="conv-1")
    kg_engine.pre_add_node_hooks[0](kg_node)
    assert "seq" not in kg_node.metadata
    assert kg_engine.meta_sqlite.current_scoped_seq("conv-1") == 0
