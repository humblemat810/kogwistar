"""Deterministic context compression contracts; source graph stays intact."""

from __future__ import annotations

import pytest

from kogwistar.agent import (
    CompressionPolicy,
    build_compression_request,
    build_compression_workflow,
    persist_summary_projection,
    policy_for,
    should_request_compression,
)


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def test_policies_are_bounded_and_threshold_hook_is_pure() -> None:
    items = [
        {"id": "n1", "conversation_id": "c1", "summary": "one"},
        {"id": "n2", "conversation_id": "c1", "summary": "two"},
        {"id": "n3", "conversation_id": "c2", "summary": "private"},
    ]
    policy = policy_for("ultra")
    before = [dict(item) for item in items]
    assert should_request_compression(items, policy=CompressionPolicy("high", 2, 5))
    request, decision = build_compression_request(
        conversation_id="c1", items=items, policy=policy
    )
    assert request.cross_conversation is False
    assert decision.source_refs == ("n1", "n2")
    assert items == before


def test_compression_runs_as_ordinary_workflow_design() -> None:
    design = build_compression_workflow()
    assert design.nodes[0].metadata["wf_mode"] == "normal"
    assert any(node.op == "agent.context_compress" for node in design.nodes)


def test_cross_conversation_requires_explicit_opt_in_and_acl_filter() -> None:
    items = [
        {"id": "own", "conversation_id": "c1", "summary": "allowed"},
        {"id": "other", "conversation_id": "c2", "summary": "secret"},
    ]
    _, denied = build_compression_request(
        conversation_id="c1", items=items, allow_cross_conversation=False
    )
    assert denied.source_refs == ("own",)
    _, filtered = build_compression_request(
        conversation_id="c1",
        items=items,
        allow_cross_conversation=True,
        authorize=lambda item: item["id"] == "own",
    )
    assert filtered.source_refs == ("own",)


class _Writer:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def add_node(self, node) -> None:
        self.nodes.append(node)

    def add_edge(self, edge) -> None:
        self.edges.append(edge)


class _Engine:
    def __init__(self) -> None:
        self.write = _Writer()


def test_summary_is_additive_and_provenance_is_reproducible() -> None:
    engine = _Engine()
    result = persist_summary_projection(
        engine,
        conversation_id="c1",
        source_refs=["n1", "n2"],
        summary_text="compressed",
        policy=policy_for("medium"),
        run_id="run-1",
    )
    assert result.source_refs == ("n1", "n2")
    assert engine.write.nodes[0].metadata["entity_type"] == "conversation_context_summary"
    assert engine.write.edges[0].relation == "summarizes"
    assert engine.write.edges[0].target_ids == ["n1", "n2"]
    again = persist_summary_projection(
        engine,
        conversation_id="c1",
        source_refs=["n1", "n2"],
        summary_text="compressed",
        policy=policy_for("medium"),
        run_id="run-1",
    )
    assert again.summary_node_id == result.summary_node_id
    assert again.provenance_edge_id == result.provenance_edge_id


def test_compression_keeps_alternate_branches_separate() -> None:
    items = [
        {"id": "a1", "conversation_id": "c1", "branch_id": "a", "summary": "old-a"},
        {"id": "b1", "conversation_id": "c1", "branch_id": "b", "summary": "new-b"},
        {"id": "b2", "conversation_id": "c1", "branch_id": "b", "summary": "latest-b"},
    ]
    _, decision = build_compression_request(
        conversation_id="c1", items=items, policy=CompressionPolicy("low", 8, 100)
    )
    assert decision.source_refs == ("b1", "b2")
