from __future__ import annotations

import pytest

from kogwistar.acl.graph import ACLGraph, ACLRecord
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.subsystems.acl import ACLSubsystem
from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.server.auth_middleware import claims_ctx
from tests._helpers.fake_backend import build_fake_backend


def _mk_span(doc_id: str) -> Span:
    span = Span.from_dummy_for_conversation()
    span.doc_id = doc_id
    return span


def _mk_node(*, node_id: str, doc_id: str) -> Node:
    return Node(
        id=node_id,
        label=node_id,
        type="entity",
        summary=node_id,
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"entity_type": "thing", "level_from_root": 0},
        embedding=[1.0, 0.0, 0.0],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _mk_edge(*, edge_id: str, src: str, tgt: str, doc_id: str) -> Edge:
    return Edge(
        id=edge_id,
        label=edge_id,
        type="relationship",
        summary=edge_id,
        relation="related_to",
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"entity_type": "relationship", "level_from_root": 0},
        embedding=[0.0, 1.0, 0.0],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def test_acl_latest_record_supersedes_previous_state():
    acl = ACLGraph()
    acl.add_record(
        truth_graph="conversation",
        entity_id="node-1",
        version=1,
        mode="private",
        owner_id="user-a",
    )
    acl.add_record(
        truth_graph="conversation",
        entity_id="node-1",
        version=2,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
        supersedes_version=1,
    )

    record = acl.latest_record(truth_graph="conversation", entity_id="node-1")
    assert record is not None
    assert record.version == 2
    assert record.mode == "shared"


def test_acl_decide_uses_truth_graph_overlay_for_any_graph_type():
    acl = ACLGraph()
    acl.add_record(
        truth_graph="workflow",
        entity_id="wf-1",
        version=1,
        mode="scope",
        security_scope="tenant-a",
    )

    allowed = acl.decide(
        truth_graph="workflow",
        entity_id="wf-1",
        principal_id="user-a",
        security_scope="tenant-a",
    )
    denied = acl.decide(
        truth_graph="workflow",
        entity_id="wf-1",
        principal_id="user-a",
        security_scope="tenant-b",
    )

    assert allowed.visible is True
    assert allowed.reason == "scope_match"
    assert denied.visible is False
    assert denied.reason == "scope_mismatch"


def test_acl_strictest_source_mode_picks_most_restrictive_visibility():
    acl = ACLGraph()
    records = [
        ACLRecord(
            target=acl.add_record(
                truth_graph="knowledge",
                entity_id="n-1",
                version=1,
                mode="group",
            ).target,
            version=1,
            mode="group",
        ),
        ACLRecord(
            target=acl.add_record(
                truth_graph="knowledge",
                entity_id="n-2",
                version=1,
                mode="private",
            ).target,
            version=1,
            mode="private",
        ),
    ]
    assert acl.strictest_source_mode(records) == "private"


def test_engine_exposes_acl_graph_overlay():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    assert isinstance(engine.acl_graph, ACLGraph)
    assert isinstance(engine.acl, ACLSubsystem)


def test_engine_acl_helpers_round_trip():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    engine.record_acl(
        truth_graph="conversation",
        entity_id="node-99",
        version=1,
        mode="private",
        created_by="user-a",
        owner_id="user-a",
    )
    decision = engine.decide_acl(
        truth_graph="conversation",
        entity_id="node-99",
        principal_id="user-b",
    )
    assert decision.visible is False
    assert decision.reason == "private"
    assert engine.acl_graph.latest_record(
        truth_graph="conversation",
        entity_id="node-99",
    ).created_by == "user-a"


def test_engine_acl_helpers_round_trip_from_persisted_graph_truth():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    engine.record_acl(
        truth_graph="conversation",
        entity_id="node-100",
        version=1,
        mode="private",
        created_by="user-a",
        owner_id="user-a",
    )
    engine.acl_graph = ACLGraph()
    decision = engine.decide_acl(
        truth_graph="conversation",
        entity_id="node-100",
        principal_id="user-b",
    )
    persisted_acl_nodes = engine.read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_truth_graph": "conversation"},
                {"acl_target_entity_id": "node-100"},
            ]
        },
    )
    assert decision.visible is False
    assert decision.reason == "private"
    assert len(persisted_acl_nodes) == 1


def test_acl_grain_mapping_supports_document_grounding_span_node_edge_and_artifact():
    acl = ACLGraph()
    grains = ("document", "grounding", "span", "node", "edge", "artifact")
    for index, grain in enumerate(grains, start=1):
        target_item_id = "span-1" if grain == "span" else None
        acl.add_record(
            grain=grain,
            truth_graph="knowledge",
            entity_id=f"target-{grain}",
            target_item_id=target_item_id,
            version=index,
            mode="private",
            owner_id="user-a",
        )

    for grain in grains:
        record = acl.latest_record(
            grain=grain,
            truth_graph="knowledge",
            entity_id=f"target-{grain}",
            target_item_id="span-1" if grain == "span" else None,
        )
        assert record is not None
        assert record.target.grain == grain
        assert record.target.target_item_id == ("span-1" if grain == "span" else None)


def test_acl_span_grain_targets_specific_span_item():
    acl = ACLGraph()
    acl.add_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="span-1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    acl.add_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="span-2",
        version=1,
        mode="private",
        owner_id="user-a",
    )

    span_1 = acl.latest_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="span-1",
    )
    span_2 = acl.latest_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="span-2",
    )
    assert span_1 is not None
    assert span_2 is not None
    assert span_1.target.target_item_id == "span-1"
    assert span_1.mode == "shared"
    assert span_2.target.target_item_id == "span-2"
    assert span_2.mode == "private"

    allowed = acl.decide(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="span-1",
        principal_id="user-b",
    )
    denied = acl.decide(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="span-2",
        principal_id="user-b",
    )

    assert allowed.visible is True
    assert allowed.reason == "principal_share"
    assert denied.visible is False
    assert denied.reason == "private"


def test_acl_span_target_item_index_maps_back_to_entity_ids():
    acl = ACLGraph()
    acl.add_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="sp:1",
        version=1,
        mode="shared",
        owner_id="user-a",
    )
    acl.add_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="edge-1",
        target_item_id="sp:2",
        version=1,
        mode="private",
        owner_id="user-a",
    )

    assert acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    ) == ("node-1",)
    assert acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:2",
    ) == ("edge-1",)
    assert acl.record_targets_for_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )[0].target.entity_id == "node-1"


def test_acl_same_span_item_can_have_distinct_node_level_acl():
    acl = ACLGraph()
    acl.add_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="sp:shared",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    acl.add_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-2",
        target_item_id="sp:shared",
        version=1,
        mode="private",
        owner_id="user-a",
    )

    node_1 = acl.decide(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="sp:shared",
        principal_id="user-b",
    )
    node_2 = acl.decide(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-2",
        target_item_id="sp:shared",
        principal_id="user-b",
    )

    assert node_1.visible is True
    assert node_1.reason == "principal_share"
    assert node_2.visible is False
    assert node_2.reason == "private"
    assert acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:shared",
    ) == ("node-1", "node-2")


def test_engine_acl_can_target_specific_span_item_for_node_and_edge():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    node_left = _mk_node(node_id="node-left", doc_id="doc-1")
    node_right = _mk_node(node_id="node-right", doc_id="doc-1")
    edge = _mk_edge(edge_id="edge-1", src=node_left.id, tgt=node_right.id, doc_id="doc-1")
    engine.write.add_node(node_left)
    engine.write.add_node(node_right)
    engine.write.add_edge(edge)

    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=node_left.id,
        target_item_id="span-node-1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=node_left.id,
        target_item_id="span-node-2",
        version=1,
        mode="private",
        owner_id="user-a",
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=edge.id,
        target_item_id="span-edge-1",
        version=1,
        mode="scope",
        owner_id="user-a",
        security_scope="tenant-a",
    )

    node_span_allowed = engine.decide_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=node_left.id,
        target_item_id="span-node-1",
        principal_id="user-b",
    )
    node_span_denied = engine.decide_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=node_left.id,
        target_item_id="span-node-2",
        principal_id="user-b",
    )
    edge_span_allowed = engine.decide_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=edge.id,
        target_item_id="span-edge-1",
        principal_id="user-c",
        security_scope="tenant-a",
    )
    edge_span_denied = engine.decide_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=edge.id,
        target_item_id="span-edge-1",
        principal_id="user-c",
        security_scope="tenant-b",
    )

    assert node_span_allowed.visible is True
    assert node_span_allowed.reason == "principal_share"
    assert node_span_denied.visible is False
    assert node_span_denied.reason == "private"
    assert edge_span_allowed.visible is True
    assert edge_span_allowed.reason == "scope_match"
    assert edge_span_denied.visible is False
    assert edge_span_denied.reason == "scope_mismatch"
    assert engine.acl_graph.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="span-node-1",
    ) == ("node-left",)
    assert engine.acl_graph.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="span-edge-1",
    ) == ("edge-1",)
    assert engine.acl_entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="span-node-1",
    ) == ("node-left",)
    assert engine.acl_entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="span-edge-1",
    ) == ("edge-1",)


def test_engine_acl_can_scope_same_span_item_across_multiple_nodes():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_id="sp:shared",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-b",
        target_item_id="sp:shared",
        version=1,
        mode="private",
        owner_id="user-a",
    )

    assert engine.decide_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_id="sp:shared",
        principal_id="user-b",
    ).visible is True
    assert engine.decide_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-b",
        target_item_id="sp:shared",
        principal_id="user-b",
    ).visible is False


def test_engine_acl_usage_requires_both_node_and_span_usage_access():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-a",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_id="sp:1",
        version=1,
        mode="private",
        owner_id="user-a",
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_id="sp:2",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )

    denied = engine.decide_acl_usage(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_id="sp:1",
        principal_id="user-b",
    )
    allowed = engine.decide_acl_usage(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_id="sp:2",
        principal_id="user-b",
    )

    assert denied.visible is False
    assert denied.reason == "span_private"
    assert denied.node_decision.visible is True
    assert denied.item_decision.visible is False
    assert allowed.visible is True
    assert allowed.reason == "node_and_span_visible"


def test_engine_acl_node_read_requires_all_underlying_span_usages():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-summary",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-summary",
        target_item_id="sp:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-summary",
        target_item_id="sp:2",
        version=1,
        mode="private",
        owner_id="user-a",
    )

    denied = engine.decide_acl_node_read(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-summary",
        target_item_ids=["sp:1", "sp:2"],
        principal_id="user-b",
    )
    allowed = engine.decide_acl_node_read(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-summary",
        target_item_ids=["sp:1"],
        principal_id="user-b",
    )

    assert denied.visible is False
    assert denied.reason == "span_private"
    assert denied.node_decision.visible is True
    assert len(denied.item_decisions) == 2
    assert denied.item_decisions[0].visible is True
    assert denied.item_decisions[1].visible is False
    assert allowed.visible is True
    assert allowed.reason == "node_and_all_items_visible"


def test_engine_acl_node_read_denies_when_node_acl_is_missing():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    engine.record_acl(
        grain="grounding",
        truth_graph="knowledge",
        entity_id="node-partial",
        target_item_id="gr:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-partial",
        target_item_id="sp:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )

    decision = engine.decide_acl_node_read(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-partial",
        grounding_item_ids=["gr:1"],
        target_item_ids=["sp:1"],
        principal_id="user-b",
    )

    assert decision.visible is False
    assert decision.reason == "node_no_acl_record"
    assert decision.node_decision.record is None
    assert [item.visible for item in decision.item_decisions] == [True, True]


def test_engine_acl_node_read_denies_when_grounding_acl_is_missing():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-partial",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-partial",
        target_item_id="sp:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )

    decision = engine.decide_acl_node_read(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-partial",
        grounding_item_ids=["gr:missing"],
        target_item_ids=["sp:1"],
        principal_id="user-b",
    )

    assert decision.visible is False
    assert decision.reason == "grounding_no_acl_record"
    assert decision.node_decision.visible is True
    assert [item.visible for item in decision.item_decisions] == [False, True]


def test_engine_acl_node_read_denies_when_span_acl_is_missing():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-partial",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="grounding",
        truth_graph="knowledge",
        entity_id="node-partial",
        target_item_id="gr:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )

    decision = engine.decide_acl_node_read(
        item_grain="span",
        truth_graph="knowledge",
        entity_id="node-partial",
        grounding_item_ids=["gr:1"],
        target_item_ids=["sp:missing"],
        principal_id="user-b",
    )

    assert decision.visible is False
    assert decision.reason == "span_no_acl_record"
    assert decision.node_decision.visible is True
    assert [item.visible for item in decision.item_decisions] == [True, False]


def test_engine_node_read_uses_persisted_acl_graph_for_node_grounding_and_span():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    node = _mk_node(node_id="node-with-grounding", doc_id="doc-1")
    engine.write.add_node(node)
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id=node.id,
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="grounding",
        truth_graph="knowledge",
        entity_id=node.id,
        target_item_id="gr:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="knowledge",
        entity_id=node.id,
        target_item_id="sp:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.acl_graph = ACLGraph()

    decision = engine.decide_acl_node_read(
        item_grain="span",
        truth_graph="knowledge",
        entity_id=node.id,
        grounding_item_ids=["gr:1"],
        target_item_ids=["sp:1"],
        principal_id="user-b",
    )
    acl_nodes = engine.read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_truth_graph": "knowledge"},
                {"acl_target_entity_id": node.id},
            ]
        },
    )

    assert decision.visible is True
    assert decision.reason == "node_and_all_items_visible"
    assert [item.visible for item in decision.item_decisions] == [True, True]
    assert sorted(n.metadata["acl_target_grain"] for n in acl_nodes) == [
        "grounding",
        "node",
        "span",
    ]


def test_engine_get_node_acl_checked_denies_partial_coverage_and_keeps_raw_read():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    node = _mk_node(node_id="node-read-entrypoint", doc_id="doc-1")
    engine.write.add_node(node)
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id=node.id,
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="grounding",
        truth_graph="knowledge",
        entity_id=node.id,
        target_item_id="gr:1",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )

    raw_nodes = engine.read.get_nodes(ids=[node.id])
    assert raw_nodes[0].id == node.id

    with pytest.raises(PermissionError, match="span_no_acl_record"):
        engine.get_node_acl_checked(
            node.id,
            grounding_item_ids=["gr:1"],
            target_item_ids=["sp:1"],
            principal_id="user-b",
        )


def test_acl_enabled_engine_makes_normal_read_write_acl_aware():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        acl_enabled=True,
    )
    node = _mk_node(node_id="node-acl-enabled", doc_id="doc-1")
    node.metadata.update(
        {
            "visibility": "private",
            "owner_agent_id": "agent-a",
            "security_scope": "tenant-a",
        }
    )

    token_a = claims_ctx.set(
        {"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        engine.write.add_node(node)
        assert engine.read.get_nodes(ids=[node.id])[0].id == node.id
    finally:
        claims_ctx.reset(token_a)

    token_b = claims_ctx.set(
        {"agent_id": "agent-b", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        assert engine.raw_read.get_nodes(ids=[node.id])[0].id == node.id
        assert engine.read.get_nodes(ids=[node.id]) == []
    finally:
        claims_ctx.reset(token_b)

    acl_nodes = engine.raw_read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_truth_graph": "knowledge"},
                {"acl_target_entity_id": node.id},
            ]
        },
    )
    assert sorted(n.metadata["acl_target_grain"] for n in acl_nodes) == [
        "grounding",
        "node",
        "span",
    ]
