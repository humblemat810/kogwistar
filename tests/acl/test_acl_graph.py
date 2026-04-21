from __future__ import annotations

import json
import threading

import pytest

from kogwistar.acl.graph import ACLGraph, ACLRecord, ACLTarget
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.subsystems import ACLAwareReadSubsystem, ACLAwareWriteSubsystem
from kogwistar.engine_core.subsystems.acl import ACLSubsystem
from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.server.auth_middleware import claims_ctx
from tests.conftest import _make_engine_pair
from tests._helpers.embeddings import build_test_embedding_function
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


def _mk_acl_fake_engine(tmp_path) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "acl_acceptance"),
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        embedding_function=build_test_embedding_function("constant", dim=3),
        acl_enabled=True,
    )


def _visible_node_ids(engine: GraphKnowledgeEngine, *, agent_id: str, scope: str) -> set[str]:
    token = claims_ctx.set({"agent_id": agent_id, "security_scope": scope, "ns": "knowledge"})
    try:
        return {
            node.id
            for node in engine.read.get_nodes(node_type=Node)
            if str((node.metadata or {}).get("entity_type") or "") != "acl_record"
        }
    finally:
        claims_ctx.reset(token)


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


def test_acceptance_insert_records_creator_owner_scope_and_truth_facts(tmp_path):
    engine = _mk_acl_fake_engine(tmp_path)
    node = _mk_node(node_id="accept-insert-a", doc_id="doc::accept::insert")
    node.metadata.update(
        {
            "visibility": "scope",
            "owner_agent_id": "agent-a",
            "security_scope": "tenant-a",
        }
    )

    token = claims_ctx.set({"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"})
    try:
        engine.write.add_node(node)
    finally:
        claims_ctx.reset(token)

    truth_nodes = engine.raw_read.get_nodes(node_type=Node, ids=[node.id])
    acl_rows = engine.raw_read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_target_entity_id": node.id},
            ]
        },
    )
    node_acl = [row for row in acl_rows if row.metadata["acl_target_grain"] == "node"][0]

    assert [row.id for row in truth_nodes] == [node.id]
    assert node_acl.metadata["created_by"] == "agent-a"
    assert node_acl.metadata["owner_id"] == "agent-a"
    assert node_acl.metadata["security_scope"] == "tenant-a"
    assert node_acl.metadata["acl_mode"] == "scope"


def test_acceptance_scope_isolation_and_user_specific_visible_sets(tmp_path):
    engine = _mk_acl_fake_engine(tmp_path)
    node_a = _mk_node(node_id="accept-visible-a-private", doc_id="doc::visible::a")
    node_a.metadata.update({"visibility": "private", "owner_agent_id": "agent-a"})
    node_shared = _mk_node(node_id="accept-visible-shared", doc_id="doc::visible::shared")
    node_shared.metadata.update(
        {
            "visibility": "shared",
            "owner_agent_id": "agent-a",
            "shared_with_agents": ["agent-b"],
            "security_scope": "tenant-a",
        }
    )
    node_b = _mk_node(node_id="accept-visible-b-private", doc_id="doc::visible::b")
    node_b.metadata.update({"visibility": "private", "owner_agent_id": "agent-b"})
    node_scope_a = _mk_node(node_id="accept-scope-a", doc_id="doc::visible::scope")
    node_scope_a.metadata.update(
        {
            "visibility": "scope",
            "owner_agent_id": "agent-a",
            "security_scope": "tenant-a",
        }
    )

    for agent, scope, node in (
        ("agent-a", "tenant-a", node_a),
        ("agent-a", "tenant-a", node_shared),
        ("agent-b", "tenant-b", node_b),
        ("agent-a", "tenant-a", node_scope_a),
    ):
        token = claims_ctx.set({"agent_id": agent, "security_scope": scope, "ns": "knowledge"})
        try:
            engine.write.add_node(node)
        finally:
            claims_ctx.reset(token)

    assert _visible_node_ids(engine, agent_id="agent-a", scope="tenant-a") == {
        node_a.id,
        node_shared.id,
        node_scope_a.id,
    }
    assert _visible_node_ids(engine, agent_id="agent-b", scope="tenant-a") == {
        node_b.id,
        node_shared.id,
        node_scope_a.id,
    }
    assert _visible_node_ids(engine, agent_id="agent-b", scope="tenant-b") == {
        node_b.id,
        node_shared.id,
    }


def test_acceptance_hidden_nodes_do_not_leak_topology_paths_or_ranking(tmp_path):
    engine = _mk_acl_fake_engine(tmp_path)
    visible = _mk_node(node_id="accept-topology-visible", doc_id="doc::topology::visible")
    visible.metadata.update({"visibility": "shared", "owner_agent_id": "agent-a", "shared_with_agents": ["agent-b"]})
    hidden = _mk_node(node_id="accept-topology-hidden", doc_id="doc::topology::hidden")
    hidden.metadata.update({"visibility": "private", "owner_agent_id": "agent-a"})
    edge = _mk_edge(
        edge_id="accept-topology-edge",
        src=visible.id,
        tgt=hidden.id,
        doc_id="doc::topology::edge",
    )
    edge.metadata.update({"visibility": "shared", "owner_agent_id": "agent-a", "shared_with_agents": ["agent-b"]})

    token = claims_ctx.set({"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"})
    try:
        engine.write.add_node(visible)
        engine.write.add_node(hidden)
        engine.write.add_edge(edge)
    finally:
        claims_ctx.reset(token)

    token = claims_ctx.set({"agent_id": "agent-b", "security_scope": "tenant-a", "ns": "knowledge"})
    try:
        visible_nodes = engine.read.get_nodes(node_type=Node)
        visible_edges = engine.read.get_edges(edge_type=Edge)
        ranked_ids = [node.id for node in sorted(visible_nodes, key=lambda item: item.summary)]
        topology = [(edge.source_ids, edge.target_ids) for edge in visible_edges]
        path_summaries = [edge.summary for edge in visible_edges]
    finally:
        claims_ctx.reset(token)

    assert [node.id for node in visible_nodes] == [visible.id]
    assert hidden.id not in ranked_ids
    assert topology == []
    assert path_summaries == []


def test_acceptance_answer_assembly_uses_visible_sources_only(tmp_path):
    engine = _mk_acl_fake_engine(tmp_path)
    visible = _mk_node(node_id="accept-answer-visible", doc_id="doc::answer::visible")
    visible.summary = "visible answer evidence"
    visible.metadata.update({"visibility": "shared", "owner_agent_id": "agent-a", "shared_with_agents": ["agent-b"]})
    hidden = _mk_node(node_id="accept-answer-hidden", doc_id="doc::answer::hidden")
    hidden.summary = "hidden answer evidence"
    hidden.metadata.update({"visibility": "private", "owner_agent_id": "agent-a"})

    token = claims_ctx.set({"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"})
    try:
        engine.write.add_node(visible)
        engine.write.add_node(hidden)
    finally:
        claims_ctx.reset(token)

    def assemble_answer() -> dict[str, object]:
        nodes = engine.read.get_nodes(node_type=Node)
        return {
            "answer": " ".join(node.summary for node in nodes),
            "evidence": [node.id for node in nodes],
            "confidence": len(nodes),
            "uncertainty": [] if nodes else ["no visible evidence"],
        }

    token = claims_ctx.set({"agent_id": "agent-b", "security_scope": "tenant-a", "ns": "knowledge"})
    try:
        assembled = assemble_answer()
    finally:
        claims_ctx.reset(token)

    assert assembled["answer"] == "visible answer evidence"
    assert assembled["evidence"] == [visible.id]
    assert hidden.id not in assembled["evidence"]


def test_acceptance_derived_artifact_mixed_sources_records_linkage_and_acl_mode(tmp_path):
    engine = _mk_acl_fake_engine(tmp_path)
    public_source = _mk_node(node_id="accept-derived-public-source", doc_id="doc::derived::public")
    public_source.metadata.update({"visibility": "public", "owner_agent_id": "agent-a"})
    private_source = _mk_node(node_id="accept-derived-private-source", doc_id="doc::derived::private")
    private_source.metadata.update({"visibility": "private", "owner_agent_id": "agent-a"})
    artifact_id = "accept-derived-artifact"

    token = claims_ctx.set({"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"})
    try:
        engine.write.add_node(public_source)
        engine.write.add_node(private_source)
        source_records = [
            engine.acl._latest_persisted_acl_record(
                grain="node",
                truth_graph="knowledge",
                entity_id=public_source.id,
            ),
            engine.acl._latest_persisted_acl_record(
                grain="node",
                truth_graph="knowledge",
                entity_id=private_source.id,
            ),
        ]
        strictest_mode = engine.acl_graph.strictest_source_mode(
            [record for record in source_records if record is not None]
        )
        engine.record_acl(
            grain="artifact",
            truth_graph="knowledge",
            entity_id=artifact_id,
            version=1,
            mode=strictest_mode,
            created_by="agent-a",
            owner_id="agent-a",
            source_ids=[public_source.id, private_source.id],
            derivation_type="summary",
        )
    finally:
        claims_ctx.reset(token)

    denied = engine.decide_acl(
        grain="artifact",
        truth_graph="knowledge",
        entity_id=artifact_id,
        principal_id="agent-b",
        security_scope="tenant-a",
    )
    owner_allowed = engine.decide_acl(
        grain="artifact",
        truth_graph="knowledge",
        entity_id=artifact_id,
        principal_id="agent-a",
        security_scope="tenant-a",
    )
    acl_rows = engine.raw_read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_target_entity_id": artifact_id},
            ]
        },
    )

    assert denied.visible is False
    assert denied.reason == "private"
    assert owner_allowed.visible is True
    assert acl_rows[0].metadata["acl_target_grain"] == "artifact"
    assert acl_rows[0].metadata["acl_mode"] == "private"
    assert acl_rows[0].properties["source_ids"] == [public_source.id, private_source.id]
    assert acl_rows[0].properties["derivation_type"] == "summary"


def test_engine_can_rebuild_acl_graph_from_persisted_truth():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    engine.record_acl(
        truth_graph="conversation",
        entity_id="node-200",
        version=1,
        mode="shared",
        created_by="user-a",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="span",
        truth_graph="conversation",
        entity_id="node-200",
        target_item_id="sp:200",
        version=1,
        mode="private",
        created_by="user-a",
        owner_id="user-a",
    )
    engine.acl_graph = ACLGraph()

    rebuilt = engine.rebuild_acl_graph_from_truth()
    node_record = engine.acl_graph.latest_record(
        grain="node",
        truth_graph="conversation",
        entity_id="node-200",
    )
    span_ids = engine.acl_graph.entity_ids_for_target_item(
        truth_graph="conversation",
        grain="span",
        target_item_id="sp:200",
    )

    assert rebuilt["rebuilt_record_count"] == 2
    assert node_record is not None
    assert node_record.mode == "shared"
    assert span_ids == ("node-200",)


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


def test_acl_graph_lazy_cache_hits_and_lru_eviction():
    acl = ACLGraph(max_record_cache_size=1, max_target_cache_size=1)
    record_calls: list[tuple[str, str | None, str, str | None]] = []
    target_calls: list[tuple[str, str, str]] = []
    record_store = {
        ("knowledge", "node", "node-1", None): (
            ACLRecord(
                target=ACLTarget(
                    truth_graph="knowledge",
                    entity_id="node-1",
                    grain="node",
                ),
                version=1,
                mode="private",
                owner_id="user-a",
            ),
        ),
        ("knowledge", "node", "node-2", None): (
            ACLRecord(
                target=ACLTarget(
                    truth_graph="knowledge",
                    entity_id="node-2",
                    grain="node",
                ),
                version=1,
                mode="shared",
                owner_id="user-a",
            ),
        ),
    }
    target_store = {
        ("knowledge", "span", "sp:1"): ("node-1", "node-2"),
    }

    def record_loader(
        *,
        truth_graph: str,
        grain: str | None,
        entity_id: str,
        target_item_id: str | None,
    ):
        record_calls.append((truth_graph, grain, entity_id, target_item_id))
        return record_store.get((truth_graph, grain, entity_id, target_item_id), ())

    def target_loader(*, truth_graph: str, grain: str, target_item_id: str):
        target_calls.append((truth_graph, grain, target_item_id))
        return target_store.get((truth_graph, grain, target_item_id), ())

    acl = ACLGraph(max_record_cache_size=1, max_target_cache_size=1)
    acl.bind_loaders(record_loader=record_loader, target_loader=target_loader)

    first = acl.latest_record(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-1",
    )
    second = acl.latest_record(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-1",
    )
    ids_first = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )
    ids_second = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )
    acl.latest_record(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-2",
    )
    third = acl.latest_record(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-1",
    )

    assert first is not None
    assert second is not None
    assert first.version == 1
    assert second.version == 1
    assert ids_first == ("node-1", "node-2")
    assert ids_second == ("node-1", "node-2")
    assert third is not None
    assert third.version == 1
    assert record_calls.count(("knowledge", "node", "node-1", None)) == 2
    assert record_calls.count(("knowledge", "node", "node-2", None)) == 1
    assert target_calls.count(("knowledge", "span", "sp:1")) == 1


def test_acl_graph_no_cache_mode_always_reads_loader_and_keeps_view_empty():
    acl = ACLGraph(cache_enabled=False)
    record_calls: list[tuple[str, str | None, str, str | None]] = []
    target_calls: list[tuple[str, str, str]] = []
    record = ACLRecord(
        target=ACLTarget(
            truth_graph="knowledge",
            entity_id="node-1",
            grain="node",
        ),
        version=1,
        mode="private",
        owner_id="user-a",
    )

    def record_loader(
        *,
        truth_graph: str,
        grain: str | None,
        entity_id: str,
        target_item_id: str | None,
    ):
        record_calls.append((truth_graph, grain, entity_id, target_item_id))
        return (record,)

    def target_loader(*, truth_graph: str, grain: str, target_item_id: str):
        target_calls.append((truth_graph, grain, target_item_id))
        return ("node-1",)

    acl.bind_loaders(record_loader=record_loader, target_loader=target_loader)

    first = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    second = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    ids_first = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )
    ids_second = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )

    assert first is record
    assert second is record
    assert ids_first == ("node-1",)
    assert ids_second == ("node-1",)
    assert record_calls == [
        ("knowledge", "node", "node-1", None),
        ("knowledge", "node", "node-1", None),
    ]
    assert target_calls == [
        ("knowledge", "span", "sp:1"),
        ("knowledge", "span", "sp:1"),
    ]
    assert acl.iter_records() == ()


def test_acl_graph_invalidate_clears_all_grains_for_entity_and_refreshes_target_index():
    acl = ACLGraph(max_record_cache_size=4, max_target_cache_size=4)
    record_store = {
        ("knowledge", "node", "node-1", None): ACLRecord(
            target=ACLTarget(
                truth_graph="knowledge",
                entity_id="node-1",
                grain="node",
            ),
            version=1,
            mode="private",
            owner_id="user-a",
        ),
        ("knowledge", "span", "node-1", "sp:1"): ACLRecord(
            target=ACLTarget(
                truth_graph="knowledge",
                entity_id="node-1",
                grain="span",
                target_item_id="sp:1",
            ),
            version=1,
            mode="shared",
            owner_id="user-a",
        ),
    }
    target_store = {
        ("knowledge", "span", "sp:1"): ("node-1",),
    }
    record_calls: list[tuple[str, str | None, str, str | None]] = []
    target_calls: list[tuple[str, str, str]] = []

    def record_loader(
        *,
        truth_graph: str,
        grain: str | None,
        entity_id: str,
        target_item_id: str | None,
    ):
        record_calls.append((truth_graph, grain, entity_id, target_item_id))
        return (record_store[(truth_graph, grain, entity_id, target_item_id)],)

    def target_loader(*, truth_graph: str, grain: str, target_item_id: str):
        target_calls.append((truth_graph, grain, target_item_id))
        return target_store[(truth_graph, grain, target_item_id)]

    acl.bind_loaders(record_loader=record_loader, target_loader=target_loader)

    node_before = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    span_before = acl.latest_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="sp:1",
    )
    target_before = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )

    record_store[("knowledge", "node", "node-1", None)] = ACLRecord(
        target=ACLTarget(
            truth_graph="knowledge",
            entity_id="node-1",
            grain="node",
        ),
        version=2,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=("user-b",),
    )
    record_store[("knowledge", "span", "node-1", "sp:1")] = ACLRecord(
        target=ACLTarget(
            truth_graph="knowledge",
            entity_id="node-1",
            grain="span",
            target_item_id="sp:1",
        ),
        version=2,
        mode="private",
        owner_id="user-a",
    )
    target_store[("knowledge", "span", "sp:1")] = ("node-1", "node-2")

    node_stale = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    span_stale = acl.latest_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="sp:1",
    )
    target_stale = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )

    acl.invalidate(truth_graph="knowledge", entity_id="node-1")

    node_fresh = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    span_fresh = acl.latest_record(
        grain="span",
        truth_graph="knowledge",
        entity_id="node-1",
        target_item_id="sp:1",
    )
    target_fresh = acl.entity_ids_for_target_item(
        truth_graph="knowledge",
        grain="span",
        target_item_id="sp:1",
    )

    assert node_before is not None and node_before.version == 1
    assert span_before is not None and span_before.version == 1
    assert target_before == ("node-1",)
    assert node_stale is not None and node_stale.version == 1
    assert span_stale is not None and span_stale.version == 1
    assert target_stale == ("node-1",)
    assert node_fresh is not None and node_fresh.version == 2
    assert span_fresh is not None and span_fresh.version == 2
    assert target_fresh == ("node-1", "node-2")
    assert record_calls.count(("knowledge", "node", "node-1", None)) == 2
    assert record_calls.count(("knowledge", "span", "node-1", "sp:1")) == 2
    assert target_calls.count(("knowledge", "span", "sp:1")) == 2


def test_acl_graph_concurrent_read_and_invalidate_remain_consistent():
    acl = ACLGraph(max_record_cache_size=4, max_target_cache_size=4)
    store_lock = threading.Lock()
    record_store = {
        ("knowledge", "node", "node-1", None): ACLRecord(
            target=ACLTarget(
                truth_graph="knowledge",
                entity_id="node-1",
                grain="node",
            ),
            version=1,
            mode="private",
            owner_id="user-a",
        ),
    }

    def record_loader(
        *,
        truth_graph: str,
        grain: str | None,
        entity_id: str,
        target_item_id: str | None,
    ):
        with store_lock:
            return (record_store[(truth_graph, grain, entity_id, target_item_id)],)

    acl.bind_loaders(record_loader=record_loader)

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(200):
                record = acl.latest_record(
                    grain="node",
                    truth_graph="knowledge",
                    entity_id="node-1",
                )
                assert record is not None
        except BaseException as exc:  # pragma: no cover - thread capture
            errors.append(exc)

    def writer() -> None:
        try:
            barrier.wait()
            for version in range(2, 5):
                with store_lock:
                    record_store[("knowledge", "node", "node-1", None)] = ACLRecord(
                        target=ACLTarget(
                            truth_graph="knowledge",
                            entity_id="node-1",
                            grain="node",
                        ),
                        version=version,
                        mode="shared" if version % 2 == 0 else "private",
                        owner_id="user-a",
                    )
                acl.invalidate(truth_graph="knowledge", entity_id="node-1")
                latest = acl.latest_record(
                    grain="node",
                    truth_graph="knowledge",
                    entity_id="node-1",
                )
                assert latest is not None
        except BaseException as exc:  # pragma: no cover - thread capture
            errors.append(exc)

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=writer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    final = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    assert not errors
    assert final is not None
    assert final.version == 4


def test_acl_graph_tombstone_write_invalidates_stale_cached_record():
    acl = ACLGraph(max_record_cache_size=4, max_target_cache_size=4)
    record_store = {
        ("knowledge", "node", "node-1", None): ACLRecord(
            target=ACLTarget(
                truth_graph="knowledge",
                entity_id="node-1",
                grain="node",
            ),
            version=1,
            mode="shared",
            owner_id="user-a",
        )
    }
    loader_calls: list[tuple[str, str | None, str, str | None]] = []

    def record_loader(
        *,
        truth_graph: str,
        grain: str | None,
        entity_id: str,
        target_item_id: str | None,
    ):
        loader_calls.append((truth_graph, grain, entity_id, target_item_id))
        return (record_store[(truth_graph, grain, entity_id, target_item_id)],)

    acl.bind_loaders(record_loader=record_loader)

    before = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    assert before is not None
    assert before.version == 1

    record_store[("knowledge", "node", "node-1", None)] = ACLRecord(
        target=ACLTarget(
            truth_graph="knowledge",
            entity_id="node-1",
            grain="node",
        ),
        version=2,
        mode="private",
        owner_id="user-a",
        tombstoned=True,
        supersedes_version=1,
    )
    acl.invalidate(truth_graph="knowledge", entity_id="node-1")

    after = acl.latest_record(grain="node", truth_graph="knowledge", entity_id="node-1")
    decision = acl.decide(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-1",
        principal_id="user-b",
    )

    assert after is not None
    assert after.version == 2
    assert after.tombstoned is True
    assert decision.visible is False
    assert decision.reason == "tombstoned"
    assert loader_calls.count(("knowledge", "node", "node-1", None)) == 2


def test_acl_prefetch_neighborhood_warms_seed_and_neighbor_items():
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
        target_item_id="sp:shared",
        version=1,
        mode="shared",
        owner_id="user-a",
        shared_with_principals=["user-b"],
    )
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-b",
        version=1,
        mode="private",
        owner_id="user-a",
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

    record_loader = engine.acl_graph._record_loader
    target_loader = engine.acl_graph._target_loader
    assert record_loader is not None
    assert target_loader is not None
    record_calls: list[tuple[str, str | None, str, str | None]] = []
    target_calls: list[tuple[str, str, str]] = []

    def wrapped_record_loader(
        *,
        truth_graph: str,
        grain: str | None,
        entity_id: str,
        target_item_id: str | None,
    ):
        record_calls.append((truth_graph, grain, entity_id, target_item_id))
        return record_loader(
            truth_graph=truth_graph,
            grain=grain,
            entity_id=entity_id,
            target_item_id=target_item_id,
        )

    def wrapped_target_loader(*, truth_graph: str, grain: str, target_item_id: str):
        target_calls.append((truth_graph, grain, target_item_id))
        return target_loader(
            truth_graph=truth_graph,
            grain=grain,
            target_item_id=target_item_id,
        )

    engine.acl_graph.clear()
    engine.acl_graph.bind_loaders(
        record_loader=wrapped_record_loader,
        target_loader=wrapped_target_loader,
    )

    warmed = engine.prefetch_acl_neighborhood(
        truth_graph="knowledge",
        entity_id="node-a",
        target_item_ids=["sp:shared", "sp:other"],
        max_items=1,
    )

    assert warmed["warmed_items"] == ("sp:shared",)
    assert warmed["warmed_neighbor_entity_ids"] == ("node-b",)
    assert record_calls == [
        ("knowledge", "node", "node-a", None),
        ("knowledge", "span", "node-a", "sp:shared"),
        ("knowledge", "node", "node-b", None),
    ]
    assert target_calls == [("knowledge", "span", "sp:shared")]


def test_engine_acl_no_cache_mode_reads_canonical_truth_each_time():
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        acl_cache_enabled=False,
    )
    engine.record_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-no-cache",
        version=1,
        mode="private",
        owner_id="user-a",
    )

    first = engine.decide_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-no-cache",
        principal_id="user-b",
    )
    second = engine.decide_acl(
        grain="node",
        truth_graph="knowledge",
        entity_id="node-no-cache",
        principal_id="user-b",
    )

    assert first.visible is False
    assert first.reason == "private"
    assert second.visible is False
    assert second.reason == "private"
    assert engine.acl_graph.iter_records() == ()


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


@pytest.mark.parametrize("backend_kind", ["fake", "chroma", "pg"], indirect=True)
def test_acl_smoke_across_fake_chroma_and_pg_backends(
    backend_kind: str, tmp_path, request
):
    sa_engine = None
    pg_schema = None
    if backend_kind == "pg":
        try:
            sa_engine = request.getfixturevalue("sa_engine")
            pg_schema = request.getfixturevalue("pg_schema")
        except Exception as exc:
            pytest.skip(f"pg backend requested but fixtures are unavailable: {exc}")

    kg_engine, _conversation_engine = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        embedding_kind="constant",
    )
    kg_engine.acl_enabled = True
    kg_engine.read = ACLAwareReadSubsystem(kg_engine, kg_engine.raw_read)
    kg_engine.write = ACLAwareWriteSubsystem(kg_engine, kg_engine.raw_write)

    node = _mk_node(node_id=f"node-acl-smoke-{backend_kind}", doc_id=f"doc::{backend_kind}")
    node.metadata.update(
        {
            "visibility": "shared",
            "owner_agent_id": "agent-a",
            "shared_with_agents": ["agent-b"],
            "security_scope": "tenant-a",
        }
    )

    token_write = claims_ctx.set(
        {"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        kg_engine.write.add_node(node)
    finally:
        claims_ctx.reset(token_write)

    grounding_ids, span_ids = kg_engine.acl.usage_ids_for_item(node)
    assert grounding_ids and span_ids

    acl_rows = kg_engine.raw_read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_truth_graph": "knowledge"},
                {"acl_target_entity_id": node.id},
            ]
        },
    )
    assert sorted(row.metadata["acl_target_grain"] for row in acl_rows) == [
        "grounding",
        "node",
        "span",
    ]

    token_read_ok = claims_ctx.set(
        {"agent_id": "agent-b", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        visible_nodes = kg_engine.read.get_nodes(ids=[node.id])
        assert [row.id for row in visible_nodes] == [node.id]
        assert kg_engine.decide_acl_node_read(
            item_grain="span",
            truth_graph="knowledge",
            entity_id=node.id,
            grounding_item_ids=grounding_ids,
            target_item_ids=span_ids,
            principal_id="agent-b",
            security_scope="tenant-a",
        ).visible is True
    finally:
        claims_ctx.reset(token_read_ok)

    token_read_bad = claims_ctx.set(
        {"agent_id": "agent-c", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        assert kg_engine.read.get_nodes(ids=[node.id]) == []
        assert (
            kg_engine.decide_acl_node_read(
                item_grain="span",
                truth_graph="knowledge",
                entity_id=node.id,
                grounding_item_ids=grounding_ids,
                target_item_ids=span_ids,
                principal_id="agent-c",
                security_scope="tenant-a",
            ).visible
            is False
        )
    finally:
        claims_ctx.reset(token_read_bad)


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"], indirect=True)
def test_acl_mid_write_failure_is_atomic_or_repairable_by_backend(
    backend_kind: str, tmp_path, request, monkeypatch
):
    sa_engine = None
    pg_schema = None
    if backend_kind == "pg":
        try:
            sa_engine = request.getfixturevalue("sa_engine")
            pg_schema = request.getfixturevalue("pg_schema")
        except Exception as exc:
            pytest.skip(f"pg backend requested but fixtures are unavailable: {exc}")

    kg_engine, _conversation_engine = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        embedding_kind="constant",
    )
    kg_engine.acl_enabled = True
    kg_engine.read = ACLAwareReadSubsystem(kg_engine, kg_engine.raw_read)
    kg_engine.write = ACLAwareWriteSubsystem(kg_engine, kg_engine.raw_write)

    node = _mk_node(
        node_id=f"node-acl-fail-{backend_kind}",
        doc_id=f"doc::fail::{backend_kind}",
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("acl write boom")

    monkeypatch.setattr(kg_engine.acl, "record_default_acl_for_item", _boom)

    token_write = claims_ctx.set(
        {"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        with pytest.raises(RuntimeError, match="acl write boom"):
            kg_engine.write.add_node(node)
    finally:
        claims_ctx.reset(token_write)

    truth_nodes = kg_engine.raw_read.get_nodes(node_type=Node, ids=[node.id])
    acl_rows = kg_engine.raw_read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_target_entity_id": node.id},
            ]
        },
    )
    entity_events = list(
        kg_engine.meta_sqlite.iter_entity_events(namespace=kg_engine.namespace)
    )
    acl_events = []
    for _seq, entity_kind, _entity_id, op, payload_json in entity_events:
        if entity_kind != "node" or op != "ADD":
            continue
        payload = json.loads(payload_json)
        metadata = payload.get("metadata") or {}
        if (
            metadata.get("entity_type") == "acl_record"
            and metadata.get("acl_target_entity_id") == node.id
        ):
            acl_events.append(payload)

    if backend_kind == "pg":
        assert truth_nodes == []
        assert acl_rows == []
        assert acl_events == []
        repair = kg_engine.repair_acl_records_from_events(limit=256)
        assert repair["repaired_acl_records"] == 0
        return

    assert [row.id for row in truth_nodes] == [node.id]
    assert acl_rows == []
    assert sorted(event["metadata"]["acl_target_grain"] for event in acl_events) == [
        "grounding",
        "node",
        "span",
    ]

    token_read = claims_ctx.set(
        {"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        assert [row.id for row in kg_engine.read.get_nodes(ids=[node.id])] == [node.id]
        repaired_acl_rows = kg_engine.raw_read.get_nodes(
            node_type=Node,
            where={
                "$and": [
                    {"entity_type": "acl_record"},
                    {"acl_target_entity_id": node.id},
                ]
            },
        )
        assert sorted(row.metadata["acl_target_grain"] for row in repaired_acl_rows) == [
            "grounding",
            "node",
            "span",
        ]
    finally:
        claims_ctx.reset(token_read)


def test_acl_startup_bounded_event_repair_rehydrates_acl_rows(tmp_path, monkeypatch):
    kg_engine, _conversation_engine = _make_engine_pair(
        backend_kind="chroma",
        tmp_path=tmp_path,
        sa_engine=None,
        pg_schema=None,
        dim=3,
        embedding_kind="constant",
        acl_enabled=True,
        acl_startup_repair_limit=0,
    )

    node = _mk_node(node_id="node-acl-startup-repair", doc_id="doc::startup::repair")

    def _boom(*args, **kwargs):
        raise RuntimeError("acl write boom")

    monkeypatch.setattr(kg_engine.acl, "record_default_acl_for_item", _boom)

    token_write = claims_ctx.set(
        {"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        with pytest.raises(RuntimeError, match="acl write boom"):
            kg_engine.write.add_node(node)
    finally:
        claims_ctx.reset(token_write)

    assert kg_engine.raw_read.get_nodes(node_type=Node, ids=[node.id])
    assert (
        kg_engine.raw_read.get_nodes(
            node_type=Node,
            where={
                "$and": [
                    {"entity_type": "acl_record"},
                    {"acl_target_entity_id": node.id},
                ]
            },
        )
        == []
    )

    kg_engine2, _conversation_engine2 = _make_engine_pair(
        backend_kind="chroma",
        tmp_path=tmp_path,
        sa_engine=None,
        pg_schema=None,
        dim=3,
        embedding_kind="constant",
        acl_enabled=True,
        acl_startup_repair_limit=256,
    )

    repaired_acl_rows = kg_engine2.raw_read.get_nodes(
        node_type=Node,
        where={
            "$and": [
                {"entity_type": "acl_record"},
                {"acl_target_entity_id": node.id},
            ]
        },
    )
    assert sorted(row.metadata["acl_target_grain"] for row in repaired_acl_rows) == [
        "grounding",
        "node",
        "span",
    ]

    token_read = claims_ctx.set(
        {"agent_id": "agent-a", "security_scope": "tenant-a", "ns": "knowledge"}
    )
    try:
        assert [row.id for row in kg_engine2.read.get_nodes(ids=[node.id])] == [node.id]
    finally:
        claims_ctx.reset(token_read)
