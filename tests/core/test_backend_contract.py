from __future__ import annotations

import pytest

from kogwistar.conversation.models import ConversationEdge, ConversationNode, ConversationRole
from kogwistar.engine_core.models import Edge, Grounding, Node
from kogwistar.id_provider import stable_id
from kogwistar.server.run_registry import RunRegistry
from tests._helpers.graph_builders import (
    build_entity_node,
    build_relationship_edge,
    mk_conversation_grounding,
)
from tests.conftest import _make_engine_pair
from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    make_real_async_chroma_uow,
    real_chroma_server,  # noqa: F401
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
pytestmark = [pytest.mark.core]


BACKEND_PARAMS = [
    pytest.param("fake", id="fake", marks=pytest.mark.ci),
    pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
    pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
]

def _mk_node(
    *,
    node_id: str,
    doc_id: str,
    entity_type: str,
    embedding: list[float],
) -> Node:
    return build_entity_node(
        node_id=node_id,
        doc_id=doc_id,
        label=node_id,
        summary=node_id,
        entity_type=entity_type,
        embedding=embedding,
        properties=None,
    )


def _mk_edge(
    *,
    edge_id: str,
    src: str,
    tgt: str,
    doc_id: str,
    relation: str = "related_to",
) -> Edge:
    return build_relationship_edge(
        edge_id=edge_id,
        src=src,
        tgt=tgt,
        doc_id=doc_id,
        label=edge_id,
        summary=edge_id,
        relation=relation,
        entity_type="relationship",
        embedding=[0.0, 0.0, 1.0],
        properties=None,
        source_edge_ids=[],
        target_edge_ids=[],
    )

def _mk_turn(
    *,
    turn_id: str,
    conversation_id: str,
    user_id: str,
    role: ConversationRole,
    turn_index: int,
) -> ConversationNode:
    doc_id = f"conv:{conversation_id}"
    return ConversationNode(
        id=turn_id,
        label=turn_id,
        type="entity",
        summary=f"{role}:{turn_index}",
        mentions=[mk_conversation_grounding(doc_id)],
        doc_id=doc_id,
        metadata={
            "entity_type": f"{role}_turn",
            "in_conversation_chain": True,
            "level_from_root": 0,
            "role": role,
            "turn_index": turn_index,
            "conversation_id": conversation_id,
            "user_id": user_id,
        },
        embedding=None,
        level_from_root=0,
        role=role,
        turn_index=turn_index,
        conversation_id=conversation_id,
        user_id=user_id,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _mk_next_turn_edge(
    *, conversation_id: str, src: str, tgt: str
) -> ConversationEdge:
    doc_id = f"conv:{conversation_id}"
    return ConversationEdge(
        id=str(
            stable_id("conversation.edge", conversation_id, "next_turn", src, tgt)
        ),
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        mentions=[mk_conversation_grounding(doc_id)],
        doc_id=doc_id,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="next_turn",
        metadata={"causal_type": "chain", "conversation_id": conversation_id},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )


def _mk_dependency_edge(
    *, conversation_id: str, src: str, tgt: str
) -> ConversationEdge:
    doc_id = f"conv:{conversation_id}"
    return ConversationEdge(
        id=str(
            stable_id("conversation.edge", conversation_id, "depends_on", src, tgt)
        ),
        label="depends_on",
        type="relationship",
        summary="Dependency",
        mentions=[mk_conversation_grounding(doc_id)],
        doc_id=doc_id,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="depends_on",
        metadata={"causal_type": "dependency", "conversation_id": conversation_id},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )


def _make_pair(
    backend_kind: str, tmp_path, request
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    sa_engine = None
    pg_schema = None
    if backend_kind == "pg":
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
    kg_engine, conversation_engine = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=3,
        embedding_kind="constant",
    )
    return kg_engine, conversation_engine


def _backend_meta_for_node(node: Node) -> dict:
    return {
        "doc_id": node.doc_id,
        "label": node.label,
        "type": node.type,
        "summary": node.summary,
        "entity_type": node.metadata.get("entity_type"),
        "level_from_root": node.metadata.get("level_from_root", 0),
    }


async def _assert_collection_crud_and_where_async(
    *,
    backend,
    uow,
    backend_kind: str,
) -> None:
    doc_id = f"doc::{backend_kind}"
    n1 = _mk_node(
        node_id="n1",
        doc_id=doc_id,
        entity_type="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    n2 = _mk_node(
        node_id="n2",
        doc_id=doc_id,
        entity_type="beta",
        embedding=[0.0, 1.0, 0.0],
    )
    async with uow.transaction():
        await backend.node_add(
            ids=[n1.id, n2.id],
            documents=[n1.summary, n2.summary],
            metadatas=[_backend_meta_for_node(n1), _backend_meta_for_node(n2)],
            embeddings=[n1.embedding, n2.embedding],
        )

    got = await backend.node_get(where={"doc_id": doc_id})
    assert set(got["ids"]) == {"n1", "n2"}

    got_and = await backend.node_get(
        where={"$and": [{"doc_id": doc_id}, {"entity_type": "alpha"}]}
    )
    assert got_and["ids"] == ["n1"]

    got_or = await backend.node_get(
        where={"$or": [{"doc_id": "missing"}, {"doc_id": doc_id}]}
    )
    assert set(got_or["ids"]) == {"n1", "n2"}

    q = await backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=1,
        where={"doc_id": doc_id},
        include=["metadatas"],
    )
    assert q["ids"][0][0] == "n1"
    assert q["metadatas"][0][0]["entity_type"] == "alpha"

    e1 = _mk_edge(edge_id="e1", src="n1", tgt="n2", doc_id=doc_id)
    async with uow.transaction():
        await backend.edge_add(
            ids=[e1.id],
            documents=[e1.summary],
            metadatas=[e1.metadata | {"relation": e1.relation, "doc_id": doc_id}],
            embeddings=[e1.embedding],
        )
    got_edge = await backend.edge_get(where={"relation": "related_to"})
    assert got_edge["ids"] == ["e1"]

    q_edge = await backend.edge_query(
        query_embeddings=[[0.0, 0.0, 1.0]],
        n_results=1,
        where={"relation": "related_to"},
        include=["metadatas"],
    )
    assert q_edge["ids"][0][0] == "e1"


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_collection_crud_and_where(
    backend_kind: str, tmp_path, request
):
    kg_engine: GraphKnowledgeEngine
    kg_engine, _conversation_engine = _make_pair(backend_kind, tmp_path, request)

    doc_id = f"doc::{backend_kind}"
    n1 = _mk_node(
        node_id="n1",
        doc_id=doc_id,
        entity_type="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    n2 = _mk_node(
        node_id="n2",
        doc_id=doc_id,
        entity_type="beta",
        embedding=[0.0, 1.0, 0.0],
    )
    kg_engine.backend.node_add(
        ids=[n1.id, n2.id],
        documents=[n1.summary, n2.summary],
        metadatas=[_backend_meta_for_node(n1), _backend_meta_for_node(n2)],
        embeddings=[n1.embedding, n2.embedding],
    )

    got = kg_engine.backend.node_get(where={"doc_id": doc_id})
    assert set(got["ids"]) == {"n1", "n2"}

    got_and = kg_engine.backend.node_get(
        where={"$and": [{"doc_id": doc_id}, {"entity_type": "alpha"}]}
    )
    assert got_and["ids"] == ["n1"]

    got_or = kg_engine.backend.node_get(
        where={"$or": [{"doc_id": "missing"}, {"doc_id": doc_id}]}
    )
    assert set(got_or["ids"]) == {"n1", "n2"}

    q = kg_engine.backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=1,
        where={"doc_id": doc_id},
        include=["metadatas"],
    )
    assert q["ids"][0][0] == "n1"
    assert q["metadatas"][0][0]["entity_type"] == "alpha"

    e1 = _mk_edge(edge_id="e1", src="n1", tgt="n2", doc_id=doc_id)
    kg_engine.backend.edge_add(
        ids=[e1.id],
        documents=[e1.summary],
        metadatas=[e1.metadata | {"relation": e1.relation, "doc_id": doc_id}],
        embeddings=[e1.embedding],
    )
    got_edge = kg_engine.backend.edge_get(where={"relation": "related_to"})
    assert got_edge["ids"] == ["e1"]

    q_edge = kg_engine.backend.edge_query(
        query_embeddings=[[0.0, 0.0, 1.0]],
        n_results=1,
        where={"relation": "related_to"},
        include=["metadatas"],
    )
    assert q_edge["ids"][0][0] == "e1"


@pytest.mark.asyncio
async def test_backend_contract_collection_crud_and_where_async_chroma(
    real_chroma_server,
):
    pytest.importorskip("chromadb")
    _backend_client, backend, _collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="contract_async_chroma"
    )
    uow = make_real_async_chroma_uow()
    await _assert_collection_crud_and_where_async(
        backend=backend, uow=uow, backend_kind="async-chroma"
    )


@pytest.mark.asyncio
async def test_backend_contract_collection_crud_and_where_async_pg(
    async_pg_backend, async_pg_uow
):
    pytest.importorskip("sqlalchemy")
    await _assert_collection_crud_and_where_async(
        backend=async_pg_backend, uow=async_pg_uow, backend_kind="async-pg"
    )


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_edge_endpoint_materialization(
    backend_kind: str, tmp_path, request
):
    kg_engine, _conversation_engine = _make_pair(backend_kind, tmp_path, request)

    doc_id = f"doc::{backend_kind}::edges"
    n1 = _mk_node(
        node_id="n1",
        doc_id=doc_id,
        entity_type="alpha",
        embedding=[1.0, 0.0, 0.0],
    )
    n2 = _mk_node(
        node_id="n2",
        doc_id=doc_id,
        entity_type="beta",
        embedding=[0.0, 1.0, 0.0],
    )
    kg_engine.write.add_node(n1)
    kg_engine.write.add_node(n2)
    e1 = _mk_edge(edge_id="e1", src="n1", tgt="n2", doc_id=doc_id)
    kg_engine.write.add_edge(e1)

    endpoints = kg_engine.backend.edge_endpoints_get(
        where={"$and": [{"edge_id": "e1"}, {"doc_id": doc_id}]},
        include=["metadatas"],
        limit=50,
    )
    assert len(endpoints["ids"]) >= 2
    assert any(m.get("edge_id") == "e1" for m in endpoints["metadatas"])
    assert {m.get("endpoint_type") for m in endpoints["metadatas"]} >= {"node"}
    assert {m.get("role") for m in endpoints["metadatas"]} >= {"src", "tgt"}


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_meta_store_and_run_registry(
    backend_kind: str, tmp_path, request
):
    _kg_engine, conversation_engine = _make_pair(backend_kind, tmp_path, request)
    meta = conversation_engine.meta_sqlite

    assert meta.current_user_seq("user-a") == 0
    assert meta.next_user_seq("user-a") == 1
    assert meta.current_user_seq("user-a") == 1
    assert meta.current_scoped_seq("scope-a") == 0
    assert meta.next_scoped_seq("scope-a") == 1
    assert meta.current_scoped_seq("scope-a") == 1
    assert meta.next_global_seq() == 1
    assert meta.current_global_seq() == 1

    assert meta.get_named_projection("bridge_governance", "interaction-1") is None
    meta.replace_named_projection(
        "bridge_governance",
        "interaction-1",
        {
            "active_agents": ["agent-a", "agent-b"],
            "latest_action": "handoff_requested",
        },
        last_authoritative_seq=7,
        last_materialized_seq=6,
        projection_schema_version=2,
        materialization_status="rebuilding",
    )
    projection = meta.get_named_projection("bridge_governance", "interaction-1")
    assert projection is not None
    assert projection["namespace"] == "bridge_governance"
    assert projection["key"] == "interaction-1"
    assert projection["payload"]["active_agents"] == ["agent-a", "agent-b"]
    assert projection["last_authoritative_seq"] == 7
    assert projection["last_materialized_seq"] == 6
    assert projection["projection_schema_version"] == 2
    assert projection["materialization_status"] == "rebuilding"
    assert isinstance(projection["updated_at_ms"], int)

    meta.replace_named_projection(
        "bridge_governance",
        "interaction-1",
        {
            "active_agents": ["agent-a", "agent-b"],
            "latest_action": "policy_approved",
        },
        last_authoritative_seq=8,
        last_materialized_seq=8,
        projection_schema_version=3,
        materialization_status="ready",
    )
    meta.replace_named_projection(
        "bridge_governance",
        "interaction-2",
        {"active_agents": ["agent-c"], "latest_action": "opened"},
        last_authoritative_seq=2,
        last_materialized_seq=2,
        projection_schema_version=1,
        materialization_status="ready",
    )
    listed = meta.list_named_projections("bridge_governance")
    assert [item["key"] for item in listed] == ["interaction-1", "interaction-2"]
    assert listed[0]["payload"]["latest_action"] == "policy_approved"
    assert listed[0]["last_materialized_seq"] == 8
    assert listed[0]["projection_schema_version"] == 3
    meta.clear_named_projection("bridge_governance", "interaction-2")
    assert meta.get_named_projection("bridge_governance", "interaction-2") is None
    meta.clear_projection_namespace("bridge_governance")
    assert meta.list_named_projections("bridge_governance") == []

    meta.replace_workflow_design_projection(
        workflow_id="wf-1",
        head={
            "current_version": 2,
            "active_tip_version": 3,
            "last_authoritative_seq": 11,
            "last_materialized_seq": 10,
            "projection_schema_version": 1,
            "snapshot_schema_version": 4,
            "materialization_status": "ready",
        },
        versions=[
            {"version": 0, "prev_version": 0, "target_seq": 0, "created_at_ms": 0},
            {"version": 2, "prev_version": 1, "target_seq": 8, "created_at_ms": 20},
        ],
        dropped_ranges=[
            {"start_seq": 9, "end_seq": 10, "start_version": 2, "end_version": 3}
        ],
    )
    workflow_projection = meta.get_workflow_design_projection(workflow_id="wf-1")
    assert workflow_projection is not None
    assert workflow_projection["current_version"] == 2
    assert workflow_projection["active_tip_version"] == 3
    assert workflow_projection["snapshot_schema_version"] == 4
    assert workflow_projection["materialization_status"] == "ready"
    assert [item["version"] for item in workflow_projection["versions"]] == [0, 2]
    workflow_named_projection = meta.get_named_projection("workflow_design", "wf-1")
    assert workflow_named_projection is not None
    assert workflow_named_projection["payload"]["current_version"] == 2
    assert workflow_named_projection["payload"]["snapshot_schema_version"] == 4
    assert workflow_named_projection["payload"]["versions"][1]["version"] == 2
    assert workflow_named_projection["payload"]["dropped_ranges"][0]["end_version"] == 3

    job_id = meta.enqueue_index_job(
        job_id="job-1",
        namespace="ns-a",
        entity_kind="node",
        entity_id="n-1",
        index_kind="node_docs",
        op="UPSERT",
    )
    assert job_id == "job-1"
    assert meta.list_index_jobs(namespace="ns-a", limit=10)[0].job_id == "job-1"

    claimed = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="ns-a")
    assert claimed and claimed[0].job_id == "job-1"
    meta.mark_index_job_done("job-1")
    assert meta.list_index_jobs(namespace="ns-a", status="DONE", limit=10)[0].job_id == "job-1"

    seq = meta.append_entity_event(
        namespace="ns-a",
        event_id="evt-1",
        entity_kind="node",
        entity_id="n-1",
        op="ADD",
        payload_json='{"hello": "world"}',
    )
    assert seq == 1
    assert list(meta.iter_entity_events(namespace="ns-a", from_seq=1))
    meta.cursor_set(namespace="ns-a", consumer="replay", last_seq=seq)
    assert meta.cursor_get(namespace="ns-a", consumer="replay") == seq

    meta.set_index_applied_fingerprint(
        namespace="ns-a",
        coalesce_key="node:n-1:node_docs",
        applied_fingerprint="fp-1",
        last_job_id="job-1",
    )
    assert (
        meta.get_index_applied_fingerprint(
            namespace="ns-a", coalesce_key="node:n-1:node_docs"
        )
        == "fp-1"
    )

    registry = RunRegistry(meta)
    run = registry.create_run(
        run_id="run-1",
        conversation_id="conv-1",
        workflow_id="wf-1",
        user_id="user-a",
        user_turn_node_id="turn-1",
    )
    assert run["run_id"] == "run-1"
    assert run["terminal"] is False
    registry.append_event("run-1", "created", {"step": 1})
    events = registry.list_events("run-1")
    assert events and events[0]["event_type"] == "created"
    updated = registry.update_status(
        "run-1",
        status="succeeded",
        result={"ok": True},
        started=True,
        finished=True,
    )
    assert updated["status"] == "succeeded"
    assert updated["terminal"] is True
    runs = meta.list_server_runs(limit=10)
    assert [row["run_id"] for row in runs] == ["run-1"]
    assert runs[0]["workflow_id"] == "wf-1"
    assert runs[0]["terminal"] is True
    meta.clear_workflow_design_projection(workflow_id="wf-1")
    assert meta.get_workflow_design_projection(workflow_id="wf-1") is None
    assert meta.get_named_projection("workflow_design", "wf-1") is None


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_conversation_phase1_rules(
    backend_kind: str, tmp_path, request
):
    _kg_engine, conversation_engine = _make_pair(backend_kind, tmp_path, request)
    conversation_engine._phase1_enable_index_jobs = False
    conversation_id = f"conv::{backend_kind}"
    user_id = "u-contract"
    t1 = _mk_turn(
        turn_id="turn-1",
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        turn_index=1,
    )
    t2 = _mk_turn(
        turn_id="turn-2",
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        turn_index=2,
    )
    t3 = _mk_turn(
        turn_id="turn-3",
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        turn_index=3,
    )
    conversation_engine.write.add_node(t1)
    conversation_engine.write.add_node(t2)
    conversation_engine.write.add_node(t3)

    seeded = _mk_next_turn_edge(
        conversation_id=conversation_id, src=t1.safe_get_id(), tgt=t2.safe_get_id()
    )
    conversation_engine.write.add_edge(seeded)
    dup_rows = conversation_engine.backend.edge_get(where={"relation": "next_turn"})
    assert len(dup_rows["ids"]) == 1
    # Phase-1 semantics: an identical add_edge(next_turn) is idempotent, not an error.
    conversation_engine.write.add_edge(
        _mk_next_turn_edge(conversation_id=conversation_id, src=t1.safe_get_id(), tgt=t2.safe_get_id())
    )
    dup_rows_after = conversation_engine.backend.edge_get(where={"relation": "next_turn"})
    assert len(dup_rows_after["ids"]) == 1
    with pytest.raises(ValueError):
        conversation_engine.write.add_edge(
            _mk_dependency_edge(conversation_id=conversation_id, src=t3.safe_get_id(), tgt=t1.safe_get_id())
        )
