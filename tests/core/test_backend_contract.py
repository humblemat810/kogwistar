from __future__ import annotations

import pytest

from kogwistar.conversation.models import ConversationEdge, ConversationNode
from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.server.run_registry import RunRegistry
from tests.conftest import _make_engine_pair

pytestmark = [pytest.mark.core, pytest.mark.ci_full]


BACKEND_PARAMS = [
    pytest.param("fake", id="fake", marks=pytest.mark.ci_full),
    pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
    pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
]


def _mk_span(doc_id: str) -> Span:
    span = Span.from_dummy_for_conversation()
    span.doc_id = doc_id
    return span


def _mk_node(
    *,
    node_id: str,
    doc_id: str,
    entity_type: str,
    embedding: list[float],
) -> Node:
    return Node(
        id=node_id,
        label=node_id,
        type="entity",
        summary=node_id,
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"entity_type": entity_type, "level_from_root": 0},
        embedding=embedding,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
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
    return Edge(
        id=edge_id,
        label=edge_id,
        type="relationship",
        summary=edge_id,
        relation=relation,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"entity_type": "relationship", "level_from_root": 0},
        embedding=[0.0, 0.0, 1.0],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _mk_turn(
    *,
    turn_id: str,
    conversation_id: str,
    user_id: str,
    role: str,
    turn_index: int,
) -> ConversationNode:
    doc_id = f"conv:{conversation_id}"
    return ConversationNode(
        id=turn_id,
        label=turn_id,
        type="entity",
        summary=f"{role}:{turn_index}",
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
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
        id=None,
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
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
        id=None,
        label="depends_on",
        type="relationship",
        summary="Dependency",
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
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
    backend_kind: str, tmp_path, sa_engine, pg_schema
) -> tuple[object, object]:
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


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_collection_crud_and_where(
    backend_kind: str, tmp_path, sa_engine, pg_schema
):
    kg_engine, _conversation_engine = _make_pair(
        backend_kind, tmp_path, sa_engine, pg_schema
    )

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


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_edge_endpoint_materialization(
    backend_kind: str, tmp_path, sa_engine, pg_schema
):
    kg_engine, _conversation_engine = _make_pair(
        backend_kind, tmp_path, sa_engine, pg_schema
    )

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
    kg_engine.add_node(n1)
    kg_engine.add_node(n2)
    e1 = _mk_edge(edge_id="e1", src="n1", tgt="n2", doc_id=doc_id)
    kg_engine.add_edge(e1)

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
    backend_kind: str, tmp_path, sa_engine, pg_schema
):
    _kg_engine, conversation_engine = _make_pair(
        backend_kind, tmp_path, sa_engine, pg_schema
    )
    meta = conversation_engine.meta_sqlite

    assert meta.current_user_seq("user-a") == 0
    assert meta.next_user_seq("user-a") == 1
    assert meta.current_user_seq("user-a") == 1
    assert meta.next_global_seq() == 1
    assert meta.current_global_seq() == 1

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


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_backend_contract_conversation_phase1_rules(
    backend_kind: str, tmp_path, sa_engine, pg_schema
):
    _kg_engine, conversation_engine = _make_pair(
        backend_kind, tmp_path, sa_engine, pg_schema
    )
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
    conversation_engine.add_node(t1)
    conversation_engine.add_node(t2)
    conversation_engine.add_node(t3)

    seeded = _mk_next_turn_edge(
        conversation_id=conversation_id, src=t1.id, tgt=t2.id
    )
    conversation_engine.add_edge(seeded)
    dup_rows = conversation_engine.backend.edge_get(where={"relation": "next_turn"})
    assert len(dup_rows["ids"]) == 1
    # Phase-1 semantics: an identical add_edge(next_turn) is idempotent, not an error.
    conversation_engine.add_edge(
        _mk_next_turn_edge(conversation_id=conversation_id, src=t1.id, tgt=t2.id)
    )
    dup_rows_after = conversation_engine.backend.edge_get(where={"relation": "next_turn"})
    assert len(dup_rows_after["ids"]) == 1
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_dependency_edge(conversation_id=conversation_id, src=t3.id, tgt=t1.id)
        )
