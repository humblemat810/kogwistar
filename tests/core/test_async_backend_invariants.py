from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.models import Edge, Node
from kogwistar.runtime.models import WorkflowRunNode, WorkflowRuntimeEdge
from kogwistar.server.run_registry import RunRegistry
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests._kg_factories import kg_document
from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    make_real_async_chroma_uow,
    real_chroma_server,  # noqa: F401
)

pytestmark = pytest.mark.ci_full


class _AsyncEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "async-backend-invariant"

    async def __call__(self, documents_or_texts):
        await asyncio.sleep(0)
        vectors: list[list[float]] = []
        for text in documents_or_texts:
            base = float(len(str(text)) % 7 + 1)
            vectors.append([base, base + 1.0, base + 2.0])
        return vectors

def _mk_node(*, node_id: str, doc_id: str, summary: str) -> Node:
    return build_entity_node(
        node_id=node_id,
        doc_id=doc_id,
        label=node_id,
        summary=summary,
        entity_type="concept",
        embedding=None,
        properties=None,
    )


async def _exercise_node_round_trip_case(
    *,
    backend_kind: str,
    backend,
    uow,
    eng: GraphKnowledgeEngine,
) -> Node:
    if backend_kind == "chroma":
        doc = kg_document(
            doc_id="doc::async-chroma-contract",
            content="alpha beta gamma",
            source="async-contract",
        )
        node = _mk_node(
            node_id="node::async-chroma-contract",
            doc_id=doc.id,
            summary="async chroma contract node",
        )
    elif backend_kind == "pg":
        doc = kg_document(
            doc_id="doc::async-pg-contract",
            content="delta epsilon zeta",
            source="async-contract",
        )
        node = _mk_node(
            node_id="node::async-pg-contract",
            doc_id=doc.id,
            summary="async pg contract node",
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported backend kind: {backend_kind}")

    async with uow.transaction():
        await asyncio.to_thread(eng.write.add_node, node, doc_id=doc.id)

    assert (await backend.node_get(ids=[node.id]))["ids"] == [node.id]
    return node


@pytest.mark.asyncio
async def test_async_chroma_backend_round_trip(real_chroma_server): # noqa: F811
    pytest.importorskip("chromadb")
    backend_client, backend, _collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="async_contract_chroma"
    )
    _ = backend_client
    uow = make_real_async_chroma_uow()
    eng = GraphKnowledgeEngine(
        persist_directory=str(real_chroma_server.persist_dir),
        embedding_function=_AsyncEmbeddingFunction(),
        backend_factory=lambda _engine: backend,
    )
    eng._phase1_enable_index_jobs = False

    await _exercise_node_round_trip_case(
        backend_kind="chroma",
        backend=backend,
        uow=uow,
        eng=eng,
    )


@pytest.mark.asyncio
async def test_async_pg_backend_invariants(async_pg_backend, async_pg_uow, tmp_path):
    pytest.importorskip("sqlalchemy")

    backend = async_pg_backend
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_pg_contract"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=backend,
    )
    eng._phase1_enable_index_jobs = False
    assert isinstance(eng.meta_sqlite, EnginePostgresMetaStore)
    registry = RunRegistry(eng.meta_sqlite)

    await _exercise_node_round_trip_case(
        backend_kind="pg",
        backend=backend,
        uow=async_pg_uow,
        eng=eng,
    )

    run_id = f"run::async-pg-contract::{uuid.uuid4().hex}"
    async with async_pg_uow.transaction():
        created = await asyncio.to_thread(
            registry.create_run,
            run_id=run_id,
            conversation_id="conv::async-pg-contract",
            workflow_id="wf::async-pg-contract",
            user_id="alice",
            user_turn_node_id="turn::async-pg-contract",
            status="queued",
        )
        assert created["run_id"] == run_id
        await asyncio.to_thread(
            registry.append_event, run_id, "run.created", {"run_id": run_id}
        )
        await asyncio.to_thread(
            registry.update_status, run_id, status="running", started=True
        )

    committed = await asyncio.to_thread(registry.get_run, run_id)
    assert committed is not None
    assert committed["status"] == "running"
    events = await asyncio.to_thread(registry.list_events, run_id)
    assert [evt["event_type"] for evt in events] == ["run.created"]

    rollback_doc = kg_document(
        doc_id="doc::async-pg-rollback",
        content="rollback alpha beta",
        source="async-contract",
    )
    rollback_node = _mk_node(
        node_id="node::async-pg-rollback",
        doc_id=rollback_doc.id,
        summary="async pg rollback node",
    )
    rollback_run_id = f"run::async-pg-rollback::{uuid.uuid4().hex}"

    with pytest.raises(RuntimeError):
        async with async_pg_uow.transaction():
            await asyncio.to_thread(eng.write.add_node, rollback_node, doc_id=rollback_doc.id)
            await asyncio.to_thread(
                registry.create_run,
                run_id=rollback_run_id,
                conversation_id="conv::async-pg-rollback",
                workflow_id="wf::async-pg-rollback",
                user_id="alice",
                user_turn_node_id="turn::async-pg-rollback",
                status="queued",
            )
            await asyncio.to_thread(
                registry.append_event,
                rollback_run_id,
                "run.created",
                {"run_id": rollback_run_id},
            )
            await asyncio.to_thread(
                registry.update_status,
                rollback_run_id,
                status="running",
                started=True,
            )
            raise RuntimeError("boom")

    assert (await backend.node_get(ids=[rollback_node.id]))["ids"] == []
    assert await asyncio.to_thread(registry.get_run, rollback_run_id) is None
    assert await asyncio.to_thread(registry.list_events, rollback_run_id) == []


@pytest.mark.asyncio
async def test_async_pg_rust_authority_writes_event_and_projection(
    async_pg_backend, async_pg_schema, async_pg_uow, tmp_path, monkeypatch
):
    """Async writes use the Rust event/projection authority as one operation."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("kogwistar._rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_pg_rust_contract"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=async_pg_backend,
    )
    assert engine.meta_sqlite.__class__.__name__ == "RustEnginePostgresMetaStore"
    meta = engine.meta_sqlite
    original_enqueue = meta.enqueue_index_job
    enqueue_transaction_states: list[bool] = []

    def capture_enqueue(**values):
        enqueue_transaction_states.append(meta.transaction_active)
        return original_enqueue(**values)

    monkeypatch.setattr(meta, "enqueue_index_job", capture_enqueue)

    doc = kg_document(
        doc_id="doc::async-pg-rust-contract",
        content="rust authority alpha beta",
        source="async-rust-contract",
    )
    node = _mk_node(
        node_id="node::async-pg-rust-contract",
        doc_id=doc.id,
        summary="async Rust authority node",
    )
    target = _mk_node(
        node_id="node::async-pg-rust-contract-target",
        doc_id=doc.id,
        summary="async Rust authority edge target",
    )
    await engine.write.add_node_async(node, doc_id=doc.id)
    await engine.write.add_node_async(target, doc_id=doc.id)

    edge: Edge = build_relationship_edge(
        edge_id="edge::async-pg-rust-contract",
        src=node.id,
        tgt=target.id,
        doc_id=doc.id,
        summary="async Rust authority edge",
    )
    await engine.write.add_edge_async(edge, doc_id=doc.id)

    assert (
        await async_pg_backend.node_get(ids=[node.id, target.id])
    )["ids"] == [node.id, target.id]
    assert (await async_pg_backend.edge_get(ids=[edge.id]))["ids"] == [edge.id]
    events = list(
        engine.meta_sqlite.iter_entity_events(namespace=engine.namespace)
    )
    assert [(kind, entity_id, op) for _, kind, entity_id, op, _ in events] == [
        ("node", node.id, "ADD"),
        ("node", target.id, "ADD"),
        ("edge", edge.id, "ADD"),
    ]
    assert enqueue_transaction_states
    assert all(enqueue_transaction_states)


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_kind", ["node", "edge"])
async def test_async_rust_authority_rolls_back_graph_write_if_index_enqueue_fails(
    entity_kind, async_pg_backend, tmp_path, monkeypatch
):
    pytest.importorskip("kogwistar._rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / f"async_pg_index_enqueue_failure_{entity_kind}"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=async_pg_backend,
    )
    source = _mk_node(
        node_id="node::async-pg-index-enqueue-source",
        doc_id="doc::async-pg-index-enqueue",
        summary="edge endpoint",
    )
    target = _mk_node(
        node_id="node::async-pg-index-enqueue-target",
        doc_id="doc::async-pg-index-enqueue",
        summary="edge endpoint",
    )
    if entity_kind == "edge":
        await engine.write.add_node_async(source)
        await engine.write.add_node_async(target)
        entity = build_relationship_edge(
            edge_id="edge::async-pg-index-enqueue-failure",
            src=source.id,
            tgt=target.id,
            doc_id="doc::async-pg-index-enqueue",
        )
    else:
        entity = _mk_node(
            node_id="node::async-pg-index-enqueue-failure",
            doc_id="doc::async-pg-index-enqueue",
            summary="must roll back with index intent",
        )

    def fail_enqueue(**_values):
        raise RuntimeError("injected index enqueue failure")

    monkeypatch.setattr(engine.meta_sqlite, "enqueue_index_job", fail_enqueue)
    write = engine.write.add_node_async if entity_kind == "node" else engine.write.add_edge_async
    with pytest.raises(RuntimeError, match="injected index enqueue failure"):
        await write(entity)

    entity_events = [
        event
        for event in engine.meta_sqlite.iter_entity_events(namespace=engine.namespace)
        if event[2] == entity.id
    ]
    assert entity_events == []
    assert all(
        job.entity_id != entity.id
        for job in engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
    )
    if entity_kind == "node":
        assert (await async_pg_backend.node_get(ids=[entity.id]))["ids"] == []
    else:
        assert (await async_pg_backend.edge_get(ids=[entity.id]))["ids"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_kind", ["node", "edge"])
async def test_async_rust_authority_rejects_inline_index_mode_before_graph_effect(
    entity_kind, async_pg_backend, tmp_path, monkeypatch
):
    pytest.importorskip("kogwistar._rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / f"async_pg_rust_inline_disabled_{entity_kind}"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=async_pg_backend,
    )
    engine._phase1_enable_index_jobs = False
    if entity_kind == "node":
        entity = _mk_node(
            node_id="node::async-pg-inline-disabled",
            doc_id="doc::async-pg-inline-disabled",
            summary="must not be committed",
        )
    else:
        entity = build_relationship_edge(
            edge_id="edge::async-pg-inline-disabled",
            src="missing-source",
            tgt="missing-target",
            doc_id="doc::async-pg-inline-disabled",
        )

    write = engine.write.add_node_async if entity_kind == "node" else engine.write.add_edge_async
    with pytest.raises(RuntimeError, match="requires durable index jobs"):
        await write(entity)

    assert list(engine.meta_sqlite.iter_entity_events(namespace=engine.namespace)) == []
    if entity_kind == "node":
        assert (await async_pg_backend.node_get(ids=[entity.id]))["ids"] == []
    else:
        assert (await async_pg_backend.edge_get(ids=[entity.id]))["ids"] == []


@pytest.mark.asyncio
async def test_async_pg_rust_authority_round_trips_derived_workflow_models(
    async_pg_backend, tmp_path, monkeypatch
):
    """Derived Pydantic models cross Python/Rust writes and reads intact."""
    pytest.importorskip("kogwistar._rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_pg_rust_derived_contract"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=async_pg_backend,
    )

    run_base = _mk_node(
        node_id="node::async-pg-derived-run",
        doc_id="doc::async-pg-derived",
        summary="derived workflow run",
    )
    run_node = WorkflowRunNode(
        id=run_base.id,
        label=run_base.label,
        type=run_base.type,
        summary=run_base.summary,
        doc_id=run_base.doc_id,
        mentions=run_base.mentions,
        metadata={
            "entity_type": "workflow_run",
            "workflow_id": "wf::derived",
            "workflow_version": "v1",
            "run_id": "run::derived",
            "conversation_id": "conversation::derived",
            "turn_node_id": "turn::derived",
            "status": "running",
        },
    )
    target = _mk_node(
        node_id="node::async-pg-derived-target",
        doc_id="doc::async-pg-derived",
        summary="derived workflow edge target",
    )
    runtime_edge = WorkflowRuntimeEdge(
        id="edge::async-pg-derived-runtime",
        label="runtime transition",
        type="relationship",
        summary="workflow run links to target",
        relation="next",
        source_ids=[run_node.id],
        target_ids=[target.id],
        source_edge_ids=None,
        target_edge_ids=None,
        doc_id="doc::async-pg-derived",
        mentions=run_base.mentions,
        metadata={
            "relation": "next",
            "conversation_id": "conversation::derived",
            "run_id": "run::derived",
            "entity_type": "conversation_edge",
        },
    )

    meta = engine.meta_sqlite
    original_apply = meta.apply_graph_mutation
    envelopes: list[dict[str, object]] = []

    def capture_envelope(**values):
        envelope = values.get("model_envelope")
        assert isinstance(envelope, dict)
        envelopes.append(envelope)
        return original_apply(**values)

    monkeypatch.setattr(meta, "apply_graph_mutation", capture_envelope)
    await engine.write.add_node_async(run_node)
    await engine.write.add_node_async(target)
    await engine.write.add_edge_async(runtime_edge)

    assert [envelope["model_class"] for envelope in envelopes] == [
        "workflow_run_node",
        "node",
        "workflow_runtime_edge",
    ]
    events = list(meta.iter_entity_events(namespace=engine.namespace))
    event_payloads = {
        entity_id: json.loads(payload_json)
        for _, _, entity_id, _, payload_json in events
    }
    assert type(WorkflowRunNode.model_validate(event_payloads[run_node.id])) is WorkflowRunNode
    assert (
        type(WorkflowRuntimeEdge.model_validate(event_payloads[runtime_edge.id]))
        is WorkflowRuntimeEdge
    )

    node_rows = await async_pg_backend.node_get(
        ids=[run_node.id], include=["documents", "metadatas"]
    )
    edge_rows = await async_pg_backend.edge_get(
        ids=[runtime_edge.id], include=["documents", "metadatas"]
    )
    assert node_rows["ids"] == [run_node.id]
    assert node_rows["metadatas"][0]["_class_name"] == "WorkflowRunNode"
    assert edge_rows["ids"] == [runtime_edge.id]
    assert edge_rows["metadatas"][0]["relation"] == "next"
    assert edge_rows["metadatas"][0]["node_endpoint_count"] == 2
    assert edge_rows["metadatas"][0]["edge_endpoint_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_kind", ["node", "edge"])
async def test_async_pg_rust_authority_rolls_back_event_and_projection_together(
    entity_kind, async_pg_backend, tmp_path, monkeypatch
):
    """A failure after Rust mutation must roll back the outer async transaction."""
    pytest.importorskip("kogwistar._rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / f"async_pg_rust_rollback_{entity_kind}"),
        embedding_function=_AsyncEmbeddingFunction(),
        backend=async_pg_backend,
    )
    node = _mk_node(
        node_id=f"node::async-pg-rollback-{entity_kind}",
        doc_id="doc::async-pg-rollback",
        summary="transaction rollback source",
    )
    target = _mk_node(
        node_id=f"node::async-pg-rollback-{entity_kind}-target",
        doc_id="doc::async-pg-rollback",
        summary="transaction rollback target",
    )
    try:
        if entity_kind == "edge":
            await engine.write.add_node_async(node)
            await engine.write.add_node_async(target)
            entity = build_relationship_edge(
                edge_id="edge::async-pg-rollback",
                src=node.id,
                tgt=target.id,
                doc_id="doc::async-pg-rollback",
                embedding=[0.1, 0.2, 0.3],
            )
        else:
            entity = node

        meta = engine.meta_sqlite
        original_apply = meta.apply_graph_mutation

        def apply_then_fail(**values):
            result = original_apply(**values)
            if values.get("entity_kind") == entity_kind:
                raise RuntimeError("injected failure after Rust graph mutation")
            return result

        monkeypatch.setattr(meta, "apply_graph_mutation", apply_then_fail)
        write = (
            engine.write.add_edge_async
            if entity_kind == "edge"
            else engine.write.add_node_async
        )
        with pytest.raises(RuntimeError, match="after Rust graph mutation"):
            await write(entity)

        ids = (
            (await async_pg_backend.edge_get(ids=[entity.id]))["ids"]
            if entity_kind == "edge"
            else (await async_pg_backend.node_get(ids=[entity.id]))["ids"]
        )
        assert ids == []
        events = list(meta.iter_entity_events(namespace=engine.namespace))
        assert not any(event[2] == entity.id for event in events)
    finally:
        engine.close()
