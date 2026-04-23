from __future__ import annotations

import asyncio
import uuid

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.models import Node
from kogwistar.server.run_registry import RunRegistry
from tests._kg_factories import kg_document
from tests._helpers.graph_builders import build_entity_node
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
