from __future__ import annotations

import asyncio
import threading

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.chroma_backend import AsyncChromaBackend
from kogwistar.workers.async_index_job_worker import AsyncIndexJobWorker
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node
from tests._helpers.graph_builders import build_relationship_edge
from tests.core.two_stage_case_catalog import two_stage_case


pytestmark = [pytest.mark.ci, pytest.mark.runtime_async]


class _AsyncCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def upsert(self, *, ids, documents, metadatas, embeddings=None, **kwargs):
        del kwargs
        for index, entity_id in enumerate(ids):
            self.rows[str(entity_id)] = {
                "document": documents[index],
                "metadata": metadatas[index],
                "embedding": None if embeddings is None else embeddings[index],
            }

    async def add(self, *, ids, documents, metadatas, embeddings=None, **kwargs):
        await self.upsert(
            ids=ids, documents=documents, metadatas=metadatas,
            embeddings=embeddings, **kwargs
        )

    async def delete(self, *, ids=None, **kwargs):
        del kwargs
        for entity_id in ids or []:
            self.rows.pop(str(entity_id), None)

    async def get(self, *, ids=None, **kwargs):
        del kwargs
        selected = [str(entity_id) for entity_id in (ids or self.rows)]
        selected = [entity_id for entity_id in selected if entity_id in self.rows]
        return {
            "ids": selected,
            "documents": [self.rows[entity_id]["document"] for entity_id in selected],
            "metadatas": [self.rows[entity_id]["metadata"] for entity_id in selected],
            "embeddings": [self.rows[entity_id]["embedding"] for entity_id in selected],
        }


def _async_chroma_backend() -> AsyncChromaBackend:
    collections = [_AsyncCollection() for _ in range(9)]
    return AsyncChromaBackend(
        node_index_collection=collections[0], node_collection=collections[1],
        edge_collection=collections[2], edge_endpoints_collection=collections[3],
        document_collection=collections[4], domain_collection=collections[5],
        node_docs_collection=collections[6], node_refs_collection=collections[7],
        edge_refs_collection=collections[8],
    )


@pytest.mark.asyncio
async def test_async_single_stage_awaits_provider_and_writes_node_and_edge(tmp_path):
    """Single-stage async admission must not require a two-stage adapter."""
    main_thread = threading.get_ident()
    calls: list[tuple[int, int]] = []

    async def provider(texts):
        calls.append((len(texts), threading.get_ident()))
        return [[0.25, 0.75] for _ in texts]

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        embedding_function=provider,
    )
    try:
        node = build_entity_node(node_id="async-single-node", doc_id="d1")
        await engine.async_add_node(node)
        assert calls and calls[0][1] == main_thread
        calls.clear()
        edge = build_relationship_edge(
            edge_id="async-single-edge", src=node.id, tgt=node.id, doc_id="d1"
        )
        await engine.async_add_edge(edge)

        node_row = engine.backend.node_get(ids=[node.id], include=["embeddings"])
        edge_row = engine.backend.edge_get(ids=[edge.id], include=["embeddings"])
        assert node_row["ids"] == [node.id]
        assert edge_row["ids"] == [edge.id]
        assert node_row["embeddings"][0] == [0.25, 0.75]
        assert edge_row["embeddings"][0] == [0.25, 0.75]
        assert calls and calls[0][1] == main_thread
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_async_chroma_single_stage_awaits_provider_and_upserts_directly(tmp_path):
    """Async Chroma single-stage uses async collection verbs, not sync facade IO."""
    backend = _async_chroma_backend()
    main_thread = threading.get_ident()
    calls: list[int] = []
    call_threads: list[int] = []

    async def provider(texts):
        calls.append(len(texts))
        call_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return [[0.5, 0.5] for _ in texts]

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=lambda _engine: backend,
        embedding_function=provider,
    )
    try:
        node = build_entity_node(node_id="async-chroma-single", doc_id="d1")
        await engine.async_add_node(node)
        stored = await backend.async_node_get(
            ids=[node.id], include=["documents", "metadatas", "embeddings"]
        )
        assert stored["ids"] == [node.id]
        assert stored["embeddings"] == [[0.5, 0.5]]
        assert calls and calls[0] == 1
        assert call_threads[0] == main_thread
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_async_single_stage_moves_sync_provider_off_event_loop(tmp_path):
    main_thread = threading.get_ident()
    call_threads: list[int] = []

    def provider(texts):
        call_threads.append(threading.get_ident())
        return [[0.1, 0.9] for _ in texts]

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        embedding_function=provider,
    )
    try:
        await engine.async_add_node(
            build_entity_node(node_id="async-sync-provider", doc_id="d1")
        )
        assert call_threads
        assert call_threads[0] != main_thread
    finally:
        engine.close()


@pytest.mark.asyncio
@two_stage_case("pending_visibility")
@two_stage_case("promotion_handoff")
async def test_async_worker_promotes_in_memory_two_stage_job_without_sync_bridge(tmp_path):
    """Async mirror: existing sync worker promotion on the same staged row."""
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="async-n1", doc_id="d1"))
        worker = AsyncIndexJobWorker(
            engine=engine,
            batch_size=8,
            max_jobs_per_tick=8,
        )

        metrics = await worker.tick()

        assert metrics.claimed == 1
        assert metrics.done == 1
        assert engine.backend.node_get(ids=["async-n1"], include=["embeddings"])[
            "embeddings"
        ][0] is not None
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_async_worker_fails_closed_without_async_projection_adapter(tmp_path):
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="async-n2", doc_id="d1"))
        engine.async_two_stage_projection_adapter = None
        metrics = await engine.indexing.make_async_index_job_worker().tick()
        assert metrics.claimed == 1
        assert metrics.done == 0
        assert metrics.retried == 1
    finally:
        engine.close()


@pytest.mark.asyncio
@two_stage_case("batch_embedding")
async def test_async_worker_batches_compatible_embedding_jobs(tmp_path):
    calls: list[int] = []

    async def provider(texts):
        calls.append(len(texts))
        return [[0.25, 0.75] for _ in texts]

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        engine.write.add_node(build_entity_node(node_id="async-b1", doc_id="d1"))
        engine.write.add_node(build_entity_node(node_id="async-b2", doc_id="d1"))
        calls.clear()
        metrics = await engine.indexing.make_async_index_job_worker(
            batch_size=8, max_jobs_per_tick=8
        ).tick()

        assert metrics.claimed == 2
        assert metrics.done == 2
        assert calls == [2]
    finally:
        engine.close()


@pytest.mark.asyncio
@two_stage_case("pending_visibility")
@two_stage_case("promotion_handoff")
@two_stage_case("batch_embedding")
async def test_async_chroma_admission_and_promotion_use_async_adapter(tmp_path):
    backend = _async_chroma_backend()
    calls: list[int] = []

    async def provider(texts):
        calls.append(len(texts))
        return [[0.5, 0.5] for _ in texts]

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=lambda _engine: backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        nodes = [
            build_entity_node(node_id="async-chroma-n1", doc_id="d1", embedding=None),
            build_entity_node(node_id="async-chroma-n2", doc_id="d1", embedding=None),
        ]
        for node in nodes:
            await engine.async_add_node(node)
        assert await backend.async_node_get(ids=[node.id for node in nodes]) == {
            "ids": [], "documents": [], "metadatas": [], "embeddings": []
        }
        calls.clear()
        metrics = await engine.indexing.make_async_index_job_worker(
            batch_size=8, max_jobs_per_tick=8
        ).tick()
        assert metrics.done == 2
        assert calls == [2]
        promoted = await backend.async_node_get(ids=[node.id for node in nodes])
        assert promoted["ids"] == [node.id for node in nodes]
        assert all(embedding is not None for embedding in promoted["embeddings"])
    finally:
        engine.close()


@pytest.mark.asyncio
@two_stage_case("stale_revision")
@two_stage_case("delete_race")
@two_stage_case("recovery_reconciliation")
async def test_async_worker_rejects_stale_delete_and_repairs_missing_job(tmp_path):
    """Async path preserves the same revision, delete, and repair contract."""
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        engine.write.add_node(build_entity_node(node_id="async-race", doc_id="d1"))
        old_job = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.index_kind == "node_embedding"
        )
        replacement = build_entity_node(node_id="async-race", doc_id="d1")
        replacement.summary = "new revision"
        engine.write.add_node(replacement)
        await engine.async_two_stage_projection_adapter.apply_embedding_job(
            entity_kind=old_job.entity_kind,
            entity_id=old_job.entity_id,
            op=old_job.op,
            payload_json=old_job.payload_json,
        )
        assert engine.backend.node_get(ids=["async-race"])["embeddings"] == [None]

        engine.write.add_node(build_entity_node(node_id="async-delete", doc_id="d1"))
        delete_job = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "async-delete" and job.index_kind == "node_embedding"
        )
        assert engine.lifecycle.tombstone_node("async-delete") is True
        await engine.async_two_stage_projection_adapter.apply_embedding_job(
            entity_kind=delete_job.entity_kind,
            entity_id=delete_job.entity_id,
            op=delete_job.op,
            payload_json=delete_job.payload_json,
        )
        assert engine.backend.node_get(ids=["async-delete"])["ids"] == []

        engine.write.add_node(build_entity_node(node_id="async-repair", doc_id="d1"))
        await engine.indexing.make_async_index_job_worker(
            batch_size=8, max_jobs_per_tick=8
        ).tick()
        engine.backend.node_clear_embeddings(ids=["async-repair"])
        assert engine.indexing.repair_missing_two_stage_embedding_jobs() == 1
        metrics = await engine.indexing.make_async_index_job_worker(
            batch_size=8, max_jobs_per_tick=8
        ).tick()
        assert metrics.done == 1
        assert engine.backend.node_get(ids=["async-repair"])["embeddings"][0] is not None
    finally:
        engine.close()
