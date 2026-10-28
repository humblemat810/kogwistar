from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from kogwistar.engine_core.chroma_backend import AsyncChromaBackend
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.workers.async_index_job_worker import AsyncIndexJobWorker
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
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
@pytest.mark.parametrize("kind", ["node", "edge"])
@pytest.mark.parametrize("batched", [False, True])
async def test_async_chroma_legacy_job_promotes_with_staged_fingerprint(
    tmp_path, kind, batched
):
    backend = _async_chroma_backend()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=lambda _engine: backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    try:
        if kind == "edge":
            for endpoint in ("a", "b"):
                await engine.async_add_node(
                    build_entity_node(node_id=endpoint, doc_id="d1", embedding=None)
                )
        entity = (
            build_entity_node(node_id="legacy", doc_id="d1", embedding=None)
            if kind == "node"
            else build_relationship_edge(edge_id="legacy", src="a", tgt="b", doc_id="d1")
        )
        await getattr(engine, f"async_add_{kind}")(entity)
        fingerprint = json.loads(engine.indexing.canonical_revision_payload(
            entity_kind=kind, entity_id="legacy"
        ))["source_fingerprint"]
        adapter = engine.async_two_stage_projection_adapter
        if batched:
            outcomes = await adapter.apply_embedding_jobs_batch([
                SimpleNamespace(job_id="legacy", entity_kind=kind, entity_id="legacy", op="UPSERT", payload_json=None)
            ])
            assert outcomes == {"legacy": None}
        else:
            await adapter.apply_embedding_job(
                entity_kind=kind, entity_id="legacy", op="UPSERT", payload_json=None
            )
        result = await backend.async_call(kind, "get", ids=["legacy"])
        assert result["metadatas"][0]["_kogwistar_source_fingerprint"] == fingerprint
        assert engine.meta_sqlite.get_stage1_node_projection(
            engine.namespace, f"{kind}:legacy"
        ) is None
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("legacy_payload", [False, True])
async def test_async_chroma_late_stale_write_restages_current_revision(
    tmp_path, batched, legacy_payload
):
    backend = _async_chroma_backend()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=lambda _engine: backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    stale_write_started = asyncio.Event()
    release_stale_write = asyncio.Event()
    original_async_call = backend.async_call

    try:
        old = build_entity_node(node_id="async-late", doc_id="old", embedding=None)
        await engine.async_add_node(old)
        old_payload = engine.indexing.canonical_revision_payload(
            entity_kind="node", entity_id="async-late"
        )
        old_fingerprint = json.loads(old_payload)["source_fingerprint"]

        async def delay_old_upsert(collection_key, method, **kwargs):
            metadatas = kwargs.get("metadatas") or []
            metadata = metadatas[0] if metadatas else {}
            if (
                collection_key == "node"
                and method == "upsert"
                and metadata.get("_kogwistar_source_fingerprint") == old_fingerprint
            ):
                stale_write_started.set()
                await release_stale_write.wait()
            return await original_async_call(collection_key, method, **kwargs)

        backend.async_call = delay_old_upsert
        adapter = engine.async_two_stage_projection_adapter
        if batched:
            stale_operation = asyncio.create_task(
                adapter.apply_embedding_jobs_batch(
                    [
                        SimpleNamespace(
                            job_id="old-async-batch",
                            entity_kind="node",
                            entity_id="async-late",
                            op="UPSERT",
                            payload_json=None if legacy_payload else old_payload,
                        )
                    ]
                )
            )
        else:
            stale_operation = asyncio.create_task(
                adapter.apply_embedding_job(
                    entity_kind="node",
                    entity_id="async-late",
                    op="UPSERT",
                    payload_json=None if legacy_payload else old_payload,
                )
            )
        await asyncio.wait_for(stale_write_started.wait(), timeout=5)

        new = build_entity_node(node_id="async-late", doc_id="new", embedding=None)
        await engine.async_add_node(new)
        current_payload = engine.indexing.canonical_revision_payload(
            entity_kind="node", entity_id="async-late"
        )
        await adapter.apply_embedding_job(
            entity_kind="node",
            entity_id="async-late",
            op="UPSERT",
            payload_json=current_payload,
        )
        assert engine.meta_sqlite.get_stage1_node_projection(
            engine.namespace, "node:async-late"
        ) is None

        release_stale_write.set()
        await stale_operation
        restored = engine.meta_sqlite.get_stage1_node_projection(
            engine.namespace, "node:async-late"
        )
        assert restored is not None
        assert json.loads(restored["payload"]["document"])["doc_id"] == "new"

        await adapter.apply_embedding_job(
            entity_kind="node",
            entity_id="async-late",
            op="UPSERT",
            payload_json=current_payload,
        )
        result = await backend.async_node_get(
            ids=["async-late"], include=["documents", "metadatas", "embeddings"]
        )
        assert json.loads(result["documents"][0])["doc_id"] == "new"
        assert result["embeddings"][0] is not None
    finally:
        release_stale_write.set()
        engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_async_chroma_stale_job_replay_restores_and_requeues_current_revision(
    tmp_path, batched
):
    backend = _async_chroma_backend()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=lambda _engine: backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    worker = AsyncIndexJobWorker(engine=engine, batch_size=8, max_jobs_per_tick=8)
    adapter = engine.async_two_stage_projection_adapter
    try:
        old = build_entity_node(node_id="async-replay", doc_id="old", embedding=None)
        await engine.async_add_node(old)
        stale_payload = engine.indexing.canonical_revision_payload(
            entity_kind="node", entity_id="async-replay"
        )
        assert (await worker.tick()).done == 1

        current = build_entity_node(
            node_id="async-replay", doc_id="current", embedding=None
        )
        await engine.async_add_node(current)
        current_payload = engine.indexing.canonical_revision_payload(
            entity_kind="node", entity_id="async-replay"
        )
        assert (await worker.tick()).done == 1
        assert engine.meta_sqlite.get_stage1_node_projection(
            engine.namespace, "node:async-replay"
        ) is None

        old_document = old.model_dump_json(
            field_mode="backend", exclude=["embedding"]
        )
        old_fingerprint = json.loads(stale_payload)["source_fingerprint"]
        await backend.async_call(
            "node",
            "upsert",
            ids=["async-replay"],
            documents=[old_document],
            metadatas=[
                {
                    "node_id": "async-replay",
                    "_kogwistar_stage2_ready": True,
                    "_kogwistar_source_fingerprint": old_fingerprint,
                }
            ],
            embeddings=[[0.1, 0.2]],
        )

        if batched:
            outcomes = await adapter.apply_embedding_jobs_batch(
                [
                    SimpleNamespace(
                        job_id="stale-async-replay",
                        entity_kind="node",
                        entity_id="async-replay",
                        op="UPSERT",
                        payload_json=stale_payload,
                    )
                ]
            )
            assert outcomes == {"stale-async-replay": None}
        else:
            await adapter.apply_embedding_job(
                entity_kind="node",
                entity_id="async-replay",
                op="UPSERT",
                payload_json=stale_payload,
            )

        stage1 = engine.meta_sqlite.get_stage1_node_projection(
            engine.namespace, "node:async-replay"
        )
        assert stage1 is not None
        assert stage1["payload"]["source_fingerprint"] == json.loads(
            current_payload
        )["source_fingerprint"]
        pending = engine.meta_sqlite.list_index_jobs(
            namespace=engine.namespace,
            entity_kind="node",
            entity_id="async-replay",
            status="PENDING",
        )
        assert any(job.payload_json == current_payload for job in pending)

        assert (await worker.tick()).done == 1
        result = await backend.async_node_get(
            ids=["async-replay"], include=["documents", "metadatas", "embeddings"]
        )
        assert json.loads(result["documents"][0])["doc_id"] == "current"
        assert result["metadatas"][0]["_kogwistar_source_fingerprint"] == json.loads(
            current_payload
        )["source_fingerprint"]
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entity_kind", "batched"),
    [("node", False), ("edge", False), ("node", True), ("edge", True)],
)
async def test_async_in_memory_rejects_stale_delete_after_resurrection(
    tmp_path, entity_kind, batched
):
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    entity_id = f"restored-{entity_kind}"

    async def write_current() -> None:
        if entity_kind == "node":
            await engine.async_add_node(build_entity_node(node_id=entity_id, doc_id="d1"))
        else:
            await engine.async_add_node(build_entity_node(node_id="source", doc_id="d1"))
            await engine.async_add_node(build_entity_node(node_id="target", doc_id="d1"))
            await engine.async_add_edge(
                build_relationship_edge(
                    edge_id=entity_id, src="source", tgt="target", doc_id="d1"
                )
            )

    try:
        await write_current()
        tombstone = getattr(engine.lifecycle, f"tombstone_{entity_kind}")
        assert tombstone(entity_id) is True
        stale_delete = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_kind == entity_kind
            and job.entity_id == entity_id
            and job.index_kind == "node_embedding"
            and job.op == "DELETE"
        )

        await write_current()
        current_revision = engine.indexing.canonical_entity_revision(
            entity_kind=entity_kind, entity_id=entity_id
        )
        assert current_revision is not None and current_revision.state == "active"
        adapter = engine.async_two_stage_projection_adapter
        if batched:
            outcomes = await adapter.apply_embedding_jobs_batch(
                [
                    {
                        "job_id": "stale-delete",
                        "entity_kind": entity_kind,
                        "entity_id": entity_id,
                        "op": stale_delete.op,
                        "payload_json": stale_delete.payload_json,
                    }
                ]
            )
            assert outcomes["stale-delete"] is None
        else:
            await adapter.apply_embedding_job(
                entity_kind=entity_kind,
                entity_id=entity_id,
                op=stale_delete.op,
                payload_json=stale_delete.payload_json,
            )

        current = getattr(engine.backend, f"{entity_kind}_get")(ids=[entity_id])
        assert current["ids"] == [entity_id]
        assert current["embeddings"] == [None]
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "meta_impl",
    [
        "python",
        pytest.param("rust", marks=pytest.mark.requires_rust, id="rust"),
    ],
)
async def test_async_in_memory_slow_embedding_cannot_overwrite_newer_revision(
    tmp_path, monkeypatch, batched, meta_impl
):
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", meta_impl)
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        persistence_mode="two_stage",
        embedding_function=build_test_embedding_function("constant", dim=2),
    )
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()

    async def slow_embedding(_texts):
        provider_started.set()
        await release_provider.wait()
        return [[1.0, 0.0]]

    monkeypatch.setattr(engine, "_ef", slow_embedding)
    try:
        await engine.async_add_node(
            build_entity_node(node_id="raced", doc_id="d1", embedding=None)
        )
        job = next(
            item
            for item in engine.meta_sqlite.list_index_jobs(namespace="default")
            if item.entity_kind == "node"
            and item.entity_id == "raced"
            and item.op == "UPSERT"
        )
        adapter = engine.async_two_stage_projection_adapter

        if batched:
            work = asyncio.create_task(
                adapter.apply_embedding_jobs_batch([job])
            )
        else:
            work = asyncio.create_task(
                adapter.apply_embedding_job(
                    entity_kind=job.entity_kind,
                    entity_id=job.entity_id,
                    op=job.op,
                    payload_json=job.payload_json,
                )
            )
        await provider_started.wait()

        current = build_entity_node(
            node_id="raced", doc_id="d1", embedding=None
        )
        current.summary = "newer canonical revision"
        await engine.async_add_node(current)
        release_provider.set()
        await work

        row = engine.backend.node_get(ids=["raced"], include=["embeddings"])
        assert row["embeddings"] == [None]
        revision = engine.indexing.canonical_entity_revision(
            entity_kind="node", entity_id="raced", namespace="default"
        )
        assert revision is not None
        assert revision.payload["summary"] == "newer canonical revision"
    finally:
        release_provider.set()
        engine.close()
