from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.graph_query import GraphQuery
from kogwistar.workers.async_index_job_worker import AsyncIndexJobWorker
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests.core.two_stage_case_catalog import two_stage_case

pytestmark = [pytest.mark.ci_full, pytest.mark.integration, pytest.mark.e2e]


def _require_postgres(sa_engine, pg_schema) -> None:
    if sa_engine is None or pg_schema is None:
        pytest.skip("live PostgreSQL/pgvector fixture unavailable")


def _engine(sa_engine, pg_schema) -> GraphKnowledgeEngine:
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    return GraphKnowledgeEngine(
        backend=backend,
        persistence_mode="two_stage",
        embedding_function=ConstantEmbeddingFunction(dim=3),
    )


def _conversation_tree():
    nodes = [
        build_entity_node(node_id="pg-conv-root", doc_id="pg-conv"),
        build_entity_node(node_id="pg-conv-child", doc_id="pg-conv"),
        build_entity_node(node_id="pg-conv-leaf", doc_id="pg-conv"),
    ]
    edges = [
        build_relationship_edge(
            edge_id="pg-conv-root-child", src="pg-conv-root", tgt="pg-conv-child",
            doc_id="pg-conv", relation="next_turn", embedding=None,
        ),
        build_relationship_edge(
            edge_id="pg-conv-child-leaf", src="pg-conv-child", tgt="pg-conv-leaf",
            doc_id="pg-conv", relation="next_turn", embedding=None,
        ),
    ]
    return nodes, edges


def _conversation_tree_snapshot(engine: GraphKnowledgeEngine) -> dict[str, object]:
    graph = GraphQuery(engine)
    ids = [
        "pg-conv-root", "pg-conv-child", "pg-conv-leaf",
        "pg-conv-root-child", "pg-conv-child-leaf",
    ]
    return {
        "neighbors": {
            item: {
                "nodes": sorted(graph.neighbors(item)["nodes"]),
                "edges": sorted(graph.neighbors(item)["edges"]),
            }
            for item in ids
        },
        "root_to_leaf": graph.shortest_path("pg-conv-root", "pg-conv-leaf"),
    }


async def _conversation_tree_snapshot_async(engine: GraphKnowledgeEngine) -> dict[str, object]:
    graph = GraphQuery(engine)
    ids = [
        "pg-conv-root", "pg-conv-child", "pg-conv-leaf",
        "pg-conv-root-child", "pg-conv-child-leaf",
    ]
    neighbors = {}
    for item in ids:
        got = await graph.neighbors_async(item)
        neighbors[item] = {
            "nodes": sorted(got["nodes"]), "edges": sorted(got["edges"])
        }
    return {"neighbors": neighbors, "root_to_leaf": []}


def _promote_pg_entities(engine: GraphKnowledgeEngine, entity_ids: set[str]) -> None:
    for job in engine.meta_sqlite.list_index_jobs(namespace="default"):
        if job.entity_id not in entity_ids:
            continue
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=job.entity_kind,
            entity_id=job.entity_id,
            op=job.op,
            payload_json=job.payload_json,
        )


@pytest.mark.conversation
@pytest.mark.runtime_sync
def test_postgres_conversation_tree_preserves_links_during_partial_promotion(
    sa_engine, pg_schema
) -> None:
    """Structural traversal stays available while vector rows promote."""
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    try:
        nodes, edges = _conversation_tree()
        for node in nodes:
            engine.write.add_node(node)
        for edge in edges:
            engine.write.add_edge(edge)

        pending = _conversation_tree_snapshot(engine)
        _promote_pg_entities(
            engine, {"pg-conv-root", "pg-conv-root-child", "pg-conv-leaf"}
        )
        mixed = _conversation_tree_snapshot(engine)
        _promote_pg_entities(engine, {"pg-conv-child", "pg-conv-child-leaf"})
        ready = _conversation_tree_snapshot(engine)

        assert mixed == pending
        assert ready == pending
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.conversation
@pytest.mark.runtime_async
async def test_async_postgres_conversation_tree_preserves_links_during_promotion(
    async_pg_backend,
):
    provider = ConstantEmbeddingFunction(dim=3)
    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        nodes, edges = _conversation_tree()
        for node in nodes:
            await engine.async_add_node(node)
        for edge in edges:
            await engine.async_add_edge(edge)

        pending = await _conversation_tree_snapshot_async(engine)
        jobs = engine.meta_sqlite.list_index_jobs(namespace="default")
        for job in jobs:
            if job.entity_id not in {"pg-conv-root", "pg-conv-root-child", "pg-conv-leaf"}:
                continue
            await engine.async_two_stage_projection_adapter.apply_embedding_job(
                entity_kind=job.entity_kind,
                entity_id=job.entity_id,
                op=job.op,
                payload_json=job.payload_json,
            )
        mixed = await _conversation_tree_snapshot_async(engine)

        jobs = engine.meta_sqlite.list_index_jobs(namespace="default")
        for job in jobs:
            if job.entity_id not in {"pg-conv-child", "pg-conv-child-leaf"}:
                continue
            await engine.async_two_stage_projection_adapter.apply_embedding_job(
                entity_kind=job.entity_kind,
                entity_id=job.entity_id,
                op=job.op,
                payload_json=job.payload_json,
            )
        ready = await _conversation_tree_snapshot_async(engine)

        assert mixed == pending
        assert ready == pending
    finally:
        engine.close()


@pytest.mark.conversation
@pytest.mark.runtime_sync
@pytest.mark.runtime_bridge_parity
def test_rust_postgres_conversation_tree_preserves_links_during_promotion(
    sa_engine, pg_schema, monkeypatch
) -> None:
    """Rust-authority facade keeps the same graph traversal contract."""
    _require_postgres(sa_engine, pg_schema)
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    engine = GraphKnowledgeEngine(
        backend=backend,
        persistence_mode="two_stage",
        embedding_function=ConstantEmbeddingFunction(dim=3),
    )
    try:
        nodes, edges = _conversation_tree()
        for node in nodes:
            engine.write.add_node(node)
        for edge in edges:
            engine.write.add_edge(edge)

        pending = _conversation_tree_snapshot(engine)
        _promote_pg_entities(
            engine, {"pg-conv-root", "pg-conv-root-child", "pg-conv-leaf"}
        )
        mixed = _conversation_tree_snapshot(engine)
        _promote_pg_entities(engine, {"pg-conv-child", "pg-conv-child-leaf"})
        ready = _conversation_tree_snapshot(engine)

        assert mixed == pending
        assert ready == pending
    finally:
        engine.close()


@two_stage_case("pending_visibility")
@two_stage_case("promotion_handoff")
def test_postgres_two_stage_pending_then_same_store_promotion(sa_engine, pg_schema) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    node = build_entity_node(node_id="n1", doc_id="d1", embedding=None)

    engine.write.add_node(node)

    stage1 = engine.backend.stage1_projection_query(
        namespace="default", entity_kind="node", ids=["n1"]
    )
    assert [row["entity_id"] for row in stage1] == ["n1"]
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []
    assert engine.backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
    )["ids"] == [[]]

    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1

    promoted = engine.backend.node_get(ids=["n1"], include=["embeddings", "metadatas"])
    assert promoted["ids"] == ["n1"]
    assert promoted["embeddings"][0] is not None
    assert promoted["metadatas"][0]["_kogwistar_stage2_ready"] is True
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None


def test_postgres_two_stage_old_job_cannot_promote_current_revision(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    jobs = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )
    old_payload = jobs[0].payload_json

    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d2", embedding=None))
    engine.two_stage_projection_adapter.apply_embedding_job(
        entity_kind="node", entity_id="n1", op="UPSERT", payload_json=old_payload
    )

    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []
    current_payload = engine.indexing.canonical_revision_payload(
        entity_kind="node", entity_id="n1"
    )
    assert json.loads(current_payload)["canonical_revision"] == 2


def test_postgres_two_stage_event_and_stage1_admission_roll_back_together(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    original = engine.two_stage_projection_adapter.add_node

    def fail_after_stage1(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected Stage-1 failure")

    monkeypatch.setattr(engine.two_stage_projection_adapter, "add_node", fail_after_stage1)
    with pytest.raises(RuntimeError, match="injected Stage-1 failure"):
        engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))

    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None
    assert list(engine.meta_sqlite.iter_entity_events(namespace="default", from_seq=1)) == []


def test_postgres_two_stage_delete_before_promotion_cleans_stage1(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))

    assert engine.tombstone_node("n1") is True
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []


@two_stage_case("recovery_reconciliation")
def test_postgres_two_stage_reconciliation_cleans_stage1_after_stage2_write(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    row = engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    )
    assert row is not None
    metadata = dict(row["metadata"])
    metadata["_kogwistar_stage2_ready"] = True
    metadata["_kogwistar_source_fingerprint"] = row["source_fingerprint"]
    engine.backend.node_upsert(
        ids=["n1"], documents=[row["document"]], metadatas=[metadata], embeddings=[[1, 0, 0]]
    )

    assert engine.two_stage_projection_adapter.reconcile_projection() == 1
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None


@two_stage_case("delete_race")
def test_postgres_two_stage_delete_during_embedding_cannot_resurrect(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    job = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )[0]
    original = engine.embed.iterative_defensive_emb
    called = {"value": False}

    def embed_then_delete(document):
        if not called["value"]:
            called["value"] = True
            assert engine.tombstone_node("n1") is True
        return original(document)

    monkeypatch.setattr(engine.embed, "iterative_defensive_emb", embed_then_delete)
    engine.two_stage_projection_adapter.apply_embedding_job(
        entity_kind="node", entity_id="n1", op="UPSERT", payload_json=job.payload_json
    )
    assert engine.backend.node_get(ids=["n1"], include=["embeddings"])["ids"] == []
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="node", entity_id="n1"
    ) is None


def test_postgres_two_stage_python_authority_rejects_stale_inflight_promotion(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    for selector in (
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_GRAPH_STORE",
    ):
        monkeypatch.setenv(selector, "python")
    engine = _engine(sa_engine, pg_schema)
    embedding_started = threading.Event()
    release_embedding = threading.Event()
    errors: list[BaseException] = []

    def provider(document: str) -> list[float]:
        if "old canonical revision" in document:
            embedding_started.set()
            if not release_embedding.wait(timeout=10):
                raise TimeoutError("test did not release the stale embedding")
            return [0.1, 0.2, 0.3]
        return [0.4, 0.5, 0.6]

    try:
        engine.write.add_node(
            build_entity_node(
                node_id="python-stale-race",
                doc_id="d1",
                summary="old canonical revision",
                embedding=None,
            )
        )
        old_job = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "python-stale-race"
            and job.index_kind == "node_embedding"
        )
        monkeypatch.setattr(engine.embed, "iterative_defensive_emb", provider)

        def promote_old_revision() -> None:
            try:
                engine.two_stage_projection_adapter.apply_embedding_job(
                    entity_kind=old_job.entity_kind,
                    entity_id=old_job.entity_id,
                    op=old_job.op,
                    payload_json=old_job.payload_json,
                )
            except BaseException as exc:
                errors.append(exc)

        promotion = threading.Thread(target=promote_old_revision)
        promotion.start()
        assert embedding_started.wait(timeout=10)
        engine.write.add_node(
            build_entity_node(
                node_id="python-stale-race",
                doc_id="d2",
                summary="new canonical revision",
                embedding=None,
            )
        )
        release_embedding.set()
        promotion.join(timeout=10)
        assert not promotion.is_alive()
        assert errors == []

        staged = engine.backend.stage1_projection_get(
            namespace="default", entity_kind="node", entity_id="python-stale-race"
        )
        assert staged is not None and "new canonical revision" in staged["document"]
        assert engine.backend.node_get(
            ids=["python-stale-race"], include=["embeddings"]
        )["ids"] == []

        new_job = max(
            (
                job
                for job in engine.meta_sqlite.list_index_jobs(namespace="default")
                if job.entity_id == "python-stale-race"
                and job.index_kind == "node_embedding"
            ),
            key=lambda job: job.created_at,
        )
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=new_job.entity_kind,
            entity_id=new_job.entity_id,
            op=new_job.op,
            payload_json=new_job.payload_json,
        )
        ready = engine.backend.node_get(
            ids=["python-stale-race"], include=["documents", "embeddings"]
        )
        assert "new canonical revision" in ready["documents"][0]
        assert ready["embeddings"] == [[0.4, 0.5, 0.6]]
    finally:
        release_embedding.set()
        engine.close()


@pytest.mark.asyncio
async def test_postgres_two_stage_async_python_authority_rejects_stale_inflight_promotion(
    async_pg_backend, monkeypatch
) -> None:
    for selector in (
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_GRAPH_STORE",
    ):
        monkeypatch.setenv(selector, "python")
    embedding_started = asyncio.Event()
    release_embedding = asyncio.Event()

    async def provider(texts: list[str]) -> list[list[float]]:
        if any("old canonical revision" in text for text in texts):
            embedding_started.set()
            await release_embedding.wait()
            return [[0.1, 0.2, 0.3] for _ in texts]
        return [[0.4, 0.5, 0.6] for _ in texts]

    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        assert engine.async_two_stage_projection_adapter.__class__.__name__ == (
            "AsyncPostgresTwoStageProjectionAdapter"
        )
        await engine.async_add_node(
            build_entity_node(
                node_id="async-python-stale-race",
                doc_id="d1",
                summary="old canonical revision",
                embedding=None,
            )
        )
        old_job = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "async-python-stale-race"
            and job.index_kind == "node_embedding"
        )
        promotion = asyncio.create_task(
            engine.async_two_stage_projection_adapter.apply_embedding_job(
                entity_kind=old_job.entity_kind,
                entity_id=old_job.entity_id,
                op=old_job.op,
                payload_json=old_job.payload_json,
            )
        )
        await asyncio.wait_for(embedding_started.wait(), timeout=10)
        await engine.async_add_node(
            build_entity_node(
                node_id="async-python-stale-race",
                doc_id="d2",
                summary="new canonical revision",
                embedding=None,
            )
        )
        release_embedding.set()
        await promotion

        staged = await async_pg_backend.stage1_projection_get_async(
            namespace="default", entity_kind="node", entity_id="async-python-stale-race"
        )
        assert staged is not None and "new canonical revision" in staged["document"]
        assert (await async_pg_backend.node_get(
            ids=["async-python-stale-race"], include=["embeddings"]
        ))["ids"] == []

        new_job = max(
            (
                job
                for job in engine.meta_sqlite.list_index_jobs(namespace="default")
                if job.entity_id == "async-python-stale-race"
                and job.index_kind == "node_embedding"
            ),
            key=lambda job: job.created_at,
        )
        await engine.async_two_stage_projection_adapter.apply_embedding_job(
            entity_kind=new_job.entity_kind,
            entity_id=new_job.entity_id,
            op=new_job.op,
            payload_json=new_job.payload_json,
        )
        ready = await async_pg_backend.node_get(
            ids=["async-python-stale-race"], include=["documents", "embeddings"]
        )
        assert "new canonical revision" in ready["documents"][0]
        assert ready["embeddings"] == [[0.4, 0.5, 0.6]]
    finally:
        release_embedding.set()
        engine.close()


@pytest.mark.asyncio
async def test_postgres_two_stage_async_admission_rolls_back_event_and_stage1_together(
    async_pg_backend, monkeypatch
) -> None:
    for selector in (
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_GRAPH_STORE",
    ):
        monkeypatch.setenv(selector, "python")
    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=ConstantEmbeddingFunction(dim=3),
    )
    adapter = engine.async_two_stage_projection_adapter
    assert adapter is not None
    add_node = adapter.add_node

    async def fail_after_stage1(node, *, doc_id=None):
        await add_node(node, doc_id=doc_id)
        raise RuntimeError("injected async Stage-1 failure")

    monkeypatch.setattr(adapter, "add_node", fail_after_stage1)
    try:
        with pytest.raises(RuntimeError, match="injected async Stage-1 failure"):
            await engine.async_add_node(
                build_entity_node(
                    node_id="async-admission-rollback", doc_id="d1", embedding=None
                )
            )

        events = list(
            engine.metadata.iter_entity_events(namespace="default", from_seq=1)
        )
        assert events == []
        assert await async_pg_backend.stage1_projection_get_async(
            namespace="default",
            entity_kind="node",
            entity_id="async-admission-rollback",
        ) is None
        assert (await async_pg_backend.node_get(
            ids=["async-admission-rollback"], include=["embeddings"]
        ))["ids"] == []
    finally:
        engine.close()


@two_stage_case("stale_revision")
def test_postgres_two_stage_late_v1_cannot_overwrite_v2(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="v1", embedding=None))
    old_job = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )[0]
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="v2", embedding=None))
    assert engine.indexing.reconcile_indexes(max_jobs=10) == 1
    current = engine.backend.node_get(ids=["n1"], include=["metadatas"])
    current_fp = current["metadatas"][0]["_kogwistar_source_fingerprint"]
    engine.two_stage_projection_adapter.apply_embedding_job(
        entity_kind="node", entity_id="n1", op="UPSERT", payload_json=old_job.payload_json
    )
    assert engine.backend.node_get(ids=["n1"], include=["metadatas"])["metadatas"][0][
        "_kogwistar_source_fingerprint"
    ] == current_fp


def test_postgres_two_stage_scanner_repairs_lost_enqueue(
    sa_engine, pg_schema
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    job = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )[0]
    claimed = engine.meta_sqlite.claim_index_jobs(limit=1, lease_seconds=60, namespace="default")
    assert len(claimed) == 1
    assert engine.meta_sqlite.mark_index_job_done(
        claimed[0].job_id, claim_token=claimed[0].claim_token
    ) is True
    assert engine.indexing.repair_missing_two_stage_embedding_jobs(max_entities=10) == 1
    repaired = engine.meta_sqlite.list_index_jobs(
        namespace="default", entity_kind="node", entity_id="n1"
    )
    assert any(item.status == "PENDING" and item.index_kind == "node_embedding" for item in repaired)


@pytest.mark.parametrize("entity_kind", ["node", "edge"])
@pytest.mark.parametrize("authority", ["python", "rust"])
def test_postgres_designated_worker_repairs_lost_enqueue(
    sa_engine, pg_schema, entity_kind, authority, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    if authority == "rust":
        monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
        monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
        monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = _engine(sa_engine, pg_schema)
    expected_adapter = (
        "RustPostgresTwoStageProjectionAdapter"
        if authority == "rust"
        else "PostgresTwoStageProjectionAdapter"
    )
    assert engine.two_stage_projection_adapter.__class__.__name__ == expected_adapter
    if entity_kind == "node":
        engine.write.add_node(build_entity_node(node_id="worker-repair", doc_id="d1", embedding=None))
    else:
        engine.write.add_node(build_entity_node(node_id="source", doc_id="d1"))
        engine.write.add_node(build_entity_node(node_id="target", doc_id="d1"))
        engine.indexing.make_index_job_worker(max_jobs_per_tick=10).tick()
        engine.write.add_edge(
            build_relationship_edge(
                edge_id="worker-repair", src="source", tgt="target", doc_id="d1", embedding=None
            )
        )
    claimed = engine.meta_sqlite.claim_index_jobs(limit=10, lease_seconds=60, namespace="default")
    for job in claimed:
        assert engine.meta_sqlite.mark_index_job_done(job.job_id, claim_token=job.claim_token)

    metrics = engine.indexing.make_index_job_worker(
        max_jobs_per_tick=10, repair_scan_interval_s=30
    ).tick()

    assert metrics.repaired == 1
    assert metrics.repair_scan_failures == 0
    assert metrics.done >= 1
    getter = getattr(engine.backend, f"{entity_kind}_get")
    assert getter(ids=["worker-repair"], include=["embeddings"])["embeddings"][0] is not None


def test_postgres_two_stage_public_api_and_batch_claim_isolate_failure(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2", embedding=None))
    public = engine.get_nodes(ids=["n1"], include=["documents", "metadatas", "embeddings"])
    assert [node.id for node in public] == ["n1"]
    assert public[0].embedding is None
    assert [node.id for node in engine.get_nodes(where={"label": "Node n1"})] == ["n1"]
    assert engine.query_nodes(query_embeddings=[[1, 0, 0]], n_results=10) == []

    original = engine.embed.iterative_defensive_emb
    calls = {"count": 0}
    batch_calls = {"count": 0}

    def batch_provider(documents):
        batch_calls["count"] += 1
        if any("n2" in document for document in documents):
            raise RuntimeError("injected batch provider failure")
        return ConstantEmbeddingFunction(dim=3)(documents)

    monkeypatch.setattr(engine, "_ef", batch_provider)

    def one_bad(document):
        calls["count"] += 1
        if "n2" in document:
            raise RuntimeError("injected provider failure")
        return original(document)

    monkeypatch.setattr(engine.embed, "iterative_defensive_emb", one_bad)
    worker = engine.indexing.make_index_job_worker(batch_size=2, max_inflight=2, max_jobs_per_tick=2)
    metrics = worker.tick()
    assert metrics.claimed == 2
    assert metrics.done == 1
    assert metrics.retried == 1
    assert calls["count"] == 2
    assert batch_calls["count"] >= 1
    semantic = engine.query_nodes(query_embeddings=[[1, 0, 0]], n_results=10)
    assert {node.id for row in semantic for node in row} == {"n1"}


def test_postgres_two_stage_edges_follow_same_promotion_contract(sa_engine, pg_schema) -> None:
    _require_postgres(sa_engine, pg_schema)
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2", embedding=None))
    edge = build_relationship_edge(
        edge_id="e1", src="n1", tgt="n2", doc_id="d-edge", embedding=None
    )

    engine.write.add_edge(edge)

    staged = engine.backend.stage1_projection_get(
        namespace="default", entity_kind="edge", entity_id="e1"
    )
    assert staged is not None
    assert engine.backend.edge_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
    )["ids"] == [[]]

    assert engine.indexing.reconcile_indexes(max_jobs=10) == 3
    promoted = engine.backend.edge_get(ids=["e1"], include=["embeddings", "metadatas"])
    assert promoted["ids"] == ["e1"]
    assert promoted["embeddings"][0] is not None
    assert engine.backend.stage1_projection_get(
        namespace="default", entity_kind="edge", entity_id="e1"
    ) is None


@two_stage_case("batch_embedding")
def test_postgres_two_stage_provider_batch_promotes_multiple_nodes(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "two-stage-embedding-batch-v1.json"
        ).read_text(encoding="utf-8")
    )
    expected_embedding = contract["expected_embedding"]
    assert len(expected_embedding) == contract["embedding_dimension"]
    engine = _engine(sa_engine, pg_schema)
    engine.write.add_node(build_entity_node(node_id="n1", doc_id="d1", embedding=None))
    engine.write.add_node(build_entity_node(node_id="n2", doc_id="d2", embedding=None))
    jobs = engine.meta_sqlite.list_index_jobs(namespace="default")
    embedding_calls: list[list[str]] = []

    def batch_embed(documents):
        embedding_calls.append(list(documents))
        return [expected_embedding for _ in documents]

    monkeypatch.setattr(engine, "_ef", batch_embed)
    outcomes = engine.two_stage_projection_adapter.apply_embedding_jobs_batch(
        [job for job in jobs if job.index_kind == "node_embedding"]
    )

    assert len(embedding_calls) == 1
    assert len(embedding_calls[0]) == contract["batch_size"]
    assert set(outcomes) == {job.job_id for job in jobs if job.index_kind == "node_embedding"}
    assert all(value is None for value in outcomes.values())
    promoted = engine.backend.node_get(ids=["n1", "n2"], include=["embeddings"])
    assert promoted["ids"] == ["n1", "n2"]
    assert promoted["embeddings"] == [expected_embedding, expected_embedding]


@pytest.mark.asyncio
async def test_postgres_single_stage_async_writes_embedding_directly(async_pg_backend) -> None:
    main_thread = threading.get_ident()
    call_threads: list[int] = []

    async def provider(documents):
        call_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return [[1.0, 0.0, 0.0] for _ in documents]

    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        embedding_function=provider,
        # Omitted intentionally: single_stage remains the default.
    )
    try:
        node = build_entity_node(node_id="async-pg-single", doc_id="d1")
        await engine.async_add_node(node)
        got = await async_pg_backend.node_get(
            ids=[node.id], include=["embeddings"]
        )
        assert got["ids"] == [node.id]
        assert got["embeddings"] == [[1.0, 0.0, 0.0]]
        assert call_threads and call_threads[0] == main_thread
        events = list(
            engine.meta_sqlite.iter_entity_events(namespace="default", from_seq=1)
        )
        assert any(event[2] == node.id and event[3] == "ADD" for event in events)
        jobs = engine.meta_sqlite.list_index_jobs(namespace="default")
        assert all(job.index_kind != "node_embedding" for job in jobs)
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_postgres_two_stage_async_backend_uses_async_adapter(
    async_pg_backend,
) -> None:
    calls: list[int] = []

    async def provider(documents):
        calls.append(len(documents))
        return [[1.0, 0.0, 0.0] for _ in documents]

    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        assert engine.two_stage_projection_capability.is_complete()
        assert engine.two_stage_projection_adapter is None
        assert engine.async_two_stage_projection_adapter is not None
        for node_id in ("async-pg-n1", "async-pg-n2"):
            await engine.async_add_node(
                build_entity_node(node_id=node_id, doc_id="d1", embedding=None)
            )
        worker = AsyncIndexJobWorker(engine=engine, batch_size=8, max_jobs_per_tick=8)
        metrics = await worker.tick()
        assert metrics.claimed == 2 and metrics.done == 2
        assert calls == [2]
        ready = await engine.backend.node_get(
            ids=["async-pg-n1", "async-pg-n2"], include=["embeddings"]
        )
        assert all(embedding is not None for embedding in ready["embeddings"])
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.conversation
@pytest.mark.regression
async def test_postgres_two_stage_async_reconciles_edge_endpoints_after_stage2_write(
    async_pg_backend,
):
    """Async recovery rebuilds structural endpoints before Stage-1 cleanup."""
    async def provider(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        edge = build_relationship_edge(
            edge_id="async-pg-recovery-edge", src="async-pg-recovery-src",
            tgt="async-pg-recovery-tgt", doc_id="async-pg-recovery",
            embedding=None,
        )
        for node_id in ("async-pg-recovery-src", "async-pg-recovery-tgt"):
            await engine.async_add_node(
                build_entity_node(node_id=node_id, doc_id="async-pg-recovery")
            )
        await engine.async_add_edge(edge)
        staged = await async_pg_backend.stage1_projection_get_async(
            namespace="default", entity_kind="edge", entity_id=edge.safe_get_id()
        )
        assert staged is not None

        metadata = dict(staged["metadata"])
        metadata["_kogwistar_stage2_ready"] = True
        metadata["_kogwistar_source_fingerprint"] = staged["source_fingerprint"]
        await async_pg_backend.edge_upsert(
            ids=[edge.safe_get_id()], documents=[staged["document"]],
            metadatas=[metadata], embeddings=[[1.0, 0.0, 0.0]],
        )
        await async_pg_backend.edge_endpoints_delete(
            where={"edge_id": edge.safe_get_id()}
        )

        assert await engine.async_two_stage_projection_adapter.reconcile_projection() == 1
        endpoints = await async_pg_backend.edge_endpoints_get(
            where={"edge_id": edge.safe_get_id()}, include=["documents"]
        )
        assert len(endpoints["ids"]) == 2
        assert await async_pg_backend.stage1_projection_get_async(
            namespace="default", entity_kind="edge", entity_id=edge.safe_get_id()
        ) is None
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_postgres_two_stage_async_rust_authority_uses_native_facade(
    async_pg_backend, monkeypatch, batched
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    calls: list[int] = []

    async def provider(texts):
        calls.append(len(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )

    async def apply_delete(job) -> None:
        adapter = engine.async_two_stage_projection_adapter
        if batched:
            outcomes = await adapter.apply_embedding_jobs_batch([job])
            assert outcomes[job.job_id] is None
        else:
            await adapter.apply_embedding_job(
                entity_kind=job.entity_kind,
                entity_id=job.entity_id,
                op=job.op,
                payload_json=job.payload_json,
            )

    try:
        assert engine.async_two_stage_projection_adapter.__class__.__name__ == (
            "AsyncRustPostgresTwoStageProjectionAdapter"
        )
        for node_id in ("async-rust-pg-n1", "async-rust-pg-n2"):
            await engine.async_add_node(
                build_entity_node(node_id=node_id, doc_id="d1", embedding=None)
            )
        metrics = await AsyncIndexJobWorker(
            engine=engine, batch_size=8, max_jobs_per_tick=8
        ).tick()
        assert metrics.done == 2
        assert calls == [2]
        ready = await engine.backend.node_get(
            ids=["async-rust-pg-n1", "async-rust-pg-n2"], include=["embeddings"]
        )
        assert all(embedding is not None for embedding in ready["embeddings"])

        delete_target = "async-rust-pg-n1"
        assert engine.lifecycle.tombstone_node(delete_target) is True
        delete_job = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == delete_target
            and job.index_kind == "node_embedding"
            and job.op == "DELETE"
        )
        await apply_delete(delete_job)
        tombstoned = await engine.backend.node_get(
            ids=[delete_target], include=["embeddings", "metadatas"]
        )
        assert tombstoned["ids"] == [delete_target]
        assert tombstoned["embeddings"] == [None]
        assert tombstoned["metadatas"][0]["lifecycle_status"] == "tombstoned"

        await engine.async_add_node(
            build_entity_node(node_id=delete_target, doc_id="d2", embedding=None)
        )
        resurrect_worker = AsyncIndexJobWorker(
            engine=engine, batch_size=8, max_jobs_per_tick=8
        )
        for _ in range(3):
            await resurrect_worker.tick()
            resurrected = await engine.backend.node_get(
                ids=[delete_target], include=["embeddings", "metadatas"]
            )
            if resurrected["embeddings"][0] is not None:
                break
        assert resurrected["embeddings"][0] is not None
        assert resurrected["metadatas"][0]["lifecycle_status"] == "active"

        await apply_delete(delete_job)
        after_stale_delete = await engine.backend.node_get(
            ids=[delete_target], include=["embeddings"]
        )
        assert after_stale_delete["embeddings"][0] is not None
        assert calls == [2, 1]
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_postgres_two_stage_async_rust_authority_rolls_back_event_and_stage1(
    async_pg_backend, monkeypatch
) -> None:
    for selector in (
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_GRAPH_STORE",
    ):
        monkeypatch.setenv(selector, "rust")
    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=ConstantEmbeddingFunction(dim=3),
    )
    adapter = engine.async_two_stage_projection_adapter
    assert adapter is not None
    assert adapter.__class__.__name__ == "AsyncRustPostgresTwoStageProjectionAdapter"
    add_node = adapter.add_node

    async def fail_after_stage1(node, *, doc_id=None):
        await add_node(node, doc_id=doc_id)
        raise RuntimeError("injected Rust-authority async Stage-1 failure")

    monkeypatch.setattr(adapter, "add_node", fail_after_stage1)
    entity_id = "async-rust-authority-admission-rollback"
    try:
        with pytest.raises(RuntimeError, match="injected Rust-authority async Stage-1 failure"):
            await engine.async_add_node(
                build_entity_node(node_id=entity_id, doc_id="d1", embedding=None)
            )

        assert list(engine.metadata.iter_entity_events(namespace="default", from_seq=1)) == []
        assert await async_pg_backend.stage1_projection_get_async(
            namespace="default", entity_kind="node", entity_id=entity_id
        ) is None
        assert (await async_pg_backend.node_get(
            ids=[entity_id], include=["embeddings"]
        ))["ids"] == []
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_postgres_two_stage_async_rust_authority_rejects_stale_inflight_promotion(
    async_pg_backend, monkeypatch
) -> None:
    """An older embedding result cannot promote over a newer canonical revision."""
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    promotion_started = threading.Event()
    release_promotion = threading.Event()
    calls: list[list[str]] = []

    async def provider(texts):
        calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    engine = GraphKnowledgeEngine(
        backend=async_pg_backend,
        persistence_mode="two_stage",
        embedding_function=provider,
    )
    try:
        await engine.async_add_node(
            build_entity_node(
                node_id="async-rust-stale-race",
                doc_id="d1",
                summary="old canonical revision",
                embedding=None,
            )
        )
        old_job = next(
            job
            for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "async-rust-stale-race"
            and job.index_kind == "node_embedding"
        )
        adapter = engine.async_two_stage_projection_adapter._sync_adapter
        promote = adapter._promote_record

        def pause_before_promote(**values):
            promotion_started.set()
            if not release_promotion.wait(timeout=10):
                raise TimeoutError("test did not release the stale promotion")
            promote(**values)

        monkeypatch.setattr(adapter, "_promote_record", pause_before_promote)

        old_promotion = asyncio.create_task(
            engine.async_two_stage_projection_adapter.apply_embedding_job(
                entity_kind=old_job.entity_kind,
                entity_id=old_job.entity_id,
                op=old_job.op,
                payload_json=old_job.payload_json,
            )
        )
        assert await asyncio.to_thread(promotion_started.wait, 10)

        await engine.async_add_node(
            build_entity_node(
                node_id="async-rust-stale-race",
                doc_id="d2",
                summary="new canonical revision",
                embedding=None,
            )
        )
        release_promotion.set()
        await old_promotion

        current = await async_pg_backend.node_get(
            ids=["async-rust-stale-race"], include=["documents", "embeddings"]
        )
        assert current["embeddings"] == [None]
        assert "new canonical revision" in current["documents"][0]

        metrics = await AsyncIndexJobWorker(
            engine=engine, batch_size=4, max_jobs_per_tick=4
        ).tick()
        assert metrics.done >= 1
        promoted = await async_pg_backend.node_get(
            ids=["async-rust-stale-race"], include=["documents", "embeddings"]
        )
        assert "new canonical revision" in promoted["documents"][0]
        assert promoted["embeddings"][0] is not None
        assert len(calls) == 2
        assert "old canonical revision" in calls[0][0]
        assert "new canonical revision" in calls[1][0]
    finally:
        release_promotion.set()
        engine.close()


@two_stage_case("pending_visibility")
@two_stage_case("promotion_handoff")
@two_stage_case("batch_embedding")
@two_stage_case("stale_revision")
@two_stage_case("delete_race")
@two_stage_case("recovery_reconciliation")
def test_postgres_two_stage_rust_authority_uses_native_two_stage_adapter(
    sa_engine, pg_schema, monkeypatch
) -> None:
    _require_postgres(sa_engine, pg_schema)
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    engine = GraphKnowledgeEngine(
        backend=backend,
        persistence_mode="two_stage",
        embedding_function=ConstantEmbeddingFunction(dim=3),
    )
    try:
        assert engine.two_stage_projection_capability.is_complete()
        assert engine.two_stage_projection_adapter.__class__.__name__ == (
            "RustPostgresTwoStageProjectionAdapter"
        )
        engine.write.add_node(build_entity_node(node_id="rust-n1", doc_id="d1", embedding=None))
        engine.write.add_node(build_entity_node(node_id="rust-n2", doc_id="d1", embedding=None))
        assert engine.backend.node_get(
            ids=["rust-n1", "rust-n2"], include=["embeddings"]
        )["embeddings"] == [None, None]
        calls: list[list[str]] = []

        def batch_provider(documents):
            calls.append(list(documents))
            return [[1.0, 0.0, 0.0] for _ in documents]

        monkeypatch.setattr(engine, "_ef", batch_provider)
        worker = engine.indexing.make_index_job_worker(
            batch_size=10, max_inflight=10, max_jobs_per_tick=10
        )
        metrics = worker.tick()
        assert metrics.claimed == 2 and metrics.done == 2
        assert len(calls) == 1 and len(calls[0]) == 2
        assert engine.backend.node_get(
            ids=["rust-n1", "rust-n2"], include=["embeddings"]
        )["embeddings"] == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        engine.write.add_node(build_entity_node(node_id="rust-stale", doc_id="d1", embedding=None))
        stale_job = next(
            job for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "rust-stale" and job.index_kind == "node_embedding"
        )
        replacement = build_entity_node(node_id="rust-stale", doc_id="d2", embedding=None)
        engine.write.add_node(replacement)
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=stale_job.entity_kind,
            entity_id=stale_job.entity_id,
            op=stale_job.op,
            payload_json=stale_job.payload_json,
        )
        assert engine.backend.node_get(ids=["rust-stale"], include=["embeddings"])["embeddings"] == [None]

        engine.write.add_node(build_entity_node(node_id="rust-delete", doc_id="d1", embedding=None))
        upsert_job = next(
            job for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "rust-delete" and job.index_kind == "node_embedding"
        )
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=upsert_job.entity_kind,
            entity_id=upsert_job.entity_id,
            op=upsert_job.op,
            payload_json=upsert_job.payload_json,
        )
        assert engine.backend.node_get(
            ids=["rust-delete"], include=["embeddings"]
        )["embeddings"][0] is not None
        assert engine.lifecycle.tombstone_node("rust-delete") is True
        delete_job = next(
            job for job in engine.meta_sqlite.list_index_jobs(namespace="default")
            if job.entity_id == "rust-delete"
            and job.index_kind == "node_embedding"
            and job.op == "DELETE"
        )
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=delete_job.entity_kind,
            entity_id=delete_job.entity_id,
            op=delete_job.op,
            payload_json=delete_job.payload_json,
        )
        deleted = engine.backend.node_get(
            ids=["rust-delete"], include=["embeddings", "metadatas"]
        )
        assert deleted["ids"] == ["rust-delete"]
        assert deleted["embeddings"] == [None]
        assert deleted["metadatas"][0]["lifecycle_status"] == "tombstoned"
        semantic = engine.backend.node_query(
            query_embeddings=[[1.0, 0.0, 0.0]], n_results=10
        )
        assert "rust-delete" not in semantic["ids"][0]

        engine.write.add_node(build_entity_node(node_id="rust-repair", doc_id="d1", embedding=None))
        claimed = engine.meta_sqlite.claim_index_jobs(
            limit=10, lease_seconds=60, namespace="default"
        )
        repair_job = next(job for job in claimed if job.entity_id == "rust-repair")
        engine.meta_sqlite.mark_index_job_done(
            repair_job.job_id, claim_token=repair_job.claim_token
        )
        assert engine.indexing.repair_missing_two_stage_embedding_jobs(max_entities=10) >= 1
        repaired = engine.indexing.make_index_job_worker(
            batch_size=10, max_inflight=10, max_jobs_per_tick=10
        ).tick()
        assert repaired.done >= 1
        assert engine.backend.node_get(ids=["rust-repair"], include=["embeddings"])["embeddings"][0] is not None
    finally:
        engine.close()
