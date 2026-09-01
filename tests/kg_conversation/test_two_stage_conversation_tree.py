from __future__ import annotations

import asyncio

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Edge
from kogwistar.graph_query import GraphQuery
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests.core.test_two_stage_chroma import _engine as _chroma_test_engine
from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    real_chroma_server,  # noqa: F401
)


pytestmark = pytest.mark.conversation


def _tree_nodes_and_edges():
    nodes = [
        build_entity_node(node_id="conv-root", doc_id="conv-tree"),
        build_entity_node(node_id="conv-child", doc_id="conv-tree"),
        build_entity_node(node_id="conv-leaf", doc_id="conv-tree"),
    ]
    edges = [
        build_relationship_edge(
            edge_id="conv-root-child", src="conv-root", tgt="conv-child",
            doc_id="conv-tree", relation="next_turn",
        ),
        build_relationship_edge(
            edge_id="conv-child-leaf", src="conv-child", tgt="conv-leaf",
            doc_id="conv-tree", relation="next_turn",
        ),
    ]
    return nodes, edges


def _tree_snapshot(engine: GraphKnowledgeEngine) -> dict[str, object]:
    graph = GraphQuery(engine)
    ids = [
        "conv-root", "conv-child", "conv-leaf",
        "conv-root-child", "conv-child-leaf",
    ]
    return {
        "neighbors": {
            item: {
                "nodes": sorted(graph.neighbors(item)["nodes"]),
                "edges": sorted(graph.neighbors(item)["edges"]),
            }
            for item in ids
        },
        "root_to_leaf": graph.shortest_path("conv-root", "conv-leaf"),
    }


def _engine(tmp_path, *, persistence_mode: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type="conversation",
        backend_factory=build_fake_backend,
        persistence_mode=persistence_mode,
        embedding_function=build_test_embedding_function("constant", dim=2),
    )


def _promote(engine: GraphKnowledgeEngine, entity_ids: set[str]) -> None:
    jobs = engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
    for job in jobs:
        if job.entity_id not in entity_ids:
            continue
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=job.entity_kind,
            entity_id=job.entity_id,
            op=job.op,
            payload_json=job.payload_json,
        )


def _promote_chroma_test_engine(engine, entity_ids: set[str]) -> None:
    for job in engine.indexing.jobs:
        if job["entity_id"] not in entity_ids:
            continue
        engine.two_stage_projection_adapter.apply_embedding_job(
            entity_kind=job["entity_kind"],
            entity_id=job["entity_id"],
            op=job["op"],
            payload_json=job["payload_json"],
        )


@pytest.mark.ci
@pytest.mark.unit
@pytest.mark.runtime_sync
def test_conversation_tree_single_stage_has_immediate_parent_child_links(tmp_path):
    engine = _engine(tmp_path / "single", persistence_mode="single_stage")
    try:
        nodes, edges = _tree_nodes_and_edges()
        for node in nodes:
            engine.write.add_node(node)
        for edge in edges:
            engine.write.add_edge(edge)

        snapshot = _tree_snapshot(engine)
        assert snapshot["root_to_leaf"] == [
            "conv-root", "conv-root-child", "conv-child",
            "conv-child-leaf", "conv-leaf",
        ]
        assert snapshot["neighbors"]["conv-root"]["nodes"] == ["conv-child"]
        assert snapshot["neighbors"]["conv-child"]["nodes"] == [
            "conv-leaf", "conv-root",
        ]
    finally:
        engine.close()


@pytest.mark.ci
@pytest.mark.unit
@pytest.mark.runtime_sync
def test_conversation_tree_two_stage_partial_promotion_preserves_exact_snapshot(tmp_path):
    engine = _engine(tmp_path / "two-stage", persistence_mode="two_stage")
    try:
        nodes, edges = _tree_nodes_and_edges()
        for node in nodes:
            engine.write.add_node(node)
        for edge in edges:
            engine.write.add_edge(edge)

        pending_snapshot = _tree_snapshot(engine)
        _promote(engine, {"conv-root", "conv-root-child", "conv-leaf"})
        mixed_snapshot = _tree_snapshot(engine)
        _promote(engine, {"conv-child", "conv-child-leaf"})
        ready_snapshot = _tree_snapshot(engine)

        assert mixed_snapshot == pending_snapshot
        assert ready_snapshot == pending_snapshot
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.runtime_async
@pytest.mark.ci
@pytest.mark.unit
async def test_conversation_tree_async_two_stage_partial_promotion_preserves_links(tmp_path):
    engine = _engine(tmp_path / "async-two-stage", persistence_mode="two_stage")
    try:
        nodes, edges = _tree_nodes_and_edges()
        for node in nodes:
            await engine.async_add_node(node)
        for edge in edges:
            await engine.async_add_edge(edge)

        pending_snapshot = await asyncio.to_thread(_tree_snapshot, engine)
        await asyncio.to_thread(
            _promote, engine, {"conv-root", "conv-root-child", "conv-leaf"}
        )
        mixed_snapshot = await asyncio.to_thread(_tree_snapshot, engine)
        await asyncio.to_thread(
            _promote, engine, {"conv-child", "conv-child-leaf"}
        )
        ready_snapshot = await asyncio.to_thread(_tree_snapshot, engine)

        assert mixed_snapshot == pending_snapshot
        assert ready_snapshot == pending_snapshot
    finally:
        engine.close()


@pytest.mark.ci
@pytest.mark.unit
@pytest.mark.runtime_sync
def test_chroma_sqlite_stage1_tree_survives_partial_and_full_promotion(tmp_path):
    engine, _adapter = _chroma_test_engine(tmp_path / "chroma")
    try:
        nodes, edges = _tree_nodes_and_edges()
        for node in nodes:
            engine.two_stage_projection_adapter.add_node(node)
        for edge in edges:
            engine.two_stage_projection_adapter.add_edge(edge)

        pending_snapshot = _tree_snapshot(engine)
        _promote_chroma_test_engine(
            engine, {"conv-root", "conv-root-child", "conv-leaf"}
        )
        mixed_snapshot = _tree_snapshot(engine)
        _promote_chroma_test_engine(
            engine, {"conv-child", "conv-child-leaf"}
        )
        ready_snapshot = _tree_snapshot(engine)

        assert mixed_snapshot == pending_snapshot
        assert ready_snapshot == pending_snapshot
    finally:
        del engine


class _AsyncTreeEmbeddingFunction:
    async def __call__(self, documents):
        await asyncio.sleep(0)
        return [[float(len(str(document)) % 7 + 1), 1.0, 2.0] for document in documents]


@pytest.mark.asyncio
@pytest.mark.runtime_async
@pytest.mark.ci_full
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
async def test_conversation_tree_async_chroma_preserves_links_across_promotion(
    real_chroma_server,  # noqa: F811
):
    _client, backend, _collections = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="conversation_async_tree"
    )
    engine = GraphKnowledgeEngine(
        persist_directory=str(real_chroma_server.persist_dir),
        kg_graph_type="conversation",
        backend_factory=lambda _engine: backend,
        persistence_mode="two_stage",
        embedding_function=_AsyncTreeEmbeddingFunction(),
    )
    try:
        nodes, edges = _tree_nodes_and_edges()
        for node in nodes:
            await engine.async_add_node(node)
        for edge in edges:
            await engine.async_add_edge(edge)

        staged_edges = await engine.async_two_stage_projection_adapter.stage1_query(
            entity_kind="edge"
        )
        staged_documents = [
            Edge.model_validate_json(row["payload"]["document"])
            for row in staged_edges
        ]
        assert {
            (edge.id, tuple(edge.source_ids), tuple(edge.target_ids))
            for edge in staged_documents
        } == {
            ("conv-root-child", ("conv-root",), ("conv-child",)),
            ("conv-child-leaf", ("conv-child",), ("conv-leaf",)),
        }
        pending_neighbors = await GraphQuery(engine).neighbors_async("conv-root")
        assert pending_neighbors == {
            "nodes": {"conv-child"}, "edges": {"conv-root-child"}
        }
        jobs = engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
        for job in jobs:
            if job.entity_id not in {"conv-root", "conv-root-child", "conv-leaf"}:
                continue
            await engine.async_two_stage_projection_adapter.apply_embedding_job(
                entity_kind=job.entity_kind,
                entity_id=job.entity_id,
                op=job.op,
                payload_json=job.payload_json,
            )
        mixed_neighbors = await GraphQuery(engine).neighbors_async("conv-root")
        assert mixed_neighbors == {
            "nodes": {"conv-child"}, "edges": {"conv-root-child"}
        }, mixed_neighbors

        jobs = engine.meta_sqlite.list_index_jobs(namespace=engine.namespace)
        for job in jobs:
            if job.entity_id not in {"conv-child", "conv-child-leaf"}:
                continue
            await engine.async_two_stage_projection_adapter.apply_embedding_job(
                entity_kind=job.entity_kind,
                entity_id=job.entity_id,
                op=job.op,
                payload_json=job.payload_json,
            )
        ready_snapshot = _tree_snapshot(engine)

        assert ready_snapshot["root_to_leaf"] == [
            "conv-root", "conv-root-child", "conv-child",
            "conv-child-leaf", "conv-leaf",
        ]
    finally:
        engine.close()
