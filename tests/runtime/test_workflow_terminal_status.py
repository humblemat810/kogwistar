from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import shutil
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.engine_core.models import (
    Grounding,
    MentionVerification,
    Span,
)
from kogwistar.runtime.models import (
    RunSuccess,
    WorkflowCompletedNode,
    WorkflowEdge,
    WorkflowNode,
)
from kogwistar.runtime.runtime import WorkflowRuntime
from tests._helpers.fake_backend import build_fake_backend
from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    real_chroma_server,
)


class FakeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.is_legacy = False

    def __call__(self, input):
        return [[0.01] * self._dim for _ in input]


def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id,
        type="entity",
        doc_id=node_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    )


def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": 100,
            "wf_is_default": True,
            "wf_predicate": None,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


ASYNC_BACKEND_PARAMS = [
    pytest.param("chroma", id="async-chroma", marks=pytest.mark.ci_full),
    pytest.param("pg", id="async-pg", marks=pytest.mark.ci_full),
]


class _AsyncFakeEmbeddingFunction3D:
    @staticmethod
    def name() -> str:
        return "async-fake-3d"

    async def __call__(self, documents_or_texts):
        vectors: list[list[float]] = []
        for text in documents_or_texts:
            base = float(len(str(text)) % 7 + 1)
            vectors.append([base, base + 1.0, base + 2.0])
        return vectors


async def _create_schema_async(async_sa_engine, schema: str) -> None:
    import sqlalchemy as sa

    async with async_sa_engine.begin() as conn:
        await conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


async def _drop_schema_async(async_sa_engine, schema: str) -> None:
    import sqlalchemy as sa

    async with async_sa_engine.begin() as conn:
        await conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@asynccontextmanager
async def _async_runtime_engine_pair(backend_kind: str, request, tmp_path):
    embedding_function = _AsyncFakeEmbeddingFunction3D()
    if backend_kind == "chroma":
        pytest.importorskip("chromadb")
        server = request.getfixturevalue("real_chroma_server")
        wf_client, wf_backend, _ = await make_real_async_chroma_backend(
            server, collection_prefix="terminal_async_wf"
        )
        conv_client, conv_backend, _ = await make_real_async_chroma_backend(
            server, collection_prefix="terminal_async_conv"
        )
        _ = wf_client, conv_client
        wf_engine = GraphKnowledgeEngine(
            persist_directory=str(server.persist_dir),
            kg_graph_type="workflow",
            embedding_function=embedding_function,
            backend_factory=lambda _engine: wf_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(server.persist_dir),
            kg_graph_type="conversation",
            embedding_function=embedding_function,
            backend_factory=lambda _engine: conv_backend,
        )
        yield wf_engine, conv_engine
        return

    if backend_kind == "pg":
        pytest.importorskip("sqlalchemy")
        async_sa_engine = request.getfixturevalue("async_sa_engine")
        if async_sa_engine is None:
            pytest.skip("async pg fixtures are unavailable")
        base_schema = f"gke_async_terminal_{uuid.uuid4().hex}"
        wf_schema = f"{base_schema}_wf"
        conv_schema = f"{base_schema}_conv"
        await _create_schema_async(async_sa_engine, wf_schema)
        await _create_schema_async(async_sa_engine, conv_schema)
        try:
            wf_backend = PgVectorBackend(
                engine=async_sa_engine, embedding_dim=3, schema=wf_schema
            )
            conv_backend = PgVectorBackend(
                engine=async_sa_engine, embedding_dim=3, schema=conv_schema
            )
            await wf_backend._ensure_schema_async()
            await conv_backend._ensure_schema_async()
            wf_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "async_wf_terminal"),
                kg_graph_type="workflow",
                embedding_function=embedding_function,
                backend=wf_backend,
            )
            conv_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "async_conv_terminal"),
                kg_graph_type="conversation",
                embedding_function=embedding_function,
                backend=conv_backend,
            )
            yield wf_engine, conv_engine
        finally:
            await _drop_schema_async(async_sa_engine, wf_schema)
            await _drop_schema_async(async_sa_engine, conv_schema)
        return

    raise ValueError(f"unsupported backend_kind: {backend_kind}")


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
    ],
)
def test_runtime_persists_completed_terminal_for_leaf_node(backend_kind):
    root = Path(".tmp_runtime_completed_terminal") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
        if backend_kind == "fake":
            workflow_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "wf"),
                kg_graph_type="workflow",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            )
            conversation_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "conv"),
                kg_graph_type="conversation",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            )
        else:
            workflow_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "wf"),
                kg_graph_type="workflow",
                embedding_function=ef,
            )
            conversation_engine = GraphKnowledgeEngine(
                persist_directory=str(root / "conv"),
                kg_graph_type="conversation",
                embedding_function=ef,
            )

        workflow_id = "wf_runtime_leaf_terminal"
        conversation_id = "conv_runtime_leaf_terminal"
        start = _wf_node(
            workflow_id=workflow_id, node_id="wf|start", op="start", start=True
        )
        leaf = _wf_node(workflow_id=workflow_id, node_id="wf|leaf", op="leaf")
        workflow_engine.write.add_node(start)
        workflow_engine.write.add_node(leaf)
        workflow_engine.write.add_edge(
            _wf_edge(
                workflow_id=workflow_id,
                edge_id="wf|start->leaf",
                src=start.id,
                dst=leaf.id,
            )
        )

        def resolve_step(op: str):
            def _fn(ctx):
                return RunSuccess(
                    conversation_node_id=None, state_update=[("a", {"last_op": op})]
                )

            return _fn

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolve_step,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
        )
        result = runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id="turn-leaf",
            initial_state={},
        )

        assert result.status == "succeeded"
        completed = conversation_engine.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_completed"},
                    {"run_id": result.run_id},
                ]
            },
            limit=10,
        )
        assert len(completed) == 1
        assert isinstance(completed[0], WorkflowCompletedNode)
        meta = completed[0].metadata or {}
        assert meta.get("last_processed_node_id") == f"wf_step|{result.run_id}|1"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_runtime_persists_completed_terminal_for_leaf_node_async_backends(
    backend_kind, request, tmp_path
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        workflow_engine,
        conversation_engine,
    ):
        workflow_id = "wf_runtime_leaf_terminal_async"
        conversation_id = "conv_runtime_leaf_terminal_async"
        start = _wf_node(
            workflow_id=workflow_id, node_id="wf|start", op="start", start=True
        )
        leaf = _wf_node(workflow_id=workflow_id, node_id="wf|leaf", op="leaf")
        await asyncio.to_thread(workflow_engine.write.add_node, start)
        await asyncio.to_thread(workflow_engine.write.add_node, leaf)
        await asyncio.to_thread(
            workflow_engine.write.add_edge,
            _wf_edge(
                workflow_id=workflow_id,
                edge_id="wf|start->leaf",
                src=start.id,
                dst=leaf.id,
            ),
        )

        def resolve_step(op: str):
            def _fn(ctx):
                return RunSuccess(
                    conversation_node_id=None, state_update=[("a", {"last_op": op})]
                )

            return _fn

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolve_step,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
        )
        result = await asyncio.to_thread(
            runtime.run,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id="turn-leaf",
            initial_state={},
        )

        assert result.status == "succeeded"
        completed = await asyncio.to_thread(
            conversation_engine.get_nodes,
            where={
                "$and": [
                    {"entity_type": "workflow_completed"},
                    {"run_id": result.run_id},
                ]
            },
            limit=10,
        )
        assert len(completed) == 1
        assert isinstance(completed[0], WorkflowCompletedNode)
        meta = completed[0].metadata or {}
        assert meta.get("last_processed_node_id") == f"wf_step|{result.run_id}|1"
