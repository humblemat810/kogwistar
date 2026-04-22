from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import uuid

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.runtime import AsyncWorkflowRuntime, WorkflowRuntime
from kogwistar.runtime.models import RunSuccess, WorkflowCompletedNode, WorkflowEdge, WorkflowNode
from kogwistar.runtime.replay import load_checkpoint, replay_to
from kogwistar.runtime.resolvers import AsyncMappingStepResolver, MappingStepResolver
from tests._helpers.fake_backend import build_fake_backend
from tests.conftest import (
    FakeEmbeddingFunction,
    _is_missing_pgvector_extension,
    _make_engine_pair,
    _make_workflow_engine,
)
from tests.core._async_chroma_real import make_real_async_chroma_backend, real_chroma_server


SYNC_BACKEND_PARAMS = [
    pytest.param("fake", id="memory-sync", marks=pytest.mark.ci),
    pytest.param("chroma", id="chroma-sync", marks=pytest.mark.ci_full),
    pytest.param("pg", id="pg-sync", marks=pytest.mark.ci_full),
]

ASYNC_BACKEND_PARAMS = [
    pytest.param("fake", id="memory-async", marks=pytest.mark.ci),
    pytest.param("chroma", id="chroma-async", marks=pytest.mark.ci_full),
    pytest.param("pg", id="pg-async", marks=pytest.mark.ci_full),
]


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
            method="human",
            is_verified=True,
            score=1.0,
            notes="test",
        ),
    )


def _grounding() -> Grounding:
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
        mentions=[_grounding()],
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
        mentions=[_grounding()],
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


def _seed_linear_workflow(workflow_engine: GraphKnowledgeEngine, workflow_id: str) -> None:
    start = _wf_node(
        workflow_id=workflow_id,
        node_id="wf|start",
        op="start",
        start=True,
    )
    end = _wf_node(
        workflow_id=workflow_id,
        node_id="wf|end",
        op="end",
        terminal=True,
    )
    workflow_engine.write.add_node(start)
    workflow_engine.write.add_node(end)
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id="wf|start->end",
            src=start.safe_get_id(),
            dst=end.safe_get_id(),
        )
    )


def _build_sync_engine_pair(*, backend_kind: str, tmp_path, request):
    ef = FakeEmbeddingFunction(dim=8)
    sa_engine = None
    pg_schema = None
    if backend_kind == "pg":
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
    workflow_engine = _make_workflow_engine(
        backend_kind=backend_kind,
        tmp_path=tmp_path / "workflow",
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=8,
        embedding_function=ef,
    )
    _kg_engine, conversation_engine = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path / "conversation",
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=8,
        embedding_function=ef,
    )
    return workflow_engine, conversation_engine


async def _create_schema_async(async_sa_engine, schema: str) -> None:
    import sqlalchemy as sa

    async with async_sa_engine.begin() as conn:
        await conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


async def _drop_schema_async(async_sa_engine, schema: str) -> None:
    import sqlalchemy as sa

    async with async_sa_engine.begin() as conn:
        await conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@asynccontextmanager
async def _build_async_engine_pair(*, backend_kind: str, request, tmp_path):
    ef = FakeEmbeddingFunction(dim=3)
    if backend_kind == "fake":
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf"),
            kg_graph_type="workflow",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv"),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        yield workflow_engine, conversation_engine
        return

    if backend_kind == "chroma":
        server = request.getfixturevalue("real_chroma_server")
        wf_client, wf_backend, _ = await make_real_async_chroma_backend(
            server,
            collection_prefix=f"runtime_smoke_wf_{uuid.uuid4().hex}",
        )
        conv_client, conv_backend, _ = await make_real_async_chroma_backend(
            server,
            collection_prefix=f"runtime_smoke_conv_{uuid.uuid4().hex}",
        )
        _ = wf_client, conv_client
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(server.persist_dir),
            kg_graph_type="workflow",
            embedding_function=ef,
            backend_factory=lambda _engine: wf_backend,
        )
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(server.persist_dir),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend_factory=lambda _engine: conv_backend,
        )
        yield workflow_engine, conversation_engine
        return

    if backend_kind == "pg":
        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
        if sa_engine is None or pg_schema is None:
            pytest.skip("pg fixtures are unavailable")
        try:
            workflow_engine = _make_workflow_engine(
                backend_kind="pg",
                tmp_path=tmp_path / "workflow",
                sa_engine=sa_engine,
                pg_schema=f"{pg_schema}_async_runtime",
                dim=3,
                embedding_function=ef,
            )
            _kg_engine, conversation_engine = _make_engine_pair(
                backend_kind="pg",
                tmp_path=tmp_path / "conversation",
                sa_engine=sa_engine,
                pg_schema=f"{pg_schema}_async_runtime",
                dim=3,
                embedding_function=ef,
            )
        except Exception as exc:
            if _is_missing_pgvector_extension(exc):
                pytest.skip(f"pg backend unavailable: {exc}")
            raise
        yield workflow_engine, conversation_engine
        return

    raise ValueError(f"unsupported backend_kind: {backend_kind}")


def _normalize_smoke_result(
    *,
    backend_kind: str,
    runtime_kind: str,
    result,
    checkpoint_state: dict,
    replayed_state: dict,
    completed_nodes: list[WorkflowCompletedNode],
) -> dict:
    rt_join = dict(checkpoint_state.get("_rt_join", {}) or {})
    pending = list(rt_join.get("pending", []) or [])
    completed_meta = (completed_nodes[0].metadata or {}) if completed_nodes else {}
    return {
        "backend": backend_kind,
        "runtime": runtime_kind,
        "status": str(result.status),
        "final_op_log": list((result.final_state or {}).get("op_log", []) or []),
        "checkpoint_op_log": list(checkpoint_state.get("op_log", []) or []),
        "checkpoint_pending_nodes": [
            str(item[0]) for item in pending if isinstance(item, (list, tuple)) and item
        ],
        "replay_op_log": list(replayed_state.get("op_log", []) or []),
        "replay_result_end": dict(replayed_state.get("result.end", {}) or {}),
        "completed_terminal_count": len(completed_nodes),
        "completed_last_step": str(completed_meta.get("last_processed_node_id") or ""),
    }


def _expected_smoke_result(*, backend_kind: str, runtime_kind: str) -> dict:
    return {
        "backend": backend_kind,
        "runtime": runtime_kind,
        "status": "succeeded",
        "final_op_log": ["start", "end"],
        "checkpoint_op_log": ["start"],
        "checkpoint_pending_nodes": ["wf|end"],
        "replay_op_log": ["start", "end"],
        "replay_result_end": {
            "value": f"{runtime_kind}:{backend_kind}:end",
        },
        "completed_terminal_count": 1,
        "completed_last_step": "suffix|1",
    }


def _assert_smoke_result(actual: dict, *, backend_kind: str, runtime_kind: str) -> None:
    expected = _expected_smoke_result(backend_kind=backend_kind, runtime_kind=runtime_kind)
    last_step = str(actual.pop("completed_last_step"))
    assert last_step.endswith("|1")
    expected.pop("completed_last_step")
    assert actual == expected


@pytest.mark.parametrize("backend_kind", SYNC_BACKEND_PARAMS)
def test_backend_e2e_smoke_matrix_sync_runtime(backend_kind, request, tmp_path):
    """Backend smoke: sync runtime preserves one-step checkpoint and terminal shape across memory, chroma, and pg."""
    workflow_engine, conversation_engine = _build_sync_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        request=request,
    )
    workflow_id = f"wf_backend_smoke_sync_{backend_kind}_{uuid.uuid4().hex}"
    conversation_id = f"conv_backend_smoke_sync_{backend_kind}_{uuid.uuid4().hex}"
    run_id = f"run_backend_smoke_sync_{backend_kind}_{uuid.uuid4().hex}"
    _seed_linear_workflow(workflow_engine, workflow_id)

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("a", {"op_log": "start"}),
                ("u", {"result.start": {"value": f"sync:{backend_kind}:start"}}),
            ],
        )

    @resolver.register("end")
    def _end(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("a", {"op_log": "end"}),
                ("u", {"result.end": {"value": f"sync:{backend_kind}:end"}}),
            ],
        )

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=9999,
        max_workers=1,
    )
    result = runtime.run(
        workflow_id=workflow_id,
        conversation_id=conversation_id,
        turn_node_id="turn-smoke-sync",
        initial_state={},
        run_id=run_id,
    )
    checkpoint_state = load_checkpoint(
        conversation_engine=conversation_engine,
        run_id=run_id,
        step_seq=0,
    )
    replayed_state = replay_to(
        conversation_engine=conversation_engine,
        run_id=run_id,
        target_step_seq=1,
    )
    completed_nodes = conversation_engine.read.get_nodes(
        where={
            "$and": [
                {"entity_type": "workflow_completed"},
                {"run_id": run_id},
            ]
        },
        limit=10,
    )
    assert len(completed_nodes) == 1
    assert isinstance(completed_nodes[0], WorkflowCompletedNode)
    normalized = _normalize_smoke_result(
        backend_kind=backend_kind,
        runtime_kind="sync",
        result=result,
        checkpoint_state=checkpoint_state,
        replayed_state=replayed_state,
        completed_nodes=completed_nodes,
    )
    _assert_smoke_result(normalized, backend_kind=backend_kind, runtime_kind="sync")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_backend_e2e_smoke_matrix_async_runtime(backend_kind, request, tmp_path):
    """Backend smoke: async runtime preserves one-step checkpoint and terminal shape across memory, chroma, and pg."""
    async with _build_async_engine_pair(
        backend_kind=backend_kind,
        request=request,
        tmp_path=tmp_path,
    ) as (workflow_engine, conversation_engine):
        workflow_id = f"wf_backend_smoke_async_{backend_kind}_{uuid.uuid4().hex}"
        conversation_id = f"conv_backend_smoke_async_{backend_kind}_{uuid.uuid4().hex}"
        run_id = f"run_backend_smoke_async_{backend_kind}_{uuid.uuid4().hex}"
        await asyncio.to_thread(_seed_linear_workflow, workflow_engine, workflow_id)

        resolver = AsyncMappingStepResolver()

        @resolver.register("start")
        async def _start(_ctx):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("a", {"op_log": "start"}),
                    ("u", {"result.start": {"value": f"async:{backend_kind}:start"}}),
                ],
            )

        @resolver.register("end")
        async def _end(_ctx):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("a", {"op_log": "end"}),
                    ("u", {"result.end": {"value": f"async:{backend_kind}:end"}}),
                ],
            )

        runtime = AsyncWorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=9999,
            max_workers=1,
            experimental_native_scheduler=True,
        )
        result = await runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id="turn-smoke-async",
            initial_state={},
            run_id=run_id,
        )
        checkpoint_state = await asyncio.to_thread(
            load_checkpoint,
            conversation_engine=conversation_engine,
            run_id=run_id,
            step_seq=0,
        )
        replayed_state = await asyncio.to_thread(
            replay_to,
            conversation_engine=conversation_engine,
            run_id=run_id,
            target_step_seq=1,
        )
        completed_nodes = await asyncio.to_thread(
            conversation_engine.read.get_nodes,
            where={
                "$and": [
                    {"entity_type": "workflow_completed"},
                    {"run_id": run_id},
                ]
            },
            limit=10,
        )
        assert len(completed_nodes) == 1
        assert isinstance(completed_nodes[0], WorkflowCompletedNode)
        normalized = _normalize_smoke_result(
            backend_kind=backend_kind,
            runtime_kind="async",
            result=result,
            checkpoint_state=checkpoint_state,
            replayed_state=replayed_state,
            completed_nodes=completed_nodes,
        )
        _assert_smoke_result(normalized, backend_kind=backend_kind, runtime_kind="async")
