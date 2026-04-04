from contextlib import asynccontextmanager
import asyncio
import threading

import pytest
pytestmark = pytest.mark.core
import json
from pathlib import Path
import uuid
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    RunSuspended,
    WorkflowEdge,
    WorkflowNode,
)
from kogwistar.runtime.runtime import WorkflowRuntime, StepContext
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.sandbox import SandboxRequest
from tests.conftest import FakeEmbeddingFunction, _is_missing_pgvector_extension
from tests._helpers.fake_backend import build_fake_backend
from tests.core._async_chroma_real import (
    make_real_async_chroma_backend,
    make_real_async_chroma_uow,
    real_chroma_server,
)

from kogwistar.engine_core.models import Span, Grounding

from kogwistar.engine_core.models import (
    MentionVerification,
)


def _get_dummy_grounding():
    sp = Span(
        collection_page_url="dummy",
        document_page_url="dummy",
        doc_id="dummy",
        insertion_method="dummy",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="a",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system", is_verified=True, score=1.0, notes=""
        ),
    )
    return Grounding(spans=[sp])


def _create_node(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
    wf_join: bool = False,
):
    engine.write.add_node(
        WorkflowNode(
            id=node_id,
            label=op,
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            mentions=[_get_dummy_grounding()],
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": wf_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
                "wf_fanout": fanout,
                "wf_join": wf_join,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _create_edge(
    engine: GraphKnowledgeEngine,
    wf_id: str,
    src: str,
    dst: str,
    *,
    label: str = "wf_next",
    predicate: str | None = None,
    is_default: bool = True,
):
    engine.write.add_edge(
        WorkflowEdge(
            id=f"{src}->{dst}",
            label=label,
            type="entity",
            doc_id=f"{src}->{dst}",
            summary="next",
            properties={},
            source_ids=[src],
            target_ids=[dst],
            source_edge_ids=[],
            target_edge_ids=[],
            relation="wf_next",
            mentions=[_get_dummy_grounding()],
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": wf_id,
                "wf_predicate": predicate,
                "wf_is_default": is_default,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )


def _latest_checkpoint_state(conv_engine: GraphKnowledgeEngine, run_id: str) -> dict:
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
    state_json = latest.metadata.get("state_json", {})
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    return dict(state_json or {})


def _workflow_step_exec_nodes(conv_engine: GraphKnowledgeEngine, run_id: str):
    return conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]}
    )


def _workflow_edge_by_id(conv_engine: GraphKnowledgeEngine, edge_id: str):
    edges = conv_engine.get_edges(ids=[edge_id])
    assert edges, f"Missing workflow edge {edge_id}"
    return edges[0]


def _runtime_entity_event_seq(engine: GraphKnowledgeEngine) -> int:
    namespace = str(getattr(engine, "namespace", "default") or "default")
    getter = getattr(engine.meta_sqlite, "get_latest_entity_event_seq", None)
    if callable(getter):
        return int(getter(namespace=namespace))
    return sum(
        1
        for _ in engine.meta_sqlite.iter_entity_events(namespace=namespace, from_seq=1)
    )


def _log_workflow_entity_event_debug(
    *,
    wf_engine: GraphKnowledgeEngine,
    wf_id: str,
    label: str,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    namespace = str(getattr(wf_engine, "namespace", "default") or "default")
    getter = getattr(wf_engine.meta_sqlite, "get_latest_entity_event_seq", None)
    iter_events = getattr(wf_engine.meta_sqlite, "iter_entity_events", None)
    counts: dict[str, int | None] = {"namespace": None, "workflow_id": None}
    sample: list[dict[str, object]] = []
    if callable(getter):
        counts["namespace"] = int(getter(namespace=namespace))
        counts["workflow_id"] = int(getter(namespace=wf_id))
    if callable(iter_events):
        for seq, entity_kind, entity_id, op, _payload in iter_events(
            namespace=namespace, from_seq=1
        ):
            sample.append(
                {
                    "seq": int(seq),
                    "entity_kind": str(entity_kind),
                    "entity_id": str(entity_id),
                    "op": str(op),
                }
            )
            if len(sample) >= 5:
                break
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "label": label,
                    "engine_namespace": namespace,
                    "workflow_id_arg": wf_id,
                    "counts": counts,
                    "sample": sample,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _build_branch_failure_join_workflow(
    wf_engine: GraphKnowledgeEngine,
    wf_id: str,
    *,
    route_on_failure: bool,
) -> None:
    _create_node(wf_engine, wf_id, "start", "start_op", start=True, fanout=True)
    _create_node(wf_engine, wf_id, "fail", "fail_op")
    _create_node(wf_engine, wf_id, "downstream", "downstream_op")
    _create_node(wf_engine, wf_id, "b", "branch_b_op")
    _create_node(wf_engine, wf_id, "join", "join_op", wf_join=True)
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)

    _create_edge(wf_engine, wf_id, "start", "fail")
    _create_edge(wf_engine, wf_id, "start", "b")
    _create_edge(
        wf_engine,
        wf_id,
        "fail",
        "downstream",
        label="recover_on_failure",
        predicate="if_failure" if route_on_failure else "if_not_failure",
        is_default=False,
    )
    _create_edge(wf_engine, wf_id, "downstream", "join")
    _create_edge(wf_engine, wf_id, "b", "join")
    _create_edge(wf_engine, wf_id, "join", "end")


def _branch_failure_join_resolver():
    class _IfFailure:
        def __call__(self, e, state, result):
            return getattr(result, "status", None) == "failure"

    class _IfNotFailure:
        def __call__(self, e, state, result):
            return getattr(result, "status", None) != "failure"

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("fail_op")
    def _fail(ctx: StepContext):
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"failed_once": True})],
            errors=["boom"],
        )

    @resolver.register("downstream_op")
    def _downstream(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"downstream_done": True})],
        )

    @resolver.register("branch_b_op")
    def _branch_b(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"b_done": True})],
        )

    @resolver.register("join_op")
    def _join(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"joined": True})],
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"ended": True})],
        )

    return resolver, _IfFailure(), _IfNotFailure()


def _build_deep_failure_join_workflow(
    wf_engine: GraphKnowledgeEngine, wf_id: str
) -> None:
    _create_node(wf_engine, wf_id, "start", "start_op", start=True, fanout=True)
    _create_node(wf_engine, wf_id, "b", "branch_b_op")
    _create_node(wf_engine, wf_id, "c", "branch_c_op", fanout=True)
    _create_node(wf_engine, wf_id, "d", "branch_d_op")
    _create_node(wf_engine, wf_id, "e", "branch_e_op")
    _create_node(wf_engine, wf_id, "f", "fail_op")
    _create_node(wf_engine, wf_id, "join", "join_op", wf_join=True)
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)

    _create_edge(wf_engine, wf_id, "start", "b")
    _create_edge(wf_engine, wf_id, "start", "c")
    _create_edge(wf_engine, wf_id, "c", "d")
    _create_edge(wf_engine, wf_id, "c", "e")
    _create_edge(wf_engine, wf_id, "b", "join")
    _create_edge(wf_engine, wf_id, "d", "join")
    _create_edge(wf_engine, wf_id, "e", "f")
    _create_edge(wf_engine, wf_id, "f", "end")


def _deep_failure_join_resolver():
    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("branch_b_op")
    def _branch_b(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"b_done": True})]
        )

    @resolver.register("branch_c_op")
    def _branch_c(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"c_done": True})]
        )

    @resolver.register("branch_d_op")
    def _branch_d(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"d_done": True})]
        )

    @resolver.register("branch_e_op")
    def _branch_e(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"e_done": True})]
        )

    @resolver.register("fail_op")
    def _fail(ctx: StepContext):
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"failed_once": True})],
            errors=["boom"],
        )

    @resolver.register("join_op")
    def _join(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"joined": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    return resolver


def _make_engine(
    tmp_path, *, graph_type: str, backend_kind: str
) -> GraphKnowledgeEngine:
    if backend_kind == "fake":
        return GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            kg_graph_type=graph_type,
            embedding_function=FakeEmbeddingFunction(),
            backend_factory=build_fake_backend,
        )
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type=graph_type,
        embedding_function=FakeEmbeddingFunction(),
    )


BACKEND_PARAMS = [
    pytest.param("fake", id="fake", marks=pytest.mark.ci),
    pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
]

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
            server, collection_prefix="runtime_async_wf"
        )
        conv_client, conv_backend, _ = await make_real_async_chroma_backend(
            server, collection_prefix="runtime_async_conv"
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
        base_schema = f"gke_async_runtime_{uuid.uuid4().hex}"
        wf_schema = f"{base_schema}_wf"
        conv_schema = f"{base_schema}_conv"
        await _create_schema_async(async_sa_engine, wf_schema)
        await _create_schema_async(async_sa_engine, conv_schema)
        try:
            try:
                wf_backend = PgVectorBackend(
                    engine=async_sa_engine, embedding_dim=3, schema=wf_schema
                )
                conv_backend = PgVectorBackend(
                    engine=async_sa_engine, embedding_dim=3, schema=conv_schema
                )
            except Exception as exc:
                if _is_missing_pgvector_extension(exc):
                    pytest.skip(f"async pg backend unavailable: {exc}")
                raise
            await wf_backend._ensure_schema_async()
            await conv_backend._ensure_schema_async()
            wf_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "async_wf"),
                kg_graph_type="workflow",
                embedding_function=embedding_function,
                backend=wf_backend,
            )
            conv_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "async_conv"),
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
    BACKEND_PARAMS,
)
def test_workflow_suspend_and_resume(tmp_path, backend_kind):
    if backend_kind == "fake":
        wf_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(),
            backend_factory=build_fake_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(),
            backend_factory=build_fake_backend,
        )
    else:
        wf_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(),
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(),
        )

    wf_id = "test_suspend_wf"

    # start -> do_suspend -> end
    _create_node(wf_engine, wf_id, "n_start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "n_suspend", "suspend_op")
    _create_node(wf_engine, wf_id, "n_end", "end_op", terminal=True)

    _create_edge(wf_engine, wf_id, "n_start", "n_suspend")
    _create_edge(wf_engine, wf_id, "n_suspend", "n_end")

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "calculate_pi"},
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"

    # 1. Run until suspension
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )

    assert res1.status == "suspended"
    assert res1.final_state.get("started") is True
    assert res1.final_state.get("ended") is None

    # Check that it actually persisted a checkpoint with pending tokens
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    assert len(ckpts) > 0
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
    state_json = latest.metadata.get("state_json", {})
    import json

    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    rt_join = state_json.get("_rt_join", {})
    suspended = rt_join.get("suspended", [])

    assert len(suspended) == 1
    assert suspended[0][0] == "n_suspend"
    suspended_token_id = suspended[0][2]

    # 2. Emulate client finishing task and providing result
    client_result = RunSuccess(
        conversation_node_id=None, state_update=[("u", {"pi": 3.14})]
    )

    # 3. Resume run
    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="n_suspend",
        suspended_token_id=suspended_token_id,
        client_result=client_result,
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("started") is True
    assert res2.final_state.get("pi") == 3.14
    assert res2.final_state.get("ended") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_suspend_and_resume_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_suspend_wf_async"
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "n_start", "start_op", True, False
        )
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "n_suspend", "suspend_op"
        )
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "n_end", "end_op", False, True
        )
        await asyncio.to_thread(
            _create_edge, wf_engine, wf_id, "n_start", "n_suspend"
        )
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "n_suspend", "n_end")

        resolver = MappingStepResolver()

        @resolver.register("start_op")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("suspend_op")
        def _suspend(ctx: StepContext):
            return RunSuspended(
                conversation_node_id=None,
                state_update=[],
                resume_payload={"task": "calculate_pi"},
            )

        @resolver.register("end_op")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
        )

        run_id = f"run_{uuid.uuid4().hex}"
        conv_id = f"conv_{uuid.uuid4().hex}"

        res1 = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=run_id,
        )

        assert res1.status == "suspended"
        assert res1.final_state.get("started") is True
        assert res1.final_state.get("ended") is None

        ckpts = await asyncio.to_thread(
            conv_engine.get_nodes,
            where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]},
        )
        assert len(ckpts) > 0
        latest = max(
            ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
        )
        state_json = latest.metadata.get("state_json", {})
        if isinstance(state_json, str):
            state_json = json.loads(state_json)
        rt_join = state_json.get("_rt_join", {})
        suspended = rt_join.get("suspended", [])
        assert len(suspended) == 1
        assert suspended[0][0] == "n_suspend"
        suspended_token_id = suspended[0][2]

        res2 = await asyncio.to_thread(
            runtime.resume_run,
            run_id=run_id,
            suspended_node_id="n_suspend",
            suspended_token_id=suspended_token_id,
            client_result=RunSuccess(
                conversation_node_id=None, state_update=[("u", {"pi": 3.14})]
            ),
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
        )

        assert res2.status == "succeeded"
        assert res2.final_state.get("started") is True
        assert res2.final_state.get("pi") == 3.14
        assert res2.final_state.get("ended") is True


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_suspend_and_resume_branching(tmp_path, backend_kind):
    wf_engine = _make_engine(tmp_path / "wf_b", graph_type="workflow", backend_kind=backend_kind)
    conv_engine = _make_engine(tmp_path / "conv_b", graph_type="conversation", backend_kind=backend_kind)

    wf_id = "test_suspend_branching_wf"

    # start -> fork (fanout) -> a (suspends) -> join -> end
    #                        \> b (normal)  /
    _create_node(wf_engine, wf_id, "start", "start", start=True)
    _create_node(wf_engine, wf_id, "fork", "noop", fanout=True)
    _create_node(wf_engine, wf_id, "a", "suspend_op")
    _create_node(wf_engine, wf_id, "b", "normal_b")
    _create_node(wf_engine, wf_id, "join", "noop", wf_join=True)
    _create_node(wf_engine, wf_id, "end", "end", terminal=True)

    _create_edge(wf_engine, wf_id, "start", "fork")
    _create_edge(wf_engine, wf_id, "fork", "a")
    _create_edge(wf_engine, wf_id, "fork", "b")
    _create_edge(wf_engine, wf_id, "a", "join")
    _create_edge(wf_engine, wf_id, "b", "join")
    _create_edge(wf_engine, wf_id, "join", "end")

    setup_seq = _runtime_entity_event_seq(conv_engine)

    resolver = MappingStepResolver()

    @resolver.register("start")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("noop")
    def _noop(ctx: StepContext):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={"task": "do_something"},
        )

    @resolver.register("normal_b")
    def _normal_b(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"b_done": True})]
        )

    @resolver.register("end")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"

    # 1. Run until suspension
    # Since branch b completes but branch a suspends, the overall run should end in suspended state
    # waiting for branch a to resume and hit the join.
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )

    assert res1.status == "suspended"
    assert res1.final_state.get("b_done") is True
    assert res1.final_state.get("ended") is None
    assert _runtime_entity_event_seq(conv_engine) > setup_seq

    # Check pending token
    ckpts = conv_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
    state_json = latest.metadata.get("state_json", {})
    import json

    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    rt_join = state_json.get("_rt_join", {})
    suspended = rt_join.get("suspended", [])

    assert len(suspended) == 1
    assert suspended[0][0] == "a"
    suspended_token_id = suspended[0][2]

    # 2. Resume
    client_result = RunSuccess(
        conversation_node_id=None, state_update=[("u", {"a_done": True})]
    )

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="a",
        suspended_token_id=suspended_token_id,
        client_result=client_result,
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("a_done") is True
    assert res2.final_state.get("b_done") is True
    assert res2.final_state.get("ended") is True
    assert _runtime_entity_event_seq(conv_engine) > setup_seq


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_suspend_and_resume_branching_async_backends(
    tmp_path, backend_kind, request, monkeypatch
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_suspend_branching_wf_async"
        debug_log = (
            Path.cwd()
            / ".tmp_runtime_sse_debug"
            / f"workflow_suspend_branching_async_{backend_kind}.jsonl"
        )
        if debug_log.exists():
            debug_log.unlink()

        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "start", "start", True, False
        )
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "fork", "noop", False, False, True)
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "a", "suspend_op")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "b", "normal_b")
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "join", "noop", False, False, False, True
        )
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "end", "end", False, True)

        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "start", "fork")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "fork", "a")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "fork", "b")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "a", "join")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "b", "join")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "join", "end")

        setup_seq = await asyncio.to_thread(_runtime_entity_event_seq, conv_engine)
        await asyncio.to_thread(
            _log_workflow_entity_event_debug,
            wf_engine=wf_engine,
            wf_id=wf_id,
            label="after_setup",
            log_path=debug_log,
        )

        def _install_event_loggers(engine, *, engine_label: str) -> None:
            orig_append = engine._append_event_for_entity
            orig_meta_append = engine.meta_sqlite.append_entity_event

            def _log_line(payload: dict[str, object]) -> None:
                debug_log.parent.mkdir(parents=True, exist_ok=True)
                with debug_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")

            def _wrap_append(*args, **kwargs):
                _log_line(
                    {
                        "label": "engine_append_event",
                        "engine": engine_label,
                        "namespace": kwargs.get("namespace"),
                        "entity_kind": kwargs.get("entity_kind"),
                        "entity_id": kwargs.get("entity_id"),
                        "op": kwargs.get("op"),
                    }
                )
                try:
                    result = orig_append(*args, **kwargs)
                except BaseException as exc:
                    _log_line(
                        {
                            "label": "engine_append_event_error",
                            "engine": engine_label,
                            "error": repr(exc),
                        }
                    )
                    raise
                _log_line(
                    {
                        "label": "engine_append_event_done",
                        "engine": engine_label,
                    }
                )
                return result

            def _wrap_meta_append(*args, **kwargs):
                _log_line(
                    {
                        "label": "meta_append_entity_event",
                        "engine": engine_label,
                        "namespace": kwargs.get("namespace"),
                        "entity_kind": kwargs.get("entity_kind"),
                        "entity_id": kwargs.get("entity_id"),
                        "op": kwargs.get("op"),
                    }
                )
                try:
                    seq = orig_meta_append(*args, **kwargs)
                except BaseException as exc:
                    _log_line(
                        {
                            "label": "meta_append_entity_event_error",
                            "engine": engine_label,
                            "error": repr(exc),
                        }
                    )
                    raise
                _log_line(
                    {
                        "label": "meta_append_entity_event_done",
                        "engine": engine_label,
                        "seq": int(seq),
                    }
                )
                return seq

            monkeypatch.setattr(engine, "_append_event_for_entity", _wrap_append)
            monkeypatch.setattr(
                engine.meta_sqlite, "append_entity_event", _wrap_meta_append
            )

        _install_event_loggers(wf_engine, engine_label="workflow")
        _install_event_loggers(conv_engine, engine_label="conversation")

        resolver = MappingStepResolver()

        @resolver.register("start")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("noop")
        def _noop(ctx: StepContext):
            return RunSuccess(conversation_node_id=None, state_update=[])

        @resolver.register("suspend_op")
        def _suspend(ctx: StepContext):
            return RunSuspended(
                conversation_node_id=None,
                state_update=[],
                resume_payload={"task": "do_something"},
            )

        @resolver.register("normal_b")
        def _normal_b(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"b_done": True})]
            )

        @resolver.register("end")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
        )

        run_id = f"run_{uuid.uuid4().hex}"
        conv_id = f"conv_{uuid.uuid4().hex}"

        res1 = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=run_id,
        )

        assert res1.status == "suspended"
        assert res1.final_state.get("b_done") is True
        assert res1.final_state.get("ended") is None
        await asyncio.to_thread(
            _log_workflow_entity_event_debug,
            wf_engine=wf_engine,
            wf_id=wf_id,
            label="after_suspend",
            log_path=debug_log,
        )
        assert await asyncio.to_thread(_runtime_entity_event_seq, conv_engine) > setup_seq

        ckpts = await asyncio.to_thread(
            conv_engine.get_nodes,
            where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]},
        )
        latest = max(
            ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
        )
        state_json = latest.metadata.get("state_json", {})
        if isinstance(state_json, str):
            state_json = json.loads(state_json)
        rt_join = state_json.get("_rt_join", {})
        suspended = rt_join.get("suspended", [])

        assert len(suspended) == 1
        assert suspended[0][0] == "a"
        suspended_token_id = suspended[0][2]

        res2 = await asyncio.to_thread(
            runtime.resume_run,
            run_id=run_id,
            suspended_node_id="a",
            suspended_token_id=suspended_token_id,
            client_result=RunSuccess(
                conversation_node_id=None, state_update=[("u", {"a_done": True})]
            ),
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
        )

        assert res2.status == "succeeded"
        assert res2.final_state.get("a_done") is True
        assert res2.final_state.get("b_done") is True
        assert res2.final_state.get("ended") is True
        await asyncio.to_thread(
            _log_workflow_entity_event_debug,
            wf_engine=wf_engine,
            wf_id=wf_id,
            label="after_resume",
            log_path=debug_log,
        )
        assert await asyncio.to_thread(_runtime_entity_event_seq, conv_engine) > setup_seq


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_does_not_route_to_terminal(tmp_path, backend_kind):
    wf_engine = _make_engine(tmp_path / "wf_fail", graph_type="workflow", backend_kind=backend_kind)
    conv_engine = _make_engine(tmp_path / "conv_fail", graph_type="conversation", backend_kind=backend_kind)

    wf_id = "test_failure_stops_routing"
    _create_node(wf_engine, wf_id, "n_start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "n_exec", "python_exec")
    _create_node(wf_engine, wf_id, "n_end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "n_start", "n_exec")
    # No outgoing edge from the failing node: unmatched failure should end the run as failure.

    class _FailingSandbox:
        def run(self, code, state, context):
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["sandbox failed"],
            )

        def close_run(self, run_id: str) -> None:
            return None

    resolver = MappingStepResolver()
    resolver.set_sandbox(_FailingSandbox())

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("python_exec", is_sandboxed=True)
    def _python_exec(ctx: StepContext):
        return SandboxRequest(
            code="result = {'state_update': [('u', {'sandbox_result': 'ok'})]}"
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "failure"
    assert res.final_state.get("started") is True
    assert res.final_state.get("ended") is None


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_waits_for_inflight_branch_drain(tmp_path, backend_kind):
    wf_engine = _make_engine(
        tmp_path / "wf_fail_drain", graph_type="workflow", backend_kind=backend_kind
    )
    conv_engine = _make_engine(
        tmp_path / "conv_fail_drain", graph_type="conversation", backend_kind=backend_kind
    )

    wf_id = "test_failure_waits_for_drain"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True, fanout=True)
    _create_node(wf_engine, wf_id, "fail", "fail_op")
    _create_node(wf_engine, wf_id, "slow", "slow_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "fail")
    _create_edge(wf_engine, wf_id, "start", "slow")
    _create_edge(wf_engine, wf_id, "slow", "end")

    slow_started = threading.Event()
    release_slow = threading.Event()
    failure_seen = threading.Event()
    result_box: dict[str, RunSuccess | RunFailure | None] = {"res": None}

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("fail_op")
    def _fail(ctx: StepContext):
        failure_seen.set()
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"failed_once": True})],
            errors=["boom"],
        )

    @resolver.register("slow_op")
    def _slow(ctx: StepContext):
        slow_started.set()
        if not release_slow.wait(timeout=10.0):
            raise AssertionError("slow branch was not released")
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"slow_done": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=4,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"

    def _run() -> None:
        result_box["res"] = runtime.run(
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=run_id,
        )

    t = threading.Thread(target=_run, name="workflow-failure-drain")
    t.start()

    assert slow_started.wait(10.0), "slow branch did not start"
    assert failure_seen.wait(10.0), "failure branch did not run"
    assert t.is_alive(), "run finished before the slow branch drained"

    release_slow.set()
    t.join(timeout=20.0)
    assert not t.is_alive(), "run did not finish"

    res = result_box["res"]
    assert res is not None
    assert res.status == "failure"
    assert res.final_state.get("started") is True
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("slow_done") is True
    assert res.final_state.get("ended") is None


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_then_downstream_before_join_is_routed_when_handled(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(
        tmp_path / "wf_fail_handled_join", graph_type="workflow", backend_kind=backend_kind
    )
    conv_engine = _make_engine(
        tmp_path / "conv_fail_handled_join",
        graph_type="conversation",
        backend_kind=backend_kind,
    )

    wf_id = "test_failure_then_downstream_before_join_handled"
    _build_branch_failure_join_workflow(wf_engine, wf_id, route_on_failure=True)
    resolver, if_failure, _if_not_failure = _branch_failure_join_resolver()

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"if_failure": if_failure},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "succeeded"
    assert res.final_state.get("started") is True
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("downstream_done") is True
    assert res.final_state.get("b_done") is True
    assert res.final_state.get("joined") is True
    assert res.final_state.get("ended") is True


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_then_downstream_before_join_is_skipped_when_unhandled(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(
        tmp_path / "wf_fail_unhandled_join",
        graph_type="workflow",
        backend_kind=backend_kind,
    )
    conv_engine = _make_engine(
        tmp_path / "conv_fail_unhandled_join",
        graph_type="conversation",
        backend_kind=backend_kind,
    )

    wf_id = "test_failure_then_downstream_before_join_unhandled"
    _build_branch_failure_join_workflow(wf_engine, wf_id, route_on_failure=False)
    resolver, _if_failure, if_not_failure = _branch_failure_join_resolver()

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"if_not_failure": if_not_failure},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "failure"
    assert res.final_state.get("started") is True
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("downstream_done") is None
    assert res.final_state.get("b_done") is True
    assert res.final_state.get("joined") is None
    assert res.final_state.get("ended") is None


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_allows_existing_join_to_finish_but_blocks_downstream(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(
        tmp_path / "wf_deep_failure_join",
        graph_type="workflow",
        backend_kind=backend_kind,
    )
    conv_engine = _make_engine(
        tmp_path / "conv_deep_failure_join",
        graph_type="conversation",
        backend_kind=backend_kind,
    )

    wf_id = "test_deep_failure_join"
    _build_deep_failure_join_workflow(wf_engine, wf_id)
    resolver = _deep_failure_join_resolver()

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "failure"
    assert res.final_state.get("started") is True
    assert res.final_state.get("b_done") is True
    assert res.final_state.get("c_done") is True
    assert res.final_state.get("d_done") is True
    assert res.final_state.get("e_done") is True
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("joined") is True
    assert res.final_state.get("ended") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_does_not_route_to_terminal_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_failure_stops_routing_async"
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "n_start", "start_op", True, False)
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "n_exec", "python_exec")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "n_end", "end_op", False, True)
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "n_start", "n_exec")

        class _FailingSandbox:
            def run(self, code, state, context):
                return RunFailure(
                    conversation_node_id=None,
                    state_update=[],
                    errors=["sandbox failed"],
                )

            def close_run(self, run_id: str) -> None:
                return None

        resolver = MappingStepResolver()
        resolver.set_sandbox(_FailingSandbox())

        @resolver.register("start_op")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("python_exec", is_sandboxed=True)
        def _python_exec(ctx: StepContext):
            return SandboxRequest(
                code="result = {'state_update': [('u', {'sandbox_result': 'ok'})]}"
            )

        @resolver.register("end_op")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
        )

        res = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=f"conv_{uuid.uuid4().hex}",
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=f"run_{uuid.uuid4().hex}",
        )

        assert res.status == "failure"
        assert res.final_state.get("started") is True
        assert res.final_state.get("ended") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_waits_for_inflight_branch_drain_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_failure_waits_for_drain_async"
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "start", "start_op", True, False, True
        )
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "fail", "fail_op")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "slow", "slow_op")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "end", "end_op", False, True)
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "start", "fail")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "start", "slow")
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "slow", "end")

        slow_started = threading.Event()
        release_slow = threading.Event()
        failure_seen = threading.Event()
        result_box: dict[str, Any] = {"res": None}

        resolver = MappingStepResolver()

        @resolver.register("start_op")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("fail_op")
        def _fail(ctx: StepContext):
            failure_seen.set()
            return RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"failed_once": True})],
                errors=["boom"],
            )

        @resolver.register("slow_op")
        def _slow(ctx: StepContext):
            slow_started.set()
            if not release_slow.wait(timeout=10.0):
                raise AssertionError("slow branch was not released")
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"slow_done": True})]
            )

        @resolver.register("end_op")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=4,
        )

        run_id = f"run_{uuid.uuid4().hex}"
        conv_id = f"conv_{uuid.uuid4().hex}"

        def _run() -> None:
            result_box["res"] = runtime.run(
                workflow_id=wf_id,
                conversation_id=conv_id,
                turn_node_id="turn_1",
                initial_state={
                    "conversation_id": "test",
                    "user_id": "test",
                    "turn_node_id": "test",
                    "turn_index": 0,
                    "role": "user",
                    "user_text": "",
                    "mem_id": "test",
                },
                run_id=run_id,
            )

        t = threading.Thread(target=_run, name="workflow-failure-drain-async")
        t.start()

        assert await asyncio.to_thread(slow_started.wait, 10.0), "slow branch did not start"
        assert await asyncio.to_thread(failure_seen.wait, 10.0), "failure branch did not run"
        assert t.is_alive(), "run finished before the slow branch drained"

        release_slow.set()
        await asyncio.to_thread(t.join, 20.0)
        assert not t.is_alive(), "run did not finish"

        res = result_box["res"]
        assert res is not None
        assert res.status == "failure"
        assert res.final_state.get("started") is True
        assert res.final_state.get("failed_once") is True
        assert res.final_state.get("slow_done") is True
        assert res.final_state.get("ended") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_then_downstream_before_join_is_routed_when_handled_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_failure_then_downstream_before_join_handled_async"
        await asyncio.to_thread(
            _build_branch_failure_join_workflow,
            wf_engine,
            wf_id,
            route_on_failure=True,
        )
        resolver, if_failure, _if_not_failure = _branch_failure_join_resolver()

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={"if_failure": if_failure},
            checkpoint_every_n_steps=1,
        )

        res = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=f"conv_{uuid.uuid4().hex}",
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=f"run_{uuid.uuid4().hex}",
        )

        assert res.status == "succeeded"
        assert res.final_state.get("started") is True
        assert res.final_state.get("failed_once") is True
        assert res.final_state.get("downstream_done") is True
        assert res.final_state.get("b_done") is True
        assert res.final_state.get("joined") is True
        assert res.final_state.get("ended") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_then_downstream_before_join_is_skipped_when_unhandled_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_failure_then_downstream_before_join_unhandled_async"
        await asyncio.to_thread(
            _build_branch_failure_join_workflow,
            wf_engine,
            wf_id,
            route_on_failure=False,
        )
        resolver, _if_failure, if_not_failure = _branch_failure_join_resolver()

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={"if_not_failure": if_not_failure},
            checkpoint_every_n_steps=1,
        )

        res = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=f"conv_{uuid.uuid4().hex}",
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=f"run_{uuid.uuid4().hex}",
        )

        assert res.status == "failure"
        assert res.final_state.get("started") is True
        assert res.final_state.get("failed_once") is True
        assert res.final_state.get("downstream_done") is None
        assert res.final_state.get("b_done") is True
        assert res.final_state.get("joined") is None
        assert res.final_state.get("ended") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_allows_existing_join_to_finish_but_blocks_downstream_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_deep_failure_join_async"
        await asyncio.to_thread(_build_deep_failure_join_workflow, wf_engine, wf_id)
        resolver = _deep_failure_join_resolver()

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
        )

        res = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=f"conv_{uuid.uuid4().hex}",
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=f"run_{uuid.uuid4().hex}",
        )

        assert res.status == "failure"
        assert res.final_state.get("started") is True
        assert res.final_state.get("b_done") is True
        assert res.final_state.get("c_done") is True
        assert res.final_state.get("d_done") is True
        assert res.final_state.get("e_done") is True
        assert res.final_state.get("failed_once") is True
        assert res.final_state.get("joined") is True
        assert res.final_state.get("ended") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_can_route_to_recovery_branch_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_failure_routes_async"
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "start", "start_op", True, False
        )
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "exec", "exec_op")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "recover", "recover_op")
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "end", "end_op", False, True
        )
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "start", "exec")
        await asyncio.to_thread(
            _create_edge,
            wf_engine,
            wf_id,
            "exec",
            "recover",
            label="recover_on_failure",
            predicate="if_failure",
            is_default=False,
        )
        await asyncio.to_thread(
            _create_edge, wf_engine, wf_id, "exec", "end", label="finish", is_default=True
        )
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "recover", "end")

        class _IfFailure:
            def __call__(self, e, state, result):
                return getattr(result, "status", None) == "failure"

        resolver = MappingStepResolver()

        @resolver.register("start_op")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("exec_op")
        def _exec(ctx: StepContext):
            return RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"failed_once": True})],
                errors=["boom"],
            )

        @resolver.register("recover_op")
        def _recover(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"recovered": True})]
            )

        @resolver.register("end_op")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={"if_failure": _IfFailure()},
            checkpoint_every_n_steps=1,
        )

        res = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=f"conv_{uuid.uuid4().hex}",
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=f"run_{uuid.uuid4().hex}",
        )

        assert res.status == "succeeded"
        assert res.final_state.get("started") is True
        assert res.final_state.get("failed_once") is True
        assert res.final_state.get("recovered") is True
        assert res.final_state.get("ended") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_workflow_failure_does_not_take_default_recovery_edge_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_failure_default_recovery_async"
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "start", "start_op", True, False
        )
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "exec", "exec_op")
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "recover", "recover_op"
        )
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "end", "end_op", False, True
        )
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "start", "exec")
        await asyncio.to_thread(
            _create_edge,
            wf_engine,
            wf_id,
            "exec",
            "recover",
            label="recover_by_default",
            is_default=True,
        )
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "recover", "end")

        resolver = MappingStepResolver()

        @resolver.register("start_op")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("exec_op")
        def _exec(ctx: StepContext):
            return RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"failed_once": True})],
                errors=["boom"],
            )

        @resolver.register("recover_op")
        def _recover(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"recovered": True})]
            )

        @resolver.register("end_op")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
        )

        res = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=f"conv_{uuid.uuid4().hex}",
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=f"run_{uuid.uuid4().hex}",
        )

        assert res.status == "failure"
        assert res.final_state.get("started") is True
        assert res.final_state.get("failed_once") is True
        assert res.final_state.get("recovered") is None
        assert res.final_state.get("ended") is None


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_can_route_to_recovery_branch(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(tmp_path / "wf_fail_route", graph_type="workflow", backend_kind=backend_kind)
    conv_engine = _make_engine(tmp_path / "conv_fail_route", graph_type="conversation", backend_kind=backend_kind)

    wf_id = "test_failure_routes"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "exec", "exec_op")
    _create_node(wf_engine, wf_id, "recover", "recover_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "exec")
    _create_edge(
        wf_engine,
        wf_id,
        "exec",
        "recover",
        label="recover_on_failure",
        predicate="if_failure",
        is_default=False,
    )
    _create_edge(wf_engine, wf_id, "exec", "end", label="finish", is_default=True)
    _create_edge(wf_engine, wf_id, "recover", "end")

    class _IfFailure:
        def __call__(self, e, state, result):
            return getattr(result, "status", None) == "failure"

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("exec_op")
    def _exec(ctx: StepContext):
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"failed_once": True})],
            errors=["boom"],
        )

    @resolver.register("recover_op")
    def _recover(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"recovered": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"if_failure": _IfFailure()},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "succeeded"
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("recovered") is True
    assert res.final_state.get("ended") is True


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_workflow_failure_does_not_take_default_recovery_edge(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(
        tmp_path / "wf_fail_default_recovery",
        graph_type="workflow",
        backend_kind=backend_kind,
    )
    conv_engine = _make_engine(
        tmp_path / "conv_fail_default_recovery",
        graph_type="conversation",
        backend_kind=backend_kind,
    )

    wf_id = "test_failure_default_recovery"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "exec", "exec_op")
    _create_node(wf_engine, wf_id, "recover", "recover_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "exec")
    _create_edge(
        wf_engine,
        wf_id,
        "exec",
        "recover",
        label="recover_by_default",
        is_default=True,
    )
    _create_edge(wf_engine, wf_id, "recover", "end")

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("exec_op")
    def _exec(ctx: StepContext):
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"failed_once": True})],
            errors=["boom"],
        )

    @resolver.register("recover_op")
    def _recover(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"recovered": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    res = runtime.run(
        workflow_id=wf_id,
        conversation_id=f"conv_{uuid.uuid4().hex}",
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=f"run_{uuid.uuid4().hex}",
    )

    assert res.status == "failure"
    assert res.final_state.get("started") is True
    assert res.final_state.get("failed_once") is True
    assert res.final_state.get("recovered") is None
    assert res.final_state.get("ended") is None


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_resume_run_failure_can_route_to_recovery_branch(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(tmp_path / "wf_resume_fail", graph_type="workflow", backend_kind=backend_kind)
    conv_engine = _make_engine(tmp_path / "conv_resume_fail", graph_type="conversation", backend_kind=backend_kind)

    wf_id = "test_resume_failure_routes"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "gate", "suspend_op")
    _create_node(wf_engine, wf_id, "recover", "recover_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "gate")
    _create_edge(
        wf_engine,
        wf_id,
        "gate",
        "recover",
        label="recover_on_failure",
        predicate="if_failure",
        is_default=False,
    )
    _create_edge(wf_engine, wf_id, "gate", "end", label="finish", is_default=True)
    _create_edge(wf_engine, wf_id, "recover", "end")

    class _IfFailure:
        def __call__(self, e, state, result):
            return getattr(result, "status", None) == "failure"

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={
                "type": "recoverable_error",
                "op": "suspend_op",
                "category": "missing_input",
                "message": "need fix",
                "errors": ["need fix"],
                "repair_payload": {"prompt": "fix it"},
            },
        )

    @resolver.register("recover_op")
    def _recover(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"failure_routed": True})]
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={"if_failure": _IfFailure()},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=suspended_token_id,
        client_result=RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"resume_failed": True})],
            errors=["still broken"],
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res1.status == "suspended"
    assert res2.status == "succeeded"
    assert res2.final_state.get("resume_failed") is True
    assert res2.final_state.get("failure_routed") is True
    assert res2.final_state.get("ended") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ASYNC_BACKEND_PARAMS)
async def test_resume_run_failure_can_route_to_recovery_branch_async_backends(
    tmp_path, backend_kind, request
):
    async with _async_runtime_engine_pair(backend_kind, request, tmp_path) as (
        wf_engine,
        conv_engine,
    ):
        wf_id = "test_resume_failure_routes_async"
        await asyncio.to_thread(
            _create_node, wf_engine, wf_id, "start", "start_op", True, False
        )
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "gate", "suspend_op")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "recover", "recover_op")
        await asyncio.to_thread(_create_node, wf_engine, wf_id, "end", "end_op", False, True)
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "start", "gate")
        await asyncio.to_thread(
            _create_edge,
            wf_engine,
            wf_id,
            "gate",
            "recover",
            label="recover_on_failure",
            predicate="if_failure",
            is_default=False,
        )
        await asyncio.to_thread(
            _create_edge, wf_engine, wf_id, "gate", "end", label="finish", is_default=True
        )
        await asyncio.to_thread(_create_edge, wf_engine, wf_id, "recover", "end")

        class _IfFailure:
            def __call__(self, e, state, result):
                return getattr(result, "status", None) == "failure"

        resolver = MappingStepResolver()

        @resolver.register("start_op")
        def _start(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"started": True})]
            )

        @resolver.register("suspend_op")
        def _suspend(ctx: StepContext):
            return RunSuspended(
                conversation_node_id=None,
                state_update=[],
                resume_payload={
                    "type": "recoverable_error",
                    "op": "suspend_op",
                    "category": "missing_input",
                    "message": "need fix",
                    "errors": ["need fix"],
                    "repair_payload": {"prompt": "fix it"},
                },
            )

        @resolver.register("recover_op")
        def _recover(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"failure_routed": True})]
            )

        @resolver.register("end_op")
        def _end(ctx: StepContext):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"ended": True})]
            )

        runtime = WorkflowRuntime(
            workflow_engine=wf_engine,
            conversation_engine=conv_engine,
            step_resolver=resolver,
            predicate_registry={"if_failure": _IfFailure()},
            checkpoint_every_n_steps=1,
        )

        run_id = f"run_{uuid.uuid4().hex}"
        conv_id = f"conv_{uuid.uuid4().hex}"
        res1 = await asyncio.to_thread(
            runtime.run,
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
            initial_state={
                "conversation_id": "test",
                "user_id": "test",
                "turn_node_id": "test",
                "turn_index": 0,
                "role": "user",
                "user_text": "",
                "mem_id": "test",
            },
            run_id=run_id,
        )
        state1 = await asyncio.to_thread(_latest_checkpoint_state, conv_engine, run_id)
        suspended_token_id = state1["_rt_join"]["suspended"][0][2]

        res2 = await asyncio.to_thread(
            runtime.resume_run,
            run_id=run_id,
            suspended_node_id="gate",
            suspended_token_id=suspended_token_id,
            client_result=RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"resume_failed": True})],
                errors=["still broken"],
            ),
            workflow_id=wf_id,
            conversation_id=conv_id,
            turn_node_id="turn_1",
        )

        assert res1.status == "suspended"
        assert res2.status == "succeeded"
        assert res2.final_state.get("resume_failed") is True
        assert res2.final_state.get("failure_routed") is True
        assert res2.final_state.get("ended") is True


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_resume_run_can_resuspend_same_token_with_updated_payload(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(tmp_path / "wf_resuspend", graph_type="workflow", backend_kind=backend_kind)
    conv_engine = _make_engine(tmp_path / "conv_resuspend", graph_type="conversation", backend_kind=backend_kind)

    wf_id = "test_resuspend"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "gate", "suspend_op")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "gate")
    _create_edge(wf_engine, wf_id, "gate", "end")

    resolver = MappingStepResolver()

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("suspend_op")
    def _suspend(ctx: StepContext):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={
                "type": "recoverable_error",
                "op": "suspend_op",
                "category": "missing_input",
                "message": "first pause",
                "errors": ["first"],
                "repair_payload": {"attempt": 1},
            },
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="gate",
        suspended_token_id=suspended_token_id,
        client_result=RunSuspended(
            conversation_node_id=None,
            state_update=[("u", {"retry_count": 1})],
            resume_payload={
                "type": "recoverable_error",
                "op": "suspend_op",
                "category": "missing_input",
                "message": "second pause",
                "errors": ["second"],
                "repair_payload": {"attempt": 2},
            },
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res1.status == "suspended"
    assert res2.status == "suspended"
    assert res2.final_state.get("retry_count") == 1

    resumed_step_edge_id = (
        f"wf_next_step_exec|{run_id}|2|last::wf_step|{run_id}|1|to::wf_step|{run_id}|2"
    )
    resumed_ckpt_edge_id = (
        f"persist_checkpoint|{run_id}|2|last::wf_step|{run_id}|2|to::wf_ckpt|{run_id}|2"
    )
    resumed_step_edge = _workflow_edge_by_id(conv_engine, resumed_step_edge_id)
    resumed_ckpt_edge = _workflow_edge_by_id(conv_engine, resumed_ckpt_edge_id)
    assert resumed_step_edge.safe_get_id() == resumed_step_edge_id
    assert resumed_step_edge.relation == "wf_next_step_exec"
    assert resumed_step_edge.source_ids == [f"wf_step|{run_id}|1"]
    assert resumed_step_edge.target_ids == [f"wf_step|{run_id}|2"]
    assert resumed_ckpt_edge.safe_get_id() == resumed_ckpt_edge_id
    assert resumed_ckpt_edge.relation == "persist_checkpoint during"
    assert resumed_ckpt_edge.source_ids == [f"wf_ckpt|{run_id}|2"]
    assert resumed_ckpt_edge.target_ids == [f"wf_step|{run_id}|2"]

    state2 = _latest_checkpoint_state(conv_engine, run_id)
    assert len((state2.get("_rt_join", {}) or {}).get("suspended", [])) == 1
    assert (state2.get("_rt_join", {}) or {}).get("suspended", [])[0][0] == "gate"

    step_execs = _workflow_step_exec_nodes(conv_engine, run_id)
    latest_step = max(
        step_execs, key=lambda n: int(getattr(n, "metadata", {}).get("step_seq", -1))
    )
    latest_result = json.loads(
        getattr(latest_step, "metadata", {}).get("result_json", "{}")
    )
    assert getattr(latest_step, "metadata", {}).get("status") == "suspended"
    assert latest_result.get("resume_payload", {}).get("message") == "second pause"


@pytest.mark.parametrize("backend_kind", BACKEND_PARAMS)
def test_sandbox_recoverable_error_can_suspend_then_resume_success(
    tmp_path, backend_kind
):
    wf_engine = _make_engine(tmp_path / "wf_sandbox_recoverable", graph_type="workflow", backend_kind=backend_kind)
    conv_engine = _make_engine(tmp_path / "conv_sandbox_recoverable", graph_type="conversation", backend_kind=backend_kind)

    wf_id = "test_sandbox_recoverable"
    _create_node(wf_engine, wf_id, "start", "start_op", start=True)
    _create_node(wf_engine, wf_id, "python_exec", "python_exec")
    _create_node(wf_engine, wf_id, "end", "end_op", terminal=True)
    _create_edge(wf_engine, wf_id, "start", "python_exec")
    _create_edge(wf_engine, wf_id, "python_exec", "end")

    class _RecoverableSandbox:
        def run(self, code, state, context):
            return RunSuspended(
                conversation_node_id=None,
                state_update=[],
                resume_payload={
                    "type": "recoverable_error",
                    "op": str(context.get("op")),
                    "category": "sandbox_code_error",
                    "message": "customer_id is not defined",
                    "errors": ["NameError: customer_id is not defined"],
                    "repair_payload": {"code": code, "state": state},
                },
            )

        def close_run(self, run_id: str) -> None:
            return None

    resolver = MappingStepResolver()
    resolver.set_sandbox(_RecoverableSandbox())

    @resolver.register("start_op")
    def _start(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"started": True})]
        )

    @resolver.register("python_exec", is_sandboxed=True)
    def _python_exec(ctx: StepContext):
        return SandboxRequest(
            code="result = {'state_update': [('u', {'sandbox_result': 'fixed'})]}"
        )

    @resolver.register("end_op")
    def _end(ctx: StepContext):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"ended": True})]
        )

    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    run_id = f"run_{uuid.uuid4().hex}"
    conv_id = f"conv_{uuid.uuid4().hex}"
    res1 = runtime.run(
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state={
            "conversation_id": "test",
            "user_id": "test",
            "turn_node_id": "test",
            "turn_index": 0,
            "role": "user",
            "user_text": "",
            "mem_id": "test",
        },  # type: ignore
        run_id=run_id,
    )
    state1 = _latest_checkpoint_state(conv_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="python_exec",
        suspended_token_id=suspended_token_id,
        client_result=RunSuccess(
            conversation_node_id=None, state_update=[("u", {"sandbox_result": "fixed"})]
        ),
        workflow_id=wf_id,
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )

    assert res1.status == "suspended"
    step_execs = _workflow_step_exec_nodes(conv_engine, run_id)
    first_suspend = next(
        n
        for n in step_execs
        if getattr(n, "metadata", {}).get("workflow_node_id") == "python_exec"
    )
    first_result = json.loads(
        getattr(first_suspend, "metadata", {}).get("result_json", "{}")
    )
    assert first_result.get("resume_payload", {}).get("type") == "recoverable_error"
    assert (
        first_result.get("resume_payload", {}).get("category") == "sandbox_code_error"
    )

    assert res2.status == "succeeded"
    assert res2.final_state.get("sandbox_result") == "fixed"
    assert res2.final_state.get("ended") is True
