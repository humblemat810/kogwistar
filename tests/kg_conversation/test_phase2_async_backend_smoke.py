from __future__ import annotations

from dataclasses import replace

import pytest

pytest_plugins = ["tests.core._async_chroma_real"]

from kogwistar.conversation.conversation_orchestrator import (# noqa: E402
    ConversationOrchestrator,
    get_id_for_conversation_turn,
)
from kogwistar.conversation.models import (# noqa: E402
    ConversationNode,
    MetaFromLastSummary,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine# noqa: E402
from kogwistar.engine_core.models import Grounding, Span# noqa: E402
from kogwistar.id_provider import stable_id# noqa: E402
from kogwistar.llm_tasks.contracts import SummarizeContextTaskResult# noqa: E402
from tests.conftest import _install_conversation_policy, _run_async_windows_safe # noqa: E402
from tests.core._async_chroma_real import make_real_async_chroma_backend # noqa: E402

pytestmark = pytest.mark.ci


class _AsyncSmokeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "phase2-async-smoke"

    def __call__(self, documents_or_texts):
        return [[0.0, 0.0, 0.0] for _ in documents_or_texts]


def _mk_span(doc_id: str) -> Span:
    span = Span.from_dummy_for_conversation()
    span.doc_id = doc_id
    return span


def _noop_filtering_callback(*_args, **_kwargs):
    from kogwistar.conversation.models import FilteringResult

    return FilteringResult(node_ids=[], edge_ids=[]), "noop"


def _build_pg_engine_pair(
    *,
    request: pytest.FixtureRequest,
    tmp_path,
    dim: int = 3,
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    async_sa_engine = request.getfixturevalue("async_sa_engine")
    async_pg_schema = request.getfixturevalue("async_pg_schema")
    backend_cls = type(request.getfixturevalue("async_pg_backend"))

    kg_backend = backend_cls(
        engine=async_sa_engine,
        embedding_dim=dim,
        schema=f"{async_pg_schema}_kg",
    )
    conv_backend = backend_cls(
        engine=async_sa_engine,
        embedding_dim=dim,
        schema=f"{async_pg_schema}_conv",
    )
    kg_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_pg_kg"),
        kg_graph_type="knowledge",
        embedding_function=_AsyncSmokeEmbeddingFunction(),
        backend=kg_backend,
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_pg_conv"),
        kg_graph_type="conversation",
        embedding_function=_AsyncSmokeEmbeddingFunction(),
        backend=conv_backend,
    )
    _install_conversation_policy(conv_engine)
    kg_engine._phase1_enable_index_jobs = False
    conv_engine._phase1_enable_index_jobs = False
    return kg_engine, conv_engine


async def _build_chroma_engine_pair(
    *,
    request: pytest.FixtureRequest,
    tmp_path,
    dim: int = 3,
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    pytest.importorskip("chromadb")
    real_chroma_server = request.getfixturevalue("real_chroma_server")
    _, kg_backend, _ = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="phase2_async_kg"
    )
    _, conv_backend, _ = await make_real_async_chroma_backend(
        real_chroma_server, collection_prefix="phase2_async_conv"
    )
    kg_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_chroma_kg"),
        kg_graph_type="knowledge",
        embedding_function=_AsyncSmokeEmbeddingFunction(),
        backend_factory=lambda _engine, backend=kg_backend: backend,
    )
    conv_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "async_chroma_conv"),
        kg_graph_type="conversation",
        embedding_function=_AsyncSmokeEmbeddingFunction(),
        backend_factory=lambda _engine, backend=conv_backend: backend,
    )
    _install_conversation_policy(conv_engine)
    kg_engine._phase1_enable_index_jobs = False
    conv_engine._phase1_enable_index_jobs = False
    return kg_engine, conv_engine


def _build_engine_pair(
    *, backend_kind: str, request: pytest.FixtureRequest, tmp_path
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine]:
    if backend_kind == "pg":
        return _build_pg_engine_pair(request=request, tmp_path=tmp_path)
    if backend_kind == "chroma":
        return _run_async_windows_safe(
            _build_chroma_engine_pair(request=request, tmp_path=tmp_path)
        )
    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


def _mk_turn(
    *,
    conversation_id: str,
    user_id: str,
    turn_id: str,
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
            "conversation_id": conversation_id,
            "user_id": user_id,
            "level_from_root": 0,
        },
        role=role,  # type: ignore[arg-type]
        turn_index=turn_index,
        conversation_id=conversation_id,
        user_id=user_id,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        level_from_root=0,
    )


def _seed_turns(
    orch: ConversationOrchestrator, *, conversation_id: str, user_id: str
) -> None:
    for idx, text in enumerate(["hi", "tell me X", "ok thanks"], start=0):
        nid = get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conversation_id,
            text,
            str(idx),
            "user",
            "conversation_turn",
            "True",
        )
        orch.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=nid,
            mem_id=f"mem-{idx}",
            role="user",
            content=text,
            filtering_callback=_noop_filtering_callback,
            add_turn_only=True,
        )


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("chroma", id="async-chroma"),
        pytest.param("pg", id="async-pg"),
    ],
)
def test_phase2c_seq_stamping_async_backends(
    backend_kind: str, tmp_path, request: pytest.FixtureRequest
) -> None:
    _kg, conv = _build_engine_pair(
        backend_kind=backend_kind, request=request, tmp_path=tmp_path
    )
    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=_kg,
        tool_call_id_factory=stable_id,
    )

    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t1",
        mem_id="m1",
        role="user",
        content="hello",
        filtering_callback=_noop_filtering_callback, # type: ignore
        add_turn_only=True,
    )
    first = conv.get_nodes()[-1]
    assert first.metadata["run_step_seq"] == 1
    assert first.metadata["attempt_seq"] == 0

    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t2",
        mem_id="m1",
        role="user",
        content="world",
        filtering_callback=_noop_filtering_callback, # type: ignore
        add_turn_only=True,
    )
    second = conv.get_nodes()[-1]
    assert second.metadata["run_step_seq"] == 2
    assert second.metadata["attempt_seq"] == 0


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("chroma", id="async-chroma"),
        pytest.param("pg", id="async-pg"),
    ],
)
def test_phase2a_accounting_async_backends(
    backend_kind: str, tmp_path, request: pytest.FixtureRequest
) -> None:
    kg, conv = _build_engine_pair(
        backend_kind=backend_kind, request=request, tmp_path=tmp_path
    )
    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        tool_call_id_factory=stable_id,
    )

    orch.create_conversation(user_id="u1", conv_id="c1")
    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t1",
        mem_id="m1",
        role="user",
        content="x" * 100,
        filtering_callback=_noop_filtering_callback, # type: ignore
        add_turn_only=True,
    )

    nodes = conv.get_nodes()
    assert "char_distance_from_last_summary" not in nodes[-1].metadata
    assert "turn_distance_from_last_summary" not in nodes[-1].metadata


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("chroma", id="async-chroma"),
        pytest.param("pg", id="async-pg"),
    ],
)
def test_phase2b_context_snapshot_async_backends(
    backend_kind: str, tmp_path, request: pytest.FixtureRequest
) -> None:
    _kg, conv = _build_engine_pair(
        backend_kind=backend_kind, request=request, tmp_path=tmp_path
    )
    conv.tool_call_id_factory = stable_id
    conv.llm_tasks = replace(
        conv.llm_tasks,
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
    )
    orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=_kg,
        tool_call_id_factory=stable_id,
    )

    conversation_id = "phase2b_c1"
    user_id = "u1"
    orch.create_conversation(user_id=user_id, conv_id=conversation_id)
    _seed_turns(orch, conversation_id=conversation_id, user_id=user_id)

    prev = MetaFromLastSummary(
        prev_node_char_distance_from_last_summary=0,
        prev_node_distance_from_last_summary=0,
        tail_turn_index=2,
    )
    orch._summarize_conversation_batch(
        conversation_id=conversation_id,
        current_index=2,
        batch_size=3,
        in_conv=True,
        user_id=user_id,
        prev_turn_meta_summary=prev,
    )

    snaps = conv.get_nodes(where={"entity_type": "context_snapshot"})
    assert snaps, "expected at least one context_snapshot node"
    snap = snaps[-1]
    assert snap.metadata.get("in_conversation_chain") is False
    assert snap.metadata.get("in_ui_chain") is False
    assert int(snap.metadata.get("run_step_seq", 0)) >= 0
    assert "cost.char_count" in snap.metadata
    assert "cost.token_count" in snap.metadata


def test_phase2d_pg_meta_sql_rollback_async_pg_only(
    tmp_path, request: pytest.FixtureRequest
) -> None:
    kg, conv = _build_engine_pair(
        backend_kind="pg", request=request, tmp_path=tmp_path
    )
    _orch = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        tool_call_id_factory=stable_id,
    )

    rollback_turn = _mk_turn(
        conversation_id="phase2d_pg",
        user_id="u1",
        turn_id="turn_rollback",
        role="user",
        turn_index=0,
    )

    with pytest.raises(RuntimeError):
        with conv.uow():
            conv.add_node(rollback_turn)
            raise RuntimeError("boom")

    assert conv.get_nodes(where={"id": "turn_rollback"}) == []
