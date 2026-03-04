import json
from pathlib import Path

import pytest

from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings

from joblib import Memory
from typing import Callable, Sequence, Any, ParamSpec, TypeVar, cast

from conversation.models import ConversationAIResponse, FilteringResult, MetaFromLastSummary
from graph_knowledge_engine.cdc.oplog import OplogWriter
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from engine_core.models import (
    Node,
    Span,
    Grounding,
    MentionVerification,
)
from graph_knowledge_engine.id_provider import stable_id
from graph_knowledge_engine.postgres_backend import PgVectorBackend


# -----------------------
# Minimal embedding func
# -----------------------
class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in documents_or_texts]


def _make_engine_pair(*, backend_kind: str, tmp_path, sa_engine, pg_schema, dim: int = 384):
    if backend_kind == "chroma":
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "kg"),
            kg_graph_type="knowledge",
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )
        return kg_engine, conv_engine

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip("pg backend requested but sa_engine/pg_schema fixtures not available")
        kg_schema = f"{pg_schema}_kg"
        conv_schema = f"{pg_schema}_conv"
        kg_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=kg_schema)
        conv_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=conv_schema)
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "kg_meta"),
            kg_graph_type="knowledge",
            embedding_function=FakeEmbeddingFunction(dim=dim),
            backend=kg_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv_meta"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(dim=dim),
            backend=conv_backend,
        )
        return kg_engine, conv_engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


def _mk_span(doc_id: str, excerpt: str) -> Span:
    return Span(
        doc_id=doc_id,
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes="test:apple",
        ),
        collection_page_url="url",
        document_page_url="url",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=len(excerpt),
        excerpt=excerpt,
        context_before="",
        context_after="",
    )


def _add_apple_knowledge(kg_engine: GraphKnowledgeEngine, *, dim: int = 384):
    """Add a tiny Apple fruit knowledge snippet.

    We deliberately add both nodes and (if supported by your models) edges/hyperedges.
    The test only *requires* nodes; edges are best-effort to avoid coupling.
    """

    apple = Node(
        id="K:apple",
        label="Apple (fruit)",
        type="entity",
        summary="Apple is a fruit.",
        mentions=[Grounding(spans=[_mk_span("D:apple", "Apple is a fruit.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    fruit = Node(
        id="K:fruit",
        label="Fruit",
        type="entity",
        summary="Fruit is a kind of food.",
        mentions=[Grounding(spans=[_mk_span("D:fruit", "Fruit is a kind of food.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    sweet = Node(
        id="K:sweet",
        label="Sweet",
        type="entity",
        summary="Sweet is a taste property.",
        mentions=[Grounding(spans=[_mk_span("D:sweet", "Sweet is a taste property.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )

    kg_engine.add_node(apple)
    kg_engine.add_node(fruit)
    kg_engine.add_node(sweet)

    # Best-effort: add edges/hyperedges if your repo exposes them in models.
    try:
        from engine_core.models import Edge  # type: ignore

        is_a = Edge(
            id="E:is_a",
            label="is_a",
            type="relationship",
            summary="Apple is a Fruit",
            source_ids=[apple.id],
            target_ids=[fruit.id],
            relation="is_a",
            doc_id="D:apple",
            domain_id=None,
            canonical_entity_id=None,
            properties=None,
            embedding=None,
            mentions=[Grounding(spans=[_mk_span("D:apple", "Apple is a fruit.")])],
            metadata={"level_from_root": 0},
            source_edge_ids=[],
            target_edge_ids=[],
        )
        has_prop = Edge(
            id="E:has_property",
            label="has_property",
            type="relationship",
            summary="Apple is Sweet",
            source_ids=[apple.id],
            target_ids=[sweet.id],
            relation="has_property",
            doc_id="D:apple",
            domain_id=None,
            canonical_entity_id=None,
            properties=None,
            embedding=None,
            mentions=[Grounding(spans=[_mk_span("D:apple", "Apples can taste sweet.")])],
            metadata={"level_from_root": 0},
            source_edge_ids=[],
            target_edge_ids=[],
        )
        kg_engine.add_edge(is_a)
        kg_engine.add_edge(has_prop)

        # Hyperedge best-effort: if Edge supports multi-target, create one.
        # This remains optional and should not fail the test if the model rejects it.
        try:
            hyper = Edge(
                id="E:apple_fruit_sweet",
                label="apple_fruit_sweet",
                type="relationship",
                summary="Apple is a fruit and sweet",
                source_ids=[apple.id],
                target_ids=[fruit.id, sweet.id],
                relation="hyperedge",
                doc_id="D:apple",
                domain_id=None,
                canonical_entity_id=None,
                properties=None,
                embedding=None,
                mentions=[Grounding(spans=[_mk_span("D:apple", "Apple is a fruit and sweet.")])],
                metadata={"level_from_root": 0},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            kg_engine.add_edge(hyper)
        except Exception:
            pass
    except Exception:
        pass


# -----------------------
# Joblib caching helpers
# -----------------------
P = ParamSpec("P")
R = TypeVar("R")


def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    """Typed wrapper over joblib.Memory.cache"""
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))


def _deterministic_filter_impl(
    conversation_content: str,
    candidates_node_ids: list[str],
    candidate_edge_ids: list[str],
) -> tuple[dict, str]:
    """Serializable deterministic filtering (cache-friendly).

    IMPORTANT: accepts only JSON-serializable inputs.
    """
    text = (conversation_content or "").lower()
    if "apple" in text:
        fr = FilteringResult(
            node_ids=["K:apple", "K:fruit", "K:sweet"],
            edge_ids=["E:is_a", "E:has_property"],
        )
        return fr.model_dump(), "deterministic: matched apple"
    fr = FilteringResult(node_ids=[], edge_ids=[])
    return fr.model_dump(), "deterministic: no match"


def _deterministic_answer_impl(
    *,
    conversation_id: str,
    prev_turn_meta_summary_dump: dict,
    used_kg_node_ids: list[str],
    used_kg_edge_ids: list[str],
) -> dict:
    """Serializable deterministic answer function (cache-friendly).

    IMPORTANT: accepts only JSON-serializable inputs.
    """
    # Super simple: if apple knowledge was selected, answer accordingly.
    if "K:apple" in used_kg_node_ids:
        text = "Apple is a fruit, and it is often sweet."
    else:
        text = "I don't know."
    return {
        "assistant_text": text,
        "llm_decision_need_summary": False,
        "used_kg_node_ids": used_kg_node_ids,
        "used_kg_edge_ids": used_kg_edge_ids,
        # response_node_id is set by orchestrator's normal assistant-turn creation.
    }


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_conversation_flow_v2_end_to_end_cached_llm(backend_kind: str, tmp_path, sa_engine, pg_schema):
    """True E2E: call ConversationOrchestrator.add_conversation_turn_workflow_v2.

    - No direct WorkflowRuntime.run calls.
    - No AgenticAnsweringAgent patching.
    - We *do* harness the orchestrator by monkeypatching its .answer_only to a cached
      deterministic function, in the same style as test_conversation_flow.py.
    """

    kg_engine, conversation_engine = _make_engine_pair(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        dim=384,
    )

    # Wire oplogs so CDC/harness tooling can observe changes if enabled.
    bundle_dir = Path(tmp_path) / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    kg_engine._oplog = OplogWriter(bundle_dir / "kg_changes.jsonl", fsync=False)
    conversation_engine._oplog = OplogWriter(bundle_dir / "conv_changes.jsonl", fsync=False)

    # Deterministic factories.
    conversation_engine.tool_call_id_factory = stable_id

    # Build orchestrator.
    user_id = "test_conversation_flow_v2::user"
    orc = conversation_engine._get_orchestrator(ref_knowledge_engine=kg_engine)
    orc.tool_runner.tool_call_id_factory = stable_id

    # Add apple knowledge.
    _add_apple_knowledge(kg_engine, dim=384)

    # Create conversation.
    conv_id = "conv_v2_test_001"
    start_node_id = "start_con_node_v2"
    conversation_engine.create_conversation(user_id, conv_id, start_node_id)

    # Cache layer (joblib) like your existing test style.
    memory = Memory(location=str(tmp_path / ".joblib"))

    # Filtering callback: cached, ignores non-deterministic args by not accepting them.
    cached_filter = cached(memory, _deterministic_filter_impl)

    def filtering_callback(*args, **kwargs):
        # Follow the engine callback shape loosely; only extract serializable bits.
        conversation_content = kwargs.get("conversation_content")
        if conversation_content is None and len(args) >= 2:
            conversation_content = args[1]
        candidates_node_ids = kwargs.get("candidates_node_ids") or kwargs.get("candidates_node_ids", [])
        candidate_edge_ids = kwargs.get("candidate_edge_ids") or kwargs.get("candidate_edge_ids", [])
        dumped, reasoning = cached_filter(
            str(conversation_content or ""),
            list(candidates_node_ids or []),
            list(candidate_edge_ids or []),
        )
        return FilteringResult.model_validate(dumped), reasoning

    # Answer-only harness: cached + ignores non-serializable state.
    cached_answer = cached(memory, _deterministic_answer_impl)

    def answer_only_harness(*, conversation_id: str, prev_turn_meta_summary: MetaFromLastSummary, **_):
        # Pull used ids from last retrieval artifacts if present; keep deterministic.
        # In v2, relevant ids are stored on the conversation via pointer nodes, but
        # the resolver state includes `kg.selected`/etc. For a harness, we rely on
        # the deterministic filter selecting the apple ids.
        used_nodes = ["K:apple", "K:fruit", "K:sweet"]
        used_edges = ["E:is_a", "E:has_property"]
        out = cached_answer(
            conversation_id=str(conversation_id),
            prev_turn_meta_summary_dump=prev_turn_meta_summary.model_dump(),
            used_kg_node_ids=list(used_nodes),
            used_kg_edge_ids=list(used_edges),
        )
        return ConversationAIResponse(
            text=str(out["assistant_text"]),
            llm_decision_need_summary=bool(out.get("llm_decision_need_summary", False)),
            used_kg_node_ids=list(out.get("used_kg_node_ids") or []),
            projected_conversation_node_ids=[],
            meta={"cached": True},
            response_node_id=None,
        )

    # Monkeypatch orchestrator (harness).
    orc.answer_only = answer_only_harness

    # Execute v2.
    res = orc.add_conversation_turn_workflow_v2(
        run_id="run_v2_e2e",
        user_id=user_id,
        conversation_id=conv_id,
        turn_id="turn_v2_001",
        mem_id="mem0",
        role="user",
        content="What is an apple? Is it sweet?",
        filtering_callback=filtering_callback,
        workflow_id="workflow_v2_test",
        in_conv=True,
        add_turn_only=False,
        max_workers=1,
    )

    # Assert: user turn exists.
    assert res.user_turn_node_id
    user_turn = conversation_engine.backend.node_get(ids=[res.user_turn_node_id])
    assert user_turn["ids"], "user turn node missing"
    user_doc = json.loads(user_turn["documents"][0])
    assert user_doc["role"] == "user"
    assert "apple" in (user_doc["summary"] or "").lower()

    # Assert: assistant exists and has fruit/sweet.
    assert res.response_turn_node_id
    asst = conversation_engine.backend.node_get(ids=[res.response_turn_node_id])
    assert asst["ids"], "assistant turn node missing"
    asst_doc = json.loads(asst["documents"][0])
    assert asst_doc["role"] == "assistant"
    assert "fruit" in (asst_doc["summary"] or "").lower()
    assert "sweet" in (asst_doc["summary"] or "").lower()

    # Assert: relevant KG ids are carried through.
    assert "K:apple" in (res.relevant_kg_node_ids or [])
