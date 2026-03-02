
import json
from pathlib import Path
from typing import Callable, Sequence, Any, TypeVar, ParamSpec, cast

import pytest
from joblib import Memory

from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.id_provider import stable_id
from graph_knowledge_engine.postgres_backend import PgVectorBackend
from graph_knowledge_engine.models import (
    Node,
    Span,
    Grounding,
    MentionVerification,
    FilteringResult,
    MetaFromLastSummary,
)
from graph_knowledge_engine.conversation_orchestrator import ConversationOrchestrator

# Optional: knowledge-edge model may not exist in some repo versions.
try:
    from graph_knowledge_engine.models import Edge  # type: ignore
except Exception:  # pragma: no cover
    Edge = None  # type: ignore

P = ParamSpec("P")
R = TypeVar("R")


def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))


class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 8):
        self._dim = dim

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in documents_or_texts]


class DeterministicLLM(BaseChatModel):
    """
    A tiny deterministic LLM stub that returns a fixed AIMessage content.
    This is "fake" vs "real" only in the sense that:
      - fake mode: we pass a deterministic filtering callback (no prompting dependency)
      - real mode: we pass candiate_filtering_callback (prompting) which uses this LLM.
    """

    def __init__(self, content: str):
        super().__init__()
        self._content = content

    @property
    def _llm_type(self) -> str:  # pragma: no cover
        return "deterministic"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return AIMessage(content=self._content)


def _make_engine(*, backend_kind: str, tmp_path: Path, sa_engine, pg_schema: str | None, graph_type: str, dim: int):
    engine = None
    if backend_kind == "chroma":
        engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / graph_type),
            kg_graph_type=graph_type,
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip("pg backend requested but sa_engine/pg_schema fixtures not available")
        schema = f"{pg_schema}_{graph_type}"
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=schema)
        engine =  GraphKnowledgeEngine(
            persist_directory=str(tmp_path / f"{graph_type}_meta"),
            kg_graph_type=graph_type,
            embedding_function=FakeEmbeddingFunction(dim=dim),
            backend=backend,
        )
    if engine:
        engine.tool_call_id_factory = stable_id
        return engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


def _mk_span(excerpt: str) -> Span:
    return Span(
        doc_id="D:test",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test"),
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


def add_knowledge_apple(kg: GraphKnowledgeEngine, *, dim: int) -> dict[str, Any]:
    """Pluggable knowledge set: apple fruit."""
    apple = Node(
        id="K:apple",
        label="Apple",
        type="entity",
        summary="Apple is a fruit.",
        mentions=[Grounding(spans=[_mk_span("Apple is a fruit.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "entity"},
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    fruit = Node(
        id="K:fruit",
        label="Fruit",
        type="entity",
        summary="Fruit is a kind of food.",
        mentions=[Grounding(spans=[_mk_span("Fruit is food.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "entity"},
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    sweet = Node(
        id="K:sweet",
        label="Sweet",
        type="entity",
        summary="Sweet is a taste.",
        mentions=[Grounding(spans=[_mk_span("Sweet is a taste.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "entity"},
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    kg.add_node(apple)
    kg.add_node(fruit)
    kg.add_node(sweet)

    if Edge is not None:
        try:
            kg.add_edge(
                Edge(
                    id="E:is_a",
                    label="is_a",
                    type="relationship",
                    summary="Apple is a Fruit",
                    source_ids=["K:apple"],
                    target_ids=["K:fruit"],
                    relation="is_a",
                    doc_id="D:test",
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "edge"},
                    embedding=None,
                    mentions=[Grounding(spans=[_mk_span("Apple is a fruit.")])],
                    metadata={"level_from_root": 0},
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
            )
            kg.add_edge(
                Edge(
                    id="E:has_property",
                    label="has_property",
                    type="relationship",
                    summary="Apple can taste sweet",
                    source_ids=["K:apple"],
                    target_ids=["K:sweet"],
                    relation="has_property",
                    doc_id="D:test",
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "edge"},
                    embedding=None,
                    mentions=[Grounding(spans=[_mk_span("Apple can taste sweet.")])],
                    metadata={"level_from_root": 0},
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
            )
        except Exception:
            # Keep test resilient if Edge signature differs across repo versions.
            pass

    return {
        "question": "What is an apple? Is it sweet?",
        "expect_terms": ["fruit", "sweet"],
        "expect_node_ids": {"K:apple"},
    }


def add_knowledge_banana(kg: GraphKnowledgeEngine, *, dim: int) -> dict[str, Any]:
    """Pluggable knowledge set: banana fruit."""
    banana = Node(
        id="K:banana",
        label="Banana",
        type="entity",
        summary="Banana is a fruit.",
        mentions=[Grounding(spans=[_mk_span("Banana is a fruit.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "entity"},
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    fruit = Node(
        id="K:fruit",
        label="Fruit",
        type="entity",
        summary="Fruit is a kind of food.",
        mentions=[Grounding(spans=[_mk_span("Fruit is food.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "entity"},
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    yellow = Node(
        id="K:yellow",
        label="Yellow",
        type="entity",
        summary="Yellow is a color.",
        mentions=[Grounding(spans=[_mk_span("Yellow is a color.")])],
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "entity"},
        embedding=[0.1] * dim,
        doc_id=None,
        level_from_root=0,
    )
    kg.add_node(banana)
    kg.add_node(fruit)
    kg.add_node(yellow)

    if Edge is not None:
        try:
            kg.add_edge(
                Edge(
                    id="E:is_a",
                    label="is_a",
                    type="relationship",
                    summary="Banana is a Fruit",
                    source_ids=["K:banana"],
                    target_ids=["K:fruit"],
                    relation="is_a",
                    doc_id="D:test",
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "edge"},
                    embedding=None,
                    mentions=[Grounding(spans=[_mk_span("Banana is a fruit.")])],
                    metadata={"level_from_root": 0},
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
            )
            kg.add_edge(
                Edge(
                    id="E:has_color",
                    label="has_color",
                    type="relationship",
                    summary="Banana is yellow",
                    source_ids=["K:banana"],
                    target_ids=["K:yellow"],
                    relation="has_color",
                    doc_id="D:test",
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "edge"},
                    embedding=None,
                    mentions=[Grounding(spans=[_mk_span("Banana is yellow.")])],
                    metadata={"level_from_root": 0},
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
            )
        except Exception:
            pass

    return {
        "question": "What is a banana? What color is it?",
        "expect_terms": ["fruit", "yellow"],
        "expect_node_ids": {"K:banana"},
    }


def _deterministic_filter_impl(*, question: str, knowledge_key: str) -> dict[str, Any]:
    """
    Cacheable: only accepts simple serializable inputs.
    Returns a dict for FilteringResult.model_validate.
    """
    q = question.lower()
    if knowledge_key == "apple" and "apple" in q:
        return {"node_ids": ["K:apple", "K:fruit", "K:sweet"], "edge_ids": ["E:is_a", "E:has_property"]}
    if knowledge_key == "banana" and "banana" in q:
        return {"node_ids": ["K:banana", "K:fruit", "K:yellow"], "edge_ids": ["E:is_a", "E:has_color"]}
    return {"node_ids": [], "edge_ids": []}


def _deterministic_answer_impl(*, question: str, knowledge_key: str) -> dict[str, Any]:
    """
    Cacheable: only accepts simple serializable inputs.
    Returns a dict used by the answer_only harness.
    """
    q = question.lower()
    if knowledge_key == "apple" and "apple" in q:
        return {"content": "Apple is a fruit, and it is often sweet.", "need_summary": False}
    if knowledge_key == "banana" and "banana" in q:
        return {"content": "Banana is a fruit, and it is typically yellow.", "need_summary": False}
    return {"content": "I don't know.", "need_summary": False}


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
@pytest.mark.parametrize("llm_mode", ["fake", "real"])
@pytest.mark.parametrize(
    "knowledge_builder,knowledge_key",
    [
        (add_knowledge_apple, "apple"),
        (add_knowledge_banana, "banana"),
    ],
)
def test_conversation_flow_v2_param_e2e(
    backend_kind: str,
    llm_mode: str,
    knowledge_builder: Callable[[GraphKnowledgeEngine], dict[str, Any]],
    knowledge_key: str,
    tmp_path: Path,
    sa_engine,
    pg_schema,
):
    """
    True E2E (v2 entrypoint): orchestrator.add_conversation_turn_workflow_v2

    Product over:
      - backend: chroma vs pg
      - llm_mode: fake vs "real" (prompt-based filtering callback)
      - knowledge sets: apple, banana (pluggable)
    """
    dim = 8
    kg = _make_engine(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, graph_type="knowledge", dim=dim)
    conv = _make_engine(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, graph_type="conversation", dim=dim)
    wf = _make_engine(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, graph_type="workflow", dim=dim)

    # Seed knowledge
    meta = knowledge_builder(kg, dim=dim)
    question = meta["question"]

    # Deterministic LLM: used only in "real" filtering mode and any summarization decisions.
    llm = DeterministicLLM(content="['K:apple']" if knowledge_key == "apple" else "['K:banana']")

    orc = ConversationOrchestrator(
        conversation_engine=conv,
        ref_knowledge_engine=kg,
        workflow_engine=wf,
        llm=llm,
    )

    # cache folder: per-test temp to avoid cross-test pollution
    memory = Memory(location=str(tmp_path / ".joblib"), verbose=0)
    cached_filter = cached(memory, _deterministic_filter_impl)
    cached_answer = cached(memory, _deterministic_answer_impl)

    # Filtering callback
    if llm_mode == "fake":
        def filtering_callback(_llm, conversation_content, cand_node_list_str, cand_edge_list_str, candidates_node_ids, candidate_edge_ids, context_text):
            dumped = cached_filter(question=conversation_content, knowledge_key=knowledge_key)
            return FilteringResult.model_validate(dumped), "cached deterministic filter"
    else:
        # Use the project's canonical prompting-based filter, but cache it (ignore llm).
        from graph_knowledge_engine.engine import candiate_filtering_callback
        def cached_inner(llm: BaseChatModel, conversation_content, cand_node_list_str, cand_edge_list_str, candidates_node_ids, candidate_edge_ids, context_text):
            fr, reason = candiate_filtering_callback(llm, conversation_content, cand_node_list_str, cand_edge_list_str, candidates_node_ids, candidate_edge_ids, context_text)
            return fr.model_dump(), reason
        cached_fn = cached(memory, cached_inner, ignore=["llm"])
        def filtering_callback(llm: BaseChatModel, conversation_content, cand_node_list_str, cand_edge_list_str, candidates_node_ids, candidate_edge_ids, context_text):
            dumped, reason = cached_fn(llm, conversation_content, cand_node_list_str, cand_edge_list_str, candidates_node_ids, candidate_edge_ids, context_text)
            return FilteringResult.model_validate(dumped), reason

    # Answer harness (cached, serializable-only inner)
    def answer_only_harness(*, conversation_id: str, prev_turn_meta_summary: MetaFromLastSummary, user_text: str, **_):
        payload = cached_answer(question=user_text, knowledge_key=knowledge_key)
        # Use the project's conversation engine API for assistant turn creation if available
        # We keep this harness minimal: the v2 orchestrator expects a ConversationAIResponse-like object.
        # If your repo defines ConversationAIResponse, use it directly.
        try:
            from graph_knowledge_engine.models import ConversationAIResponse
            return ConversationAIResponse(
                response_node_id=None,  # will be filled by engine call below if available
                llm_decision_need_summary=bool(payload.get("need_summary", False)),
                content=payload["content"],
            )
        except Exception:
            class _Resp:
                response_node_id = None
                llm_decision_need_summary = bool(payload.get("need_summary", False))
                content = payload["content"]
            return _Resp()

    # Monkeypatch: orchestrator v2 still takes answer_only from deps;
    # once Phase B2/B3 is complete, remove this and let agentic path drive.
    orc.answer_only = answer_only_harness  # type: ignore[attr-defined]

    user_id = "u:test"
    conv_id = f"conv_v2_{backend_kind}_{llm_mode}_{knowledge_key}"
    turn_id = f"turn_v2_{backend_kind}_{llm_mode}_{knowledge_key}"
    run_id = f"run_v2_{backend_kind}_{llm_mode}_{knowledge_key}"

    conv.create_conversation(user_id, conv_id, "start_node_v2")

    res = orc.add_conversation_turn_workflow_v2(
        run_id=run_id,
        user_id=user_id,
        conversation_id=conv_id,
        turn_id=turn_id,
        mem_id="mem0",
        role="user",
        content=question,
        filtering_callback=filtering_callback,
        workflow_id="conversation.add_turn.v2.test",
        in_conv=True,
        add_turn_only=False,
    )

    # Verify: user turn exists
    assert res.user_turn_node_id
    got_user = conv.backend.node_get(ids=[res.user_turn_node_id])
    assert got_user["ids"], "user turn node missing"
    user_doc = json.loads(got_user["documents"][0])
    assert user_doc["role"] == "user"
    assert question[:5].lower() in user_doc["summary"].lower()

    # Verify: relevant KG ids contain expected
    for nid in meta["expect_node_ids"]:
        assert nid in set(res.relevant_kg_node_ids), f"expected KG node id {nid} to be referenced"

    # Verify: assistant response exists (if your pipeline creates it)
    if res.response_turn_node_id is not None:
        got_ai = conv.backend.node_get(ids=[res.response_turn_node_id])
        assert got_ai["ids"], "assistant turn node missing"
        ai_doc = json.loads(got_ai["documents"][0])
        assert ai_doc["role"] == "assistant"
        for term in meta["expect_terms"]:
            assert term in ai_doc["summary"].lower()
