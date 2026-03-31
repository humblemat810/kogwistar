import json
import re
from pathlib import Path
from typing import Callable, Sequence, Any, TypeVar, ParamSpec, cast

import pytest
from joblib import Memory

pytest.importorskip("chromadb")
pytest.importorskip("langchain_core")
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig

from kogwistar.conversation.filtering import candiate_filtering_callback
from kogwistar.conversation.models import (
    FilteringResult,
    MetaFromLastSummary,
)
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskRequest,
    FilterCandidatesTaskResult,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)
from kogwistar.id_provider import stable_id
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.engine_core.models import (
    Node,
    Span,
    Grounding,
    MentionVerification,
)

pytestmark = [
    pytest.mark.conversation,
    pytest.mark.workflow,
    pytest.mark.e2e,
]

# Optional: knowledge-edge model may not exist in some repo versions.
try:
    from kogwistar.engine_core.models import Edge  # type: ignore
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
        return cast(Embeddings, [[0.01] * self._dim for _ in documents_or_texts])


class DeterministicLLM(BaseChatModel):
    """
    A tiny deterministic LLM stub that returns a fixed AIMessage content.
    This is "fake" vs "real" only in the sense that:
      - fake mode: we pass a deterministic filtering callback (no prompting dependency)
      - real mode: we pass candiate_filtering_callback (prompting) which uses this LLM.
    """

    class _StructuredRunnable(Runnable):
        def __init__(self, llm: "DeterministicLLM", schema, include_raw: bool = False):
            self._llm = llm
            self._schema = schema
            self._include_raw = include_raw

        def invoke(
            self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
        ) -> Any:
            parsed = self._llm._structured_payload(self._schema, input=input)
            if self._include_raw:
                return {"raw": None, "parsed": parsed, "parsing_error": None}
            return parsed

        async def ainvoke(
            self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
        ) -> Any:
            return self.invoke(input, config=config, **kwargs)

    def __init__(self, content: str):
        super().__init__()
        self._content = content

    @property
    def _llm_type(self) -> str:  # pragma: no cover
        return "deterministic"

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        return DeterministicLLM._StructuredRunnable(
            self, schema=schema, include_raw=include_raw
        )

    @staticmethod
    def _flatten_prompt_text(input_obj: Any) -> str:
        if input_obj is None:
            return ""
        if isinstance(input_obj, str):
            return input_obj
        if isinstance(input_obj, dict):
            return "\n".join(f"{k}: {v}" for k, v in input_obj.items())
        if isinstance(input_obj, (list, tuple, set)):
            return "\n".join(
                DeterministicLLM._flatten_prompt_text(x) for x in input_obj
            )

        msgs = []
        to_messages = getattr(input_obj, "to_messages", None)
        if callable(to_messages):
            try:
                msgs = to_messages()
            except Exception:
                msgs = []
        if msgs:
            return "\n".join(str(getattr(m, "content", "")) for m in msgs)

        content = getattr(input_obj, "content", None)
        if content is not None:
            return str(content)
        return str(input_obj)

    @staticmethod
    def _extract_ids(text: str) -> tuple[list[str], list[str]]:
        node_ids = sorted(set(re.findall(r"K:[A-Za-z0-9_\-]+", text)))
        edge_ids = sorted(set(re.findall(r"E:[A-Za-z0-9_\-]+", text)))
        return node_ids, edge_ids

    @staticmethod
    def _score_id(id_: str, text_l: str) -> int:
        score = 0
        key = id_.split(":", 1)[-1].lower()
        key_tokens = [t for t in re.split(r"[_\-]+", key) if t]
        for t in key_tokens:
            if t and t in text_l:
                score += 2
        if "apple" in text_l and "apple" in key:
            score += 5
        if "banana" in text_l and "banana" in key:
            score += 5
        if "sweet" in text_l and "sweet" in key:
            score += 3
        if "yellow" in text_l and "yellow" in key:
            score += 3
        if "fruit" in text_l and "fruit" in key:
            score += 2
        return score

    def _heuristic_pick(self, text: str) -> tuple[list[str], list[str]]:
        text_l = text.lower()
        node_ids, edge_ids = self._extract_ids(text)

        scored_nodes = sorted(
            ((self._score_id(nid, text_l), nid) for nid in node_ids),
            key=lambda x: (-x[0], x[1]),
        )
        scored_edges = sorted(
            ((self._score_id(eid, text_l), eid) for eid in edge_ids),
            key=lambda x: (-x[0], x[1]),
        )

        picked_nodes = [nid for s, nid in scored_nodes if s > 0][:8]
        picked_edges = [eid for s, eid in scored_edges if s > 0][:8]

        # deterministic fallback: still pick plausible non-empty candidates when present
        if not picked_nodes and node_ids:
            picked_nodes = node_ids[: min(3, len(node_ids))]
        if not picked_edges and edge_ids:
            picked_edges = edge_ids[: min(2, len(edge_ids))]
        return picked_nodes, picked_edges

    def _build_answer_with_citations(self, text: str, schema):
        text_l = text.lower()
        # evidence pack lines are like: NODE <id> | ...
        evidence_node_ids = re.findall(r"NODE\s+([^\s|]+)", text)
        first_node = evidence_node_ids[0] if evidence_node_ids else None

        if "apple" in text_l:
            ans_text = "Apple is a fruit, and it is often sweet."
        elif "banana" in text_l:
            ans_text = "Banana is a fruit, and it is typically yellow."
        else:
            ans_text = (
                "Based on the provided evidence, this item is a fruit-like entity."
            )

        claim = {
            "claim": ans_text,
            "citations": (
                [{"source_node_id": first_node, "mention_index": 0, "span_index": 0}]
                if first_node
                else []
            ),
        }
        payload = {
            "text": ans_text,
            "reasoning": "heuristic evidence-grounded deterministic answer",
            "claims": [claim],
        }
        return schema.model_validate(payload)

    def _build_answer_evaluation(self, text: str, schema):
        text_l = text.lower()
        weak = ("i don't know" in text_l) or ("insufficient" in text_l)
        payload = {
            "is_sufficient": not weak,
            "needs_more_info": bool(weak),
            "missing_aspects": [] if not weak else ["insufficient grounded evidence"],
            "notes": "heuristic deterministic evaluation",
        }
        return schema.model_validate(payload)

    def _structured_payload(self, schema, *, input: Any):
        schema_name = getattr(schema, "__name__", str(schema))
        text = self._flatten_prompt_text(input)

        # Filtering callback path from conversation.filtering.candiate_filtering_callback
        if "FilteringResponse" in schema_name:
            node_ids, edge_ids = self._heuristic_pick(text)
            return schema.model_validate(
                {
                    "reasoning": "heuristic lexical overlap from user question and candidate labels",
                    "relevant_ids": {"node_ids": node_ids, "edge_ids": edge_ids},
                }
            )
        if "FilteringResult" in schema_name:
            node_ids, edge_ids = self._heuristic_pick(text)
            return schema.model_validate({"node_ids": node_ids, "edge_ids": edge_ids})

        # Agentic answering path
        if "EvidenceSelection" in schema_name:
            node_ids, edge_ids = self._heuristic_pick(text)
            return schema.model_validate(
                {
                    "used_node_ids": node_ids,
                    "used_edge_ids": edge_ids,
                    "reasoning": "heuristic evidence selection from candidate overlap",
                }
            )
        if "AnswerWithCitations" in schema_name:
            return self._build_answer_with_citations(text, schema)
        if "AnswerEvaluation" in schema_name:
            return self._build_answer_evaluation(text, schema)

        # fallback to neutral empty validated instance if possible
        try:
            return schema.model_validate({})
        except Exception:
            return {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._content))]
        )


def _make_engine(
    *,
    backend_kind: str,
    tmp_path: Path,
    sa_engine,
    pg_schema: str | None,
    graph_type: str,
    dim: int,
):
    engine = None
    if backend_kind == "chroma":
        engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / graph_type),
            kg_graph_type=graph_type,
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip(
                "pg backend requested but sa_engine/pg_schema fixtures not available"
            )
        schema = f"{pg_schema}_{graph_type}"
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=schema)
        engine = GraphKnowledgeEngine(
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
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
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
        return {
            "node_ids": ["K:apple", "K:fruit", "K:sweet"],
            "edge_ids": ["E:is_a", "E:has_property"],
        }
    if knowledge_key == "banana" and "banana" in q:
        return {
            "node_ids": ["K:banana", "K:fruit", "K:yellow"],
            "edge_ids": ["E:is_a", "E:has_color"],
        }
    return {"node_ids": [], "edge_ids": []}


def _deterministic_answer_impl(*, question: str, knowledge_key: str) -> dict[str, Any]:
    """
    Cacheable: only accepts simple serializable inputs.
    Returns a dict used by the answer_only harness.
    """
    q = question.lower()
    if knowledge_key == "apple" and "apple" in q:
        return {
            "content": "Apple is a fruit, and it is often sweet.",
            "need_summary": False,
        }
    if knowledge_key == "banana" and "banana" in q:
        return {
            "content": "Banana is a fruit, and it is typically yellow.",
            "need_summary": False,
        }
    return {"content": "I don't know.", "need_summary": False}


def _deterministic_llm_tasks(*, knowledge_key: str) -> LLMTaskSet:
    def _filter(req: FilterCandidatesTaskRequest) -> FilterCandidatesTaskResult:
        picked = _deterministic_filter_impl(
            question=req.conversation_content, knowledge_key=knowledge_key
        )
        return FilterCandidatesTaskResult(
            node_ids=tuple(picked.get("node_ids", [])),
            edge_ids=tuple(picked.get("edge_ids", [])),
            reasoning="deterministic",
            raw=None,
            parsing_error=None,
        )

    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload=None, parsing_error="unused"
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload=None, raw=None, parsing_error="unused"
        ),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error="unused"
        ),
        filter_candidates=_filter,
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error="unused"
        ),
        repair_citations=lambda _req: RepairCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error="unused"
        ),
        provider_hints=LLMTaskProviderHints(filter_candidates_provider="custom"),
    )


@pytest.mark.parametrize(
    "backend_kind,llm_mode",
    [
        pytest.param("chroma", "fake", id="chroma_fake", marks=[pytest.mark.ci_full]),
        pytest.param("pg", "fake", id="pg_fake", marks=[pytest.mark.ci_full]),
        pytest.param(
            "chroma",
            "real",
            id="chroma_real",
            marks=[pytest.mark.nightly, pytest.mark.llm_real],
        ),
        pytest.param(
            "pg",
            "real",
            id="pg_real",
            marks=[pytest.mark.nightly, pytest.mark.llm_real],
        ),
    ],
)
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
    knowledge_builder: Callable[..., dict[str, Any]],
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
    kg = _make_engine(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        graph_type="knowledge",
        dim=dim,
    )
    conv = _make_engine(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        graph_type="conversation",
        dim=dim,
    )
    wf = _make_engine(
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        sa_engine=sa_engine,
        pg_schema=pg_schema,
        graph_type="workflow",
        dim=dim,
    )

    # Seed knowledge
    meta = knowledge_builder(kg, dim=dim)
    question = meta["question"]

    conv.llm_tasks = _deterministic_llm_tasks(knowledge_key=knowledge_key)
    svc = ConversationService.from_engine(
        conv,
        knowledge_engine=kg,
        workflow_engine=wf,
    )
    orc = svc.orchestrator

    # cache folder: per-test temp to avoid cross-test pollution
    memory = Memory(location=str(tmp_path / ".joblib"), verbose=0)
    cached_filter = cached(memory, _deterministic_filter_impl)
    cached_answer = cached(memory, _deterministic_answer_impl)

    # Filtering callback
    if llm_mode == "fake":

        def filtering_callback(
            _llm_tasks,
            conversation_content,
            cand_node_list_str,
            cand_edge_list_str,
            candidates_node_ids,
            candidate_edge_ids,
            context_text,
        ):
            dumped = cached_filter(
                question=conversation_content, knowledge_key=knowledge_key
            )
            return FilteringResult.model_validate(dumped), "cached deterministic filter"
    else:
        # Use the project's canonical prompting-based filter, but cache it (ignore llm).
        def cached_inner(
            llm_tasks: LLMTaskSet,
            conversation_content,
            cand_node_list_str,
            cand_edge_list_str,
            candidates_node_ids,
            candidate_edge_ids,
            context_text,
        ):
            fr, reason = candiate_filtering_callback(
                llm_tasks,
                conversation_content,
                cand_node_list_str,
                cand_edge_list_str,
                candidates_node_ids,
                candidate_edge_ids,
                context_text,
            )
            return fr.model_dump(), reason

        cached_fn = cached(memory, cached_inner, ignore=["llm_tasks"])

        def filtering_callback(
            llm_tasks: LLMTaskSet,
            conversation_content,
            cand_node_list_str,
            cand_edge_list_str,
            candidates_node_ids,
            candidate_edge_ids,
            context_text,
        ):
            dumped, reason = cached_fn(
                llm_tasks,
                conversation_content,
                cand_node_list_str,
                cand_edge_list_str,
                candidates_node_ids,
                candidate_edge_ids,
                context_text,
            )
            return FilteringResult.model_validate(dumped), reason

    # Answer harness (cached, serializable-only inner)
    def answer_only_harness(
        *, conversation_id: str, prev_turn_meta_summary: MetaFromLastSummary, **_
    ):

        payload = cached_answer(question=question, knowledge_key=knowledge_key)
        # Use the project's conversation engine API for assistant turn creation if available
        # We keep this harness minimal: the v2 orchestrator expects a ConversationAIResponse-like object.
        # If your repo defines ConversationAIResponse, use it directly.
        try:
            from kogwistar.conversation.models import (
                ConversationAIResponse,
            )

            return ConversationAIResponse(
                response_node_id=None,  # will be filled by engine call below if available
                llm_decision_need_summary=bool(payload.get("need_summary", False)),
                text=payload["content"],
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

    svc.create_conversation(user_id, conv_id, "start_node_v2")

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
        assert nid in set(res.relevant_kg_node_ids), (
            f"expected KG node id {nid} to be referenced"
        )

    # Verify: assistant response exists (if your pipeline creates it)
    if res.response_turn_node_id is not None:
        got_ai = conv.backend.node_get(ids=[res.response_turn_node_id])
        assert got_ai["ids"], "assistant turn node missing"
        ai_doc = json.loads(got_ai["documents"][0])
        assert ai_doc["role"] == "assistant"
        for term in meta["expect_terms"]:
            assert term in ai_doc["summary"].lower()
