import sys
import types
from dataclasses import dataclass, field
from typing import Any, Optional, List

# ---------------------------------------------------------------------------
# Test-time shims
#   - LangChain: optional dependency (not needed in these unit tests)
#   - models: we stub the small subset needed by agentic_answering
# ---------------------------------------------------------------------------
import json
import os
from types import SimpleNamespace
from typing import Callable, ParamSpec, TypeVar, cast
from joblib import Memory
from pydantic import BaseModel



P = ParamSpec("P")
R = TypeVar("R")

def cached(memory: Memory, fn: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn))
def _llm_cache_key(model_name: str, temperature: float, messages: list[dict], 
                   out_model_schema: dict, evidence_pack: dict | list,  system_prompt: str
                   ) -> str:
    # ensure stable cache key
    payload = {
        # "model": model_name,
        # "temperature": temperature,
        "evidence_pack": evidence_pack,
        "system_prompt": system_prompt,
        "messages": messages,
        "out_model_schema" : out_model_schema,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)

def make_cached_invoke(memory: Memory, llm) -> Callable[[str], str]:
    """
    Returns a cached function that invokes llm given a serialized payload.
    `llm` is captured in closure (not cached), only the payload->string is cached.
    """
    def _invoke(payload_json: str) -> str:
        payload = json.loads(payload_json)
        # IMPORTANT: this uses the *real* llm, but caching happens on payload_json.
        resp = llm.invoke(payload["messages"])
        # resp can be AIMessage; normalize to text.
        return getattr(resp, "content", str(resp))
    return cached(memory, _invoke)

def _install_langchain_core_stubs() -> None:
    if "langchain_core" in sys.modules:
        return

    langchain_core = types.ModuleType("langchain_core")
    prompts = types.ModuleType("langchain_core.prompts")
    language_models = types.ModuleType("langchain_core.language_models")
    chat_models = types.ModuleType("langchain_core.language_models.chat_models")

    class _StubRunnable:
        def __or__(self, other):
            return self
        def invoke(self, *args, **kwargs):
            raise RuntimeError("Stub: should not be invoked in these unit tests")

    class ChatPromptTemplate(_StubRunnable):
        @classmethod
        def from_messages(cls, messages):
            return cls()

    class BaseChatModel:
        def with_structured_output(self, *args, **kwargs):
            return self

    prompts.ChatPromptTemplate = ChatPromptTemplate
    chat_models.BaseChatModel = BaseChatModel

    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.prompts"] = prompts
    sys.modules["langchain_core.language_models"] = language_models
    sys.modules["langchain_core.language_models.chat_models"] = chat_models


def _install_gke_models_stub() -> None:
    """Stub graph_knowledge_engine.models to avoid importing the full Pydantic graph.

    These tests focus on agent orchestration + projection idempotency.
    """
    if "graph_knowledge_engine.models" in sys.modules:
        return

    m = types.ModuleType("graph_knowledge_engine.models")

    @dataclass
    class Span:
        collection_page_url: str = ""
        document_page_url: str = ""
        doc_id: str = "_conv:_dummy"
        insertion_method: str = "system"
        page_number: int = 1
        start_char: int = 0
        end_char: int = 1
        excerpt: str = ""
        context_before: str = ""
        context_after: str = ""

        @staticmethod
        def from_dummy_for_conversation():
            return Span()

    @dataclass
    class Grounding:
        spans: List[Span] = field(default_factory=list)

    @dataclass
    class ConversationNode:
        id: str
        label: str
        type: str
        summary: str
        conversation_id: str
        role: str
        turn_index: Optional[int]
        properties: dict
        mentions: list
        metadata: dict
        domain_id: Optional[str]
        canonical_entity_id: Optional[str]

    @dataclass
    class ConversationEdge:
        id: str
        source_ids: list
        target_ids: list
        relation: str
        label: str
        type: str
        summary: str
        doc_id: str
        mentions: list
        domain_id: Optional[str]
        canonical_entity_id: Optional[str]
        properties: dict
        embedding: Any = None
        metadata: dict = field(default_factory=dict)
        source_edge_ids: list = field(default_factory=list)
        target_edge_ids: list = field(default_factory=list)

    m.Span = Span
    m.Grounding = Grounding
    m.ConversationNode = ConversationNode
    m.ConversationEdge = ConversationEdge

    sys.modules["graph_knowledge_engine.models"] = m


_install_langchain_core_stubs()
_install_gke_models_stub()

from langchain_google_genai import ChatGoogleGenerativeAI
import pytest

from graph_knowledge_engine.agentic_answering import (
    AgenticAnsweringAgent,
    AnswerWithCitations,
    AnswerEvaluation,
    AgentConfig,
    pointer_id,
    edge_id,
    snapshot_hash,
    EvidenceSelection,
)
from graph_knowledge_engine.models import Span


class FakeCollection:
    """Very small in-memory stand-in for a Chroma collection used by the agent."""

    def __init__(self):
        # id -> {"document": str|None, "metadata": dict|None}
        self._rows: dict[str, dict] = {}

    def get(self, *, ids, include=None):
        found_ids = []
        metas = []
        docs = []
        for _id in ids:
            if _id in self._rows:
                found_ids.append(_id)
                metas.append(self._rows[_id].get("metadata"))
                docs.append(self._rows[_id].get("document"))
        out = {"ids": found_ids}
        if include and "metadatas" in include:
            out["metadatas"] = metas
        if include and "documents" in include:
            out["documents"] = docs
        return out

    def query(self, *, query_embeddings, n_results, where=None):
        ids = list(self._rows.keys())[:n_results]
        metas = [self._rows[i].get("metadata") for i in ids]
        docs = [self._rows[i].get("document") for i in ids]
        return {"ids": [ids], "metadatas": [metas], "documents": [docs]}

    def add(self, _id: str, *, metadata=None, document=None):
        self._rows[_id] = {"metadata": metadata or {}, "document": document}
    def upsert(self, _id: str, *, metadata=None, document=None):
        self._rows[_id] = {"metadata": metadata or {}, "document": document}
    def update(self, _id: str, *, metadata=None, document=None):
        self._rows[_id] = {"metadata": metadata or {}, "document": document}        
    def __len__(self):
        return len(self._rows)

class FakeEmbedding:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

class _BackendShim:
    """Chroma-like backend shim exposing node_get/edge_get for AgenticAnsweringAgent."""
    def __init__(self, node_collection: FakeCollection, edge_collection: FakeCollection):
        self._nodes = node_collection
        self._edges = edge_collection

    def node_get(self, *, ids, include=None, where=None, limit=None, offset=None):
        # where/limit/offset ignored for this shim
        return self._nodes.get(ids=ids, include=include)
    
    def node_query(self, *, query_embeddings, n_results, include=None, where=None, limit=None, offset=None):
        # where/limit/offset ignored for this shim
        return self._nodes.query(query_embeddings = query_embeddings, n_results = n_results)

    def edge_get(self, *, ids, include=None, where=None, limit=None, offset=None):
        return self._edges.get(ids=ids, include=include)

class FakeConversationEngine:
    def __init__(self, conversation_id: str, messages):
        self._conversation_id = conversation_id
        self._messages = messages
        self.node_collection = FakeCollection()
        self.edge_collection = FakeCollection()
        self._iterative_defensive_emb = self.iterative_defensive_emb
        self.backend = _BackendShim(self.node_collection, self.edge_collection)
    def get_conversation_view(
        self,
        *,
        conversation_id: str,
        user_id=None,
        purpose=None,
        budget_tokens=None,
    ):
        return SimpleNamespace(
            conversation_id=conversation_id,
            user_id=user_id,
            system_prompt="",
            question="dummy question",
        )
    def get_conversation(self, conversation_id: str):
        assert conversation_id == self._conversation_id
        return self._messages

    def get_system_prompt(self, conversation_id: str):
        assert conversation_id == self._conversation_id
        return "You are a helpful assistant."

    def add_node(self, node):
        self.node_collection.add(node.id, metadata=getattr(node, "metadata", {}), document=getattr(node, "summary", None))

    def add_edge(self, edge):
        self.edge_collection.add(
            edge.id,
            metadata={"relation": edge.relation, "src": edge.source_ids, "dst": edge.target_ids},
            document=getattr(edge, "summary", None),
        )
    def _get_conversation_tail(self, conversation_id: str):
        # If your fake already tracks turns, return the real last one.
        # Otherwise, minimal stub:
        return SimpleNamespace(turn_index=-1)    
    def iterative_defensive_emb(self, text: str):
        return FakeEmbedding([0.0])
class FakeKnowledgeEngine:
    def __init__(self):
        self.node_collection = FakeCollection()
        self.edge_collection = FakeCollection()
        self._iterative_defensive_emb = self.iterative_defensive_emb
        self.backend = _BackendShim(self.node_collection, self.edge_collection)

    def iterative_defensive_emb(self, text: str):
        return FakeEmbedding([0.0])
    

@pytest.fixture()
def engines():
    conversation_id = "conv_1"
    conv = FakeConversationEngine(
        conversation_id,
        messages=[
            {"role": "system", "content": "hello"},
            {"role": "user", "content": "What is Foo?"},
        ],
    )
    kg = FakeKnowledgeEngine()
    kg.node_collection.add("A", metadata={"label": "Foo", "summary": "Foo is a thing.", "type": "concept"})
    kg.node_collection.add("B", metadata={"label": "Bar", "summary": "Bar is related.", "type": "concept"})
    kg.node_collection.add("C", metadata={"label": "Baz", "summary": "Baz is extra.", "type": "concept"})
    return conv, kg, conversation_id


def test_snapshot_hash_is_stable():
    payload = {"b": 2, "a": 1}
    h1 = snapshot_hash(payload)
    h2 = snapshot_hash({"a": 1, "b": 2})
    assert h1 == h2


def test_pointer_and_edge_ids_are_deterministic():
    pid1 = pointer_id(scope="conv:1", pointer_kind="kg_node", target_kind="node", target_id="A")
    pid2 = pointer_id(scope="conv:1", pointer_kind="kg_node", target_kind="node", target_id="A")
    assert pid1 == pid2

    eid1 = edge_id(scope="conv:1", rel="used_evidence", src="X", dst="Y")
    eid2 = edge_id(scope="conv:1", rel="used_evidence", src="X", dst="Y")
    assert eid1 == eid2
class FakeLLM:
    def with_structured_output(self, *args, **kwargs):
        return self  # must be pipe-compatible

    def invoke(self, *args, **kwargs):
        # Only needed if chain.invoke(...) is called.
        # Return whatever your code expects (dict or pydantic).
        return {"parsed": None, "raw": None}

from langchain_core.language_models import BaseChatModel

class NullLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "null"

    def with_structured_output(self, *args, **kwargs):
        raise AssertionError("LLM should not be used in this test")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("LLM should not be used in this test")    
from langchain_core.language_models import BaseChatModel
from typing import cast




def test_agent_answer_creates_run_anchor_projects_used_evidence(monkeypatch, engines):
    conv, kg, conversation_id = engines

    agent = AgenticAnsweringAgent(
        conversation_engine=conv,
        knowledge_engine=kg,
        llm=NullLLM(), 
        config=AgentConfig(max_candidates=5, max_used=2),
    )

    def fake_select(*, system_prompt, question, candidates):
        return EvidenceSelection(used_node_ids=[candidates[0]["id"], candidates[1]["id"]], used_edge_ids=[], reasoning="")

    def fake_generate(*, system_prompt, question, used_nodes):
        return "Answer using A and B"

    def fake_evaluate_answer(
        agent: AgenticAnsweringAgent,
        *,
        system_prompt: str,
        question: str,
        answer_text: str,
        used_node_ids: list[str],
        evidence_pack: dict,
        out_model_schema: dict,
        out_model,
    ):
        # evaluator returns BaseM instance (model), usually out_model
        return AnswerEvaluation(
            is_sufficient=True,
            needs_more_info=False,
            missing_aspects=[],
            notes="",
        )
    def fake_generate_answer_with_citations(
        agent: AgenticAnsweringAgent,
        *,
        system_prompt: str,
        question: str,
        evidence_pack: dict[str, object],
        used_node_ids: list[str],
        out_model_schema: dict[str, object],
        out_model,
    ):
        # Minimal valid output
        return AnswerWithCitations(
            text="Answer using A and B",
            reasoning="",
            claims=[],  # citations optional
        ).model_dump()
    monkeypatch.setattr(
        AgenticAnsweringAgent,
        "_generate_answer_with_citations",
        staticmethod(fake_generate_answer_with_citations),
    )
    monkeypatch.setattr(
        AgenticAnsweringAgent,
        "_evaluate_answer",
        staticmethod(fake_evaluate_answer),
    )

    def fake_get_conversation_view(*, conversation_id, user_id=None, purpose=None, budget_tokens=None):
        # Minimal shape for the agent to proceed.
        # Add fields here if your agent accesses more attributes.
        return SimpleNamespace(
            conversation_id=conversation_id,
            user_id=user_id,
            system_prompt="",
            question="dummy question",
            # minimal “chat history”
            messages=[
                {"role": "user", "content": "dummy question"},
            ],
        )




    monkeypatch.setattr(
        AgenticAnsweringAgent,
        "_generate_answer_with_citations",
        staticmethod(fake_generate_answer_with_citations),
    )
    monkeypatch.setattr(conv, "get_conversation_view", fake_get_conversation_view)
    monkeypatch.setattr(agent, "_select_used_evidence", fake_select)
    from graph_knowledge_engine.models import MetaFromLastSummary
    
    out = agent.answer(conversation_id=conversation_id, prev_turn_meta_summary = MetaFromLastSummary(0,0))

    assert out["run_node_id"] in conv.node_collection._rows
    # assert len(out["projected_pointer_ids"]) == 2
    for pid in out["projected_pointer_ids"]:
        assert pid in conv.node_collection._rows

    scope = f"conv:{conversation_id}"
    for pid in out["projected_pointer_ids"]:
        eid = edge_id(scope=scope, rel="used_evidence", src=out["run_node_id"], dst=pid)
        assert eid in conv.edge_collection._rows

    gen_eid = edge_id(scope=scope, rel="generated", src=out["run_node_id"], dst=out["assistant_turn_node_id"])
    assert gen_eid in conv.edge_collection._rows


def test_projection_is_idempotent(engines):
    conv, kg, conversation_id = engines
    from graph_knowledge_engine.models import MetaFromLastSummary
    agent = AgenticAnsweringAgent(
        conversation_engine=conv,
        knowledge_engine=kg,
        llm=NullLLM(),
        config=AgentConfig(max_candidates=5, max_used=2),
    )

    run_node_id = agent._ensure_run_anchor(conversation_id=conversation_id, run_id="run_test")

    before_nodes = len(conv.node_collection)
    before_edges = len(conv.edge_collection)
    prev_turn_meta_summary = MetaFromLastSummary(0,0)
    sp = Span.from_dummy_for_conversation()
    pid1 = agent._project_kg_node(conversation_id=conversation_id, run_node_id=run_node_id, kg_node_id="A", 
                                  provenance_span=sp, prev_turn_meta_summary=prev_turn_meta_summary)
    mid_nodes = len(conv.node_collection)
    mid_edges = len(conv.edge_collection)

    pid2 = agent._project_kg_node(conversation_id=conversation_id, run_node_id=run_node_id, kg_node_id="A", 
                                  provenance_span=sp, prev_turn_meta_summary=prev_turn_meta_summary)

    assert pid1 == pid2
    assert len(conv.node_collection) == mid_nodes
    assert len(conv.edge_collection) == mid_edges

    assert mid_nodes >= before_nodes + 1
    assert mid_edges >= before_edges + 1
    
@pytest.mark.integration
def test_agent_with_real_llm_cached(monkeypatch, engine, conversation_engine):
    from graph_knowledge_engine.engine import GraphKnowledgeEngine
    engine: GraphKnowledgeEngine
    conversation_engine: GraphKnowledgeEngine
    # If no real model configured, skip (keeps CI clean)
    llm: BaseChatModel | None = None
    
    try:
        if not os.getenv("OPENAI_API_KEY"):
            # Choose your real model (example: ChatOpenAI). Adjust to your stack.
            from langchain_openai import ChatOpenAI

            model_name = os.getenv("TEST_LLM_MODEL", "gpt-4o-mini")
            temperature = float(os.getenv("TEST_LLM_TEMPERATURE", "0"))

            llm = ChatOpenAI(model=model_name, temperature=temperature)        
    except:
        pass
    try:
        model_name="gemini-2.5-flash"
        temperature=0.1
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    except:
        pass
        
        
    if llm is None:
        pytest.skip("OPENAI_API_KEY not set; skipping real-LLM integration test")
    

    
    memory = Memory(location=".joblib/llm_cache", verbose=0)
    cached_invoke = make_cached_invoke(memory, llm)

    # Build a stable messages payload for each LLM call boundary.
    # NOTE: `messages` must be JSON-serializable; use dicts, not Message objects.
    def select_used_evidence_cached(query: str, candidates: list[dict]) -> list[str]:
        messages = [
            {"role": "system", "content": "Select the IDs of evidence items that are truly USED."},
            {"role": "user", "content": json.dumps({"query": query, "candidates": candidates}, sort_keys=True)},
        ]
        payload = _llm_cache_key(model_name, temperature, messages)
        text = cached_invoke(payload)

        # Parse whatever format you enforce (recommend strict JSON!)
        # For example, ask the LLM to return {"used_ids":[...]}.
        try:
            obj = json.loads(text)
            return list(obj.get("used_ids", []))
        except Exception:
            # fallback: very strict/defensive parse
            # e.g. if model returns `["N1"]`
            return list(json.loads(text))
    from typing import Type, TypeVar
    BaseM = TypeVar("BaseM", bound = BaseModel)
    def generate_answer_with_citations_cached(
                agent: "AgenticAnsweringAgent",
        *,
        system_prompt: str,
        question: str,
        evidence_pack: dict[str, Any],
        used_node_ids: list[str],
        out_model_schema: dict[str, Any],
        out_model: Type[BaseM]
        ) -> str:
        messages = [
            {"role": "system", "content": "Answer the user using ONLY the provided evidence."},
            {"role": "user", "content": json.dumps({"query": question, "used": used_node_ids}, sort_keys=True)},
        ]
        payload = _llm_cache_key(model_name, temperature, messages, out_model_schema, evidence_pack, system_prompt)
        return cached_invoke(payload)

    # Patch agent internals to use cached real-LLM boundaries
    from graph_knowledge_engine.agentic_answering import AgenticAnsweringAgent
    from graph_knowledge_engine.models import MetaFromLastSummary
    prev_turn_meta_summary = MetaFromLastSummary(0,0,0)

    agent = AgenticAnsweringAgent(conversation_engine=conversation_engine, 
                                  knowledge_engine=engine,
                                  llm=llm)

    monkeypatch.setattr(agent, "_select_used_evidence", select_used_evidence_cached)
    monkeypatch.setattr(agent, "_generate_answer_with_citations", generate_answer_with_citations_cached)

    # Now run your end-to-end agent call
    user_id = "test_agent_with_llm_cached"
    conv_id, start_node_id = conversation_engine.create_conversation(user_id)
    conversation_engine.add_conversation_turn(user_id, conv_id, role = 'user', turn_id = "user_turn1", 
                                              mem_id = "mem1", content = "what is an LLM ?", ref_knowledge_engine = engine)
    # out = agent.answer(conversation_id=conv_id, 
                       
    #                    prev_turn_meta_summary=prev_turn_meta_summary)

    # assert out["assistant_text"]
    # # Assert projection happened (e.g. used_evidence edges exist)
    # used_edges = conversation_engine.edge_collection.get(where={"relation": "used_evidence"})
    # assert len(used_edges["ids"]) >= 1