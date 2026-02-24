
import pytest

from graph_knowledge_engine.conversation_orchestrator import ConversationOrchestrator, _estimate_tokens_from_chars
from graph_knowledge_engine.models import FilteringResult


class FakeConversationEngine:
    def __init__(self):
        self.kg_graph_type = "conversation"
        self.llm = None
        self.added_nodes = []
        self.added_edges = []
        self._tail = None

    def _iterative_defensive_emb(self, text: str):
        return [0.0, 0.0, 0.0]

    def add_node(self, node, _=None):
        self.added_nodes.append(node)
        self._tail = node

    def add_edge(self, edge):
        self.added_edges.append(edge)
    def query_nodes(self, *arg, query_embeddings = None, **kwargs):
        return self.added_nodes
    def _get_conversation_tail(self, conversation_id: str, *a, **k):
        return self._tail
from typing import Sequence, Callable
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.postgres_backend import PgVectorBackend
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings
class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, model_name: str = "all-minilm:l6-v2", dim = 3):

        def ef(prompts: Sequence[str]) -> Embeddings:
            res: Embeddings = []
            for p in prompts:
                # Boundary: ollama types are weak -> cast once.
                r = [0.01] * dim
                
                res.append(r)
            return res

        self._emb: Callable[[Sequence[str]], Embeddings] = ef

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return self._emb(documents_or_texts)

def _make_engine_pair(*, backend_kind: str, tmp_path, sa_engine, pg_schema, dim: int = 3):
    """
    Build (kg_engine, conv_engine) for either chroma or pgvector.
    """
    # ef = _fake_ef_dim(dim)

    if backend_kind == "chroma":
        kg_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg"), kg_graph_type="knowledge", embedding_function=FakeEmbeddingFunction(dim=dim)
                                         )
        conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv"), kg_graph_type="conversation", embedding_function=FakeEmbeddingFunction(dim=dim)
                                           )
        return kg_engine, conv_engine

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip("pg backend requested but sa_engine/pg_schema fixtures not available")
        kg_schema = f"{pg_schema}_kg"
        conv_schema = f"{pg_schema}_conv"
        kg_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=kg_schema)
        conv_backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=conv_schema)
        kg_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "kg_meta"), 
                                         kg_graph_type="knowledge", embedding_function=FakeEmbeddingFunction(dim=dim), backend=kg_backend)
        conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv_meta"),
                                           kg_graph_type="conversation", embedding_function=FakeEmbeddingFunction(dim=dim), backend=conv_backend)
        return kg_engine, conv_engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")

def _noop_filtering_callback(llm, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidate_node_ids: list[str], candidate_edge_ids: list[str], context_text):
    
    return FilteringResult(node_ids=candidate_node_ids, edge_ids=candidate_edge_ids), 'this is a noop testing pass through fake filtering'


def test_estimate_tokens_from_chars_default_proxy():
    assert _estimate_tokens_from_chars(0) == 0
    assert _estimate_tokens_from_chars(1) == 1
    assert _estimate_tokens_from_chars(8) == 2  # 4 chars/token proxy

@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])

def test_summary_trigger_can_use_token_threshold(monkeypatch, backend_kind: str, tmp_path, sa_engine, pg_schema):
    kg, eng = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=384)


    orch = ConversationOrchestrator(
        conversation_engine=eng,
        ref_knowledge_engine=kg,
        tool_call_id_factory=lambda *a, **k: "tool_call_id",
        llm=None,
    )

    called = {"n": 0}

    def _fake_summarize(*a, **k):
        called["n"] += 1
        return "summary_node_id"

    monkeypatch.setattr(orch, "_summarize_conversation_batch", _fake_summarize)
    orch.create_conversation(user_id='u1', conv_id='c1')
    # Seed first turn
    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t1",
        mem_id="m1",
        role="user",
        content="x" * 100,
        filtering_callback=_noop_filtering_callback,
        add_turn_only=False,
        summary_turn_threshold=1000,
        summary_char_threshold=10**9,  # don't trigger via chars
        summary_token_threshold=10**9,  # don't trigger via tokens yet
    )

    # Second turn: make token proxy exceed threshold but keep char threshold very high.
    orch.add_conversation_turn(
        user_id="u1",
        conversation_id="c1",
        turn_id="t2",
        mem_id="m1",
        role="user",
        content="y" * 9000,
        filtering_callback=_noop_filtering_callback,
        add_turn_only=False,
        summary_turn_threshold=1000,
        summary_char_threshold=10**9,
        summary_token_threshold=1000,  # 9000 chars -> ~2250 tokens -> triggers
    )

    assert called["n"] == 1

    # Accounting keys must not land on the turn node itself.
    nodes = eng.get_nodes()
    n2 = nodes[-1]
    assert "char_distance_from_last_summary" not in n2.metadata
    assert "turn_distance_from_last_summary" not in n2.metadata
