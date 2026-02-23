import json
from pathlib import Path

import pytest
import functools
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from graph_knowledge_engine.cdc.oplog import OplogWriter
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import FilteringResult, MetaFromLastSummary, Node, Span, Grounding, MentionVerification, ConversationNode
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from typing import Callable, TypeVar, ParamSpec, cast, Sequence, Any
from joblib import Memory

from graph_knowledge_engine.id_provider import stable_id

from graph_knowledge_engine.postgres_backend import PgVectorBackend

# def _fake_ef_dim(dim: int):
#     def _ef(texts):
#         return [[0.01] * dim for _ in texts]
#     return _ef


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


P = ParamSpec("P")
R = TypeVar("R")
from itertools import count

def tool_id_factory(prefix: str = "tool-id"):
    counter = count(1)
    return lambda: f"{prefix}-{next(counter)}"
def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))
@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_conversation_flow(backend_kind: str, tmp_path, sa_engine, pg_schema):
    engine, conversation_engine = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=384)
    # wf_dir = tmp_path / "wf"
    # conv_dir = tmp_path / "conv"
    # kg_dir = tmp_path / "kg"
    
    engine._oplog = OplogWriter(Path(".") / "bundle" / "kg_changes.jsonl", fsync=False)
    conversation_engine._oplog = OplogWriter(Path(".") / "bundle" / "conv_changes.jsonl", fsync=False)
    user_id = "test_conversation_flow::test_user"
    # my_factory = tool_id_factory()
    my_factory = stable_id
    conversation_engine.tool_call_id_factory = my_factory
    orc=conversation_engine._get_orchestrator(ref_knowledge_engine=engine)
    orc.tool_runner.tool_call_id_factory = my_factory
    # 1. Pre-populate Knowledge Graph
    # We need a node "N1" for the filter to find/return.
    n1 = Node(
        id="N1", 
        label="Test Knowledge Node", 
        type="entity", 
        summary="A test node",
        mentions=[Grounding(spans=[Span(
            doc_id="D1", 
            chunk_id = None,
            source_cluster_id = None,
            verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test:verbatim user/system input"),
            collection_page_url="url", document_page_url="url", 
            insertion_method="test",
            page_number=1, start_char=0, end_char=10, excerpt="test content", context_before="", context_after=""
        )])],
        metadata={"level_from_root": 0},
        domain_id=None , canonical_entity_id=None , properties=None , embedding=None, doc_id=None,
        level_from_root = 0
        # properties = {"conversation": "concept"}
    )
    n2 = Node(
        id="N2", 
        label="Greet user polite", 
        type="entity", 
        summary="In general, as an AI assistant, you should entertain all users requests.",
        mentions=[Grounding(spans=[Span(
            doc_id="D1", 
            chunk_id = None,
            source_cluster_id = None,
            verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test:verbatim user/system input"),
            collection_page_url="url", document_page_url="url", 
            insertion_method="test",
            page_number=1, start_char=0, end_char=10, excerpt="test content", context_before="", context_after=""
        )])],
        metadata={"level_from_root": 0},
        domain_id=None , canonical_entity_id=None , properties=None , embedding=None, doc_id=None,
        level_from_root = 0
        # properties = {"conversation": "concept"}
    )    
    # Add manually to bypass ingestion logic complexity
    doc, meta = engine._node_doc_and_meta(n1)

    engine.backend.node_add(
        ids=[n1.id],
        documents=[doc],
        embeddings=[[0.1]*384], # Dummy embedding
        metadatas=[meta]
    )
    engine.add_node(n1) 
    engine.add_node(n2) 
    # Mock embedding function to return same dimension
    # conversation_engine._ef = lambda x: [[0.1]*384 for _ in x]
    
    # 2. Create Conversation
    conv_id = "conv_test_001"
    start_node_id = "start_con_node_001"
    conv_id, start_node_id_returned = conversation_engine.create_conversation(user_id, conv_id, start_node_id)
    assert conv_id
    assert start_node_id_returned == start_node_id
    # Check start node
    start_nodes_dict = conversation_engine.backend.node_get(where={"conversation_id": conv_id})
    assert len(start_nodes_dict['ids']) == 1
    start_nodes : list[Node] = conversation_engine.get_nodes(ids = start_nodes_dict['ids'])
    assert start_nodes
    
    assert start_nodes_dict['metadatas'][0]['entity_type'] == "conversation_start"
    memory = Memory(location = '.joblib')
    
    # Monkey patch with typed cacheing
    from graph_knowledge_engine.engine import candiate_filtering_callback
    candiate_filtering_callback_cached = cached(memory, candiate_filtering_callback)
    from functools import partial
    cached_deco = partial(cached, memory)
    # haha = 0
    # @cached_deco(ignore = ["llm"])
    # def my_cached_callback0():
    #     nonlocal haha
    #     haha += 1
    #     pass
    # my_cached_callback0()
    # my_cached_callback0()
    def cached_inner(llm: BaseChatModel, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidates_node_ids: list[str], candidate_edge_ids: list[str], context_text):
            filtering_result, filtering_reasining = candiate_filtering_callback(llm, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidates_node_ids, candidate_edge_ids, context_text)
            return filtering_result.model_dump(), filtering_reasining
    cached_fn = cached_deco(ignore = ["llm"], fn=cached_inner)
    def wrapped_cached_callback(llm: BaseChatModel, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidates_node_ids: list[str], candidate_edge_ids: list[str], context_text):
        
        
        dumped, reasoning = cached_fn(llm, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidates_node_ids, candidate_edge_ids, context_text)
        return FilteringResult.model_validate(dumped), reasoning
    # 3. Add Turn 1
    turn_id = "turn_cached_id_1"
    res = conversation_engine.add_conversation_turn(user_id, conv_id, turn_id, "mem0", 
                                                    role="user", content="Hello computer", 
                                                    ref_knowledge_engine=engine,
                                                    filtering_callback = candiate_filtering_callback_cached)
    prev_turn_meta_summary : MetaFromLastSummary = res.prev_turn_meta_summary
    assert res.turn_index == 2
    turn_id = res.user_turn_node_id
    
    # Verify Turn Node
    user_turn_node_data = conversation_engine.backend.node_get(ids=[turn_id])
    assert user_turn_node_data['ids']

    user_tn_doc = json.loads(user_turn_node_data['documents'][0])
    assert user_tn_doc['role'] == "user"
    assert user_tn_doc['summary'] == "Hello computer"
    
    # Verify Reference (FakeLLM returns ['N1'])
    # The code: relevant_kg_ids = ['N1'] -> creates reference node -> creates edge
    ref_nodes = conversation_engine.backend.node_get(where={"entity_type": "knowledge_reference"})
    
    
    # template_html = Path("graph_knowledge_engine/templates/d3.html").read_text(encoding="utf-8")
    # out_dir = Path(".") / "bundle"
    # from graph_knowledge_engine.utils.kge_debug_dump import dump_paired_bundles
    # dump_paired_bundles(
    #     kg_engine=engine,
    #     conversation_engine=conversation_engine,
    #     template_html=template_html,
    #     out_dir=out_dir,
    # )    
    
    assert len(ref_nodes['ids']) == 1
    ref_doc = json.loads(ref_nodes['documents'][0])
    referred_ids =[ json.loads(i)['properties']['refers_to_id'] for i in ref_nodes['documents']]
    assert ref_doc['properties']['refers_to_id'] == "N2"
    
    # Verify Edge (Turn -> Ref)
    edges = conversation_engine.backend.edge_get(where={"relation": "references"})
    assert len(edges['ids']) == 1
    e_doc = json.loads(edges['documents'][0])
    assert e_doc['source_ids'] == [turn_id]
    assert e_doc['target_ids'] == [ref_doc['id']]


    template_html = Path("graph_knowledge_engine/templates/d3.html").read_text(encoding="utf-8")
    out_dir = Path(".") / "bundle" / "turn1"
    from graph_knowledge_engine.utils.kge_debug_dump import dump_paired_bundles
    dump_paired_bundles(
        kg_engine=engine,
        conversation_engine=conversation_engine,
        template_html=template_html,
        out_dir=out_dir,
    )        
    # 4. Trigger Summarization (Batch size is 5, so we need turn index 0, 1, 2, 3, 4, 5... actually check is `if new_index > 0 and new_index % 5 == 0`)
    # We added index 0.
    # Add 5 more turns to guarantee reach index 5.
    
    #hard code index below
    last_turn_node = conversation_engine._get_conversation_tail(conv_id)
    i_start = (last_turn_node.turn_index or 0) if last_turn_node is not None else 0
    i_end = i_start + 5
    last_node_ids = []
    last_node_ids.append(conversation_engine._get_conversation_tail(conv_id))
    for i in range(i_start, i_end):
        res = conversation_engine.add_conversation_turn(
            user_id,
            conv_id,
            turn_id=f"assistant_turn_{i}" if i % 2 else f"user_turn_{i}",
            mem_id=f"msg turn_{i}",
            role="system",
            content=f"turn dummy filler turn {i}",
            ref_knowledge_engine=engine,
            filtering_callback=candiate_filtering_callback_cached,
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only=False,
        )
        last_node_ids.append(conversation_engine._get_conversation_tail(conv_id))
        prev_turn_meta_summary : MetaFromLastSummary = res.prev_turn_meta_summary
        template_html = Path("graph_knowledge_engine/templates/d3.html").read_text(encoding="utf-8")
        out_dir = Path(".") / "bundle" / f"turn {res.turn_index}-{i}"
        from graph_knowledge_engine.utils.kge_debug_dump import dump_paired_bundles
        dump_paired_bundles(
            kg_engine=engine,
            conversation_engine=conversation_engine,
            template_html=template_html,
            out_dir=out_dir,
        )        
    # # Now add index 5 -> Ensure Trigger
    # res_5 = conversation_engine.add_conversation_turn(user_id, conv_id, "assistant", "trigger summary",
    #                                                   role="system", content="turn dummy filler", 
    #                                               ref_knowledge_engine=engine,
    #                                              filtering_callback = candiate_filtering_callback_cached)
    assert res.turn_index >= 5
    
    # Check Summary Node
    summaries = conversation_engine.get_nodes(where={"entity_type": "conversation_summary"})
    assert len(summaries) >= 1
    sum_doc = summaries[0].summary
    # FakeLLM returned "['N1']" as content
    assert sum_doc
    
    # Check Summary Edges
    # It should link to the last 5 turns (Indices 1 to 5)
    sum_edges = conversation_engine.get_edges(where={"relation": "summarizes"})
    assert len(sum_edges) >= 1
    se_doc = sum_edges[0].summary
    assert se_doc
    # Target IDs should be the turn IDs
    assert len(sum_edges[0].target_ids) > 0
    # Note: Logic in _summarize uses `start_index = max(0, current_index - batch_size + 1)`
    # current=5, batch=5 -> start = 5-5+1 = 1. So covers indices 1,2,3,4,5. Total 5 nodes.
    # Turn 0 is left out of this batch (correct for sliding/tumbling window?)
    
    print("Conversation flow test passed!")


    template_html = Path("graph_knowledge_engine/templates/d3.html").read_text(encoding="utf-8")
    out_dir = Path(".") / "bundle"
    from graph_knowledge_engine.utils.kge_debug_dump import dump_paired_bundles
    dump_paired_bundles(
        kg_engine=engine,
        conversation_engine=conversation_engine,
        template_html=template_html,
        out_dir=out_dir,
    )    