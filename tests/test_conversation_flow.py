import json

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import FilteringResult, Node, Span, Grounding, MentionVerification, ConversationNode
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")
from itertools import count

def tool_id_factory(prefix: str = "tool-id"):
    counter = count(1)
    return lambda: f"{prefix}-{next(counter)}"
def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))
def test_conversation_flow(engine:GraphKnowledgeEngine, conversation_engine:GraphKnowledgeEngine):
    user_id = "test_conversation_flow::test_user"
    my_factory = tool_id_factory()
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
    engine.node_collection.add(
        ids=[n1.id],
        documents=[doc],
        embeddings=[[0.1]*384], # Dummy embedding
        metadatas=[meta]
    )
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
    start_nodes_dict = conversation_engine.node_collection.get(where={"conversation_id": conv_id})
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
                                                    filtering_callback = wrapped_cached_callback)
    
    assert res['turn_index'] == 0
    turn_id = res['turn_node_id']
    
    # Verify Turn Node
    turn_node_data = conversation_engine.node_collection.get(ids=[turn_id])
    assert turn_node_data['ids']

    tn_doc = json.loads(turn_node_data['documents'][0])
    assert tn_doc['role'] == "user"
    assert tn_doc['summary'] == "Hello computer"
    
    # Verify Reference (FakeLLM returns ['N1'])
    # The code: relevant_kg_ids = ['N1'] -> creates reference node -> creates edge
    ref_nodes = conversation_engine.node_collection.get(where={"entity_type": "knowledge_reference"})
    assert len(ref_nodes['ids']) == 1
    ref_doc = json.loads(ref_nodes['documents'][0])
    assert ref_doc['properties']['refers_to_id'] == "N1"
    
    # Verify Edge (Turn -> Ref)
    edges = conversation_engine.edge_collection.get(where={"relation": "references"})
    assert len(edges['ids']) == 1
    e_doc = json.loads(edges['documents'][0])
    assert e_doc['source_ids'] == [turn_id]
    assert e_doc['target_ids'] == [ref_doc['id']]

    # 4. Trigger Summarization (Batch size is 5, so we need turn index 0, 1, 2, 3, 4, 5... actually check is `if new_index > 0 and new_index % 5 == 0`)
    # We added index 0.
    # Add 4 more turns to reach index 4.
    for i in range(1, 5):
        conversation_engine.add_conversation_turn(user_id, conv_id, "assistant" if i%2 else "user", f"msg {i}",
                                                  ref_knowledge_engine=engine,
                                                 filtering_callback = candiate_filtering_callback_cached)
    
    # Now add index 5 -> Trigger
    res_5 = conversation_engine.add_conversation_turn(user_id, conv_id, "assistant", "trigger summary",
                                                  ref_knowledge_engine=engine,
                                                 filtering_callback = candiate_filtering_callback_cached)
    assert res_5['turn_index'] == 5
    
    # Check Summary Node
    summaries = conversation_engine.node_collection.get(where={"type": "memory_summary"})
    assert len(summaries['ids']) == 1
    sum_doc = json.loads(summaries['documents'][0])
    # FakeLLM returned "['N1']" as content
    assert sum_doc['summary'] == "['N1']" 
    
    # Check Summary Edges
    # It should link to the last 5 turns (Indices 1 to 5)
    sum_edges = conversation_engine.edge_collection.get(where={"relation": "summarizes"})
    assert len(sum_edges['ids']) == 1
    se_doc = json.loads(sum_edges['documents'][0])
    # Target IDs should be the turn IDs
    assert len(se_doc['target_ids']) > 0
    # Note: Logic in _summarize uses `start_index = max(0, current_index - batch_size + 1)`
    # current=5, batch=5 -> start = 5-5+1 = 1. So covers indices 1,2,3,4,5. Total 5 nodes.
    # Turn 0 is left out of this batch (correct for sliding/tumbling window?)
    
    print("Conversation flow test passed!")
