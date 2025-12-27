import uuid
from graph_knowledge_engine.engine import GraphKnowledgeEngine, uuid_to_base62, base62_to_uuid, AliasBook
from graph_knowledge_engine.models import Node, Edge, Document, Span, MentionVerification, Grounding
import json
def _span_for(doc_id: str, start_page = 1, end_page = 1, start_char = 0, end_char = 1, collection_page_url = None, document_page_url = None) -> Span:
    if collection_page_url is None:
        collection_page_url = f"url/{doc_id}"
    if not document_page_url:
        document_page_url = f"document/{doc_id}"
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        start_page=start_page, end_page=end_page, start_char=start_char, end_char=end_char,
        verification=MentionVerification(method="heuristic", is_verified=False, notes = None, score = 0.9), 
        insertion_method="pytest-manual",
        doc_id = doc_id,
        source_cluster_id = None,
        snippet = None
    )
def test_base62_roundtrip():
    u = str(uuid.uuid4())
    s = uuid_to_base62(u)
    back = base62_to_uuid(s)
    assert back.lower() == u.lower()

def test_alias_book_stability_and_delta():
    book = AliasBook()
    nodes = [f"{uuid.uuid4()}" for _ in range(3)]
    edges = [f"{uuid.uuid4()}" for _ in range(2)]

    # first assignment
    new_nodes, new_edges = book.legend_delta(nodes, edges)
    assert len(new_nodes) == 3 and len(new_edges) == 2
    # second time should be empty delta (cache-friendly)
    new_nodes2, new_edges2 = book.legend_delta(nodes, edges)
    assert new_nodes2 == [] and new_edges2 == []
    # aliases are stable
    a1 = [book.real_to_alias[nodes[0]], book.real_to_alias[edges[0]]]
    # re-ask and ensure same
    book.legend_delta([nodes[0]], [edges[0]])
    a2 = [book.real_to_alias[nodes[0]], book.real_to_alias[edges[0]]]
    assert a1 == a2

def test_de_alias_ids_in_result_session_alias(monkeypatch):
    eng = GraphKnowledgeEngine()
    doc = Document(content="x", type="test", metadata = {"source": "test_de_alias_ids_in_result_session_alias"}, domain_id = None, processed = False)
    eng.add_document(doc)
    # pretend we already have context with two nodes and an edge
    n1 = Node(label="A", type="entity", summary="a", metadata = {"source": "test_commit_cross_kind_creates_reifies"},
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, mentions=[Grounding([_span_for(doc.id)])])
    n2 = Node(label="B", type="entity", summary="b", domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, metadata = {},
                mentions=[Grounding([_span_for(doc.id, 1,1,2,3)])])
    eng.add_node(n1, doc_id=doc.id)
    eng.add_node(n2, doc_id=doc.id)
    e = Edge(label="A->B", type="relationship", summary="ab", relation="rel", source_ids=[n1.id], target_ids=[n2.id],
             source_edge_ids = [], target_edge_ids = [], domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, metadata = {},
             mentions=[Grounding([_span_for(doc.id, 1,1,0,3)])])
    eng.add_edge(e, doc_id=doc.id)

    # allocate aliases for the context
    book = eng._alias_book(doc.id)
    book.assign_for_sets([n1.id, n2.id], [e.id])

    # fake an LLMGraphExtraction-shaped thing with aliases
    from graph_knowledge_engine.models import LLMGraphExtraction, LLMNode, LLMEdge, MentionVerification
    # immitate llm slice return from llm
    dumped = LLMGraphExtraction['llm'](
        nodes=[LLMNode['llm'](id=book.real_to_alias[n1.id], label="A", type="entity", summary="a",  domain_id=None, canonical_entity_id=None, properties=None, local_id = "nn:1",
                    mentions=[Grounding([_span_for(doc.id)])])
                                   ],
        edges=[LLMEdge['llm'](id=book.real_to_alias[e.id], label="A->B", type="relationship", summary="ab",  domain_id=None, canonical_entity_id=None, properties=None, local_id = "ne:1",
                       relation="rel", source_ids=[book.real_to_alias[n1.id]], target_ids=[book.real_to_alias[n2.id]],
                       source_edge_ids = [], target_edge_ids = [],
                       mentions=[Grounding([_span_for(doc.id,1,1,2,3, collection_page_url="c")])]
                       )]).model_dump()
    parsed = LLMGraphExtraction['llm'].model_validate( dumped, context = dict(insertion_method = "pytest-graph_extractor")) # re try with injection
    # invoke custom slice to base conversion
    parsed = LLMGraphExtraction.FromLLMSlice(parsed, insertion_method = "pytest-graph_extractor")
    # de-alias
    out = eng._de_alias_ids_in_result(doc.id, parsed)
    assert out.nodes[0].id == n1.id
    assert out.edges[0].id == e.id
    assert out.edges[0].source_ids == [n1.id]
    assert out.edges[0].target_ids == [n2.id]

def test_commit_merge_creates_same_as_and_endpoints():
    eng = GraphKnowledgeEngine()
    doc = Document(content="y", type="test", metadata={"source": "test_commit_merge_creates_same_as_and_endpoints"}, domain_id = None, processed = False)
    eng.add_document(doc)
    mentions=[Grounding([_span_for(doc.id,1,1,0,5, collection_page_url="c")])]
    # grounding = Span(collection_page_url="c", document_page_url=f"document/{doc.id}", 
                        #    insertion_method = "pytest-manual",
                        #    doc_id = doc.id, start_page=1, end_page=1, start_char=0, end_char=5)
    a = Node(label="Einstein", type="entity", summary="person", mentions=mentions,  
             domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, metadata = {})
    b = Node(label="Einstein", type="entity", summary="person (dup)", mentions=mentions,  
             domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, metadata = {})
    eng.add_node(a, doc_id=doc.id)
    eng.add_node(b, doc_id=doc.id)
    from graph_knowledge_engine.models import AdjudicationVerdict
    # minimal verdict-like object
    adj_verd = AdjudicationVerdict.model_validate(dict(same_entity=True,
        confidence=0.9,
        reason="dup",
        canonical_entity_id=None))
        
    
    eng.commit_merge(a, b, adj_verd)  # should create same_as edge with left_id/right_id or endpoints

    # verify an edge exists that ties a<->b
    edges = eng.edge_collection.get(include=["metadatas","documents"])
    found = False
    from typing import cast
    for eid, meta, docj in zip(edges["ids"], cast(list[dict],edges["metadatas"]), cast(list[dict],edges["documents"])):
        if (meta or {}).get("relation") == "same_as":
            found = True
            break
    assert found, "same_as edge not created by commit_merge"