import os
import pytest
import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Span, LLMGraphExtraction
from typing import cast
@pytest.fixture(scope="module")
def engine() -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine()
def test_ingest_documentS_with_llm(engine:GraphKnowledgeEngine)-> None: 
    engine.llm.model = "models/gemini-2.5-pro"
    doc_content_list = [
        "1. Science & Facts"
        "The Pacific Ocean is the largest and deepest body of water on Earth, covering more than 60 million square miles. It contains countless ecosystems, from vibrant coral reefs to mysterious deep-sea trenches. Scientists continue to discover new species in its depths each year."
,
        "2. Opinion & Subjectivity"
        "Electric cars represent a significant shift in personal transportation, but their long-term sustainability depends on battery recycling and renewable energy adoption. While some drivers embrace the quiet and efficiency, others still worry about range limitations."
,
        "3. Narrative / Storytelling"
        "Under the dim glow of the streetlamp, Mia tightened her scarf and hurried across the empty park. Snow crunched under her boots, and each breath made a small cloud in the cold night air. She glanced back once, but the path was empty."
,
        "4. Technical / Instructional"
        "To brew a smooth cup of coffee, start with freshly ground beans and water heated to about 93°C. Pour slowly over the grounds in a circular motion, allowing the bloom to release trapped gases. Let the coffee steep for three to four minutes before serving."
,
        "5. Abstract / Philosophical"
        "Time feels both infinite and fleeting. We measure it in seconds and years, yet our perception shifts with mood and memory. Perhaps the real challenge is not in keeping time, but in learning how to truly inhabit it."
,
        "6. Happiness"
        "Happiness often comes in small, quiet moments—a shared laugh with a friend, the warmth of sunlight on your skin, or the first sip of tea on a cold morning. It’s less about grand achievements and more about noticing the simple joys that fill each day."
    ]
    instruction_for_node_edge_contents_parsing_inclusion = ("Extract knowledge as nodes and edges. Extract node as entity, edge as relationship. "
                                            "Both nodes and edges can be endpoint of an edge to represent nested ideas. All edge endpoints must be resolvable. "
                                            "use source_ids or target_ids for nodes; use source_edge_ids or target_edge_ids for nodes. ")
    import pathlib
    from joblib import Memory
    cache_dir = os.path.join(".cache","test",pathlib.Path(__file__).name,"extract")
    os.makedirs(cache_dir, exist_ok = True)
    memory_embeddings = Memory(location=cache_dir, verbose=0)
    
    location=os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "_extract_only")
    os.makedirs(location, exist_ok = True)
    memory_extract_only = Memory(location=location, verbose=0)
    for doc_content in doc_content_list:
        @memory_embeddings.cache
        def get_ef(doc_content) -> list[list[float]]:# -> Any | list[list[float]]:
            res : list[list[float]]= engine._ef(doc_content) # type: ignore
            return res
        doc = Document(content=doc_content,
                   type="text", metadata={"source":"test"}, domain_id = None, processed = False, embeddings = cast(list[list[float]], get_ef(doc_content))[0])
           # cache ONLY the pure extraction on the doc content
        
        

        
        
        @memory_extract_only.cache
        def _extract_only(content: str, instruction_for_node_edge_contents_parsing_inclusion: None | str = None):
            raw_with_parsed = engine.extract_graph_with_llm(content=content, doc_type = doc.type,
                                                 instruction_for_node_edge_contents_parsing_inclusion = instruction_for_node_edge_contents_parsing_inclusion)
            parsed: LLMGraphExtraction['llm']= raw_with_parsed["parsed"]
            return {"raw": raw_with_parsed["raw"], "parsed": parsed.model_dump()}
        

        extracted = _extract_only(doc.content, instruction_for_node_edge_contents_parsing_inclusion)
        parsed = LLMGraphExtraction['llm'].model_validate(extracted["parsed"], context = {"insertion_method" : "test_case/test_ingest_documentS_with_llm"})

        # persist deterministically (choose replace/append/skip-if-exists)
        out = engine.persist_graph_extraction(document=doc, parsed=parsed, mode="replace")
    pass
def test_ingest_document_with_llm(engine):
    # Example document content
    doc_content = (
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Chlorophyll is the molecule that absorbs sunlight. "
        "Plants perform photosynthesis in their leaves."
    )
    document = Document(
        content=doc_content,
        type="ocr",
        metadata={"source": "test"},
        processed=False,
        domain_id = None,
    )

    # The .env file must be configured with Azure OpenAI credentials
    result = engine.ingest_document_with_llm(document)
    assert "document_id" in result
    assert result["nodes_added"] >= 1
    assert result["edges_added"] >= 1

    # Retrieve nodes and edges from ChromaDB
    nodes = engine.node_collection.get()
    edges = engine.edge_collection.get()

    # Check that at least one node and one edge has a non-empty references field
    node_found = False
    for node_json in nodes["documents"]:
        node = json.loads(node_json)
        if node.get("groundings"):
            node_found = True
            # Check ReferenceSession structure
            ref = node["references"][0]
            assert "collection_page_url" in ref and "document_page_url" in ref
    assert node_found, "No node with references found"

    edge_found = False
    for edge_json in edges["documents"]:
        edge = json.loads(edge_json)
        if edge.get("references"):
            edge_found = True
            ref = edge["references"][0]
            assert "collection_page_url" in ref and "document_page_url" in ref
    assert edge_found, "No edge with references found"

    for node_json in nodes["documents"]:
        node = json.loads(node_json)
        assert "id" in node and isinstance(node["id"], str)
        assert node["label"] and node["type"]

    for edge_json in edges["documents"]:
        edge = json.loads(edge_json)
        assert isinstance(edge.get("source_ids", []), list)
        assert isinstance(edge.get("target_ids", []), list)
        assert isinstance(edge.get("relation", ""), str)
    print("Ingested document:", result)

#{'document_id': 'cbd56c36-7664-11f0-9315-0456e5e49702', 'node_ids': ['668ad754-4a3f-4ba9-a6ab-9178daa45e8c', '21a81be9-4257-41c7-93fd-a3ef32273eec', '81f7100a-7ecb-4645-ab2c-00b9323dcf27', '7a8e0857-ae61-483f-84bc-e246b92b31d7'], 'edge_ids': ['2a3d53e2-06dc-4b41-9368-71d56216035a', 'dd5fe3b6-9e7b-4583-b6a5-2cc35db3725e', '49d92d0a-1ab3-42d3-91af-1bf80563d9e5'], 'nodes_added': 4, 'edges_added': 3}