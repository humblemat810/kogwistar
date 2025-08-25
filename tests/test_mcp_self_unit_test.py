def test_doc_node_edge_adjudicate(small_test_docs_nodes_edge_adjudcate):

    docs=small_test_docs_nodes_edge_adjudcate.get("docs")
    nodes=small_test_docs_nodes_edge_adjudcate.get("nodes")
    edges=small_test_docs_nodes_edge_adjudcate.get("edges")
    adjudication_pairs=small_test_docs_nodes_edge_adjudcate.get("adjudication_pairs")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    graph_rag_port = 28110
    import requests
    
    ####### create doc 1 and doc 2 graph here. with some nodes and edge adjudicatable, very similar meaning, with 10 nodes and ten edges each
    
    
    insertion_method = 'llm_graph_extraction'
    # insert graph1 from doc1
    requests.post(f"http://127.0.0.1:{graph_rag_port}/api/graph/upsert", json = {})
    # insert graph1 from doc2
    requests.post(f"http://127.0.0.1:{graph_rag_port}/api/graph/upsert", json = {})
    import os
    folder_path = os.path.join("..","haast-data","split_pages","Contract Samples - cleaned","Samples for sending","Contract 4 - East Australia - Chris Miller")
    async with streamablehttp_client(f"http://127.0.0.1:{graph_rag_port}/mcp",sse_read_timeout = None, timeout = None,
                                        ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()  # SDK negotiates protocol

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {'kg_extract', 'doc_parse', 'kg_crossdoc_adjudicate_anykind'} <= names
            import json
            
            res = await session.call_tool("kg_load_persisted", 
                        arguments={"inp": {"doc_ids": os.listdir(folder_path), "insertion_method": "llm_graph_extraction" }})
            
            res = await session.call_tool("kg_crossdoc_adjudicate_anykind", 
                        arguments={"inp": {"doc_ids": os.listdir(folder_path), "insertion_method": "llm_graph_extraction" }})
            assert (res.content[0].type == "json") or (res.content[0].type == "text" and json.loads(res.content[0].text))
