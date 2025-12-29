from requests import Response

from typing import cast, Callable, List, Any, Dict
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
def test_semantic_ocr_document_splitting():
    
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = "GPT-5-chat"
    if model_name=='GPT-5-chat':
        temp = 0.1
        from langchain_openai import AzureChatOpenAI        
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        handler = UsageMetadataCallbackHandler()
        llm = AzureChatOpenAI(deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT5_CHAT"),
                            model_name = os.getenv("OPENAI_DEPLOYMENT_NAME_GPT5_CHAT"),
                            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT5_CHAT"),
                            cache=None,
                            openai_api_key=os.getenv("OPENAI_API_KEY_GPT5_CHAT"),
                            api_version="2024-08-01-preview",
                            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT5_CHAT"),
                            temperature=temp,
                            max_tokens = 12000,
                            callbacks=[handler],
                            #reasoning_effort="minimal",
                            openai_api_type="azure",
        )
    elif model_name.startwith("gemini"):
        from graph_knowledge_engine.utils.langchain import get_gemini_callback_cost
    
        cm =  get_gemini_callback_cost()
        cb = cm.__enter__()  
        llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        cache=None,
                        max_tokens=12000,
                        callbacks=[cb]
                    )
    else:
        print("using default model")
        llm = None
    def filter_callback (file_path):
        # folder at least have some page ocr that ends with .json
        for i, f in enumerate(os.listdir(file_path)):
            if f.endswith('.json'):
                return True
        else:
            return False
        pass
        
        
    compare_root = os.path.join('..', 'doc_data', 'split_pages')
    from graph_knowledge_engine.utils.file_loader import RawFileLoader
    loader = RawFileLoader(env_flist_path=None,
                           walk_root=os.path.join('..', 'doc_data', 'split_pages', 'run_set'),
                           compare_root = os.path.join('..', 'doc_data', 'split_pages'),
                           filtering_callbacks = [filter_callback],
                           include = ['dirs']
                           )
    from graph_knowledge_engine.ingesters.top_down_CRD_intgester import parse_doc, semantic_tree_to_kge_payload, kge_payload_to_semantic_tree,build_index_terms_for_semantic_node,all_child_from_root
    from graph_knowledge_engine.ocr import regen_doc
    from joblib import Memory
    memory = Memory(location = '.joblib')
    for f in loader:
        f_name = pathlib.Path(f).name
        doc = {f_name : regen_doc(os.path.join(compare_root, f), use_raw = True)}
        document_tree, source_map = parse_doc(doc, llm = llm)
        cached_semantic_tree_to_kge_payload: Callable[[Any], Dict[str, Any] ] = cast(Callable[[Any], Dict[str, Any] ], memory.cache(semantic_tree_to_kge_payload))
        graph_to_persist = cached_semantic_tree_to_kge_payload(document_tree)
        reconstrcted_root = kge_payload_to_semantic_tree(graph_to_persist)
        assert reconstrcted_root.model_dump() == document_tree.model_dump()
        import requests
        res = requests.post("http://127.0.0.1:28110/api/contract.validate_graph", json = graph_to_persist)
        res.raise_for_status()
        nodes = all_child_from_root(reconstrcted_root)
        batch_index_list = build_index_terms_for_semantic_node(nodes)        
        payload = {"index": [i.model_dump(mode='json') for i in batch_index_list]}
        for k in payload['index']:
            k.update({'doc_id': str(reconstrcted_root.node_id)})
        @memory.cache
        def get_index_entries(payload) -> Response:
            res = requests.post("http://127.0.0.1:28110/api/add_index_entries", json = payload)
            return res
        res: Response = get_index_entries(payload)
        res.raise_for_status()
        @memory.cache
        def get_upsert_result(graph_to_persist):
            res = requests.post("http://127.0.0.1:28110/api/contract.upsert_tree", json = graph_to_persist)
            return res
        res2: Response = get_upsert_result(graph_to_persist)
        res2.raise_for_status()
        # search using index
        
        res3 = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", 
                            params = {'q' : '"6.7"'})
        res3.raise_for_status()
        res4 = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", 
                            params = {'q' : '"6.7"', "resolve_node": True})
        res4.raise_for_status()