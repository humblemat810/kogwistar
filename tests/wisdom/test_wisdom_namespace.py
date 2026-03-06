# test_wisdom_namespace.py
# import jwt, os
# from fastapi.testclient import TestClient
# from graph_knowledge_engine.server_mcp_with_admin import app, JWT_SECRET, JWT_ALG

# client = TestClient(app)

# def make_token(role="rw", ns="wisdom"):
#     payload = {"role": role, "ns": ns}
#     return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

# def test_docs_and_wisdom_are_isolated(tmp_path):
#     t_wis = make_token(ns="wisdom")
#     t_doc = make_token(ns="docs")

#     # --- store something in wisdom
#     body = {
#         "nodes": [{
#             "id": "n1",
#             "label": "Example wisdom node",
#             "type": "entity",
#             "summary": "Wisdom layer test",
#         }],
#         "edges": []
#     }
#     r = client.post(
#         "/mcp/tools/wisdom.kg_upsert_graph/invoke",
#         headers={"Authorization": f"Bearer {t_wis}"},
#         json={"arguments": body}
#     )
#     assert r.status_code == 200
#     wisdom_ids = r.json()["result"]["node_ids"]
#     assert wisdom_ids

#     # --- querying the normal engine should not see it
#     r2 = client.post(
#         "/mcp/tools/kg_semantic_seed_then_expand_text/invoke",
#         headers={"Authorization": f"Bearer {t_doc}"},
#         json={"arguments": {"text": "Wisdom layer test"}}
#     )
#     assert r2.status_code == 200
#     data2 = r2.json()["result"]
#     assert not any("Wisdom layer test" in str(data2) for _ in range(1)), "Document namespace should not see wisdom data"

# def test_ns_permission_enforced():
#     t_doc = make_token(ns="docs")  # not allowed for wisdom
#     r = client.post(
#         "/mcp/tools/wisdom.kg_upsert_graph/invoke",
#         headers={"Authorization": f"Bearer {t_doc}"},
#         json={"arguments": {"nodes": [], "edges": []}},
#     )
#     assert r.status_code == 200
#     err = r.json()["error"]["message"].lower()
#     assert "forbidden" in err
    
    
# def test_wisdom_semantic_seed_expand_roundtrip():
#     t_wis = make_token(ns="wisdom")
#     body = {
#         "nodes": [{
#             "id": "n1",
#             "label": "Variance Analysis",
#             "type": "entity",
#             "summary": "Compare expected vs observed values to locate anomalies."
#         }],
#         "edges": []
#     }
#     up = client.post(
#         "/mcp/tools/wisdom.kg_upsert_graph/invoke",
#         headers={"Authorization": f"Bearer {t_wis}"},
#         json={"arguments": body}
#     )
#     assert up.status_code == 200

#     q = client.post(
#         "/mcp/tools/wisdom.semantic_seed_then_expand/invoke",
#         headers={"Authorization": f"Bearer {t_wis}"},
#         json={"arguments": {"text": "variance analysis"}}
#     )
#     data = q.json()["result"]
#     assert "layers" in data
#     assert isinstance(data["layers"], list)
    
    
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import PureChromaNode, PureChromaEdge, PureGraph

def test_puregraph_persist(tmp_path):
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path))
    node = PureChromaNode(domain_id = "wisdom-math", label="T", type="entity", summary="x")
    edge = PureChromaEdge(domain_id = "wisdom-math", label="E", type="relationship", summary="r",
                          relation="rel", source_ids=[node.id], target_ids=[node.id],
                          source_edge_ids=[], target_edge_ids=[])
    g = PureGraph(nodes=[node], edges=[edge])
    out = eng.persist_graph(parsed=g, session_id="unit-test")
    assert out["node_ids"] and out["edge_ids"]
    
# def test_dev_token_endpoint_ns_field():
#     r = client.post("/auth/dev-token", params={"role": "rw", "ns": "wisdom"})
#     assert r.status_code == 200
#     tok = r.json()["token"]
#     decoded = jwt.decode(tok, JWT_SECRET, algorithms=[JWT_ALG])
#     assert decoded["ns"] == "wisdom"
    
# def test_tool_listing_respects_namespace(client):
#     token_docs = make_token(ns="docs")
#     token_wisdom = make_token(ns="wisdom")

#     r_docs = client.get("/mcp/tools", headers={"Authorization": f"Bearer {token_docs}"})
#     r_wis = client.get("/mcp/tools", headers={"Authorization": f"Bearer {token_wisdom}"})

#     names_docs = {t["name"] for t in r_docs.json()["tools"]}
#     names_wis  = {t["name"] for t in r_wis.json()["tools"]}

#     # Each side should only see its own prefix
#     assert all(n.startswith("docs.") for n in names_docs)
#     assert all(n.startswith("wisdom.") for n in names_wis)

#     # And there should be no overlap
#     assert not names_docs & names_wis