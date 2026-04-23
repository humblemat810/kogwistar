
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    PureChromaNode,
    PureChromaEdge,
    PureGraph,
)
from tests._helpers.embeddings import ConstantEmbeddingFunction
import pytest
pytestmark = pytest.mark.ci_full

def test_puregraph_persist(tmp_path):
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=ConstantEmbeddingFunction(dim=384),
    )
    node = PureChromaNode(
        id="nn:wisdom-node-1",
        domain_id="wisdom-math",
        label="T",
        type="entity",
        summary="x",
    )
    edge = PureChromaEdge(
        domain_id="wisdom-math",
        label="E",
        type="relationship",
        summary="r",
        relation="rel",
        source_ids=[node.id],
        target_ids=[node.id],
        source_edge_ids=[],
        target_edge_ids=[],
    )
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
