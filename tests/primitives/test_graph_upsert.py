import pytest
pytestmark = pytest.mark.ci_full

pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")

from fastapi.testclient import TestClient

from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend


def _rw_headers(client: TestClient) -> dict[str, str]:
    resp = client.post(
        "/auth/dev-token",
        json={"username": "tester", "role": "rw", "ns": "docs,conversation,workflow"},
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['token']}"}

# Import your FastAPI app

DOC_ID = "pytest-doc-upsert-1"


def test_graph_upsert_llm_batch_with_references():
    # clean slate (best-effort)

    from kogwistar.server_mcp_with_admin import app
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    import kogwistar.server_mcp_with_admin as server

    client = TestClient(app)
    headers = _rw_headers(client)

    fake_engine = GraphKnowledgeEngine(
        persist_directory="./.tmp_pytest/graph_upsert_fake",
        kg_graph_type="knowledge",
        embedding_function=ConstantEmbeddingFunction(dim=8),
        backend_factory=build_fake_backend,
    )
    server.engine._value = fake_engine

    client.delete(f"/admin/doc/{DOC_ID}", headers=headers)

    payload = {
        "doc_id": DOC_ID,
        "content": "A short contract: Alice contracts with Bob. Includes a penalty clause.",
        "doc_type": "text",
        "insertion_method": "api_upsert",
        "nodes": [
            {
                "id": "nn:alice",
                "label": "Alice Pty Ltd",
                "type": "entity",
                "summary": "Party A",
                "references": [
                    {
                        "collection_page_url": f"document_collection/{DOC_ID}",
                        "document_page_url": f"document/{DOC_ID}",
                        "doc_id": f"{DOC_ID}",
                        "insertion_method": "test_manual_insert",
                        "start_page": 1,
                        "end_page": 1,
                        "start_char": 18,
                        "end_char": 42,
                        "excerpt": "Alice contracts with Bob",
                    }
                ],
            },
            {
                "id": "nn:bob",
                "label": "Bob Co",
                "type": "entity",
                "summary": "Party B",
                "references": [
                    {
                        "collection_page_url": f"document_collection/{DOC_ID}",
                        "document_page_url": f"document/{DOC_ID}",
                        "doc_id": f"{DOC_ID}",
                        "insertion_method": "test_manual_insert",
                        "start_page": 1,
                        "end_page": 1,
                        "start_char": 39,
                        "end_char": 42,
                        "excerpt": "Bob",
                    }
                ],
            },
        ],
        "edges": [
            {
                "id": "ne:contract_e",
                "label": "Contract relationship",
                "type": "relationship",
                "summary": "Alice contracts with Bob",
                "relation": "contracts_with",
                "source_ids": ["nn:alice"],
                "target_ids": ["nn:bob"],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "references": [
                    {
                        "collection_page_url": f"document_collection/{DOC_ID}",
                        "document_page_url": f"document/{DOC_ID}",
                        "doc_id": f"{DOC_ID}",
                        "insertion_method": "test_manual_insert",
                        "start_page": 1,
                        "end_page": 1,
                        "start_char": 18,
                        "end_char": 42,
                        "excerpt": "Alice contracts with Bob",
                    }
                ],
            },
            {
                "id": "ne:penalty_meta",
                "label": "Penalty references contract",
                "type": "relationship",
                "summary": "Penalty clause refers to the contract edge",
                "relation": "refers_to",
                "source_ids": [],  # required (empty is OK)
                "target_ids": ["nn:bob"],
                "source_edge_ids": ["ne:contract_e"],  # hyperedge
                "target_edge_ids": [],  # required (empty is OK)
                "references": [
                    {
                        "collection_page_url": f"document_collection/{DOC_ID}",
                        "document_page_url": f"document/{DOC_ID}",
                        "doc_id": f"{DOC_ID}",
                        "insertion_method": "test_manual_insert",
                        "start_page": 1,
                        "end_page": 1,
                        "start_char": 55,
                        "end_char": 69,
                        "excerpt": "penalty clause",
                    }
                ],
            },
        ],
    }

    # upsert batch
    r = client.post("/api/graph/upsert", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["document_id"] == DOC_ID
    assert data["nodes_added"] >= 2
    assert data["edges_added"] >= 2
    assert len(data["node_ids"]) >= 2
    assert len(data["edge_ids"]) >= 2

    # minimal smoke test of viz JSON (ensures persistence + endpoints tables are coherent)
    viz = client.get(f"/api/viz/d3.json?doc_id={DOC_ID}&mode=reify", headers=headers)
    assert viz.status_code == 200
    g = viz.json()
    assert "nodes" in g and "links" in g
    assert isinstance(g["nodes"], list) and isinstance(g["links"], list)
    # spot-check there is a contracts_with relation somewhere
    rels = {(l.get("relation") or l.get("label") or "") for l in g.get("links", [])}
    assert ("contracts_with" in rels) or ("Contract relationship" in rels)


import types

from kogwistar.visualization import graph_viz


def test_to_d3_force_many_to_one_hyperedge():
    """
    Sanity check: many sources -> one target should produce one diamond (edge-node)
    with one incoming link per source and one outgoing link to the target.
    """
    # Fake edge-node with 2 sources and 1 target
    edge = types.SimpleNamespace(
        id="E",
        relation="contracts_with",
        label="contracts_with",
        type="edge-node",
        properties={},
        source_ids=["Alice", "Carol"],
        target_ids=["Bob"],
        source_edge_ids=[],
        target_edge_ids=[],
    )

    nodes = [
        types.SimpleNamespace(id="Alice", label="Alice", type="entity", properties={}),
        types.SimpleNamespace(id="Bob", label="Bob", type="entity", properties={}),
        types.SimpleNamespace(id="Carol", label="Carol", type="entity", properties={}),
        types.SimpleNamespace(
            id="E", label="contracts_with", type="edge-node", properties={}
        ),
    ]

    d3 = graph_viz.to_d3_force(nodes, [edge])

    # Collect the (src, tgt, role) triples
    got = {(L["source"], L["target"], L["role"]) for L in d3["links"]}
    want = {
        ("Alice", "E", "src"),
        ("Carol", "E", "src"),
        ("E", "Bob", "tgt"),
    }

    assert got == want, f"Expected {want}, got {got}"
