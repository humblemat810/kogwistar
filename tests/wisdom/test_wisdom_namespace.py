
import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Node,
    PureChromaEdge,
    PureChromaNode,
    PureGraph,
)
from kogwistar.server.auth_middleware import claims_ctx
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = pytest.mark.ci_full

def test_puregraph_persist(tmp_path):
    with GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=ConstantEmbeddingFunction(dim=384),
    ) as eng:
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


def test_puregraph_persistence_scopes_node_and_edge_projections(tmp_path):
    with GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        namespace="wisdom",
        kg_graph_type="wisdom",
        embedding_function=ConstantEmbeddingFunction(dim=2),
        backend_factory=build_fake_backend,
    ) as eng:
        node = PureChromaNode(
            id="wisdom-node",
            label="Lesson",
            type="entity",
            summary="A Wisdom lesson",
        )
        edge = PureChromaEdge(
            id="wisdom-edge",
            label="relates",
            type="relationship",
            summary="A relation",
            relation="related_to",
            source_ids=[node.id],
            target_ids=[node.id],
            source_edge_ids=[],
            target_edge_ids=[],
        )

        eng.persist_graph(
            parsed=PureGraph(nodes=[node], edges=[edge]),
            session_id="wisdom-namespace-contract",
        )

        node_metadata = eng.backend.node_get(
            ids=[node.id], include=["metadatas"]
        )["metadatas"][0]
        edge_metadata = eng.backend.edge_get(
            ids=[edge.id], include=["metadatas"]
        )["metadatas"][0]
        assert node_metadata["namespace"] == "wisdom"
        assert edge_metadata["namespace"] == "wisdom"


def test_acl_enabled_puregraph_persistence_records_authoritative_acl(tmp_path):
    with GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        namespace="wisdom",
        kg_graph_type="wisdom",
        embedding_function=ConstantEmbeddingFunction(dim=2),
        backend_factory=build_fake_backend,
        acl_enabled=True,
    ) as eng:
        node = PureChromaNode(
            id="acl-wisdom-node",
            label="Private lesson",
            type="entity",
            summary="A private Wisdom lesson",
        )
        edge = PureChromaEdge(
            id="acl-wisdom-edge",
            label="relates",
            type="relationship",
            summary="A private relation",
            relation="related_to",
            source_ids=[node.id],
            target_ids=[node.id],
            source_edge_ids=[],
            target_edge_ids=[],
        )

        owner_token = claims_ctx.set({"agent_id": "wisdom-owner", "ns": "wisdom"})
        try:
            eng.persist_graph(
                parsed=PureGraph(nodes=[node], edges=[edge]),
                session_id="acl-wisdom-namespace-contract",
            )
        finally:
            claims_ctx.reset(owner_token)

        for entity_id in (node.id, edge.id):
            acl_rows = eng.raw_read.get_nodes(
                node_type=Node,
                where={
                    "$and": [
                        {"entity_type": "acl_record"},
                        {"acl_truth_graph": "wisdom"},
                        {"acl_target_entity_id": entity_id},
                    ]
                },
            )
            assert any(row.metadata["acl_target_grain"] in {"node", "edge"} for row in acl_rows)

        owner_token = claims_ctx.set({"agent_id": "wisdom-owner", "ns": "wisdom"})
        try:
            assert [
                item.id
                for item in eng.read.get_nodes(
                    ids=[node.id], node_type=PureChromaNode
                )
            ] == [node.id]
            assert [
                item.id
                for item in eng.read.get_edges(
                    ids=[edge.id], edge_type=PureChromaEdge
                )
            ] == [edge.id]
        finally:
            claims_ctx.reset(owner_token)

        other_token = claims_ctx.set({"agent_id": "other-user", "ns": "wisdom"})
        try:
            assert eng.read.get_nodes(
                ids=[node.id], node_type=PureChromaNode
            ) == []
            assert eng.read.get_edges(
                ids=[edge.id], edge_type=PureChromaEdge
            ) == []
        finally:
            claims_ctx.reset(other_token)


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
