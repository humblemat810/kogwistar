from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

pytestmark = pytest.mark.ci_full

import kogwistar.server_mcp_with_admin as server
from kogwistar.conversation.models import ConversationNode
from kogwistar.engine_core import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.graph_query import GraphQuery
from kogwistar.runtime.models import WorkflowCheckpointNode
from kogwistar.server import resources as server_resources
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
)
from kogwistar.server.run_registry import RunRegistry
from tests._helpers.engine_factories import FakeEmbeddingFunction
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests._helpers.server_fixtures import build_engine_triplet


class _FixedResource:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def __getattr__(self, name):
        return getattr(self._value, name)


@pytest.fixture()
def engine_triplet():
    root = Path(".tmp_chat_mcp_tests") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
        yield build_engine_triplet(
            root=root,
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _configure_server(
    monkeypatch,
    engine,
    conversation_engine,
    workflow_engine,
    answer_runner,
    runtime_runner=None,
):
    registry = RunRegistry(conversation_engine.meta_sqlite)
    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=registry,
        answer_runner=answer_runner,
        runtime_runner=runtime_runner,
    )
    # Patch both the live server module and the underlying resources module.
    # The refactor moved the real service wiring behind lazy resources, so tests
    # need to keep the two module views in sync.
    monkeypatch.setattr(
        server_resources, "engine", _FixedResource(engine), raising=False
    )
    monkeypatch.setattr(
        server_resources,
        "conversation_engine",
        _FixedResource(conversation_engine),
        raising=False,
    )
    monkeypatch.setattr(
        server_resources, "workflow_engine", _FixedResource(workflow_engine), raising=False
    )
    monkeypatch.setattr(
        server_resources, "run_registry", _FixedResource(registry), raising=False
    )
    monkeypatch.setattr(
        server_resources, "chat_service", _FixedResource(service), raising=False
    )
    monkeypatch.setattr(server, "engine", _FixedResource(engine), raising=False)
    monkeypatch.setattr(
        server,
        "conversation_engine",
        _FixedResource(conversation_engine),
        raising=False,
    )
    monkeypatch.setattr(
        server, "workflow_engine", _FixedResource(workflow_engine), raising=False
    )
    monkeypatch.setattr(server, "run_registry", _FixedResource(registry), raising=False)
    monkeypatch.setattr(server, "chat_service", _FixedResource(service), raising=False)
    return service


@contextmanager
def _claims(
    role: str,
    ns: str,
    sub: str | None = None,
    security_scope: str | None = None,
    scope_parts: dict[str, str] | None = None,
):
    claims = {"role": role, "ns": ns}
    if sub is not None:
        claims["sub"] = sub
    if security_scope is not None:
        claims["security_scope"] = security_scope
    if scope_parts is not None:
        claims.update(scope_parts)
    token = claims_ctx.set(claims)
    try:
        yield
    finally:
        claims_ctx.reset(token)


def _structured(result):
    data = getattr(result, "structured_content", None) or getattr(
        result, "structured_content", None
    )
    if data:
        return data
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if isinstance(text, str) and text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {}


@pytest.mark.asyncio
async def test_mcp_stage1_read_and_semantic_visibility(monkeypatch, tmp_path):
    """MCP must expose Stage 1 references but never Stage 1 semantic hits."""
    from kogwistar.server import mcp_tools
    from tests.core.test_two_stage_chroma import _engine

    engine, adapter = _engine(tmp_path)
    engine._iterative_defensive_emb = lambda _text: [0.0, 0.0]
    semantic_queries = []
    engine.read.query_nodes = lambda **kwargs: semantic_queries.append(kwargs) or []
    source = build_entity_node(node_id="mcp-stage1-source", doc_id="mcp-doc")
    target = build_entity_node(node_id="mcp-stage1-target", doc_id="mcp-doc")
    edge = build_relationship_edge(
        edge_id="mcp-stage1-edge", src=source.id, tgt=target.id, doc_id="mcp-doc"
    )
    adapter.add_node(source)
    adapter.add_node(target)
    adapter.add_edge(edge)
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    monkeypatch.setattr(
        mcp_tools.shortids, "s2l_id", lambda doc_id: f"canonical:{doc_id}"
    )

    with _claims("rw", "knowledge", sub="mcp-agent"):
        neighbors = _structured(
            await server.mcp.call_tool("kg_neighbors", {"rid": source.id})
        )
        semantic = _structured(
            await server.mcp.call_tool(
                "kg_semantic_seed_then_expand_text",
                {"text": "stage one pending", "top_k": 5, "hops": 1},
            )
        )
        semantic_by_doc = _structured(
            await server.mcp.call_tool(
                "kg_semantic_seed_then_expand_text",
                {"text": "stage one pending", "doc_ids": "mcp-doc"},
            )
        )

    assert target.id in neighbors["nodes"]
    assert edge.id in neighbors["edges"]
    assert semantic["seeds"] == []
    assert semantic_by_doc["seeds"] == []
    assert semantic_queries[-1]["where"] == {"doc_id": "canonical:mcp-doc"}


@pytest.mark.asyncio
async def test_python_only_mcp_graph_read_does_not_load_rust(monkeypatch, tmp_path):
    from kogwistar._rust_bridge import RustExtensionUnavailableError
    from kogwistar.server import mcp_tools

    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.setenv(selector, "python")

    def unavailable() -> None:
        raise RustExtensionUnavailableError("native loader must not be called")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "python_only_mcp"),
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        embedding_function=FakeEmbeddingFunction(),
    )
    source = build_entity_node(node_id="python-mcp-source", doc_id="python-mcp-doc")
    target = build_entity_node(node_id="python-mcp-target", doc_id="python-mcp-doc")
    edge = build_relationship_edge(
        edge_id="python-mcp-edge",
        src=source.id,
        tgt=target.id,
        doc_id="python-mcp-doc",
    )
    engine.write.add_node(source)
    engine.write.add_node(target)
    engine.write.add_edge(edge)
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))

    with _claims("rw", "wisdom", sub="python-only-mcp-denied"):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "kg_neighbors", {"rid": source.id}, enforce_visibility=True
            )

    with _claims("rw", "docs", sub="python-only-mcp"):
        neighbors = _structured(
            await server.mcp._execute(
                "kg_neighbors", {"rid": source.id}, enforce_visibility=True
            )
        )

    assert neighbors == {"nodes": [target.id], "edges": [edge.id]}


@pytest.mark.asyncio
async def test_mcp_graph_traversal_matches_shared_golden_contract(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts/golden/graph-traversal-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 1

    graph = fixture["graph"]
    engine, _, _ = engine_triplet
    for node_id in graph["nodes"]:
        engine.write.add_node(
            build_entity_node(
                node_id=node_id,
                doc_id=graph["edges"][0]["doc_id"],
                label=graph["node_labels"][node_id],
            )
        )
    for edge_row in graph["edges"]:
        engine.write.add_edge(
            build_relationship_edge(
                edge_id=edge_row["id"],
                src=edge_row["source_ids"][0],
                tgt=edge_row["target_ids"][0],
                doc_id=edge_row["doc_id"],
                relation=edge_row["relation"],
            )
        )

    from kogwistar.server import mcp_tools

    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    cases = {case["name"]: case for case in fixture["cases"]}
    with _claims("rw", "knowledge", sub="mcp-traversal-contract"):
        edge_case = cases["edge_to_nodes"]
        edge_neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": edge_case["rid"], "doc_id": "D1"}
            )
        )
        assert edge_neighbors == edge_case["expected_neighbors"]

        hop_case = cases["k_hop"]
        hop_result = _structured(
            await server.mcp.call_tool(
                "kg_k_hop",
                {
                    "start_ids": hop_case["start_ids"],
                    "k": hop_case["k"],
                    "doc_id": "D1",
                },
            )
        )
        assert hop_result["layers"] == hop_case["expected_layers"]

        path_case = cases["shortest_path"]
        path_result = _structured(
            await server.mcp.call_tool(
                "kg_shortest_path",
                {
                    "src_id": path_case["src_id"],
                    "dst_id": path_case["dst_id"],
                    "doc_id": "D1",
                },
            )
        )
        assert path_result["path"] == path_case["expected_path"]

        find_case = cases["find_edges"]
        find_result = _structured(
            await server.mcp.call_tool(
                "kg_find_edges",
                {
                    "relation": find_case["relation"],
                    "src_label_contains": find_case["source_label_contains"],
                    "tgt_label_contains": find_case["target_label_contains"],
                    "doc_id": find_case["doc_id"],
                },
            )
        )
        assert find_result["edges"] == find_case["expected_edge_ids"]

        # The MCP neighbors operation intentionally follows the graph query's
        # default and includes the opposite endpoint for node IDs.
        node_neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": "A", "doc_id": "D1"}
            )
        )
        assert node_neighbors == {"nodes": ["B"], "edges": ["E1"]}


@pytest.mark.asyncio
async def test_mcp_neighbors_preserve_acl_visibility_and_hide_topology(
    monkeypatch, tmp_path
):
    from kogwistar.server import mcp_tools

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/graph-acl-traversal-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == 1
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "acl_mcp"),
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        embedding_function=build_test_embedding_function("constant", dim=3),
        acl_enabled=True,
    )
    source_spec, target_spec = fixture["nodes"]
    source = build_entity_node(
        node_id=source_spec["id"],
        doc_id=fixture["doc_id"],
        label="Public source",
        metadata={"visibility": source_spec["mode"]},
    )
    target = build_entity_node(
        node_id=target_spec["id"],
        doc_id=fixture["doc_id"],
        label="Private target",
        metadata={
            "visibility": target_spec["mode"],
            "owner_agent_id": target_spec["owner_id"],
        },
    )
    edge_spec = fixture["edge"]
    edge = build_relationship_edge(
        edge_id=edge_spec["id"],
        src=source.id,
        tgt=target.id,
        doc_id=fixture["doc_id"],
        relation=edge_spec["relation"],
        metadata={
            "visibility": edge_spec["mode"],
            "owner_agent_id": edge_spec["owner_id"],
        },
    )
    owner_token = claims_ctx.set(
        {"agent_id": "owner-a", "sub": "owner-a", "role": "rw", "ns": "knowledge"}
    )
    try:
        engine.write.add_node(source)
        engine.write.add_node(target)
        engine.write.add_edge(edge)
    finally:
        claims_ctx.reset(owner_token)

    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    owner = fixture["principals"][0]
    with _claims("rw", fixture["truth_graph"], sub=owner["id"]):
        owner_neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": source.id, "doc_id": fixture["doc_id"]}
            )
        )
        owner_k_hop = _structured(
            await server.mcp.call_tool(
                "kg_k_hop",
                {"start_ids": [source.id], "k": 2, "doc_id": fixture["doc_id"]},
            )
        )
        owner_path = _structured(
            await server.mcp.call_tool(
                "kg_shortest_path",
                {
                    "src_id": source.id,
                    "dst_id": target.id,
                    "max_depth": 8,
                    "doc_id": fixture["doc_id"],
                },
            )
        )
        owner_edges = _structured(
            await server.mcp.call_tool(
                "kg_find_edges",
                {"relation": edge_spec["relation"], "doc_id": fixture["doc_id"]},
            )
        )
    assert owner_neighbors == owner["expected_neighbors"]
    assert [
        {"nodes": sorted(layer["nodes"]), "edges": sorted(layer["edges"])}
        for layer in owner_k_hop["layers"]
    ] == owner["expected_k_hop"]
    assert owner_path["path"] == owner["expected_shortest_path"]
    assert owner_edges["edges"] == owner["expected_find_edges"]

    reader = fixture["principals"][1]
    with _claims("ro", fixture["truth_graph"], sub=reader["id"]):
        visible_source = engine.read.get_nodes(ids=[source.id])
        assert len(visible_source) == 1
        graph_query = GraphQuery(engine)
        async_reader_neighbors = await graph_query.neighbors_async(
            source.id, doc_id=fixture["doc_id"]
        )
        reader_neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": source.id, "doc_id": fixture["doc_id"]}
            )
        )
        reader_k_hop = _structured(
            await server.mcp.call_tool(
                "kg_k_hop",
                {"start_ids": [source.id], "k": 2, "doc_id": fixture["doc_id"]},
            )
        )
        reader_path = _structured(
            await server.mcp.call_tool(
                "kg_shortest_path",
                {
                    "src_id": source.id,
                    "dst_id": target.id,
                    "max_depth": 8,
                    "doc_id": fixture["doc_id"],
                },
            )
        )
        reader_edges = _structured(
            await server.mcp.call_tool(
                "kg_find_edges",
                {"relation": edge_spec["relation"], "doc_id": fixture["doc_id"]},
            )
        )
    assert reader_neighbors == reader["expected_neighbors"]
    assert async_reader_neighbors == {
        "nodes": set(reader["expected_neighbors"]["nodes"]),
        "edges": set(reader["expected_neighbors"]["edges"]),
    }
    assert [
        {"nodes": sorted(layer["nodes"]), "edges": sorted(layer["edges"])}
        for layer in reader_k_hop["layers"]
    ] == reader["expected_k_hop"]
    assert reader_path["path"] == reader["expected_shortest_path"]
    assert reader_edges["edges"] == reader["expected_find_edges"]


@pytest.mark.asyncio
async def test_mcp_neighbors_hide_private_endpoint_of_visible_edge(monkeypatch, tmp_path):
    from kogwistar.server import mcp_tools

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "acl_visible_edge"),
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        embedding_function=build_test_embedding_function("constant", dim=3),
        acl_enabled=True,
    )
    source = build_entity_node(
        node_id="source-public",
        doc_id="D-visible-edge",
        metadata={"visibility": "public"},
    )
    target = build_entity_node(
        node_id="target-private",
        doc_id="D-visible-edge",
        metadata={"visibility": "private", "owner_agent_id": "owner-a"},
    )
    edge = build_relationship_edge(
        edge_id="edge-public",
        src=source.id,
        tgt=target.id,
        doc_id="D-visible-edge",
        metadata={"visibility": "public"},
    )
    owner_token = claims_ctx.set(
        {"agent_id": "owner-a", "sub": "owner-a", "role": "rw", "ns": "knowledge"}
    )
    try:
        engine.write.add_node(source)
        engine.write.add_node(target)
        engine.write.add_edge(edge)
    finally:
        claims_ctx.reset(owner_token)

    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    with _claims("ro", "knowledge", sub="reader-b"):
        assert [node.id for node in engine.read.get_nodes(ids=[source.id])] == [
            source.id
        ]
        assert not engine.read.get_nodes(ids=[target.id])
        assert not engine.read.get_edges(ids=[edge.id])
        neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": source.id, "doc_id": "D-visible-edge"}
            )
        )
        async_neighbors = await GraphQuery(engine).neighbors_async(
            source.id, doc_id="D-visible-edge"
        )

    assert neighbors == {"nodes": [], "edges": []}
    assert async_neighbors == {"nodes": set(), "edges": set()}


@pytest.mark.asyncio
async def test_mcp_neighbors_keep_same_id_node_and_edge_acl_isolated(
    monkeypatch, tmp_path
):
    from kogwistar.server import mcp_tools

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "same_id_acl_grains"),
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        embedding_function=build_test_embedding_function("constant", dim=3),
        acl_enabled=True,
    )
    shared_id = "same-id-node-edge"
    shared_node = build_entity_node(
        node_id=shared_id,
        doc_id="D-same-id",
        metadata={"visibility": "public"},
    )
    source = build_entity_node(
        node_id="same-id-source",
        doc_id="D-same-id",
        metadata={"visibility": "public"},
    )
    target = build_entity_node(
        node_id="same-id-target",
        doc_id="D-same-id",
        metadata={"visibility": "public"},
    )
    shared_edge = build_relationship_edge(
        edge_id=shared_id,
        src=shared_node.id,
        tgt=target.id,
        doc_id="D-same-id",
        metadata={"visibility": "private", "owner_agent_id": "edge-owner"},
    )
    parent_edge = build_relationship_edge(
        edge_id="same-id-parent-edge",
        src=shared_node.id,
        tgt=source.id,
        doc_id="D-same-id",
        metadata={"visibility": "public"},
    )
    with _claims("rw", "knowledge", sub="edge-owner"):
        engine.write.add_node(shared_node)
        engine.write.add_node(source)
        engine.write.add_node(target)
        engine.write.add_edge(shared_edge)
        engine.write.add_edge(parent_edge)

    node_grounding_ids, node_span_ids = engine.acl.usage_ids_for_item(shared_node)
    edge_grounding_ids, edge_span_ids = engine.acl.usage_ids_for_item(shared_edge)
    assert set(node_grounding_ids).isdisjoint(edge_grounding_ids)
    assert set(node_span_ids).isdisjoint(edge_span_ids)

    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    with _claims("ro", "knowledge", sub="reader"):
        reader_node = engine.read.get_nodes(ids=[shared_id])
        reader_neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": shared_id, "doc_id": "D-same-id"}
            )
        )
    with _claims("ro", "knowledge", sub="edge-owner"):
        owner_neighbors = _structured(
            await server.mcp.call_tool(
                "kg_neighbors", {"rid": shared_id, "doc_id": "D-same-id"}
            )
        )

    assert [node.id for node in reader_node] == [shared_id]
    assert reader_neighbors == {
        "nodes": [source.id],
        "edges": [parent_edge.id],
    }
    assert owner_neighbors == {
        "nodes": sorted([shared_id, source.id, target.id]),
        "edges": sorted([parent_edge.id, shared_id]),
    }


def test_propose_vector_returns_only_acl_visible_pairs(monkeypatch, tmp_path):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/graph-vector-proposal-acl-v1.json"
        ).read_text(encoding="utf-8")
    )
    from kogwistar.server import mcp_tools

    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "acl_vector_proposals"),
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
        embedding_function=build_test_embedding_function("constant", dim=3),
        acl_enabled=True,
    )
    visible_a = build_entity_node(
        node_id=fixture["expected_pairs"][0]["left_id"],
        doc_id="D-proposals",
        metadata={"visibility": "public"},
    )
    visible_b = build_entity_node(
        node_id=fixture["expected_pairs"][0]["right_id"],
        doc_id="D-proposals",
        metadata={"visibility": "public"},
    )
    private = build_entity_node(
        node_id="proposal-private",
        doc_id="D-proposals",
        metadata={"visibility": "private", "owner_agent_id": "proposal-owner"},
    )
    private_endpoint = build_entity_node(
        node_id="proposal-private-endpoint",
        doc_id="D-proposals",
        metadata={"visibility": "private", "owner_agent_id": "proposal-owner"},
    )
    visible_edge = build_relationship_edge(
        edge_id="proposal-visible-edge",
        src=visible_a.id,
        tgt=visible_b.id,
        doc_id="D-proposals",
        metadata={"visibility": "public"},
    )
    hidden_endpoint_edge = build_relationship_edge(
        edge_id="proposal-hidden-endpoint-edge",
        src=visible_a.id,
        tgt=private_endpoint.id,
        doc_id="D-proposals",
        metadata={"visibility": "public"},
    )

    owner_token = claims_ctx.set(
        {
            "agent_id": "proposal-owner",
            "sub": "proposal-owner",
            "role": "rw",
            "ns": "docs",
        }
    )
    try:
        for node in (visible_a, visible_b, private, private_endpoint):
            engine.write.add_node(node)
        engine.write.add_edge(visible_edge)
        engine.write.add_edge(hidden_endpoint_edge)
    finally:
        claims_ctx.reset(owner_token)

    class _FakeProposer:
        def __init__(self, _engine):
            pass

        def generate_merge_candidates(self, **kwargs):
            assert kwargs["new_node"] == fixture["request"]["new_node_ids"]
            assert kwargs["top_k"] == fixture["request"]["top_k"]
            assert kwargs["include_edges"] == fixture["request"]["include_edges"]
            assert kwargs["max_distance"] == fixture["request"]["max_distance"]
            assert kwargs["where"] == fixture["request"]["where"]
            return {
                ("visible", "visible"): (visible_a, visible_b, 0.1),
                ("visible", "private"): (visible_a, private, 0.1),
                ("visible", "hidden-edge"): (visible_a, hidden_endpoint_edge, 0.1),
            }

    monkeypatch.setattr(mcp_tools, "engine", _FixedResource(engine))
    monkeypatch.setattr(mcp_tools, "VectorProposer", _FakeProposer)

    with _claims("ro", "docs", sub="proposal-reader"):
        result = mcp_tools.propose_vector(
            mcp_tools.ProposeVectorIn.model_validate(fixture["request"])
        )

    assert [pair.model_dump() for pair in result.pairs] == fixture["expected_pairs"]


def test_acl_vector_query_matches_shared_python_rust_ranking_fixture(tmp_path):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/graph-acl-vector-query-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "acl_vector_query"),
        backend_factory=build_fake_backend,
        kg_graph_type=fixture["truth_graph"],
        embedding_function=build_test_embedding_function("constant", dim=2),
        acl_enabled=True,
    )
    nodes = {}
    for spec in fixture["nodes"]:
        metadata = {"visibility": spec["mode"]}
        if "security_scope" in spec:
            metadata["security_scope"] = spec["security_scope"]
        if "owner_id" in spec:
            metadata["owner_agent_id"] = spec["owner_id"]
        if "lifecycle_status" in spec:
            metadata["lifecycle_status"] = spec["lifecycle_status"]
        if "redirect_to_id" in spec:
            metadata["redirect_to_id"] = spec["redirect_to_id"]
        node = build_entity_node(
            node_id=spec["id"],
            doc_id="D-vector-acl",
            label=spec["id"],
            metadata=metadata,
        )
        node.embedding = spec["embedding"]
        nodes[node.id] = node

    edges = []
    for spec in fixture["edges"]:
        metadata = {"visibility": spec["mode"]}
        if "security_scope" in spec:
            metadata["security_scope"] = spec["security_scope"]
        if "owner_id" in spec:
            metadata["owner_agent_id"] = spec["owner_id"]
        if "lifecycle_status" in spec:
            metadata["lifecycle_status"] = spec["lifecycle_status"]
        if "redirect_to_id" in spec:
            metadata["redirect_to_id"] = spec["redirect_to_id"]
        edge = build_relationship_edge(
            edge_id=spec["id"],
            src=spec["source_ids"][0],
            tgt=spec["target_ids"][0],
            doc_id="D-vector-acl",
            relation="links",
            metadata=metadata,
        )
        edge.embedding = spec["embedding"]
        edges.append(edge)

    with _claims("rw", fixture["truth_graph"], sub="owner-b"):
        for node in nodes.values():
            engine.write.add_node(node)
        for edge in edges:
            engine.write.add_edge(edge)


    for principal in fixture["principals"]:
        with _claims(
            "ro",
            fixture["truth_graph"],
            sub=principal["id"],
            security_scope=principal.get("security_scope"),
            scope_parts=principal.get("scope_parts"),
        ):
            node_results = engine.read.query_nodes(
                query_embeddings=fixture["query_embeddings"],
                n_results=fixture["n_results"],
            )
            assert [[node.id for node in batch] for batch in node_results] == principal[
                "expected"
            ]["nodes"], principal["id"]
            assert all(
                fixture["current_vector_redirect_contract"]["node"]["source_id"]
                not in [node.id for node in batch]
                for batch in node_results
            )
            assert all(
                fixture["current_vector_redirect_contract"]["node"]["target_id"]
                in [node.id for node in batch]
                for batch in node_results
            )
            edge_results = engine.read.query_edges(
                query_embeddings=fixture["query_embeddings"],
                n_results=fixture["n_results"],
            )
            assert [[edge.id for edge in batch] for batch in edge_results] == principal[
                "expected"
            ]["edges"]
            assert all(
                fixture["current_vector_redirect_contract"]["edge"]["source_id"]
                not in [edge.id for edge in batch]
                for batch in edge_results
            )
            assert all(
                fixture["current_vector_redirect_contract"]["edge"]["target_id"]
                in [edge.id for edge in batch]
                for batch in edge_results
            )
            if principal["id"] == "reader-a":
                limited = engine.read.query_nodes(
                    query_embeddings=[fixture["query_embeddings"][0]], n_results=1
                )
                assert [[node.id for node in batch] for batch in limited] == [[]]
            # Historical scored reads resolve lifecycle and ACL after candidate retrieval.
            scored_node_results = [
                engine.read.search_nodes_as_of_scored(
                    query_embeddings=[embedding],
                    as_of_ts="2099-01-01T00:00:00+00:00",
                    n_results=max(fixture["n_results"], 10),
                )
                for embedding in fixture["query_embeddings"]
            ]
        for query_index, hits in enumerate(scored_node_results):
            expected = principal["expected"]
            expected_nodes = expected.get("scored_nodes", expected["nodes"])
            expected_distances = expected.get(
                "scored_distances", expected["distances"]
            )["nodes"]
            assert [hit.node.id for hit in hits] == expected_nodes[query_index]
            assert [hit.raw_distance for hit in hits] == pytest.approx(
                expected_distances[query_index]
            )
            assert [hit.similarity for hit in hits] == pytest.approx(
                [1.0 - distance for distance in expected_distances[query_index]]
            )

    with _claims("ro", fixture["truth_graph"], sub="owner-b"):
        public_nodes = engine.read.query_nodes(
            query_embeddings=fixture["query_embeddings"],
            n_results=fixture["n_results"],
            where={"visibility": "public"},
        )
    assert [[node.id for node in batch] for batch in public_nodes] == [
        ["visible-public"],
        ["visible-public"],
    ]

    source_restricted = build_entity_node(
        node_id="node-source-span-private",
        doc_id="D-vector-acl-source-span",
        label="node-source-span-private",
        metadata={"visibility": "public"},
    )
    source_restricted.mentions = [Grounding(spans=[Span.from_dummy_for_conversation()])]
    with _claims("rw", fixture["truth_graph"], sub="owner-b"):
        engine.write.add_node(source_restricted)
        engine.acl.record_acl(
            grain="span",
            truth_graph=fixture["truth_graph"],
            entity_id=source_restricted.id,
            target_item_id=f"{source_restricted.id}::mention::0::span::0",
            version=2,
            mode="private",
            owner_id="owner-b",
        )
    with _claims("ro", fixture["truth_graph"], sub="reader-a"):
        source_restricted_hits = engine.read.query_nodes(
            query_embeddings=[[1.0, 0.0]],
            where={"label": "node-source-span-private"},
            n_results=1,
        )
    assert [[node.id for node in batch] for batch in source_restricted_hits] == [[]]


def test_vector_query_requires_exact_source_span_acl_reference(tmp_path):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/graph-vector-source-acl-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "source_acl_query"),
        backend_factory=build_fake_backend,
        kg_graph_type=fixture["truth_graph"],
        embedding_function=build_test_embedding_function("constant", dim=2),
        acl_enabled=True,
    )
    for case in fixture["cases"]:
        node = build_entity_node(
            node_id=case["id"],
            doc_id="D-source-acl",
            label=case["id"],
            metadata={"visibility": "public"},
        )
        node.mentions = [Grounding(spans=[Span.from_dummy_for_conversation()])]
        node.embedding = fixture["query_embedding"]
        item_id = f"{case['id']}::mention::0"
        current_span_id = f"{item_id}::span::0"
        with _claims("rw", fixture["truth_graph"], sub="writer"):
            engine.raw_write.add_node(node)
            engine.acl.record_acl(
                grain="node",
                truth_graph=fixture["truth_graph"],
                entity_id=case["id"],
                version=1,
                mode="public",
            )
            engine.acl.record_acl(
                grain="grounding",
                truth_graph=fixture["truth_graph"],
                entity_id=case["id"],
                target_item_id=item_id,
                version=1,
                mode="public",
            )
            engine.acl.record_acl(
                grain="span",
                truth_graph=fixture["truth_graph"],
                entity_id=case["span_acl_entity_id"],
                target_item_id=case["span_acl_target_item_id"],
                version=1,
                mode="public",
            )

        raw_hits = engine.raw_read.query_nodes(
            query_embeddings=[fixture["query_embedding"]],
            where={"label": case["id"]},
            n_results=1,
        )
        assert [[candidate.id for candidate in batch] for batch in raw_hits] == [
            [case["id"]]
        ]
        with _claims("ro", fixture["truth_graph"], sub="reader"):
            visible_hits = engine.read.query_nodes(
                query_embeddings=[fixture["query_embedding"]],
                where={"label": case["id"]},
                n_results=1,
            )
        expected = [case["id"]] if case["expected_visible"] else []
        assert [[candidate.id for candidate in batch] for batch in visible_hits] == [
            expected
        ], {"case": case["id"], "span_id": current_span_id}


def _success_runner(req: AnswerRunRequest) -> dict:
    assistant_text = f"Assistant reply: {req.user_text}"
    assistant_turn_node_id = f"assistant|{uuid.uuid4().hex}"
    embedding = req.conversation_engine.iterative_defensive_emb(assistant_text)
    req.conversation_engine.write.add_node(
        ConversationNode(
            id=assistant_turn_node_id,
            label="Assistant turn",
            type="entity",
            summary=assistant_text,
            conversation_id=req.conversation_id,
            role="assistant",  # type: ignore[arg-type]
            turn_index=req.prev_turn_meta_summary.tail_turn_index + 1,
            properties={"content": assistant_text, "entity_type": "assistant_turn"},
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={
                "level_from_root": 0,
                "entity_type": "assistant_turn",
                "in_conversation_chain": True,
                "in_ui_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            ),
        )
    )
    req.prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
        assistant_text
    )
    req.prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
    req.prev_turn_meta_summary.tail_turn_index += 1
    return {
        "assistant_turn_node_id": assistant_turn_node_id,
        "assistant_text": assistant_text,
        "workflow_status": "succeeded",
    }


def _cancel_runner(req: AnswerRunRequest) -> dict:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected the run to be cancelled")


def _runtime_success_runner(req: RuntimeRunRequest) -> dict:
    req.publish("run.stage", {"stage": "execute"})
    req.publish(
        "reasoning.summary",
        {"stage": "execute", "summary": "Executing workflow runtime."},
    )
    if req.is_cancel_requested():
        raise RunCancelledError()
    return {
        "workflow_status": "succeeded",
        "final_state": {"workflow_id": req.workflow_id},
    }


def _runtime_cancel_runner(req: RuntimeRunRequest) -> dict:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected runtime workflow to be cancelled")


@pytest.mark.asyncio
async def test_mcp_chat_tool_visibility_by_namespace(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with _claims("rw", "conversation"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "conversation.create" in names
        assert "conversation.ask" in names
        assert "conversation.cancel_run" in names
        assert "workflow.run_replay" not in names
        assert "doc_parse" not in names

    with _claims("ro", "workflow"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "workflow.run_status" in names
        assert "workflow.run_events" in names
        assert "workflow.run_submit" not in names
        assert "workflow.design_history" in names
        assert "workflow.design_undo" not in names
        assert "workflow.run_checkpoint_get" in names
        assert "workflow.run_replay" in names
        assert "workflow.process_table" in names
        assert "workflow.operator_inbox" in names
        assert "workflow.blocked_runs" in names
        assert "workflow.process_timeline" in names
        assert "conversation.ask" not in names


@pytest.mark.asyncio
async def test_mcp_workflow_run_status_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(service, "get_run", lambda run_id: reads.append(run_id))

    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_status", arguments={"run_id": "private-run"}
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_events_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(
        service, "list_run_events", lambda run_id, **_kwargs: reads.append(run_id)
    )

    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_events", arguments={"run_id": "private-run"}
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_events_denies_missing_read_capability_before_read(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(
        service._run_execution,
        "list_run_events",
        lambda run_id, **_kwargs: reads.append(run_id),
    )

    with _claims("ro", "workflow", sub="outsider", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.run_events",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )

    assert denied.value.status_code == 403
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_events_matches_shared_contract(
    monkeypatch, engine_triplet
):
    from pathlib import Path

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-events-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    event = {
        "seq": 7,
        "run_id": contract["arguments"]["run_id"],
        "event_type": "run.started",
        "payload": {"status": "running"},
        "created_at_ms": 1700000000000,
    }
    calls: list[tuple[str, int, int]] = []

    def list_events(run_id: str, *, after_seq: int, limit: int) -> list[dict]:
        calls.append((run_id, after_seq, limit))
        return [event]

    monkeypatch.setattr(service._run_execution, "list_run_events", list_events)
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await server.mcp.call_tool(contract["operation"], contract["arguments"])
        )

    assert set(contract["response_fields"]).issubset(result)
    assert result == {"run_id": event["run_id"], "events": [event]}
    assert set(contract["required_event_fields"]).issubset(result["events"][0])
    assert calls == [(event["run_id"], 4, 200)]


@pytest.mark.asyncio
async def test_mcp_workflow_process_timeline_matches_shared_contract(
    monkeypatch, engine_triplet
):
    from pathlib import Path
    from types import SimpleNamespace

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-process-timeline-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    event = {
        "seq": 9,
        "run_id": contract["arguments"]["run_id"],
        "event_type": "run.stage",
        "payload": {"stage": "execute"},
        "created_at_ms": 1700000000000,
    }
    calls: list[tuple[str, int, int]] = []

    def list_events(run_id: str, *, after_seq: int, limit: int) -> list[dict]:
        calls.append((run_id, after_seq, limit))
        return [event]

    monkeypatch.setattr(
        service,
        "_execution_meta_store",
        lambda: SimpleNamespace(list_server_run_events=list_events),
    )
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await server.mcp.call_tool(contract["operation"], contract["arguments"])
        )

    assert set(contract["response_fields"]).issubset(result)
    assert result == {"run_id": event["run_id"], "events": [event]}
    assert set(contract["required_event_fields"]).issubset(result["events"][0])
    assert calls == [(event["run_id"], 4, 200)]


@pytest.mark.asyncio
async def test_mcp_workflow_process_timeline_denies_wrong_scope_before_read(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[bool] = []
    monkeypatch.setattr(
        service,
        "_execution_meta_store",
        lambda: reads.append(True),
    )

    with _claims(
        "ro",
        "conversation",
        sub="outsider",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.process_timeline",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )

    with _claims("ro", "workflow", sub="outsider", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.process_timeline",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
    assert denied.value.status_code == 403
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_steps_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(service, "get_run", lambda run_id: reads.append(run_id))

    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_steps", arguments={"run_id": "private-run"}
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_checkpoint_get_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[tuple[str, int]] = []

    def read_checkpoint(run_id: str, step_seq: int) -> dict[str, Any]:
        reads.append((run_id, step_seq))
        return {}

    monkeypatch.setattr(service, "get_checkpoint", read_checkpoint)
    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_checkpoint_get",
                arguments={"run_id": "private-run", "step_seq": 4},
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_replay_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[tuple[str, int]] = []

    def replay_run(run_id: str, target_step_seq: int) -> dict[str, Any]:
        reads.append((run_id, target_step_seq))
        return {}

    monkeypatch.setattr(service, "replay_run", replay_run)
    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_replay",
                arguments={"run_id": "private-run", "target_step_seq": 4},
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_steps_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-steps-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    run_id = contract["arguments"]["run_id"]
    monkeypatch.setattr(
        service,
        "get_run",
        lambda requested_id: {"run_id": requested_id, "workflow_id": "workflow-1"},
    )
    monkeypatch.setattr(
        service,
        "list_steps",
        lambda requested_id: contract["result"]["steps"]
        if requested_id == run_id
        else [],
    )

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_run_submit_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-submit-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, object]] = []

    def submit_workflow_run(**kwargs):
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service, "submit_workflow_run", submit_workflow_run)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_scheduler_timeline_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-scheduler-timeline-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, object]] = []

    def list_scheduler_timeline(**kwargs):
        calls.append(kwargs)
        return contract["result"]["events"]

    monkeypatch.setattr(service, "list_scheduler_timeline", list_scheduler_timeline)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_declare_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-declare-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, object]] = []

    def declare_service(**kwargs):
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service, "declare_service", declare_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_disable_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-disable-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def disable_service(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "disable_service", disable_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_enable_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-enable-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def enable_service(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "enable_service", enable_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_events_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-events-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, int]] = []

    def list_service_events(service_id, *, limit):
        calls.append((service_id, limit))
        return contract["result"]["events"]

    monkeypatch.setattr(service, "list_service_events", list_service_events)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["limit"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_get_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-get-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def get_service(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "get_service", get_service)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_heartbeat_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-heartbeat-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def record_service_heartbeat(service_id, *, instance_id, payload):
        calls.append((service_id, instance_id, payload))
        return contract["result"]

    monkeypatch.setattr(service, "record_service_heartbeat", record_service_heartbeat)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["instance_id"],
            contract["python_service_arguments"]["payload"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_list_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-list-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[int] = []

    def list_services(*, limit):
        calls.append(limit)
        return contract["result"]["services"]

    monkeypatch.setattr(service, "list_services", list_services)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["limit"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_repair_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-repair-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def repair_service_projection(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "repair_service_projection", repair_service_projection)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_services_repair_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-services-repair-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[int] = []

    def repair_service_projections(*, limit):
        calls.append(limit)
        return contract["result"]

    monkeypatch.setattr(
        service, "repair_service_projections", repair_service_projections
    )

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["limit"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_tool_audit_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-tool-audit-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str | None, int]] = []

    def list_tool_audit(*, conversation_id, limit):
        calls.append((conversation_id, limit))
        return contract["result"]

    monkeypatch.setattr(service, "list_tool_audit", list_tool_audit)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["conversation_id"],
            contract["python_service_arguments"]["limit"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_visibility_snapshot_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-visibility-snapshot-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[None] = []

    def visibility_snapshot():
        calls.append(None)
        return contract["result"]

    monkeypatch.setattr(service, "visibility_snapshot", visibility_snapshot)

    with _claims(
        "ro",
        "workflow",
        sub=contract["result"]["current_subject"],
        scope_parts={"capabilities": contract["result"]["current_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [None]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_visibility_snapshot_denies_missing_capability_and_namespace(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    scope_reads: list[None] = []
    original_scope_snapshot = service._scope_snapshot

    def track_scope_snapshot():
        scope_reads.append(None)
        return original_scope_snapshot()

    monkeypatch.setattr(service, "_scope_snapshot", track_scope_snapshot)

    with _claims("ro", "workflow", sub="visibility-no-cap", scope_parts={"capabilities": ["project_view"]}):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "workflow.visibility_snapshot", {}, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="visibility-wrong-namespace",
        scope_parts={"capabilities": ["project_view", "read_security_scope"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.visibility_snapshot", {}, enforce_visibility=True
            )

    assert scope_reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_service_trigger_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-trigger-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def trigger_service(service_id, *, trigger_type, payload):
        calls.append((service_id, trigger_type, payload))
        return contract["result"]

    monkeypatch.setattr(service, "trigger_service", trigger_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["trigger_type"],
            contract["python_service_arguments"]["payload"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_create_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-create-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("rw", "conversation"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_ask_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-ask-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("rw", "conversation"):
        await server.mcp.call_tool(
            "conversation.create",
            {
                "user_id": contract["arguments"]["user_id"],
                "conversation_id": contract["arguments"]["conversation_id"],
            },
        )
        result = _structured(
            await server.mcp.call_tool(contract["tool"], contract["arguments"])
        )

    assert result["run_id"]
    assert result["user_turn_node_id"]
    result["run_id"] = "<generated>"
    result["user_turn_node_id"] = "<generated>"
    assert result == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_cancel_run_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-cancel-run-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("rw", "conversation"):
        await server.mcp.call_tool(
            "conversation.create",
            {"user_id": "alice", "conversation_id": "conv-cancel-contract"},
        )
        service.run_registry.create_run(
            run_id=contract["arguments"]["run_id"],
            conversation_id="conv-cancel-contract",
            workflow_id="agentic_answering.v2",
            user_id="alice",
            user_turn_node_id="turn-cancel-contract",
        )
        result = _structured(
            await server.mcp.call_tool(
                contract["tool"], contract["arguments"]
            )
        )

    assert result["status"] == "cancelling"
    assert result["cancel_requested"] is True
    result["created_at_ms"] = "<generated>"
    result["updated_at_ms"] = "<generated>"
    assert result == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_cancel_run_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )
    service.run_registry.create_run(
        run_id="cancel-auth-run",
        conversation_id="cancel-auth-conversation",
        workflow_id="agentic_answering.v2",
        user_id="cancel-auth-owner",
        user_turn_node_id="cancel-auth-turn",
    )
    args = {"run_id": "cancel-auth-run"}

    with _claims(
        "ro",
        "conversation",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": ["workflow.run.write"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.cancel_run", args, enforce_visibility=True
            )

    with _claims(
        "rw",
        "workflow",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": ["workflow.run.write"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.cancel_run", args, enforce_visibility=True
            )

    with _claims(
        "rw",
        "conversation",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": []},
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.cancel_run", args, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    run = service.run_registry.get_run("cancel-auth-run")
    assert run is not None
    assert run["terminal"] is False
    assert run["cancel_requested"] is False


@pytest.mark.asyncio
async def test_mcp_conversation_run_status_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-run-status-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    service.run_registry.create_run(
        run_id=contract["arguments"]["run_id"],
        conversation_id=contract["result"]["conversation_id"],
        workflow_id=contract["result"]["workflow_id"],
        user_id=contract["result"]["user_id"],
        user_turn_node_id=contract["result"]["user_turn_node_id"],
        status=contract["result"]["status"],
    )

    with _claims(
        "ro",
        "conversation",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await server.mcp.call_tool(contract["tool"], contract["arguments"])
        )

    result["created_at_ms"] = "<generated>"
    result["updated_at_ms"] = "<generated>"
    assert result == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_run_status_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service.run_registry.create_run(
        run_id="status-auth-run",
        conversation_id="status-auth-conversation",
        workflow_id="agentic_answering.v2",
        user_id="status-auth-owner",
        user_turn_node_id="status-auth-turn",
        status="running",
    )
    args = {"run_id": "status-auth-run"}

    with _claims(
        "ro",
        "conversation",
        sub="status-auth-owner",
        scope_parts={"capabilities": []},
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.run_status", args, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro",
        "workflow",
        sub="status-auth-owner",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.run_status", args, enforce_visibility=True
            )

    run = service.run_registry.get_run("status-auth-run")
    assert run is not None
    assert run["status"] == "running"


@pytest.mark.asyncio
async def test_mcp_conversation_create_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    tool_args = {
        "user_id": "unauthorized-create",
        "conversation_id": "unauthorized-create-conversation",
        "start_node_id": "unauthorized-create-start",
    }

    with _claims(
        "rw",
        "conversation",
        sub="unauthorized-create",
        scope_parts={"capabilities": []},
    ):
        with pytest.raises(HTTPException):
            await server.mcp._execute(
                "conversation.create", tool_args, enforce_visibility=True
            )

    with _claims(
        "rw",
        "workflow",
        sub="unauthorized-create",
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.create", tool_args, enforce_visibility=True
            )

    assert not conversation_engine.read.get_nodes(
        where={"conversation_id": tool_args["conversation_id"]}, limit=10
    )


@pytest.mark.asyncio
async def test_mcp_conversation_ask_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    owner = "unauthorized-ask-owner"
    conversation_id = "unauthorized-ask-conversation"
    with _claims(
        "rw",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        await server.mcp.call_tool(
            "conversation.create",
            {"user_id": owner, "conversation_id": conversation_id},
        )

    ask_args = {
        "conversation_id": conversation_id,
        "user_id": owner,
        "text": "This must not be admitted",
    }
    with _claims(
        "rw", "conversation", sub=owner, scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.ask", ask_args, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "rw",
        "workflow",
        sub=owner,
        scope_parts={"capabilities": ["send_message"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.ask", ask_args, enforce_visibility=True
            )

    conversation_nodes = conversation_engine.read.get_nodes(
        where={"conversation_id": conversation_id}, limit=10
    )
    assert len(conversation_nodes) == 1
    assert service.run_registry.meta_store.list_server_runs(
        conversation_id=conversation_id, limit=10
    ) == []


@pytest.mark.asyncio
async def test_mcp_conversation_get_transcript_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-get-transcript-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    owner = "transcript-contract-owner"

    with _claims(
        "rw",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        await server.mcp.call_tool(
            "conversation.create",
            {
                "user_id": owner,
                "conversation_id": contract["arguments"]["conversation_id"],
            },
        )

    with _claims(
        "ro",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["read_graph"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_get_transcript_denies_missing_capability_and_namespace(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_id = "transcript-authorization-check"
    owner = "transcript-authorization-owner"
    with _claims(
        "rw",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        await server.mcp.call_tool(
            "conversation.create",
            {"user_id": owner, "conversation_id": conversation_id},
        )

    arguments = {"conversation_id": conversation_id}
    with _claims(
        "ro", "conversation", sub=owner, scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.get_transcript", arguments, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro", "workflow", sub=owner, scope_parts={"capabilities": ["read_graph"]}
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.get_transcript", arguments, enforce_visibility=True
            )


@pytest.mark.asyncio
async def test_wisdom_graph_upsert_mcp_requires_authorized_wisdom_writer(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    from kogwistar.server import mcp_tools

    persist_calls: list[tuple[int, int, str]] = []

    class WisdomStore:
        def persist_graph(self, *, parsed, session_id: str) -> dict[str, object]:
            persist_calls.append((len(parsed.nodes), len(parsed.edges), session_id))
            return {
                "node_ids": [],
                "edge_ids": [],
                "nodes_added": 0,
                "edges_added": 0,
            }

    monkeypatch.setattr(mcp_tools, "wisdom_engine", _FixedResource(WisdomStore()))
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "wisdom-graph-upsert-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "wisdom.kg_upsert_graph"
    arguments = contract["arguments"]

    for role, namespace in (("ro", "wisdom"), ("rw", "docs")):
        with _claims(role, namespace):
            names = {tool.name for tool in await server.mcp.list_tools()}
            assert "wisdom.kg_upsert_graph" not in names
            with pytest.raises(PermissionError):
                await server.mcp._execute(
                    "wisdom.kg_upsert_graph", arguments, enforce_visibility=True
                )
    assert persist_calls == []

    with _claims("rw", "wisdom"):
        names = {tool.name for tool in await server.mcp.list_tools()}
        assert "wisdom.kg_upsert_graph" in names
        result = await server.mcp._execute(
            "wisdom.kg_upsert_graph", arguments, enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]
    assert len(persist_calls) == 1
    assert persist_calls[0][:2] == (0, 0)
    assert persist_calls[0][2].startswith("wisdom:")


@pytest.mark.asyncio
async def test_kg_load_persisted_mcp_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    from kogwistar.server import mcp_tools

    class PersistedGraph:
        def __init__(self, metadata):
            self.metadata = metadata

        def nodes_by_doc(self, doc_id: str, *, where: dict[str, str]) -> list[str]:
            assert doc_id == "doc-a"
            assert where == {"insertion_method": "parser"}
            return ["n2", "n1"]

        def edges_by_doc(self, doc_id: str, *, where: dict[str, str]) -> list[str]:
            assert doc_id == "doc-a"
            assert where == {"insertion_method": "parser"}
            return ["e1"]

    monkeypatch.setattr(
        mcp_tools, "engine", _FixedResource(PersistedGraph(engine.metadata))
    )
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "kg-load-persisted-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "kg_load_persisted"

    with _claims("ro", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_kg_neighbors_mcp_matches_shared_contract(monkeypatch, tmp_path):
    from kogwistar.server import mcp_tools
    from tests.core.test_two_stage_chroma import _engine

    engine, adapter = _engine(tmp_path)
    source = build_entity_node(node_id="n1", doc_id="doc-a")
    target = build_entity_node(node_id="n2", doc_id="doc-a")
    edge = build_relationship_edge(
        edge_id="e1", src=source.id, tgt=target.id, doc_id="doc-a"
    )
    adapter.add_node(source)
    adapter.add_node(target)
    adapter.add_edge(edge)
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "kg-neighbors-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "kg_neighbors"

    with _claims("rw", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_kg_find_edges_mcp_matches_shared_contract(monkeypatch, tmp_path):
    from kogwistar.server import mcp_tools
    from tests.core.test_two_stage_chroma import _engine

    engine, adapter = _engine(tmp_path)
    nodes = [
        build_entity_node(node_id=f"n{index}", doc_id=doc_id)
        for index, doc_id in ((1, "doc-a"), (2, "doc-a"), (3, "doc-b"))
    ]
    for node in nodes:
        adapter.add_node(node)
    adapter.add_edge(
        build_relationship_edge(edge_id="e1", src="n1", tgt="n2", doc_id="doc-a")
    )
    adapter.add_edge(
        build_relationship_edge(edge_id="e2", src="n1", tgt="n3", doc_id="doc-b")
    )
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "kg-find-edges-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "kg_find_edges"

    with _claims("rw", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_kg_k_hop_mcp_matches_shared_contract(monkeypatch, tmp_path):
    from kogwistar.server import mcp_tools
    from tests.core.test_two_stage_chroma import _engine

    engine, adapter = _engine(tmp_path)
    for node_id in ("n1", "n2"):
        adapter.add_node(build_entity_node(node_id=node_id, doc_id="doc-a"))
    adapter.add_edge(
        build_relationship_edge(edge_id="e1", src="n1", tgt="n2", doc_id="doc-a")
    )
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "kg-k-hop-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "kg_k_hop"

    with _claims("rw", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_mcp_workflow_process_views(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    service.run_registry.create_run(
        run_id="run-process-1",
        conversation_id="conv-1",
        workflow_id="wf-1",
        user_id="user-1",
        user_turn_node_id="turn-1",
        status="running",
    )
    service.run_registry.append_event(
        "run-process-1", "run.stage", {"stage": "execute"}
    )
    service.run_registry.update_status(
        "run-process-1",
        status="suspended",
        started=True,
    )

    with _claims("ro", "workflow"):
        processes = _structured(
            await server.mcp.call_tool("workflow.process_table", {"limit": 10})
        )["processes"]
        assert processes
        assert processes[0]["process_id"] == "run-process-1"
        assert processes[0]["process_kind"] == "workflow_run"

        blocked = _structured(
            await server.mcp.call_tool("workflow.blocked_runs", {"limit": 10})
        )["processes"]
        assert any(item["process_id"] == "run-process-1" for item in blocked)

        timeline = _structured(
            await server.mcp.call_tool(
                "workflow.process_timeline", {"run_id": "run-process-1", "limit": 10}
            )
        )
        assert timeline["run_id"] == "run-process-1"
        assert timeline["events"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_tools_round_trip(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )

    with _claims("rw", "workflow", sub="svc-mcp"):
        claims = claims_ctx.get() or {}
        claims["capabilities"] = [
            "spawn_process",
            "workflow.run.read",
            "workflow.run.write",
            "project_view",
            "service.manage",
            "service.inspect",
            "service.heartbeat",
        ]
        created = service.create_conversation(user_id="svc-mcp-user")

    with _claims("rw", "workflow", sub="svc-mcp"):
        claims = claims_ctx.get() or {}
        claims["capabilities"] = [
            "service.manage",
            "service.inspect",
            "service.heartbeat",
            "project_view",
            "spawn_process",
            "workflow.run.read",
            "workflow.run.write",
        ]
        declared = _structured(
            await server.mcp.call_tool(
                "workflow.service_declare",
                {
                    "service_id": "svc.mcp.demo",
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf.service.mcp",
                    "target_config": {"conversation_id": created["conversation_id"]},
                },
            )
        )
        assert declared["service_id"] == "svc.mcp.demo"

        heartbeat = _structured(
            await server.mcp.call_tool(
                "workflow.service_heartbeat",
                {
                    "service_id": "svc.mcp.demo",
                    "instance_id": "mcp-1",
                    "payload": {"beat": 1},
                },
            )
        )
        assert heartbeat["health_status"] == "healthy"

        listed = _structured(
            await server.mcp.call_tool("workflow.service_list", {"limit": 20})
        )["services"]
        assert any(item["service_id"] == "svc.mcp.demo" for item in listed)

        triggered = _structured(
            await server.mcp.call_tool(
                "workflow.service_trigger",
                {"service_id": "svc.mcp.demo", "trigger_type": "external event"},
            )
        )
        assert triggered["current_child_run_id"]

        events = _structured(
            await server.mcp.call_tool(
                "workflow.service_events", {"service_id": "svc.mcp.demo", "limit": 50}
            )
        )["events"]
        assert any(evt["event_type"] == "service.triggered" for evt in events)


def test_chat_run_service_execution_meta_comes_from_conversation_engine(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    assert service._execution_meta_store() is conversation_engine.meta_sqlite
    assert service.run_registry.meta_store is conversation_engine.meta_sqlite


def test_chat_run_service_process_table_uses_claim_scope_not_backend_namespace(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_engine.namespace = "backend-tenant"
    service.run_registry.create_run(
        run_id="run-scope-1",
        conversation_id="conv-scope-1",
        workflow_id="wf-scope-1",
        user_id="user-scope-1",
        user_turn_node_id="turn-scope-1",
        status="running",
    )
    token = claims_ctx.set(
        {
            "ns": "conversation",
            "storage_ns": "store-tenant",
            "execution_ns": "exec-tenant",
            "security_scope": "tenant-a",
        }
    )
    try:
        rows = service.list_process_table(limit=10)
    finally:
        claims_ctx.reset(token)
    assert rows
    row = rows[0]
    assert row["namespace"] == "exec-tenant"
    assert row["storage_namespace"] == "store-tenant"
    assert row["security_scope"] == "tenant-a"


def test_chat_run_service_resume_contract_exposes_checkpoint_metadata(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_engine.write.add_node(
        WorkflowCheckpointNode(
            id="wf_ckpt|run-resume-1|0",
            label="Checkpoint 0",
            type="entity",
            doc_id="wf_ckpt|run-resume-1|0",
            summary="checkpoint",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={},
            metadata={
                "entity_type": "workflow_checkpoint",
                "run_id": "run-resume-1",
                "workflow_id": "wf-resume-1",
                "step_seq": 0,
                "checkpoint_schema_version": 1,
                "state_json": "{\"foo\": \"bar\"}",
            },
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )
    contract = service.resume_contract("run-resume-1")
    assert contract["run_id"] == "run-resume-1"
    assert contract["latest_checkpoint_step_seq"] == 0
    assert contract["checkpoint_schema_version"] == 1
    assert "checkpoint_schema_version" in contract["persisted_keys"]
    assert "_rt_join" in contract["ephemeral_keys"]


def test_chat_run_service_operator_inbox_respects_security_scope(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    token_send = claims_ctx.set(
        {
            "ns": "conversation",
            "storage_ns": "conversation",
            "execution_ns": "conversation",
            "security_scope": "tenant-a",
        }
    )
    try:
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.private",
            payload={"hello": "world"},
            security_scope="tenant-a",
        )
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.shared",
            payload={"hello": "shared"},
            security_scope="tenant-a",
            shared_scope=True,
        )
    finally:
        claims_ctx.reset(token_send)

    token_read = claims_ctx.set(
        {
            "ns": "workflow",
            "storage_ns": "conversation",
            "execution_ns": "conversation",
            "security_scope": "tenant-b",
        }
    )
    try:
        rows = service.list_operator_inbox(inbox_id="inbox:ops", limit=10)
    finally:
        claims_ctx.reset(token_read)
    assert len(rows) == 1
    assert rows[0]["msg_type"] == "request.shared"
    assert rows[0]["visibility"] == "shared"


@pytest.mark.asyncio
async def test_mcp_chat_submit_cancel_and_workflow_diagnostics(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )

    with _claims("rw", "conversation"):
        created = _structured(
            await server.mcp.call_tool("conversation.create", {"user_id": "u-mcp"})
        )
        conversation_id = created["conversation_id"]
        asked = _structured(
            await server.mcp.call_tool(
                "conversation.ask",
                {
                    "conversation_id": conversation_id,
                    "user_id": "u-mcp",
                    "text": "cancel this",
                },
            )
        )
        run_id = asked["run_id"]

        cancelled = _structured(
            await server.mcp.call_tool("conversation.cancel_run", {"run_id": run_id})
        )
        assert cancelled["status"] == "cancelling"
        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(
                await server.mcp.call_tool(
                    "conversation.run_status", {"run_id": run_id}
                )
            )
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Run did not reach cancelled status")

        transcript = _structured(
            await server.mcp.call_tool(
                "conversation.get_transcript", {"conversation_id": conversation_id}
            )
        )
        assert [turn["role"] for turn in transcript["turns"]] == ["user"]

    checkpoint = WorkflowCheckpointNode(
        id="wf_ckpt|diag-run|0",
        label="Checkpoint 0",
        type="entity",
        doc_id="wf_ckpt|diag-run|0",
        summary="workflow checkpoint",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_checkpoint",
            "run_id": "diag-run",
            "workflow_id": "wf.diag",
            "step_seq": 0,
            "state_json": json.dumps({"answer": "draft"}),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conversation_engine.write.add_node(checkpoint)

    with _claims("ro", "workflow"):
        checkpoint_args = {"run_id": "diag-run", "step_seq": 0}
        got = _structured(
            await server.mcp.call_tool(
                "workflow.run_checkpoint_get", checkpoint_args
            )
        )
        checkpoint_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-run-checkpoint-get-mcp-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert checkpoint_contract["operation"] == "workflow.run_checkpoint_get"
        assert set(checkpoint_args) == set(checkpoint_contract["request_fields"])
        assert set(got) == set(checkpoint_contract["response_fields"])
        assert got["run_id"] == "diag-run"
        assert got["step_seq"] == 0
        assert isinstance(got["state"], dict)
        assert got["state"]["answer"] == "draft"

        # replay_to needs at least one checkpoint and no subsequent steps for target=0
        replay = _structured(
            await server.mcp.call_tool(
                "workflow.run_replay", {"run_id": "diag-run", "target_step_seq": 0}
            )
        )
        assert replay["state"]["answer"] == "draft"

        resume_contract = _structured(
            await server.mcp.call_tool(
                "workflow.run_resume_contract", {"run_id": "missing-mcp-run"}
            )
        )
        fixture = json.loads(
            (Path(__file__).resolve().parents[2] / "contracts/golden/run-resume-contract-v1.json")
            .read_text(encoding="utf-8")
        )
        assert set(resume_contract) == set(fixture["required_fields"])
        assert resume_contract["run_id"] == "missing-mcp-run"
        for field, expected in fixture["no_checkpoint"].items():
            assert resume_contract[field] == expected
        assert resume_contract["supported_wait_reasons"] == fixture[
            "supported_wait_reasons"
        ]


@pytest.mark.asyncio
async def test_mcp_workflow_runtime_submit_cancel(monkeypatch, engine_triplet):
    run_status_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-status-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_cancel_runner,
    )

    with _claims("rw", "conversation"):
        created = _structured(
            await server.mcp.call_tool("conversation.create", {"user_id": "u-wf-mcp"})
        )
        conversation_id = created["conversation_id"]

    with _claims("rw", "workflow"):
        submitted = _structured(
            await server.mcp.call_tool(
                "workflow.run_submit",
                {
                    "workflow_id": "wf.runtime.mcp.cancel",
                    "conversation_id": conversation_id,
                    "initial_state": {"counter": 1},
                },
            )
        )
        run_id = submitted["run_id"]
        cancelling = _structured(
            await server.mcp.call_tool("workflow.run_cancel", {"run_id": run_id})
        )
        assert cancelling["status"] == "cancelling"

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(
                await server.mcp.call_tool("workflow.run_status", {"run_id": run_id})
            )
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Workflow run did not reach cancelled status")
        assert set(run_status_contract["required_fields"]).issubset(status)
        assert status["status"] in run_status_contract["terminal_statuses"]
        assert status["terminal"] is True

        events = _structured(
            await server.mcp.call_tool("workflow.run_events", {"run_id": run_id})
        )
        names = [str(evt.get("event_type") or "") for evt in events.get("events", [])]
        assert "run.cancelling" in names
        assert "run.cancelled" in names

        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1


@pytest.mark.asyncio
async def test_mcp_workflow_run_cancel_denies_missing_write_capability(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service.run_registry.create_run(
        run_id="mcp-cancel-auth-run",
        conversation_id="mcp-cancel-auth-conversation",
        workflow_id="mcp-cancel-auth-workflow",
        user_id="alice",
        user_turn_node_id="mcp-cancel-auth-turn",
        status="running",
    )

    with _claims("rw", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.run_cancel",
                {"run_id": "mcp-cancel-auth-run"},
                enforce_visibility=True,
            )
    assert denied.value.status_code == 403

    run = service.run_registry.get_run("mcp-cancel-auth-run")
    assert run is not None
    assert run["status"] == "running"
    assert run["cancel_requested"] is False


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_redo(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    workflow_id = "wf.design.mcp.undo_redo"
    designer_id = "designer-mcp"
    with _claims("rw", "workflow", sub=designer_id):
        n1 = _structured(
            await server.mcp.call_tool(
                "workflow.design_node_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "node_id": f"wf|{workflow_id}|start",
                    "label": "Start",
                    "op": "start",
                    "start": True,
                },
            )
        )
        assert n1["node_id"] == f"wf|{workflow_id}|start"
        n2 = _structured(
            await server.mcp.call_tool(
                "workflow.design_node_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "node_id": f"wf|{workflow_id}|end",
                    "label": "End",
                    "op": "end",
                    "terminal": True,
                },
            )
        )
        assert n2["node_id"] == f"wf|{workflow_id}|end"
        e1 = _structured(
            await server.mcp.call_tool(
                "workflow.design_edge_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "src": f"wf|{workflow_id}|start",
                    "dst": f"wf|{workflow_id}|end",
                    "relation": "wf_next",
                    "is_default": True,
                },
            )
        )
        assert e1["edge_id"] == f"wf|{workflow_id}|e|start_end"

        hist_before = _structured(
            await server.mcp.call_tool(
                "workflow.design_history", {"workflow_id": workflow_id}
            )
        )
        assert int(hist_before["current_version"]) >= 3
        assert bool(hist_before["can_undo"]) is True

        undone = _structured(
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": workflow_id, "designer_id": designer_id},
            )
        )
        assert undone["status"] in {"ok", "noop"}

        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_edge_delete",
                {
                    "workflow_id": workflow_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "designer_id": designer_id,
                },
            )

        redone = _structured(
            await server.mcp.call_tool(
                "workflow.design_redo",
                {"workflow_id": workflow_id, "designer_id": designer_id},
            )
        )
        assert redone["status"] in {"ok", "noop"}

        deleted_after_redo = _structured(
            await server.mcp.call_tool(
                "workflow.design_edge_delete",
                {
                    "workflow_id": workflow_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "designer_id": designer_id,
                },
            )
        )
        assert bool(deleted_after_redo.get("deleted")) is True
        hist_after = _structured(
            await server.mcp.call_tool(
                "workflow.design_history", {"workflow_id": workflow_id}
            )
        )
        timeline_ops = [
            str(item.get("op") or "") for item in hist_after.get("timeline", [])
        ]
        assert "UNDO_APPLIED" in timeline_ops
        assert "REDO_APPLIED" in timeline_ops


@pytest.mark.asyncio
async def test_mcp_workflow_design_requires_designer_id_and_enforces_subject(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with _claims("rw", "workflow", sub="alice"):
        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": "wf.design.mcp.missing_designer"},
            )
        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": "wf.design.mcp.mismatch", "designer_id": "bob"},
            )
