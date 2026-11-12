from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
async def test_document_id_from_file_name_mcp_matches_shared_contract(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from kogwistar.server import mcp_tools
    from kogwistar.shortids import ShortIdMapper

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "document-id-from-file-name-mcp-v1.json"
    )
    contract = json.loads(fixture_path.read_text(encoding="utf-8"))
    mapper = ShortIdMapper(f"document-id-mcp-{uuid.uuid4()}", str(tmp_path))
    canonical_id = contract["expected_canonical_id"]
    backend = SimpleNamespace(
        document_get=lambda *, ids: {"ids": [canonical_id] if canonical_id in ids else []}
    )
    monkeypatch.setattr(mcp_tools, "engine", _FixedResource(SimpleNamespace(backend=backend)))
    monkeypatch.setattr(mcp_tools.shortids, "_mapper_for_current_run", lambda: mapper)

    with _claims("ro", "docs", scope_parts={"capabilities": ["read_graph"]}):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    mapping = _structured(result)["id_mapping"]
    assert mapping == [
        {
            "file_name": contract["expected_file_name"],
            "id": mapping[0]["id"],
        }
    ]
    assert mapping[0]["id"].startswith(contract["short_id_prefix"])
    assert mapper.s2l_id(mapping[0]["id"]) == canonical_id


@pytest.mark.asyncio
async def test_document_id_from_file_name_denies_scope_and_capability_before_lookup(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from kogwistar.server import mcp_tools
    from kogwistar.shortids import ShortIdMapper

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "document-id-from-file-name-mcp-v1.json"
    )
    contract = json.loads(fixture_path.read_text(encoding="utf-8"))
    mapper = ShortIdMapper(f"document-id-auth-{uuid.uuid4()}", str(tmp_path))
    canonical_id = contract["expected_canonical_id"]
    lookup_calls: list[list[str]] = []

    def document_get(*, ids: list[str]) -> dict[str, list[str]]:
        lookup_calls.append(ids)
        return {"ids": [canonical_id] if canonical_id in ids else []}

    backend = SimpleNamespace(document_get=document_get)
    monkeypatch.setattr(
        mcp_tools, "engine", _FixedResource(SimpleNamespace(backend=backend))
    )
    monkeypatch.setattr(mcp_tools.shortids, "_mapper_for_current_run", lambda: mapper)

    for role, namespace, capabilities in (
        ("ro", "docs", []),
        ("ro", "conversation", ["read_graph"]),
        ("rw", "docs", ["write_graph"]),
    ):
        with _claims(role, namespace, scope_parts={"capabilities": capabilities}):
            with pytest.raises(PermissionError, match="not permitted"):
                await server.mcp._execute(
                    contract["operation"],
                    contract["arguments"],
                    enforce_visibility=True,
                )
    assert lookup_calls == []

    with _claims("ro", "docs", scope_parts={"capabilities": ["read_graph"]}):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert _structured(result)["id_mapping"][0]["file_name"] == canonical_id
    assert lookup_calls == [[canonical_id]]

    # Legacy tokens without explicit capability claims retain Python defaults.
    with _claims("ro", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert _structured(result)["id_mapping"][0]["file_name"] == canonical_id
    assert lookup_calls == [[canonical_id], [canonical_id]]


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


@pytest.mark.asyncio
async def test_propose_vector_returns_only_acl_visible_pairs(monkeypatch, tmp_path):
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
            proposal_calls.append(None)
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

    proposal_calls: list[None] = []
    monkeypatch.setattr(mcp_tools, "engine", _FixedResource(engine))
    monkeypatch.setattr(mcp_tools, "VectorProposer", _FakeProposer)

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                fixture["tool"],
                {"inp": fixture["request"]},
                enforce_visibility=True,
            )
    assert proposal_calls == []

    with _claims("ro", "docs", sub="proposal-reader"):
        result = mcp_tools.propose_vector(
            mcp_tools.ProposeVectorIn.model_validate(fixture["request"])
        )

    assert [pair.model_dump() for pair in result.pairs] == fixture["expected_pairs"]
    assert len(proposal_calls) == 1

    with _claims("ro", "docs", sub="proposal-reader"):
        mcp_result = await server.mcp.call_tool(
            fixture["tool"], fixture["mcp_arguments"]
        )

    assert mcp_result.is_error is False
    assert _structured(mcp_result) == fixture["expected_result"]
    assert len(proposal_calls) == 2


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
    _, source_span_ids = engine.acl.usage_ids_for_item(source_restricted)
    assert len(source_span_ids) == 1
    with _claims("rw", fixture["truth_graph"], sub="owner-b"):
        engine.write.add_node(source_restricted)
        engine.acl.record_acl(
            grain="span",
            truth_graph=fixture["truth_graph"],
            entity_id=source_restricted.id,
            target_item_id=source_span_ids[0],
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
async def test_mcp_workflow_run_evidence_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "contracts/golden/run-evidence-v1.json"
    )
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    snapshot = {
        "snapshot_node_id": expected["snapshot_node_id"],
        "metadata": expected["snapshot_metadata"],
        "properties": {
            "evidence_pack_digest": json.dumps(
                expected["source_summary"]["evidence_pack_digest"]
            )
        },
    }
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: {
            "run_id": run_id,
            "workflow_id": "evidence-workflow",
            "conversation_id": expected["conversation_id"],
        },
    )
    monkeypatch.setattr(
        service._conversation_queries,
        "latest_snapshot",
        lambda conversation_id, *, run_id, stage=None: snapshot,
    )

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read", "project_view"]},
    ):
        result = _structured(
            await server.mcp.call_tool(
                "workflow.run_evidence", {"run_id": expected["run_id"]}
            )
        )

    assert result == expected


@pytest.mark.asyncio
async def test_mcp_workflow_run_evidence_denies_scope_and_missing_capabilities(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_reads: list[str] = []
    snapshot_reads: list[str] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: run_reads.append(run_id)
        or {
            "workflow_id": "evidence-workflow",
            "conversation_id": "conversation-1",
        },
    )
    monkeypatch.setattr(
        service._conversation_queries,
        "latest_snapshot",
        lambda conversation_id, **_kwargs: snapshot_reads.append(conversation_id)
        or {},
    )

    with _claims("ro", "conversation", sub="outsider"):
        wrong_scope = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_evidence", arguments={"run_id": "private-run"}
            ),
        )
    assert wrong_scope.is_error is True
    assert "403 Forbidden" in wrong_scope.content[0].text
    assert run_reads == []

    with _claims("ro", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as missing_run_read:
            await server.mcp._execute(
                "workflow.run_evidence",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
    assert missing_run_read.value.status_code == 403
    assert run_reads == []

    run_reads.clear()
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        with pytest.raises(HTTPException) as missing_project_view:
            await server.mcp._execute(
                "workflow.run_evidence",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
    assert missing_project_view.value.status_code == 403
    assert run_reads == ["private-run", "private-run"]
    assert snapshot_reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_lineage_matches_shared_contract(
    monkeypatch, engine_triplet
):
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles

    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-lineage-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    access_checks: list[tuple[str, str]] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: {
            "run_id": run_id,
            "workflow_id": "wf-child",
            "conversation_id": "lineage-conversation",
        },
    )
    monkeypatch.setattr(
        service,
        "workflow_run_lineage",
        lambda run_id: contract["response"],
    )
    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=lambda workflow_id, role: access_checks.append(
            (workflow_id, role)
        ),
    )

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await workflow_mcp.call_tool(
                "workflow.run_lineage",
                {"run_id": contract["response"]["run_id"], "limit": 2},
            )
        )

    assert result == {**contract["response"], "truncated": False}
    assert access_checks == [
        ("wf-child", "ro"),
        ("wf-child", "ro"),
        ("wf-parent", "ro"),
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_run_lineage_denies_scope_capability_and_child_acl(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-lineage-v1.json"
        ).read_text(encoding="utf-8")
    )
    run_reads: list[str] = []
    lineage_reads: list[str] = []
    access_checks: list[str] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: run_reads.append(run_id)
        or {"workflow_id": "wf-child", "conversation_id": "lineage-conversation"},
    )
    monkeypatch.setattr(
        service,
        "workflow_run_lineage",
        lambda run_id: lineage_reads.append(run_id) or contract["response"],
    )

    def check_workflow_access(workflow_id: str, _role: str) -> None:
        access_checks.append(workflow_id)
        if workflow_id == "wf-parent":
            raise HTTPException(status_code=403, detail="workflow denied")

    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=check_workflow_access,
    )

    with _claims("ro", "conversation", sub="alice"):
        wrong_scope = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_lineage",
                arguments={"run_id": contract["response"]["run_id"]},
            ),
        )
    assert wrong_scope.is_error is True
    assert "403 Forbidden" in wrong_scope.content[0].text
    assert run_reads == []

    with _claims("ro", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as missing_capability:
            await workflow_mcp._execute(
                "workflow.run_lineage",
                {"run_id": contract["response"]["run_id"]},
                enforce_visibility=True,
            )
    assert missing_capability.value.status_code == 403
    assert run_reads == []
    assert lineage_reads == []

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        denied_child = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_lineage",
                arguments={"run_id": contract["response"]["run_id"]},
            ),
        )
    assert denied_child.is_error is True
    assert denied_child.content[0].text == "403: workflow denied"
    assert "wf-parent" in access_checks
    assert lineage_reads == [contract["response"]["run_id"]]


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
async def test_mcp_workflow_resume_contract_denies_scope_and_missing_capability(
    monkeypatch, engine_triplet
):
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []

    def resume_contract(run_id: str) -> dict:
        reads.append(run_id)
        return {"run_id": run_id}

    monkeypatch.setattr(service._run_inspection, "resume_contract", resume_contract)
    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=lambda _workflow_id, _role: None,
    )

    with _claims("ro", "conversation", sub="alice"):
        wrong_scope = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_resume_contract",
                arguments={"run_id": "private-run"},
            ),
        )
    assert wrong_scope.is_error is True
    assert "403 Forbidden" in wrong_scope.content[0].text
    assert reads == []

    with _claims("ro", "workflow", sub="alice", scope_parts={"capabilities": []}):
        missing_capability = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_resume_contract",
                arguments={"run_id": "private-run"},
            ),
        )
    assert missing_capability.is_error is True
    assert "403" in missing_capability.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_replay_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-replay-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, int]] = []

    def replay_run(run_id: str, target_step_seq: int) -> dict:
        calls.append((run_id, target_step_seq))
        return contract["result"]

    monkeypatch.setattr(service, "replay_run", replay_run)
    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [(contract["arguments"]["run_id"], 3)]


@pytest.mark.asyncio
async def test_mcp_workflow_run_cancel_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-cancel-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def cancel_run(run_id: str) -> dict:
        calls.append(run_id)
        return contract["result"]

    monkeypatch.setattr(service, "cancel_run", cancel_run)
    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [contract["arguments"]["run_id"]]


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
async def test_mcp_workflow_scheduler_timeline_denies_before_reads(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    timeline_reads: list[dict[str, object]] = []

    def track_timeline(**kwargs: object) -> list[dict[str, object]]:
        timeline_reads.append(kwargs)
        return []

    monkeypatch.setattr(service.scheduler, "timeline", track_timeline)
    arguments = {"run_id": None, "limit": 10}
    with _claims(
        "ro", "workflow", sub="scheduler-no-cap", scope_parts={"capabilities": []}
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            "workflow.scheduler_timeline", arguments, enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="scheduler-wrong-scope",
        scope_parts={"capabilities": ["project_view"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            "workflow.scheduler_timeline", arguments, enforce_visibility=True
        )
    assert timeline_reads == []

    with _claims(
        "rw",
        "workflow",
        sub="scheduler-authorized",
        scope_parts={"capabilities": ["project_view"]},
    ):
        result = await server.mcp._execute(
            "workflow.scheduler_timeline", arguments, enforce_visibility=True
        )
    assert timeline_reads == [{"run_id": None, "limit": 10}]
    assert _structured(result) == {"run_id": None, "events": []}


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
async def test_mcp_workflow_service_declare_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

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
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        service.service_supervisor,
        "declare_service",
        lambda **kwargs: writes.append(kwargs) or contract["result"],
    )

    for role, namespace, capabilities in (
        ("ro", "workflow", ["service.manage"]),
        ("rw", "conversation", ["service.manage"]),
        ("rw", "workflow", []),
    ):
        with _claims(
            role,
            namespace,
            scope_parts={"capabilities": capabilities},
        ), pytest.raises((HTTPException, PermissionError)):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert writes == []

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": ["service.manage"]}
    ):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert writes == [contract["python_service_arguments"]]


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
@pytest.mark.parametrize(
    ("tool", "method", "fixture_name"),
    [
        (
            "workflow.service_enable",
            "enable_service",
            "workflow-service-enable-mcp-v1.json",
        ),
        (
            "workflow.service_disable",
            "disable_service",
            "workflow-service-disable-mcp-v1.json",
        ),
    ],
)
async def test_mcp_workflow_service_lifecycle_denies_unauthorized_effects(
    monkeypatch, engine_triplet, tool, method, fixture_name
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (Path(__file__).resolve().parents[2] / f"contracts/golden/{fixture_name}").read_text(
            encoding="utf-8"
        )
    )
    effects: list[str] = []
    monkeypatch.setattr(
        service.service_supervisor,
        method,
        lambda service_id: effects.append(service_id) or contract["result"],
    )

    for role, namespace, capabilities in (
        ("ro", "workflow", ["service.manage"]),
        ("rw", "conversation", ["service.manage"]),
        ("rw", "workflow", []),
    ):
        with _claims(
            role,
            namespace,
            scope_parts={"capabilities": capabilities},
        ), pytest.raises((HTTPException, PermissionError)):
            await server.mcp._execute(
                tool, contract["arguments"], enforce_visibility=True
            )
    assert effects == []

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": ["service.manage"]}
    ):
        result = await server.mcp._execute(
            tool, contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert effects == [contract["python_service_arguments"]["service_id"]]


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
async def test_mcp_workflow_service_events_deny_before_read(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[tuple[str, int]] = []

    def list_service_events(service_id: str, *, limit: int) -> list[dict[str, str]]:
        reads.append((service_id, limit))
        return [{"event_id": "evt-1"}]

    monkeypatch.setattr(
        service.service_supervisor, "list_service_events", list_service_events
    )
    arguments = {"service_id": "svc-1", "limit": 5}

    for claims_ns, capabilities in [
        ("workflow", []),
        ("workflow", ["service.inspect"]),
        ("workflow", ["project_view"]),
        ("conversation", ["service.inspect", "project_view"]),
    ]:
        reads_before = len(reads)
        with _claims(
            "ro",
            claims_ns,
            sub=f"service-events-denied-{reads_before}",
            scope_parts={"capabilities": capabilities},
        ), pytest.raises((HTTPException, PermissionError)):
            await server.mcp._execute(
                "workflow.service_events", arguments, enforce_visibility=True
            )
        assert len(reads) == reads_before

    with _claims(
        "ro",
        "workflow",
        sub="service-events-authorized",
        scope_parts={"capabilities": ["service.inspect", "project_view"]},
    ):
        result = await server.mcp._execute(
            "workflow.service_events", arguments, enforce_visibility=True
        )

    assert result.is_error is False
    assert _structured(result) == {
        "service_id": "svc-1",
        "events": [{"event_id": "evt-1"}],
    }
    assert reads == [("svc-1", 5)]


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
async def test_mcp_workflow_service_reads_deny_before_lookup(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    lookups: list[tuple[str, str]] = []

    def get_service(service_id: str) -> dict[str, str]:
        lookups.append(("get", service_id))
        return {"service_id": service_id}

    def list_services(*, limit: int) -> list[dict[str, int]]:
        lookups.append(("list", str(limit)))
        return []

    monkeypatch.setattr(service.service_supervisor, "get_service", get_service)
    monkeypatch.setattr(service.service_supervisor, "list_services", list_services)
    operations = [
        ("workflow.service_get", {"service_id": "svc-1"}),
        ("workflow.service_list", {"limit": 20}),
    ]

    for tool, arguments in operations:
        reads_before = len(lookups)
        with _claims(
            "ro", "workflow", sub=f"{tool}-no-cap", scope_parts={"capabilities": []}
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(tool, arguments, enforce_visibility=True)
        assert denied.value.status_code == 403
        assert len(lookups) == reads_before

        with _claims(
            "ro",
            "conversation",
            sub=f"{tool}-wrong-scope",
            scope_parts={"capabilities": ["service.inspect"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(tool, arguments, enforce_visibility=True)
        assert len(lookups) == reads_before

        for capabilities in (["service.inspect"], ["project_view"]):
            with _claims(
                "ro",
                "workflow",
                sub=f"{tool}-missing-{capabilities[0]}",
                scope_parts={"capabilities": capabilities},
            ), pytest.raises(HTTPException) as denied:
                await server.mcp._execute(tool, arguments, enforce_visibility=True)
            assert denied.value.status_code == 403
            assert len(lookups) == reads_before

        with _claims(
            "ro",
            "workflow",
            sub=f"{tool}-authorized",
            scope_parts={"capabilities": ["service.inspect", "project_view"]},
        ):
            result = await server.mcp._execute(
                tool, arguments, enforce_visibility=True
            )
        assert result.is_error is False

    assert lookups == [
        ("get", "svc-1"),
        ("list", "20"),
    ]


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
async def test_mcp_workflow_budget_history_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-budget-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[int] = []

    def budget_history(*, limit):
        calls.append(limit)
        return contract["result"]

    monkeypatch.setattr(service, "budget_history", budget_history)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["limit"]]
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
async def test_mcp_workflow_capabilities_snapshot_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capabilities-snapshot-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    snapshot_calls: list[None] = []

    def capability_kernel_snapshot():
        snapshot_calls.append(None)
        return {
            field: contract["result"][field]
            for field in ("specs", "approvals", "revoked", "audit_log")
        }

    monkeypatch.setattr(service.capability_kernel, "snapshot", capability_kernel_snapshot)

    with _claims(
        "ro",
        "workflow",
        sub=contract["result"]["current_subject"],
        scope_parts={"capabilities": contract["result"]["effective_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert snapshot_calls == [None]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_capabilities_snapshot_denies_before_snapshot(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    snapshot_calls: list[None] = []
    original_snapshot = service.capability_kernel.snapshot

    def track_snapshot():
        snapshot_calls.append(None)
        return original_snapshot()

    monkeypatch.setattr(service.capability_kernel, "snapshot", track_snapshot)
    for capabilities in (["project_view"], ["read_security_scope"]):
        with _claims(
            "ro",
            "workflow",
            sub=f"capabilities-snapshot-denied-{len(capabilities)}",
            scope_parts={"capabilities": capabilities},
        ):
            with pytest.raises(HTTPException) as denied:
                await server.mcp._execute(
                    "workflow.capabilities_snapshot", {}, enforce_visibility=True
                )
            assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="capabilities-snapshot-wrong-namespace",
        scope_parts={"capabilities": ["project_view", "read_security_scope"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.capabilities_snapshot", {}, enforce_visibility=True
            )

    assert snapshot_calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_capability_approve_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-approve-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "rw",
        "workflow",
        sub=contract["expected"]["current_subject"],
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    payload = _structured(result)
    for field in contract["expected"]["required_fields"]:
        assert field in payload
    assert payload["current_subject"] == contract["expected"]["current_subject"]
    assert contract["expected"]["approval"] in payload["approvals"]
    assert set(contract["expected"]["effective_capabilities_include"]).issubset(
        payload["effective_capabilities"]
    )
    audit = payload["audit_log"][-2:]
    assert [entry["action"] for entry in audit] == contract["expected"]["audit_actions"]
    assert [entry["required"] for entry in audit] == contract["expected"]["audit_required"]
    assert all(entry["outcome"] == "allow" for entry in audit)


@pytest.mark.asyncio
async def test_mcp_workflow_capability_approve_denies_without_mutation(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-approve-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    prior_approvals = {
        key: set(capabilities)
        for key, capabilities in service.capability_kernel.approvals.items()
    }
    mutation_calls: list[dict[str, Any]] = []
    original_grant = service.capability_kernel.grant

    def track_grant(**kwargs: Any) -> None:
        mutation_calls.append(kwargs)
        original_grant(**kwargs)

    monkeypatch.setattr(service.capability_kernel, "grant", track_grant)
    for index, capabilities in enumerate(
        (
            ["project_view"],
            ["approve_action"],
            ["approve_action", "project_view"],
            ["approve_action", "read_security_scope"],
        )
    ):
        with _claims(
            "rw",
            "workflow",
            sub=f"approval-missing-capability-{index}",
            scope_parts={"capabilities": capabilities},
        ):
            with pytest.raises(HTTPException) as missing_capability:
                await server.mcp._execute(
                    fixture["tool"], fixture["arguments"], enforce_visibility=True
                )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro",
        "workflow",
        sub="approval-read-only-role",
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                fixture["tool"], fixture["arguments"], enforce_visibility=True
            )

    with _claims(
        "rw",
        "conversation",
        sub="approval-wrong-namespace",
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                fixture["tool"], fixture["arguments"], enforce_visibility=True
            )

    assert mutation_calls == []
    assert service.capability_kernel.approvals == prior_approvals


@pytest.mark.asyncio
async def test_mcp_workflow_capability_revoke_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-revoke-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    service.capability_kernel.grant(
        subject=contract["expected"]["current_subject"],
        action="seed",
        capabilities="spawn_process",
    )

    with _claims(
        "rw",
        "workflow",
        sub=contract["expected"]["current_subject"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    payload = _structured(result)
    for field in contract["expected"]["required_fields"]:
        assert field in payload
    assert payload["current_subject"] == contract["expected"]["current_subject"]
    assert contract["expected"]["revocation"] in payload["revoked"]
    assert set(contract["expected"]["effective_capabilities_exclude"]).isdisjoint(
        payload["effective_capabilities"]
    )
    audit = payload["audit_log"][-2:]
    assert [entry["action"] for entry in audit] == contract["expected"]["audit_actions"]
    assert [entry["required"] for entry in audit] == contract["expected"]["audit_required"]
    assert all(entry["outcome"] == "allow" for entry in audit)


@pytest.mark.asyncio
async def test_mcp_workflow_capability_revoke_denies_without_mutation(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-revoke-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    prior_revoked = {
        subject: set(capabilities)
        for subject, capabilities in service.capability_kernel.revoked.items()
    }
    mutation_calls: list[dict[str, Any]] = []
    original_revoke = service.capability_kernel.revoke

    def track_revoke(**kwargs: Any) -> None:
        mutation_calls.append(kwargs)
        return original_revoke(**kwargs)

    monkeypatch.setattr(service.capability_kernel, "revoke", track_revoke)
    for index, capabilities in enumerate(
        (
            ["project_view"],
            ["approve_action"],
            ["approve_action", "project_view"],
            ["approve_action", "read_security_scope"],
        )
    ):
        with _claims(
            "rw",
            "workflow",
            sub=f"revoke-missing-capability-{index}",
            scope_parts={"capabilities": capabilities},
        ):
            with pytest.raises(HTTPException) as denied:
                await server.mcp._execute(
                    contract["tool"], contract["arguments"], enforce_visibility=True
                )
        assert denied.value.status_code == 403

    with _claims(
        "ro",
        "workflow",
        sub="revoke-read-only-role",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )

    with _claims(
        "rw",
        "conversation",
        sub="revoke-wrong-namespace",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )

    assert mutation_calls == []
    assert service.capability_kernel.revoked == prior_revoked


@pytest.mark.asyncio
async def test_mcp_workflow_capability_revoke_can_revoke_own_snapshot_access(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    subject = "self-revoking-admin"

    with _claims(
        "rw",
        "workflow",
        sub=subject,
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        result = await server.mcp.call_tool(
            "workflow.capability_revoke",
            {"capability": "project_view", "subject": subject},
        )

    assert result.is_error is False
    payload = _structured(result)
    assert {"subject": subject, "capabilities": ["project_view"]} in payload["revoked"]
    assert "project_view" not in payload["effective_capabilities"]
    assert [entry["action"] for entry in payload["audit_log"][-2:]] == [
        "approve_action",
        "project_view",
    ]
    assert all(entry["outcome"] == "allow" for entry in payload["audit_log"][-2:])


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
async def test_mcp_workflow_dead_letter_replay_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letter-replay-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "rw",
        "workflow",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letter_replay_denies_without_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letter-replay-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    scheduler_calls: list[str] = []
    monkeypatch.setattr(
        service.scheduler,
        "dead_letter",
        lambda: scheduler_calls.append("read") or [],
    )

    with _claims(
        "rw",
        "workflow",
        scope_parts={"capabilities": []},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )

    assert denied.value.status_code == 403
    assert scheduler_calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letters_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letters-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    scheduler_calls: list[str] = []
    monkeypatch.setattr(
        service.scheduler,
        "dead_letter",
        lambda: scheduler_calls.append("read") or [],
    )

    with _claims(
        "ro",
        "workflow",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert scheduler_calls == ["read"]


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letters_denies_before_read(monkeypatch, engine_triplet):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letters-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    scheduler_calls: list[str] = []
    monkeypatch.setattr(
        service.scheduler,
        "dead_letter",
        lambda: scheduler_calls.append("read") or [],
    )

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": []}
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "rw",
        "conversation",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert scheduler_calls == []

    assert scheduler_calls == []


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

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"],
                contract["arguments"],
                enforce_visibility=True,
            )

    with _claims("ro", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        authorized = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert authorized.structured_content == contract["expected_structured_content"]


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

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"],
                contract["arguments"],
                enforce_visibility=True,
            )

    with _claims("rw", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        authorized = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert authorized.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_root_graph_mcp_honors_explicit_capability_claims(monkeypatch):
    from kogwistar.server import mcp_tools

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        assert not mcp_tools._tool_allowed(
            "kg_shortest_path", role="ro", namespaces={"docs"}
        )
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                "kg_shortest_path",
                {"src_id": "source", "dst_id": "target"},
                enforce_visibility=True,
            )
        assert mcp_tools._filter_tool_list(
            [{"name": "kg_shortest_path"}]
        ) == []

    with _claims("ro", "docs"):
        assert mcp_tools._tool_allowed(
            "kg_shortest_path", role="ro", namespaces={"docs"}
        )

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        assert mcp_tools._tool_allowed(
            "kg_shortest_path", role="ro", namespaces={"docs"}
        )

    for name, capability in mcp_tools.TOOL_CAPABILITIES.items():
        role = "rw" if capability == "write_graph" else "ro"
        namespace = "wisdom" if name.startswith("wisdom.") else "docs"
        with _claims(role, namespace, scope_parts={"capabilities": []}):
            assert not mcp_tools._tool_allowed(
                name, role=role, namespaces={namespace}
            ), name
        with _claims(role, namespace, scope_parts={"capabilities": [capability]}):
            assert mcp_tools._tool_allowed(
                name, role=role, namespaces={namespace}
            ), name


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

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"],
                contract["arguments"],
                enforce_visibility=True,
            )

    with _claims("rw", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        authorized = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert authorized.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_kg_propose_bruteforce_mcp_matches_shared_contract(monkeypatch):
    from kogwistar.server import mcp_tools
    from types import SimpleNamespace

    nodes = {
        node_id: build_entity_node(node_id=node_id, doc_id="doc-a")
        for node_id in ("n1", "n2")
    }

    class GraphRead:
        @staticmethod
        def node_ids_by_doc(doc_id):
            return list(nodes) if doc_id == "doc-a" else []

        @staticmethod
        def edge_ids_by_doc(_doc_id):
            return []

    class GraphBackend:
        @staticmethod
        def node_get(*, ids, **_kwargs):
            selected = [nodes[node_id] for node_id in ids if node_id in nodes]
            return {
                "documents": [
                    node.model_dump_json(field_mode="backend") for node in selected
                ]
            }

    engine = SimpleNamespace(
        read=GraphRead(), backend=GraphBackend(), metadata=object()
    )
    monkeypatch.setattr(mcp_tools, "engine", _FixedResource(engine))
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "kg-propose-bruteforce-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "kg_propose_bruteforce"

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"],
                contract["arguments"],
                enforce_visibility=True,
            )

    with _claims("ro", "docs"):
        legacy = await server.mcp._execute(
            contract["operation"],
            contract["arguments"],
            enforce_visibility=True,
        )
    assert legacy.structured_content == contract["expected_structured_content"]

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        authorized = await server.mcp._execute(
            contract["operation"],
            contract["arguments"],
            enforce_visibility=True,
        )
    assert authorized.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_kg_semantic_seed_expand_mcp_matches_shared_contract(monkeypatch):
    from kogwistar import shortids
    from kogwistar.graph_query import GraphQuery
    from kogwistar.server import mcp_tools
    from tests.primitives.test_graph_query import FakeEngine

    engine = FakeEngine()
    nodes = {
        node_id: build_entity_node(node_id=node_id, doc_id="doc-a")
        for node_id in ("n1", "n2")
    }
    for node_id, node in nodes.items():
        payload = node.model_dump_json(field_mode="backend")
        engine.node_collection.add(
            ids=[node_id], documents=[payload], metadatas=[{"doc_id": "doc-a"}]
        )
        node_doc = {"id": f"{node_id}::doc-a", "node_id": node_id, "doc_id": "doc-a"}
        engine.node_docs_collection.add(
            ids=[node_doc["id"]],
            documents=[json.dumps(node_doc)],
            metadatas=[node_doc],
        )

    endpoint_rows = []
    for edge_id in ("e1", "e4"):
        edge = build_relationship_edge(
            edge_id=edge_id, src="n1", tgt="n2", doc_id="doc-a"
        )
        engine.edge_collection.add(
            ids=[edge_id],
            documents=[edge.model_dump_json(field_mode="backend")],
            metadatas=[{"doc_id": "doc-a"}],
        )
        for role, node_id in (("src", "n1"), ("tgt", "n2")):
            row = {
                "id": f"{edge_id}::{role}::node::{node_id}",
                "edge_id": edge_id,
                "endpoint_id": node_id,
                "endpoint_type": "node",
                "role": role,
                "doc_id": "doc-a",
            }
            endpoint_rows.append(row)
    engine.edge_endpoints_collection.add(
        ids=[row["id"] for row in endpoint_rows],
        documents=[json.dumps(row) for row in endpoint_rows],
        metadatas=endpoint_rows,
    )

    def query_n2(*, query_texts=None, query_embeddings=None, n_results=5, **_kwargs):
        del query_texts, query_embeddings
        ids = ["n2"][:n_results]
        return {
            "ids": [ids],
                "documents": [
                    [
                        nodes[node_id].model_dump_json(field_mode="backend")
                        for node_id in ids
                    ]
                ],
        }

    engine.backend.node_query = query_n2
    engine.metadata = object()
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))
    monkeypatch.setattr(shortids, "s2l_id", lambda value: value)
    monkeypatch.setattr(shortids, "l2s_doc", lambda value: value)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "kg-semantic-seed-expand-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"], contract["arguments"], enforce_visibility=True
            )

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    actual = result.structured_content
    assert [json.loads(seed)["id"] for seed in actual["seeds"]] == contract[
        "expected_seed_ids"
    ]
    assert sorted(
        json.loads(edge)["id"] for edge in actual["layers"][0]["edges"]
    ) == contract["expected_first_layer_edge_ids"]


@pytest.mark.asyncio
async def test_wisdom_semantic_seed_expand_mcp_contract_and_authorization(monkeypatch):
    from kogwistar.graph_query import GraphQuery
    from kogwistar.server import mcp_tools
    from tests.primitives.test_graph_query import FakeEngine

    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "wisdom-semantic-seed-expand-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine = FakeEngine()
    seed_id = contract["expected_seed_ids"][0]
    edge_id = contract["expected_first_layer_edge_ids"][0]
    seed = build_entity_node(node_id=seed_id, doc_id="wisdom-doc")
    neighbor = build_entity_node(node_id="wisdom-neighbor", doc_id="wisdom-doc")
    for node in (seed, neighbor):
        engine.node_collection.add(
            ids=[node.id],
            documents=[node.model_dump_json(field_mode="backend")],
            metadatas=[{"doc_id": node.doc_id}],
        )
        node_doc = {"id": f"{node.id}::{node.doc_id}", "node_id": node.id, "doc_id": node.doc_id}
        engine.node_docs_collection.add(
            ids=[node_doc["id"]],
            documents=[json.dumps(node_doc)],
            metadatas=[node_doc],
        )
    edge = build_relationship_edge(
        edge_id=edge_id,
        src=seed_id,
        tgt=neighbor.id,
        doc_id="wisdom-doc",
    )
    engine.edge_collection.add(
        ids=[edge_id],
        documents=[edge.model_dump_json(field_mode="backend")],
        metadatas=[{"doc_id": edge.doc_id}],
    )
    endpoint_rows = [
        {
            "id": f"{edge_id}::{role}::node::{node_id}",
            "edge_id": edge_id,
            "endpoint_id": node_id,
            "endpoint_type": "node",
            "role": role,
            "doc_id": edge.doc_id,
        }
        for role, node_id in (("src", seed_id), ("tgt", neighbor.id))
    ]
    engine.edge_endpoints_collection.add(
        ids=[row["id"] for row in endpoint_rows],
        documents=[json.dumps(row) for row in endpoint_rows],
        metadatas=endpoint_rows,
    )
    query_calls = []

    def query_seed(*, query_texts=None, query_embeddings=None, n_results=5, **_kwargs):
        query_calls.append((query_texts, n_results))
        del query_embeddings
        ids = [seed_id][:n_results]
        return {
            "ids": [ids],
            "documents": [[seed.model_dump_json(field_mode="backend")] for _ in ids],
        }

    engine.backend.node_query = query_seed
    engine.metadata = object()
    monkeypatch.setattr(mcp_tools, "wisdom_gq", _FixedResource(GraphQuery(engine)))

    with _claims(
        "ro", "wisdom", sub="wisdom-reader", scope_parts={"capabilities": []}
    ):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"], contract["arguments"], enforce_visibility=True
            )
    assert query_calls == []

    with _claims(
        "ro", "wisdom", sub="wisdom-reader", scope_parts={"capabilities": ["read_graph"]}
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    actual = result.structured_content
    assert [json.loads(seed_json)["id"] for seed_json in actual["seeds"]] == contract[
        "expected_seed_ids"
    ]
    assert [
        json.loads(edge_json)["id"] for edge_json in actual["layers"][0]["edges"]
    ] == contract["expected_first_layer_edge_ids"]
    assert query_calls == [([contract["arguments"]["text"]], contract["arguments"]["top_k"])]


@pytest.mark.asyncio
async def test_kg_viz_d3_mcp_matches_shared_contract(monkeypatch):
    from types import SimpleNamespace

    from kogwistar.server import mcp_tools

    nodes = {
        node_id: build_entity_node(node_id=node_id, doc_id="doc-a")
        for node_id in ("n1", "n2")
    }
    edge = build_relationship_edge(
        edge_id="e1", src="n1", tgt="n2", doc_id="doc-a"
    )
    graph = SimpleNamespace(
        kg_graph_type="knowledge",
        backend=SimpleNamespace(
            node_get=lambda **_kwargs: {"ids": list(nodes)},
            edge_get=lambda **_kwargs: {"ids": [edge.id]},
        ),
        read=SimpleNamespace(
            node_ids_by_doc=lambda doc_id: list(nodes) if doc_id == "doc-a" else [],
            edge_ids_by_doc=lambda doc_id: [edge.id] if doc_id == "doc-a" else [],
            load_node_map=lambda ids, **_kwargs: {
                node_id: nodes[node_id] for node_id in ids if node_id in nodes
            },
            load_edge_map=lambda ids, **_kwargs: {
                edge.id: edge for edge_id in ids if edge_id == edge.id
            },
        ),
    )
    graph.metadata = object()
    monkeypatch.setattr(mcp_tools, "engine", _FixedResource(graph))
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "kg-viz-d3-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"], contract["arguments"], enforce_visibility=True
            )

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    actual = result.structured_content
    assert sorted(node["id"] for node in actual["nodes"]) == contract[
        "expected_node_ids"
    ]
    actual_links = sorted(
        (
            {"source": link["source"], "target": link["target"], "role": link["role"]}
            for link in actual["links"]
        ),
        key=lambda link: (link["source"], link["target"], link["role"]),
    )
    assert actual_links == contract["expected_links"]


@pytest.mark.asyncio
async def test_kg_viz_cytoscape_mcp_matches_shared_contract(monkeypatch):
    from types import SimpleNamespace

    from kogwistar.server import mcp_tools

    nodes = {
        node_id: build_entity_node(node_id=node_id, doc_id="doc-a")
        for node_id in ("n1", "n2")
    }
    edge = build_relationship_edge(
        edge_id="e1", src="n1", tgt="n2", doc_id="doc-a"
    )
    graph = SimpleNamespace(
        kg_graph_type="knowledge",
        backend=SimpleNamespace(
            node_get=lambda **_kwargs: {"ids": list(nodes)},
            edge_get=lambda **_kwargs: {"ids": [edge.id]},
        ),
        read=SimpleNamespace(
            node_ids_by_doc=lambda doc_id: list(nodes) if doc_id == "doc-a" else [],
            edge_ids_by_doc=lambda doc_id: [edge.id] if doc_id == "doc-a" else [],
            load_node_map=lambda ids, **_kwargs: {
                node_id: nodes[node_id] for node_id in ids if node_id in nodes
            },
            load_edge_map=lambda ids, **_kwargs: {
                edge.id: edge for edge_id in ids if edge_id == edge.id
            },
        ),
        metadata=object(),
    )
    monkeypatch.setattr(mcp_tools, "engine", _FixedResource(graph))
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "kg-viz-cytoscape-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"], contract["arguments"], enforce_visibility=True
            )

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    actual = result.structured_content
    actual_elements = []
    for element in actual["elements"]:
        data = element["data"]
        normalized = {"id": data["id"], "classes": element.get("classes", "")}
        if "source" in data:
            normalized["source"] = data["source"]
            normalized["target"] = data["target"]
        actual_elements.append(normalized)
    assert sorted(actual_elements, key=lambda item: item["id"]) == contract[
        "expected_elements"
    ]


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

    with _claims("ro", "docs", scope_parts={"capabilities": []}):
        with pytest.raises(PermissionError, match="not permitted"):
            await server.mcp._execute(
                contract["operation"],
                contract["arguments"],
                enforce_visibility=True,
            )

    with _claims("rw", "docs"):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]

    with _claims(
        "ro", "docs", scope_parts={"capabilities": ["read_graph"]}
    ):
        authorized = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert authorized.structured_content == contract["expected_structured_content"]


@pytest.mark.asyncio
async def test_mcp_workflow_process_views(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-blocked-runs-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    run = fixture["run"]
    process_table_fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-process-table-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    service.run_registry.create_run(
        run_id=run["run_id"],
        conversation_id=run["conversation_id"],
        workflow_id=run["workflow_id"],
        user_id=run["user_id"],
        user_turn_node_id=run["user_turn_node_id"],
        status="running",
    )
    service.run_registry.append_event(
        run["run_id"], "run.stage", {"stage": "execute"}
    )
    service.run_registry.update_status(
        run["run_id"],
        status=run["status"],
        started=True,
    )
    from fastapi import HTTPException

    process_reads: list[int] = []
    meta_store = service._execution_meta_store()
    original_list_server_runs = meta_store.list_server_runs

    def track_run_store_read(
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        process_reads.append(limit)
        return original_list_server_runs(
            status=status,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            limit=limit,
        )

    monkeypatch.setattr(meta_store, "list_server_runs", track_run_store_read)
    with _claims(
        "ro", "workflow", sub="blocked-runs-no-cap", scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                fixture["operation"], fixture["arguments"], enforce_visibility=True
            )
    assert denied.value.status_code == 403
    with _claims(
        "ro", "workflow", sub="process-table-no-cap", scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as denied_process_table:
            await server.mcp._execute(
                process_table_fixture["operation"],
                process_table_fixture["arguments"],
                enforce_visibility=True,
            )
    assert denied_process_table.value.status_code == 403
    assert process_reads == []

    with _claims("ro", "workflow"):
        processes = _structured(
            await server.mcp.call_tool(
                process_table_fixture["operation"],
                process_table_fixture["arguments"],
            )
        )["processes"]
        actual_processes = [
            {
                field: process[field]
                for field in (
                    "process_id",
                    "process_kind",
                    "status",
                    "workflow_id",
                    "conversation_id",
                )
            }
            for process in processes
        ]
        assert actual_processes == process_table_fixture["expected_processes"]

        blocked = _structured(
            await server.mcp.call_tool(
                fixture["operation"], fixture["arguments"]
            )
        )["processes"]
        actual = [
            {
                field: item[field]
                for field in ("process_id", "process_kind", "status")
            }
            for item in blocked
            if item["process_id"] == run["run_id"]
        ]
        assert actual == fixture["expected_processes"]

        timeline = _structured(
            await server.mcp.call_tool(
                "workflow.process_timeline",
                {"run_id": run["run_id"], "limit": 10},
            )
        )
        assert timeline["run_id"] == run["run_id"]
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
async def test_mcp_workflow_run_submit_denies_missing_capabilities_before_admission(
    monkeypatch, engine_triplet
):
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    submissions: list[dict] = []
    monkeypatch.setattr(
        service._run_execution,
        "submit_workflow_run",
        lambda **kwargs: submissions.append(kwargs) or {"status": "queued"},
    )
    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=lambda _workflow_id, _role: None,
    )
    arguments = {
        "workflow_id": "unauthorized-submit-workflow",
        "conversation_id": "unauthorized-submit-conversation",
    }

    for capabilities in (["workflow.run.write"], ["spawn_process"]):
        with _claims(
            "rw",
            "workflow",
            sub="alice",
            scope_parts={"capabilities": capabilities},
        ):
            denied = await workflow_mcp._handle_call_tool(
                None,
                types.CallToolRequestParams(
                    name="workflow.run_submit",
                    arguments=arguments,
                ),
            )
        assert denied.is_error is True
        assert "403" in denied.content[0].text

    assert submissions == []


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
async def test_mcp_workflow_design_edge_delete_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []

    def delete_edge(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service._workflow_design, "workflow_design_delete_edge", delete_edge)
    with _claims(
        "rw",
        "workflow",
        sub=contract["arguments"]["designer_id"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [
        {
            **contract["arguments"],
            "actor_sub": contract["arguments"]["designer_id"],
            "source": "mcp",
        }
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_delete_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_delete_edge",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_upsert_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []

    def upsert_edge(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service._workflow_design, "workflow_design_upsert_edge", upsert_edge)
    with _claims(
        "rw",
        "workflow",
        sub=contract["arguments"]["designer_id"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [
        {
            **contract["arguments"],
            "actor_sub": contract["arguments"]["designer_id"],
            "source": "mcp",
        }
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_upsert_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_upsert_edge",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_history_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []

    def history(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service._workflow_design, "workflow_design_history", history)
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [contract["arguments"]]


@pytest.mark.asyncio
async def test_mcp_workflow_design_history_empty_state_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_design_history_denies_before_read(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_history",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": []},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "rw",
        "conversation",
        sub="alice",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_delete_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    workflow_id = contract["arguments"]["workflow_id"]
    node_id = contract["arguments"]["node_id"]
    designer_id = contract["arguments"]["designer_id"]
    with _claims(
        "rw",
        "workflow",
        sub=designer_id,
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        created = await server.mcp.call_tool(
            "workflow.design_node_upsert",
            {
                "workflow_id": workflow_id,
                "designer_id": designer_id,
                "node_id": node_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
        )
        assert created.is_error is False
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_delete_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_delete_node",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_upsert_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    with _claims(
        "rw",
        "workflow",
        sub=contract["arguments"]["designer_id"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    body = _structured(result)
    for field, expected in contract["fixed_result_fields"].items():
        assert body[field] == expected
    for field in contract["integer_result_fields"]:
        assert isinstance(body[field], int)


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_upsert_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_upsert_node",
        lambda **kwargs: calls.append(kwargs) or contract["fixed_result_fields"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-undo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    arguments = contract["arguments"]
    with _claims(
        "rw",
        "workflow",
        sub=arguments["designer_id"],
        scope_parts={
            "capabilities": [*contract["caller_capabilities"], "write_graph"]
        },
    ):
        created = await server.mcp.call_tool(
            "workflow.design_node_upsert",
            {
                "workflow_id": arguments["workflow_id"],
                "designer_id": arguments["designer_id"],
                "node_id": "undo-node-contract",
                "label": "Start",
                "op": "start",
                "start": True,
            },
        )
        assert created.is_error is False
        result = await server.mcp.call_tool(contract["tool"], arguments)

    assert result.is_error is False
    body = _structured(result)
    for field, expected in contract["fixed_result_fields"].items():
        assert body[field] == expected


@pytest.mark.asyncio
async def test_mcp_workflow_design_redo_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-redo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    arguments = contract["arguments"]
    with _claims(
        "rw",
        "workflow",
        sub=arguments["designer_id"],
        scope_parts={
            "capabilities": [*contract["caller_capabilities"], "write_graph"]
        },
    ):
        created = await server.mcp.call_tool(
            "workflow.design_node_upsert",
            {
                "workflow_id": arguments["workflow_id"],
                "designer_id": arguments["designer_id"],
                "node_id": "redo-node-contract",
                "label": "Start",
                "op": "start",
                "start": True,
            },
        )
        assert created.is_error is False
        undone = await server.mcp.call_tool(
            "workflow.design_undo", arguments
        )
        assert undone.is_error is False
        result = await server.mcp.call_tool(contract["tool"], arguments)

    assert result.is_error is False
    body = _structured(result)
    for field, expected in contract["fixed_result_fields"].items():
        assert body[field] == expected, f"{field}: {body!r}"
    for field in contract["integer_result_fields"]:
        assert isinstance(body[field], int)
    for field, length in contract["array_lengths"].items():
        assert len(body[field]) == length
    assert body["commits"]["1"]["action"] == contract["version_one_action"]
    for field in contract["absent_result_fields"]:
        assert field not in body


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-undo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_undo",
        lambda **kwargs: calls.append(kwargs) or contract["fixed_result_fields"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_redo_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-redo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_redo",
        lambda **kwargs: calls.append(kwargs) or contract["fixed_result_fields"],
    )

    for missing_capability in contract["caller_capabilities"]:
        capabilities = [
            capability
            for capability in contract["caller_capabilities"]
            if capability != missing_capability
        ]
        with _claims(
            "rw", "workflow", sub="alice", scope_parts={"capabilities": capabilities}
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
        assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_message_orphans_repair_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    meta_store = service._execution_meta_store()
    accesses: list[str] = []
    original_list = meta_store.list_projected_lane_messages
    original_update = meta_store.update_projected_lane_message_status

    def track_list(**kwargs: Any) -> Any:
        accesses.append("list")
        return original_list(**kwargs)

    def track_update(**kwargs: Any) -> Any:
        accesses.append("update")
        return original_update(**kwargs)

    monkeypatch.setattr(meta_store, "list_projected_lane_messages", track_list)
    monkeypatch.setattr(meta_store, "update_projected_lane_message_status", track_update)
    arguments = {"inbox_id": "repair-auth", "limit": 10}

    with _claims("rw", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.message_orphans_repair", arguments, enforce_visibility=True
            )
        assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": ["project_view"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.message_orphans_repair", arguments, enforce_visibility=True
            )
    assert accesses == []


@pytest.mark.asyncio
async def test_mcp_workflow_operator_dashboard_denies_before_reads(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    required = {"project_view", "workflow.run.read", "read_security_scope"}
    meta_store = service._execution_meta_store()
    reads: list[str] = []
    original_list_runs = meta_store.list_server_runs
    original_list_messages = meta_store.list_projected_lane_messages

    def track_runs(**kwargs: Any) -> Any:
        reads.append("runs")
        return original_list_runs(**kwargs)

    def track_messages(**kwargs: Any) -> Any:
        reads.append("messages")
        return original_list_messages(**kwargs)

    monkeypatch.setattr(meta_store, "list_server_runs", track_runs)
    monkeypatch.setattr(meta_store, "list_projected_lane_messages", track_messages)
    contract = {
        "operation": "workflow.operator_dashboard",
        "arguments": {"limit": 10},
    }

    for missing in required:
        capabilities = sorted(required - {missing})
        with _claims(
            "ro",
            "workflow",
            sub=f"dashboard-missing-{missing}",
            scope_parts={"capabilities": capabilities},
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                contract["operation"], contract["arguments"], enforce_visibility=True
            )
        assert denied.value.status_code == 403

    for role, namespace in (("ro", "conversation"),):
        with _claims(
            role,
            namespace,
            sub="dashboard-scope-denied",
            scope_parts={"capabilities": sorted(required)},
        ):
            denied = await server.mcp._handle_call_tool(
                None,
                types.CallToolRequestParams(
                    name=contract["operation"], arguments=contract["arguments"]
                ),
            )
        assert denied.is_error is True
    assert reads == []

    with _claims(
        "ro", "workflow", sub="dashboard-authorized", scope_parts={"capabilities": sorted(required)}
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert set(_structured(result)) >= {
        "process_table",
        "operator_inbox",
        "blocked_runs",
        "blocked_flow_graph",
        "message_queue",
        "dead_letters",
        "capabilities",
        "resources",
    }


@pytest.mark.asyncio
async def test_mcp_workflow_operator_inbox_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-operator-inbox-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "ro",
        "workflow",
        sub="operator-inbox-contract",
        scope_parts={"capabilities": ["project_view"]},
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert _structured(result) == contract["expected_result"]


@pytest.mark.asyncio
async def test_mcp_workflow_operator_inbox_denies_before_reads(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    meta_store = service._execution_meta_store()
    reads: list[str] = []
    original_list_messages = meta_store.list_projected_lane_messages
    original_get_nodes = conversation_engine.read.get_nodes

    def track_messages(**kwargs: Any) -> Any:
        reads.append("messages")
        return original_list_messages(**kwargs)

    def track_nodes(*args: Any, **kwargs: Any) -> Any:
        reads.append("nodes")
        return original_get_nodes(*args, **kwargs)

    monkeypatch.setattr(meta_store, "list_projected_lane_messages", track_messages)
    monkeypatch.setattr(conversation_engine.read, "get_nodes", track_nodes)
    arguments = {"inbox_id": "auth-deny", "limit": 10}

    with _claims(
        "ro", "workflow", sub="inbox-no-cap", scope_parts={"capabilities": []}
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            "workflow.operator_inbox", arguments, enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="inbox-wrong-scope",
        scope_parts={"capabilities": ["project_view"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            "workflow.operator_inbox", arguments, enforce_visibility=True
        )
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_budget_snapshot_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-budget-http-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "ro",
        "workflow",
        sub="budget-snapshot-contract",
        scope_parts={"capabilities": contract["required_capabilities"]},
    ):
        result = await server.mcp._execute(
            "workflow.budget_snapshot", {}, enforce_visibility=True
        )

    payload = _structured(result)
    for field in contract["required_fields"]:
        assert field in payload
    for field in contract["required_cost_ledger_fields"]:
        assert field in payload["cost_ledger"]
    assert payload["budget_model"] == contract["budget_model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("workflow.budget_snapshot", {}),
        ("workflow.budget_history", {"limit": 10}),
    ],
)
async def test_mcp_workflow_budget_reads_deny_before_ledger_read(
    monkeypatch, engine_triplet, operation, arguments
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    ledger_reads: list[bool] = []
    original_snapshot = service.cost_ledger.snapshot

    def track_snapshot() -> Any:
        ledger_reads.append(True)
        return original_snapshot()

    monkeypatch.setattr(service.cost_ledger, "snapshot", track_snapshot)
    required = {"project_view", "read_security_scope"}
    for missing in required:
        with _claims(
            "ro",
            "workflow",
            sub=f"budget-missing-{missing}",
            scope_parts={"capabilities": sorted(required - {missing})},
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                operation, arguments, enforce_visibility=True
            )
        assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="budget-wrong-scope",
        scope_parts={"capabilities": sorted(required)},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(operation, arguments, enforce_visibility=True)

    assert ledger_reads == []
    with _claims(
        "rw",
        "workflow",
        sub="budget-authorized-rw",
        scope_parts={"capabilities": sorted(required)},
    ):
        authorized = await server.mcp._execute(
            operation, arguments, enforce_visibility=True
        )
    assert _structured(authorized)
    assert ledger_reads == [True]


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
