"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

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
        namespace=fixture["scope"]["namespace"],
        kg_graph_type=fixture["truth_graph"],
        embedding_function=build_test_embedding_function("constant", dim=2),
        acl_enabled=True,
    )
    nodes = {}
    for spec in fixture["nodes"]:
        metadata = {
            "visibility": spec["mode"],
            "graph_space": spec.get("graph_space", "facts"),
        }
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
        metadata = {
            "visibility": spec["mode"],
            "graph_space": spec.get("graph_space", "facts"),
        }
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
            scoped_expected = fixture["graph_space_expected"][principal["id"]]
            scoped_nodes = engine.read.query_nodes(
                query_embeddings=fixture["query_embeddings"],
                n_results=fixture["n_results"],
                where={"graph_space": scoped_expected["graph_space"]},
            )
            scoped_edges = engine.read.query_edges(
                query_embeddings=fixture["query_embeddings"],
                n_results=fixture["n_results"],
                where={"graph_space": scoped_expected["graph_space"]},
            )
            assert [[node.id for node in batch] for batch in scoped_nodes] == (
                scoped_expected["nodes"]
            ), principal["id"]
            assert [[edge.id for edge in batch] for batch in scoped_edges] == (
                scoped_expected["edges"]
            ), principal["id"]
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
