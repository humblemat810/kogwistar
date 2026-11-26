"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

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

    class ForbiddenGraphQuery:
        def get(self):
            pytest.fail("denied MCP request reached graph query handler")

    monkeypatch.setattr(mcp_tools, "gq", ForbiddenGraphQuery())
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

    with _claims(
        "ro",
        "docs",
        scope_parts={"capabilities": [], "caps": ["read_graph"]},
    ):
        assert mcp_tools._tool_allowed(
            "kg_shortest_path", role="ro", namespaces={"docs"}
        )

    with _claims(
        "ro",
        "docs",
        scope_parts={"capabilities": "read_graph,write_graph"},
    ):
        assert not mcp_tools._tool_allowed(
            "kg_shortest_path", role="ro", namespaces={"docs"}
        )

    for malformed_namespace in ("other,docs", " docs "):
        with _claims(
            "ro",
            malformed_namespace,
            scope_parts={"capabilities": ["read_graph"]},
        ):
            namespaces = mcp_tools.get_current_namespaces()
            assert not mcp_tools._tool_allowed(
                "kg_shortest_path", role="ro", namespaces=namespaces
            )

    for name, capability in mcp_tools.TOOL_CAPABILITIES.items():
        role = "rw" if capability == "write_graph" else "ro"
        default_namespace = "wisdom" if name.startswith("wisdom.") else "docs"
        allowed_namespaces = mcp_tools.TOOL_NAMESPACE.get(
            name, {default_namespace}
        )
        namespace = next(iter(allowed_namespaces))
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
