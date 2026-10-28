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
def _claims(role: str, ns: str, sub: str | None = None):
    claims = {"role": role, "ns": ns}
    if sub is not None:
        claims["sub"] = sub
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
    engine.read.query_nodes = lambda **_kwargs: []
    source = build_entity_node(node_id="mcp-stage1-source", doc_id="mcp-doc")
    target = build_entity_node(node_id="mcp-stage1-target", doc_id="mcp-doc")
    edge = build_relationship_edge(
        edge_id="mcp-stage1-edge", src=source.id, tgt=target.id, doc_id="mcp-doc"
    )
    adapter.add_node(source)
    adapter.add_node(target)
    adapter.add_edge(edge)
    monkeypatch.setattr(mcp_tools, "gq", _FixedResource(GraphQuery(engine)))

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

    assert target.id in neighbors["nodes"]
    assert edge.id in neighbors["edges"]
    assert semantic["seeds"] == []


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
        if "owner_id" in spec:
            metadata["owner_agent_id"] = spec["owner_id"]
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
        edge = build_relationship_edge(
            edge_id=spec["id"],
            src=spec["source_ids"][0],
            tgt=spec["target_ids"][0],
            doc_id="D-vector-acl",
            relation="links",
            metadata={"visibility": spec["mode"]},
        )
        edge.embedding = spec["embedding"]
        edges.append(edge)

    with _claims("rw", fixture["truth_graph"], sub="owner-b"):
        for node in nodes.values():
            engine.write.add_node(node)
        for edge in edges:
            engine.write.add_edge(edge)

    for principal in fixture["principals"]:
        with _claims("ro", fixture["truth_graph"], sub=principal["id"]):
            node_results = engine.read.query_nodes(
                query_embeddings=fixture["query_embeddings"],
                n_results=fixture["n_results"],
            )
            scored_node_results = [
                engine.read.search_nodes_as_of_scored(
                    query_embeddings=[embedding],
                    as_of_ts="2099-01-01T00:00:00+00:00",
                    n_results=fixture["n_results"],
                )
                for embedding in fixture["query_embeddings"]
            ]
            edge_results = engine.read.query_edges(
                query_embeddings=fixture["query_embeddings"],
                n_results=fixture["n_results"],
            )
        assert [[node.id for node in batch] for batch in node_results] == principal[
            "expected"
        ]["nodes"]
        assert [[edge.id for edge in batch] for batch in edge_results] == principal[
            "expected"
        ]["edges"]
        for query_index, hits in enumerate(scored_node_results):
            assert [hit.node.id for hit in hits] == principal["expected"]["nodes"][
                query_index
            ]
            assert [hit.raw_distance for hit in hits] == pytest.approx(
                principal["expected"]["distances"]["nodes"][query_index]
            )
            assert [hit.similarity for hit in hits] == pytest.approx(
                [1.0 - distance for distance in principal["expected"]["distances"]["nodes"][query_index]]
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
        got = _structured(
            await server.mcp.call_tool(
                "workflow.run_checkpoint_get", {"run_id": "diag-run", "step_seq": 0}
            )
        )
        assert got["state"]["answer"] == "draft"

        # replay_to needs at least one checkpoint and no subsequent steps for target=0
        replay = _structured(
            await server.mcp.call_tool(
                "workflow.run_replay", {"run_id": "diag-run", "target_step_seq": 0}
            )
        )
        assert replay["state"]["answer"] == "draft"


@pytest.mark.asyncio
async def test_mcp_workflow_runtime_submit_cancel(monkeypatch, engine_triplet):
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
