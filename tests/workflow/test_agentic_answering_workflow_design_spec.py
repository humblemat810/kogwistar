from __future__ import annotations

import json
from pathlib import Path

import pytest

from kogwistar.conversation.agentic_answering_design import (
    AGENTIC_ANSWERING_WORKFLOW_ID,
    DEBUG_RAG_WORKFLOW_ID,
    agentic_answering_expected_ops,
    debug_rag_expected_ops,
    build_agentic_answering_backend_payload,
    build_agentic_answering_frontend_payload,
    build_agentic_answering_workflow_design,
    build_debug_rag_workflow_design,
    materialize_workflow_design_artifact,
)
from kogwistar.conversation.designer import AgenticAnsweringWorkflowDesigner
from kogwistar.conversation.designer import ConversationWorkflowDesigner
from kogwistar.conversation.resolvers import default_resolver
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.design import load_workflow_design
from kogwistar.runtime.models import WorkflowDesignArtifact
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = pytest.mark.workflow


def test_answer_designs_match_shared_runtime_admission_contract():
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "answer-workflow-runtime-contract-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    workflows = contract["workflows"]

    for workflow_id, expected_ops in (
        (AGENTIC_ANSWERING_WORKFLOW_ID, agentic_answering_expected_ops()),
        (DEBUG_RAG_WORKFLOW_ID, debug_rag_expected_ops()),
    ):
        design = (
            build_debug_rag_workflow_design(workflow_id=workflow_id)
            if workflow_id == DEBUG_RAG_WORKFLOW_ID
            else build_agentic_answering_workflow_design(workflow_id=workflow_id)
        )
        actual_ops = {str(node.metadata.get("wf_op")) for node in design.nodes}
        actual_fanout_ops = {
            str(node.metadata.get("wf_op"))
            for node in design.nodes
            if node.metadata.get("wf_fanout") is True
        }
        actual_join_ops = {
            str(node.metadata.get("wf_op"))
            for node in design.nodes
            if node.metadata.get("wf_join") is True
        }
        actual_predicates = {
            str(edge.metadata["wf_predicate"])
            for edge in design.edges
            if edge.metadata.get("wf_predicate")
        }
        node_ops_by_id = {node.id: str(node.metadata.get("wf_op")) for node in design.nodes}
        actual_routes = [
            {
                "from": node_ops_by_id[edge.source_ids[0]],
                "to": node_ops_by_id[edge.target_ids[0]],
                "predicate": edge.metadata.get("wf_predicate"),
                "default": bool(edge.metadata.get("wf_is_default", False)),
                "priority": int(edge.metadata.get("wf_priority", 100)),
                "multiplicity": str(edge.metadata.get("wf_multiplicity", "one")),
            }
            for edge in design.edges
        ]
        def route_order(route: dict[str, object]) -> tuple[object, ...]:
            return (
                route["from"],
                route["to"],
                route["priority"],
                route["predicate"] or "",
            )

        actual_routes.sort(key=route_order)
        expected = set(expected_ops)
        assert workflows[workflow_id]["required_ops"] == list(expected_ops)
        assert actual_ops == expected
        assert design.workflow_version == "v2"
        assert {node.metadata.get("wf_version", "v1") for node in design.nodes} == {
            workflows[workflow_id]["python_op_version"]
        }
        assert all("wf_version" not in edge.metadata for edge in design.edges)
        assert set(workflows[workflow_id]["python_executor_ops"]) == expected
        assert sorted(workflows[workflow_id]["required_fanout_ops"]) == sorted(
            actual_fanout_ops
        )
        assert sorted(workflows[workflow_id]["required_join_ops"]) == sorted(
            actual_join_ops
        )
        assert actual_predicates == set(workflows[workflow_id]["required_predicates"])
        assert actual_routes == sorted(
            workflows[workflow_id]["required_routes"], key=route_order
        )
        assert workflows[workflow_id]["requires_default_route"] is True
        assert any(edge.metadata.get("wf_is_default") for edge in design.edges)


def test_agentic_answering_design_round_trips_and_persists(tmp_path):
    workflow_id = AGENTIC_ANSWERING_WORKFLOW_ID

    _design = build_agentic_answering_workflow_design(workflow_id=workflow_id)
    backend_payload = build_agentic_answering_backend_payload(workflow_id=workflow_id)
    frontend_payload = build_agentic_answering_frontend_payload(workflow_id=workflow_id)
    validated = WorkflowDesignArtifact.model_validate(backend_payload)

    assert validated.workflow_id == workflow_id
    assert validated.start_node_id == f"wf:{workflow_id}:start"
    assert len(validated.nodes) == len(agentic_answering_expected_ops())
    assert len(validated.edges) == 14
    assert json.loads(json.dumps(frontend_payload))

    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "workflow"),
        kg_graph_type="workflow",
        embedding_function=ConstantEmbeddingFunction(dim=8),
        backend_factory=build_fake_backend,
    )

    materialize_workflow_design_artifact(workflow_engine, validated)

    start, nodes, adj, rev_adj = load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )

    assert start.id == validated.start_node_id
    assert start.metadata["wf_start"] is True
    assert set(node.metadata["wf_op"] for node in nodes.values()) == set(
        agentic_answering_expected_ops()
    )
    assert len(adj[start.id]) == 1
    assert rev_adj[validated.start_node_id] == []


def test_answer_designer_rebuilds_persisted_operation_version_mismatch(tmp_path):
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "workflow"),
        kg_graph_type="workflow",
        embedding_function=ConstantEmbeddingFunction(dim=8),
        backend_factory=build_fake_backend,
    )
    legacy_design = build_agentic_answering_workflow_design()
    for node in legacy_design.nodes:
        node.metadata["wf_version"] = "v2"
        workflow_engine.write.add_node(node)
    for edge in legacy_design.edges:
        workflow_engine.write.add_edge(edge)

    designer = AgenticAnsweringWorkflowDesigner(
        workflow_engine=workflow_engine,
        predicate_registry={
            "always": lambda *_args: True,
            "aa_should_iterate": lambda *_args: False,
        },
    )
    designer.ensure_answer_flow(workflow_id=AGENTIC_ANSWERING_WORKFLOW_ID)

    nodes = workflow_engine.read.get_nodes(
        where={
            "$and": [
                {"entity_type": "workflow_node"},
                {"workflow_id": AGENTIC_ANSWERING_WORKFLOW_ID},
            ]
        },
        limit=5000,
    )
    assert nodes
    assert {node.metadata.get("wf_version", "v1") for node in nodes} == {"v1"}


def test_conversation_design_operation_versions_match_python_resolver(tmp_path):
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "workflow"),
        kg_graph_type="workflow",
        embedding_function=ConstantEmbeddingFunction(dim=8),
        backend_factory=build_fake_backend,
    )
    designer = ConversationWorkflowDesigner(
        workflow_engine=workflow_engine,
        predicate_registry={
            "always": lambda *_args: True,
            "should_pin_memory": lambda *_args: False,
            "should_pin_kg": lambda *_args: False,
            "should_summarize": lambda *_args: False,
        },
    )
    workflow_id = "conversation.add_turn.v2"

    _start, nodes, _adjacency = designer.ensure_add_turn_flow(
        workflow_id=workflow_id,
        mode="full",
    )
    designer.validate(workflow_id=workflow_id, resolver=default_resolver)

    assert nodes
    assert {node.metadata.get("wf_version", "v1") for node in nodes.values()} == {
        "v1"
    }
