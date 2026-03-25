from __future__ import annotations

import pytest
import json

from kogwistar.conversation.agentic_answering_design import (
    AGENTIC_ANSWERING_WORKFLOW_ID,
    agentic_answering_expected_ops,
    build_agentic_answering_backend_payload,
    build_agentic_answering_frontend_payload,
    build_agentic_answering_workflow_design,
    materialize_workflow_design_artifact,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.design import load_workflow_design
from kogwistar.runtime.models import WorkflowDesignArtifact
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = pytest.mark.workflow


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
