import pytest

from graph_knowledge_engine.workflow.design import build_workflow_from_engine

# Adjust this import to your actual engine location:
# e.g. from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine import GraphKnowledgeEngine

from graph_knowledge_engine.models import (
    WorkflowNode,
    WorkflowEdge,
    Grounding,
    Span,
    MentionVerification,
)

def _span():
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test"),
    )

def test_workflow_design_creation_and_persistence(tmp_path):
    wf_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "wf"), kg_graph_type="workflow")
    workflow_id = "wf_demo"

    g = Grounding(spans=[_span()])

    n1 = WorkflowNode(
        id="wf|wf_demo|start",
        label="start",
        type="entity",
        doc_id="wf|wf_demo|start",
        summary="start",
        mentions=[g],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": "a",
            "wf_start": True,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
    )
    n2 = WorkflowNode(
        id="wf|wf_demo|end",
        label="end",
        type="entity",
        doc_id="wf|wf_demo|end",
        summary="end",
        mentions=[g],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": "b",
            "wf_terminal": True,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
    )
    wf_engine.add_node(n1)
    wf_engine.add_node(n2)

    e = WorkflowEdge(
        id="wf|wf_demo|e|start|end|default",
        source_ids=[n1.id],
        target_ids=[n2.id],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id="wf_demo",
        mentions=[g],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": 0,
            "wf_is_default": True,
            "wf_predicate": None,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )
    wf_engine.add_edge(e)

    # Reopen: ensures persistence works
    wf_engine2 = GraphKnowledgeEngine(persist_directory=str(tmp_path / "wf"), kg_graph_type="workflow")

    spec = build_workflow_from_engine(workflow_engine=wf_engine2, workflow_id=workflow_id)

    assert spec.workflow_id == workflow_id
    assert spec.start_node_id == n1.id
    assert n2.id in spec.nodes
    assert spec.out_edges[n1.id][0].dst == n2.id
