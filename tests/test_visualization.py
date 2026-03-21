from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from kogwistar.visualization.basic_visualization import Visualizer

from kogwistar.utils.kge_debug_dump import dump_d3_bundle
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
from tests._helpers.fake_backend import build_fake_backend

@pytest.mark.ci
def test_pretty_print_graph():
    doc_id = "doc::test::pretty_print_graph"
    engine = GraphKnowledgeEngine(
        persist_directory=str(Path("test_pretty_print_graph") / "kg"),
        backend_factory=build_fake_backend,
        embedding_function=lambda texts: [[0.1] * 384 for _ in texts],
    )

    def _span(excerpt: str) -> Span:
        return Span(
            collection_page_url=f"document_collection/{doc_id}",
            document_page_url=f"document/{doc_id}",
            doc_id=doc_id,
            insertion_method="test",
            page_number=1,
            start_char=0,
            end_char=min(len(excerpt), 4),
            excerpt=excerpt,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="human", is_verified=True, score=1.0, notes="test"
            ),
        )

    def _g(excerpt: str) -> Grounding:
        return Grounding(spans=[_span(excerpt)])

    node_a = Node(
        id="A",
        label="Smoking",
        type="entity",
        summary="Smoking",
        mentions=[_g("smok")],
        properties={},
        metadata={"entity_type": "test_node", "level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        doc_id=doc_id,
        level_from_root=0,
    )
    node_b = Node(
        id="B",
        label="Lung Cancer",
        type="entity",
        summary="Lung Cancer",
        mentions=[_g("lung")],
        properties={},
        metadata={"entity_type": "test_node", "level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        doc_id=doc_id,
        level_from_root=0,
    )
    edge = Edge(
        id="E1",
        label="Smoking causes Lung Cancer",
        type="relationship",
        summary="causal",
        relation="causes",
        source_ids=["A"],
        target_ids=["B"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[_g("smok")],
        doc_id=doc_id,
        properties={},
        metadata={"entity_type": "test_edge", "level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )

    engine.add_node(node_a, doc_id=doc_id)
    engine.add_node(node_b, doc_id=doc_id)
    engine.add_edge(edge, doc_id=doc_id)
    visualiser = Visualizer(engine=engine)

    txt = visualiser.pretty_print_graph(by_doc_id=doc_id, include_refs=True)

    assert txt.startswith("Nodes:")
    assert "Smoking" in txt
    assert "Lung Cancer" in txt
    assert "Edges:" in txt
    assert "causes" in txt
    assert "(empty)" not in txt


from kogwistar.engine_core.models import (
    Span,
    Grounding,
    MentionVerification,
)


def _span() -> Span:
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
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(
    *, workflow_id: str, node_id: str, op: str, start=False, terminal=False
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=workflow_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
    )


def _wf_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None,
    priority: int,
    is_default: bool,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


@pytest.mark.ci_full
def test_d3_litmus_workflow_self_and_parallel():

    wf_id = "wf_d3_litmus"
    wf_dir = Path("test_d3_litmus_workflow_self_and_parallel") / "wf"
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(wf_dir), kg_graph_type="workflow"
    )

    # nodes
    workflow_engine.add_node(
        _wf_node(
            workflow_id=wf_id,
            node_id="A",
            op="noop",
        )
    )
    workflow_engine.add_node(
        _wf_node(
            workflow_id=wf_id,
            node_id="B",
            op="noop",
        )
    )

    # edges
    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=wf_id,
            edge_id="A->B-1",
            src="A",
            dst="B",
            priority=100,
            is_default=True,
            predicate=None,
        )
    )

    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=wf_id,
            edge_id="A->B-2",
            src="A",
            dst="B",
            priority=100,
            is_default=True,
            predicate=None,
        )
    )

    workflow_engine.add_edge(
        _wf_edge(
            workflow_id=wf_id,
            edge_id="A->A",
            src="A",
            dst="A",
            priority=100,
            is_default=True,
            predicate=None,
        )
    )

    # dump
    out_dir = Path("test_d3_litmus_workflow_self_and_parallel") / "d3"
    out_dir.mkdir(parents=True, exist_ok=True)
    template_html = Path("kogwistar") / "templates" / "d3.html"
    template_html = template_html.read_text(encoding="utf-8")
    dump_d3_bundle(
        template_html=template_html,
        engine=workflow_engine,
        # doc_id=wf_id,
        out_html=out_dir / "wf_self_and_multiple.html",
    )
    assert (out_dir / "wf_self_and_multiple.html").exists()
