import logging

from graph_knowledge_engine.engine_core.models import Document
from graph_knowledge_engine.visualization.basic_visualization import Visualizer
from joblib import Memory
import pathlib

from graph_knowledge_engine.utils.kge_debug_dump import dump_d3_bundle
from graph_knowledge_engine.runtime.models import WorkflowEdge, WorkflowNode


def test_pretty_print_graph(engine):
    import os
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine

    engine2: GraphKnowledgeEngine = engine
    doc_content = (
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Chlorophyll is the molecule that absorbs sunlight. "
        "Plants perform photosynthesis in their leaves."
    )
    doc = Document(
        id="doc::test::test_pretty_print_graph",
        content=doc_content,
        type="text",
        metadata={"source": "test"},
        processed=False,
        source_map=None,
        embeddings=engine2._iterative_defensive_emb(doc_content),
        domain_id="test-cases",
    )

    # cache ONLY the pure extraction on the doc content

    location = os.path.join(
        ".cache", "test", pathlib.Path(__file__).parts[-1], "_extract_only"
    )
    os.makedirs(location, exist_ok=True)
    memory = Memory(location=location, verbose=0)

    @memory.cache
    def _extract_only(content: str, doc_type):
        return engine2.extract_graph_with_llm(content=content, doc_type="text")

    extracted = _extract_only(
        doc.model_dump(field_mode="dto")["content"], doc_type="text"
    )
    parsed = extracted["parsed"]
    visualiser = Visualizer(engine=engine2)
    # persist deterministically (choose replace/append/skip-if-exists)
    out = engine2.persist_graph_extraction(document=doc, parsed=parsed, mode="replace")

    txt = visualiser.pretty_print_graph(by_doc_id=doc.id, include_refs=True)
    print(txt)
    logging.info(txt)


from graph_knowledge_engine.engine_core.models import (
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


def test_d3_litmus_workflow_self_and_parallel(tmp_path):
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine

    wf_id = "wf_d3_litmus"
    wf_dir = tmp_path / "wf"
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
    out_dir = pathlib.Path(".") / "test_d3_litmus_workflow_self_and_parallel" / "d3"
    out_dir.mkdir(parents=True, exist_ok=True)
    template_html = (
        pathlib.Path(".") / "graph_knowledge_engine" / "templates" / "d3.html"
    ).read_text(encoding="utf-8")
    dump_d3_bundle(
        template_html=template_html,
        engine=workflow_engine,
        # doc_id=wf_id,
        out_html=out_dir / "wf_self_and_multiple.html",
    )
    import os

    os.startfile(str(out_dir))
    assert (out_dir / "wf_self_and_multiple.html").exists()
