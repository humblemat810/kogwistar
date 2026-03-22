import shutil

import pytest
pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Edge, Node
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend
from tests._kg_factories import kg_document, kg_grounding


@pytest.fixture(scope="function")
def tmp_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db_ep")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="function")
def engine(tmp_chroma_dir, backend_kind):
    kwargs = {
        "persist_directory": tmp_chroma_dir,
        "embedding_function": build_test_embedding_function("constant", dim=384),
    }
    if backend_kind == "fake":
        kwargs["backend_factory"] = build_fake_backend
    eng = GraphKnowledgeEngine(**kwargs)
    assert getattr(eng, "backend", None) is not None
    return eng


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
def test_rollback_single_document(engine):
    doc = kg_document(
        doc_id="doc::test_rollback_single_document",
        content="Alpha relates to Beta in rollback test.",
        source="test_rollback_single_document",
    )
    engine.write.add_document(doc)

    ref1 = kg_grounding(doc.id, start_char=0, end_char=5, excerpt="Alpha")
    ref2 = kg_grounding(doc.id, start_char=17, end_char=21, excerpt="Beta")
    node1 = Node(label="A", type="entity", summary="n1", mentions=[ref1], doc_id=doc.id)
    node2 = Node(label="B", type="entity", summary="n2", mentions=[ref2], doc_id=doc.id)
    engine.write.add_node(node1)
    engine.write.add_node(node2)

    edge = Edge(
        label="A->B",
        type="relationship",
        summary="edge",
        source_ids=[node1.id],
        target_ids=[node2.id],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="related",
        mentions=[ref1],
    )
    engine.write.add_edge(edge, doc_id=doc.id)

    result = engine.rollback_document(doc.id)

    assert result["deleted_nodes"] == 2
    assert result["deleted_edges"] == 1
    assert result["updated_edges"] == 0

    node1_row = engine.backend.node_get(ids=[node1.id], include=["metadatas"])
    node2_row = engine.backend.node_get(ids=[node2.id], include=["metadatas"])
    edge_row = engine.backend.edge_get(ids=[edge.id], include=["metadatas"])

    assert node1_row["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert node2_row["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert edge_row["metadatas"][0]["lifecycle_status"] == "tombstoned"

    assert not (
        engine.backend.node_docs_get(where={"node_id": node1.id}).get("ids") or []
    )
    assert not (
        engine.backend.node_docs_get(where={"node_id": node2.id}).get("ids") or []
    )
    assert not (
        engine.backend.edge_endpoints_get(where={"edge_id": edge.id}).get("ids") or []
    )


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
def test_rollback_multiple_documents(tmp_path, backend_kind):
    kwargs = {
        "persist_directory": str(tmp_path / "chroma"),
        "embedding_function": build_test_embedding_function("constant", dim=384),
    }
    if backend_kind == "fake":
        kwargs["backend_factory"] = build_fake_backend
    engine = GraphKnowledgeEngine(**kwargs)

    docs = [
        kg_document(
            doc_id=f"doc::test_rollback_multiple_documents::{i}",
            content=f"Doc {i}",
            source="test_rollback_multiple_documents",
        )
        for i in range(2)
    ]
    for doc in docs:
        engine.write.add_document(doc)

    ref1 = kg_grounding(docs[0].id, excerpt="Doc 0", end_char=5)
    ref2 = kg_grounding(docs[1].id, excerpt="Doc 1", end_char=5)

    shared_node = Node(
        label="X", type="entity", summary="shared", mentions=[ref1, ref2]
    )
    engine.write.add_node(shared_node)

    n1 = Node(label="Y", type="entity", summary="n1", mentions=[ref1])
    engine.write.add_node(n1)

    n2 = Node(label="Z", type="entity", summary="n2", mentions=[ref2])
    engine.write.add_node(n2)

    e1 = Edge(
        label="shared->n1",
        type="relationship",
        summary="e1",
        source_ids=[shared_node.id],
        target_ids=[n1.id],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="related",
        mentions=[ref1],
    )
    e2 = Edge(
        label="n2->shared",
        type="relationship",
        summary="e2",
        source_ids=[n2.id],
        target_ids=[shared_node.id],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="related",
        mentions=[ref2],
    )
    engine.write.add_edge(e1, doc_id=docs[0].id)
    engine.write.add_edge(e2, doc_id=docs[1].id)

    result = engine.rollback_many_documents([docs[0].id, docs[1].id])

    assert result["deleted_nodes"] >= 3
    assert result["deleted_edges"] >= 2
