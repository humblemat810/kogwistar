# tests/test_graph_query_real_engine.py
from __future__ import annotations
import pytest

import kogwistar.engine_core.engine as engmod
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.graph_query import GraphQuery
from kogwistar.engine_core.models import (
    Document,
    Node,
    Edge,
    Span,
    Grounding,
)
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [
    pytest.mark.core,
    pytest.mark.ci_full,
]


# ---- Test helpers ----
class _DummyLLM:
    """Prevents AzureChatOpenAI from initializing/networking during tests."""

    def __init__(self, *_, **__):
        pass

    def with_structured_output(self, *_a, **_k):
        class _Chain:
            def invoke(self, *_x, **_y):
                return {
                    "raw": "(disabled)",
                    "parsed": None,
                    "parsing_error": "LLM disabled in tests",
                }

        return _Chain()


def _ref(doc_id: str, excerpt: str = "") -> Span:
    return Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        doc_id=doc_id,
        page_number=1,
        start_char=0,
        insertion_method="test-manual",
        end_char=max(1, len(excerpt) or 1),
        excerpt=excerpt or "x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
    )


def make_engine(tmp_path, *, backend_kind: str) -> GraphKnowledgeEngine:
    # Monkeypatch the engine to avoid external LLM initialization/networking in tests
    engmod.AzureChatOpenAI = _DummyLLM
    engmod.ChatGoogleGenerativeAI = _DummyLLM
    ef = build_test_embedding_function("constant", dim=384)
    if backend_kind == "fake":
        e = GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
    else:
        e = GraphKnowledgeEngine(
            persist_directory=str(tmp_path), embedding_function=ef
        )
    # Keep indexing deterministic in this primitive test: write join indexes synchronously.
    e._phase1_enable_index_jobs = False
    return e


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci_full),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
)
def test_graph_query_structural_end_to_end(tmp_path, backend_kind):
    e = make_engine(tmp_path, backend_kind=backend_kind)

    # 1) Real document row
    content = "Smoking causes lung cancer."
    doc = Document(
        id="D1",
        content=content,
        type="text",
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        # For plain "text" docs, source_map is not used; keep explicit for schema compatibility.
        source_map=None,
    )
    e.add_document(doc)

    # 2) Real nodes (persisted into the selected backend)
    n_smoke = Node(
        label="Smoking",
        type="entity",
        summary="habit",
        mentions=[Grounding(spans=[_ref(doc.id, "Smoking")])],
        doc_id=doc.id,
        metadata={},
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )
    n_cancer = Node(
        label="Lung cancer",
        type="entity",
        summary="disease",
        mentions=[Grounding(spans=[_ref(doc.id, "lung cancer")])],
        doc_id=doc.id,
        metadata={},
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )
    e.add_node(n_smoke, doc_id=doc.id)
    e.add_node(n_cancer, doc_id=doc.id)
    assert n_smoke.id is not None and n_cancer.id is not None
    smoke_id = n_smoke.id
    cancer_id = n_cancer.id

    # 3) Real edge (engine will fan out edge_endpoints rows)
    e_causes = Edge(
        label="Smoking→Cancer",
        type="relationship",
        relation="causes",
        source_ids=[smoke_id],
        target_ids=[cancer_id],
        source_edge_ids=[],
        target_edge_ids=[],
        summary="causal claim",
        mentions=[Grounding(spans=[_ref(doc.id, "causes")])],
        doc_id=doc.id,
        metadata={"causal_type": "test"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )
    e.add_edge(e_causes, doc_id=doc.id)
    assert e_causes.id is not None
    causes_id = e_causes.id

    # 4) Graph queries against the *real* collections
    gq = GraphQuery(e)

    # neighbors(node) should include the edge + opposite node
    nbrs = gq.neighbors(smoke_id)
    assert causes_id in nbrs["edges"], f"Expected {causes_id} in edges, got {nbrs}"
    assert cancer_id in nbrs["nodes"], f"Expected {cancer_id} in nodes, got {nbrs}"
    nbrs = gq.neighbors(smoke_id, allow_jump_edge=False)
    assert len(nbrs["nodes"]) == 0, (
        "node should not jump over edge to another jode in no jump mode"
    )
    # k-hop expansion
    layers = gq.k_hop([smoke_id], k=2)
    assert causes_id in layers[0]["edges"]
    assert cancer_id in layers[1]["nodes"]

    # shortest path (unweighted)
    path = gq.shortest_path(smoke_id, cancer_id)
    assert path and path[0] == smoke_id and path[-1] == cancer_id
    assert causes_id in path

    # filtering edges by relation + endpoint labels + doc filter
    results = gq.find_edges(
        relation="causes",
        src_label_contains="smok",
        tgt_label_contains="cancer",
        doc_id=doc.id,
    )
    assert causes_id in results


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci_full),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
)
def test_semantic_seed_then_expand_text(tmp_path, backend_kind):
    e = make_engine(tmp_path, backend_kind=backend_kind)
    doc = Document(
        id="D2",
        content="Smoking causes disease.",
        type="text",
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
    e.add_document(doc)

    n_smoke = Node(
        label="Smoking",
        type="entity",
        summary="habit",
        mentions=[Grounding(spans=[_ref(doc.id, "Smoking")])],
        doc_id=doc.id,
        metadata={},
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )
    n_disease = Node(
        label="Disease",
        type="entity",
        summary="outcome",
        mentions=[Grounding(spans=[_ref(doc.id, "disease")])],
        doc_id=doc.id,
        metadata={},
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )
    e.add_node(n_smoke, doc_id=doc.id)
    e.add_node(n_disease, doc_id=doc.id)
    assert n_smoke.id is not None and n_disease.id is not None
    smoke_id = n_smoke.id
    disease_id = n_disease.id

    rel = Edge(
        label="Smoking→Disease",
        type="relationship",
        relation="causes",
        source_ids=[smoke_id],
        target_ids=[disease_id],
        source_edge_ids=[],
        target_edge_ids=[],
        summary="causal",
        mentions=[Grounding(spans=[_ref(doc.id, "causes")])],
        doc_id=doc.id,
        metadata={"causal_type": "test"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )
    e.add_edge(rel, doc_id=doc.id)

    gq = GraphQuery(e)
    out = gq.semantic_seed_then_expand_text("smok", top_k=5, hops=1)
    assert out["seeds"]  # should find A or related
    assert isinstance(out["layers"], list) and out["layers"]
