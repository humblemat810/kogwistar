import uuid

import pytest

from kogwistar.engine_core.models import (
    Edge,
    GraphExtractionWithIDs,
    LLMEdge,
    LLMGraphExtraction,
    LLMNode,
    Node,
)
from tests._kg_factories import kg_document, kg_grounding


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_preserves_explicit_ids(engine):
    doc = kg_document(
        doc_id="doc::graph_id_contract_direct_ids",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    alpha_id = str(uuid.uuid4())
    beta_id = str(uuid.uuid4())
    edge_id = str(uuid.uuid4())

    alpha = Node(
        id=alpha_id,
        label="Alpha",
        type="entity",
        summary="explicit node id",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
        level_from_root=0,
    )
    beta = Node(
        id=beta_id,
        label="Beta",
        type="entity",
        summary="explicit node id",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
        level_from_root=0,
    )
    edge = Edge(
        id=edge_id,
        label="relates",
        type="relationship",
        summary="explicit edge id",
        relation="relates_to",
        source_ids=["Alpha"],
        target_ids=["Beta"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )

    out = engine.persist.persist_document_graph_extraction(
        doc_id=doc.id,
        parsed=GraphExtractionWithIDs(nodes=[alpha, beta], edges=[edge]),
    )

    assert out["nodes_added"] == 2
    assert out["edges_added"] == 1
    assert out["node_ids"] == [alpha_id, beta_id]
    assert out["edge_ids"] == [edge_id]

    got_nodes = engine.backend.node_get(ids=[alpha_id, beta_id], include=["documents"])
    assert got_nodes["ids"] == [alpha_id, beta_id]
    got_edges = engine.backend.edge_get(ids=[edge_id], include=["documents"])
    assert got_edges["ids"] == [edge_id]


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_uses_node_local_id_over_id(engine):
    doc = kg_document(
        doc_id="doc::graph_id_contract_node_precedence",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    node_id = str(uuid.uuid4())
    temp_local_id = "nn:ignored"
    node = LLMNode(
        id=node_id,
        local_id=temp_local_id,
        label="Node",
        type="entity",
        summary="node local_id should win over id",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        doc_id=doc.id,
    )

    parsed = LLMGraphExtraction(nodes=[node], edges=[])
    engine.persist.resolve_llm_ids(doc.id, parsed)

    assert node.id != node_id
    assert node.id != temp_local_id


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_uses_local_ids_and_resolves_same_batch_endpoints(
    engine,
):
    doc = kg_document(
        doc_id="doc::graph_id_contract_local_ids",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    source = LLMNode(
        local_id="nn:source",
        label="Source",
        type="entity",
        summary="local id source node",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        doc_id=doc.id,
    )
    target = LLMNode(
        local_id="nn:target",
        label="Target",
        type="entity",
        summary="local id target node",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        doc_id=doc.id,
    )
    base_edge = LLMEdge(
        id="edge:base_should_be_ignored",
        local_id="ne:base",
        label="base",
        type="relationship",
        summary="base edge",
        relation="links_to",
        source_ids=["nn:source"],
        target_ids=["nn:target"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        doc_id=doc.id,
    )
    meta_edge = LLMEdge(
        id="edge:meta_should_be_ignored",
        local_id="ne:meta",
        label="meta",
        type="relationship",
        summary="edge that references same-batch edges",
        relation="depends_on",
        source_ids=["nn:target"],
        target_ids=["nn:source"],
        source_edge_ids=["ne:base"],
        target_edge_ids=["ne:base"],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        doc_id=doc.id,
    )

    parsed = LLMGraphExtraction(nodes=[source, target], edges=[base_edge, meta_edge])
    engine.persist.resolve_llm_ids(doc.id, parsed)

    assert source.id != "nn:source"
    assert target.id != "nn:target"
    assert base_edge.id != "ne:base"
    assert base_edge.id != "edge:base_should_be_ignored"
    assert meta_edge.id != "ne:meta"
    assert meta_edge.id != "edge:meta_should_be_ignored"
    assert base_edge.source_ids == [source.id]
    assert base_edge.target_ids == [target.id]
    assert meta_edge.source_ids == [target.id]
    assert meta_edge.target_ids == [source.id]
    assert meta_edge.source_edge_ids == [base_edge.id]
    assert meta_edge.target_edge_ids == [base_edge.id]


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_resolves_existing_alias_and_uuid_endpoints(
    engine,
):
    doc = kg_document(
        doc_id="doc::graph_id_contract_aliases",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    existing_source = Node(
        id=str(uuid.uuid4()),
        label="Existing Source",
        type="entity",
        summary="already persisted source node",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    existing_target = Node(
        id=str(uuid.uuid4()),
        label="Existing Target",
        type="entity",
        summary="already persisted target node",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    engine.write.add_node(existing_source, doc_id=doc.id)
    engine.write.add_node(existing_target, doc_id=doc.id)

    existing_edge = Edge(
        id=str(uuid.uuid4()),
        label="Existing Edge",
        type="relationship",
        summary="already persisted edge",
        relation="connects_to",
        source_ids=[existing_source.id],
        target_ids=[existing_target.id],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )
    engine.write.add_edge(existing_edge, doc_id=doc.id)

    alias_book = engine._alias_book(doc.id)
    source_alias = alias_book.alias_for_node(existing_source.id)
    edge_alias = alias_book.alias_for_edge(existing_edge.id)

    new_edge = LLMEdge(
        local_id="ne:new_alias_edge",
        label="Alias Edge",
        type="relationship",
        summary="mixes alias and UUID endpoints",
        relation="connects_to",
        source_ids=[source_alias],
        target_ids=[existing_target.id],
        source_edge_ids=[edge_alias],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        doc_id=doc.id,
    )

    parsed = LLMGraphExtraction(nodes=[], edges=[new_edge])
    engine.persist.resolve_llm_ids(doc.id, parsed)

    assert new_edge.source_ids == [existing_source.id]
    assert new_edge.target_ids == [existing_target.id]
    assert new_edge.source_edge_ids == [existing_edge.id]


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_resolves_node_label_fallback(engine):
    doc = kg_document(
        doc_id="doc::graph_id_contract_label_fallback",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    label_node = Node(
        id=str(uuid.uuid4()),
        label="Label Target",
        type="entity",
        summary="node resolved by label fallback",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
        level_from_root=0,
    )
    explicit_node = Node(
        id=str(uuid.uuid4()),
        label="Explicit",
        type="entity",
        summary="node resolved by uuid endpoint",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
        level_from_root=0,
    )
    edge = Edge(
        id=str(uuid.uuid4()),
        label="Label Edge",
        type="relationship",
        summary="edge uses label fallback for a node endpoint",
        relation="connects_to",
        source_ids=["label target"],
        target_ids=[explicit_node.id],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )

    out = engine.persist.persist_document_graph_extraction(
        doc_id=doc.id,
        parsed=GraphExtractionWithIDs(nodes=[label_node, explicit_node], edges=[edge]),
    )

    assert out["nodes_added"] == 2
    assert out["edges_added"] == 1
    assert edge.source_ids == [label_node.id]
    assert edge.target_ids == [explicit_node.id]


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_rejects_missing_uuid_endpoints(engine):
    doc = kg_document(
        doc_id="doc::graph_id_contract_bad",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    new_node = Node(
        id="nn:new_node",
        label="New Node",
        type="entity",
        summary="new node in the same batch",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
        level_from_root=0,
    )
    missing_uuid = str(uuid.uuid4())
    link = Edge(
        id="ne:new_link",
        label="links",
        type="relationship",
        summary="new node links to a missing node",
        relation="links_to",
        source_ids=["nn:new_node"],
        target_ids=[missing_uuid],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )

    with pytest.raises(ValueError, match="Dangling references"):
        engine.persist.persist_document_graph_extraction(
            doc_id=doc.id,
            parsed=GraphExtractionWithIDs(nodes=[new_node], edges=[link]),
        )


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_persist_document_graph_extraction_rejects_missing_edge_endpoints(engine):
    doc = kg_document(
        doc_id="doc::graph_id_contract_missing_edge",
        content="x",
        source="test_graph_extraction_id_contract",
        doc_type="text",
    )
    engine.write.add_document(doc)

    existing = Node(
        id=str(uuid.uuid4()),
        label="Existing",
        type="entity",
        summary="existing node",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    engine.write.add_node(existing, doc_id=doc.id)

    edge = Edge(
        id=str(uuid.uuid4()),
        label="Missing Edge Ref",
        type="relationship",
        summary="references a missing edge endpoint",
        relation="depends_on",
        source_ids=[existing.id],
        target_ids=[existing.id],
        source_edge_ids=[str(uuid.uuid4())],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )

    with pytest.raises(ValueError, match="Dangling references"):
        engine.persist.persist_document_graph_extraction(
            doc_id=doc.id,
            parsed=GraphExtractionWithIDs(nodes=[], edges=[edge]),
        )
