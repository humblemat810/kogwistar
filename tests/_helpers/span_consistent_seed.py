from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kogwistar.engine_core.models import Document, Edge, GraphExtractionWithIDs, Node

from tests.conftest import mk_grounding, mk_span


@dataclass
class SpanConsistentSeedGraph:
    document: Document
    nodes: list[Node]
    edges: list[Edge]

    def as_graph_extraction(self) -> GraphExtractionWithIDs:
        return GraphExtractionWithIDs(
            nodes=[node.model_copy(deep=True) for node in self.nodes],
            edges=[edge.model_copy(deep=True) for edge in self.edges],
        )

    def as_document_upsert_payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.document.id,
            "nodes": [
                node.model_dump(field_mode="backend", dump_format="json")
                for node in self.nodes
            ],
            "edges": [
                edge.model_dump(field_mode="backend", dump_format="json")
                for edge in self.edges
            ],
        }


def build_span_consistent_debug_rag_seed(
    *,
    doc_id: str = "manual-debug-rag-doc",
    insertion_method: str = "manual_debug_rag_seed",
) -> SpanConsistentSeedGraph:
    alpha_excerpt = "Manual Debug RAG Alpha anchors deterministic retrieval."
    beta_excerpt = "Manual Debug RAG Beta demonstrates validator-safe seeding."
    edge_excerpt = "Manual Debug RAG Alpha supports Manual Debug RAG Beta."
    doc_text = " ".join(
        [
            alpha_excerpt,
            beta_excerpt,
            edge_excerpt,
            "This document exists only to provide span-consistent provenance for debug rag tests.",
        ]
    )

    document = Document(
        id=doc_id,
        content=doc_text,
        type="text",
        metadata={"insertion_method": insertion_method},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )

    def _grounding(excerpt: str, anchor: str):
        start_char = doc_text.index(excerpt)
        end_char = start_char + len(excerpt)
        return mk_grounding(
            mk_span(
                doc_id=doc_id,
                full_text=doc_text,
                start_char=start_char,
                end_char=end_char,
                insertion_method=insertion_method,
                collection_page_url=f"collection/{doc_id}",
                document_page_url=f"doc/{doc_id}#{anchor}",
            )
        )

    alpha = Node(
        id="manual-debug-rag-alpha",
        label="Manual Debug RAG Alpha",
        type="entity",
        summary="Alpha node summary for the manual debug rag flow.",
        mentions=[_grounding(alpha_excerpt, "manual-debug-rag-alpha")],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=doc_id,
        level_from_root=0,
    )
    beta = Node(
        id="manual-debug-rag-beta",
        label="Manual Debug RAG Beta",
        type="entity",
        summary="Beta node summary for the manual debug rag flow.",
        mentions=[_grounding(beta_excerpt, "manual-debug-rag-beta")],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=doc_id,
        level_from_root=0,
    )
    edge = Edge(
        id="manual-debug-rag-supports",
        label="supports",
        type="relationship",
        summary="Alpha supports Beta for the manual debug rag flow.",
        source_ids=[alpha.label],
        target_ids=[beta.label],
        relation="supports",
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[_grounding(edge_excerpt, "manual-debug-rag-supports")],
        metadata={"level_from_root": 0, "entity_type": "kg_edge"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_edge"},
        embedding=None,
        doc_id=doc_id,
    )

    return SpanConsistentSeedGraph(
        document=document,
        nodes=[alpha, beta],
        edges=[edge],
    )
