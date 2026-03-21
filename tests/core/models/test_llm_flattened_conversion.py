from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci

from kogwistar.engine_core.models import (
    AssocFlattenedLLMGraphExtraction,
    FlattenedLLMGraphExtraction,
    Grounding,
    LLMEdge,
    LLMGraphExtraction,
    LLMNode,
    Span,
)


def make_span(
    *,
    excerpt: str = "a^2 + b^2 = c^2",
    page_number: int = 1,
    start_char: int = 10,
    end_char: int = 25,
    chunk_id: str | None = "chunk-1",
    source_cluster_id: str | None = "cluster-1",
    insertion_method: str = "llm",
) -> Span:
    return Span.model_validate(
        {
            "collection_page_url": "collection/proof",
            "document_page_url": "document/proof",
            "doc_id": "doc:proof",
            "insertion_method": insertion_method,
            "page_number": page_number,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": "Before proof",
            "context_after": "After proof",
            "chunk_id": chunk_id,
            "source_cluster_id": source_cluster_id,
        },
        context={"insertion_method": insertion_method},
    )


def make_mention(*spans: Span) -> Grounding:
    return Grounding.model_validate({"spans": list(spans)})


def make_node(*mentions: Grounding) -> LLMNode:
    return LLMNode.model_validate(
        {
            "local_id": "nn:pythagorean_theorem",
            "label": "Pythagorean theorem",
            "type": "entity",
            "summary": "A theorem relating sides of a right triangle.",
            "mentions": list(mentions),
        }
    )


def make_edge(*mentions: Grounding) -> LLMEdge:
    return LLMEdge.model_validate(
        {
            "local_id": "ne:proved_by",
            "label": "proved by",
            "type": "relationship",
            "summary": "The theorem is supported by a proof step.",
            "mentions": list(mentions),
            "source_ids": ["nn:pythagorean_theorem"],
            "target_ids": ["nn:proof_step_1"],
            "relation": "proved_by",
            "source_edge_ids": None,
            "target_edge_ids": None,
        }
    )


def make_content_for_offsets(
    *, excerpt: str, start_char: int, end_char: int, total_len: int = 256
) -> str:
    assert end_char - start_char == len(excerpt)
    prefix = "x" * start_char
    suffix_len = max(0, total_len - end_char)
    suffix = "y" * suffix_len
    return prefix + excerpt + suffix


def assert_span_equal(a: Span, b: Span) -> None:
    assert a.collection_page_url == b.collection_page_url
    assert a.document_page_url == b.document_page_url
    assert a.doc_id == b.doc_id
    assert a.page_number == b.page_number
    assert a.start_char == b.start_char
    assert a.end_char == b.end_char
    assert a.excerpt == b.excerpt
    assert a.context_before == b.context_before
    assert a.context_after == b.context_after
    assert a.chunk_id == b.chunk_id
    assert a.source_cluster_id == b.source_cluster_id


def assert_mention_equal(a: Grounding, b: Grounding) -> None:
    assert len(a.spans) == len(b.spans)
    for sa, sb in zip(a.spans, b.spans):
        assert_span_equal(sa, sb)


def assert_node_equal(a: LLMNode, b: LLMNode) -> None:
    assert a.id == b.id
    assert a.local_id == b.local_id
    assert a.label == b.label
    assert a.type == b.type
    assert a.summary == b.summary
    assert a.domain_id == b.domain_id
    assert a.canonical_entity_id == b.canonical_entity_id
    assert a.properties == b.properties
    assert len(a.mentions) == len(b.mentions)
    for ma, mb in zip(a.mentions, b.mentions):
        assert_mention_equal(ma, mb)


def assert_edge_equal(a: LLMEdge, b: LLMEdge) -> None:
    assert a.id == b.id
    assert a.local_id == b.local_id
    assert a.label == b.label
    assert a.type == b.type
    assert a.summary == b.summary
    assert a.domain_id == b.domain_id
    assert a.canonical_entity_id == b.canonical_entity_id
    assert a.properties == b.properties
    assert a.source_ids == b.source_ids
    assert a.target_ids == b.target_ids
    assert a.relation == b.relation
    assert a.source_edge_ids == b.source_edge_ids
    assert a.target_edge_ids == b.target_edge_ids
    assert len(a.mentions) == len(b.mentions)
    for ma, mb in zip(a.mentions, b.mentions):
        assert_mention_equal(ma, mb)


def test_llm_graph_extraction_roundtrip_canonical_to_flattened_to_canonical():
    sp1 = make_span(excerpt="Proof step one", page_number=1, start_char=0, end_char=14)
    sp2 = make_span(excerpt="Proof step two", page_number=2, start_char=5, end_char=19)

    m1 = make_mention(sp1)
    m2 = make_mention(sp2)

    node = make_node(m1, m2)
    edge = make_edge(m1)

    graph = LLMGraphExtraction.model_validate(
        {
            "nodes": [node],
            "edges": [edge],
        }
    )

    flat = graph.to_flattened(insertion_method="llm")
    roundtrip: LLMGraphExtraction = flat.to_canonical(insertion_method="llm")

    assert len(roundtrip.nodes) == 1
    assert len(roundtrip.edges) == 1

    assert_node_equal(graph.nodes[0], roundtrip.nodes[0])
    assert_edge_equal(graph.edges[0], roundtrip.edges[0])


def test_llm_graph_extraction_roundtrip_flattened_to_canonical_to_flattened():
    flat = FlattenedLLMGraphExtraction.model_validate(
        {
            "spans": [
                {
                    "id": "sp:1",
                    "collection_page_url": "collection/proof",
                    "document_page_url": "document/proof",
                    "doc_id": "doc:proof",
                    "insertion_method": "llm",
                    "page_number": 1,
                    "start_char": 0,
                    "end_char": 14,
                    "excerpt": "Proof step one",
                    "context_before": "",
                    "context_after": " continues",
                    "chunk_id": "chunk-1",
                    "source_cluster_id": "cluster-1",
                },
                {
                    "id": "sp:2",
                    "collection_page_url": "collection/proof",
                    "document_page_url": "document/proof",
                    "doc_id": "doc:proof",
                    "insertion_method": "llm",
                    "page_number": 2,
                    "start_char": 5,
                    "end_char": 19,
                    "excerpt": "Proof step two",
                    "context_before": "previous",
                    "context_after": "",
                    "chunk_id": "chunk-2",
                    "source_cluster_id": "cluster-1",
                },
            ],
            "nodes": [
                {
                    "local_id": "nn:pythagorean_theorem",
                    "label": "Pythagorean theorem",
                    "type": "entity",
                    "summary": "A theorem relating sides of a right triangle.",
                    "mentions": [
                        {"span_ids": ["sp:1"]},
                        {"span_ids": ["sp:2"]},
                    ],
                }
            ],
            "edges": [
                {
                    "local_id": "ne:proved_by",
                    "label": "proved by",
                    "type": "relationship",
                    "summary": "The theorem is supported by a proof step.",
                    "source_ids": ["nn:pythagorean_theorem"],
                    "target_ids": ["nn:proof_step_1"],
                    "relation": "proved_by",
                    "source_edge_ids": None,
                    "target_edge_ids": None,
                    "mentions": [{"span_ids": ["sp:1"]}],
                }
            ],
        },
        context={"insertion_method": "llm"},
    )

    canonical = flat.to_canonical(insertion_method="llm")
    flat2 = canonical.to_flattened(insertion_method="llm")

    assert len(canonical.nodes) == 1
    assert len(canonical.edges) == 1
    assert len(canonical.nodes[0].mentions) == 2
    assert len(canonical.edges[0].mentions) == 1

    assert len(flat2.spans) == 2
    assert len(flat2.nodes[0].mentions) == 2
    assert len(flat2.edges[0].mentions) == 1


def test_llm_node_roundtrip_via_flattened():
    sp1 = make_span(
        excerpt="Node mention one", page_number=1, start_char=1, end_char=17
    )
    sp2 = make_span(
        excerpt="Node mention two", page_number=2, start_char=2, end_char=18
    )

    node = make_node(
        make_mention(sp1),
        make_mention(sp2),
    )

    graph = LLMGraphExtraction.model_validate({"nodes": [node], "edges": []})
    flat_graph = graph.to_flattened(insertion_method="llm")

    assert len(flat_graph.nodes) == 1
    flat_node = flat_graph.nodes[0]

    restored_node = flat_node.to_canonical(
        span_by_id={
            sp.id: sp.to_canonical(insertion_method="llm") for sp in flat_graph.spans
        }
    )

    assert_node_equal(node, restored_node)


def test_llm_edge_roundtrip_via_flattened():
    sp1 = make_span(excerpt="Edge mention", page_number=3, start_char=4, end_char=16)

    edge = make_edge(make_mention(sp1))

    graph = LLMGraphExtraction.model_validate({"nodes": [], "edges": [edge]})
    flat_graph = graph.to_flattened(insertion_method="llm")

    assert len(flat_graph.edges) == 1
    flat_edge = flat_graph.edges[0]

    restored_edge = flat_edge.to_canonical(
        span_by_id={
            sp.id: sp.to_canonical(insertion_method="llm") for sp in flat_graph.spans
        }
    )

    assert_edge_equal(edge, restored_edge)


def test_from_normal_llm_accepts_mentions_shape():
    payload = {
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
                "mentions": [
                    {
                        "spans": [
                            {
                                "collection_page_url": "collection/proof",
                                "document_page_url": "document/proof",
                                "doc_id": "doc:proof",
                                "page_number": 1,
                                "start_char": 0,
                                "end_char": 14,
                                "excerpt": "Proof step one",
                                "context_before": "",
                                "context_after": "",
                                "chunk_id": "chunk-1",
                                "source_cluster_id": "cluster-1",
                            }
                        ]
                    }
                ],
            }
        ],
        "edges": [],
    }

    graph = LLMGraphExtraction.from_normal_llm(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].mentions) == 1
    assert len(graph.nodes[0].mentions[0].spans) == 1


def test_from_flattened_llm_accepts_flattened_shape():
    payload = {
        "spans": [
            {
                "id": "sp:1",
                "collection_page_url": "collection/proof",
                "document_page_url": "document/proof",
                "doc_id": "doc:proof",
                "insertion_method": "llm",
                "page_number": 1,
                "start_char": 0,
                "end_char": 14,
                "excerpt": "Proof step one",
                "context_before": "",
                "context_after": "",
                "chunk_id": "chunk-1",
                "source_cluster_id": "cluster-1",
            }
        ],
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
                "mentions": [{"span_ids": ["sp:1"]}],
            }
        ],
        "edges": [],
    }

    graph = LLMGraphExtraction.from_flattened_llm(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].mentions) == 1
    assert len(graph.nodes[0].mentions[0].spans) == 1


def test_dispatcher_from_llm_slice_detects_flattened():
    payload = {
        "spans": [
            {
                "id": "sp:1",
                "collection_page_url": "collection/proof",
                "document_page_url": "document/proof",
                "doc_id": "doc:proof",
                "insertion_method": "llm",
                "page_number": 1,
                "start_char": 0,
                "end_char": 14,
                "excerpt": "Proof step one",
                "context_before": "",
                "context_after": "",
                "chunk_id": "chunk-1",
                "source_cluster_id": "cluster-1",
            }
        ],
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
                "mentions": [{"span_ids": ["sp:1"]}],
            }
        ],
        "edges": [],
    }

    graph = LLMGraphExtraction.FromLLMSlice(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Pythagorean theorem"


def test_flattened_validation_rejects_unknown_span_id():
    with pytest.raises(
        ValueError, match="unknown span id|Unknown span id|references unknown span id"
    ):
        FlattenedLLMGraphExtraction.model_validate(
            {
                "spans": [],
                "nodes": [
                    {
                        "local_id": "nn:pythagorean_theorem",
                        "label": "Pythagorean theorem",
                        "type": "entity",
                        "summary": "A theorem relating sides of a right triangle.",
                        "mentions": [{"span_ids": ["sp:missing"]}],
                    }
                ],
                "edges": [],
            },
            context={"insertion_method": "llm"},
        )


def test_flattened_validation_rejects_orphan_span():
    with pytest.raises(ValueError, match="Unreferenced spans|orphan"):
        FlattenedLLMGraphExtraction.model_validate(
            {
                "spans": [
                    {
                        "id": "sp:1",
                        "collection_page_url": "collection/proof",
                        "document_page_url": "document/proof",
                        "doc_id": "doc:proof",
                        "insertion_method": "llm",
                        "page_number": 1,
                        "start_char": 0,
                        "end_char": 14,
                        "excerpt": "Proof step one",
                        "context_before": "",
                        "context_after": "",
                        "chunk_id": "chunk-1",
                        "source_cluster_id": "cluster-1",
                    }
                ],
                "nodes": [],
                "edges": [],
            },
            context={"insertion_method": "llm"},
        )


def test_flattening_dedupes_same_span_used_in_multiple_mentions():
    shared = make_span(
        excerpt="Shared supporting text", page_number=7, start_char=10, end_char=32
    )

    node = make_node(
        make_mention(shared),
        make_mention(shared),
    )
    edge = make_edge(make_mention(shared))

    graph = LLMGraphExtraction.model_validate({"nodes": [node], "edges": [edge]})
    flat = graph.to_flattened(insertion_method="llm")

    assert len(flat.spans) == 1

    node_span_ids = [sid for m in flat.nodes[0].mentions for sid in m.span_ids]
    edge_span_ids = [sid for m in flat.edges[0].mentions for sid in m.span_ids]

    assert node_span_ids == ["sp:1", "sp:1"]
    assert edge_span_ids == ["sp:1"]


def test_from_normal_llm_accepts_legacy_groundings_alias():
    payload = {
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
                "groundings": [
                    {
                        "spans": [
                            {
                                "collection_page_url": "collection/proof",
                                "document_page_url": "document/proof",
                                "doc_id": "doc:proof",
                                "page_number": 1,
                                "start_char": 0,
                                "end_char": 14,
                                "excerpt": "Proof step one",
                                "context_before": "",
                                "context_after": "",
                                "chunk_id": "chunk-1",
                                "source_cluster_id": "cluster-1",
                            }
                        ]
                    }
                ],
            }
        ],
        "edges": [],
    }

    graph = LLMGraphExtraction.from_normal_llm(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].mentions) == 1


def test_from_flattened_llm_accepts_legacy_groundings_alias():
    payload = {
        "spans": [
            {
                "id": "sp:1",
                "collection_page_url": "collection/proof",
                "document_page_url": "document/proof",
                "doc_id": "doc:proof",
                "insertion_method": "llm",
                "page_number": 1,
                "start_char": 0,
                "end_char": 14,
                "excerpt": "Proof step one",
                "context_before": "",
                "context_after": "",
                "chunk_id": "chunk-1",
                "source_cluster_id": "cluster-1",
            }
        ],
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
                "groundings": [{"span_ids": ["sp:1"]}],
            }
        ],
        "edges": [],
    }

    graph = LLMGraphExtraction.from_flattened_llm(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].mentions) == 1


def _assoc_payload_one_node_one_grounding() -> dict:
    return {
        "spans": [
            {
                "id": "sp:1",
                "collection_page_url": "collection/proof",
                "document_page_url": "document/proof",
                "doc_id": "doc:proof",
                "insertion_method": "llm",
                "page_number": 1,
                "start_char": 0,
                "end_char": 14,
                "excerpt": "Proof step one",
                "context_before": "",
                "context_after": "",
                "chunk_id": "chunk-1",
                "source_cluster_id": "cluster-1",
            }
        ],
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
            }
        ],
        "edges": [],
        "groundings": [{"id": "gr:1"}],
        "node_groundings": [{"node_index": 0, "grounding_id": "gr:1"}],
        "edge_groundings": [],
        "grounding_spans": [{"grounding_id": "gr:1", "span_id": "sp:1"}],
    }


def test_assoc_flattened_roundtrip_canonical_to_flattened_to_canonical():
    sp1 = make_span(excerpt="Proof step one", page_number=1, start_char=0, end_char=14)
    sp2 = make_span(excerpt="Proof step two", page_number=2, start_char=5, end_char=19)

    node = make_node(make_mention(sp1), make_mention(sp2))
    edge = make_edge(make_mention(sp1))
    graph = LLMGraphExtraction.model_validate({"nodes": [node], "edges": [edge]})

    assoc = AssocFlattenedLLMGraphExtraction.from_canonical(
        graph, insertion_method="llm"
    )
    roundtrip = assoc.to_canonical(insertion_method="llm")

    assert len(assoc.spans) == 2
    assert len(assoc.groundings) == 3
    assert len(assoc.node_groundings) == 2
    assert len(assoc.edge_groundings) == 1

    assert_node_equal(graph.nodes[0], roundtrip.nodes[0])
    assert_edge_equal(graph.edges[0], roundtrip.edges[0])


def test_from_flattened_llm_accepts_assoc_flattened_shape():
    payload = _assoc_payload_one_node_one_grounding()

    graph = LLMGraphExtraction.from_flattened_llm(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].mentions) == 1
    assert len(graph.nodes[0].mentions[0].spans) == 1


def test_dispatcher_from_llm_slice_detects_assoc_flattened():
    payload = _assoc_payload_one_node_one_grounding()

    graph = LLMGraphExtraction.FromLLMSlice(payload, insertion_method="llm")
    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Pythagorean theorem"


def test_assoc_flattened_validation_rejects_unknown_grounding_id():
    payload = _assoc_payload_one_node_one_grounding()
    payload["node_groundings"] = [{"node_index": 0, "grounding_id": "gr:missing"}]

    with pytest.raises(ValueError, match="unknown grounding id"):
        AssocFlattenedLLMGraphExtraction.model_validate(
            payload, context={"insertion_method": "llm"}
        )


def test_assoc_flattened_validation_rejects_unknown_span_id():
    payload = _assoc_payload_one_node_one_grounding()
    payload["grounding_spans"] = [{"grounding_id": "gr:1", "span_id": "sp:missing"}]

    with pytest.raises(ValueError, match="unknown span id"):
        AssocFlattenedLLMGraphExtraction.model_validate(
            payload, context={"insertion_method": "llm"}
        )


def test_assoc_flattened_validation_rejects_orphan_span_and_grounding():
    payload = _assoc_payload_one_node_one_grounding()
    payload["groundings"].append({"id": "gr:2"})
    payload["spans"].append(
        {
            "id": "sp:2",
            "collection_page_url": "collection/proof",
            "document_page_url": "document/proof",
            "doc_id": "doc:proof",
            "insertion_method": "llm",
            "page_number": 2,
            "start_char": 10,
            "end_char": 22,
            "excerpt": "Orphan text",
            "context_before": "",
            "context_after": "",
            "chunk_id": "chunk-2",
            "source_cluster_id": "cluster-1",
        }
    )

    with pytest.raises(ValueError, match="Unreferenced groundings|Unreferenced spans"):
        AssocFlattenedLLMGraphExtraction.model_validate(
            payload, context={"insertion_method": "llm"}
        )


def test_assoc_flattened_validation_rejects_invalid_node_index():
    payload = _assoc_payload_one_node_one_grounding()
    payload["node_groundings"] = [{"node_index": 3, "grounding_id": "gr:1"}]

    with pytest.raises(ValueError, match="invalid node_index"):
        AssocFlattenedLLMGraphExtraction.model_validate(
            payload, context={"insertion_method": "llm"}
        )


def _lean_assoc_payload_one_node_one_grounding(
    *,
    excerpt: str = "Proof step one",
    page_number: int = 1,
    start_char: int = 10,
    end_char: int = 24,
) -> dict:
    return {
        "spans": [
            {
                "id": "sp:1",
                "page_number": page_number,
                "start_char": start_char,
                "end_char": end_char,
                "excerpt": excerpt,
            }
        ],
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
            }
        ],
        "edges": [],
        "groundings": [{"id": "gr:1"}],
        "node_groundings": [{"node_index": 0, "grounding_id": "gr:1"}],
        "edge_groundings": [],
        "grounding_spans": [{"grounding_id": "gr:1", "span_id": "sp:1"}],
    }


def test_lean_assoc_roundtrip_canonical_to_flattened_to_canonical():
    excerpt = "Proof step one"
    start_char = 10
    end_char = 24
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=start_char, end_char=end_char, total_len=96
    )

    span = Span.model_validate(
        {
            "collection_page_url": "document_collection/doc:lean",
            "document_page_url": "document/doc:lean",
            "doc_id": "doc:lean",
            "insertion_method": "llm",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": content[max(0, start_char - 20) : start_char],
            "context_after": content[end_char : end_char + 20],
            "chunk_id": None,
            "source_cluster_id": None,
        },
        context={"insertion_method": "llm"},
    )
    mention = make_mention(span)
    node = make_node(mention)
    edge = make_edge(mention)
    graph = LLMGraphExtraction.model_validate({"nodes": [node], "edges": [edge]})

    lean_assoc = graph.to_lean_assoc_flattened(insertion_method="llm")
    roundtrip = LLMGraphExtraction.from_flattened_llm(
        lean_assoc,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(lean_assoc.spans) == 1
    assert len(lean_assoc.groundings) == 2
    assert len(roundtrip.nodes) == 1
    assert len(roundtrip.edges) == 1
    assert roundtrip.nodes[0].domain_id is None
    assert roundtrip.nodes[0].canonical_entity_id is None
    assert roundtrip.nodes[0].properties is None

    span_rt = roundtrip.nodes[0].mentions[0].spans[0]
    assert span_rt.doc_id == "doc:lean"
    assert span_rt.collection_page_url == "document_collection/doc:lean"
    assert span_rt.document_page_url == "document/doc:lean"
    assert span_rt.excerpt == excerpt
    assert content[span_rt.start_char : span_rt.end_char] == span_rt.excerpt


def test_from_flattened_llm_accepts_lean_assoc_shape_with_rehydration():
    excerpt = "Proof step one"
    payload = _lean_assoc_payload_one_node_one_grounding(
        excerpt=excerpt,
        start_char=10,
        end_char=24,
    )
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=10, end_char=24, total_len=80
    )

    graph = LLMGraphExtraction.from_flattened_llm(
        payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].mentions) == 1
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.doc_id == "doc:lean"
    assert span.collection_page_url == "document_collection/doc:lean"
    assert span.document_page_url == "document/doc:lean"


def test_dispatcher_from_llm_slice_detects_lean_assoc():
    payload = _lean_assoc_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    content = make_content_for_offsets(
        excerpt="Proof step one", start_char=10, end_char=24, total_len=80
    )

    graph = LLMGraphExtraction.FromLLMSlice(
        payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Pythagorean theorem"


def test_lean_assoc_edge_defaults_source_and_target_edge_ids_to_empty_lists():
    excerpt = "Proof step one"
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=10, end_char=24, total_len=96
    )
    payload = {
        "spans": [
            {
                "id": "sp:1",
                "page_number": 1,
                "start_char": 10,
                "end_char": 24,
                "excerpt": excerpt,
            },
        ],
        "nodes": [
            {
                "local_id": "nn:source",
                "label": "Source",
                "type": "entity",
                "summary": "Source node",
            },
            {
                "local_id": "nn:target",
                "label": "Target",
                "type": "entity",
                "summary": "Target node",
            },
        ],
        "edges": [
            {
                "local_id": "ne:r",
                "label": "rel",
                "type": "relationship",
                "summary": "hyperedge-compatible relation",
                "source_ids": ["nn:source"],
                "target_ids": ["nn:target"],
                "relation": "relates",
            }
        ],
        "groundings": [{"id": "gr:1"}, {"id": "gr:2"}, {"id": "gr:3"}],
        "node_groundings": [
            {"node_index": 0, "grounding_id": "gr:1"},
            {"node_index": 1, "grounding_id": "gr:2"},
        ],
        "edge_groundings": [{"edge_index": 0, "grounding_id": "gr:3"}],
        "grounding_spans": [
            {"grounding_id": "gr:1", "span_id": "sp:1"},
            {"grounding_id": "gr:2", "span_id": "sp:1"},
            {"grounding_id": "gr:3", "span_id": "sp:1"},
        ],
    }

    graph = LLMGraphExtraction.from_flattened_llm(
        payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )
    assert len(graph.edges) == 1
    assert graph.edges[0].source_edge_ids == []
    assert graph.edges[0].target_edge_ids == []


def test_lean_assoc_rehydrates_context_before_and_after():
    payload = _lean_assoc_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    content = "0123456789Proof step oneABCDEFGHIJ"

    graph = LLMGraphExtraction.from_flattened_llm(
        payload,
        insertion_method="llm",
        doc_id="doc:ctx",
        content=content,
        context_window_chars=4,
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.context_before == "6789"
    assert span.context_after == "ABCD"


def test_lean_assoc_rehydration_fails_when_excerpt_does_not_match_offsets():
    payload = _lean_assoc_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    # Force mismatch at the expected offsets.
    content = "x" * 10 + "WRONG EXCERPT!" + "y" * 50

    with pytest.raises(ValueError, match="excerpt mismatch"):
        LLMGraphExtraction.from_flattened_llm(
            payload,
            insertion_method="llm",
            doc_id="doc:bad",
            content=content,
        )


def test_lean_assoc_requires_content_for_rehydration():
    payload = _lean_assoc_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    with pytest.raises(ValueError, match="content is required"):
        LLMGraphExtraction.from_flattened_llm(
            payload,
            insertion_method="llm",
            doc_id="doc:no-content",
            content=None,
        )


def _lean_non_flat_payload_one_node_one_grounding(
    *,
    excerpt: str = "Proof step one",
    page_number: int = 1,
    start_char: int = 10,
    end_char: int = 24,
) -> dict:
    return {
        "nodes": [
            {
                "local_id": "nn:pythagorean_theorem",
                "label": "Pythagorean theorem",
                "type": "entity",
                "summary": "A theorem relating sides of a right triangle.",
                "mentions": [
                    {
                        "spans": [
                            {
                                "page_number": page_number,
                                "start_char": start_char,
                                "end_char": end_char,
                                "excerpt": excerpt,
                            }
                        ]
                    }
                ],
            }
        ],
        "edges": [],
    }


def test_lean_non_flat_roundtrip_canonical_to_lean_to_canonical():
    excerpt = "Proof step one"
    start_char = 10
    end_char = 24
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=start_char, end_char=end_char, total_len=96
    )

    span = Span.model_validate(
        {
            "collection_page_url": "document_collection/doc:lean",
            "document_page_url": "document/doc:lean",
            "doc_id": "doc:lean",
            "insertion_method": "llm",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": content[max(0, start_char - 20) : start_char],
            "context_after": content[end_char : end_char + 20],
            "chunk_id": None,
            "source_cluster_id": None,
        },
        context={"insertion_method": "llm"},
    )
    mention = make_mention(span)
    node = make_node(mention)
    edge = make_edge(mention)
    graph = LLMGraphExtraction.model_validate({"nodes": [node], "edges": [edge]})

    lean = graph.to_lean()
    roundtrip = LLMGraphExtraction.from_llm_in_payload(
        lean,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(roundtrip.nodes) == 1
    assert len(roundtrip.edges) == 1
    assert roundtrip.nodes[0].domain_id is None
    assert roundtrip.nodes[0].canonical_entity_id is None
    assert roundtrip.nodes[0].properties is None
    span_rt = roundtrip.nodes[0].mentions[0].spans[0]
    assert span_rt.doc_id == "doc:lean"
    assert span_rt.collection_page_url == "document_collection/doc:lean"
    assert span_rt.document_page_url == "document/doc:lean"
    assert content[span_rt.start_char : span_rt.end_char] == span_rt.excerpt


def test_from_normal_llm_accepts_lean_non_flat_shape_with_rehydration():
    excerpt = "Proof step one"
    payload = _lean_non_flat_payload_one_node_one_grounding(
        excerpt=excerpt,
        start_char=10,
        end_char=24,
    )
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=10, end_char=24, total_len=80
    )

    graph = LLMGraphExtraction.from_normal_llm(
        payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(graph.nodes) == 1
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.doc_id == "doc:lean"
    assert span.collection_page_url == "document_collection/doc:lean"
    assert span.document_page_url == "document/doc:lean"


def test_dispatcher_from_llm_slice_detects_lean_non_flat():
    excerpt = "Proof step one"
    payload = _lean_non_flat_payload_one_node_one_grounding(
        excerpt=excerpt,
        start_char=10,
        end_char=24,
    )
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=10, end_char=24, total_len=80
    )

    graph = LLMGraphExtraction.FromLLMSlice(
        payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Pythagorean theorem"


def test_lean_non_flat_edge_defaults_source_and_target_edge_ids_to_empty_lists():
    excerpt = "Proof step one"
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=10, end_char=24, total_len=96
    )
    payload = {
        "nodes": [
            {
                "local_id": "nn:source",
                "label": "Source",
                "type": "entity",
                "summary": "Source node",
                "mentions": [
                    {
                        "spans": [
                            {
                                "page_number": 1,
                                "start_char": 10,
                                "end_char": 24,
                                "excerpt": excerpt,
                            }
                        ]
                    }
                ],
            },
            {
                "local_id": "nn:target",
                "label": "Target",
                "type": "entity",
                "summary": "Target node",
                "mentions": [
                    {
                        "spans": [
                            {
                                "page_number": 1,
                                "start_char": 10,
                                "end_char": 24,
                                "excerpt": excerpt,
                            }
                        ]
                    }
                ],
            },
        ],
        "edges": [
            {
                "local_id": "ne:r",
                "label": "rel",
                "type": "relationship",
                "summary": "hyperedge-compatible relation",
                "source_ids": ["nn:source"],
                "target_ids": ["nn:target"],
                "relation": "relates",
                "mentions": [
                    {
                        "spans": [
                            {
                                "page_number": 1,
                                "start_char": 10,
                                "end_char": 24,
                                "excerpt": excerpt,
                            }
                        ]
                    }
                ],
            }
        ],
    }

    graph = LLMGraphExtraction.from_normal_llm(
        payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )
    assert len(graph.edges) == 1
    assert graph.edges[0].source_edge_ids == []
    assert graph.edges[0].target_edge_ids == []


def test_lean_non_flat_rehydrates_context_before_and_after():
    payload = _lean_non_flat_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    content = "0123456789Proof step oneABCDEFGHIJ"

    graph = LLMGraphExtraction.from_normal_llm(
        payload,
        insertion_method="llm",
        doc_id="doc:ctx",
        content=content,
        context_window_chars=4,
    )
    span = graph.nodes[0].mentions[0].spans[0]
    assert span.context_before == "6789"
    assert span.context_after == "ABCD"


def test_lean_non_flat_rehydration_fails_when_excerpt_does_not_match_offsets():
    payload = _lean_non_flat_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    content = "x" * 10 + "WRONG EXCERPT!" + "y" * 50

    with pytest.raises(ValueError, match="excerpt mismatch"):
        LLMGraphExtraction.from_normal_llm(
            payload,
            insertion_method="llm",
            doc_id="doc:bad",
            content=content,
        )


def test_lean_non_flat_requires_content_for_rehydration():
    payload = _lean_non_flat_payload_one_node_one_grounding(
        excerpt="Proof step one",
        start_char=10,
        end_char=24,
    )
    with pytest.raises(ValueError, match="content is required"):
        LLMGraphExtraction.from_normal_llm(
            payload,
            insertion_method="llm",
            doc_id="doc:no-content",
            content=None,
        )


def test_llm_in_slice_for_non_flat_shape_excludes_full_only_fields():
    excerpt = "Proof step one"
    start_char = 10
    end_char = 24
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=start_char, end_char=end_char, total_len=96
    )
    span = Span.model_validate(
        {
            "collection_page_url": "document_collection/doc:lean",
            "document_page_url": "document/doc:lean",
            "doc_id": "doc:lean",
            "insertion_method": "llm",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": content[max(0, start_char - 20) : start_char],
            "context_after": content[end_char : end_char + 20],
            "chunk_id": None,
            "source_cluster_id": None,
        },
        context={"insertion_method": "llm"},
    )
    mention = make_mention(span)
    node = make_node(mention)
    edge = make_edge(mention)
    graph = LLMGraphExtraction.model_validate({"nodes": [node], "edges": [edge]})

    llm_in_payload = graph.model_dump(field_mode="llm_in")
    span_payload = llm_in_payload["nodes"][0]["mentions"][0]["spans"][0]
    edge_payload = llm_in_payload["edges"][0]
    node_payload = llm_in_payload["nodes"][0]

    assert set(span_payload.keys()) == {
        "page_number",
        "start_char",
        "end_char",
        "excerpt",
    }
    assert "domain_id" not in node_payload
    assert "canonical_entity_id" not in node_payload
    assert "properties" not in node_payload
    assert "source_edge_ids" not in edge_payload
    assert "target_edge_ids" not in edge_payload
    assert "domain_id" not in edge_payload
    assert "canonical_entity_id" not in edge_payload
    assert "properties" not in edge_payload


def test_llm_in_slice_for_assoc_flat_shape_excludes_full_only_fields():
    excerpt = "Proof step one"
    start_char = 10
    end_char = 24
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=start_char, end_char=end_char, total_len=96
    )
    span = Span.model_validate(
        {
            "collection_page_url": "document_collection/doc:lean",
            "document_page_url": "document/doc:lean",
            "doc_id": "doc:lean",
            "insertion_method": "llm",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": content[max(0, start_char - 20) : start_char],
            "context_after": content[end_char : end_char + 20],
            "chunk_id": None,
            "source_cluster_id": None,
        },
        context={"insertion_method": "llm"},
    )
    mention = make_mention(span)
    node = make_node(mention)
    edge = make_edge(mention)
    assoc = LLMGraphExtraction.model_validate(
        {"nodes": [node], "edges": [edge]}
    ).to_assoc_flattened(insertion_method="llm")

    llm_in_payload = assoc.model_dump(field_mode="llm_in")
    span_payload = llm_in_payload["spans"][0]
    edge_payload = llm_in_payload["edges"][0]
    node_payload = llm_in_payload["nodes"][0]

    assert set(span_payload.keys()) == {
        "id",
        "page_number",
        "start_char",
        "end_char",
        "excerpt",
    }
    assert "domain_id" not in node_payload
    assert "canonical_entity_id" not in node_payload
    assert "properties" not in node_payload
    assert "source_edge_ids" not in edge_payload
    assert "target_edge_ids" not in edge_payload
    assert "domain_id" not in edge_payload
    assert "canonical_entity_id" not in edge_payload
    assert "properties" not in edge_payload


def test_non_flat_roundtrip_via_llm_in_payload():
    excerpt = "Proof step one"
    start_char = 10
    end_char = 24
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=start_char, end_char=end_char, total_len=96
    )
    span = Span.model_validate(
        {
            "collection_page_url": "document_collection/doc:lean",
            "document_page_url": "document/doc:lean",
            "doc_id": "doc:lean",
            "insertion_method": "llm",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": content[max(0, start_char - 20) : start_char],
            "context_after": content[end_char : end_char + 20],
            "chunk_id": None,
            "source_cluster_id": None,
        },
        context={"insertion_method": "llm"},
    )
    mention = make_mention(span)
    graph = LLMGraphExtraction.model_validate(
        {"nodes": [make_node(mention)], "edges": [make_edge(mention)]}
    )

    llm_in_payload = graph.model_dump(field_mode="llm_in")
    roundtrip = LLMGraphExtraction.from_llm_in_payload(
        llm_in_payload,
        insertion_method="llm",
        doc_id="doc:lean",
        content=content,
    )

    assert len(roundtrip.nodes) == 1
    assert len(roundtrip.edges) == 1
    assert roundtrip.edges[0].source_edge_ids == []
    assert roundtrip.edges[0].target_edge_ids == []
    span_rt = roundtrip.nodes[0].mentions[0].spans[0]
    assert span_rt.doc_id == "doc:lean"
    assert span_rt.collection_page_url == "document_collection/doc:lean"
    assert span_rt.document_page_url == "document/doc:lean"


def test_flattened_roundtrip_via_llm_in_payload():
    excerpt = "Proof step one"
    start_char = 10
    end_char = 24
    content = make_content_for_offsets(
        excerpt=excerpt, start_char=start_char, end_char=end_char, total_len=96
    )
    span = Span.model_validate(
        {
            "collection_page_url": "document_collection/doc:lean",
            "document_page_url": "document/doc:lean",
            "doc_id": "doc:lean",
            "insertion_method": "llm",
            "page_number": 1,
            "start_char": start_char,
            "end_char": end_char,
            "excerpt": excerpt,
            "context_before": content[max(0, start_char - 20) : start_char],
            "context_after": content[end_char : end_char + 20],
            "chunk_id": None,
            "source_cluster_id": None,
        },
        context={"insertion_method": "llm"},
    )
    mention = make_mention(span)
    assoc = LLMGraphExtraction.model_validate(
        {"nodes": [make_node(mention)], "edges": [make_edge(mention)]}
    ).to_assoc_flattened(insertion_method="llm")

    llm_in_payload = assoc.model_dump(field_mode="llm_in")
    roundtrip = AssocFlattenedLLMGraphExtraction.to_canonical_from_llm_in_payload(
        llm_in_payload,
        doc_id="doc:lean",
        content=content,
        insertion_method="llm",
    )

    assert len(roundtrip.nodes) == 1
    assert len(roundtrip.edges) == 1
    assert roundtrip.edges[0].source_edge_ids == []
    assert roundtrip.edges[0].target_edge_ids == []
    span_rt = roundtrip.nodes[0].mentions[0].spans[0]
    assert span_rt.doc_id == "doc:lean"
    assert span_rt.collection_page_url == "document_collection/doc:lean"
    assert span_rt.document_page_url == "document/doc:lean"
