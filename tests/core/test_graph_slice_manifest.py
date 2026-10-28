from __future__ import annotations

from collections.abc import Mapping

import pytest
from typing import Annotated

from pydantic import BaseModel
from pydantic_extension.model_slicing.mixin import BackendField, LLMField, ModeSlicingMixin

pytestmark = pytest.mark.ci

from kogwistar.engine_core.graph_slice_manifest import (
    build_slice_manifest,
    project_with_manifest,
)
from kogwistar.engine_core.models import Grounding, Node, Span


class _NestedSliceModel(ModeSlicingMixin, BaseModel):
    visible: Annotated[str, BackendField(), LLMField()]
    secret: Annotated[str, BackendField()]


class _MappingSliceModel(ModeSlicingMixin, BaseModel):
    values: Annotated[Mapping[str, _NestedSliceModel], BackendField(), LLMField()]


class _MixinOnlyModel(ModeSlicingMixin):
    pass


def _node() -> Node:
    return Node(
        id="node-1",
        label="alpha",
        type="entity",
        summary="Alpha",
        doc_id="doc-1",
        embedding=[0.1, 0.2],
        metadata={"acl": "backend-only"},
        mentions=[
            Grounding(
                spans=[
                    Span(
                        collection_page_url="collection",
                        document_page_url="document",
                        doc_id="doc-1",
                        insertion_method="parser",
                        page_number=1,
                        start_char=0,
                        end_char=5,
                        excerpt="Alpha",
                        context_before="before",
                        context_after="after",
                        chunk_id="chunk-1",
                        source_cluster_id="cluster-1",
                    )
                ]
            )
        ],
    )


def test_manifest_matches_python_llm_input_slice_and_drops_backend_fields() -> None:
    node = _node()
    projected = project_with_manifest(type(node), node, "llm_in")
    assert projected == node.model_dump(dump_format="json", field_mode="llm_in")

    span = node.mentions[0].spans[0]
    span_projected = project_with_manifest(type(span), span, "llm_in")
    assert span.excerpt == "Alpha"
    assert span_projected["excerpt"] == "Alpha"
    assert "doc_id" not in span_projected
    assert "insertion_method" not in span_projected
    assert "context_before" not in span_projected
    assert "chunk_id" not in span_projected


def test_backend_manifest_projection_is_lossless_for_known_model() -> None:
    node = _node()
    manifest = build_slice_manifest(type(node))
    projected = project_with_manifest(type(node), node, "backend")
    expected = node.model_dump(dump_format="json", field_mode="backend")
    assert projected == expected
    assert manifest["schema_version"] == 1
    assert manifest["model_name"] == "Node"


def test_manifest_cache_returns_defensive_deterministic_copies() -> None:
    first = build_slice_manifest(Node)
    first["fields"]["label"]["modes"].clear()

    second = build_slice_manifest(Node)
    assert second["fields"]["label"]["modes"]
    assert second == build_slice_manifest(Node)


def test_manifest_rejects_slicing_mixin_without_pydantic_model_fields() -> None:
    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        build_slice_manifest(_MixinOnlyModel)


def test_manifest_preserves_mapping_value_slicing() -> None:
    value = _MappingSliceModel(
        values={"one": _NestedSliceModel(visible="yes", secret="no")}
    )
    projected = project_with_manifest(_MappingSliceModel, value, "llm")

    assert projected == {"values": {"one": {"visible": "yes"}}}
    assert (
        build_slice_manifest(_MappingSliceModel)["fields"]["values"]["child_location"]
        == "values"
    )
