from __future__ import annotations

import pytest
from pydantic import ValidationError

from kogwistar.engine_core.embedding_profile import EmbeddingProfile
from kogwistar.engine_core.models import (
    FlattenedGrounding,
    Grounding,
    LLMGraphExtraction,
    MultimodalSpan,
)
from kogwistar.engine_core.multimodal import (
    EmbeddingReference,
    SpatialRegionLocator,
    TemporalIntervalLocator,
    TextRangeLocator,
    VideoRegionTrackLocator,
    PinnedLogicalRef,
)
from kogwistar.logical_refs import LogicalRef


CONTENT_SHA = "a" * 64
PROFILE_SHA = "b" * 64


def _image_span() -> MultimodalSpan:
    return MultimodalSpan(
        source_namespace="project-a",
        resource_id="image:lecture-1",
        resource_revision_id="rev:1",
        content_sha256=CONTENT_SHA,
        modality="image",
        locator=SpatialRegionLocator(
            x=0.05,
            y=0.10,
            width=0.20,
            height=0.25,
        ),
    )


def _source_map_ref() -> PinnedLogicalRef:
    return PinnedLogicalRef(
        logical_ref=LogicalRef("project-a", "artifact", "source-map:image:lecture-1"),
        role="source_map",
        revision_id="rev:1",
    )


def test_grounding_requires_at_least_one_evidence_span() -> None:
    with pytest.raises(ValidationError, match="span"):
        Grounding(spans=[], multimodal_spans=[])

    grounding = Grounding(multimodal_spans=[_image_span()])
    assert list(grounding.iter_evidence()) == grounding.multimodal_spans


def test_multimodal_locators_validate_temporal_and_spatial_boundaries() -> None:
    with pytest.raises(ValidationError):
        TextRangeLocator(start_char=4, end_char=4)
    with pytest.raises(ValidationError):
        TemporalIntervalLocator(start_ms=20, end_ms=20)
    with pytest.raises(ValidationError):
        SpatialRegionLocator(x=0.9, y=0.0, width=0.2, height=0.1)

    track = VideoRegionTrackLocator(
        start_ms=0,
        end_ms=1000,
        track_manifest_ref="lake://tracks/lecture-1.json",
        track_manifest_sha256=CONTENT_SHA,
    )
    span = MultimodalSpan(
        source_namespace="project-a",
        resource_id="video:lecture-1",
        resource_revision_id="rev:1",
        content_sha256=CONTENT_SHA,
        modality="video",
        locator=track,
    )
    assert span.locator_digest
    assert span.evidence_key == span.evidence_key


def test_embedding_reference_requires_pinned_source_map_and_preserves_higher_order_targets() -> None:
    semantic_edge = PinnedLogicalRef(
        logical_ref=LogicalRef("project-a", "edge", "edge:lecture-about-rabbits"),
        role="semantic",
        mode="live",
    )
    reference = EmbeddingReference(
        source_namespace="project-a",
        profile_fingerprint=PROFILE_SHA,
        embedding_set_id="view:lecture-1",
        span=_image_span(),
        targets=(_source_map_ref(), semantic_edge),
    )
    assert reference.reference_id == reference.reference_id

    with pytest.raises(ValidationError, match="source_map"):
        EmbeddingReference(
            source_namespace="project-a",
            profile_fingerprint=PROFILE_SHA,
            embedding_set_id="view:lecture-1",
            span=_image_span(),
            targets=(semantic_edge,),
        )


def test_legacy_profile_fingerprint_is_unchanged_by_optional_multimodal_fields() -> None:
    legacy = EmbeddingProfile(provider="fake", model="model", dimension=1024)
    assert "embedding_kind" not in legacy.as_dict()
    assert legacy.fingerprint == EmbeddingProfile.from_mapping(legacy.as_dict()).fingerprint

    late = EmbeddingProfile(
        provider="fake",
        model="colqwen",
        dimension=128,
        embedding_kind="late_interaction",
        model_revision="r1",
        preprocessing_fingerprint="crop:v1",
        max_image_patches=200,
    )
    assert late.fingerprint != legacy.fingerprint


def test_flattened_grounding_accepts_multimodal_only_and_rejects_empty() -> None:
    span = _image_span()
    flattened = FlattenedGrounding(
        multimodal_span_ids=[span.evidence_key],
    )
    assert flattened.multimodal_span_ids == [span.evidence_key]
    with pytest.raises(ValidationError, match="span id"):
        FlattenedGrounding(span_ids=[], multimodal_span_ids=[])


def test_graph_flattening_keeps_multimodal_evidence_outside_text_span_table() -> None:
    span = _image_span()
    graph = LLMGraphExtraction.model_validate(
        {
            "nodes": [
                {
                    "local_id": "nn:rabbit",
                    "label": "rabbit",
                    "type": "entity",
                    "summary": "A rabbit in the image.",
                    "mentions": [{"multimodal_spans": [span.model_dump(mode="json")]}],
                }
            ],
            "edges": [],
        }
    )
    flattened = graph.to_flattened(insertion_method="llm")
    assert flattened.spans == []
    assert len(flattened.multimodal_spans) == 1
    restored = flattened.to_canonical(insertion_method="llm")
    assert restored.nodes[0].mentions[0].multimodal_spans[0] == span
