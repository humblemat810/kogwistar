"""Immutable multimodal evidence and embedding-reference contracts.

These models describe evidence and references around the canonical graph.  They
do not contain vector values and do not turn individual late-interaction
vectors into graph entities.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kogwistar.logical_refs import LogicalRef


Modality = Literal[
    "text",
    "image",
    "audio",
    "video",
    "pdf_page",
    "table",
    "chart",
    "webpage",
]
CoordinateSystem = Literal["normalized_0_1", "pixels"]


class TextRangeLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text_range"] = "text_range"
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    page_number: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> "TextRangeLocator":
        if self.end_char <= self.start_char:
            raise ValueError("text range end_char must be greater than start_char")
        return self


class SpatialRegionLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["spatial_region"] = "spatial_region"
    coordinate_system: CoordinateSystem = "normalized_0_1"
    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    width: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)
    page_number: int | None = Field(None, ge=1)
    frame_index: int | None = Field(None, ge=0)
    timestamp_ms: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "SpatialRegionLocator":
        if self.coordinate_system == "normalized_0_1":
            if self.x + self.width > 1.0 or self.y + self.height > 1.0:
                raise ValueError("normalized spatial region must stay within 0..1")
        return self


class TemporalIntervalLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["temporal_interval"] = "temporal_interval"
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TemporalIntervalLocator":
        if self.end_ms <= self.start_ms:
            raise ValueError("temporal interval end_ms must be greater than start_ms")
        return self


class VideoRegionTrackLocator(BaseModel):
    """A bounded locator for an external per-frame region manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["video_region_track"] = "video_region_track"
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., gt=0)
    track_manifest_ref: str = Field(..., min_length=1)
    track_manifest_sha256: str = Field(..., min_length=64, max_length=64)
    manifest_schema_version: int = Field(1, ge=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> "VideoRegionTrackLocator":
        if self.end_ms <= self.start_ms:
            raise ValueError("video track end_ms must be greater than start_ms")
        try:
            int(self.track_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("track_manifest_sha256 must be hexadecimal") from exc
        return self


class VideoTrackRegion(BaseModel):
    """One per-frame region in an external video-track manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["bounding_box", "polygon", "mask_ref"]
    x: float | None = Field(None, ge=0.0)
    y: float | None = Field(None, ge=0.0)
    width: float | None = Field(None, gt=0.0)
    height: float | None = Field(None, gt=0.0)
    points: tuple[tuple[float, float], ...] = ()
    mask_ref: str | None = None
    mask_sha256: str | None = Field(None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_shape(self) -> "VideoTrackRegion":
        if self.kind == "bounding_box":
            if None in {self.x, self.y, self.width, self.height}:
                raise ValueError("bounding_box regions require x, y, width, and height")
        elif self.kind == "polygon":
            if len(self.points) < 3:
                raise ValueError("polygon regions require at least three points")
        elif not self.mask_ref or not self.mask_sha256:
            raise ValueError("mask_ref regions require mask_ref and mask_sha256")
        if self.mask_sha256 is not None:
            try:
                int(self.mask_sha256, 16)
            except ValueError as exc:
                raise ValueError("mask_sha256 must be hexadecimal") from exc
        if self.x is not None and self.width is not None and self.x + self.width > 1.0:
            raise ValueError("video region x bounds must stay within 0..1")
        if self.y is not None and self.height is not None and self.y + self.height > 1.0:
            raise ValueError("video region y bounds must stay within 0..1")
        if any(not 0.0 <= coordinate <= 1.0 for point in self.points for coordinate in point):
            raise ValueError("video polygon coordinates must stay within 0..1")
        return self


class VideoTrackFrame(BaseModel):
    """All regions associated with one frame or timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_index: int = Field(..., ge=0)
    timestamp_ms: int = Field(..., ge=0)
    regions: tuple[VideoTrackRegion, ...] = Field(..., min_length=1)


class VideoTrackManifest(BaseModel):
    """Versioned, immutable external manifest for per-frame video grounding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    frames: tuple[VideoTrackFrame, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_order(self) -> "VideoTrackManifest":
        frame_indices = [frame.frame_index for frame in self.frames]
        timestamps = [frame.timestamp_ms for frame in self.frames]
        if frame_indices != sorted(set(frame_indices)):
            raise ValueError("video track frame indices must be unique and ordered")
        if timestamps != sorted(timestamps):
            raise ValueError("video track timestamps must be ordered")
        return self


class LegacyLocator(BaseModel):
    """Read-only compatibility wrapper for historical locator payloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["legacy"] = "legacy"
    payload: Mapping[str, Any]


Locator = Annotated[
    Union[
        TextRangeLocator,
        SpatialRegionLocator,
        TemporalIntervalLocator,
        VideoRegionTrackLocator,
        LegacyLocator,
    ],
    Field(discriminator="kind"),
]


class MultimodalSpan(BaseModel):
    """Immutable source evidence for text, image, audio, or video content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    source_namespace: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    resource_revision_id: str = Field(..., min_length=1)
    content_sha256: str = Field(..., min_length=64, max_length=64)
    modality: Modality
    locator: Locator

    @model_validator(mode="after")
    def validate_evidence(self) -> "MultimodalSpan":
        try:
            int(self.content_sha256, 16)
        except ValueError as exc:
            raise ValueError("content_sha256 must be hexadecimal") from exc
        allowed_locators = {
            "text": {"text_range", "legacy"},
            "image": {"spatial_region", "legacy"},
            "audio": {"temporal_interval", "legacy"},
            "video": {
            "temporal_interval",
            "spatial_region",
            "video_region_track",
            "legacy",
            },
            "pdf_page": {"text_range", "spatial_region", "legacy"},
            "table": {"text_range", "spatial_region", "legacy"},
            "chart": {"spatial_region", "legacy"},
            "webpage": {"text_range", "spatial_region", "legacy"},
        }
        if self.locator.kind not in allowed_locators[self.modality]:
            raise ValueError(
                f"{self.modality} evidence has incompatible locator {self.locator.kind}"
            )
        return self

    @property
    def locator_digest(self) -> str:
        payload = self.locator.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def evidence_key(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "source_namespace": self.source_namespace,
            "resource_id": self.resource_id,
            "resource_revision_id": self.resource_revision_id,
            "content_sha256": self.content_sha256,
            "modality": self.modality,
            "locator_digest": self.locator_digest,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


ReferenceRole = Literal["source_map", "semantic", "conversation", "artifact"]
ReferenceMode = Literal["pinned", "live"]


class PinnedLogicalRef(BaseModel):
    """A namespace-scoped graph reference with explicit freshness semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    logical_ref: LogicalRef
    role: ReferenceRole
    mode: ReferenceMode = "pinned"
    revision_id: str | None = None
    event_seq: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_pin(self) -> "PinnedLogicalRef":
        if self.role == "source_map" and self.mode != "pinned":
            raise ValueError("source_map references must be pinned")
        if self.mode == "pinned" and self.revision_id is None and self.event_seq is None:
            raise ValueError("pinned references require revision_id or event_seq")
        if self.mode == "live" and (self.revision_id is not None or self.event_seq is not None):
            raise ValueError("live references cannot carry a revision or event sequence")
        return self


class EmbeddingReference(BaseModel):
    """Immutable bridge from a profile-scoped vector result to graph evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    source_namespace: str = Field(..., min_length=1)
    profile_fingerprint: str = Field(..., min_length=64, max_length=64)
    embedding_set_id: str = Field(..., min_length=1)
    span: MultimodalSpan
    targets: tuple[PinnedLogicalRef, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_targets(self) -> "EmbeddingReference":
        source_maps = [target for target in self.targets if target.role == "source_map"]
        if not source_maps:
            raise ValueError("embedding references require a source_map target")
        if self.span.source_namespace != self.source_namespace:
            raise ValueError("embedding span must stay in the source namespace")
        if any(target.logical_ref.target_namespace != self.source_namespace for target in self.targets):
            raise ValueError("embedding targets must stay in the source namespace")
        try:
            int(self.profile_fingerprint, 16)
        except ValueError as exc:
            raise ValueError("profile_fingerprint must be hexadecimal") from exc
        return self

    @property
    def reference_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "source_namespace": self.source_namespace,
            "profile_fingerprint": self.profile_fingerprint,
            "embedding_set_id": self.embedding_set_id,
            "evidence_key": self.span.evidence_key,
            "targets": [target.model_dump(mode="json") for target in self.targets],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


__all__ = [
    "EmbeddingReference",
    "LegacyLocator",
    "Locator",
    "MultimodalSpan",
    "PinnedLogicalRef",
    "SpatialRegionLocator",
    "TemporalIntervalLocator",
    "TextRangeLocator",
    "VideoRegionTrackLocator",
    "VideoTrackFrame",
    "VideoTrackManifest",
    "VideoTrackRegion",
]
