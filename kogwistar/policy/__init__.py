from __future__ import annotations

"""Core knowledge-management policy protocols and conservative defaults."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


_VISIBILITY_VALUES = {"internal", "review", "knowledge", "projection", "wisdom"}


@dataclass(frozen=True, slots=True)
class PromotionContext:
    promotion_mode: str
    auto_accept_threshold: float
    default_accept_threshold: float = 0.95
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    should_promote: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceQueryDecision:
    where: dict[str, Any] = field(default_factory=dict)


class PromotionPolicy(Protocol):
    def decide(self, context: PromotionContext) -> PromotionDecision: ...


class ArtifactVisibilityPolicy(Protocol):
    def visibility_for(self, metadata: Mapping[str, Any]) -> str: ...


class ProjectionEligibilityPolicy(Protocol):
    def is_projection_eligible(self, metadata: Mapping[str, Any]) -> bool: ...


class DerivedKnowledgePolicy(Protocol):
    def group_key(self, node: Any) -> str: ...

    def source_query(self, *, workspace_id: str) -> SourceQueryDecision: ...

    def match_where(self, *, workspace_id: str, label: str) -> dict[str, Any]: ...

    def build_metadata(
        self,
        *,
        workspace_id: str,
        label: str,
        source_node_ids: Sequence[str],
        replaces_ids: Sequence[str],
        created_at_ms: int,
        artifact_kind: str,
    ) -> dict[str, Any]: ...


class WisdomPolicy(Protocol):
    @property
    def min_failure_signals(self) -> int: ...

    def source_query(self, *, workspace_id: str) -> SourceQueryDecision: ...

    def match_where(self, *, workspace_id: str, step_op: str) -> dict[str, Any]: ...

    def build_metadata(
        self,
        *,
        workspace_id: str,
        step_op: str,
        failure_count: int,
        evidence_run_ids: Sequence[str],
        replaces_ids: Sequence[str],
        created_at_ms: int,
        artifact_kind: str,
    ) -> dict[str, Any]: ...


class KnowledgeLifecyclePolicy(Protocol):
    def requires_provenance(self, artifact_kind: str) -> bool: ...

    def replacement_ids(self, existing: Sequence[Any]) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class DefaultPromotionPolicy:
    """Conservative promotion default: only auto-promote explicit sync requests."""

    default_accept_threshold: float = 0.95

    def decide(self, context: PromotionContext) -> PromotionDecision:
        mode = str(context.promotion_mode or "").strip().lower()
        if mode != "sync":
            return PromotionDecision(
                should_promote=False,
                reason="promotion_mode is not sync",
            )
        if float(context.auto_accept_threshold) > float(self.default_accept_threshold):
            return PromotionDecision(
                should_promote=False,
                reason="auto_accept_threshold is above the default accept threshold",
            )
        return PromotionDecision(
            should_promote=True,
            reason="sync promotion accepted by default policy",
            metadata={
                "promotion_mode": mode,
                "auto_accept_threshold": float(context.auto_accept_threshold),
                "default_accept_threshold": float(self.default_accept_threshold),
            },
        )


@dataclass(frozen=True, slots=True)
class DefaultArtifactVisibilityPolicy:
    """Metadata-driven visibility classifier with conservative fallbacks."""

    def visibility_for(self, metadata: Mapping[str, Any]) -> str:
        meta = dict(metadata or {})
        explicit = str(meta.get("visibility") or "").strip().lower()
        if explicit in _VISIBILITY_VALUES:
            return explicit
        if meta.get("projection_visible") is True:
            return "projection"
        stage = str(meta.get("knowledge_stage") or "").strip().lower()
        if stage in _VISIBILITY_VALUES:
            return stage
        if stage in {"promoted", "durable", "knowledge"}:
            return "knowledge"
        if stage in {"derived", "wisdom"}:
            return "wisdom"
        if stage in {"candidate", "review"}:
            return "review"
        return "internal"


@dataclass(frozen=True, slots=True)
class DefaultProjectionEligibilityPolicy:
    """Only explicit projection-visible artifacts may be projected."""

    visibility_policy: ArtifactVisibilityPolicy = field(
        default_factory=DefaultArtifactVisibilityPolicy
    )

    def is_projection_eligible(self, metadata: Mapping[str, Any]) -> bool:
        meta = dict(metadata or {})
        if meta.get("projection_visible") is True:
            return True
        return self.visibility_policy.visibility_for(meta) == "projection"


@dataclass(frozen=True, slots=True)
class DefaultDerivedKnowledgePolicy:
    """Stable grouping and versioned replacement defaults for derived knowledge."""

    def group_key(self, node: Any) -> str:
        metadata = dict(getattr(node, "metadata", {}) or {})
        label = metadata.get("label") or getattr(node, "label", None) or getattr(node, "summary", None)
        text = str(label or "").strip()
        return text or "Unknown Entity"

    def source_query(self, *, workspace_id: str) -> SourceQueryDecision:
        return SourceQueryDecision(where={"workspace_id": workspace_id})

    def match_where(self, *, workspace_id: str, label: str) -> dict[str, Any]:
        return {
            "workspace_id": workspace_id,
            "label": label,
        }

    def build_metadata(
        self,
        *,
        workspace_id: str,
        label: str,
        source_node_ids: Sequence[str],
        replaces_ids: Sequence[str],
        created_at_ms: int,
        artifact_kind: str,
    ) -> dict[str, Any]:
        return {
            "workspace_id": workspace_id,
            "artifact_kind": artifact_kind,
            "label": label,
            "source_node_ids": [str(item) for item in source_node_ids if str(item)],
            "replaces_ids": [str(item) for item in replaces_ids if str(item)],
            "created_at_ms": int(created_at_ms),
        }


@dataclass(frozen=True, slots=True)
class DefaultWisdomPolicy:
    """Execution-wisdom defaults for repeated-failure synthesis."""

    min_failure_signals: int = 2

    def source_query(self, *, workspace_id: str) -> SourceQueryDecision:
        return SourceQueryDecision(where={"workspace_id": workspace_id})

    def match_where(self, *, workspace_id: str, step_op: str) -> dict[str, Any]:
        return {
            "workspace_id": workspace_id,
            "step_op": step_op,
        }

    def build_metadata(
        self,
        *,
        workspace_id: str,
        step_op: str,
        failure_count: int,
        evidence_run_ids: Sequence[str],
        replaces_ids: Sequence[str],
        created_at_ms: int,
        artifact_kind: str,
    ) -> dict[str, Any]:
        return {
            "workspace_id": workspace_id,
            "artifact_kind": artifact_kind,
            "step_op": step_op,
            "failure_count": int(failure_count),
            "evidence_run_ids": [str(item) for item in evidence_run_ids if str(item)],
            "replaces_ids": [str(item) for item in replaces_ids if str(item)],
            "created_at_ms": int(created_at_ms),
        }


@dataclass(frozen=True, slots=True)
class DefaultKnowledgeLifecyclePolicy:
    """Lifecycle helpers for versioned, provenance-preserving artifacts."""

    provenance_required_stages: frozenset[str] = frozenset(
        {"knowledge", "promoted", "durable", "derived", "wisdom"}
    )

    def requires_provenance(self, artifact_kind: str) -> bool:
        return str(artifact_kind or "").strip().lower() in self.provenance_required_stages

    def replacement_ids(self, existing: Sequence[Any]) -> list[str]:
        out: list[str] = []
        for item in existing:
            item_id = getattr(item, "id", item)
            text = str(item_id or "").strip()
            if text:
                out.append(text)
        return out


__all__ = [
    "ArtifactVisibilityPolicy",
    "DefaultArtifactVisibilityPolicy",
    "DefaultDerivedKnowledgePolicy",
    "DefaultKnowledgeLifecyclePolicy",
    "DefaultPromotionPolicy",
    "DefaultProjectionEligibilityPolicy",
    "DefaultWisdomPolicy",
    "DerivedKnowledgePolicy",
    "KnowledgeLifecyclePolicy",
    "PromotionContext",
    "PromotionDecision",
    "PromotionPolicy",
    "ProjectionEligibilityPolicy",
    "SourceQueryDecision",
    "WisdomPolicy",
]
