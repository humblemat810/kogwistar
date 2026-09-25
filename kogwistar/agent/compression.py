"""Bounded conversation-context compression over ordinary graph artifacts.

Compression is a derived view. It never deletes or rewrites source nodes.
Summary persistence uses existing conversation node/edge models so normal ACL,
provenance, and graph traversal remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from kogwistar.conversation.models import ConversationEdge, ConversationNode
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.id_provider import stable_id
from .workflows import build_normal_workflow


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    """Versioned retention policy independent from model reasoning effort."""

    name: str
    max_items: int
    max_chars: int
    version: str = "v1"

    def __post_init__(self) -> None:
        if self.name not in {"low", "medium", "high", "ultra"}:
            raise ValueError("compression policy must be low, medium, high, or ultra")
        if int(self.max_items) < 1 or int(self.max_chars) < 1:
            raise ValueError("compression limits must be positive")


POLICIES: dict[str, CompressionPolicy] = {
    "low": CompressionPolicy("low", max_items=64, max_chars=48_000),
    "medium": CompressionPolicy("medium", max_items=32, max_chars=32_000),
    "high": CompressionPolicy("high", max_items=16, max_chars=16_000),
    "ultra": CompressionPolicy("ultra", max_items=8, max_chars=8_000),
}


@dataclass(frozen=True, slots=True)
class CompressionRequest:
    conversation_id: str
    source_items: tuple[dict[str, Any], ...]
    policy: CompressionPolicy
    cross_conversation: bool = False
    requested_by_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompressionDecision:
    requested: bool
    selected_items: tuple[dict[str, Any], ...]
    source_refs: tuple[str, ...]
    estimated_chars: int
    reason: str
    policy_version: str


def should_request_compression(
    items: Sequence[Mapping[str, Any]],
    *,
    policy: CompressionPolicy,
    max_source_chars: int | None = None,
) -> bool:
    """Pure threshold hook; only requests work and never mutates history."""

    char_count = sum(len(str(item.get("content") or item.get("summary") or "")) for item in items)
    return len(items) > policy.max_items or char_count > int(max_source_chars or policy.max_chars)


@dataclass(frozen=True, slots=True)
class CompressionResult:
    summary_node_id: str
    source_refs: tuple[str, ...]
    provenance_edge_id: str
    policy_version: str
    source_fingerprint: str


def policy_for(name: str) -> CompressionPolicy:
    try:
        return POLICIES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown compression policy: {name}") from exc


def build_compression_workflow(*, workflow_id: str = "agent.context-compression.v1"):
    """Return an ordinary audited workflow design for compression execution."""

    return build_normal_workflow(
        workflow_id=workflow_id,
        execute_op="agent.context_compress",
    )


def select_for_compression(
    items: Sequence[Mapping[str, Any]],
    *,
    policy: CompressionPolicy,
    authorize: Callable[[Mapping[str, Any]], bool] | None = None,
) -> CompressionDecision:
    """Select newest authorized items without mutating source history."""

    # A summary must not silently merge alternate conversation branches.  The
    # newest source selects the branch; callers needing a deliberate merge
    # must normalize branches before invoking this contract.
    branch_items = list(items)
    if branch_items:
        newest_branch = branch_items[-1].get("branch_id")
        if newest_branch is not None:
            branch_items = [
                item for item in branch_items if item.get("branch_id") == newest_branch
            ]
    selected: list[dict[str, Any]] = []
    chars = 0
    for raw in reversed(branch_items):
        item = dict(raw)
        if authorize is not None and not authorize(item):
            continue
        node_id = str(item.get("id") or item.get("node_id") or "")
        if not node_id:
            continue
        text = str(item.get("content") or item.get("summary") or "")
        if selected and (len(selected) >= policy.max_items or chars + len(text) > policy.max_chars):
            break
        selected.append(item)
        chars += len(text)
    selected.reverse()
    refs = tuple(str(item.get("id") or item.get("node_id")) for item in selected)
    return CompressionDecision(
        requested=bool(selected),
        selected_items=tuple(selected),
        source_refs=refs,
        estimated_chars=chars,
        reason="threshold_or_explicit_request" if selected else "no_authorized_source_items",
        policy_version=policy.version,
    )


def build_compression_request(
    *,
    conversation_id: str,
    items: Sequence[Mapping[str, Any]],
    policy: CompressionPolicy | str = "medium",
    allow_cross_conversation: bool = False,
    requested_by_run_id: str | None = None,
    authorize: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[CompressionRequest, CompressionDecision]:
    selected_policy = policy_for(policy) if isinstance(policy, str) else policy
    if not allow_cross_conversation:
        items = [item for item in items if str(item.get("conversation_id", conversation_id)) == conversation_id]
    request = CompressionRequest(
        conversation_id=conversation_id,
        source_items=tuple(dict(item) for item in items),
        policy=selected_policy,
        cross_conversation=allow_cross_conversation,
        requested_by_run_id=requested_by_run_id,
    )
    return request, select_for_compression(items, policy=selected_policy, authorize=authorize)


def _summary_span(conversation_id: str, summary_id: str, excerpt: str) -> Span:
    return Span(
        collection_page_url=f"conversation/{conversation_id}",
        document_page_url=f"conversation/{conversation_id}#{summary_id}",
        doc_id=f"conv:{conversation_id}",
        insertion_method="agent_context_compression",
        page_number=1,
        start_char=0,
        end_char=max(1, len(excerpt)),
        excerpt=excerpt[:512],
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="derived summary"),
    )


def persist_summary_projection(
    engine: Any,
    *,
    conversation_id: str,
    source_refs: Sequence[str],
    summary_text: str,
    policy: CompressionPolicy,
    run_id: str | None = None,
) -> CompressionResult:
    """Persist one additive summary node and provenance edge."""

    refs = tuple(str(ref) for ref in source_refs)
    fingerprint = hashlib.sha256(
        json.dumps({"refs": refs, "summary": summary_text, "policy": policy.version}, sort_keys=True).encode()
    ).hexdigest()
    summary_id = str(stable_id("conversation.summary", conversation_id, fingerprint))
    edge_id = str(stable_id("conversation.summary.edge", summary_id, *refs))
    node = ConversationNode(
        id=summary_id,
        label="Compressed context",
        type="entity",
        doc_id=f"conv:{conversation_id}",
        summary=str(summary_text),
        role="system",
        conversation_id=conversation_id,
        turn_index=None,
        user_id=None,
        mentions=[Grounding(spans=[_summary_span(conversation_id, summary_id, str(summary_text))])],
        properties={"source_refs": list(refs), "policy": policy.name},
        metadata={
            "level_from_root": 1,
            "entity_type": "conversation_context_summary",
            "summary_policy": policy.name,
            "summary_policy_version": policy.version,
            "source_fingerprint": fingerprint,
            "run_id": run_id,
        },
        domain_id=None,
        canonical_entity_id=None,
    )
    edge = ConversationEdge(
        id=edge_id,
        source_ids=[summary_id],
        target_ids=list(refs),
        source_edge_ids=[],
        target_edge_ids=[],
        relation="summarizes",
        label="summarizes",
        type="relationship",
        doc_id=f"conv:{conversation_id}",
        summary="Compressed context provenance",
        properties={"policy": policy.name, "source_fingerprint": fingerprint},
        metadata={"causal_type": "summary", "conversation_id": conversation_id},
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        mentions=[Grounding(spans=[_summary_span(conversation_id, summary_id, str(summary_text))])],
    )
    engine.write.add_node(node)
    engine.write.add_edge(edge)
    return CompressionResult(
        summary_node_id=summary_id,
        source_refs=refs,
        provenance_edge_id=edge_id,
        policy_version=policy.version,
        source_fingerprint=fingerprint,
    )


__all__ = [
    "CompressionDecision",
    "CompressionPolicy",
    "CompressionRequest",
    "CompressionResult",
    "POLICIES",
    "build_compression_request",
    "build_compression_workflow",
    "persist_summary_projection",
    "policy_for",
    "select_for_compression",
    "should_request_compression",
]
