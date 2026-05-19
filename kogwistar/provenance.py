from __future__ import annotations

"""Shared provenance primitives used across conversation and promotion flows."""

from collections.abc import Mapping
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidencePackDigest(BaseModel):
    """A compact, rehydratable description of an evidence pack."""

    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    depth: str = Field(
        "shallow", description="Materialization depth hint (e.g. shallow/deep)"
    )
    max_chars_per_item: int = Field(0, ge=0)
    max_total_chars: int = Field(0, ge=0)
    evidence_pack_hash: str | None = None

    model_config = ConfigDict(extra="allow")


def canonicalize_evidence_pack_digest(
    digest: Mapping[str, Any] | EvidencePackDigest,
) -> dict[str, Any]:
    """Normalize an evidence-pack payload for stable hashing."""

    if isinstance(digest, EvidencePackDigest):
        payload = digest.model_dump(mode="python")
    else:
        payload = dict(digest)

    payload["node_ids"] = sorted(
        str(node_id) for node_id in payload.get("node_ids") or [] if str(node_id)
    )
    payload["edge_ids"] = sorted(
        str(edge_id) for edge_id in payload.get("edge_ids") or [] if str(edge_id)
    )
    payload.pop("evidence_pack_hash", None)
    return payload


def evidence_pack_digest_hash(
    digest: Mapping[str, Any] | EvidencePackDigest,
) -> str:
    """Return a deterministic content hash for an evidence-pack payload."""

    payload = canonicalize_evidence_pack_digest(digest)
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return sha256(blob).hexdigest()
