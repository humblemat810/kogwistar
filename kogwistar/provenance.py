from __future__ import annotations

"""Shared provenance primitives used across conversation and promotion flows."""

from collections.abc import Mapping
from hashlib import sha256
import json
from typing import cast

from .json_types import JsonValue
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


def _string_ids(value: JsonValue | None) -> list[str]:
    """Read a JSON list as string identifiers, ignoring non-list values."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def canonicalize_evidence_pack_digest(
    digest: Mapping[str, JsonValue] | EvidencePackDigest,
) -> dict[str, JsonValue]:
    """Normalize an evidence-pack payload for stable hashing."""

    if isinstance(digest, EvidencePackDigest):
        payload = cast(dict[str, JsonValue], digest.model_dump(mode="python"))
    else:
        payload = dict(digest)

    payload["node_ids"] = cast(
        list[JsonValue], sorted(_string_ids(payload.get("node_ids")))
    )
    payload["edge_ids"] = cast(
        list[JsonValue], sorted(_string_ids(payload.get("edge_ids")))
    )
    payload.pop("evidence_pack_hash", None)
    return payload


def evidence_pack_digest_hash(
    digest: Mapping[str, JsonValue] | EvidencePackDigest,
) -> str:
    """Return a deterministic content hash for an evidence-pack payload."""

    if isinstance(digest, EvidencePackDigest):
        bridge_payload = cast(dict[str, JsonValue], digest.model_dump(mode="python"))
    else:
        bridge_payload = dict(digest)
    payload = canonicalize_evidence_pack_digest(digest)
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    python_value = sha256(blob).hexdigest()
    from ._rust_bridge import contract_evidence_pack_digest_hash

    return contract_evidence_pack_digest_hash(
        value=bridge_payload,
        python_value=python_value,
    )
