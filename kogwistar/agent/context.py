"""Serializable agent context contracts over existing graph read primitives."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContextPolicy(BaseModel):
    """Bounded context selection policy; it never owns conversation history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    include_current_conversation: bool = True
    allow_cross_conversation: bool = False
    max_items: int = 32
    max_chars: int = 32_000
    compression: Literal["low", "medium", "high", "ultra"] = "medium"
    memory_query: str | None = None
    knowledge_query: str | None = None
    wisdom_query: str | None = None

    @field_validator("max_items", "max_chars")
    @classmethod
    def _positive_bound(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("context bounds must be positive")
        return int(value)


class ContextSnapshot(BaseModel):
    """A bounded, auditable view; source graph remains authoritative."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = "v1"
    conversation_id: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    truncated: bool = False
    compression: str = "medium"

    def stable_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()


__all__ = ["ContextPolicy", "ContextSnapshot"]
