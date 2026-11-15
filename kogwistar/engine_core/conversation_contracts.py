"""Opt-in conversation wire contracts shared with the Rust contract crate."""

from __future__ import annotations

from typing import Literal

from kogwistar.json_types import JsonValue
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONVERSATION_WIRE_VERSION = 1


class ConversationSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = CONVERSATION_WIRE_VERSION
    conversation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    start_node_id: str = Field(min_length=1)
    tail_node_id: str = ""
    turn_count: int = Field(ge=0)


class ConversationTurnV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = CONVERSATION_WIRE_VERSION
    node_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    role: str = Field(min_length=1)
    content: str
    entity_type: Literal["conversation_turn", "assistant_turn"]


class ConversationTranscriptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = CONVERSATION_WIRE_VERSION
    conversation_id: str = Field(min_length=1)
    turns: tuple[ConversationTurnV1, ...]

    @field_validator("turns")
    @classmethod
    def _turns_are_ordered(cls, turns: tuple[ConversationTurnV1, ...]):
        indices = [turn.turn_index for turn in turns]
        if indices != sorted(indices):
            raise ValueError("conversation turns must be ordered by turn_index")
        return turns


TOOL_AUDIT_WIRE_VERSION = 1


class ToolAuditItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = TOOL_AUDIT_WIRE_VERSION
    node_id: str = Field(min_length=1)
    entity_type: Literal["tool_call", "tool_result"]
    tool_name: str = Field(min_length=1)
    conversation_id: str | None = None
    turn_index: int | None = Field(default=None, ge=0)
    summary: str
    properties: dict[str, JsonValue] = Field(default_factory=dict)


class ToolAuditResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = TOOL_AUDIT_WIRE_VERSION
    conversation_id: str | None = None
    items: tuple[ToolAuditItemV1, ...]

    @field_validator("conversation_id")
    @classmethod
    def _conversation_id_is_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("conversation_id must not be empty")
        return value

    @field_validator("items")
    @classmethod
    def _items_match_conversation(
        cls, items: tuple[ToolAuditItemV1, ...], info: object
    ) -> tuple[ToolAuditItemV1, ...]:
        conversation_id = getattr(info, "data", {}).get("conversation_id")
        if conversation_id is not None and any(
            item.conversation_id not in (None, conversation_id) for item in items
        ):
            raise ValueError("tool audit item conversation_id does not match response")
        return items


__all__ = [
    "CONVERSATION_WIRE_VERSION",
    "ConversationSummaryV1",
    "ConversationTranscriptV1",
    "ConversationTurnV1",
    "TOOL_AUDIT_WIRE_VERSION",
    "ToolAuditItemV1",
    "ToolAuditResponseV1",
]
