"""Opt-in conversation wire contracts shared with the Rust contract crate."""

from __future__ import annotations

from typing import Literal

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


__all__ = [
    "CONVERSATION_WIRE_VERSION",
    "ConversationSummaryV1",
    "ConversationTranscriptV1",
    "ConversationTurnV1",
]
