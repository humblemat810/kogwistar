"""Typed streaming-agent events and bounded structured-output parsing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kogwistar.json_types import JsonObject, JsonValue


class ProviderRequest(BaseModel):
    """Provider-neutral request for one model response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["v1"] = "v1"
    model: str
    instructions: str = ""
    input: str
    response_schema: JsonObject | None = None
    tools: tuple[JsonObject, ...] = ()
    previous_response_id: str | None = None
    max_output_tokens: int | None = None


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    structured_output: bool = False
    tool_calls: bool = False
    reasoning_summaries: bool = False
    steering: bool = False
    cancellation: bool = False


class ProviderEventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["v1"] = "v1"
    provider_event_id: str | None = None
    provider_sequence: int | None = None
    response_id: str | None = None


class ProviderResponseStarted(ProviderEventBase):
    kind: Literal["response_started"] = "response_started"


class ProviderTextDelta(ProviderEventBase):
    kind: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderFieldCompleted(ProviderEventBase):
    kind: Literal["field_completed"] = "field_completed"
    field_name: str
    value: JsonValue


class ProviderReasoningSummary(ProviderEventBase):
    kind: Literal["reasoning_summary"] = "reasoning_summary"
    text: str


class ProviderToolCallCompleted(ProviderEventBase):
    kind: Literal["tool_call_completed"] = "tool_call_completed"
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: JsonObject


class ProviderResponseCompleted(ProviderEventBase):
    kind: Literal["response_completed"] = "response_completed"
    text: str = ""
    structured_output: JsonObject | None = None
    has_tool_calls: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderResponseFailed(ProviderEventBase):
    kind: Literal["response_failed"] = "response_failed"
    error_code: str
    message: str
    retryable: bool = False


ProviderEvent: TypeAlias = Annotated[
    ProviderResponseStarted
    | ProviderTextDelta
    | ProviderFieldCompleted
    | ProviderReasoningSummary
    | ProviderToolCallCompleted
    | ProviderResponseCompleted
    | ProviderResponseFailed,
    Field(discriminator="kind"),
]


@runtime_checkable
class StreamingModelProvider(Protocol):
    """Async event source; provider events carry no Kogwistar authority."""

    provider_id: str
    capabilities: ProviderCapabilities

    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]: ...


class StructuredOutputError(ValueError):
    """Invalid, ambiguous, or oversized streamed JSON object."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StructuredOutputError(f"non-standard JSON number is not allowed: {value}")


class IncrementalJsonObjectParser:
    """Parse top-level object fields as they complete, with a strict size cap."""

    def __init__(self, *, max_chars: int = 64_000) -> None:
        if max_chars < 2:
            raise ValueError("max_chars must be at least 2")
        self.max_chars = int(max_chars)
        self._buffer = ""
        self._cursor = 0
        self._started = False
        self._finished = False
        self._seen_keys: set[str] = set()
        self._pending_key: str | None = None
        self._awaiting_value = False
        self._value_start: int | None = None
        self._value_scan = 0
        self._value_depth = 0
        self._in_string = False
        self._escaped = False

    def feed(self, chunk: str) -> tuple[tuple[str, JsonValue], ...]:
        if self._finished:
            raise StructuredOutputError("cannot feed after finish")
        if not isinstance(chunk, str):
            raise TypeError("JSON stream chunks must be text")
        self._buffer += chunk
        if len(self._buffer) > self.max_chars:
            raise StructuredOutputError("structured output exceeds configured limit")
        fields: list[tuple[str, JsonValue]] = []
        while True:
            self._skip_ws()
            if not self._started:
                if self._cursor >= len(self._buffer):
                    break
                if self._buffer[self._cursor] != "{":
                    raise StructuredOutputError("structured output must be a JSON object")
                self._started = True
                self._cursor += 1
                continue
            if self._value_start is not None:
                boundary = self._scan_value()
                if boundary is None:
                    break
                raw = self._buffer[self._value_start:boundary]
                try:
                    value = json.loads(
                        raw,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except (json.JSONDecodeError, StructuredOutputError) as exc:
                    raise StructuredOutputError(f"invalid field value: {exc}") from exc
                key = self._pending_key
                if key is None:
                    raise StructuredOutputError("field value has no key")
                fields.append((key, value))
                self._pending_key = None
                self._value_start = None
                self._cursor = boundary
                continue
            if self._pending_key is not None:
                if not self._awaiting_value:
                    if self._cursor >= len(self._buffer):
                        break
                    if self._buffer[self._cursor] != ":":
                        raise StructuredOutputError("expected ':' after object key")
                    self._cursor += 1
                    self._awaiting_value = True
                    self._skip_ws()
                    if self._cursor >= len(self._buffer):
                        break
                self._value_start = self._cursor
                self._value_scan = self._cursor
                self._value_depth = 0
                self._in_string = False
                self._escaped = False
                self._awaiting_value = False
                continue
            if self._cursor >= len(self._buffer):
                break
            char = self._buffer[self._cursor]
            if char == "}":
                self._cursor += 1
                self._skip_ws()
                if self._cursor < len(self._buffer):
                    raise StructuredOutputError("trailing content after JSON object")
                break
            if char == ",":
                self._cursor += 1
                self._skip_ws()
            key, end = self._scan_key()
            if key is None:
                break
            if key in self._seen_keys:
                raise StructuredOutputError(f"duplicate JSON key: {key}")
            self._seen_keys.add(key)
            self._pending_key = key
            self._cursor = end
            continue
        return tuple((key, value) for key, value in fields)

    def finish(self) -> JsonObject:
        if self._finished:
            raise StructuredOutputError("finish may only be called once")
        self.feed("")
        self._finished = True
        try:
            value = json.loads(
                self._buffer,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, StructuredOutputError) as exc:
            raise StructuredOutputError(f"incomplete or invalid JSON object: {exc}") from exc
        if not isinstance(value, dict):
            raise StructuredOutputError("structured output must be a JSON object")
        return value  # type: ignore[return-value]

    def _skip_ws(self) -> None:
        while self._cursor < len(self._buffer) and self._buffer[self._cursor] in " \t\r\n":
            self._cursor += 1

    def _scan_key(self) -> tuple[str | None, int]:
        if self._cursor >= len(self._buffer):
            return None, self._cursor
        if self._buffer[self._cursor] != '"':
            if self._buffer[self._cursor] == "}":
                return None, self._cursor
            raise StructuredOutputError("expected a quoted object key")
        start = self._cursor
        index = start + 1
        escaped = False
        while index < len(self._buffer):
            char = self._buffer[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                try:
                    key = json.loads(self._buffer[start:index + 1])
                except json.JSONDecodeError as exc:
                    raise StructuredOutputError(f"invalid object key: {exc}") from exc
                return str(key), index + 1
            index += 1
        return None, self._cursor

    def _scan_value(self) -> int | None:
        index = self._value_scan
        while index < len(self._buffer):
            char = self._buffer[index]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char in "[{":
                self._value_depth += 1
            elif char in "]}":
                if self._value_depth == 0:
                    return index
                self._value_depth -= 1
                if self._value_depth == 0:
                    index += 1
                    self._value_scan = index
                    return index
            elif char == "," and self._value_depth == 0:
                return index
            index += 1
        self._value_scan = index
        return None


__all__ = [
    "IncrementalJsonObjectParser",
    "ProviderCapabilities",
    "ProviderEvent",
    "ProviderFieldCompleted",
    "ProviderReasoningSummary",
    "ProviderRequest",
    "ProviderResponseCompleted",
    "ProviderResponseFailed",
    "ProviderResponseStarted",
    "ProviderTextDelta",
    "ProviderToolCallCompleted",
    "StreamingModelProvider",
    "StructuredOutputError",
]
