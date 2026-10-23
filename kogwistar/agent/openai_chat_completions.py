"""OpenAI Chat Completions SSE adapter for the normalized streaming contract."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import cast

import httpx

from kogwistar.json_types import JsonObject, JsonValue

from .schema_validation import validate_json_schema
from .streaming import (
    IncrementalJsonObjectParser,
    ProviderCapabilities,
    ProviderEvent,
    ProviderFieldCompleted,
    ProviderRequest,
    ProviderResponseCompleted,
    ProviderResponseFailed,
    ProviderResponseStarted,
    ProviderTextDelta,
    ProviderToolCallCompleted,
    StructuredOutputError,
)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


async def _iter_sse_lines(
    response: httpx.Response,
    *,
    max_line_bytes: int,
) -> AsyncIterator[str]:
    """Decode SSE lines with a cap applied before accumulating an entire line."""
    pending = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=16_384):
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline
            segment = chunk[start:end]
            if len(pending) + len(segment) > max_line_bytes:
                raise StructuredOutputError("SSE line exceeds configured limit")
            pending.extend(segment)
            if newline < 0:
                break
            if pending.endswith(b"\r"):
                pending.pop()
            try:
                yield pending.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StructuredOutputError("SSE line is not valid UTF-8") from exc
            pending.clear()
            start = newline + 1
    if pending:
        if pending.endswith(b"\r"):
            pending.pop()
        try:
            yield pending.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StructuredOutputError("SSE line is not valid UTF-8") from exc


@dataclass
class _ToolCallBuffer:
    call_id_parts: list[str] = field(default_factory=list)
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)

    @property
    def call_id(self) -> str:
        return "".join(self.call_id_parts)

    @property
    def name(self) -> str:
        return "".join(self.name_parts)

    @property
    def arguments(self) -> str:
        return "".join(self.argument_parts)


@dataclass
class _ChatStreamState:
    parser: IncrementalJsonObjectParser | None
    response_schema: JsonObject | None = None
    defer_structured_fields: bool = False
    response_id: str | None = None
    sequence: int = 0
    started: bool = False
    finish_reason: str | None = None
    output_parts: list[str] = field(default_factory=list)
    parsed_fields: list[tuple[str, JsonValue]] = field(default_factory=list)
    output_chars: int = 0
    tool_calls: dict[int, _ToolCallBuffer] = field(default_factory=dict)
    tool_argument_chars: int = 0
    usage: Mapping[str, object] = field(default_factory=dict)
    terminal: bool = False

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


class OpenAIChatCompletionsProvider:
    """Normalize streamed Chat Completions chunks; never execute tools here."""

    provider_id = "openai.chat_completions"
    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calls=True,
        reasoning_summaries=False,
        steering=False,
        cancellation=False,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
        max_structured_chars: int = 64_000,
        max_response_chars: int = 256_000,
        max_sse_line_bytes: int = 1_048_576,
        max_sse_event_chars: int = 1_048_576,
        max_tool_calls: int = 32,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = str(
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY") or ""
        ).strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        self._max_structured_chars = int(max_structured_chars)
        self._max_response_chars = int(max_response_chars)
        self._max_sse_line_bytes = int(max_sse_line_bytes)
        self._max_sse_event_chars = int(max_sse_event_chars)
        self._max_tool_calls = int(max_tool_calls)
        self._client = client
        if self._timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            self._max_structured_chars < 2
            or self._max_response_chars < 1
            or self._max_sse_line_bytes < 1
            or self._max_sse_event_chars < 1
        ):
            raise ValueError("stream character limits are too small")
        if self._max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")

    def _payload(self, request: ProviderRequest) -> JsonObject:
        if request.previous_response_id:
            raise ValueError("Chat Completions requires history in messages, not previous_response_id")
        messages: list[JsonObject] = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        messages.append({"role": "user", "content": request.input})
        payload: JsonObject = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "n": 1,
        }
        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "kogwistar_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        if request.tools:
            payload["tools"] = list(request.tools)
        return payload

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        if not self._api_key:
            yield ProviderResponseFailed(
                error_code="missing_api_key",
                message="OPENAI_API_KEY is not configured",
            )
            return
        try:
            payload = self._payload(request)
        except ValueError as exc:
            yield ProviderResponseFailed(
                error_code="unsupported_request",
                message=str(exc)[:300],
            )
            return

        parser = (
            IncrementalJsonObjectParser(max_chars=self._max_structured_chars)
            if request.response_schema is not None
            else None
        )
        state = _ChatStreamState(
            parser=parser,
            response_schema=request.response_schema,
            defer_structured_fields=parser is not None and bool(request.tools),
        )
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        close_client = self._client is None
        saw_done = False
        data_lines: list[str] = []
        data_chars = 0
        try:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                async for line in _iter_sse_lines(
                    response,
                    max_line_bytes=self._max_sse_line_bytes,
                ):
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data = line[5:].lstrip()
                        data_chars += len(data)
                        if data_chars > self._max_sse_event_chars:
                            raise StructuredOutputError("SSE event exceeds configured limit")
                        data_lines.append(data)
                        continue
                    if line or not data_lines:
                        continue
                    raw = "\n".join(data_lines)
                    data_lines.clear()
                    data_chars = 0
                    if raw == "[DONE]":
                        saw_done = True
                        for event in self._finish(state):
                            yield event
                        break
                    for event in self._consume_chunk(
                        json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant),
                        state,
                    ):
                        yield event
                    if state.terminal:
                        return
                if data_lines and not saw_done:
                    raw = "\n".join(data_lines)
                    data_lines.clear()
                    data_chars = 0
                    if raw == "[DONE]":
                        saw_done = True
                        for event in self._finish(state):
                            yield event
                    else:
                        for event in self._consume_chunk(
                            json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant),
                            state,
                        ):
                            yield event
                        if state.terminal:
                            return
                if not state.terminal:
                    if not saw_done:
                        yield self._fail(state, "incomplete_stream", "Chat Completions stream ended before [DONE]")
                    else:
                        yield self._fail(state, "incomplete_response", "Chat Completions stream ended without a terminal result")
        except (httpx.HTTPError, json.JSONDecodeError, StructuredOutputError, TypeError, ValueError) as exc:
            message = (
                str(exc)[:300]
                if isinstance(exc, StructuredOutputError)
                else "OpenAI Chat Completions streaming request failed"
            )
            yield self._fail(
                state,
                type(exc).__name__.lower(),
                message,
                retryable=isinstance(exc, httpx.TimeoutException),
            )
        finally:
            if close_client:
                await client.aclose()

    def _consume_chunk(self, raw: object, state: _ChatStreamState) -> list[ProviderEvent]:
        if not isinstance(raw, Mapping):
            raise StructuredOutputError("Chat Completions SSE data must be an object")
        chunk = cast(Mapping[str, object], raw)
        if isinstance(chunk.get("error"), Mapping):
            return [self._fail(state, "provider_error", "Chat Completions returned an API error")]
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            if state.response_id is not None and state.response_id != chunk_id:
                raise StructuredOutputError("Chat Completions response ID changed mid-stream")
            state.response_id = chunk_id
        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            state.usage = cast(Mapping[str, object], usage)
        choices = chunk.get("choices")
        if choices is None:
            return []
        if not isinstance(choices, list) or len(choices) > 1:
            raise StructuredOutputError("Chat Completions must contain at most one choice")
        if not choices:
            return []
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise StructuredOutputError("Chat Completions choice must be an object")
        choice_map = cast(Mapping[str, object], choice)
        index = choice_map.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index != 0:
            raise StructuredOutputError("Chat Completions returned an unexpected choice index")
        if state.finish_reason is not None:
            raise StructuredOutputError("received a choice chunk after finish_reason")

        events: list[ProviderEvent] = []
        if not state.started:
            state.started = True
            events.append(
                ProviderResponseStarted(
                    provider_sequence=state.next_sequence(), response_id=state.response_id
                )
            )
        delta = _mapping(choice_map.get("delta"))
        unsupported_delta = set(delta) - {"role", "content", "tool_calls", "refusal"}
        if unsupported_delta:
            raise StructuredOutputError("Chat Completions returned an unsupported delta event")
        if delta.get("refusal"):
            events.append(self._fail(state, "provider_refusal", "provider refused the request"))
            return events
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise StructuredOutputError("Chat Completions content delta must be text")
            if state.output_chars + len(content) > self._max_response_chars:
                raise StructuredOutputError("response text exceeds configured limit")
            state.output_parts.append(content)
            state.output_chars += len(content)
            common = {
                "provider_sequence": state.next_sequence(),
                "response_id": state.response_id,
            }
            events.append(ProviderTextDelta(**common, delta=content))
            if state.parser is not None and not state.defer_structured_fields:
                state.parsed_fields.extend(state.parser.feed(content))

        tool_deltas = delta.get("tool_calls")
        if tool_deltas is not None:
            if not isinstance(tool_deltas, list):
                raise StructuredOutputError("tool_calls delta must be a list")
            for tool_delta in tool_deltas:
                item = _mapping(tool_delta)
                tool_type = item.get("type")
                if tool_type is not None and tool_type != "function":
                    raise StructuredOutputError(
                        "provider-managed Chat Completions tools are unsupported"
                    )
                tool_index = item.get("index")
                if not isinstance(tool_index, int) or isinstance(tool_index, bool) or tool_index < 0:
                    raise StructuredOutputError("tool-call delta has invalid index")
                if tool_index >= self._max_tool_calls:
                    raise StructuredOutputError("tool-call count exceeds configured limit")
                buffer = state.tool_calls.setdefault(tool_index, _ToolCallBuffer())
                call_id = item.get("id")
                if isinstance(call_id, str) and call_id:
                    current_id = buffer.call_id
                    if call_id.startswith(current_id):
                        buffer.call_id_parts[:] = [call_id]
                    elif not current_id.startswith(call_id):
                        buffer.call_id_parts.append(call_id)
                    if len(buffer.call_id) > 256:
                        raise StructuredOutputError("tool-call ID exceeds configured limit")
                function = _mapping(item.get("function"))
                name = function.get("name")
                arguments = function.get("arguments")
                if name is not None:
                    if not isinstance(name, str):
                        raise StructuredOutputError("tool-call name fragment must be text")
                    buffer.name_parts.append(name)
                    if len(buffer.name) > 256:
                        raise StructuredOutputError("tool-call name exceeds configured limit")
                if arguments is not None:
                    if not isinstance(arguments, str):
                        raise StructuredOutputError("tool-call arguments fragment must be text")
                    buffer.argument_parts.append(arguments)
                    state.tool_argument_chars += len(arguments)
                if state.tool_argument_chars > self._max_structured_chars:
                    raise StructuredOutputError("tool-call arguments exceed configured limit")
        finish_reason = choice_map.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str):
                raise StructuredOutputError("finish_reason must be text")
            state.finish_reason = finish_reason
            if finish_reason in {"length", "content_filter"}:
                message = "Chat Completions output was truncated" if finish_reason == "length" else "Chat Completions output was filtered"
                events.append(self._fail(state, finish_reason, message))
            elif finish_reason not in {"stop", "tool_calls"}:
                events.append(self._fail(state, "unsupported_finish_reason", "Chat Completions used an unsupported finish reason"))
        return events

    def _finish(self, state: _ChatStreamState) -> list[ProviderEvent]:
        if state.terminal:
            return []
        if not state.started or state.finish_reason is None:
            return [self._fail(state, "incomplete_stream", "Chat Completions ended without a completed choice")]
        if state.finish_reason not in {"stop", "tool_calls"}:
            return [self._fail(state, "incomplete_response", "Chat Completions did not finish successfully")]
        if state.tool_calls and state.finish_reason != "tool_calls":
            return [self._fail(state, "incomplete_tool_calls", "tool-call deltas were not completed successfully")]
        if state.finish_reason == "tool_calls" and not state.tool_calls:
            return [self._fail(state, "missing_tool_calls", "provider finished as tool_calls without any tool call")]

        events: list[ProviderEvent] = []
        completed_calls: list[ProviderToolCallCompleted] = []
        for index in sorted(state.tool_calls):
            call = state.tool_calls[index]
            if not call.call_id or not call.name or len(call.name) > 256:
                return [self._fail(state, "invalid_tool_call", "completed tool call is missing a valid ID or name")]
            try:
                arguments = json.loads(
                    call.arguments,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, StructuredOutputError) as exc:
                message = str(exc)[:300] if isinstance(exc, StructuredOutputError) else "completed tool arguments are invalid JSON"
                return [self._fail(state, "invalid_tool_call_arguments", message)]
            if not isinstance(arguments, Mapping):
                return [self._fail(state, "invalid_tool_call_arguments", "completed tool arguments must be an object")]
            completed_calls.append(
                ProviderToolCallCompleted(
                    provider_sequence=state.next_sequence(),
                    response_id=state.response_id,
                    call_id=call.call_id,
                    name=call.name,
                    arguments=cast(JsonObject, dict(arguments)),
                )
            )
        structured: JsonObject | None = None
        if state.parser is not None and state.finish_reason == "stop":
            if state.defer_structured_fields:
                state.parsed_fields.extend(state.parser.feed("".join(state.output_parts)))
            structured = state.parser.finish()
            if state.response_schema is not None:
                validate_json_schema(structured, state.response_schema)
            for name, value in state.parsed_fields:
                events.append(
                    ProviderFieldCompleted(
                        provider_sequence=state.next_sequence(),
                        response_id=state.response_id,
                        field_name=name,
                        value=value,
                    )
                )
        events.extend(completed_calls)
        usage = state.usage
        events.append(
            ProviderResponseCompleted(
                provider_sequence=state.next_sequence(),
                response_id=state.response_id,
                text="".join(state.output_parts),
                structured_output=structured,
                has_tool_calls=bool(state.tool_calls),
                input_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
            )
        )
        state.terminal = True
        return events

    @staticmethod
    def _fail(
        state: _ChatStreamState,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> ProviderResponseFailed:
        state.terminal = True
        return ProviderResponseFailed(
            provider_sequence=state.next_sequence(),
            response_id=state.response_id,
            error_code=code,
            message=message,
            retryable=retryable,
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StructuredOutputError(f"non-standard JSON number is not allowed: {value}")


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["OpenAIChatCompletionsProvider"]
