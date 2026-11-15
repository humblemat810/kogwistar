"""OpenAI Responses API streaming adapter using the existing httpx dependency."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import cast

import httpx

from kogwistar.json_types import JsonObject, JsonValue

from .schema_validation import validate_json_schema
from .streaming import (
    IncrementalJsonObjectParser,
    ProviderCapabilities,
    ProviderEvent,
    ProviderFieldCompleted,
    ProviderReasoningSummary,
    ProviderRequest,
    ProviderResponseCompleted,
    ProviderResponseFailed,
    ProviderResponseStarted,
    ProviderTextDelta,
    ProviderToolCallCompleted,
    StructuredOutputError,
)


def _json_object(value: object) -> JsonObject:
    if isinstance(value, Mapping):
        return cast(JsonObject, dict(value))
    return {}


class OpenAIResponsesProvider:
    """Stream Responses events and normalize them without executing tools."""

    provider_id = "openai.responses"
    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calls=True,
        reasoning_summaries=True,
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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = str(
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY") or ""
        ).strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        self._max_structured_chars = int(max_structured_chars)
        self._client = client
        if self._timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self._max_structured_chars < 2:
            raise ValueError("max_structured_chars must be at least 2")

    def _payload(self, request: ProviderRequest) -> JsonObject:
        payload: JsonObject = {
            "model": request.model,
            "input": request.input,
            "stream": True,
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.previous_response_id:
            payload["previous_response_id"] = request.previous_response_id
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.response_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "kogwistar_response",
                    "strict": True,
                    "schema": request.response_schema,
                }
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
        parser = (
            IncrementalJsonObjectParser(max_chars=self._max_structured_chars)
            if request.response_schema is not None
            else None
        )
        response_id: str | None = None
        sequence: int | None = None
        output_text: list[str] = []
        parsed_fields: list[tuple[str, JsonValue]] = []
        tool_calls_seen: list[bool] = []
        terminal = False
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        close_client = self._client is None
        try:
            async with client.stream(
                "POST",
                f"{self._base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=self._payload(request),
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if not line:
                        if data_lines:
                            raw = "\n".join(data_lines)
                            data_lines.clear()
                            if raw != "[DONE]":
                                async for event in self._normalize_event(
                                    json.loads(
                                        raw,
                                        object_pairs_hook=_unique_object,
                                        parse_constant=_reject_constant,
                                    ),
                                    parser=parser,
                                    output_text=output_text,
                                    parsed_fields=parsed_fields,
                                    response_schema=request.response_schema,
                                    tool_calls_seen=tool_calls_seen,
                                    response_id=response_id,
                                    provider_sequence=sequence,
                                ):
                                    if event.response_id:
                                        response_id = event.response_id
                                    if event.provider_sequence is not None:
                                        sequence = event.provider_sequence
                                    yield event
                                    if isinstance(event, (ProviderResponseCompleted, ProviderResponseFailed)):
                                        terminal = True
                                        break
                                if terminal:
                                    break
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    raw = "\n".join(data_lines)
                    if raw != "[DONE]":
                        async for event in self._normalize_event(
                            json.loads(
                                raw,
                                object_pairs_hook=_unique_object,
                                parse_constant=_reject_constant,
                            ),
                            parser=parser,
                            output_text=output_text,
                            parsed_fields=parsed_fields,
                            response_schema=request.response_schema,
                            tool_calls_seen=tool_calls_seen,
                            response_id=response_id,
                            provider_sequence=sequence,
                        ):
                            if event.response_id:
                                response_id = event.response_id
                            if event.provider_sequence is not None:
                                sequence = event.provider_sequence
                            yield event
                            if isinstance(event, (ProviderResponseCompleted, ProviderResponseFailed)):
                                terminal = True
                if not terminal:
                    yield ProviderResponseFailed(
                        provider_sequence=sequence,
                        response_id=response_id,
                        error_code="incomplete_stream",
                        message="OpenAI Responses stream ended without a terminal event",
                    )
        except (httpx.HTTPError, json.JSONDecodeError, StructuredOutputError, ValueError) as exc:
            safe_message = (
                str(exc)[:300]
                if isinstance(exc, StructuredOutputError)
                else "OpenAI Responses streaming request failed"
            )
            yield ProviderResponseFailed(
                provider_event_id=None,
                provider_sequence=sequence,
                response_id=response_id,
                error_code=type(exc).__name__.lower(),
                message=safe_message,
                retryable=isinstance(exc, httpx.TimeoutException),
            )
        finally:
            if close_client:
                await client.aclose()

    async def _normalize_event(
        self,
        raw: object,
        *,
        parser: IncrementalJsonObjectParser | None,
        output_text: list[str],
        parsed_fields: list[tuple[str, JsonValue]],
        response_schema: JsonObject | None,
        tool_calls_seen: list[bool],
        response_id: str | None,
        provider_sequence: int | None,
    ) -> AsyncIterator[ProviderEvent]:
        if not isinstance(raw, Mapping):
            raise StructuredOutputError("Responses event must be an object")
        event_type = str(raw.get("type") or "")
        event_id = str(raw.get("event_id") or raw.get("id") or "") or None
        sequence_value = raw.get("sequence_number")
        sequence = int(sequence_value) if isinstance(sequence_value, int) else provider_sequence
        response = _json_object(raw.get("response"))
        resolved_response_id = str(response.get("id") or response_id or "") or None
        common = {
            "provider_event_id": event_id,
            "provider_sequence": sequence,
            "response_id": resolved_response_id,
        }
        if event_type == "response.created":
            yield ProviderResponseStarted(**common)
        elif event_type == "response.output_text.delta":
            delta = raw.get("delta")
            if isinstance(delta, str):
                output_text.append(delta)
                yield ProviderTextDelta(**common, delta=delta)
                if parser is not None:
                    parsed_fields.extend(parser.feed(delta))
        elif event_type in {
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
        }:
            summary = raw.get("text")
            if summary is None:
                part = _json_object(raw.get("part"))
                summary = part.get("text")
            if isinstance(summary, str):
                yield ProviderReasoningSummary(**common, text=summary)
        elif event_type == "response.output_item.done":
            item = _json_object(raw.get("item"))
            if item.get("type") == "function_call":
                args_value = item.get("arguments")
                try:
                    args = (
                        json.loads(
                            args_value,
                            object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant,
                        )
                        if isinstance(args_value, str)
                        else args_value
                    )
                except json.JSONDecodeError as exc:
                    raise StructuredOutputError("function call arguments are invalid JSON") from exc
                if not isinstance(args, Mapping):
                    raise StructuredOutputError("function call arguments must be an object")
                yield ProviderToolCallCompleted(
                    **common,
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=cast(JsonObject, dict(args)),
                )
                tool_calls_seen.append(True)
            elif item.get("type") in {"message", "reasoning"}:
                pass
            else:
                raise StructuredOutputError("provider-managed or unknown output item is unsupported")
        elif event_type == "response.completed":
            usage = _json_object(response.get("usage"))
            structured = parser.finish() if parser is not None else None
            if structured is not None and response_schema is not None:
                validate_json_schema(structured, response_schema)
                for field_name, value in parsed_fields:
                    yield ProviderFieldCompleted(
                        **common, field_name=field_name, value=value
                    )
            yield ProviderResponseCompleted(
                **common,
                text="".join(output_text),
                structured_output=structured,
                has_tool_calls=bool(tool_calls_seen),
                input_tokens=_optional_int(usage.get("input_tokens")),
                output_tokens=_optional_int(usage.get("output_tokens")),
            )
        elif event_type in {"response.failed", "response.incomplete", "error"}:
            error = _json_object(response.get("error") or raw.get("error"))
            yield ProviderResponseFailed(
                **common,
                error_code=str(error.get("code") or "provider_error"),
                message=str(error.get("message") or "provider response failed")[:500],
            )
        elif event_type not in _IGNORED_RESPONSE_EVENTS:
            raise StructuredOutputError(f"unsupported Responses event type: {event_type or 'missing'}")


_IGNORED_RESPONSE_EVENTS = frozenset(
    {
        "response.queued",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        "response.output_text.annotation.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_part.added",
        "response.reasoning_text.delta",
    }
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


__all__ = ["OpenAIResponsesProvider"]
