from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.agent import (
    IncrementalJsonObjectParser,
    ModelProvider,
    OpenAIChatCompletionsProvider,
    OpenAIResponsesProvider,
    ProviderCapabilities,
    ProviderEvent,
    ProviderFieldCompleted,
    ProviderRequest,
    ProviderResponseCompleted,
    ProviderResponseFailed,
    ProviderResponseStarted,
    ProviderTextDelta,
    SequenceFakeModel,
    StreamingModelProvider,
    StructuredOutputError,
    build_event_workflow,
    validate_agent_design,
)

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class _ChunkedHttpBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


def _chat_sse(*chunks: dict[str, object], include_done: bool = True) -> bytes:
    body = b"".join(
        b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\r\n\r\n"
        for chunk in chunks
    )
    return body + (b"data: [DONE]\r\n\r\n" if include_done else b"")


def _chat_provider_for_body(body: bytes, *, api_key: str = "test-key") -> OpenAIChatCompletionsProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {api_key}"
        return httpx.Response(
            200,
            stream=_ChunkedHttpBody([body]),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChatCompletionsProvider(api_key=api_key, client=client)


def _collect_chat(
    provider: OpenAIChatCompletionsProvider,
    request: ProviderRequest,
) -> list[ProviderEvent]:
    async def collect() -> list[ProviderEvent]:
        try:
            return [event async for event in provider.stream(request)]
        finally:
            if provider._client is not None:
                await provider._client.aclose()

    return asyncio.run(collect())


def _collect_responses(
    body: str, request: ProviderRequest
) -> list[ProviderEvent]:
    async def collect() -> list[ProviderEvent]:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, text=body, headers={"content-type": "text/event-stream"}
                )
            )
        )
        provider = OpenAIResponsesProvider(api_key="test-key", client=client)
        try:
            return [event async for event in provider.stream(request)]
        finally:
            await client.aclose()

    return asyncio.run(collect())


def test_streaming_contract_preserves_existing_sync_model_provider_api() -> None:
    legacy = SequenceFakeModel([{"answer": "sync"}])
    assert isinstance(legacy, ModelProvider)
    assert legacy.complete("prompt", {}) == {"answer": "sync"}
    assert legacy.prompts == ["prompt"]

    class StreamOnlyProvider:
        provider_id = "test.stream-only"
        capabilities = ProviderCapabilities()

        async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponseCompleted]:
            yield ProviderResponseCompleted(response_id="response", text=request.input)

    stream_provider = StreamOnlyProvider()
    assert isinstance(stream_provider, StreamingModelProvider)
    assert not isinstance(stream_provider, ModelProvider)


def test_incremental_json_parser_handles_split_nested_escaped_and_unicode_values() -> None:
    parser = IncrementalJsonObjectParser()
    fields = []
    for chunk in ['{"title":"a \\', '"quote\\"", "nested":{"items":[1,', '2]}, "unicode":"雪"}']:
        fields.extend(parser.feed(chunk))
    assert fields == [
        ("title", 'a "quote"'),
        ("nested", {"items": [1, 2]}),
        ("unicode", "雪"),
    ]
    assert parser.finish() == {
        "title": 'a "quote"',
        "nested": {"items": [1, 2]},
        "unicode": "雪",
    }


def test_incremental_json_parser_accepts_every_two_chunk_boundary() -> None:
    source = '{"status":"ok","nested":{"values":[1,"雪"]},"value":7}'
    for split in range(len(source) + 1):
        parser = IncrementalJsonObjectParser()
        parser.feed(source[:split])
        parser.feed(source[split:])
        assert parser.finish() == {
            "status": "ok",
            "nested": {"values": [1, "雪"]},
            "value": 7,
        }


@pytest.mark.parametrize(
    "chunks,match",
    [
        (["{\"a\":1,\"a\":2}"], "duplicate JSON key"),
        (["{\"a\":{\"x\":1,\"x\":2}}"], "duplicate JSON key"),
        (["[1,2]"], "JSON object"),
        (["{\"a\":"], "incomplete or invalid"),
        (["{\"a\":1} extra"], "trailing content"),
    ],
)
def test_incremental_json_parser_rejects_ambiguous_or_invalid_input(
    chunks: list[str], match: str
) -> None:
    parser = IncrementalJsonObjectParser()
    with pytest.raises(StructuredOutputError, match=match):
        for chunk in chunks:
            parser.feed(chunk)
        parser.finish()


def test_incremental_json_parser_enforces_character_limit() -> None:
    parser = IncrementalJsonObjectParser(max_chars=8)
    with pytest.raises(StructuredOutputError, match="configured limit"):
        parser.feed('{"value":')


def test_event_workflow_is_ordinary_predicate_routed_graph() -> None:
    design = build_event_workflow()
    assert validate_agent_design(design, mode="event") == ()
    assert any(node.op == "agent.event.receive" for node in design.nodes)
    predicates = [edge.metadata.get("wf_predicate") for edge in design.edges]
    assert "event_is_response_complete" in predicates
    assert "event_is_response_failed" in predicates
    assert "event_is_tool_call" in predicates
    assert "event_run_succeeded" in predicates
    assert any(node.op == "agent.event.ack" for node in design.nodes)
    assert any(node.op == "agent.event.control" for node in design.nodes)
    assert any(node.op == "agent.event.control_ack" for node in design.nodes)
    assert any(
        edge.metadata.get("wf_predicate") == "event_turn_requires_model_resume"
        and str(edge.target_ids[0]).endswith(":prepare")
        for edge in design.edges
    )
    assert all(node.metadata.get("wf_cacheable") is False for node in design.nodes)


def test_openai_responses_adapter_normalizes_stream_and_structured_fields() -> None:
    wire_events = [
        {"type": "response.created", "sequence_number": 1, "response": {"id": "resp_1"}},
        {"type": "response.output_text.delta", "sequence_number": 2, "delta": '{"answer":"ok",'},
        {"type": "response.output_text.delta", "sequence_number": 3, "delta": '"count":2}'},
        {"type": "response.completed", "sequence_number": 4, "response": {"id": "resp_1", "usage": {"input_tokens": 4, "output_tokens": 3}}},
    ]
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in wire_events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def collect() -> list[ProviderEvent]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIResponsesProvider(api_key="test-key", client=client)
        events = [event async for event in provider.stream(ProviderRequest(
            model="test-model",
            input="return a small object",
            response_schema={"type": "object", "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}}},
        ))]
        await client.aclose()
        return events

    events = asyncio.run(collect())
    assert isinstance(events[0], ProviderResponseStarted)
    assert sum(isinstance(event, ProviderTextDelta) for event in events) == 2
    assert [(event.field_name, event.value) for event in events if isinstance(event, ProviderFieldCompleted)] == [
        ("answer", "ok"),
        ("count", 2),
    ]
    completed = next(event for event in events if isinstance(event, ProviderResponseCompleted))
    assert completed.response_id == "resp_1"
    assert completed.structured_output == {"answer": "ok", "count": 2}
    assert completed.input_tokens == 4 and completed.output_tokens == 3


def test_openai_responses_adapter_reports_missing_credentials_without_network() -> None:
    async def collect() -> list[ProviderEvent]:
        provider = OpenAIResponsesProvider(api_key="")
        return [event async for event in provider.stream(ProviderRequest(model="m", input="x"))]

    events = asyncio.run(collect())
    assert len(events) == 1
    assert events[0].error_code == "missing_api_key"


@pytest.mark.parametrize(
    "schema,output",
    [
        ({"type": "object", "required": ["answer"]}, {"other": "x"}),
        ({"type": "object", "properties": {"count": {"type": "integer"}}}, {"count": "one"}),
        ({"type": "object", "properties": {"status": {"enum": ["ok"]}}}, {"status": "bad"}),
        (
            {"type": "object", "properties": {"answer": {"type": "string"}}, "additionalProperties": False},
            {"answer": "ok", "extra": True},
        ),
    ],
)
@pytest.mark.parametrize("adapter", ["responses", "chat"])
def test_openai_adapters_reject_schema_invalid_output_before_field_events(
    adapter: str, schema: JsonObject, output: dict[str, JsonValue]
) -> None:
    encoded = json.dumps(output)
    if adapter == "responses":
        wire = [
            {"type": "response.output_text.delta", "sequence_number": 1, "delta": encoded},
            {"type": "response.completed", "sequence_number": 2, "response": {"id": "bad-schema"}},
        ]
        events = _collect_responses(
            "".join(f"data: {json.dumps(item)}\n\n" for item in wire),
            ProviderRequest(model="m", input="x", response_schema=schema),
        )
    else:
        body = _chat_sse(
            {"id": "bad-schema", "choices": [{"index": 0, "delta": {"content": encoded}, "finish_reason": None}]},
            {"id": "bad-schema", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        events = _collect_chat(
            _chat_provider_for_body(body),
            ProviderRequest(model="m", input="x", response_schema=schema),
        )

    assert isinstance(events[-1], ProviderResponseFailed)
    assert not any(isinstance(event, ProviderFieldCompleted) for event in events)
    assert not any(isinstance(event, ProviderResponseCompleted) for event in events)


@pytest.mark.parametrize("adapter", ["responses", "chat"])
def test_openai_adapters_reject_invalid_host_schema(adapter: str) -> None:
    if adapter == "responses":
        wire = [
            {"type": "response.output_text.delta", "sequence_number": 1, "delta": "{}"},
            {"type": "response.completed", "sequence_number": 2, "response": {"id": "bad-schema"}},
        ]
        events = _collect_responses(
            "".join(f"data: {json.dumps(item)}\n\n" for item in wire),
            ProviderRequest(model="m", input="x", response_schema={"type": "not-a-json-schema-type"}),
        )
    else:
        body = _chat_sse(
            {"id": "bad-schema", "choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": None}]},
            {"id": "bad-schema", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        events = _collect_chat(
            _chat_provider_for_body(body),
            ProviderRequest(model="m", input="x", response_schema={"type": "not-a-json-schema-type"}),
        )

    assert isinstance(events[-1], ProviderResponseFailed)
    assert not any(isinstance(event, ProviderFieldCompleted) for event in events)
    assert not any(isinstance(event, ProviderResponseCompleted) for event in events)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.future_event", "sequence_number": 1},
        {"type": "response.output_item.done", "sequence_number": 1, "item": {"type": "computer_call"}},
    ],
)
def test_responses_rejects_unknown_events_and_provider_managed_tools(event: dict[str, object]) -> None:
    events = _collect_responses(
        f"data: {json.dumps(event)}\n\n",
        ProviderRequest(model="m", input="x"),
    )
    assert isinstance(events[-1], ProviderResponseFailed)
    assert not any(isinstance(item, ProviderResponseCompleted) for item in events)
    assert not any(item.kind == "tool_call_completed" for item in events)


def test_chat_completions_rejects_unknown_delta_fields() -> None:
    body = _chat_sse(
        {"id": "unknown-delta", "choices": [{"index": 0, "delta": {"audio": "opaque"}, "finish_reason": None}]},
    )
    events = _collect_chat(
        _chat_provider_for_body(body), ProviderRequest(model="m", input="x")
    )
    assert isinstance(events[-1], ProviderResponseFailed)
    assert not any(isinstance(event, ProviderResponseCompleted) for event in events)


def test_chat_completions_rejects_provider_managed_tool_types() -> None:
    body = _chat_sse(
        {"id": "managed-tool", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call-1", "type": "computer", "function": {"name": "search", "arguments": "{}"}},
        ]}, "finish_reason": None}]},
        {"id": "managed-tool", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    )
    events = _collect_chat(
        _chat_provider_for_body(body), ProviderRequest(model="m", input="x")
    )
    assert isinstance(events[-1], ProviderResponseFailed)
    assert not any(event.kind == "tool_call_completed" for event in events)
    assert not any(isinstance(event, ProviderResponseCompleted) for event in events)


def test_openai_response_completion_marks_tool_call_turn_for_safe_resume() -> None:
    wire_events = [
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{}",
            },
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": {"id": "resp-tool", "usage": {}},
        },
    ]
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in wire_events)

    async def collect() -> list[ProviderEvent]:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
        )
        provider = OpenAIResponsesProvider(api_key="test-key", client=client)
        events = [event async for event in provider.stream(ProviderRequest(model="test", input="call a tool"))]
        await client.aclose()
        return events

    events = asyncio.run(collect())
    assert [event.kind for event in events] == ["tool_call_completed", "response_completed"]
    assert isinstance(events[-1], ProviderResponseCompleted)
    assert events[-1].has_tool_calls is True


def test_chat_completions_builds_strict_schema_request_and_normalizes_split_sse() -> None:
    wire = _chat_sse(
        {"id": "chat-1", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"id": "chat-1", "choices": [{"index": 0, "delta": {"content": '{"answer":"雪\\n'}, "finish_reason": None}]},
        {"id": "chat-1", "choices": [{"index": 0, "delta": {"content": 'line","nested":{"ok":true}}'}, "finish_reason": None}]},
        {"id": "chat-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "chat-1", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 5}},
    )
    split = wire.index("雪".encode()) + 1

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["n"] == 1
        assert body["stream_options"] == {"include_usage": True}
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["max_completion_tokens"] == 80
        return httpx.Response(
            200,
            stream=_ChunkedHttpBody([wire[:split], wire[split:]]),
            headers={"content-type": "text/event-stream"},
        )

    async def collect() -> list[ProviderEvent]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIChatCompletionsProvider(api_key="test-key", client=client)
        try:
            return [event async for event in provider.stream(ProviderRequest(
                model="test-model",
                input="return JSON",
                instructions="be concise",
                response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
                max_output_tokens=80,
            ))]
        finally:
            await client.aclose()

    events = asyncio.run(collect())
    assert events[0].kind == "response_started"
    assert [event.field_name for event in events if isinstance(event, ProviderFieldCompleted)] == ["answer", "nested"]
    completed = next(event for event in events if isinstance(event, ProviderResponseCompleted))
    assert completed.response_id == "chat-1"
    assert completed.structured_output == {"answer": "雪\nline", "nested": {"ok": True}}
    assert (completed.input_tokens, completed.output_tokens) == (9, 5)
    assert [event.provider_sequence for event in events] == list(range(1, len(events) + 1))


def test_chat_completions_does_not_emit_fragmented_tool_calls_before_done() -> None:
    chunks = [
        {"id": "chat-tools", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call-", "type": "function", "function": {"name": "look", "arguments": '{"q"'}},
            {"index": 1, "id": "call-2", "type": "function", "function": {"name": "write", "arguments": '{"v":'}},
        ]}, "finish_reason": None}]},
        {"id": "chat-tools", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "1", "function": {"name": "up", "arguments": ':"x"}'}},
            {"index": 1, "function": {"arguments": "2}"}},
        ]}, "finish_reason": None}]},
        {"id": "chat-tools", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = _collect_chat(
        _chat_provider_for_body(_chat_sse(*chunks)),
        ProviderRequest(
            model="test",
            input="use tools",
            tools=({"type": "function"},),
            response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        ),
    )
    calls = [event for event in events if event.kind == "tool_call_completed"]
    assert [(call.call_id, call.name, call.arguments) for call in calls] == [
        ("call-1", "lookup", {"q": "x"}),
        ("call-2", "write", {"v": 2}),
    ]
    first_sequence = calls[0].provider_sequence
    second_sequence = calls[1].provider_sequence
    assert isinstance(first_sequence, int) and isinstance(second_sequence, int)
    assert first_sequence < second_sequence
    assert events[-1].kind == "response_completed"
    assert isinstance(events[-1], ProviderResponseCompleted)
    assert events[-1].has_tool_calls is True
    assert events[-1].structured_output is None


@pytest.mark.parametrize(
    "body,expected_code",
    [
        (_chat_sse({"error": {"message": "bad"}}), "provider_error"),
        (_chat_sse({"id": "r", "choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": None}]}, include_done=False), "incomplete_stream"),
        (_chat_sse({"id": "r", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}), "length"),
        (b"data: {not-json}\n\n", "jsondecodeerror"),
    ],
)
def test_chat_completions_fails_closed_for_provider_and_stream_errors(body: bytes, expected_code: str) -> None:
    events = _collect_chat(_chat_provider_for_body(body), ProviderRequest(model="m", input="x"))
    assert isinstance(events[-1], ProviderResponseFailed)
    assert events[-1].error_code == expected_code


def test_chat_completions_missing_credentials_fails_before_transport() -> None:
    async def collect() -> list[ProviderEvent]:
        provider = OpenAIChatCompletionsProvider(api_key="")
        return [event async for event in provider.stream(ProviderRequest(model="m", input="x"))]

    events = asyncio.run(collect())
    assert len(events) == 1
    assert isinstance(events[0], ProviderResponseFailed)
    assert events[0].error_code == "missing_api_key"


def test_chat_completions_timeout_is_retryable_and_response_identity_is_stable() -> None:
    def timeout_handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private transport details")

    async def collect_timeout() -> list[ProviderEvent]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
        provider = OpenAIChatCompletionsProvider(api_key="test-key", client=client)
        try:
            return [event async for event in provider.stream(ProviderRequest(model="m", input="x"))]
        finally:
            await client.aclose()

    timeout_events = asyncio.run(collect_timeout())
    assert isinstance(timeout_events[-1], ProviderResponseFailed)
    assert timeout_events[-1].retryable is True
    assert "private transport details" not in timeout_events[-1].message

    body = _chat_sse(
        {"id": "first", "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}]},
        {"id": "second", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    events = _collect_chat(_chat_provider_for_body(body), ProviderRequest(model="m", input="x"))
    assert isinstance(events[-1], ProviderResponseFailed)
    assert events[-1].error_code == "structuredoutputerror"
    assert not any(event.kind == "response_completed" for event in events)


def test_chat_completions_rejects_oversized_text_and_missing_finish_reason() -> None:
    body = _chat_sse(
        {"id": "r", "choices": [{"index": 0, "delta": {"content": "123456"}, "finish_reason": None}]},
        {"id": "r", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    provider = _chat_provider_for_body(body)
    provider._max_response_chars = 5
    events = _collect_chat(provider, ProviderRequest(model="m", input="x"))
    assert isinstance(events[-1], ProviderResponseFailed)
    assert "response text" in events[-1].message

    missing_finish = _chat_sse(
        {"id": "r", "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
    )
    events = _collect_chat(
        _chat_provider_for_body(missing_finish),
        ProviderRequest(model="m", input="x"),
    )
    assert isinstance(events[-1], ProviderResponseFailed)
    assert events[-1].error_code == "incomplete_stream"


def test_chat_completions_fails_closed_on_incomplete_structured_json() -> None:
    body = _chat_sse(
        {"id": "r", "choices": [{"index": 0, "delta": {"content": '{"answer":'}, "finish_reason": None}]},
        {"id": "r", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    events = _collect_chat(
        _chat_provider_for_body(body),
        ProviderRequest(
            model="m",
            input="return JSON",
            response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        ),
    )
    assert isinstance(events[-1], ProviderResponseFailed)
    assert not any(event.kind == "response_completed" for event in events)
    assert not any(event.kind == "field_completed" for event in events)


def test_chat_completions_bounds_sse_event_size_and_rejects_duplicate_tool_keys() -> None:
    oversized = b"data: " + (b"x" * 64) + b"\n\n"
    provider = _chat_provider_for_body(oversized)
    provider._max_sse_event_chars = 32
    events = _collect_chat(provider, ProviderRequest(model="m", input="x"))
    assert isinstance(events[-1], ProviderResponseFailed)
    assert events[-1].error_code == "structuredoutputerror"

    long_line_provider = _chat_provider_for_body(b"data:" + b"x" * 100)
    long_line_provider._max_sse_line_bytes = 16
    events = _collect_chat(long_line_provider, ProviderRequest(model="m", input="x"))
    assert isinstance(events[-1], ProviderResponseFailed)
    assert events[-1].message == "SSE line exceeds configured limit"

    body = _chat_sse(
        {"id": "r", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c", "function": {"name": "tool", "arguments": '{"x":1,"x":2}'}}
        ]}, "finish_reason": None}]},
        {"id": "r", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    )
    events = _collect_chat(_chat_provider_for_body(body), ProviderRequest(model="m", input="x"))
    assert not any(event.kind == "tool_call_completed" for event in events)
    assert isinstance(events[-1], ProviderResponseFailed)
