"""Dependency-light structured provider primitives.

This module intentionally contains no LangChain, parser, or application
dependencies. Higher-level packages provide vendor-specific construction.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol, TypeVar, runtime_checkable

TStructuredModel = TypeVar("TStructuredModel")


@runtime_checkable
class SupportsStructuredOutput(Protocol):
    def with_structured_output(
        self,
        schema: type[TStructuredModel],
        include_raw: bool = True,
        **kwargs: Any,
    ) -> Any: ...


class StructuredBridgeChatModel:
    """Authenticated client for a bounded host-side structured-output bridge."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        model: str,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        if not endpoint or not token:
            raise ValueError("structured bridge requires a non-empty endpoint and token")
        self.endpoint = endpoint.rstrip("/") + "/v1/structured"
        self.token = token
        self.model = model
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))

    def with_structured_output(
        self,
        schema: type[TStructuredModel],
        include_raw: bool = True,
        **kwargs: Any,
    ) -> "_StructuredBridgeResponse":
        _ = include_raw, kwargs
        return _StructuredBridgeResponse(self, schema)

    def complete(self, messages: Any, schema: Any) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": bridge_messages(messages),
            "response_schema": schema.model_json_schema(),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code not in {408, 429} and exc.code < 500:
                    raise ValueError(f"structured bridge rejected request with HTTP {exc.code}: {detail}") from exc
                if attempt >= self.max_retries:
                    raise TimeoutError(f"structured bridge retryable HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise TimeoutError(f"structured bridge unavailable: {exc}") from exc
        else:
            raise TimeoutError("structured bridge exhausted retries")
        if not isinstance(result, dict) or not isinstance(result.get("output"), dict):
            raise TypeError("structured bridge returned no structured output object")
        return result["output"]


class _StructuredBridgeResponse:
    def __init__(self, model: StructuredBridgeChatModel, schema: type[TStructuredModel]) -> None:
        self.model = model
        self.schema = schema

    def invoke(self, messages: Any, config: Any = None) -> dict[str, Any]:
        _ = config
        payload = self.model.complete(messages, self.schema)
        parsed = self.schema.model_validate(payload)
        return {"parsed": parsed, "raw": payload, "parsing_error": None}


def bridge_messages(messages: Any) -> list[dict[str, str]]:
    """Convert common chat-message objects without exposing local paths."""
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    result: list[dict[str, str]] = []
    for message in messages:
        role = str(getattr(message, "type", None) or getattr(message, "role", None) or "user")
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        content = getattr(message, "content", message)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        result.append({"role": role, "content": content})
    return result


class ProviderChainChatModel:
    """Try structured providers in order; fallback is limited to availability failures."""

    def __init__(self, models: list[tuple[str, SupportsStructuredOutput]]) -> None:
        if not models:
            raise ValueError("provider chain is empty")
        self.models = models

    def with_structured_output(
        self,
        schema: type[TStructuredModel],
        include_raw: bool = True,
        **kwargs: Any,
    ) -> "_ProviderChainResponse":
        _ = include_raw, kwargs
        return _ProviderChainResponse(self.models, schema)


class _ProviderChainResponse:
    def __init__(self, models: list[tuple[str, SupportsStructuredOutput]], schema: type[TStructuredModel]) -> None:
        self.models = models
        self.schema = schema

    def invoke(self, messages: Any, config: Any = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for index, (provider, model) in enumerate(self.models):
            try:
                result = model.with_structured_output(self.schema, include_raw=True).invoke(messages, config=config)
                if not isinstance(result, dict):
                    raise TypeError(f"{provider} returned an invalid structured result")
                result = dict(result)
                result["provider"] = provider
                if index:
                    result["fallback_from"] = self.models[0][0]
                    result["fallback_reason"] = str(last_error or "retryable provider failure")
                return result
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("provider chain is empty")


__all__ = [
    "ProviderChainChatModel",
    "StructuredBridgeChatModel",
    "SupportsStructuredOutput",
    "bridge_messages",
]
