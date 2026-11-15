"""Host-authorized dispatch for trusted, registered in-process event tools."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from kogwistar.engine_core.embedding_profile import NamedProjectionStore
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.runtime import ProjectionPayload
from kogwistar.runtime.runtime import StepContext

from .catalog import CatalogStore
from .providers import ProviderInactiveError, ProviderRegistry, ToolProvider
from .read_tools import ReadScope
from .schema_validation import SchemaValidationError, validate_json_object_schema, validate_json_schema
from .streaming import ProviderToolCallCompleted


def _json_value(value: object) -> JsonValue:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return cast(JsonValue, json.loads(encoded))


class ToolCallIdempotency(Protocol):
    """Host-owned durable deduplication boundary for tool side effects."""

    def execute_once(
        self, key: str, operation: Callable[[], JsonValue], /
    ) -> JsonValue: ...


class AmbiguousToolCallError(RuntimeError):
    """A prior tool effect may have happened and must not be replayed blindly."""


class NamedProjectionToolCallIdempotency:
    """Durably deduplicate tool calls using Kogwistar's metadata CAS primitive.

    The durable ``executing`` record is written before the side effect. If the
    process dies before recording the result, a replay fails closed instead of
    invoking the tool again. This provides at-most-once invocation, not atomic
    exactly-once effects; the downstream tool must still be idempotent when
    callers need retry/reconciliation after an ambiguous outcome.
    """

    def __init__(
        self,
        metadata: NamedProjectionStore,
        *,
        namespace: str = "agent_tool_call_idempotency_v1",
    ) -> None:
        self._metadata = metadata
        self._namespace = str(namespace).strip()
        if not self._namespace:
            raise ValueError("idempotency namespace must be non-empty")

    @staticmethod
    def _sequence(row: ProjectionPayload, key: str) -> int:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AmbiguousToolCallError("tool idempotency record has invalid sequence metadata")
        return value

    def _write(
        self,
        key: str,
        payload: JsonObject,
        *,
        expected: ProjectionPayload | None,
        status: str,
    ) -> bool:
        expected_authoritative = (
            self._sequence(expected, "last_authoritative_seq") if expected is not None else None
        )
        expected_materialized = (
            self._sequence(expected, "last_materialized_seq") if expected is not None else None
        )
        next_authoritative = (expected_authoritative or 0) + 1
        next_materialized = (expected_materialized or 0) + 1
        return self._metadata.compare_and_swap_named_projection(
            self._namespace,
            key,
            cast(ProjectionPayload, payload),
            expected_last_authoritative_seq=expected_authoritative,
            expected_last_materialized_seq=expected_materialized,
            last_authoritative_seq=next_authoritative,
            last_materialized_seq=next_materialized,
            projection_schema_version=1,
            materialization_status=status,
        )

    def execute_once(self, key: str, operation: Callable[[], JsonValue], /) -> JsonValue:
        identity = str(key).strip()
        if not identity:
            raise ValueError("idempotency key must be non-empty")
        existing = self._metadata.get_named_projection(self._namespace, identity)
        if existing is not None:
            payload = existing.get("payload")
            if isinstance(payload, Mapping) and payload.get("status") == "completed" and "result" in payload:
                return cast(JsonValue, payload["result"])
            raise AmbiguousToolCallError(
                "tool call has an unfinished durable record; refusing unsafe replay"
            )

        if not self._write(
            identity,
            {"status": "executing"},
            expected=None,
            status="executing",
        ):
            # A competing caller won the create CAS. Re-read once and return a
            # completed result if available; otherwise fail closed.
            raced = self._metadata.get_named_projection(self._namespace, identity)
            raced_payload = raced.get("payload") if raced is not None else None
            if (
                isinstance(raced_payload, Mapping)
                and raced_payload.get("status") == "completed"
                and "result" in raced_payload
            ):
                return cast(JsonValue, raced_payload["result"])
            raise AmbiguousToolCallError(
                "tool call is already executing or has an ambiguous prior outcome"
            )

        result = operation()
        durable = self._metadata.get_named_projection(self._namespace, identity)
        if durable is None or not self._write(
            identity,
            {"status": "completed", "result": result},
            expected=durable,
            status="completed",
        ):
            raise AmbiguousToolCallError(
                "tool effect completed but its result could not be durably recorded"
            )
        return result


class CatalogAuthorizedToolDispatcher:
    """Resolve and authorize provider tool calls before invoking trusted code.

    This adapter is deliberately limited to trusted in-process ``ToolProvider``
    implementations. MCP or other untrusted executables require a separately
    isolated host dispatcher. Tool names and capability requirements are
    supplied by host configuration; catalog manifests cannot grant authority.
    """

    def __init__(
        self,
        *,
        catalog: CatalogStore,
        providers: ProviderRegistry,
        scope: ReadScope,
        tool_name_to_logical_id: Mapping[str, str],
        capability_requirements: Mapping[str, frozenset[str]],
        argument_schemas: Mapping[str, JsonObject],
        idempotency: ToolCallIdempotency,
    ) -> None:
        self._catalog = catalog
        self._providers = providers
        self._scope = scope
        self._tool_name_to_logical_id = {
            str(name): str(logical_id)
            for name, logical_id in tool_name_to_logical_id.items()
        }
        self._capability_requirements = {
            str(logical_id): frozenset(str(cap) for cap in capabilities)
            for logical_id, capabilities in capability_requirements.items()
        }
        self._argument_schemas = {
            str(logical_id): copy.deepcopy(schema)
            for logical_id, schema in argument_schemas.items()
        }
        self._idempotency = idempotency
        if not catalog.acl_enabled or catalog.acl_checker is None:
            raise PermissionError("event tool dispatch requires an ACL-enabled catalog")
        if any(not name.strip() or not logical_id.strip() for name, logical_id in self._tool_name_to_logical_id.items()):
            raise ValueError("tool names and logical IDs must be non-empty")
        configured_tools = set(self._tool_name_to_logical_id.values())
        if set(self._argument_schemas) != configured_tools:
            raise ValueError("every configured tool requires exactly one host argument schema")
        for schema in self._argument_schemas.values():
            validate_json_object_schema(schema)

    def _dispatch_sync(
        self,
        context: StepContext,
        call: ProviderToolCallCompleted,
        *,
        idempotency_key: str,
    ) -> JsonObject:
        """Perform the final scoped authorization and idempotent invocation."""

        logical_id = self._tool_name_to_logical_id.get(call.name)
        if logical_id is None:
            raise PermissionError("tool name is not in the host execution allowlist")

        entry = self._catalog.get(
            logical_id,
            principal=self._scope.principal_id,
            scope=self._scope.security_scope,
            tenant_id=self._scope.tenant_id,
            project_id=self._scope.project_id,
        )
        if entry is None or entry.kind != "tool":
            raise PermissionError("tool is not visible in the caller's catalog scope")

        required = self._capability_requirements.get(logical_id)
        if required is None:
            raise PermissionError("tool has no host authorization policy")
        authority = context.authority_context
        raw_capabilities = authority.get("effective_capabilities", ())
        if not isinstance(raw_capabilities, (list, tuple, set, frozenset)):
            raise PermissionError("trusted effective capabilities are unavailable")
        effective = {str(item) for item in raw_capabilities}
        if not required.issubset(effective):
            raise PermissionError("tool capability requirements are not satisfied")

        schema = self._argument_schemas.get(logical_id)
        if schema is None:
            raise PermissionError("tool has no host argument schema policy")
        try:
            validate_json_schema(call.arguments, schema)
        except SchemaValidationError as exc:
            raise PermissionError("tool arguments do not satisfy host schema") from exc

        try:
            registration = self._providers.get(
                entry.provider_id, entry.provider_version
            )
        except KeyError as exc:
            raise PermissionError("catalog tool provider is not active") from exc
        provider = registration.provider
        if not isinstance(provider, ToolProvider) or provider.provider_id != entry.provider_id:
            raise PermissionError("catalog tool provider is not an invocable tool provider")

        def invoke() -> JsonValue:
            result = provider.invoke(call.arguments)
            return _json_value(result)

        try:
            result = self._idempotency.execute_once(
                idempotency_key,
                lambda: self._providers.run_if_active(registration, invoke),
            )
        except ProviderInactiveError as exc:
            raise PermissionError("tool provider is no longer active") from exc
        return {"result": _json_value(result)}

    async def __call__(
        self,
        context: StepContext,
        call: ProviderToolCallCompleted,
        *,
        idempotency_key: str,
    ) -> JsonObject:
        """Keep synchronous providers off the async workflow event loop."""

        return await asyncio.to_thread(
            self._dispatch_sync,
            context,
            call,
            idempotency_key=idempotency_key,
        )


__all__ = [
    "AmbiguousToolCallError",
    "CatalogAuthorizedToolDispatcher",
    "NamedProjectionToolCallIdempotency",
    "ToolCallIdempotency",
]
