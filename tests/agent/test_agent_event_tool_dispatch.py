from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from kogwistar.agent import (
    AmbiguousToolCallError,
    CatalogAuthorizedToolDispatcher,
    CatalogEntry,
    CatalogStore,
    NamedProjectionToolCallIdempotency,
    ProviderRegistry,
    ProviderToolCallCompleted,
    ReadScope,
    ToolCallIdempotency,
)
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.json_types import JsonValue
from kogwistar.runtime.runtime import StepContext

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class LookupTool:
    provider_id = "trusted.lookup"

    def __init__(self) -> None:
        self.calls: list[dict[str, JsonValue]] = []

    def invoke(self, arguments: Mapping[str, JsonValue]) -> JsonValue:
        self.calls.append(dict(arguments))
        return {"matches": ["alpha"]}


class IdempotencyStore:
    def __init__(self) -> None:
        self.results: dict[str, JsonValue] = {}

    def execute_once(
        self, key: str, operation: Callable[[], JsonValue], /
    ) -> JsonValue:
        if key not in self.results:
            self.results[key] = operation()
        return self.results[key]


def _context(*, capabilities: tuple[str, ...]) -> StepContext:
    return StepContext(
        run_id="run-1",
        workflow_id="agent.event.v1",
        workflow_node_id="wf:agent.event.v1:tool",
        op="agent.event.tool",
        token_id="token-1",
        attempt=1,
        step_seq=4,
        cache_dir=None,
        conversation_id="conversation-1",
        authority_context={"effective_capabilities": capabilities},
        state={},
    )


def _entry() -> CatalogEntry:
    return CatalogEntry(
        logical_id="trusted.lookup:search",
        provider_id="trusted.lookup",
        provider_local_id="search",
        kind="tool",
        name="search",
        source_fingerprint="fixture",
        # Descriptors are not authority. The dispatcher uses host policy.
        required_capabilities=[],
        tenant_id="tenant-a",
        project_id="project-a",
        scope="private",
    )


def _dispatcher(
    *,
    acl: Callable[[CatalogEntry, str], bool] | None = None,
    tenant_id: str = "tenant-a",
    acl_enabled: bool = True,
    idempotency: ToolCallIdempotency | None = None,
    argument_schemas: Mapping[str, dict[str, JsonValue]] | None = None,
) -> tuple[CatalogAuthorizedToolDispatcher, LookupTool, ToolCallIdempotency, ProviderRegistry]:
    tool = LookupTool()
    providers = ProviderRegistry()
    providers.register(provider_id=tool.provider_id, provider=tool)
    catalog = CatalogStore(
        acl_enabled=acl_enabled,
        acl_checker=acl or (lambda _entry, principal: principal == "alice"),
    )
    catalog.upsert(_entry())
    selected_idempotency = idempotency or IdempotencyStore()
    dispatcher = CatalogAuthorizedToolDispatcher(
        catalog=catalog,
        providers=providers,
        scope=ReadScope(
            principal_id="alice",
            security_scope="private",
            tenant_id=tenant_id,
            project_id="project-a",
        ),
        tool_name_to_logical_id={"search": "trusted.lookup:search"},
        capability_requirements={"trusted.lookup:search": frozenset({"knowledge.read"})},
        argument_schemas=(
            {"trusted.lookup:search": {"type": "object"}}
            if argument_schemas is None
            else argument_schemas
        ),
        idempotency=selected_idempotency,
    )
    return dispatcher, tool, selected_idempotency, providers


def test_event_dispatch_requires_host_capability_not_manifest_claim() -> None:
    dispatcher, tool, _, _ = _dispatcher()
    call = ProviderToolCallCompleted(
        response_id="response-1",
        call_id="call-1",
        name="search",
        arguments={"query": "alpha"},
    )

    try:
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.write",)),
            call,
            idempotency_key="tool-key-1",
        )
    except PermissionError as exc:
        assert "capability" in str(exc)
    else:
        raise AssertionError("tool executed without host-granted knowledge.read")
    assert tool.calls == []


def test_event_dispatch_validates_arguments_against_host_owned_schema() -> None:
    dispatcher, tool, _, _ = _dispatcher(
        argument_schemas={
            "trusted.lookup:search": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            }
        }
    )
    call = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-invalid", name="search", arguments={"query": 7}
    )

    with pytest.raises(PermissionError, match="host schema"):
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)),
            call,
            idempotency_key="tool-invalid-args",
        )
    assert tool.calls == []


def test_event_dispatch_requires_schema_for_every_host_tool() -> None:
    with pytest.raises(ValueError, match="host argument schema"):
        _dispatcher(argument_schemas={})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": "ok", "unexpected": True},
        {"query": 42},
    ],
)
def test_event_dispatch_rejects_missing_extra_or_wrong_type_arguments(
    arguments: dict[str, JsonValue],
) -> None:
    dispatcher, tool, _, _ = _dispatcher(
        argument_schemas={
            "trusted.lookup:search": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            }
        }
    )
    call = ProviderToolCallCompleted(
        response_id="response-1", call_id="bad-args", name="search", arguments=arguments
    )

    with pytest.raises(PermissionError, match="host schema"):
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)), call, idempotency_key="bad-args"
        )
    assert tool.calls == []


def test_event_dispatch_ignores_capabilities_forged_in_workflow_state() -> None:
    dispatcher, tool, _, _ = _dispatcher()
    context = _context(capabilities=())
    context._state["effective_capabilities"] = ["knowledge.read"]
    call = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-1", name="search", arguments={}
    )

    try:
        dispatcher._dispatch_sync(context, call, idempotency_key="tool-key-1")
    except PermissionError as exc:
        assert "capability" in str(exc)
    else:
        raise AssertionError("workflow state granted authority to the tool call")
    assert tool.calls == []


def test_event_dispatch_checks_catalog_acl_and_scope_before_invocation() -> None:
    dispatcher, tool, _, _ = _dispatcher(acl=lambda _entry, _principal: False)
    call = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-1", name="search", arguments={}
    )

    try:
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)),
            call,
            idempotency_key="tool-key-1",
        )
    except PermissionError as exc:
        assert "visible" in str(exc)
    else:
        raise AssertionError("ACL-hidden catalog tool was invoked")
    assert tool.calls == []


def test_event_dispatch_reuses_durable_idempotency_result_without_repeating_effect(
    tmp_path,
) -> None:
    metadata = EngineSQLite(tmp_path / "dispatcher-meta")
    metadata.ensure_initialized()
    dispatcher, tool, _, _ = _dispatcher(
        idempotency=NamedProjectionToolCallIdempotency(metadata)
    )
    call = ProviderToolCallCompleted(
        response_id="response-1",
        call_id="call-1",
        name="search",
        arguments={"query": "alpha"},
    )
    context = _context(capabilities=("knowledge.read",))

    first = dispatcher._dispatch_sync(context, call, idempotency_key="tool-key-1")
    replay = dispatcher._dispatch_sync(context, call, idempotency_key="tool-key-1")

    assert first == replay == {"result": {"matches": ["alpha"]}}
    assert tool.calls == [{"query": "alpha"}]


def test_event_dispatch_rejects_unmapped_and_non_json_provider_results() -> None:
    dispatcher, tool, _, _ = _dispatcher()
    unknown = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-1", name="admin", arguments={}
    )
    try:
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)),
            unknown,
            idempotency_key="tool-key-unknown",
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("unmapped provider tool name was executed")

    tool.invoke = lambda _: object()  # type: ignore[method-assign,return-value]
    known = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-2", name="search", arguments={}
    )
    try:
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)),
            known,
            idempotency_key="tool-key-non-json",
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("non-JSON provider output escaped validation")


def test_event_dispatch_rejects_scope_mismatch_and_inactive_provider() -> None:
    dispatcher, tool, _, providers = _dispatcher(tenant_id="tenant-b")
    call = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-1", name="search", arguments={}
    )
    try:
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)),
            call,
            idempotency_key="tool-key-1",
        )
    except PermissionError as exc:
        assert "visible" in str(exc)
    else:
        raise AssertionError("cross-tenant tool entry was invoked")

    dispatcher, tool, _, providers = _dispatcher()
    providers.unload(tool.provider_id)
    try:
        dispatcher._dispatch_sync(
            _context(capabilities=("knowledge.read",)),
            call,
            idempotency_key="tool-key-2",
        )
    except PermissionError as exc:
        assert "active" in str(exc)
    else:
        raise AssertionError("retired provider was invoked")
    assert tool.calls == []


def test_event_dispatch_requires_acl_enabled_catalog() -> None:
    try:
        _dispatcher(acl_enabled=False)
    except PermissionError as exc:
        assert "ACL-enabled" in str(exc)
    else:
        raise AssertionError("dispatcher accepted a catalog without ACL enforcement")


def test_named_projection_idempotency_returns_durable_result_after_reopen(tmp_path) -> None:
    metadata = EngineSQLite(tmp_path / "metadata")
    metadata.ensure_initialized()
    first_executor = NamedProjectionToolCallIdempotency(metadata)
    calls: list[str] = []

    result = first_executor.execute_once(
        "stable-key", lambda: calls.append("invoked") or {"ok": True}
    )
    reopened_metadata = EngineSQLite(tmp_path / "metadata")
    reopened_metadata.ensure_initialized()
    reopened_executor = NamedProjectionToolCallIdempotency(reopened_metadata)
    replay = reopened_executor.execute_once(
        "stable-key", lambda: (_ for _ in ()).throw(AssertionError("must not replay"))
    )

    assert result == replay == {"ok": True}
    assert calls == ["invoked"]


def test_named_projection_idempotency_fails_closed_after_ambiguous_effect(tmp_path) -> None:
    metadata = EngineSQLite(tmp_path / "metadata")
    metadata.ensure_initialized()
    executor = NamedProjectionToolCallIdempotency(metadata)
    calls: list[str] = []

    def fail_after_effect() -> JsonValue:
        calls.append("effect-may-have-happened")
        raise RuntimeError("simulated crash after side effect")

    try:
        executor.execute_once("ambiguous-key", fail_after_effect)
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)
    else:
        raise AssertionError("first call should surface the tool failure")

    reopened_metadata = EngineSQLite(tmp_path / "metadata")
    reopened_metadata.ensure_initialized()
    reopened_executor = NamedProjectionToolCallIdempotency(reopened_metadata)
    try:
        reopened_executor.execute_once("ambiguous-key", lambda: {"ok": True})
    except AmbiguousToolCallError as exc:
        assert "refusing unsafe replay" in str(exc)
    else:
        raise AssertionError("unfinished effect was replayed")
    assert calls == ["effect-may-have-happened"]
