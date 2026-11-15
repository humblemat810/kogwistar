from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from kogwistar._rust_bridge import (
    RustExtensionUnavailableError,
    RustParityError,
    api_health,
    contract_canonical_json,
    contract_stable_id,
    runtime_apply_state_update,
    store_memory_read,
)

pytestmark = [pytest.mark.ci, pytest.mark.core]


class _ExtensionStub:
    def __init__(self) -> None:
        self.payloads: list[str] = []

    def api_health(self, payload: str) -> str:
        self.payloads.append(payload)
        return '{"ok":true}'


class _StableIdExtension:
    def __init__(self, result: UUID) -> None:
        self.result = result

    def stable_id_json(self, payload: str) -> str:
        return str(self.result)


@pytest.mark.parametrize(
    "invalid_value",
    [
        {"value": float("nan")},
        {"value": 1 << 64},
        {"nested": {1: "not a JSON object key"}},
    ],
)
def test_rust_bridge_rejects_invalid_json_before_native_call(
    monkeypatch: pytest.MonkeyPatch, invalid_value: Any
) -> None:
    extension = _ExtensionStub()
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")
    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", lambda: extension)

    with pytest.raises(RustParityError, match="violates strict JSON"):
        api_health(payload=invalid_value, python_value={"ok": True})

    assert extension.payloads == []


def test_rust_bridge_preserves_valid_json_payload_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _ExtensionStub()
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")
    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", lambda: extension)

    assert api_health(
        payload={"service": "wiki", "enabled": True}, python_value={"ok": False}
    ) == {"ok": True}
    assert extension.payloads == ['{"service":"wiki","enabled":true}']


def test_python_server_mode_keeps_legacy_path_without_loading_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "python")

    def unavailable() -> None:
        raise RustExtensionUnavailableError("extension intentionally unavailable")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)
    expected = {"ok": True, "implementation": "python"}

    assert api_health(payload={}, python_value=expected) is expected


def test_unset_implementation_selectors_preserve_python_bridge_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar import _rust_bridge as bridge
    from kogwistar.implementation_selection import _IMPLEMENTATION_ENV_NAMES

    for selector in _IMPLEMENTATION_ENV_NAMES:
        monkeypatch.delenv(selector, raising=False)

    def rust_must_not_load() -> None:
        raise AssertionError("unset selector default unexpectedly loaded Rust")

    monkeypatch.setattr(bridge, "_load_extension", rust_must_not_load)
    stable_id = UUID("8d9f6f96-e5a5-4ec8-9794-b9bd4938f9f0")
    cases = [
        ("server", lambda: bridge.api_health(payload={}, python_value={"ok": True}), {"ok": True}),
        (
            "contracts stable id",
            lambda: bridge.contract_stable_id(kind="node", parts=(), python_value=stable_id),
            stable_id,
        ),
        (
            "contracts canonical json",
            lambda: bridge.contract_canonical_json(value={"a": 1}, python_value='{"a":1}'),
            '{"a":1}',
        ),
        (
            "contracts evidence digest",
            lambda: bridge.contract_evidence_pack_digest_hash(value={}, python_value="digest"),
            "digest",
        ),
        (
            "contracts metadata filter",
            lambda: bridge.contract_metadata_filter_matches(
                metadata={}, where={}, python_value=True
            ),
            True,
        ),
        (
            "contracts short id",
            lambda: bridge.short_id_transform(payload={}, python_value={"id": "python"}),
            {"id": "python"},
        ),
        (
            "contracts canonical event",
            lambda: bridge.contract_canonical_entity_event(payload={}, python_value="event"),
            "event",
        ),
        (
            "contracts replay",
            lambda: bridge.contract_replay_entity_events(payload={}, python_value="replay"),
            "replay",
        ),
        (
            "runtime state update",
            lambda: bridge.runtime_apply_state_update(payload={}, python_value={"state": 1}),
            {"state": 1},
        ),
        (
            "runtime join reachability",
            lambda: bridge.runtime_workflow_may_reach_join(
                node_ids=[], edges=[], join_ids=[], python_value={"node": 1}
            ),
            {"node": 1},
        ),
        (
            "runtime terminal reachability",
            lambda: bridge.runtime_workflow_terminal_reachable(
                node_ids=[], edges=[], start_node_id="start", terminal_ids=[], python_value=True
            ),
            True,
        ),
        (
            "runtime successors",
            lambda: bridge.runtime_plan_successors(payload={}, python_value={"tokens": []}),
            {"tokens": []},
        ),
        (
            "runtime retry",
            lambda: bridge.runtime_decide_retry(payload={}, python_value={"retry": False}),
            {"retry": False},
        ),
        (
            "runtime nested invocation",
            lambda: bridge.runtime_plan_nested_invocation(payload={}, python_value={"run": "py"}),
            {"run": "py"},
        ),
        (
            "runtime dispatch",
            lambda: bridge.runtime_decide_dispatch(payload={}, python_value={"dispatch": False}),
            {"dispatch": False},
        ),
        (
            "runtime budget",
            lambda: bridge.runtime_decide_budget_suspend(payload={}, python_value={"suspend": False}),
            {"suspend": False},
        ),
        (
            "runtime scheduler",
            lambda: bridge.runtime_scheduler_tick(payload={}, python_value={"effects": []}),
            {"effects": []},
        ),
        (
            "graph store",
            lambda: bridge.store_memory_read(
                snapshot={}, operation={"kind": "list_nodes"}, python_value=[], store="graph"
            ),
            [],
        ),
        (
            "metadata store",
            lambda: bridge.store_memory_read(
                snapshot={}, operation={"kind": "list_jobs"}, python_value=[], store="meta"
            ),
            [],
        ),
    ]
    for name, invoke, expected in cases:
        assert invoke() == expected, name


def test_selected_rust_server_fails_closed_without_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")

    def unavailable() -> None:
        raise RustExtensionUnavailableError("extension intentionally unavailable")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)

    with pytest.raises(RustExtensionUnavailableError, match="intentionally unavailable"):
        api_health(payload={}, python_value={"ok": True})


def test_rust_stable_id_selection_does_not_evaluate_python_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = UUID("8d9f6f96-e5a5-4ec8-9794-b9bd4938f9f0")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "rust")
    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension",
        lambda: _StableIdExtension(expected),
    )

    def unexpected_python_oracle() -> UUID:
        raise AssertionError("Rust selection evaluated the Python stable-id oracle")

    assert (
        contract_stable_id(
            kind="node", parts=("title",), python_value=unexpected_python_oracle
        )
        == expected
    )


@pytest.mark.parametrize("mode", ["python", "shadow"])
def test_stable_id_oracle_runs_only_when_selected_or_shadowed(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    expected = UUID("8d9f6f96-e5a5-4ec8-9794-b9bd4938f9f0")
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", mode)
    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension",
        lambda: _StableIdExtension(expected),
    )
    calls = 0

    def python_oracle() -> UUID:
        nonlocal calls
        calls += 1
        return expected

    assert (
        contract_stable_id(kind="node", parts=("title",), python_value=python_oracle)
        == expected
    )
    assert calls == 1


@pytest.mark.parametrize(
    ("selector", "invoke"),
    [
        (
            "KOGWISTAR_IMPL_CONTRACTS",
            lambda: contract_canonical_json(value={"a": 1}, python_value='{"a":1}'),
        ),
        (
            "KOGWISTAR_IMPL_RUNTIME",
            lambda: runtime_apply_state_update(payload={"updates": []}, python_value=[]),
        ),
        (
            "KOGWISTAR_IMPL_GRAPH_STORE",
            lambda: store_memory_read(
                snapshot={}, operation={"kind": "list_nodes"}, python_value=[], store="graph"
            ),
        ),
        (
            "KOGWISTAR_IMPL_META_STORE",
            lambda: store_memory_read(
                snapshot={}, operation={"kind": "list_jobs"}, python_value=[], store="meta"
            ),
        ),
    ],
)
def test_explicit_rust_selectors_fail_closed_without_native_extension(
    monkeypatch: pytest.MonkeyPatch, selector: str, invoke
) -> None:
    monkeypatch.setenv(selector, "rust")

    def unavailable() -> None:
        raise RustExtensionUnavailableError("extension intentionally unavailable")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)

    with pytest.raises(RustExtensionUnavailableError, match="intentionally unavailable"):
        invoke()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: api_health(payload={}, python_value={"ok": True}),
        lambda: contract_canonical_json(value={"value": 1}, python_value='{"value":1}'),
        lambda: runtime_apply_state_update(payload={"updates": []}, python_value=[]),
        lambda: store_memory_read(
            snapshot={}, operation={"kind": "list_nodes"}, python_value=[], store="graph"
        ),
        lambda: store_memory_read(
            snapshot={}, operation={"kind": "list_jobs"}, python_value=[], store="meta"
        ),
    ],
    ids=["server", "contracts", "runtime", "graph-store", "meta-store"],
)
def test_global_rust_selector_fails_closed_when_component_selectors_are_unset(
    monkeypatch: pytest.MonkeyPatch, invoke
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
    for selector in (
        "KOGWISTAR_IMPL_SERVER",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
    ):
        monkeypatch.delenv(selector, raising=False)

    def unavailable() -> None:
        raise RustExtensionUnavailableError("extension intentionally unavailable")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)

    with pytest.raises(RustExtensionUnavailableError, match="intentionally unavailable"):
        invoke()
