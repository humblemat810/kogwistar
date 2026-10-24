from __future__ import annotations

from typing import Any

import pytest

from kogwistar._rust_bridge import (
    RustExtensionUnavailableError,
    RustParityError,
    api_health,
)

pytestmark = [pytest.mark.ci, pytest.mark.core]


class _ExtensionStub:
    def __init__(self) -> None:
        self.payloads: list[str] = []

    def api_health(self, payload: str) -> str:
        self.payloads.append(payload)
        return '{"ok":true}'


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


def test_selected_rust_server_fails_closed_without_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")

    def unavailable() -> None:
        raise RustExtensionUnavailableError("extension intentionally unavailable")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)

    with pytest.raises(RustExtensionUnavailableError, match="intentionally unavailable"):
        api_health(payload={}, python_value={"ok": True})
