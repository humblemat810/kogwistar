from __future__ import annotations

from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, api_health

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
