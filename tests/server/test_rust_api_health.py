from __future__ import annotations

import pytest

from kogwistar._rust_bridge import (
    api_authorize,
    api_cli_health,
    api_health,
    api_mcp_result,
    api_sse_frame,
)
import kogwistar.server_mcp_with_admin as server

pytestmark = [pytest.mark.ci]


@pytest.mark.parametrize(
    "mode",
    [
        "python",
        pytest.param("rust", marks=pytest.mark.requires_rust),
    ],
)
def test_rust_api_health_preserves_python_contract(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", mode)
    payload = {
        "backend": "pg",
        "persist_directory": "root",
        "conversation_persist_directory": "conversation",
        "workflow_persist_directory": "workflow",
        "wisdom_persist_directory": "wisdom",
        "pg_schema_base": "kogwistar",
    }
    python_value = {"ok": True, **payload}

    assert api_health(payload=payload, python_value=python_value) == python_value


def test_python_health_handler_keeps_legacy_default_without_native_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request-level Python server path must not require Rust."""

    for name in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)

    def native_must_not_load() -> object:
        raise AssertionError("legacy Python server route loaded the Rust extension")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", native_must_not_load)

    body = server.health()
    assert body["ok"] is True
    assert "backend" in body


@pytest.mark.requires_rust
def test_rust_api_auth_sse_mcp_and_cli_contracts() -> None:
    assert api_authorize(roles=["admin"], required_roles=["admin"]) is True
    assert api_authorize(roles=["reader"], required_roles=["admin"]) is False
    assert api_sse_frame(event="run", data={"ok": True}, event_id="7") == (
        'id: 7\nevent: run\ndata: {"ok":true}\n\n'
    )
    assert api_mcp_result(request_id=1, result={"ok": True}) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"ok": True},
    }
    assert api_cli_health(
        payload={
            "backend": "pg",
            "persist_directory": "root",
            "conversation_persist_directory": "conversation",
            "workflow_persist_directory": "workflow",
            "wisdom_persist_directory": "wisdom",
            "pg_schema_base": "kogwistar",
        }
    ) == "ok backend=pg"
