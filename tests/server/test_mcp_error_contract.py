from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from kogwistar.server_mcp_with_admin import app
from tests._helpers.server_http_helpers import token_header

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "golden"
    / "mcp-tool-call-error-v1.json"
)


def _sse_message(body: str) -> dict[str, object]:
    data = next(
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ")
    )
    return json.loads(data)


def test_python_streamable_mcp_invalid_tool_call_matches_shared_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    expected = fixture["expected"]

    with TestClient(app) as client:
        headers = token_header(client, role="ro", ns="conversation")
        headers["Accept"] = "application/json, text/event-stream"
        initialized = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "contract-test", "version": "1"},
                },
            },
        )
        assert initialized.status_code == 200
        headers["Mcp-Session-Id"] = initialized.headers["mcp-session-id"]
        client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": fixture["tool_name"],
                    "arguments": fixture["arguments"],
                },
            },
        )

    message = _sse_message(response.text)
    result = message["result"]
    assert response.status_code == expected["http_status"]
    assert response.headers["content-type"].startswith(expected["content_type"])
    assert result["isError"] is expected["result_is_error"]
    assert result["content"][0]["type"] == expected["content_item_type"]
    assert bool(result["content"][0]["text"]) is expected["content_text_non_empty"]
    assert ("error" in message) is expected["top_level_error"]
