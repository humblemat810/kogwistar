import pytest
import requests
import jwt
import json
from typing import Dict
import os
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

GRAPH_RAG_PORT = 28110
BASE = f"http://127.0.0.1:{GRAPH_RAG_PORT}"

def _mint_token(role: str) -> str:
    """Uses your /auth/dev-token helper added on the server to get a JWT."""
    r = requests.post(f"{BASE}/auth/dev-token", json={"username": "pytest", "role": role})
    r.raise_for_status()
    return r.json()["token"]

@pytest.mark.asyncio
async def test_readonly_token_blocks_writes_and_allows_reads():
    # 1) Get a read-only token
    ro_token = _mint_token("ro")
    headers: Dict[str, str] = {"Authorization": f"Bearer {ro_token}"}

    # 2) REST: write endpoint should be forbidden
    # (even if the doc doesn't exist, the RBAC gate should trip first)
    resp = requests.delete(f"{BASE}/admin/doc/NONEXISTENT_DOC", headers=headers)
    assert resp.status_code == 403, f"Expected 403 for RO delete, got {resp.status_code}: {resp.text}"

    # 3) MCP: connect with RO token
    async with streamablehttp_client(f"{BASE}/mcp", headers=headers,
                                     sse_read_timeout=None, timeout=None) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3a) Read tool should succeed
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "kg_find_edges" in names, f"Read tool not found; have: {sorted(names)}"

            read_res = await session.call_tool("kg_find_edges", arguments={})
            # We just assert the call returned something without raising
            assert read_res is not None

            # 3b) Write tool should fail for RO token
            # Try a minimal write tool. Use store_document (or doc_parse / kg_extract).
            err_resp = await session.call_tool("store_document", arguments={
                "inp": {"id": "RO_BLOCK_TEST", "content": "hello", "type": "plain"}
            })
            # If the server properly guards with require_role("rw"), this should not succeed.
            # Some MCP client libs return an error envelope instead of raising — check both paths:
            # If your implementation returns a JSON error payload instead of raising, assert 403-ish message.
            assert err_resp.content[0].text == "Error calling tool 'store_document': 403: Forbidden: requires role 'rw', you have 'ro'"
            



GRAPH_RAG_PORT = int(os.getenv("GRAPH_RAG_PORT", "28110"))
BASE = f"http://127.0.0.1:{GRAPH_RAG_PORT}"
MCP_URL = f"{BASE}/mcp"

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = os.getenv("JWT_ALG", "HS256")

WRITE_TOOLS = {
    # Marked @tool_roles(Role.RW)
    "kg_extract",
    "store_document",
    "kg_crossdoc_adjudicate_anykind",
}

READ_TOOLS_MIN = {
    # A few representative RO tools (there can be more)
    "kg_find_edges",
    "kg_neighbors",
    "kg_k_hop",
    "kg_shortest_path",
    "kg_viz_d3_json",
    "kg_viz_cytoscape_json",
}

def _mint(role: str) -> str:
    return jwt.encode({"role": role}, JWT_SECRET, algorithm=JWT_ALG)

def _ensure_server():
    try:
        r = requests.get(f"{BASE}/health", timeout=1.5)
        r.raise_for_status()
        return True
    except Exception:
        pytest.skip(f"Server not reachable at {BASE}")

@pytest.mark.asyncio
async def test_tools_visibility_filtered_by_role():
    _ensure_server()

    ro_headers = {"Authorization": f"Bearer {_mint('ro')}"}
    rw_headers = {"Authorization": f"Bearer {_mint('rw')}"}

    # --- RO: list tools (write tools must be filtered out)
    async with streamablehttp_client(MCP_URL, headers=ro_headers, sse_read_timeout=None, timeout=None) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

            # Sanity: we see some RO tools
            assert READ_TOOLS_MIN & names, f"Expected some read-only tools, got: {sorted(names)}"
            # Ensure write tools are NOT visible
            assert WRITE_TOOLS.isdisjoint(names), f"RO client should not see write tools, but got: {WRITE_TOOLS & names}"

    # --- RW: list tools (write tools should be visible)
    async with streamablehttp_client(MCP_URL, headers=rw_headers, sse_read_timeout=None, timeout=None) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

            missing = WRITE_TOOLS - names
            assert not missing, f"RW client should see write tools, missing: {sorted(missing)}"

@pytest.mark.asyncio
async def test_ro_blocked_on_write_tool_and_rw_allowed():
    _ensure_server()

    ro_headers = {"Authorization": f"Bearer {_mint('ro')}"}
    rw_headers = {"Authorization": f"Bearer {_mint('rw')}"}

    # Pick a write tool that exists on your server. `kg_extract` is a good canary:
    write_tool = "kg_extract"

    # ---- RO should be blocked on invocation
    async with streamablehttp_client(MCP_URL, headers=ro_headers, sse_read_timeout=None, timeout=None) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Even if tool is hidden from list, we try to call it directly to ensure server-side guard works.
            try:
                res = await session.call_tool(write_tool, arguments={"inp": {"mode": "skip-if-exists", "id": "DUMMY"}})
                # Some servers return an error envelope; some return text/json with an error message
                payload = None
                if res.content:
                    c = res.content[0]
                    if c.type == "json":
                        payload = c.json
                    elif c.type == "text":
                        try:
                            payload = json.loads(c.text)
                        except Exception:
                            payload = {"text": c.text}
                # Must indicate permission denied somewhere
                body = payload or {}
                blob = json.dumps(body)
                assert ("requires read-write role" in blob) or ("403" in blob) or ("Forbidden" in blob.lower()), \
                    f"RO must be blocked; got response: {blob}"
            except Exception as e:
                # Also acceptable: transport-level error from tool invocation
                assert "Permission" in str(e) or "requires read-write" in str(e)

    # ---- RW should be allowed (we don't assert business success, just not permission denied)
    async with streamablehttp_client(MCP_URL, headers=rw_headers, sse_read_timeout=None, timeout=None) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(write_tool, arguments={"inp":{"mode": "skip-if-exists", "id": "DUMMY"}})
            # Not a permission error
            blob = ""
            if res.content:
                c = res.content[0]
                blob = c.text if c.type == "text" else json.dumps(c.json)
            assert "requires read-write role" not in blob, f"RW must be allowed; got: {blob}"
    
    resp = requests.delete(f"{BASE}/admin/doc/DUMMY", headers=rw_headers)
