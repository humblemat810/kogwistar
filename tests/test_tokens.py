import json
import os
from typing import Dict

import jwt
import pytest
import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = os.getenv("JWT_ALG", "HS256")

WRITE_TOOLS = {
    "kg_extract",
    "store_document",
    "kg_crossdoc_adjudicate_anykind",
}

READ_TOOLS_MIN = {
    "kg_find_edges",
    "kg_neighbors",
    "kg_k_hop",
    "kg_shortest_path",
    "kg_viz_d3_json",
    "kg_viz_cytoscape_json",
}


def _mint_token(base_http: str, role: str) -> str:
    response = requests.post(
        f"{base_http}/auth/dev-token",
        json={"username": "pytest", "role": role},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()["token"]


def _mint(role: str) -> str:
    return jwt.encode({"role": role}, JWT_SECRET, algorithm=JWT_ALG)


@pytest.mark.asyncio
async def test_readonly_token_blocks_writes_and_allows_reads(mcp_admin_server):
    base_http = str(mcp_admin_server["base_http"])
    mcp_url = str(mcp_admin_server["base_mcp"])

    ro_token = _mint_token(base_http, "ro")
    headers: Dict[str, str] = {"Authorization": f"Bearer {ro_token}"}

    resp = requests.delete(
        f"{base_http}/admin/doc/NONEXISTENT_DOC",
        headers=headers,
        timeout=20,
    )
    assert resp.status_code == 403, (
        f"Expected 403 for RO delete, got {resp.status_code}: {resp.text}"
    )

    async with streamablehttp_client(
        mcp_url,
        headers=headers,
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "kg_find_edges" in names, (
                f"Read tool not found; have: {sorted(names)}"
            )

            read_res = await session.call_tool("kg_find_edges", arguments={})
            assert read_res is not None

            err_resp = await session.call_tool(
                "store_document",
                arguments={
                    "inp": {"id": "RO_BLOCK_TEST", "content": "hello", "type": "text"}
                },
            )
            assert err_resp.content
            first = err_resp.content[0]
            blob = (
                getattr(first, "text", "")
                if getattr(first, "type", "") == "text"
                else json.dumps(first.json)
            )
            assert ("403" in blob) or ("Forbidden" in blob)


@pytest.mark.asyncio
async def test_tools_visibility_filtered_by_role(mcp_admin_server):
    mcp_url = str(mcp_admin_server["base_mcp"])

    ro_headers = {"Authorization": f"Bearer {_mint('ro')}"}
    rw_headers = {"Authorization": f"Bearer {_mint('rw')}"}

    async with streamablehttp_client(
        mcp_url,
        headers=ro_headers,
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

            assert READ_TOOLS_MIN & names, (
                f"Expected some read-only tools, got: {sorted(names)}"
            )
            assert WRITE_TOOLS.isdisjoint(names), (
                f"RO client should not see write tools, but got: {WRITE_TOOLS & names}"
            )

    async with streamablehttp_client(
        mcp_url,
        headers=rw_headers,
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

            missing = WRITE_TOOLS - names
            assert not missing, (
                f"RW client should see write tools, missing: {sorted(missing)}"
            )


@pytest.mark.asyncio
async def test_ro_blocked_on_write_tool_and_rw_allowed(mcp_admin_server):
    base_http = str(mcp_admin_server["base_http"])
    mcp_url = str(mcp_admin_server["base_mcp"])

    ro_headers = {"Authorization": f"Bearer {_mint('ro')}"}
    rw_headers = {"Authorization": f"Bearer {_mint('rw')}"}
    write_tool = "kg_extract"

    async with streamablehttp_client(
        mcp_url,
        headers=ro_headers,
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                res = await session.call_tool(
                    write_tool,
                    arguments={"inp": {"mode": "skip-if-exists", "id": "DUMMY"}},
                )
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
                blob = json.dumps(payload or {})
                assert (
                    ("403" in blob)
                    or ("Forbidden" in blob)
                    or ("not permitted" in blob)
                ), f"RO must be blocked; got response: {blob}"
            except Exception as exc:
                text = str(exc)
                assert (
                    ("Permission" in text)
                    or ("Forbidden" in text)
                    or ("not permitted" in text)
                )

    async with streamablehttp_client(
        mcp_url,
        headers=rw_headers,
        sse_read_timeout=None,
        timeout=None,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                write_tool, arguments={"inp": {"mode": "skip-if-exists", "id": "DUMMY"}}
            )
            blob = ""
            if res.content:
                c = res.content[0]
                blob = c.text if c.type == "text" else json.dumps(c.json)
            assert "not permitted" not in blob
            assert "Forbidden" not in blob

    cleanup = requests.delete(
        f"{base_http}/admin/doc/DUMMY", headers=rw_headers, timeout=20
    )
    assert cleanup.status_code in {200, 204, 404}
