from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any


async def _tool_contracts(registry: Any) -> list[dict[str, Any]]:
    tools = await registry.list_tools()
    contracts = [
        {
            "name": tool.name,
            "title": tool.title,
            "description": tool.description,
            "inputSchema": tool.input_schema,
            "outputSchema": tool.output_schema,
        }
        for tool in tools
    ]
    contracts.sort(key=lambda item: item["name"])
    names = [item["name"] for item in contracts]
    if len(names) != len(set(names)):
        raise RuntimeError("MCP tool registry contains duplicate names")
    return contracts


async def collect_surfaces() -> dict[str, list[dict[str, Any]]]:
    from kogwistar.server.auth_middleware import (
        reset_claims_ctx,
        reset_current_role,
        set_claims_ctx,
        set_current_role,
    )
    from kogwistar.server.mcp_tools import conversation_mcp, workflow_mcp
    from kogwistar.server_mcp_with_admin import mcp

    root = await _tool_contracts(mcp)
    claims_token = set_claims_ctx(
        {
            "role": "rw",
            "ns": ["wisdom"],
            "capabilities": ["read_graph", "write_graph"],
        }
    )
    role_token = set_current_role("rw")
    try:
        wisdom_tools = await _tool_contracts(mcp)
    finally:
        reset_current_role(role_token)
        reset_claims_ctx(claims_token)
    root_by_name = {tool["name"]: tool for tool in root}
    root_by_name.update({tool["name"]: tool for tool in wisdom_tools})
    return {
        "root": sorted(root_by_name.values(), key=lambda item: item["name"]),
        "conversation": await _tool_contracts(conversation_mcp),
        "workflow": await _tool_contracts(workflow_mcp),
    }


def _render_catalog(surfaces: dict[str, list[dict[str, Any]]]) -> str:
    return json.dumps(
        {"contract_version": "1.0.0", "surfaces": surfaces},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contracts/golden/mcp-tools.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the Python runtime catalog differs from the committed contract",
    )
    args = parser.parse_args()
    surfaces = asyncio.run(collect_surfaces())
    rendered = _render_catalog(surfaces)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"MCP tool contract is stale: regenerate {args.output}")
            return 1
        print(f"MCP tool contract is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {sum(len(tools) for tools in surfaces.values())} MCP tools "
        f"across {len(surfaces)} surfaces to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
