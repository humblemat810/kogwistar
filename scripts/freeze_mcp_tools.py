from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def _tool_contracts(registry) -> list[dict]:
    tools = await registry.list_tools()
    contracts = [
        {
            "name": tool.name,
            "title": tool.title,
            "description": tool.description,
            "inputSchema": tool.parameters,
            "outputSchema": tool.output_schema,
        }
        for tool in tools
    ]
    contracts.sort(key=lambda item: item["name"])
    names = [item["name"] for item in contracts]
    if len(names) != len(set(names)):
        raise RuntimeError("MCP tool registry contains duplicate names")
    return contracts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contracts/golden/mcp-tools.json"),
    )
    args = parser.parse_args()
    async def collect() -> dict[str, list[dict]]:
        from kogwistar.server.mcp_tools import conversation_mcp, workflow_mcp
        from kogwistar.server_mcp_with_admin import mcp

        return {
            "root": await _tool_contracts(mcp),
            "conversation": await _tool_contracts(conversation_mcp),
            "workflow": await _tool_contracts(workflow_mcp),
        }

    surfaces = asyncio.run(collect())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"contract_version": "1.0.0", "surfaces": surfaces},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {sum(len(tools) for tools in surfaces.values())} MCP tools "
        f"across {len(surfaces)} surfaces to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
