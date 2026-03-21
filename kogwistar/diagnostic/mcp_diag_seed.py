# mcp_diag_seed.py
# Seed a small doc into any MCP server, run extraction, and sanity-query it.
# Uses the SAME session pattern as your agent (MultiServerMCPClient + manual __aenter__/__aexit__).
#
# Example:
#   python mcp_diag_seed.py --url http://127.0.0.1:28110/mcp --doc-id D1 \
#       --text "Smoking causes lung cancer." --relation causes --mode replace
#
# Optional admin cleanup (non-MCP endpoint mounted next to /mcp):
#   python mcp_diag_seed.py --url http://127.0.0.1:28110/mcp --doc-id D1 --delete

from __future__ import annotations
import argparse
import asyncio
import json
import re

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------- helpers ----------


def _base_from_mcp(url: str) -> str:
    # strip trailing /mcp or /mcp/
    return re.sub(r"/mcp/?$", "", url)


def _needs_inp_wrapper(input_schema: dict | None) -> bool:
    if not input_schema:
        return False
    props = input_schema.get("properties") or {}
    # common pattern in your server where the tool signature is (inp: Model)
    return "inp" in props and (props["inp"].get("type") in (None, "object"))


def _wrap_if_needed(input_schema: dict | None, payload: dict) -> dict:
    return {"inp": payload} if _needs_inp_wrapper(input_schema) else payload


async def _find_tool(session, name: str) -> dict | None:
    tools = await session.list_tools()
    for t in tools.tools:
        if t.name == name:
            return {
                "name": t.name,
                "input_schema": t.inputSchema,
                "output_schema": t.outputSchema,
            }
    return None


async def _call_tool_json(session, name: str, args: dict) -> dict:
    res = await session.call_tool(name, arguments=args)
    block = res.content[0]
    if (
        getattr(block, "type", None) == "json"
        and getattr(block, "json", None) is not None
    ):
        return block.json
    # normalize text results so the CLI never crashes
    return {"text": getattr(block, "text", "")}


# ---------- main ----------


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url",
        required=True,
        help="MCP Streamable HTTP endpoint (e.g., http://127.0.0.1:28110/mcp)",
    )
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--text", default="Smoking causes lung cancer.")
    ap.add_argument("--type", default="text")
    ap.add_argument(
        "--mode", default="replace", choices=["replace", "append", "skip-if-exists"]
    )
    ap.add_argument(
        "--relation", default="causes", help="Relation to list in the sanity query"
    )
    ap.add_argument(
        "--delete",
        action="store_true",
        help="After test, call non-MCP admin delete endpoint",
    )
    args = ap.parse_args()
    # Optional: non-MCP admin cleanup
    if args.delete:
        base = _base_from_mcp(args.url)
        async with httpx.AsyncClient(timeout=10) as client_http:
            resp = await client_http.delete(f"{base}/admin/doc/{args.doc_id}")
            resp.raise_for_status()
            print("[admin delete]", resp.json())
        return

    # Adapter expects a NAME -> CONFIG mapping (your fixed convention)
    # --- add this helper near the top ---
    def _normalize_base(url: str) -> str:
        # ensure exactly one trailing slash so path joins keep the /mcp prefix
        return url.rstrip("/") + "/"

    # ... inside main() right after parsing args ...
    base_url = _normalize_base(args.url)

    SERVERS = {
        "KnowledgeEngine": {
            "transport": "streamable_http",
            "url": base_url,
            # Optional: custom headers if your gateway needs them
            # "headers": {"MCP-Protocol-Version": "2025-03-26"},
        }
    }

    client = MultiServerMCPClient(SERVERS)

    # Create the context manager and ENTER it to get a real MCP session
    ctxs = [client.session(s) for s in SERVERS]
    sessions = await asyncio.gather(*(s.__aenter__() for s in ctxs))
    try:
        session = sessions[0]
        # 1) doc_parse (required before extraction)
        doc_parse = await _find_tool(session, "doc_parse")
        if not doc_parse:
            raise SystemExit("Server does not expose 'doc_parse' tool.")
        dp_args = _wrap_if_needed(
            doc_parse["input_schema"],
            {"doc_id": args.doc_id, "content": args.text, "type": args.type},
        )
        dp_out = await _call_tool_json(session, "doc_parse", dp_args)
        print("[doc_parse]", json.dumps(dp_out, ensure_ascii=False))

        # 2) kg_extract
        kg_extract = await _find_tool(session, "kg_extract")
        if not kg_extract:
            raise SystemExit("Server does not expose 'kg_extract' tool.")
        ke_args = _wrap_if_needed(
            kg_extract["input_schema"],
            {"doc_id": args.doc_id, "mode": args.mode},
        )
        ke_out = await _call_tool_json(session, "kg_extract", ke_args)
        print("[kg_extract]", json.dumps(ke_out, ensure_ascii=False))

        # 3) sanity query
        find_edges = await _find_tool(session, "kg_find_edges")
        if find_edges:
            fe_out = await _call_tool_json(
                session,
                "kg_find_edges",
                {"relation": args.relation, "doc_id": args.doc_id},
            )
            print("[kg_find_edges]", json.dumps(fe_out, ensure_ascii=False))
        else:
            # fallback: kg_query multiplexer, if you still expose it
            kg_query = await _find_tool(session, "kg_query")
            if kg_query:
                q_args = _wrap_if_needed(
                    kg_query["input_schema"],
                    {
                        "op": "find_edges",
                        "args": {"relation": args.relation, "doc_id": args.doc_id},
                    },
                )
                q_out = await _call_tool_json(session, "kg_query", q_args)
                print("[kg_query(find_edges)]", json.dumps(q_out, ensure_ascii=False))
            else:
                print(
                    "No 'kg_find_edges' or 'kg_query' tools found; skipping sanity query."
                )

    finally:
        # Properly EXIT the context (so sockets/processes are cleaned up)
        await asyncio.gather(*(s.__aexit__(None, None, None) for s in sessions))


if __name__ == "__main__":
    asyncio.run(main())
