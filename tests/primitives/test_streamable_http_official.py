# tests/test_streamable_http_e2e.py
import asyncio
import os
import socket
import subprocess
import sys
import time
import pathlib
import pytest
pytestmark = pytest.mark.ci_full
import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _wait(url: str, timeout=15):
    async with httpx.AsyncClient() as c:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = await c.get(url, timeout=0.5)
                if r.status_code < 500:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
    raise TimeoutError("server not ready")


# --- Preseed helper (same models/engine your server uses) ---
def _preseed_chroma_dir(persist_dir: str):
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Edge, Node

    from tests._kg_factories import kg_document, kg_grounding

    eng = GraphKnowledgeEngine(persist_directory=persist_dir)
    doc = kg_document(
        doc_id="D1",
        content="Smoking causes lung cancer.",
        source="test_streamable_http_e2e",
    )
    eng.write.add_document(doc)

    n_smoke = Node(
        label="Smoking",
        type="entity",
        summary="habit",
        mentions=[kg_grounding(doc.id, excerpt="Smoking", end_char=7)],
        doc_id=doc.id,
    )
    n_cancer = Node(
        label="Lung cancer",
        type="entity",
        summary="disease",
        mentions=[kg_grounding(doc.id, excerpt="lung cancer", end_char=11)],
        doc_id=doc.id,
    )
    eng.write.add_node(n_smoke, doc_id=doc.id)
    eng.write.add_node(n_cancer, doc_id=doc.id)

    e_causes = Edge(
        label="Smoking→Cancer",
        type="relationship",
        relation="causes",
        source_ids=[n_smoke.id],
        target_ids=[n_cancer.id],
        summary="causal claim",
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id, excerpt="causes", end_char=6)],
        doc_id=doc.id,
    )
    eng.write.add_edge(e_causes, doc_id=doc.id)


@pytest.mark.asyncio
async def test_streamable_http_e2e(tmp_path):
    chroma_dir = str(tmp_path / "chroma")
    _preseed_chroma_dir(chroma_dir)  # <-- seed BEFORE starting the server

    port = _free_port()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MCP_CHROMA_DIR": chroma_dir}
    srv = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server_mcp:mcp.streamable_http_app",
            "--factory",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        await _wait(f"http://127.0.0.1:{port}/mcp", timeout=100)

        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()  # SDK negotiates protocol

                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {
                    "kg_shortest_path",
                    "kg_find_edges",
                    "kg_semantic_seed_then_expand_text",
                    "kg_k_hop",
                    "kg_neighbors",
                } <= names

                # Prefer a non-embedding call for CI determinism:
                # res = await session.call_tool(
                #     "kg_query",
                #     arguments={"inp": {"op": "find_edges",
                #                "args": {"relation": "causes", "doc_id": "D1"}}}
                # )
                # find_edges
                import json
                from server_mcp import FindEdgesOut, KHopOut

                res = await session.call_tool(
                    "kg_find_edges", arguments={"relation": "causes", "doc_id": "D1"}
                )
                assert (res.content[0].type == "json") or (
                    res.content[0].type == "text" and json.loads(res.content[0].text)
                )
                FindEdgesOut.model_validate_json(res.content[0].text)

                # k_hop
                res = await session.call_tool(
                    "kg_k_hop", arguments={"start_ids": ["A"], "k": 2}
                )
                assert (res.content[0].type == "json") or (
                    res.content[0].type == "text" and json.loads(res.content[0].text)
                )
                KHopOut.model_validate_json(res.content[0].text)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=2)
        except Exception:
            srv.kill()
