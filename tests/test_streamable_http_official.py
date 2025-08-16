# tests/test_streamable_http_official.py
import asyncio, socket, subprocess, sys, time, os, pathlib, pytest
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _free_port():
    s = socket.socket(); s.bind(("",0)); p=s.getsockname()[1]; s.close(); return p

async def _wait(url: str, timeout=10):
    async with httpx.AsyncClient() as c:
        t0=time.time()
        while time.time()-t0 < timeout:
            try:
                r=await c.get(url, timeout=0.5)
                if r.status_code < 500: return
            except Exception: pass
            await asyncio.sleep(0.1)
    raise TimeoutError("server not ready")

@pytest.mark.asyncio
async def test_streamable_http_e2e(tmp_path):
    port = _free_port()
    env = {**os.environ, "PYTHONUNBUFFERED":"1"}
    srv = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server_mcp:mcp.streamable_http_app", "--factory", "--port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        await _wait(f"http://127.0.0.1:{port}/mcp")
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()  # protocol negotiation handled by SDK
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                assert {"kg_extract","doc_parse","kg_query","doc_query"} <= set(names)

                r = await session.call_tool("doc_query", arguments={"text":"smok", "top_k":1, "hops":0})
                # The README shows list_tools/call_tool usage with this client pattern. 
                assert r.content, "expected content blocks"
    finally:
        srv.terminate(); 
        try: srv.wait(timeout=2)
        except Exception: srv.kill()
