from fastapi.testclient import TestClient
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from mcp_http_server import app

client = TestClient(app)

def rpc(method, params=None, id="1"):
    return {"jsonrpc":"2.0","id":id,"method":method,"params":params or {}}

def test_http_roundtrip():
    r1 = client.post("/mcp", json=rpc("initialize", {"protocolVersion":"1.0.0"})).json()
    assert r1["result"]["serverInfo"]["name"] == "demo-http"

    r2 = client.post("/mcp", json=rpc("tools/list")).json()
    assert isinstance(r2["result"]["tools"], list)

    r3 = client.post("/mcp", json=rpc("tools/call",
        {"name":"kg.query","arguments":{"op":"final_summary_node_id","doc_id":"D1"}}, id="3")).json()
    assert "content" in r3["result"]

    r4 = client.post("/mcp", json=rpc("shutdown", id="4")).json()
    assert r4["result"] == {}
