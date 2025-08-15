import json, subprocess, sys, time, os, pathlib

def _send(p, obj):
    p.stdin.write((json.dumps(obj)+"\n").encode()); p.stdin.flush()

def _recv(p, timeout=2.0):
    t0=time.time()
    while time.time()-t0<timeout:
        line=p.stdout.readline()
        if line: return json.loads(line.decode())
    raise TimeoutError("no response")

def test_stdio_roundtrip(tmp_path):
    env = {**os.environ, "PYTHONUNBUFFERED":"1"}
    server = subprocess.Popen(
        [sys.executable, "mcp_stdio_server.py"],
        cwd=str(pathlib.Path(__file__).parent.parent),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    try:
        _send(server, {"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"1.0.0"}})
        r1 = _recv(server); assert r1["result"]["serverInfo"]["name"] == "demo-stdio"

        _send(server, {"jsonrpc":"2.0","id":"2","method":"tools/list","params":{}})
        r2 = _recv(server); assert isinstance(r2["result"]["tools"], list)

        _send(server, {"jsonrpc":"2.0","id":"3","method":"tools/call",
                       "params":{"name":"kg.query","arguments":{"op":"final_summary_node_id","doc_id":"D1"}}})
        r3 = _recv(server); assert "content" in r3["result"]

        _send(server, {"jsonrpc":"2.0","id":"4","method":"shutdown","params":{}})
        _recv(server)
    finally:
        server.terminate()
        try: server.wait(timeout=1)
        except Exception: server.kill()
