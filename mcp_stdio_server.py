# Run: python mcp_stdio_server.py
import sys, json, uuid
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.mcp_endpoints import KnowledgeMCP  # your dispatcher

engine = GraphKnowledgeEngine(persist_directory="./.chroma-test")
mcp = KnowledgeMCP(engine, ingester_llm=None)

def send(id_, *, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg["result" if error is None else "error"] = result if error is None else error
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()

for line in sys.stdin:
    if not line.strip(): 
        continue
    req = json.loads(line)
    rid = req.get("id", str(uuid.uuid4()))
    method = req.get("method")
    params = req.get("params") or {}
    try:
        if method == "initialize":
            send(rid, result={"protocolVersion":"1.0.0",
                              "capabilities":{"tools":{"listChanged":True}},
                              "serverInfo":{"name":"demo-stdio","version":"0.1"}})
        elif method == "tools/list":
            send(rid, result={"tools": mcp.list_tools()})
        elif method == "tools/call":
            name = params["name"]; args = params.get("arguments", {})
            send(rid, result={"content":[{"type":"json","json": mcp.call(name, args)}]})
        elif method == "shutdown":
            send(rid, result={}); break
        else:
            send(rid, error={"code":-32601,"message":f"Method not found: {method}"})
    except Exception as e:
        send(rid, error={"code":-32000,"message":str(e)})