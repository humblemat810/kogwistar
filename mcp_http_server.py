# Run: uvicorn mcp_http_server:app --port 8765
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional
import uuid

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.mcp_endpoints import KnowledgeMCP

engine = GraphKnowledgeEngine(persist_directory="./.chroma-test")
mcp = KnowledgeMCP(engine, ingester_llm=None)

app = FastAPI()

class RPC(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[dict[str, Any]] = None

@app.post("/mcp")
async def mcp_endpoint(req: RPC):
    rid = req.id or str(uuid.uuid4())
    try:
        if req.method == "initialize":
            return {"jsonrpc":"2.0","id":rid,"result":{
                "protocolVersion":"1.0.0",
                "capabilities":{"tools":{"listChanged":True}},
                "serverInfo":{"name":"demo-http","version":"0.1"}}}
        if req.method == "tools/list":
            return {"jsonrpc":"2.0","id":rid,"result":{"tools": mcp.list_tools()}}
        if req.method == "tools/call":
            p = req.params or {}
            out = mcp.call(p["name"], p.get("arguments", {}))
            return {"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"json","json": out}]}}
        if req.method == "shutdown":
            return {"jsonrpc":"2.0","id":rid,"result":{}}
        return {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":f"Method not found: {req.method}"}}
    except Exception as e:
        return {"jsonrpc":"2.0","id":rid,"error":{"code":-32000,"message":str(e)}}
