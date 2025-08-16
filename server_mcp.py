# server_mcp.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import os
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.models import Document, Node, Edge

# ---------- App context ----------
class AppCtx(BaseModel):
    persist_directory: str
persist_directory=os.environ.get("MCP_CHROMA_DIR") or "./.chroma-mcp"
engine = GraphKnowledgeEngine(persist_directory=persist_directory)
gq = GraphQuery(engine)
mcp = FastMCP("KnowledgeEngine")

# ---------- I/O models for structured output ----------# server_mcp.py


# ---- Typed outputs per op ----
class FindEdgesOut(BaseModel):
    edges: List[str]

class NeighborsOut(BaseModel):
    nodes: List[str]
    edges: List[str]

class KHopLayer(BaseModel):
    nodes: List[str]
    edges: List[str]

class KHopOut(BaseModel):
    layers: List[KHopLayer]

class ShortestPathOut(BaseModel):
    path: List[str]

class SeedExpandOut(BaseModel):
    seeds: List[str]
    layers: List[KHopLayer]

# ---- Tools (one per op) ----
@mcp.tool(structured_output=True)
def kg_find_edges(
    relation: Optional[str] = None,
    src_label_contains: Optional[str] = None,
    tgt_label_contains: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> FindEdgesOut:
    eids = gq.find_edges(
        relation=relation,
        src_label_contains=src_label_contains,
        tgt_label_contains=tgt_label_contains,
        doc_id=doc_id,
    )
    return FindEdgesOut(edges=eids)

@mcp.tool(structured_output=True)
def kg_neighbors(rid: str, doc_id: Optional[str] = None) -> NeighborsOut:
    nb = gq.neighbors(rid, doc_id=doc_id)
    return NeighborsOut(nodes=sorted(nb["nodes"]), edges=sorted(nb["edges"]))

@mcp.tool(structured_output=True)
def kg_k_hop(start_ids: List[str], k: int = 1, doc_id: Optional[str] = None) -> KHopOut:
    layers = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"]))
        for L in gq.k_hop(start_ids, k=k, doc_id=doc_id)
    ]
    return KHopOut(layers=layers)

@mcp.tool(structured_output=True)
def kg_shortest_path(src_id: str, dst_id: str, doc_id: Optional[str] = None, max_depth: int = 8) -> ShortestPathOut:
    return ShortestPathOut(path=gq.shortest_path(src_id, dst_id, doc_id=doc_id, max_depth=max_depth))

@mcp.tool(structured_output=True)
def kg_semantic_seed_then_expand_text(text: str, top_k: int = 5, hops: int = 1, doc_id: Optional[str] = None) -> SeedExpandOut:
    out = gq.semantic_seed_then_expand_text(text, top_k=top_k, hops=hops)  # add doc_id if you extended it
    layers = [KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"])) for L in out["layers"]]
    return SeedExpandOut(seeds=out["seeds"], layers=layers)

# ---------- Entrypoints ----------
if __name__ == "__main__":
    # Streamable HTTP is the recommended production transport
    mcp.run(transport="streamable-http")
    # For stdio, run: mcp.run(transport="stdio")