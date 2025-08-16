# server_mcp.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.models import Document, Node, Edge

# ---------- App context ----------
class AppCtx(BaseModel):
    persist_directory: str

engine = GraphKnowledgeEngine(persist_directory="./.chroma-mcp")
gq = GraphQuery(engine)
mcp = FastMCP("KnowledgeEngine")

# ---------- I/O models for structured output ----------
class KGExtractIn(BaseModel):
    doc_id: str

class KGExtractOut(BaseModel):
    node_ids: List[str]
    edge_ids: List[str]

class DocParseIn(BaseModel):
    doc_id: str
    content: str
    type: str = "plain"

class DocParseOut(BaseModel):
    doc_id: str
    chunk_count: int
    summary_node_id: Optional[str] = None

class KGQueryIn(BaseModel):
    op: str
    args: Dict[str, Any] = {}

class KGQueryOut(BaseModel):
    ok: bool
    result: Dict[str, Any]

class DocQueryIn(BaseModel):
    text: str
    top_k: int = 5
    hops: int = 0

class DocQueryOut(BaseModel):
    seeds: List[str]
    layers: List[Dict[str, Any]]

# ---------- Tools ----------
@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    """(1) Knowledge graph extraction method for an already ingested document."""
    # You likely have an engine method that extracts KG into collections; adapt here.
    # Example: engine.extract_graph(doc_id=inp.doc_id)
    node_ids = engine._nodes_by_doc(inp.doc_id)
    edge_ids = engine._edge_ids_by_doc(inp.doc_id)
    return KGExtractOut(node_ids=node_ids, edge_ids=edge_ids)

@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    """(2) Document parsing -> chunks + summary tree (final summary node if present)."""
    doc = Document(id=inp.doc_id, content=inp.content, type=inp.type)
    engine.add_document(doc)
    # Your existing chunking/summarization pipeline should be invoked here
    # (left as-is to avoid pulling external LLMs in this snippet).
    summary_node = gq.final_summary_node_id(inp.doc_id)
    # If you store chunks in a collection, compute count; here we approximate by node count.
    chunk_count = len(engine._nodes_by_doc(inp.doc_id))
    return DocParseOut(doc_id=inp.doc_id, chunk_count=chunk_count, summary_node_id=summary_node)

@mcp.tool()
def kg_query(inp: KGQueryIn) -> KGQueryOut:
    """(3) Query the knowledge graph (GraphQuery entry point)."""
    op = inp.op
    args = dict(inp.args)
    # Small router: add more ops as needed
    if op == "final_summary_node_id":
        res = {"id": gq.final_summary_node_id(args["doc_id"])}
    elif op == "neighbors":
        res = {k: sorted(v) for k, v in gq.neighbors(args["rid"], doc_id=args.get("doc_id")).items()}
    elif op == "k_hop":
        res = gq.k_hop(args["start_ids"], k=int(args.get("k", 1)), doc_id=args.get("doc_id"))
    elif op == "shortest_path":
        res = gq.shortest_path(args["src_id"], args["dst_id"], doc_id=args.get("doc_id"), max_depth=int(args.get("max_depth", 8)))
    else:
        return KGQueryOut(ok=False, result={"error": f"unknown op: {op}"})
    return KGQueryOut(ok=True, result=res)

@mcp.tool()
def doc_query(inp: DocQueryIn) -> DocQueryOut:
    """(4) Query the parsed document (semantic seed + optional K-hop expansion)."""
    out = gq.semantic_seed_then_expand_text(inp.text, top_k=inp.top_k, hops=inp.hops)
    return DocQueryOut(seeds=out["seeds"], layers=out["layers"])

# ---------- Entrypoints ----------
if __name__ == "__main__":
    # Streamable HTTP is the recommended production transport
    mcp.run(transport="streamable-http")
    # For stdio, run: mcp.run(transport="stdio")