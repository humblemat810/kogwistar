# server_mcp_with_admin.py
from __future__ import annotations
from typing import List, Optional
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.models import Document

# ---- Engine + MCP ----
persist_directory = os.environ.get("MCP_CHROMA_DIR") or "./.chroma-mcp"
engine = GraphKnowledgeEngine(persist_directory=persist_directory)
gq = GraphQuery(engine)

# Fastapi

mcp = FastMCP("KnowledgeEngine + MCP + Admin")
# ---- Build a unified FastAPI app: /mcp + /admin ----
mcp_app = mcp.http_app(path='/mcp')
app = FastAPI(title="KnowledgeEngine + MCP + Admin", lifespan=mcp_app.lifespan)

# Mount the MCP server
app.mount("/", mcp_app)

# health
@app.get("/health")
def health():
    return {"ok": True, "persist_directory": persist_directory}

# DELETE /admin/doc/{doc_id}  (non-MCP utility)
@app.delete("/admin/doc/{doc_id}")
def admin_delete_doc(doc_id: str):
    # Collect counts before deletion
    try:
        node_ids = engine._nodes_by_doc(doc_id)
        edge_ids = engine._edge_ids_by_doc(doc_id)
    except Exception:
        node_ids, edge_ids = [], []

    # Delete endpoints and mapping tables first
    try:
        engine.edge_endpoints_collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    try:
        engine.node_docs_collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    # Delete primary rows
    try:
        if edge_ids:
            engine.edge_collection.delete(ids=edge_ids)
        else:
            engine.edge_collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    try:
        if node_ids:
            engine.node_collection.delete(ids=node_ids)
        else:
            engine.node_collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    # Optional: document row (if you keep one)
    try:
        engine.document_collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    return {
        "ok": True,
        "doc_id": doc_id,
        "deleted": {"nodes": len(node_ids), "edges": len(edge_ids)},
    }


# mcp = FastMCP.from_fastapi(app=app)

# ---- Query I/O (same as before) ----
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

@mcp.tool()
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

@mcp.tool()
def kg_neighbors(rid: str, doc_id: Optional[str] = None) -> NeighborsOut:
    nb = gq.neighbors(rid, doc_id=doc_id)
    return NeighborsOut(nodes=sorted(nb["nodes"]), edges=sorted(nb["edges"]))

@mcp.tool()
def kg_k_hop(start_ids: List[str], k: int = 1, doc_id: Optional[str] = None) -> KHopOut:
    layers = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"]))
        for L in gq.k_hop(start_ids, k=k, doc_id=doc_id)
    ]
    return KHopOut(layers=layers)

@mcp.tool()
def kg_shortest_path(src_id: str, dst_id: str, doc_id: Optional[str] = None, max_depth: int = 8) -> ShortestPathOut:
    return ShortestPathOut(path=gq.shortest_path(src_id, dst_id, doc_id=doc_id, max_depth=max_depth))

@mcp.tool()
def kg_semantic_seed_then_expand_text(text: str, top_k: int = 5, hops: int = 1, doc_id: Optional[str] = None) -> SeedExpandOut:
    out = gq.semantic_seed_then_expand_text(text, top_k=top_k, hops=hops)
    layers = [KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"])) for L in out["layers"]]
    return SeedExpandOut(seeds=out["seeds"], layers=layers)

# ---- Ingestion tools ----
class DocParseIn(BaseModel):
    doc_id: str
    content: str
    type: str = "plain"

class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: List[str]
    summary_node_id: Optional[str]
from graph_knowledge_engine.ingester import PagewiseSummaryIngestor
@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    doc = Document(id=inp.doc_id, content=inp.content, type=inp.type)
    ingester = PagewiseSummaryIngestor(engine=engine, llm=engine.llm, cache_dir=str(os.path.join(".",".llm_cache")))
    res: dict = ingester.ingest_document(document = doc)
    # plug your real chunker/summary tree here and populate outputs
    return DocParseOut(doc_id=doc.id, chunk_ids=res.get("chunk_ids"), summary_node_id=res.get("final_node_id"))

class KGExtractIn(BaseModel):
    doc_id: str
    mode: str = "replace"  # "append" | "replace" | "skip-if-exists"

class KGExtractOut(BaseModel):
    doc_id: str
    node_ids: List[str]
    edge_ids: List[str]
    nodes_added: int
    edges_added: int

@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    content = engine._fetch_document_text(inp.doc_id)
    if not content:
        raise ValueError(f"Document '{inp.doc_id}' not found; run doc_parse first.")
    extracted = engine.extract_graph_with_llm(content=content)
    parsed = extracted["parsed"]
    engine._preflight_validate(parsed, inp.doc_id)
    persisted = engine.persist_graph_extraction(
        document=Document(id=inp.doc_id, content=content, type="plain"),
        parsed=parsed, mode=inp.mode,
    )
    return KGExtractOut(
        doc_id=inp.doc_id,
        node_ids=persisted["node_ids"],
        edge_ids=persisted["edge_ids"],
        nodes_added=persisted.get("nodes_added", len(persisted["node_ids"])),
        edges_added=persisted.get("edges_added", len(persisted["edge_ids"])),
    )


# Run with:
#   uvicorn server_mcp_with_admin:app --port 8765
