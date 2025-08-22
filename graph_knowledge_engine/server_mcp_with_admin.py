from __future__ import annotations
# server_mcp_with_admin.py

from typing import List, Optional
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.models import Document
from graph_knowledge_engine.visualization.graph_viz import to_cytoscape, to_d3_force
import json
from fastapi.templating import Jinja2Templates
from fastapi import Request
# ---- Engine + MCP ----
persist_directory = os.environ.get("MCP_CHROMA_DIR") or "./.chroma-mcp"
engine = GraphKnowledgeEngine(persist_directory=persist_directory)
gq = GraphQuery(engine)
import pathlib
templates = Jinja2Templates(directory=os.path.join(str(pathlib.Path(__file__).parent),"templates"))
# Fastapi

mcp = FastMCP("KnowledgeEngine + MCP + Admin")

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
from .models import Document
class DocParseIn(Document['dto']):
    id: Optional[str]
    content: str
    type: str = "plain"

class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: List[str]
    summary_node_id: Optional[str]
from graph_knowledge_engine.ingester import PagewiseSummaryIngestor
@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    ingester = PagewiseSummaryIngestor(engine=engine, llm=engine.llm, cache_dir=str(os.path.join(".",".llm_cache")))
    res: dict = ingester.ingest_document(document = doc)
    # plug your real chunker/summary tree here and populate outputs
    return DocParseOut(doc_id=doc.id, chunk_ids=res.get("chunk_ids"), summary_node_id=res.get("final_node_id"))

class KGExtractIn(BaseModel):
    id: Optional[str]
    mode: str = "skip-if-exists"  # "append" | "replace" | "skip-if-exists"

class KGExtractOut(BaseModel):
    doc_id: str
    node_ids: List[str]
    edge_ids: List[str]
    nodes_added: int
    edges_added: int
class DocStoreOut(BaseModel):
    success: bool
    
@mcp.tool()
def store_document(inp: DocParseIn):
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    engine.add_document(doc)
    return DocStoreOut(**{"success": True})
@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    content = engine._fetch_document_text(inp.id)
    if not content:
        raise ValueError(f"Document '{inp.id}' not found; run doc_parse first.")
    from .models import LLMGraphExtraction
    # import pickle, os
    # cdir = os.path.join('.', '.kg_extract_cache')
    # os.makedirs(cdir, exist_ok = True)
    # if inp.id in os.listdir(cdir):
    #     res = pickle.load(os.path.join(cdir, inp.id))
        
    # extract_graph_with_llm
    # extracted: LLMGraphExtraction = engine.extract_graph_with_llm(content=content)
    from joblib import Memory
    location = os.path.join(".", '.kg_extract')
    os.makedirs(location, exist_ok = True)
    memory = Memory(location = location)
    @memory.cache()
    def get_reparsed_extraction(content):
        extracted = engine._cached_extract_graph_with_llm(content=content)
        # parsed = extracted["parsed"]
        # if not isinstance(parsed, LLMGraphExtraction):
        #     dumped = parsed.model_dump(field_mode = 'backend')
        parsed_LLM: LLMGraphExtraction['llm'] = extracted["parsed"]
        # ctx = {"insertion_method": "graph_extractor"}
        # dumped = parsed_LLM.model_dump()
        parsed = LLMGraphExtraction.FromLLMSlice(parsed_LLM, insertion_method = "graph_extractor")
        return parsed
    parsed = get_reparsed_extraction(content)
    batch_node_ids, batch_edge_ids = engine._preflight_validate(parsed, inp.id)
    persisted = engine.persist_graph_extraction(
        document=Document(id=inp.id, content=content, type="plain"),
        parsed=parsed, mode=inp.mode,
    )
    return KGExtractOut(
        doc_id=inp.id,
        node_ids=persisted["node_ids"],
        edge_ids=persisted["edge_ids"],
        nodes_added=persisted.get("nodes_added", len(persisted["node_ids"])),
        edges_added=persisted.get("edges_added", len(persisted["edge_ids"])),
    )

class CytoscapeOut(BaseModel):
    elements: List[dict]
    mode: str
    doc_id: Optional[str]

class D3Out(BaseModel):
    nodes: List[dict]
    links: List[dict]
    mode: str
    doc_id: Optional[str]

@mcp.tool()
def kg_viz_cytoscape_json(doc_id: Optional[str] = None, mode: str = "reify") -> CytoscapeOut:
    payload = to_cytoscape(engine, doc_id=doc_id, mode=mode)
    return CytoscapeOut(**payload)

@mcp.tool()
def kg_viz_d3_json(doc_id: Optional[str] = None, mode: str = "reify") -> D3Out:
    payload = to_d3_force(engine, doc_id=doc_id, mode=mode)
    return D3Out(**payload)



# ---- Build a unified FastAPI app: /mcp + /admin ----
mcp_app = mcp.http_app(path='/mcp')
app = FastAPI(title="KnowledgeEngine + MCP + Admin", lifespan=mcp_app.lifespan)



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
# http://localhost:28110/api/viz/d3.json?doc_id=pytest-doc-upsert-1&mode=reify
# http://localhost:28110/api/viz/d3.json?doc_id=&mode=reify
@app.get("/api/viz/cytoscape.json")
def api_viz_cytoscape(
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,   # NEW
):
    payload = to_cytoscape(engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method)
    return JSONResponse(payload)

@app.get("/api/viz/d3.json")
def api_viz_d3(
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,   # NEW
):
    payload = to_d3_force(engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method)
    return JSONResponse(payload)
# --- Add near your other imports ---
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from graph_knowledge_engine.models import LLMGraphExtraction, Document

class GraphUpsertLLMIn(BaseModel):
    """
    Strict LLM-conformant upsert:
    - nodes / edges must follow LLM* models (including REQUIRED references with spans).
    - endpoints may use 'nn:*' / 'ne:*' temp ids that resolve in-batch.
    - references may use document alias token ::DOC:: in URLs; we’ll de-alias to doc_id.
    """
    doc_id: str = Field(..., description="Document id to scope persistence")
    content: Optional[str] = Field(None, description="If provided and doc is new, store this as document content")
    doc_type: str = Field("plain", description="Document type")
    insertion_method: str = Field("api_upsert", description="Provenance tag copied into each ReferenceSession")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="LLMNode['llm']-shaped dicts")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="LLMEdge['llm']-shaped dicts")

class GraphUpsertOut(BaseModel):
    document_id: str
    node_ids: List[str]
    edge_ids: List[str]
    nodes_added: int
    edges_added: int

@app.post("/api/graph/upsert", response_model=GraphUpsertOut)
def api_graph_upsert_llm(inp: GraphUpsertLLMIn):
    """
    Upsert a (hyper)graph in one shot:
    - Validates against the LLM Graph schema (which REQUIRES non-empty 'references').
    - Injects `insertion_method` into every ReferenceSession.
    - Supports hyperedges (edge->edge) in a single batch via engine’s topo-sorted ingest.
    """
    # 1) Ensure the document row exists (idempotent)
    if inp.content is not None:
        engine.add_document(Document(id=inp.doc_id, content=inp.content, type=inp.doc_type))
    else:
        # create a placeholder if completely missing so refs can anchor to doc_id
        if not engine._fetch_document_text(inp.doc_id):
            engine.add_document(Document(id=inp.doc_id, content="", type=inp.doc_type))

    # 2) Validate to your LLM models (references REQUIRED by GraphEntityBase)
    #    This matches your extractor path which later calls FromLLMSlice.
    llm_like = LLMGraphExtraction.model_validate({
        "nodes": inp.nodes,
        "edges": inp.edges,
    })

    # 3) Copy insertion_method into every ReferenceSession (backend-only field),
    #    exactly like kg_extract does, so refs carry provenance.
    parsed = LLMGraphExtraction.FromLLMSlice(llm_like, insertion_method=inp.insertion_method)

    # 4) Persist using your first-class persistence path (allocates nn:/ne:, topo-sorts, enforces endpoints)
    persisted = engine.persist_graph_extraction(
        document=Document(id=inp.doc_id, content=inp.content or engine._fetch_document_text(inp.doc_id) or "", type=inp.doc_type),
        parsed=parsed,
        mode="append",
    )

    return GraphUpsertOut(
        document_id=persisted["document_id"],
        node_ids=persisted["node_ids"],
        edge_ids=persisted["edge_ids"],
        nodes_added=persisted["nodes_added"],
        edges_added=persisted["edges_added"],
    )

@app.get("/viz/cytoscape", response_class=HTMLResponse)
def viz_cytoscape(
    request: Request,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
):
    return templates.TemplateResponse(
        "cytoscape.html",
        {"request": request, "doc_id": doc_id, "mode": mode, "insertion_method": insertion_method},
    )

@app.get("/viz/d3", response_class=HTMLResponse)
def viz_d3(
    request: Request,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
):
    return templates.TemplateResponse(
        "d3.html",
        {"request": request, "doc_id": doc_id, "mode": mode, "insertion_method": insertion_method},
    )
    
# Mount the MCP server
app.mount("/", mcp_app)

# Run with:
#   uvicorn server_mcp_with_admin:app --port 8765
