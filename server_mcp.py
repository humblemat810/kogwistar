# server_mcp.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
import os

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.engine_core.models import Document

# ---------- Engine + MCP ----------
persist_directory = os.environ.get("MCP_CHROMA_DIR") or "./.chroma-mcp"
engine = GraphKnowledgeEngine(persist_directory=persist_directory)
gq = GraphQuery(engine)
mcp = FastMCP("KnowledgeEngine")


# ---------- Query I/O models ----------
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


# ---------- Query tools ----------
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
def kg_shortest_path(
    src_id: str, dst_id: str, doc_id: Optional[str] = None, max_depth: int = 8
) -> ShortestPathOut:
    return ShortestPathOut(
        path=gq.shortest_path(src_id, dst_id, doc_id=doc_id, max_depth=max_depth)
    )


@mcp.tool(structured_output=True)
def kg_semantic_seed_then_expand_text(
    text: str, top_k: int = 5, hops: int = 1, doc_id: Optional[str] = None
) -> SeedExpandOut:
    # NOTE: current GraphQuery.semantic_seed_then_expand_text ignores doc_id.
    out = gq.semantic_seed_then_expand_text(text, top_k=top_k, hops=hops)
    layers = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"]))
        for L in out["layers"]
    ]
    return SeedExpandOut(seeds=out["seeds"], layers=layers)


# ========== Document parsing (text injection layer) ==========
class DocParseIn(BaseModel):
    doc_id: str
    content: str
    type: str = "text"


class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: List[str]
    summary_node_id: Optional[str]


@mcp.tool(structured_output=True)
def doc_parse(inp: DocParseIn) -> DocParseOut:
    """
    Parse/ingest raw text for a document and register it with the engine.
    (Placeholder for chunking/summary-tree; currently stores doc row only.)
    """
    doc = Document(id=inp.doc_id, content=inp.content, type=inp.type)
    engine.add_document(doc)
    # TODO: when your chunking + summary-tree pipeline is ready, call it here and populate outputs.
    return DocParseOut(doc_id=doc.id, chunk_ids=[], summary_node_id=None)


# ========== Graph extraction (from existing document text) ==========
class KGExtractIn(BaseModel):
    doc_id: str
    mode: str = "append"  # "append" | "replace" | "skip-if-exists"


class KGExtractOut(BaseModel):
    doc_id: str
    node_ids: List[str]
    edge_ids: List[str]
    nodes_added: int
    edges_added: int


@mcp.tool(structured_output=True)
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    """
    Run LLM extraction for the given doc_id and persist nodes/edges/endpoints.
    """
    # 1) fetch the document text already stored via doc_parse
    content = engine._fetch_document_text(inp.doc_id)
    if not content:
        raise ValueError(
            f"Document '{inp.doc_id}' not found or empty; run doc_parse first."
        )

    # 2) LLM extraction (pure; no writes)
    extracted = engine.extract_graph_with_llm(content=content)
    parsed = extracted["parsed"]

    # 3) Validate against this doc scope (resolve aliases, ids, endpoints)
    engine._preflight_validate(parsed, inp.doc_id)

    # 4) Persist
    persisted = engine.persist_graph_extraction(
        document=Document(id=inp.doc_id, content=content, type="text"),
        parsed=parsed,
        mode=inp.mode,
    )

    return KGExtractOut(
        doc_id=inp.doc_id,
        node_ids=persisted["node_ids"],
        edge_ids=persisted["edge_ids"],
        nodes_added=persisted.get("nodes_added", len(persisted["node_ids"])),
        edges_added=persisted.get("edges_added", len(persisted["edge_ids"])),
    )


# ---------- Entrypoints ----------
if __name__ == "__main__":
    # Recommended: streamable HTTP (works with uvicorn --factory via mcp.streamable_http_app)
    mcp.run(transport="streamable-http")
    # For stdio: mcp.run(transport="stdio")
