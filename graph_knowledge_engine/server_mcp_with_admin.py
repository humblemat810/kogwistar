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
        parsed = LLMGraphExtraction.FromLLMSlice(parsed_LLM, insertion_method = "llm_graph_extraction")
        batch_node_ids, batch_edge_ids = engine._preflight_validate(parsed, inp.id)
        return parsed
    parsed = get_reparsed_extraction(content)
    
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

#=====================
# Adjundicate persisted nodes
#=====================
from graph_knowledge_engine.models import AdjudicationQuestionCode, Node, Edge, AdjudicationVerdict
from typing import Literal, Tuple
class LoadPersistedIn(BaseModel):
    doc_ids: List[str]
    insertion_method: Optional[str] = None  # e.g. "graph_extractor", "api_upsert"

class LoadPersistedOut(BaseModel):
    node_ids: List[str]
    edge_ids: List[str]

def _load_persisted_graph(doc_ids: List[str], insertion_method: Optional[str] = None) -> LoadPersistedOut:
    """Return ID sets pulled from persistence (optionally filter by insertion_method)."""
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    for did in doc_ids:
        if insertion_method:
            node_ids.update(engine.nodes_by_doc(did, where={"insertion_method": insertion_method}))
            edge_ids.update(engine.edges_by_doc(did, where={"insertion_method": insertion_method}))
        else:
            node_ids.update(engine.node_ids_by_doc(did))
            edge_ids.update(engine.edge_ids_by_doc(did))
    return LoadPersistedOut(node_ids=sorted(node_ids), edge_ids=sorted(edge_ids))

@mcp.tool()
def kg_load_persisted(inp: LoadPersistedIn) -> LoadPersistedOut:
    """MCP: read back persisted IDs for the given docs (fast; no LLM)."""
    return _load_persisted_graph(inp.doc_ids, insertion_method=inp.insertion_method)

# ---------- Cross-document adjudication (same-kind & cross-kind) ----------
class CrossDocAdjIn(BaseModel):
    doc_ids: List[str]
    kind: Literal["node", "edge", "any"] = "any"   # "node" (node↔node only), "edge" (edge↔edge only), "any" (includes node↔edge)
    insertion_method: Optional[str] = None         # optional provenance filter
    max_pairs_per_bucket: int = 50                 # guardrail per logical bucket
    commit: bool = False                           # if True, commit positives
    scope: Literal["cross-doc", "within-doc"] = "cross-doc"  # NEW
    strict_crossdoc: bool = True                              # NEW: drop pairs if a side lacks doc_id

class CrossDocAdjItem(BaseModel):
    left: str
    right: str
    left_kind: Literal["entity", "relationship"]
    right_kind: Literal["entity", "relationship"]
    same_entity: Optional[bool]
    confidence: Optional[float] = None
    reason: Optional[str] = None
    canonical_id: Optional[str] = None            # set when commit succeeds

class CrossDocAdjOut(BaseModel):
    question_key: str
    total_pairs: int
    positives: int
    negatives: int
    abstain: int
    committed_ids: List[str]
    results: List[CrossDocAdjItem]


def _fetch_nodes(ids: List[str]) -> List[Node]:
    if hasattr(engine, "get_nodes"):
        return engine.get_nodes(ids)
    # fallback
    got = engine.node_collection.get(ids=ids, include=["documents"])
    return [Node.model_validate_json(j) for j in (got.get("documents") or [])]

def _fetch_edges(ids: List[str]) -> List[Edge]:
    if hasattr(engine, "get_edges"):
        return engine.get_edges(ids)
    # fallback
    got = engine.edge_collection.get(ids=ids, include=["documents"])
    return [Edge.model_validate_json(j) for j in (got.get("documents") or [])]

def _primary_doc_of(n: Node) -> Optional[str]:
    # prefer explicit .doc_id, else try references[].doc_id
    if getattr(n, "doc_id", None):
        return n.doc_id
    for r in (n.references or []):
        did = getattr(r, "doc_id", None)
        if did:
            return did
    return None

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _sigtext(n_or_e: Any) -> Optional[str]:
    """Signature text used to match reified node <-> edge or edge <-> edge."""
    props = getattr(n_or_e, "properties", None) or {}
    st = props.get("signature_text")
    return st if isinstance(st, str) else None


@mcp.tool()
def kg_crossdoc_adjudicate_anykind(inp: CrossDocAdjIn) -> CrossDocAdjOut:
    """
    Cross-document adjudication:
      - kind="node": node↔node
      - kind="edge": edge↔edge
      - kind="any" : node↔node, edge↔edge, and node↔edge (cross-type)

    Pairing heuristic (minimal / fast):
      • node↔node: bucket by (type, normalized label) across different doc_ids.
      • edge↔edge: match by properties.signature_text when present; otherwise (relation, normalized label).
      • node↔edge: match by properties.signature_text (reified event) when present; otherwise (normalized node.label == normalized edge.label).
    """
    def _pairable(di: Optional[str], dj: Optional[str]) -> bool:
        if inp.scope == "cross-doc":
            if inp.strict_crossdoc and (not di or not dj):
                return False
            return (di and dj and di != dj)
        else:  # within-doc
            if inp.strict_crossdoc and (not di or not dj):
                return False
            return (di and dj and di == dj)
    loaded = _load_persisted_graph(inp.doc_ids, insertion_method=inp.insertion_method)
    node_objs: List[Node] = _fetch_nodes(loaded.node_ids) if loaded.node_ids else []
    edge_objs: List[Edge] = _fetch_edges(loaded.edge_ids) if loaded.edge_ids else []

    # Build quick lookups
    nodes_by_key: Dict[Tuple[str, str], List[Tuple[Node, Optional[str]]]] = {}
    for n in node_objs:
        key = (n.type, _norm(n.label))
        nodes_by_key.setdefault(key, []).append((n, _primary_doc_of(n)))

    # For edges use signature_text when available; else fallback to (relation, normalized label)
    edges_by_sig: Dict[str, List[Tuple[Edge, Optional[str]]]] = {}
    edges_by_rel_label: Dict[Tuple[str, str], List[Tuple[Edge, Optional[str]]]] = {}
    for e in edge_objs:
        did = _primary_doc_of(e)  # same helper works (Edge ⊂ Node in your model)
        st = _sigtext(e)
        if st:
            edges_by_sig.setdefault(st, []).append((e, did))
        else:
            key = (e.relation or "", _norm(e.label))
            edges_by_rel_label.setdefault(key, []).append((e, did))

    # ---------- Build pairs ----------
    pairs: List[Tuple[Any, Any]] = []

    def _cap_pairing(items: List[Tuple[Any, Optional[str]]]):
        made = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                di, dj = items[i][1], items[j][1]
                if not di or not dj or di == dj:
                    continue  # cross-doc only
                if not _pairable(di, dj):
                    continue
                pairs.append((items[i][0], items[j][0]))
                made += 1
                if made >= inp.max_pairs_per_bucket:
                    return
    # simple adjudicate proposal by label
    if inp.kind in ("node", "any"):
        for _, items in nodes_by_key.items():
            if len(items) >= 2:
                _cap_pairing(items)

    if inp.kind in ("edge", "any"):
        for _, items in edges_by_sig.items():
            if len(items) >= 2:
                _cap_pairing(items)
        for _, items in edges_by_rel_label.items():
            if len(items) >= 2:
                _cap_pairing(items)

    if inp.kind == "any":
        # node↔edge (cross-type): use signature_text; else label match
        # map node signature_text to nodes
        nodes_by_sig: Dict[str, List[Tuple[Node, Optional[str]]]] = {}
        for n in node_objs:
            st = _sigtext(n)
            if st:
                nodes_by_sig.setdefault(st, []).append((n, _primary_doc_of(n)))

        # sig-based matches
        for st, n_items in nodes_by_sig.items():
            e_items = edges_by_sig.get(st) or []
            if not e_items or not n_items:
                continue
            made = 0
            for (n, dn) in n_items:
                for (e, de) in e_items:
                    if dn and de and dn != de:
                        pairs.append((n, e))
                        made += 1
                        if made >= inp.max_pairs_per_bucket:
                            break
                if made >= inp.max_pairs_per_bucket:
                    break

        # fallback: label match (rare but useful)
        edge_label_map: Dict[str, List[Tuple[Edge, Optional[str]]]] = {}
        for e in edge_objs:
            edge_label_map.setdefault(_norm(e.label), []).append((e, _primary_doc_of(e)))
        for n in node_objs:
            label_key = _norm(n.label)
            e_items = edge_label_map.get(label_key)
            if not e_items:
                continue
            made = 0
            dn = _primary_doc_of(n)
            for (e, de) in e_items:
                if dn and de and dn != de:
                    pairs.append((n, e))
                    made += 1
                    if made >= inp.max_pairs_per_bucket:
                        break

    if not pairs:
        return CrossDocAdjOut(
            question_key=str(AdjudicationQuestionCode.SAME_ENTITY.value),
            total_pairs=0, positives=0, negatives=0, abstain=0,
            committed_ids=[], results=[],
        )

    # ---------- Adjudicate with the engine's batch method ----------
    adjudications, qkey = engine.batch_adjudicate_merges(
        pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY
    )

    # ---------- Optionally commit positives ----------
    def _kind(o: Any) -> Literal["entity", "relationship"]:
        # Edge ⊂ Node in your model; distinguish by presence of .relation
        return "relationship" if isinstance(o, Edge) or getattr(o, "relation", None) else "entity"

    results: List[CrossDocAdjItem] = []
    pos = neg = abst = 0
    committed: List[str] = []

    for (left, right), out in zip(pairs, adjudications):
        verdict: AdjudicationVerdict = getattr(out, "verdict", out)

        lkind = _kind(left)
        rkind = _kind(right)

        if verdict.same_entity is True:
            pos += 1
            canonical_id = None
            if inp.commit:
                # same-kind → commit_merge; cross-type → commit_any_kind (node↔edge “reifies” style)
                if lkind == rkind:
                    canonical_id = engine.commit_merge(left, right, verdict)
                else:
                    # Requires your engine to expose commit_any_kind(Node|Edge, Node|Edge, verdict)
                    canonical_id = engine.commit_any_kind(left, right, verdict)
                if canonical_id:
                    committed.append(str(canonical_id))
            results.append(CrossDocAdjItem(
                left=left.id, right=right.id,
                left_kind=lkind, right_kind=rkind,
                same_entity=True, confidence=verdict.confidence,
                reason=verdict.reason, canonical_id=canonical_id
            ))
        elif verdict.same_entity is False:
            neg += 1
            results.append(CrossDocAdjItem(
                left=left.id, right=right.id,
                left_kind=lkind, right_kind=rkind,
                same_entity=False, confidence=verdict.confidence,
                reason=verdict.reason
            ))
        else:
            abst += 1
            results.append(CrossDocAdjItem(
                left=left.id, right=right.id,
                left_kind=lkind, right_kind=rkind,
                same_entity=None, confidence=verdict.confidence,
                reason=verdict.reason
            ))

    return CrossDocAdjOut(
        question_key=qkey,
        total_pairs=len(pairs),
        positives=pos,
        negatives=neg,
        abstain=abst,
        committed_ids=committed,
        results=results,
    )
    
class PairDTO(BaseModel):
    left_id: str
    left_kind: Literal["node", "edge"]
    left_label: str | None = None
    right_id: str
    right_kind: Literal["node", "edge"]
    right_label: str | None = None

class ProposePairsOut(BaseModel):
    pairs: list[PairDTO]
    
def _load_node(engine: GraphKnowledgeEngine, nid: str) -> Node | None:
    got = engine.node_collection.get(ids=[nid], include=["documents"])
    docs = got.get("documents") or []
    if not docs:
        return None
    try:
        return Node.model_validate_json(docs[0])
    except Exception:
        return None

def _label_of(obj: Node | Edge | None) -> str | None:
    if not obj:
        return None
    return getattr(obj, "label", None) or getattr(obj, "summary", None)

@mcp.tool()
def kg_propose_vector(
    new_node_ids: list[str],
    top_k: int = 12,
    # doc scoping
    allowed_docs: list[str] | None = None,
    anchor_doc_id: str | None = None,
    cross_doc_only: bool = False,
    anchor_only: bool = True,
    # vector thresholds
    score_mode: Literal["distance", "similarity"] = "distance",
    max_distance: float = 0.25,   # used when score_mode="distance"
    min_similarity: float = 0.85, # used when score_mode="similarity"
    include_edges: bool = True,
) -> ProposePairsOut:
    """
    Batch vector search over the graph for the given node IDs.
    Returns (query_node, matched_entity) pairs (entity can be a Node or an Edge).
    """
    if not new_node_ids:
        return ProposePairsOut(pairs=[])

    # Fetch concrete Node objects for the query set
    q_nodes: list[Node] = []
    for nid in new_node_ids:
        n = _load_node(engine, nid)
        if n and getattr(n, "embedding", None) is not None:
            q_nodes.append(n)

    if not q_nodes:
        return ProposePairsOut(pairs=[])

    proposer = engine.proposer
    pairs = proposer.generate_merge_candidates(
        engine=engine,
        new_node=q_nodes,                # batch
        top_k=top_k,
        allowed_docs=allowed_docs,
        anchor_doc_id=anchor_doc_id,
        cross_doc_only=cross_doc_only,
        anchor_only=anchor_only,
        score_mode=score_mode,
        max_distance=max_distance,
        min_similarity=min_similarity,
        include_edges=include_edges,
    )

    out: list[PairDTO] = []
    for q, m in pairs:
        right_kind = "edge" if isinstance(m, Edge) else "node"
        out.append(
            PairDTO(
                left_id=q.id, left_kind="node", left_label=_label_of(q),
                right_id=m.id, right_kind=right_kind, right_label=_label_of(m),
            )
        )
    return ProposePairsOut(pairs=out)
@mcp.tool()
def kg_propose_bruteforce(
    pair_kind: Literal["node_node", "edge_edge", "node_edge"] = "node_node",
    allowed_docs: list[str] | None = None,
    anchor_doc_id: str | None = None,
    cross_doc_only: bool = False,
    anchor_only: bool = True,
    limit_per_bucket: int = 200,
) -> ProposePairsOut:
    """
    Heuristic/brute-force candidate generation across the graph with doc scoping.
    - pair_kind chooses Node↔Node, Edge↔Edge, or Node↔Edge.
    - If anchor_doc_id is provided and anchor_only=True, at least one side must be from the anchor doc.
    - If cross_doc_only=True, pairs from the same doc are filtered out.
    """
    proposer = engine.proposer
    pairs = proposer.propose_any_kind_any_doc(
        engine=engine,
        pair_kind=pair_kind,
        allowed_docs=allowed_docs,
        anchor_doc_id=anchor_doc_id,
        cross_doc_only=cross_doc_only,
        anchor_only=anchor_only,
        limit_per_bucket=limit_per_bucket,
    )

    out: list[PairDTO] = []
    for l, r in pairs:
        lk = "edge" if isinstance(l, Edge) else "node"
        rk = "edge" if isinstance(r, Edge) else "node"
        out.append(
            PairDTO(
                left_id=l.id, left_kind=lk, left_label=_label_of(l),
                right_id=r.id, right_kind=rk, right_label=_label_of(r),
            )
        )
    return ProposePairsOut(pairs=out)

"""
Quick usage notes

Vector (best for “given these N fresh nodes, what’s similar anywhere in the graph?”):

// MCP tool: kg_propose_vector
{
  "new_node_ids": ["N123", "N456"],
  "top_k": 12,
  "allowed_docs": ["D1", "D2"],
  "anchor_doc_id": "D1",
  "cross_doc_only": true,
  "anchor_only": true,
  "score_mode": "distance",
  "max_distance": 0.22,
  "include_edges": true
}


Brute-force / heuristic (good for cross-doc sweeps):

// MCP tool: kg_propose_bruteforce
{
  "pair_kind": "node_edge",
  "allowed_docs": ["D1", "D2", "D3"],
  "anchor_doc_id": "D1",
  "cross_doc_only": true,
  "anchor_only": true,
  "limit_per_bucket": 300
}


Both tools return:

{
  "pairs": [
    {
      "left_id": "…",
      "left_kind": "node",
      "left_label": "…",
      "right_id": "…",
      "right_kind": "edge",
      "right_label": "…"
    }
  ]
}    
"""

# Mount the MCP server
app.mount("/", mcp_app)

# Run with:
#   uvicorn server_mcp_with_admin:app --port 8765
