from __future__ import annotations
# server_mcp_with_admin.py
import contextvars
import functools
from contextvars import ContextVar
from regex import P
from starlette.types import Scope, Receive, Send
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.models import Document
from graph_knowledge_engine.visualization.graph_viz import to_cytoscape, to_d3_force
from fastapi.templating import Jinja2Templates
from fastapi import Request
from enum import Enum
from jose import jwt, JWTError
import json
import os
import pathlib
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import List, Optional
from typing import Any, Dict, Set
import uuid
from graph_knowledge_engine.shortids import run_id_ctx, run_id_scope
from graph_knowledge_engine import shortids
# --- JWT config (env-driven) ---
JWT_ALG = os.getenv("JWT_ALG", "HS256")          # HS256 (shared secret) or RS256 (public key)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")  # HS256 secret OR RS256 public key
JWT_ISS = os.getenv("JWT_ISS")                   # optional issuer to check
JWT_AUD = os.getenv("JWT_AUD")                   # optional audience to check
PROTECTED_PREFIXES = tuple(
    (os.getenv("JWT_PROTECTED_PATHS") or "/mcp,/admin")
    .split(",")
)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG    = os.getenv("JWT_ALG", "HS256")

class Role(str, Enum):
    RO = "ro"
    RW = "rw"
class NameSpace(str, Enum):
    DOCS = "docs"
    WISDOM = "wisdom"
# tool name -> allowed roles
TOOL_ROLES: dict[str, set[str]] = {}
TOOL_NAMESPACE: dict[str, set[str]] = {}
def tool_roles(roles: set[Role] | Role):
    """Annotate a tool with allowed roles (e.g. {Role.RO, Role.RW} or Role.RW)."""
    allowed = {roles} if isinstance(roles, Role) else set(roles)
    def deco(fn: FunctionTool):
        name = getattr(fn, "name", None) or fn.name
        # we record by function name here; FastMCP uses that as tool name by default
        TOOL_ROLES[name] = {r.value for r in allowed}
        original_fn = fn.fn
        @functools.wraps(fn.fn)
        def wrapper(*args, **kwargs):
            user_role = get_current_role()
            if user_role not in allowed:
                raise HTTPException(status_code=403, detail=f"Forbidden: role {user_role} not permitted call this tool")
            # if ROLE_ORDER.get(user_role, 0) < ROLE_ORDER.get(min_role, 0):
            #     raise HTTPException(status_code=403, detail=f"Forbidden: requires role '{min_role}', you have '{user_role}'")
            return original_fn(*args, **kwargs)
        fn.fn = wrapper
        return fn
    

    
    return deco


def _decode_role_from_headers(scope: Scope) -> str:
    try:
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return Role.RO.value
        token = auth.split(" ", 1)[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        role = payload.get("role", Role.RO.value)
        return role if role in (Role.RO.value, Role.RW.value) else Role.RO.value
    except Exception:
        return Role.RO.value
# ---- Engine + MCP ----
persist_directory = os.environ.get("MCP_CHROMA_DIR") or "./.chroma-mcp"
pathparts = list(pathlib.Path(os.environ.get("MCP_CHROMA_DIR")).parts)
pathparts.insert(-1, 'index')
index_dir = os.path.join(*pathparts) or "./index/chroma-mcp"
os.makedirs(index_dir, exist_ok = True)
index_db_path = os.path.join(*(pathparts + ['index.db']))

engine = GraphKnowledgeEngine(persist_directory=persist_directory)
gq = GraphQuery(engine)
# ---- Wisdom + MCP ----
wisdom_persist_directory = os.environ.get("MCP_CHROMA_DIR_WISDOM") or (persist_directory + "-wisdom")
wisdom_engine = GraphKnowledgeEngine(persist_directory=wisdom_persist_directory)
wisdom_gq = GraphQuery(wisdom_engine)
templates = Jinja2Templates(directory=os.path.join(str(pathlib.Path(__file__).parent),"templates"))

# Fastapi
def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()

def verify_jwt(token: str) -> dict:
    """
    Validates a JWT and returns claims. Works for HS256 or RS256 depending on env.
    - HS256: set JWT_ALG=HS256 and JWT_SECRET=<shared secret>
    - RS256: set JWT_ALG=RS256 and JWT_SECRET=<PEM public key string>
    Optionally set JWT_ISS and/or JWT_AUD for issuer/audience checks.
    """
    try:
        options = {"verify_aud": bool(JWT_AUD)}
        claims = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALG],
            audience=JWT_AUD,
            issuer=JWT_ISS,
            options=options,
        )
        return claims
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
class MCPRoleMiddleware:
    """
    - Sets ContextVar current_role per request (so tools can read it).
    - If JSON-RPC result contains a 'tools' list, filter by TOOL_ROLES & current_role.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope.get("type") != "http":  # pass-through
            return await self.app(scope, receive, send)

        # set role for this request
        token = current_role.set(_decode_role_from_headers(scope))

        # We need to possibly rewrite the JSON response body -> buffer
        started = {}
        body_chunks: list[bytes] = []

        async def _send(message):
            if message["type"] == "http.response.start":
                # stash headers/status; we may need to re-send after filtering
                started["message"] = message
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"") or b""
                more = message.get("more_body", False)
                body_chunks.append(chunk)
                if not more:
                    # combine + maybe filter
                    raw = b"".join(body_chunks)
                    if raw.startswith(b'event: message\r\n'):
                        raw_lines = raw.split(b'\r\n')
                        try:
                            _, json_str = raw_lines[1].decode("utf-8").split('data: ')
                            prefix = "data: "
                            data = json.loads(json_str)
                            # If it looks like an MCP tools/list response, filter it
                            

                            changed = False

                            # JSON-RPC result style: {"jsonrpc":"2.0","id":...,"result":{"tools":[...]}}
                            res = data.get("result")
                            if isinstance(res, dict) and isinstance(res.get("tools"), list):
                                res["tools"] = _filter_tool_list(res["tools"])
                                changed = True

                            # Plain payload style: {"tools":[...]}
                            elif isinstance(data.get("tools"), list):
                                data["tools"] = _filter_tool_list(data["tools"])
                                changed = True

                            if changed:
                                raw_lines[1] = (prefix + json.dumps(data)).encode("utf-8")
                                # raw = json.dumps(data).encode("utf-8")
                                raw = b"\r\n".join(raw_lines)
                        except Exception:
                            pass  # non-JSON or unexpected shape; return as-is

                    # now actually send start + full body
                    await send(started.get("message", {"type": "http.response.start", "status": 200, "headers": []}))
                    await send({"type": "http.response.body", "body": raw, "more_body": False})
                # else: keep buffering until more_body=False
            else:
                await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            current_role.reset(token)
from fastmcp.tools.tool import FunctionTool
def _filter_tool_list(lst: list[dict]) -> list[dict]:
    role = current_role.get()
    namespace = get_current_namespace()
    out = []
    for item in lst:
        name = getattr(item, 'name', None) if type(item) is FunctionTool else None or item.get("name") or item.get("tool") or ""
        # accept either exact name or any recorded alias
        tool_roles = TOOL_ROLES.get(name, {Role.RO.value})
        tool_namespace = TOOL_NAMESPACE.get(name, {NameSpace.DOCS.value})
        if role in tool_roles and namespace in tool_namespace:
            out.append(item)
    return out

from fastmcp.tools.tool_manager import ToolManager
import inspect
from functools import wraps
from typing import Type
def patch_toolmanager_list_tools(ToolManager: Type):
    """
    Monkey-patch ToolManager.list_tools to post-filter its result based on policy_ctx.
    Works whether the original is sync or async.
    """
    orig = getattr(ToolManager, "list_tools")

    # Detect if the original is defined as coroutine
    is_async = inspect.iscoroutinefunction(orig)

    if is_async:
        @wraps(orig)
        async def wrapped(self, *args, **kwargs): # type: ignore
            res = await orig(self, *args, **kwargs)
            return _filter_tool_list(res)
    else:
        @wraps(orig)
        def wrapped(self, *args, **kwargs):
            res = orig(self, *args, **kwargs)
            return _filter_tool_list(res)

    # Bind as an instance method on the class
    setattr(ToolManager, "list_tools", wrapped)
patch_toolmanager_list_tools(ToolManager)
class JWTProtectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if claims_ctx.get() is None:
            path = request.url.path
            if any(path.startswith(p) for p in PROTECTED_PREFIXES):
                token = _extract_bearer(request)
                if not token:
                    return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
                try:
                    claims = verify_jwt(token)
                    request.state.claims = claims
                    token_ = claims_ctx.set(claims)
                    try:
                        with run_id_scope(token):
                            return await call_next(request)
                    finally:
                        claims_ctx.reset(token_)
                except HTTPException as e:
                    return JSONResponse({"detail": e.detail}, status_code=e.status_code)
        return await call_next(request)


def get_current_role() -> str:
    claims = claims_ctx.get() or {}
    return (claims.get("role") or DEFAULT_ROLE).lower()

def require_role(min_role: str = "ro"):
    user_role = get_current_role()
    if ROLE_ORDER.get(user_role, 0) < ROLE_ORDER.get(min_role, 0):
        raise HTTPException(status_code=403, detail=f"Forbidden: requires role '{min_role}', you have '{user_role}'")

# Context var to expose claims in any handler/tool
claims_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("claims", default=None)
current_role: ContextVar[str] = ContextVar("current_role", default=Role.RO.value)

# Simple two-role lattice
ROLE_ORDER = {"ro": 0, "rw": 1}  # read-only < read-write
DEFAULT_ROLE = "ro"


def get_current_namespace() -> str:
    claims = claims_ctx.get() or {}
    ns = (claims.get("ns") or "docs").lower()
    return "wisdom" if ns == "wisdom" else "docs"
from fastmcp.tools import FunctionTool
def require_ns(expected: set[NameSpace] | NameSpace):
    # only wrap fast mcp function tool
    if type(expected) is Set:
        if len(expected) == 0:
            raise ValueError("At least 1 name space has to be specified")
        if not all( i in NameSpace for i in expected):
            raise ValueError("expected not in NameSpace")
        expected2 = expected
    if expected in NameSpace or type(expected) is str:
        expected2 = {expected}
    
    def deco(fn: FunctionTool):
        # allowed : set[NameSpace | str] = expected2
        allowed = expected2
        name = getattr(fn, "name", None) or fn.name
        # we record by function name here; FastMCP uses that as tool name by default
        TOOL_NAMESPACE[name] = allowed
        original_fn = fn.fn
        @functools.wraps(fn.fn)
        def wrapper(*args, **kwargs):
            actual = get_current_namespace()
            if actual != expected:
                # MCP tools throw regular exceptions; FastMCP wraps as tool error
                raise HTTPException(status_code=403, detail=f"Forbidden: namespace '{actual}' cannot call this tool")
                
            return original_fn(*args, **kwargs)
        fn.fn = wrapper
        return fn
        
    return deco
    #     @functools.wraps(fn)
    #     def wrapper(*args, **kwargs):
    #         actual = get_current_namespace()
    #         if actual != expected:
    #             # MCP tools throw regular exceptions; FastMCP wraps as tool error
    #             raise PermissionError(f"Forbidden: namespace '{actual}' cannot call this tool (expected '{expected}').")
    #         return fn(*args, **kwargs)
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
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_find_edges(
    relation: Optional[str] = None,
    src_label_contains: Optional[str] = None,
    tgt_label_contains: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> FindEdgesOut:
    """Find an edge given filter of exact value of relation/ source label/ target label/ document id"""
    eids = gq.find_edges(
        relation=relation,
        src_label_contains=src_label_contains,
        tgt_label_contains=tgt_label_contains,
        doc_id=doc_id,
    )
    return FindEdgesOut(edges=eids)
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_neighbors(rid: str, doc_id: Optional[str] = None) -> NeighborsOut:
    """find all direct neighbour given id of a node or an edge, optionally filtered by document id
    - rid: string of node or edge id to find its neigbour
    """
    nb = gq.neighbors(rid, doc_id=doc_id)
    return NeighborsOut(nodes=sorted(nb["nodes"]), edges=sorted(nb["edges"]))
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_k_hop(start_ids: List[str], k: int = 1, doc_id: Optional[str] = None) -> KHopOut:
    """Run k-hop algorithm to obtain node edge relationship and its neighbour. Use when you have a node or edge"""
    layers = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"]))
        for L in gq.k_hop(start_ids, k=k, doc_id=doc_id)
    ]
    return KHopOut(layers=layers)
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_shortest_path(src_id: str, dst_id: str, doc_id: Optional[str] = None, max_depth: int = 8) -> ShortestPathOut:
    """Shortest path from source id to target id (both can be node or edges as it is hypergraph.)"""
    return ShortestPathOut(path=gq.shortest_path(src_id, dst_id, doc_id=doc_id, max_depth=max_depth))
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_semantic_seed_then_expand_text(text: str, top_k: int = 5, hops: int = 1, doc_ids: Optional[str] = None) -> SeedExpandOut:
    """
    set doc_ids to None/null if you have file name but not the exact id.
    Simple semantic search with optional graph search. [possible entrypoint]
    set hops to 0 to reduce to simple vector semantic index search.
    hops > 0 will return graph neighbours
    set top k to limit the number of neighbour
    """
    doc_ids_coerced = None
    if doc_ids:
        doc_ids_coerced: str | list[str] = shortids.s2l_id(doc_ids) if type(doc_ids is str) else [shortids.s2l_id(i) for i in doc_ids]
    out = gq.semantic_seed_then_expand_text(text, top_k=top_k, hops=hops, doc_ids = doc_ids_coerced)
    layers= [{"nodes": [shortids.l2s_doc(n) for n in L['nodes']], "edges": [shortids.l2s_doc(n) for n in L['edges']]} for L in out["layers"]]
    layers2 = [KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"])) for L in layers]
    
    outsid = SeedExpandOut(seeds = [shortids.l2s_doc(i) for i in out["seeds"]], 
                                            layers= layers2
                                            )
    return outsid

# ---- Ingestion tools ----
from .models import Document, PureGraph
class DocParseIn(Document['dto']):
    id: Optional[str]
    content: str
    type: str = "plain"

class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: List[str]
    summary_node_id: Optional[str]
from graph_knowledge_engine.ingester import PagewiseSummaryIngestor

@tool_roles({Role.RW})
@require_ns("docs")
@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    """Parse a document into leaf and relationships between chunks with summaries from low to high abstraction levels."""
    require_role("rw")
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    ingester = PagewiseSummaryIngestor(engine=engine, llm=engine.llm, 
                                       cache_dir=str(os.path.join(".",".llm_cache"))
                                       )
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

class DocIdsOut(BaseModel):
    id_mapping: list[dict[str, str]]
    
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def document_id_from_file_name(file_name: str):
    """get document id given filename, [possible entrypoint]"""
    
    # """return document_id from document name, temporarily, filename is actually doc id but this layer may subject to changes"""
    if type(file_name) is str:
        filenames = [file_name]
    elif type(file_name) is list:
        filenames = file_name
        
    docs = engine.document_collection.get(ids = filenames) # this is only where we do not use shortid
    sids = [shortids.l2s_id(i) for i in docs['ids']]
    to_return = [{"file_name": i, "id": shortids.l2s_id(i)} for i in docs['ids']]
    return DocIdsOut(id_mapping = to_return)



@tool_roles({Role.RW})
@require_ns("docs")
@mcp.tool()
def store_document(inp: DocParseIn):
    """Store document in graph database"""
    require_role("rw")
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    engine.add_document(doc)
    return DocStoreOut.model_validate({"success": True})
@tool_roles({Role.RW})
@require_ns("docs")
@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    """From documents extract knowledge and relationships as a hypergraph between entities, ideas, concepts with each other."""
    require_role("rw")
    content = engine._fetch_document_text(inp.id)
    if not content:
        raise ValueError(f"Document '{inp.id}' not found; run store_document first.")
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
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_viz_cytoscape_json(doc_id: Optional[str] = None, mode: str = "reify") -> CytoscapeOut:
    """get cytoscape format json data for visual rendering"""
    payload = to_cytoscape(engine, doc_id=doc_id, mode=mode)
    return CytoscapeOut.model_validate(payload)
@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_viz_d3_json(doc_id: Optional[str] = None, mode: str = "reify") -> D3Out:
    """get d3 format json data for visual rendering"""
    payload = to_d3_force(engine, doc_id=doc_id, mode=mode)
    return D3Out.model_validate(payload)



# ---- Build a unified FastAPI app: /mcp + /admin ----
mcp_app = mcp.http_app(path='/mcp')
app = FastAPI(title="KnowledgeEngine + MCP + Admin", lifespan=mcp_app.lifespan)
app.add_middleware(MCPRoleMiddleware)
app.add_middleware(JWTProtectMiddleware)
from datetime import datetime, timedelta, timezone

class DevTokenInp(BaseModel):
    username: str = "dev"
    role: str = "ro"
    ns : Literal["docs", "wisdom"]= "docs"
@app.post("/auth/dev-token")
async def dev_token(request: Request):
    inp = DevTokenInp.model_validate((await request.json()))
    if inp.role not in ROLE_ORDER:
        raise HTTPException(400, f"role must be one of {list(ROLE_ORDER)}")
    payload = {
        "sub": inp.username,
        "ns" : inp.ns,
        "role": inp.role,
        "iat": int(time.time()),
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=4)).timestamp()),
        "iss": JWT_ISS or "local",
        "aud": JWT_AUD or None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token}

# health
@app.get("/health")
def health():
    return {"ok": True, "persist_directory": persist_directory}

# DELETE /admin/doc/{doc_id}  (non-MCP utility)
@app.delete("/admin/doc/{doc_id}")
def admin_delete_doc(doc_id: str):
    require_role("rw")
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
            engine.edge_refs_collection.delete(ids=edge_ids)
            engine.edge_collection.delete(ids=edge_ids)
        else:
            engine.edge_collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    try:
        if node_ids:
            engine.node_refs_collection.delete(ids=edge_ids)
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
class DocumentGraphProposal(BaseModel):
    doc_id: str
    insertion_method: str = "document_parser_v1"
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]] = []

class DocumentGraphValidationResult(BaseModel):
    ok: bool
    node_errors: Dict[str, str] = {}
    edge_errors: Dict[str, str] = {}
from pydantic import ValidationError
@app.post("/api/contract.validate_graph", response_model=DocumentGraphValidationResult)
async def contract_validate_graph(payload: DocumentGraphProposal):
    # inp =await request.json()
    # payload = ContractGraphProposal.model_validate(inp['payload'])
    node_errors: Dict[str, str] = {}
    edge_errors: Dict[str, str] = {}

    # try to coerce nodes
    for n in payload.nodes:
        nid = n.get("id") or n.get("label") or "unknown"
        try:
            # IMPORTANT: your Node requires references;
            # user may send "metadata.pointers" instead.
            # so we do a small shim here:
            if "references" not in n:
                # try to lift from metadata.pointers -> references
                md = n.get("metadata") or {}
                ptrs = md.get("pointers") or []
                if ptrs:
                    n["references"] = [
                        {
                            "doc_id": payload.doc_id,
                            "collection_page_url": f"doc://{payload.doc_id}",
                            "document_page_url": f"doc://{payload.doc_id}#{p['source_cluster_id']}",
                            "insertion_method": payload.insertion_method,
                            "start_page": 1,
                            "end_page": 1,
                            "start_char": p["start_char"],
                            "end_char": (10**9 if p["end_char"] == -1 else p["end_char"]),
                            "snippet": p.get("verbatim_text", "")[:400],
                        }
                        for p in ptrs
                    ]
            Node.model_validate(n)
        except ValidationError as e:
            node_errors[str(nid)] = e.json()

    # try to coerce edges
    for e in payload.edges:
        eid = e.get("id") or "unknown-edge"
        try:
            Edge.model_validate(e)
        except ValidationError as e2:
            edge_errors[str(eid)] = e2.json()

    ok = not node_errors and not edge_errors
    return DocumentGraphValidationResult(ok=ok, node_errors=node_errors, edge_errors=edge_errors)

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
    require_role("rw")
    # 1) Ensure the document row exists (idempotent)
    if inp.content is not None:
        engine.add_document(Document(id=inp.doc_id, content=inp.content, type=inp.doc_type))
    else:
        # create a placeholder if completely missing so refs can anchor to doc_id
        if not engine._fetch_document_text(inp.doc_id):
            engine.add_document(Document(id=inp.doc_id, content="", type=inp.doc_type))

    # 2) Validate to your LLM models (references REQUIRED by GraphEntityBase)
    #    This matches your extractor path which later calls FromLLMSlice.
    try:
        parsed = LLMGraphExtraction.FromLLMSlice({
            "nodes": inp.nodes,
            "edges": inp.edges,
        }, insertion_method=inp.insertion_method)
    except:
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
class ContractGraphUpsert(BaseModel):
    doc_id: str
    insertion_method: str = "document_parser_v1"
    nodes: List[Node]
    edges: List[Edge] = []

class ContractGraphUpsertResult(BaseModel):
    status: str
    inserted_nodes: int
    inserted_edges: int
    engine_result: Dict[str, Any] | None = None

@app.post("/api/contract.upsert_tree", response_model=ContractGraphUpsertResult)
async def contract_upsert_tree(payload: ContractGraphUpsert):
    from .models import GraphExtractionWithIDs
    try:
        res = engine.persist_document_graph_extraction(
            parsed = GraphExtractionWithIDs(
                nodes=[n.model_dump(field_mode = 'backend') for n in payload.nodes],
                edges=[e.model_dump(field_mode = 'backend') for e in payload.edges]),
            # insertion_method=payload.insertion_method,
            doc_id=payload.doc_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ContractGraphUpsertResult(
        status="ok",
        inserted_nodes=len(payload.nodes),
        inserted_edges=len(payload.edges),
        engine_result=res,
    )
class KGUpsertIn(BaseModel):
    """
    Strict LLM-conformant upsert:
    - nodes / edges must follow LLM* models (including REQUIRED references with spans).
    - endpoints may use 'nn:*' / 'ne:*' temp ids that resolve in-batch.
    - references may use document alias token ::DOC:: in URLs; we’ll de-alias to doc_id.
    """
    content: Optional[str] = Field(None, description="If provided and doc is new, store this as document content")
    insertion_method: str = Field("api_upsert", description="Provenance tag copied into each ReferenceSession")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="PureNode-shaped dicts")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="PureEdge-shaped dicts")

class GraphUpsertOut(BaseModel):
    node_ids: List[str]
    edge_ids: List[str]
    nodes_added: int
    edges_added: int
@tool_roles({Role.RW})
@require_ns(NameSpace.WISDOM)
@mcp.tool(name="wisdom.kg_upsert_graph")
def kg_upsert_graph_wisdom(inp: KGUpsertIn) -> GraphUpsertOut:
    # You can keep doc_id prefixes or types here for clarity
    
    # Optional: tag the insertion method so you can rollback by provenance
    inp.insertion_method = inp.insertion_method or "wisdom_runtime"
    pure_graph = PureGraph.model_validate(dict(nodes = inp.nodes, edges = inp.edges))
    
    return GraphUpsertOut.model_validate(wisdom_engine.persist_graph(parsed = pure_graph, session_id = "wisdom:"+str(uuid.uuid1())))
    # return _kg_upsert_graph_impl(wisdom_engine, wisdom_gq, inp)
@tool_roles({Role.RO, Role.RW})
@require_ns(NameSpace.WISDOM)
@mcp.tool(name="wisdom.semantic_seed_then_expand")
def wisdom_semantic_seed_then_expand(text: str, top_k: int = 10, hops: int = 2):
    out = wisdom_gq.semantic_seed_then_expand_text(text, top_k=top_k, hops=hops)
    return out
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
@app.get("/viz/go", response_class=HTMLResponse)
def viz_go(
    request: Request,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
):
    return templates.TemplateResponse(
        "go.html",
        {"request": request, "doc_id": doc_id, "mode": mode, "insertion_method": insertion_method},
    )
import sqlite3
from typing import List, Optional
from pydantic import BaseModel
import json, uuid

class IndexingItem(BaseModel):
    node_id: str
    canonical_title: str
    keywords: List[str]
    aliases: List[str]
    provision: str
    doc_id: Optional[str]

class AddIndexEntriesInput(BaseModel):
    index: List[IndexingItem]

import sqlite3
def maybe_index_vector(item: IndexingItem):
    engine.node_index_collection.upsert(
        # collection_name="semantic_index",
        ids=[f"idx:{uuid.uuid1()}:{item.node_id}"],
        metadatas=[{
            "target_node_id": item.node_id,
            "canonical_title": item.canonical_title,
            "provision": item.provision,
            "keywords": json.dumps(item.keywords),
            "aliases": json.dumps(item.aliases),
        }],
        documents=[str(item.model_dump())],
    )
def ensure_index_tables(conn: sqlite3.Connection):
    cur = conn.cursor()

    # 1) base table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS semantic_index (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id         TEXT    NOT NULL,
        canonical_title TEXT    NOT NULL,
        keywords        TEXT    NOT NULL DEFAULT '',
        aliases         TEXT    NOT NULL DEFAULT '',
        provision       TEXT    NOT NULL,
        document_id     TEXT,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (node_id, canonical_title, provision)
    );
    """)

    # 2) FTS5 table (no DROP here!)
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS semantic_index_fts
    USING fts5(
        canonical_title,
        provision,
        keywords,
        aliases,
        content='semantic_index',
        content_rowid='id'
    );
    """)

    # 3) triggers to keep FTS in sync
    # we write *all* 4 fields, not just title+provision
    cur.executescript("""
    CREATE TRIGGER IF NOT EXISTS semantic_index_ai
    AFTER INSERT ON semantic_index
    BEGIN
        INSERT INTO semantic_index_fts(
            rowid,
            canonical_title,
            provision,
            keywords,
            aliases
        )
        VALUES (
            new.id,
            new.canonical_title,
            new.provision,
            new.keywords,
            new.aliases
        );
    END;

    CREATE TRIGGER IF NOT EXISTS semantic_index_ad
    AFTER DELETE ON semantic_index
    BEGIN
        DELETE FROM semantic_index_fts WHERE rowid = old.id;
    END;

    CREATE TRIGGER IF NOT EXISTS semantic_index_au
    AFTER UPDATE ON semantic_index
    BEGIN
        UPDATE semantic_index_fts
        SET
            canonical_title = new.canonical_title,
            provision       = new.provision,
            keywords        = new.keywords,
            aliases         = new.aliases
        WHERE rowid = new.id;
    END;
    """)

    # 4) extra helper tables (optional)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS semantic_index_keyword (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        index_id INTEGER NOT NULL REFERENCES semantic_index(id) ON DELETE CASCADE,
        keyword  TEXT NOT NULL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_index_keyword_kw ON semantic_index_keyword(keyword);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS semantic_index_alias (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        index_id INTEGER NOT NULL REFERENCES semantic_index(id) ON DELETE CASCADE,
        alias    TEXT NOT NULL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_index_alias_alias ON semantic_index_alias(alias);")

    conn.commit()
def _safe_upsert_fts(cur, idx_id, item):
    kw = " ".join(item.keywords) if item.keywords else ""
    al = " ".join(item.aliases) if item.aliases else ""
    try:
        cur.execute("DELETE FROM semantic_index_fts WHERE rowid = ?", (idx_id,))
        cur.execute(
            """
            INSERT INTO semantic_index_fts (rowid, canonical_title, keywords, aliases, provision)
            VALUES (?, ?, ?, ?, ?)
            """,
            (idx_id, item.canonical_title, kw, al, item.provision)
        )
    except sqlite3.DatabaseError as e:
        # mark to rebuild later, but don't kill the whole request
        print("WARNING: FTS table bad / missing, need rebuild:", e)
        # you could set a flag in another table, or just skip    
@app.post("/api/add_index_entries")
def add_index_entries(payload: AddIndexEntriesInput):
    import sqlite3
    conn = sqlite3.connect(index_db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    try:
        for item in payload.index:
            kw = " ".join(item.keywords) if item.keywords else ""
            al = " ".join(item.aliases) if item.aliases else ""

            # upsert base table ONLY
            cur.execute(
                """
                INSERT INTO semantic_index
                    (node_id, canonical_title, keywords, aliases, provision, document_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id, canonical_title, provision) DO UPDATE SET
                    canonical_title = excluded.canonical_title,
                    keywords        = excluded.keywords,
                    aliases         = excluded.aliases,
                    provision       = excluded.provision,
                    document_id     = excluded.document_id,
                    updated_at      = CURRENT_TIMESTAMP
                """,
                (
                    item.node_id,
                    item.canonical_title,
                    kw,
                    al,
                    item.provision,
                    item.doc_id,
                ),
            )
            maybe_index_vector(item)
        conn.commit()
    finally:
        conn.close()

    return {"ok": True}
conn = sqlite3.connect(index_db_path)
cur = conn.cursor()
cur.execute("PRAGMA integrity_check;")
print(cur.fetchall())


ensure_index_tables(conn)
conn.close()
from numpy import interp
@app.get("/api/search_index_hybrid")
def search_index_hybrid(q: str, limit: int = 10, resolve_node = False) -> Dict[str, Any]:
    conn = sqlite3.connect(index_db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Lexical (FTS5)
    cur.execute(
        """
        SELECT
            s.id,
            s.node_id,
            s.canonical_title,
            s.provision,
            bm25(semantic_index_fts) AS fts_score
        FROM semantic_index AS s
        JOIN semantic_index_fts
        ON semantic_index_fts.rowid = s.id
        WHERE semantic_index_fts MATCH ?
        ORDER BY fts_score
        LIMIT ?;
        """,
        (q, limit),
    )
    fts_rows = cur.fetchall()

    # 2. Semantic (vector)
    vector_results = engine.node_index_collection.query(
        # collection_name="semantic_index",
        query_texts=[q],
        n_results=limit
    )

    # 3. Merge and normalize
    combined = {}
    # normalize scores to 0–1
    fts_scores = [r["fts_score"] for r in fts_rows]
    if fts_scores:
        min_f, max_f = min(fts_scores), max(fts_scores)
    else:
        min_f, max_f = 0, 1
    for r in fts_rows:
        sid = r["id"]
        combined[sid] = {
            "node_id": r["node_id"],
            "canonical_title": r["canonical_title"],
            "provision": r["provision"],
            "fts_score": interp(r["fts_score"], [max_f, min_f], [1.0, 0.0]),  # lower bm25 = better
            "vec_score": 0.0,
        }

    for idx, id_ in enumerate(vector_results["ids"][0]):
        vec_score = vector_results["distances"][0][idx]
        if id_ not in combined:
            combined[id_] = {
                "node_id": vector_results["metadatas"][0][idx]["target_node_id"],
                "canonical_title": vector_results["metadatas"][0][idx]["canonical_title"],
                "provision": vector_results["metadatas"][0][idx]["provision"],
                "fts_score": 0.0,
                "vec_score": vec_score,
            }
        else:
            combined[id_]["vec_score"] = vec_score

    # 4. Combine ranks
    ranked = sorted(combined.values(), key=lambda x: 0.6*x["fts_score"] + 0.4*x["vec_score"], reverse=True)
    if resolve_node:
        ids = {}
        for i in ranked:
            if i["node_id"] in ids:
                pass
            else:
                ids[i["node_id"]] = i
        res = engine.node_collection.get(ids = list(ids.keys()))
        cols = ['ids'] + res['included']
        rows = {}
        for i, id in enumerate(res["ids"]):
            rows[id] = {}
            row = rows[id]
            for col in cols:
                row[col] = res[col][i]
        to_return = ranked[:limit]
        for i in to_return:
            i.update(rows[i['node_id']])
        return {"query": q, "results": to_return}
    return {"query": q, "results": ranked[:limit]}
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
@tool_roles({Role.RO, Role.RW})
@require_ns("docs")
@mcp.tool()
def kg_load_persisted(inp: LoadPersistedIn) -> LoadPersistedOut:
    """MCP: read back persisted IDs for the given docs (fast; no LLM)."""
    all_persisted = _load_persisted_graph(inp.doc_ids, insertion_method=inp.insertion_method)
    all_persisted.node_ids = sorted( shortids.l2s_id(nid) for nid in all_persisted.node_ids)
    all_persisted.edge_ids = sorted( shortids.l2s_id(eid) for eid in all_persisted.edge_ids)
    return all_persisted

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
    # NEED-FIX
    for r in (n.mentions or []):
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

@tool_roles({Role.RW})
@require_ns("docs")
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
    require_role("rw")
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
    
# --- Propose vector candidates with optional 'where' filter ---
from graph_knowledge_engine.strategies.proposer import VectorProposer

class ProposePair(BaseModel):
    left_id: str
    left_kind: Literal["node", "edge"]
    right_id: str
    right_kind: Literal["node", "edge"]

class ProposeOut(BaseModel):
    pairs: List[ProposePair]
class ProposeVectorIn(BaseModel):
    new_node_ids: Optional[List[str]] = None
    new_edge_ids: Optional[List[str]] = None
    top_k: int = 10
    score_mode: Literal["distance", "similarity"] = "distance"
    max_distance: float = 0.35
    min_similarity: float = 0.65
    include_edges: bool = True
    allowed_docs: Optional[List[str]] = None
    anchor_doc_id: Optional[str] = None
    cross_doc_only: bool = False
    anchor_only: bool = True
    where: Optional[str| dict] = None
    
@tool_roles({Role.RO, Role.RW})
@require_ns("docs")
@mcp.tool()
def propose_vector(inp : ProposeVectorIn
    # new_node_ids: List[str],
    # top_k: int = 10,
    # score_mode: Literal["distance", "similarity"] = "distance",
    # max_distance: float = 0.25,
    # min_similarity: float = 0.85,
    # include_edges: bool = True,
    # allowed_docs: Optional[List[str]] = None,
    # anchor_doc_id: Optional[str] = None,
    # cross_doc_only: bool = False,
    # anchor_only: bool = True,
    # where: Optional[str| dict] = None,  # JSON string for Chroma where, e.g. {"insertion_method":"graph_extractor"}
) -> ProposeOut:
    "Propose merging of nodes and edges, use when editing knowledge graph. Do not use this tool when making queries."
    # where_dict = dict | None
    # if type(where) is dict:
    #     where_dict = where
    # else:
    #     try:
    #         where_dict = json.loads(where)
    #     except Exception as e:
    #         raise HTTPException(status_code=400, detail=f"Invalid where JSON: {e}")
    new_node_ids: Optional[List[str]] = inp.new_node_ids
    new_edge_ids: Optional[List[str]] = inp.new_edge_ids
    top_k: int = inp.top_k
    score_mode: Literal["distance", "similarity"] = inp.score_mode
    max_distance: float = inp.max_distance
    min_similarity: float = inp.min_similarity
    include_edges: bool = inp.include_edges
    allowed_docs: Optional[List[str]] = inp.allowed_docs
    anchor_doc_id: Optional[str] = inp.anchor_doc_id
    cross_doc_only: bool = inp.cross_doc_only
    anchor_only: bool = inp.anchor_only
    where = inp.where
    prop = VectorProposer(engine)
    pairs = prop.generate_merge_candidates(
        engine=engine,
        new_node=new_node_ids,              # can be ids or Node objects
        new_edge = new_edge_ids,
        top_k=top_k,
        allowed_docs=allowed_docs,
        anchor_doc_id=anchor_doc_id,
        cross_doc_only=cross_doc_only,
        anchor_only=anchor_only,
        score_mode=score_mode,
        max_distance=max_distance,
        min_similarity=min_similarity,
        include_edges=include_edges,
        where=where                   # <--- threads through to Chroma
    )
    out = []
    for (lid, rid), (l, r, score) in pairs.items():
        out.append(ProposePair(
            left_id=getattr(l, "id", ""),
            left_kind="edge" if isinstance(l, Edge) else "node" ,            # query side is a node by contract
            right_id=getattr(r, "id", ""),
            right_kind="edge" if isinstance(r, Edge) else "node",
        ))
    return ProposeOut(pairs=out)
def _ids_matching_where(kind: Literal["node", "edge"], where: Dict[str, Any]) -> Set[str]:
    """
    Minimal, server-side filter: fetch ids matching metadata `where`.
    Avoids changing the proposer. If `where` is None/empty, return empty set to mean 'no restriction'.
    """
    if not where:
        return set()
    if kind == "node":
        res = engine.node_collection.get(where=where)  # returns dict with 'ids'
    else:
        res = engine.edge_collection.get(where=where)
    return set(res.get("ids") or [])


from itertools import product
from typing import ClassVar
class ProposeBruteForceIn(BaseModel):
    PairableNodeTypes : ClassVar[List[Literal['node', 'edge', 'any']]] = ['node', 'edge', 'any']
    pair_kind: Literal[*["_".join(i) for i in list(product(PairableNodeTypes, PairableNodeTypes))]] = "any_any"
    allowed_docs: Optional[List[str]] = None
    anchor_doc_id: Optional[str] = None
    cross_doc_only: bool = False
    anchor_only: bool = True
    limit_per_bucket: Optional[int] = 200
    where: Optional[dict] = None # JSON string, e.g. {"insertion_method":"graph_extractor"}
@tool_roles({Role.RO, Role.RW})
@require_ns("docs")
@mcp.tool()
def kg_propose_bruteforce(
    inp : ProposeBruteForceIn
) -> ProposeOut:
    """
    Brute-force candidate proposal across documents with optional Chroma metadata filtering.
    - pair_kind: which pairs to propose
    - allowed_docs / anchor_doc_id / cross_doc_only / anchor_only: document scoping knobs
    - where: JSON applied as a metadata filter; enforced *server-side* post-proposal
             
    """
    PairableNodeTypes : ClassVar[List[Literal['node', 'edge', 'any']]] = ['node', 'edge', 'any']
    pair_kind: Literal[*["_".join(i) for i in list(product(PairableNodeTypes, PairableNodeTypes))]] = inp.pair_kind
    allowed_docs: Optional[List[str]] = inp.allowed_docs
    anchor_doc_id: Optional[str] = inp.anchor_doc_id
    cross_doc_only: bool = inp.cross_doc_only
    anchor_only: bool = inp.anchor_only
    limit_per_bucket: Optional[int] = inp.limit_per_bucket
    where: Optional[dict] = inp.where
    

    # Produce raw pairs (no where filter yet)
    proposer = VectorProposer(engine)
    raw_pairs = proposer.propose_any_kind_any_doc(
        engine=engine,
        pair_kind=pair_kind,
        allowed_docs=allowed_docs,
        anchor_doc_id=anchor_doc_id,
        cross_doc_only=cross_doc_only,
        anchor_only=anchor_only,
        limit_per_bucket=limit_per_bucket,
    )

    # If no 'where' provided, return as-is (minimal overhead)
    if not where:
        out = [
            ProposePair(
                left_id=getattr(l, "id", ""),
                left_kind="edge" if isinstance(l, Edge) else "node",
                right_id=getattr(r, "id", ""),
                right_kind="edge" if isinstance(r, Edge) else "node",
            )
            for (l, r) in raw_pairs
        ]
        return ProposeOut(pairs=out)

    # Server-side filtering by metadata:
    node_ok = _ids_matching_where("node", where)
    edge_ok = _ids_matching_where("edge", where)

    def _passes_where(obj: Node | Edge) -> bool:
        if isinstance(obj, Edge):
            return (not edge_ok) or (obj.id in edge_ok)
        else:
            return (not node_ok) or (obj.id in node_ok)

    filtered = []
    for l, r in raw_pairs:
        # Enforce kind-aware membership
        if not (_passes_where(l) and _passes_where(r)):
            continue
        filtered.append(
            ProposePair(
                left_id=getattr(l, "id", ""),
                left_kind="edge" if isinstance(l, Edge) else "node",
                right_id=getattr(r, "id", ""),
                right_kind="edge" if isinstance(r, Edge) else "node",
            )
        )

    return ProposeOut(pairs=filtered)

"""
Quick usage notes

Vector (best for “given these N fresh nodes, what’s similar anywhere in the graph?”):

// MCP tool: propose_vector
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

// MCP tool: propose_bruteforce
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

class AdjPairsIn(BaseModel):
    pairs: List["ProposePair"]
    commit : bool = False
@tool_roles({Role.RW})
@require_ns("docs")
@mcp.tool()
def commit_merge(inp: CrossDocAdjOut):
    """ merge pair of nodes of edges after running adjudication proposal
    """
    require_role("rw")
    adj_out = inp
    committed = []
    for pairs in adj_out.results:
        left, right = engine.get_nodes(pairs.left)[0], engine.get_nodes(pairs.right)[0]
        lkind, rkind, same_entity = pairs.left_kind, pairs.right_kind, pairs.same_entity
        if same_entity:
            verdict = AdjudicationVerdict(same_entity=pairs.same_entity,
                                          confidence =pairs.confidence,
                                          reason = pairs.reason, canonical_entity_id= None)
            if lkind == rkind == 'node': # legacy and later implemented others
                canonical_id = engine.commit_merge(left, right, verdict)
            else:
                # Requires your engine to expose commit_any_kind(Node|Edge, Node|Edge, verdict)
                canonical_id = engine.commit_any_kind(left, right, verdict)
            verdict.canonical_entity_id = canonical_id
            if canonical_id:
                committed.append(str(canonical_id))
@tool_roles({Role.RW})
@require_ns("docs")
@mcp.tool()
def adjudicate_pairs(inp: AdjPairsIn) -> CrossDocAdjOut:
    """ Propose pairs of similar meaning nodes/ edges for subsequent merging
    """
    if inp.commit:
        require_role("rw")
    pairs: List[Tuple[Node| Edge, Node| Edge]] = [None] * len(inp.pairs) # type: ignore
    for i, pair_info in enumerate(inp.pairs):
        left_id, left_kind, right_id, right_kind = pair_info.left_id, pair_info.left_kind, pair_info.right_id, pair_info.right_kind
        def fetch_any(id, kind):
            if kind == 'node':
                return _fetch_nodes([id])
            elif kind == 'edge':
                return _fetch_edges([id])
            
        l: Node| Edge = fetch_any(left_id, left_kind)
        r: Node| Edge = fetch_any(right_id, right_kind)
        pairs[i] = (l[0],r[0])
    
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
# Mount the MCP server
app.mount("/", mcp_app)

# Run with:
#   uvicorn server_mcp_with_admin:app --port 8765
