from __future__ import annotations
# server_mcp_with_admin.py
import contextvars
import functools
from contextvars import ContextVar
from starlette.types import Scope, Receive, Send
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.engine_core.models import Document
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
import sqlite3
import time
from threading import Lock
from typing import Callable, List, Optional
from typing import Any, cast, Dict, Set
import uuid
from graph_knowledge_engine.shortids import run_id_ctx, run_id_scope
from graph_knowledge_engine import shortids
from graph_knowledge_engine.id_provider import new_id_str, stable_id
from graph_knowledge_engine.server.chat_api import create_chat_router
from graph_knowledge_engine.server.runtime_api import create_runtime_router
from graph_knowledge_engine.runtime.models import WorkflowNodeMetadata, WorkflowEdgeMetadata
from graph_knowledge_engine.server.chat_mcp import build_conversation_mcp, build_workflow_mcp
from graph_knowledge_engine.server.bootstrap import (
    build_graph_engine,
    build_sqlalchemy_engine,
    load_server_storage_settings,
)
from graph_knowledge_engine.server.chat_service import ChatRunService
from graph_knowledge_engine.server.run_registry import RunRegistry
from graph_knowledge_engine.engine_core.search_index.models import AddIndexEntriesInput
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
    CONVERSATION = "conversation"
    WORKFLOW = "workflow"
    WISDOM = "wisdom"
# tool name -> allowed roles
TOOL_ROLES: dict[str, set[str]] = {}
TOOL_NAMESPACE: dict[str, set[str]] = {}
def tool_roles(roles: set[Role] | Role):
    """Annotate a tool with allowed roles (e.g. {Role.RO, Role.RW} or Role.RW)."""
    allowed = {roles} if isinstance(roles, Role) else set(roles)
    def deco(fn: Callable):
        name = getattr(fn, "name", None) or getattr(fn, "__name__", None)
        if name is None:
            raise Exception('name not found')
        # we record by function name here; FastMCP uses that as tool name by default
        TOOL_ROLES[name] = {r.value for r in allowed}
        original_fn = fn
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            user_role = get_current_role()
            if user_role not in allowed:
                raise HTTPException(status_code=403, detail=f"Forbidden: role {user_role} not permitted call this tool")
            # if ROLE_ORDER.get(user_role, 0) < ROLE_ORDER.get(min_role, 0):
            #     raise HTTPException(status_code=403, detail=f"Forbidden: requires role '{min_role}', you have '{user_role}'")
            return original_fn(*args, **kwargs)
        fn = wrapper
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
storage_settings = load_server_storage_settings()
persist_directory = storage_settings.knowledge_dir
conversation_persist_directory = storage_settings.conversation_dir
workflow_persist_directory = storage_settings.workflow_dir
wisdom_persist_directory = storage_settings.wisdom_dir


class _LazyResource:
    def __init__(self, factory: Callable[[], object], name: str) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_value", None)
        object.__setattr__(self, "_lock", Lock())

    def get(self) -> object:
        value = object.__getattribute__(self, "_value")
        if value is None:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                value = object.__getattribute__(self, "_value")
                if value is None:
                    value = object.__getattribute__(self, "_factory")()
                    object.__setattr__(self, "_value", value)
        return value

    def __getattr__(self, name: str) -> object:
        return getattr(self.get(), name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_factory", "_name", "_value", "_lock"}:
            object.__setattr__(self, name, value)
            return
        setattr(self.get(), name, value)

    def __repr__(self) -> str:
        value = object.__getattribute__(self, "_value")
        state = "initialized" if value is not None else "lazy"
        return f"<_LazyResource {object.__getattribute__(self, '_name')} ({state})>"


def _build_pg_sqlalchemy_engine():
    return build_sqlalchemy_engine(storage_settings)


def _shared_sqlalchemy_engine():
    if storage_settings.backend != "pg":
        return None
    return pg_sqlalchemy_engine.get()


def _build_engine() -> GraphKnowledgeEngine:
    return build_graph_engine(
        settings=storage_settings,
        graph_type="knowledge",
        sa_engine=_shared_sqlalchemy_engine(),
    )


def _build_conversation_engine() -> GraphKnowledgeEngine:
    eng = build_graph_engine(
        settings=storage_settings,
        graph_type="conversation",
        sa_engine=_shared_sqlalchemy_engine(),
    )
    return eng


def _build_workflow_engine() -> GraphKnowledgeEngine:
    return build_graph_engine(
        settings=storage_settings,
        graph_type="workflow",
        sa_engine=_shared_sqlalchemy_engine(),
    )


def _build_wisdom_engine() -> GraphKnowledgeEngine:
    return build_graph_engine(
        settings=storage_settings,
        graph_type="wisdom",
        sa_engine=_shared_sqlalchemy_engine(),
    )


pg_sqlalchemy_engine = _LazyResource(_build_pg_sqlalchemy_engine, "pg_sqlalchemy_engine")
engine: GraphKnowledgeEngine = _LazyResource(_build_engine, "knowledge_engine")
conversation_engine = _LazyResource(_build_conversation_engine, "conversation_engine")
workflow_engine = _LazyResource(_build_workflow_engine, "workflow_engine")
wisdom_engine = _LazyResource(_build_wisdom_engine, "wisdom_engine")
gq = _LazyResource(lambda: GraphQuery(engine.get()), "knowledge_graph_query")
conversation_gq = _LazyResource(lambda: GraphQuery(conversation_engine.get()), "conversation_graph_query")
wisdom_gq = _LazyResource(lambda: GraphQuery(wisdom_engine.get()), "wisdom_graph_query")
run_registry = _LazyResource(
    lambda: RunRegistry(workflow_engine.get().meta_sqlite),
    "chat_run_registry",
)
chat_service = _LazyResource(
    lambda: ChatRunService(
        get_knowledge_engine=lambda: engine.get(),
        get_conversation_engine=lambda: conversation_engine.get(),
        get_workflow_engine=lambda: workflow_engine.get(),
        run_registry=run_registry.get(),
    ),
    "chat_run_service",
)
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
from fastmcp.tools.function_tool import FunctionTool
def _tool_name_candidates(name: str) -> list[str]:
    base = str(name or "")
    out = [base]
    if "." in base:
        out.append(base.replace(".", "_"))
    if "_" in base:
        out.append(base.replace("_", "."))
    seen = []
    for item in out:
        if item and item not in seen:
            seen.append(item)
    return seen

def _filter_tool_list(lst: list[dict]) -> list[dict]:
    role = current_role.get()
    namespace = get_current_namespace()
    out = []
    for item in lst:
        name = getattr(item, 'name', None) if type(item) is FunctionTool else None or item.get("name") or item.get("tool") or ""
        # accept either exact name or any recorded alias
        if name:
            tool_roles = {Role.RO.value}
            tool_namespace = {NameSpace.DOCS.value}
            for candidate in _tool_name_candidates(str(name)):
                tool_roles = TOOL_ROLES.get(candidate, tool_roles)
                tool_namespace = TOOL_NAMESPACE.get(candidate, tool_namespace)
            if role in tool_roles and namespace in tool_namespace:
                out.append(item)
        else:
            tool_roles = None
            tool_namespace = None
        
    return out


def create_conversation():
    """create ref node in other conversation collection"""
    raise NotImplementedError()
    pass

# --- FastMCP 3.x RBAC middleware (replaces ToolManager monkeypatch from 2.x) ---
#
# FastMCP 3 removed ToolManager, so the supported way to filter tool visibility is
# via middleware hooks (on_list_tools / on_call_tool).
try:
    from fastmcp.server.middleware import Middleware  # type: ignore
except Exception:  # pragma: no cover
    class Middleware:  # minimal fallback so import-time doesn't explode
        pass


def _tool_allowed(tool_name: str, *, role: str, namespace: str) -> bool:
    allowed_roles = {Role.RO.value}
    allowed_namespaces = {NameSpace.DOCS.value}
    for candidate in _tool_name_candidates(str(tool_name)):
        allowed_roles = TOOL_ROLES.get(candidate, allowed_roles)
        allowed_namespaces = TOOL_NAMESPACE.get(candidate, allowed_namespaces)
    return (role in allowed_roles) and (namespace in allowed_namespaces)


class RbacMiddleware(Middleware):
    """RBAC for MCP tools.

    - Filters tools from tools/list so unauthorized tools are not visible.
    - Blocks tools/call even if a client guesses the name.

    Authorization is derived from JWT claims populated by JWTProtectMiddleware
    (claims_ctx ContextVar). This preserves seam-boundary behavior for both MCP
    and FastAPI endpoints.
    """

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        role = get_current_role()
        ns = get_current_namespace()
        out = []
        for t in list(tools):
            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
            if not name:
                out.append(t)
                continue
            if _tool_allowed(str(name), role=role, namespace=ns):
                out.append(t)
        return out

    async def on_call_tool(self, context, call_next):
        role = get_current_role()
        ns = get_current_namespace()
        req = getattr(context, "request", None)
        tool_name = None
        for attr in ("name", "tool_name", "tool"):
            if req is not None and hasattr(req, attr):
                tool_name = getattr(req, attr)
                break
        if tool_name is None and isinstance(req, dict):
            tool_name = req.get("name") or req.get("tool_name") or req.get("tool")
        if isinstance(tool_name, dict):
            tool_name = tool_name.get("name")
        if tool_name:
            if not _tool_allowed(str(tool_name), role=role, namespace=ns):
                raise HTTPException(status_code=403, detail=f"Tool '{tool_name}' not permitted for role '{role}' in namespace '{ns}'")
        return await call_next(context)
class JWTProtectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # IMPORTANT: some /api endpoints enforce auth via require_role()/require_ns(),
        # which read from claims_ctx. So we must populate claims_ctx whenever a
        # bearer token is present, not only for PROTECTED_PREFIXES.
        if claims_ctx.get() is not None:
            return await call_next(request)

        path = request.url.path
        is_protected = any(path.startswith(p) for p in PROTECTED_PREFIXES)
        token = _extract_bearer(request)

        if not token:
            if is_protected:
                return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
            return await call_next(request)

        try:
            claims = verify_jwt(token)
        except HTTPException as e:
            # If endpoint isn't globally protected, treat invalid token as anonymous.
            # Handlers that require auth will still reject via require_role/ns.
            if is_protected:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code)
            return await call_next(request)

        request.state.claims = claims
        token_ = claims_ctx.set(claims)
        try:
            with run_id_scope(token):
                return await call_next(request)
        finally:
            claims_ctx.reset(token_)


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
DEFAULT_NAMESPACE = NameSpace.DOCS.value


def get_current_namespace() -> str:
    claims = claims_ctx.get() or {}
    ns = str(claims.get("ns") or DEFAULT_NAMESPACE).lower()
    allowed = {item.value for item in NameSpace}
    return ns if ns in allowed else DEFAULT_NAMESPACE


def get_current_subject() -> str | None:
    claims = claims_ctx.get() or {}
    sub = str(claims.get("sub") or "").strip()
    return sub or None

def _normalize_namespaces(expected: set[NameSpace] | NameSpace | set[str] | str) -> set[str]:
    if isinstance(expected, set):
        raw_items = list(expected)
    else:
        raw_items = [expected]
    if not raw_items:
        raise ValueError("At least 1 namespace has to be specified")
    allowed = set()
    valid = {item.value for item in NameSpace}
    for item in raw_items:
        value = str(item.value if isinstance(item, NameSpace) else item).lower()
        if value not in valid:
            raise ValueError(f"Unknown namespace: {item!r}")
        allowed.add(value)
    return allowed

def require_namespace(expected: set[NameSpace] | NameSpace | set[str] | str):
    allowed = _normalize_namespaces(expected)
    actual = get_current_namespace()
    if actual not in allowed:
        raise HTTPException(status_code=403, detail=f"Forbidden: namespace '{actual}' is not permitted")
    return actual

from fastmcp.tools import FunctionTool
def require_ns(expected: set[NameSpace] | NameSpace):
    allowed = _normalize_namespaces(expected)
    
    def deco(fn: Callable):
        name = getattr(fn, "name", None) or getattr(fn, "__name__", None)
        if name is None:
            raise Exception('name not found')
        # we record by function name here; FastMCP uses that as tool name by default
        TOOL_NAMESPACE[name] = allowed
        original_fn = fn
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            actual = get_current_namespace()
            if actual not in allowed:
                # MCP tools throw regular exceptions; FastMCP wraps as tool error
                raise HTTPException(status_code=403, detail=f"Forbidden: namespace '{actual}' cannot call this tool")
                
            return original_fn(*args, **kwargs)
        fn = wrapper
        return fn
        
    return deco
    #     @functools.wraps(fn)
    #     def wrapper(*args, **kwargs):
    #         actual = get_current_namespace()
    #         if actual != expected:
    #             # MCP tools throw regular exceptions; FastMCP wraps as tool error
    #             raise PermissionError(f"Forbidden: namespace '{actual}' cannot call this tool (expected '{expected}').")
    #         return fn(*args, **kwargs)
mcp = FastMCP("KnowledgeEngine + MCP + Admin", middleware=[RbacMiddleware()])

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
from .engine_core.models import Document, PureGraph
class DocParseIn(Document['dto']):
    id: Optional[str] # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    content: str
    type: str = "text"

class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: List[str]
    summary_node_id: Optional[str]
from graph_knowledge_engine.ingester import PagewiseSummaryIngestor

@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    """Parse a document into leaf and relationships between chunks with summaries from low to high abstraction levels."""
    require_role("rw")
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        raise RuntimeError(
            "doc_parse requires optional dependency group 'gemini'. "
            "Install with: pip install 'kogwistar[gemini]'"
        ) from e
    ingester_llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro"),
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    ingester = PagewiseSummaryIngestor(engine=engine, llm=ingester_llm, 
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
        
    docs = engine.backend.document_get(ids = filenames) # this is only where we do not use shortid
    sids = [shortids.l2s_id(i) for i in docs['ids']]
    to_return = [{"file_name": i, "id": shortids.l2s_id(i)} for i in docs['ids']]
    return DocIdsOut(id_mapping = to_return)



@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def store_document(inp: DocParseIn):
    """Store document in graph database"""
    require_role("rw")
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    engine.write.add_document(doc)
    return DocStoreOut.model_validate({"success": True})
@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    """From documents extract knowledge and relationships as a hypergraph between entities, ideas, concepts with each other."""
    require_role("rw")
    content = engine.extract.fetch_document_text(inp.id)
    if not content:
        raise ValueError(f"Document '{inp.id}' not found; run store_document first.")
    from .engine_core.models import LLMGraphExtraction
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
        extracted = engine.extract.cached_extract_graph_with_llm(content=content)
        # parsed = extracted["parsed"]
        # if not isinstance(parsed, LLMGraphExtraction):
        #     dumped = parsed.model_dump(field_mode = 'backend')
        parsed_LLM: LLMGraphExtraction['llm'] = extracted["parsed"]
        # ctx = {"insertion_method": "graph_extractor"}
        # dumped = parsed_LLM.model_dump()
        parsed = LLMGraphExtraction.FromLLMSlice(parsed_LLM, insertion_method = "llm_graph_extraction")
        batch_node_ids, batch_edge_ids = engine.persist.preflight_validate(parsed, inp.id)
        return parsed
    parsed = get_reparsed_extraction(content)
    
    persisted = engine.persist.persist_graph_extraction(
        document=Document(id=inp.id, content=content, type="text"),
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



conversation_mcp = build_conversation_mcp(
    get_service=lambda: chat_service.get(),
    tool_roles=tool_roles,
    require_ns=require_ns,
    role_ro=Role.RO,
    role_rw=Role.RW,
    ns_conversation=NameSpace.CONVERSATION,
)
workflow_mcp = build_workflow_mcp(
    get_service=lambda: chat_service.get(),
    tool_roles=tool_roles,
    require_ns=require_ns,
    role_ro=Role.RO,
    role_rw=Role.RW,
    ns_workflow=NameSpace.WORKFLOW,
    get_subject=get_current_subject,
)
mcp.mount(conversation_mcp)
mcp.mount(workflow_mcp)

# ---- Build a unified FastAPI app: /mcp + /admin ----
mcp_app = mcp.http_app(path='/mcp')
app = FastAPI(title="KnowledgeEngine + MCP + Admin", lifespan=mcp_app.lifespan)
app.add_middleware(MCPRoleMiddleware)
app.add_middleware(JWTProtectMiddleware)
app.include_router(
    create_chat_router(
        get_service=lambda: chat_service.get(),
        require_role=require_role,
        require_namespace=require_namespace,
        conversation_namespace=NameSpace.CONVERSATION.value,
        workflow_namespaces={NameSpace.CONVERSATION.value, NameSpace.WORKFLOW.value},
    )
)
app.include_router(
    create_runtime_router(
        get_service=lambda: chat_service.get(),
        require_role=require_role,
        require_namespace=require_namespace,
        runtime_namespaces={NameSpace.WORKFLOW.value},
        get_subject=get_current_subject,
    )
)
from datetime import datetime, timedelta, timezone

class DevTokenInp(BaseModel):
    username: str = "dev"
    role: str = "ro"
    ns : Literal["docs", "conversation", "workflow", "wisdom"]= "docs"
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


def _designer_resolver_candidates(service: Any) -> list[Any]:
    candidates: list[Any] = []
    seen: set[int] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        candidates.append(value)

    for attr in (
        "resolver",
        "step_resolver",
        "workflow_resolver",
        "runtime_resolver",
        "workflow_step_resolver",
    ):
        try:
            _add(getattr(service, attr, None))
        except Exception:
            pass

    for attr in (
        "get_resolver",
        "get_step_resolver",
        "get_workflow_resolver",
        "get_runtime_resolver",
    ):
        fn = getattr(service, attr, None)
        if callable(fn):
            try:
                _add(fn())
            except TypeError:
                pass
            except Exception:
                pass

    return candidates


def _designer_runtime_capabilities() -> dict[str, Any]:
    builtin_ops: set[str] = set()
    sandboxed_ops: set[str] = set()
    nested_ops: set[str] = set()
    state_schema: dict[str, str] = {}
    sandbox_runtime_configured = False
    sandbox_support = False
    resolver_found = False

    try:
        service = chat_service.get()
    except Exception:
        service = None

    for resolver in _designer_resolver_candidates(service):
        resolver_found = True
        try:
            ops = getattr(resolver, "ops", None)
            if ops is not None:
                builtin_ops.update(str(op) for op in ops)
        except Exception:
            pass
        try:
            nested_ops.update(str(op) for op in (getattr(resolver, "nested_ops", set()) or set()))
        except Exception:
            pass
        try:
            sandboxed = getattr(resolver, "sandboxed_ops", None)
            if sandboxed is not None:
                sandbox_support = True
                sandboxed_ops.update(str(op) for op in sandboxed)
        except Exception:
            pass
        try:
            describe_state = getattr(resolver, "describe_state", None)
            if callable(describe_state):
                state_schema.update({str(k): str(v) for k, v in (describe_state() or {}).items()})
        except Exception:
            pass
        try:
            if getattr(resolver, "_sandbox", None) is not None:
                sandbox_runtime_configured = True
                sandbox_support = True
        except Exception:
            pass

    return {
        "resolver_found": resolver_found,
        "builtin_ops": sorted(builtin_ops),
        "nested_ops": sorted(nested_ops),
        "sandboxed_ops": sorted(sandboxed_ops),
        "sandbox": {
            "supports_sandboxed_ops": sandbox_support,
            "runtime_configured": sandbox_runtime_configured,
        },
        "state_schema": dict(sorted(state_schema.items())),
    }


@app.get("/designer/capabilities")
def designer_capabilities():
    require_role("ro")
    require_namespace({NameSpace.WORKFLOW})

    node_schema = WorkflowNodeMetadata.model_json_schema()
    edge_schema = WorkflowEdgeMetadata.model_json_schema()
    runtime_caps = _designer_runtime_capabilities()

    return {
        "schema_version": "workflow-designer-capabilities/v1",
        "projection_schema": "workflow_design_v1",
        "design_features": {
            "undo_redo": True,
            "delta_history": True,
            "snapshot_restore": True,
            "dry_run_validation": False,
        },
        "custom_ops": {
            "allow_unregistered_ops_in_design": True,
            "allow_execution_of_unregistered_ops": False,
            "binding_statuses": ["resolved", "unresolved", "sandboxed", "plugin"],
        },
        "node_types": [
            {
                "type": "workflow_node",
                "display_name": "Workflow Node",
                "metadata_schema": node_schema,
                "flags": {
                    "supports_start": True,
                    "supports_terminal": True,
                    "supports_fanout": True,
                    "supports_join": True,
                },
            }
        ],
        "edge_types": [
            {
                "type": "workflow_edge",
                "display_name": "Workflow Edge",
                "metadata_schema": edge_schema,
                "flags": {
                    "supports_predicate": True,
                    "supports_priority": True,
                    "supports_default": True,
                    "supports_multiplicity": True,
                },
            }
        ],
        "runtime": runtime_caps,
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "backend": storage_settings.backend,
        "persist_directory": persist_directory,
        "conversation_persist_directory": conversation_persist_directory,
        "workflow_persist_directory": workflow_persist_directory,
        "wisdom_persist_directory": wisdom_persist_directory,
        "pg_schema_base": storage_settings.pg_schema_base if storage_settings.backend == "pg" else None,
    }

# DELETE /admin/doc/{doc_id}  (non-MCP utility)
@app.delete("/admin/doc/{doc_id}")
def admin_delete_doc(doc_id: str):
    require_role("rw")
    # Collect counts before deletion
    try:
        node_ids = engine.read.node_ids_by_doc(doc_id)
        edge_ids = engine.read.edge_ids_by_doc(doc_id)
    except Exception:
        node_ids, edge_ids = [], []

    # Delete endpoints and mapping tables first
    try:
        engine.backend.edge_endpoints_delete(where={"doc_id": doc_id})
    except Exception:
        pass
    try:
        engine.backend.node_docs_delete(where={"doc_id": doc_id})
    except Exception:
        pass

    # Delete primary rows
    
    try:
        if edge_ids:
            engine.backend.edge_refs_delete(ids=edge_ids)
            engine.backend.edge_delete(ids=edge_ids)
        else:
            engine.backend.edge_delete(where={"doc_id": doc_id})
    except Exception:
        pass
    try:
        if node_ids:
            engine.backend.node_refs_delete(ids=edge_ids)
            engine.backend.node_delete(ids=node_ids)
        else:
            engine.backend.node_delete(where={"doc_id": doc_id})
    except Exception:
        pass

    # Optional: document row (if you keep one)
    try:
        engine.backend.document_delete(where={"doc_id": doc_id})
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
    graph_type: str = "knowledge",
):
    gt = (graph_type or "knowledge").lower()
    if gt == "conversation":
        use_engine = conversation_engine
    elif gt == "workflow":
        use_engine = workflow_engine
    elif gt == "wisdom":
        use_engine = wisdom_engine
    else:
        use_engine = engine
    payload = to_cytoscape(use_engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method)
    return JSONResponse(payload)

@app.get("/viz/d3.bundle", response_class=HTMLResponse)
def viz_d3_bundle(
    request: Request,
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
    graph_type: str = "knowledge",  # knowledge|conversation|workflow|wisdom
):
    gt = (graph_type or "knowledge").lower()
    if gt == "conversation":
        use_engine = conversation_engine
    elif gt == "workflow":
        use_engine = workflow_engine
    elif gt == "wisdom":
        use_engine = wisdom_engine
    else:
        use_engine = engine

    payload = to_d3_force(use_engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method)

    # Optional bundle meta (single bundle won't know peer paths; pair bundling is handled by CLI/debugger)
    bundle_meta = {
        "graph_type": gt,
        "mode": mode,
        "insertion_method": insertion_method,
        "doc_id": doc_id,
    }

    return templates.TemplateResponse(
        "d3.html",
        {
            "request": request,
            "doc_id": doc_id,
            "mode": mode,
            "insertion_method": insertion_method,
            "embedded_data": json.dumps(payload),
            "bundle_meta": json.dumps(bundle_meta),
            "is_bundle": True,
        },
    )
@app.get("/api/viz/d3.json")
def api_viz_d3(
    doc_id: Optional[str] = None,
    mode: str = "reify",
    insertion_method: Optional[str] = None,
    graph_type: Optional[str] = None,   # NEW: knowledge|conversation|workflow|wisdom
):
    graph_type = (graph_type or "knowledge").lower()
    if graph_type == "conversation":
        use_engine = conversation_engine
    elif graph_type == "workflow":
        use_engine = workflow_engine
    elif graph_type == "wisdom":
        use_engine = wisdom_engine
    else:
        use_engine = engine

    payload = to_d3_force(use_engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method)
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
@app.post("/api/document.validate_graph", response_model=DocumentGraphValidationResult)
async def document_validate_graph(payload: DocumentGraphProposal):
    # inp =await request.json()
    # payload = documentGraphProposal.model_validate(inp['payload'])
    node_errors: Dict[str, str] = {}
    edge_errors: Dict[str, str] = {}

    # try to coerce nodes
    for n in payload.nodes:
        nid = n.get("id") or n.get("label") or "unknown"
        try:
            # IMPORTANT: your Node requires references;
            # user may send "metadata.pointers" instead.
            # so we do a small shim here:
            if "mentions" not in n:
                # try to lift from metadata.pointers -> references
                md = n.get("metadata") or {}
                ptrs = md.get("pointers") or []
                if ptrs:
                    n["mentions"] = [
                        {
                            "doc_id": payload.doc_id,
                            "collection_page_url": f"doc://{payload.doc_id}",
                            "document_page_url": f"doc://{payload.doc_id}#{p['source_cluster_id']}",
                            "insertion_method": payload.insertion_method,
                            "start_page": 1,
                            "end_page": 1,
                            "start_char": p["start_char"],
                            "end_char": (10**9 if p["end_char"] == -1 else p["end_char"]),
                            "excerpt": p.get("verbatim_text", "")[:400],
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

from graph_knowledge_engine.engine_core.models import LLMGraphExtraction, Document

class GraphUpsertLLMIn(BaseModel):
    """
    Strict LLM-conformant upsert:
    - nodes / edges must follow LLM* models (including REQUIRED references with spans).
    - endpoints may use 'nn:*' / 'ne:*' temp ids that resolve in-batch.
    - references may use document alias token ::DOC:: in URLs; we’ll de-alias to doc_id.
    """
    doc_id: str = Field(..., description="Document id to scope persistence")
    content: Optional[str] = Field(None, description="If provided and doc is new, store this as document content")
    doc_type: str = Field("text", description="Document type")
    insertion_method: str = Field("api_upsert", description="Provenance tag copied into each ReferenceSession")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="LLMNode['llm']-shaped dicts")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="LLMEdge['llm']-shaped dicts")

class DocumentGraphUpsertOut(BaseModel):
    document_id: str
    node_ids: List[str]
    edge_ids: List[str]
    nodes_added: int
    edges_added: int

@app.post("/api/graph/upsert", response_model=DocumentGraphUpsertOut)
def api_graph_upsert_llm(inp: GraphUpsertLLMIn):
    """
    Upsert a (hyper)graph in one shot:
    - High level api with less control and more defaults
    - Validates against the LLM Graph schema (which REQUIRES non-empty 'references').
    - Injects `insertion_method` into every ReferenceSession.
    - Supports hyperedges (edge->edge) in a single batch via engine’s topo-sorted ingest.
    - insertion method and other non llm fields such as backend/ frontend fields are dropped, and will be inserted with overrides
       e.g. insertion method will be overridden with llm_graph_extraction because it assume data is uploaded by llm
    - null doc id and content if just upload a node edge graph without dangling node/edges
    
    """
    require_role("rw")
    # 1) Ensure the document row exists (idempotent) but metadata are all missing if created in this fun
    if inp.content is not None:
        engine.write.add_document(
            Document(
                id=inp.doc_id,
                content=inp.content,
                type=inp.doc_type,
                metadata={},          # if you want
                embeddings=None,      # REQUIRED by Pydantic because Field(...)
                source_map=None,      # REQUIRED by Pydantic because Field(...)
                domain_id=None,
                processed=False
            )
        )
    else:
        # create a placeholder if completely missing so refs can anchor to doc_id
        if not engine.extract.fetch_document_text(inp.doc_id):
            engine.write.add_document(
                Document(
                    id=inp.doc_id,
                    content=inp.content,
                    type=inp.doc_type,
                    metadata={},          # if you want
                    embeddings=None,      # REQUIRED by Pydantic because Field(...)
                    source_map=None,      # REQUIRED by Pydantic because Field(...)
                    domain_id=None,
                    processed=False
                )
        )

    # 2) Validate to your LLM models (references REQUIRED by GraphEntityBase)
    #    This matches your extractor path which later calls FromLLMSlice.
    try:
        parsed = LLMGraphExtraction.FromLLMSlice({
            "nodes": inp.nodes,
            "edges": inp.edges,
        }, insertion_method=inp.insertion_method)
    except Exception as _e:
        llm_like = LLMGraphExtraction.model_validate({
            "nodes": inp.nodes,
            "edges": inp.edges,
        })

        # 3) Copy insertion_method into every ReferenceSession (backend-only field),
        #    exactly like kg_extract does, so refs carry provenance.
        parsed = LLMGraphExtraction.FromLLMSlice(llm_like, insertion_method=inp.insertion_method)

    # 4) Persist using your first-class persistence path (allocates nn:/ne:, topo-sorts, enforces endpoints)
    persisted = engine.persist.persist_graph_extraction(
        document=Document(id=inp.doc_id, content=inp.content or engine.extract.fetch_document_text(inp.doc_id) or "", type=inp.doc_type, 
                          embeddings = None, source_map = None, metadata=None, domain_id=None, processed=None),
        parsed=parsed,
        mode="append",
    )

    return DocumentGraphUpsertOut(
        document_id=persisted["document_id"],
        node_ids=persisted["node_ids"],
        edge_ids=persisted["edge_ids"],
        nodes_added=persisted["nodes_added"],
        edges_added=persisted["edges_added"],
    )
class DocumentGraphUpsert(BaseModel):
    doc_id: str
    insertion_method: str = "document_parser_v1"
    nodes: List[Node]
    edges: List[Edge] = []

class DocumentGraphUpsertResult(BaseModel):
    status: str
    inserted_nodes: int
    inserted_edges: int
    engine_result: Dict[str, Any] | None = None

class DocumentUpsert(BaseModel):
    doc_id: str
    doc_type: str
    insertion_method: str = "document_parser_v1"
    content: str # json string
class DocumentUpsertResult(BaseModel):
    status: str

@app.post("/api/document")
async def document_upsert(inp: DocumentUpsert, response_model=DocumentUpsertResult):
    if inp.doc_type == 'text':
        doc =  Document(
                id=inp.doc_id,
                content=inp.content,
                type=inp.doc_type,
                metadata={},          # if you want
                embeddings=None,      # REQUIRED by Pydantic because Field(...)
                source_map=None,      # REQUIRED by Pydantic because Field(...)
                domain_id=None,
                processed=False
            )
    elif inp.doc_type == "ocr":
        ocr_doc_dict = json.loads(inp.content)
        doc = Document.from_ocr(id=inp.doc_id, ocr_content=ocr_doc_dict, type=inp.doc_type)
    else:
        raise Exception(f"Unrecognised doc type{inp.doc_type}")
    
    if doc.metadata is None:
        
        doc.metadata = {"insertion_method": inp.insertion_method}
    else:
        doc.metadata["insertion_method"] = inp.insertion_method
    engine.write.add_document(doc)
@app.post("/api/document.upsert_tree", response_model=DocumentGraphUpsertResult)
async def document_upsert_tree(payload: DocumentGraphUpsert):
    """Upsert a generic tree with document root, use only when complete control of all backend fields are clearly known."""
    from .engine_core.models import GraphExtractionWithIDs
    try:
        res = engine.persist.persist_document_graph_extraction(
            parsed = GraphExtractionWithIDs(
                nodes=[Node.model_validate(n.model_dump(field_mode = 'backend')) for n in payload.nodes],
                edges=[Edge.model_validate(e.model_dump(field_mode = 'backend')) for e in payload.edges]),
            # insertion_method=payload.insertion_method,
            doc_id=payload.doc_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return DocumentGraphUpsertResult(
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
    
    return GraphUpsertOut.model_validate(wisdom_engine.persist_graph(parsed = pure_graph, 
                    session_id = "wisdom:"+str(stable_id("wisdom_graph", str(pure_graph.model_dump_json())))))
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

# @app.get("/viz/d3.bundle", response_class=HTMLResponse)
# def viz_d3_bundle(
#     request: Request,
#     doc_id: Optional[str] = None,
#     mode: str = "reify",
#     insertion_method: Optional[str] = None,
# ):
#     # Embed the JSON payload directly into the HTML so it can be opened offline.
#     import json as _json
#     payload = to_d3_force(engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method)
#     return templates.TemplateResponse(
#         "d3.html",
#         {
#             "request": request,
#             "doc_id": doc_id,
#             "mode": mode,
#             "insertion_method": insertion_method,
#             "embedded_data": _json.dumps(payload),
#             "is_bundle": True,
#         },
#     )

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
    engine.search_index.upsert_entries(payload.index)
    return {"ok": True}

@app.get("/api/search_index_hybrid")
def search_index_hybrid(q: str, limit: int = 10, resolve_node: bool = False):
    return engine.search_index.search_hybrid(
        q=q,
        limit=limit,
        resolve_node=resolve_node,
    )
#=====================
# Adjundicate persisted nodes
#=====================
from graph_knowledge_engine.engine_core.models import AdjudicationQuestionCode, Node, Edge, AdjudicationVerdict
from typing import Literal, Tuple
class LoadPersistedIn(BaseModel):
    doc_ids: List[str]
    insertion_method: Optional[str] = None  # e.g. "graph_extractor", "api_upsert"
    sid : bool = True

class LoadPersistedOut(BaseModel):
    node_ids: List[str]
    edge_ids: List[str]

def _load_persisted_graph(doc_ids: List[str]
                          , insertion_method: Optional[str] = None
                          ) -> LoadPersistedOut:
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
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_load_persisted(inp: LoadPersistedIn) -> LoadPersistedOut:
    """MCP: read back persisted IDs for the given docs (fast; no LLM)."""
    all_persisted = _load_persisted_graph(inp.doc_ids, insertion_method=getattr(inp, "insertion_method", None))
    if inp.sid:
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
    got = engine.backend.node_get(ids=ids, include=["documents"])
    return [Node.model_validate_json(j) for j in (got.get("documents") or [])]

def _fetch_edges(ids: List[str]) -> List[Edge]:
    if hasattr(engine, "get_edges"):
        return engine.get_edges(ids)
    # fallback
    got = engine.backend.edge_get(ids=ids, include=["documents"])
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
@require_ns(NameSpace.DOCS)
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
            return (bool(di) and bool(dj) and (di != dj))
        else:  # within-doc
            if inp.strict_crossdoc and (not di or not dj):
                return False
            return (bool(di) and bool(dj) and (di != dj))
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
@require_ns(NameSpace.DOCS)
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
            left_kind="edge" if isinstance(l, Edge) else "node" ,            # query side is a node by document
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
        res = engine.backend.node_get(where=where)  # returns dict with 'ids'
    else:
        res = engine.backend.edge_get(where=where)
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
@require_ns(NameSpace.DOCS)
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
@require_ns(NameSpace.DOCS)
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
@require_ns(NameSpace.DOCS)
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
def main() -> None:
    """Console entrypoint for `knowledge-mcp`."""
    import os
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run("graph_knowledge_engine.server_mcp_with_admin:app", host=host, port=port)


if __name__ == "__main__":
    main()
