from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from jose import jwt
from pydantic import BaseModel, Field, ValidationError

from graph_knowledge_engine.engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationVerdict,
    Document,
    Edge,
    GraphExtractionWithIDs,
    LLMGraphExtraction,
    Node,
)
from graph_knowledge_engine.engine_core.search_index.models import AddIndexEntriesInput
from graph_knowledge_engine.runtime.models import WorkflowEdgeMetadata, WorkflowNodeMetadata
from graph_knowledge_engine.server.auth_middleware import (
    DEFAULT_NAMESPACE,
    DEFAULT_ROLE,
    JWT_ALG,
    JWT_AUD,
    JWT_ISS,
    JWT_SECRET,
    PROTECTED_PREFIXES,
    ROLE_ORDER,
    JWTProtectMiddleware,
    NameSpace,
    Role,
    _normalize_namespaces,
    claims_ctx,
    current_role,
    get_current_namespaces,
    get_current_role,
    get_current_subject,
    get_current_user_id,
    require_namespace,
    require_role,
    require_workflow_access,
    set_auth_app,
)
from graph_knowledge_engine.server.chat_api import create_chat_router
from graph_knowledge_engine.server.mcp_tools import MCPRoleMiddleware, mcp
from graph_knowledge_engine.server.mcp_tools import *  # noqa: F401,F403
from graph_knowledge_engine.server.resources import (
    auth_engine_resource,
    chat_service,
    conversation_engine,
    conversation_gq,
    conversation_persist_directory,
    engine,
    gq,
    persist_directory,
    run_registry,
    storage_settings,
    templates,
    wisdom_engine,
    wisdom_gq,
    wisdom_persist_directory,
    workflow_engine,
    workflow_persist_directory,
)
from graph_knowledge_engine.server.runtime_api import create_runtime_router
from graph_knowledge_engine.visualization.graph_viz import to_cytoscape, to_d3_force

load_dotenv()

try:
    from graph_knowledge_engine.server.auth.db import get_session
    from graph_knowledge_engine.server.auth.oidc import OIDCClient
    from graph_knowledge_engine.server.auth.router import router as auth_router
    from graph_knowledge_engine.server.auth.seeding import seed_auth_data
    from graph_knowledge_engine.server.auth.service import AuthService
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("sqlalchemy"):
        get_session = None  # type: ignore[assignment]
        OIDCClient = None  # type: ignore[assignment]
        seed_auth_data = None  # type: ignore[assignment]
        AuthService = None  # type: ignore[assignment]
        auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
    else:
        raise

mcp_app = mcp.http_app(path="/mcp")


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    if get_session is not None and AuthService is not None and seed_auth_data is not None:
        auth_engine_resource.get()
        seed_auth_data(get_session())
        auth_mode = getattr(app.state, "auth_mode", None)
        if auth_mode is None:
            auth_mode = "oidc" if getattr(app.state, "auth_service", None) is not None else (os.getenv("AUTH_MODE") or "oidc")
        auth_mode = str(auth_mode).strip().lower()
        app.state.auth_mode = auth_mode
        if getattr(app.state, "auth_service", None) is None:
            app.state.auth_service = AuthService(
                session=get_session(),
                jwt_secret=JWT_SECRET,
                jwt_alg=JWT_ALG,
                jwt_iss=JWT_ISS,
                jwt_aud=JWT_AUD,
            )
        if getattr(app.state, "oidc_client", None) is None:
            app.state.oidc_client = None
        if auth_mode != "dev" and OIDCClient is not None:
            existing_oidc = getattr(app.state, "oidc_client", None)
            app.state.oidc_client = existing_oidc or OIDCClient(
                client_id=os.getenv("OIDC_CLIENT_ID", ""),
                client_secret=os.getenv("OIDC_CLIENT_SECRET", ""),
                discovery_url=os.getenv("OIDC_DISCOVERY_URL", ""),
                redirect_uri=os.getenv("OIDC_REDIRECT_URI", ""),
            )

    async with mcp_app.lifespan(app):
        yield


app = FastAPI(title="KnowledgeEngine + MCP + Admin", lifespan=combined_lifespan)
set_auth_app(app)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MCPRoleMiddleware)
app.add_middleware(JWTProtectMiddleware)
app.include_router(
    create_chat_router(
        get_service=lambda: chat_service.get(),
        require_role=require_role,
        require_namespace=require_namespace,
        conversation_namespace=NameSpace.CONVERSATION.value,
        workflow_namespaces={NameSpace.CONVERSATION.value, NameSpace.WORKFLOW.value},
        get_user_id=get_current_user_id,
    )
)
app.include_router(
    create_runtime_router(
        get_service=lambda: chat_service.get(),
        require_role=require_role,
        require_namespace=require_namespace,
        require_workflow_access=require_workflow_access,
        runtime_namespaces={NameSpace.WORKFLOW.value},
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
    )
)
app.include_router(auth_router)


class DevTokenInp(BaseModel):
    username: str = "dev"
    role: str = "ro"
    ns: Literal["docs", "conversation", "workflow", "wisdom"] = "docs"


@app.post("/auth/dev-token")
async def dev_token(request: Request):
    inp = DevTokenInp.model_validate((await request.json()))
    if inp.role not in ROLE_ORDER:
        raise HTTPException(400, f"role must be one of {list(ROLE_ORDER)}")
    payload = {
        "sub": inp.username,
        "ns": inp.ns,
        "role": inp.role,
        "iat": int(time.time()),
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=4)).timestamp()),
        "iss": JWT_ISS or "local",
        "aud": JWT_AUD or None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token}
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

    if service and hasattr(service, "workflow_catalog_ops"):
        try:
            catalog_ops = service.workflow_catalog_ops()
            for op_def in catalog_ops:
                op_name = op_def.get("op")
                if op_name:
                    builtin_ops.add(str(op_name))
        except Exception:
            pass

    resolver_candidates = _designer_resolver_candidates(service)
    if not resolver_candidates:
        # Fallback: use the default conversation resolver so ops are exposed even
        # when ChatRunService doesn't surface resolver attributes.
        try:
            from graph_knowledge_engine.conversation.resolvers import default_resolver
            resolver_candidates = [default_resolver]
        except Exception:
            resolver_candidates = []

    for resolver in resolver_candidates:
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
# Mount the MCP server
app.mount("/", mcp_app)

# Run with:
#   uvicorn server_mcp_with_admin:app --port 28110
def main() -> None:
    """Console entrypoint for `knowledge-mcp`."""
    import os
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "28110"))
    uvicorn.run("graph_knowledge_engine.server_mcp_with_admin:app", host=host, port=port)


if __name__ == "__main__":
    main()
