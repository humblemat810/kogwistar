from __future__ import annotations

""" server_mcp_with_admin.py
mcp and fast rest api server for conversation service, add node edge, parsing service with authorization authentication
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from jose import jwt
from pydantic import BaseModel, Field, ValidationError, field_validator

from kogwistar.engine_core.models import (
    Document,
    Edge,
    GraphExtractionWithIDs,
    LLMGraphExtraction,
    Node,
)
from kogwistar.engine_core.search_index.models import AddIndexEntriesInput
from kogwistar.runtime.models import (
    WorkflowEdgeMetadata,
    WorkflowNodeMetadata,
)
from kogwistar.server.auth_middleware import (
    JWT_ALG,
    JWT_AUD,
    JWT_ISS,
    JWT_SECRET,
    ROLE_ORDER,
    DevStreamGuardMiddleware,
    JWTProtectMiddleware,
    NameSpace,
    get_current_subject,
    get_current_user_id,
    require_capability,
    require_namespace,
    require_role,
    require_workflow_access,
    set_auth_app,
)
from kogwistar.server.chat_api import create_chat_router
from kogwistar.server.syscall_api import create_syscall_router
from kogwistar.server.error_reporting import internal_http_error
from kogwistar.server.mcp_tools import MCPRoleMiddleware, mcp
from kogwistar.server.mcp_tools import *  # noqa: F401,F403
from kogwistar.server.resources import (
    auth_engine_resource,
    chat_service,
    conversation_engine,
    conversation_persist_directory,
    engine,
    persist_directory,
    storage_settings,
    templates,
    wisdom_engine,
    wisdom_persist_directory,
    workflow_engine,
    workflow_persist_directory,
)
from kogwistar.server.runtime_api import create_runtime_router
from kogwistar.visualization.graph_viz import to_cytoscape, to_d3_force

load_dotenv()

logger = logging.getLogger(__name__)
request_logger = logging.getLogger("kogwistar.request")

def _configure_console_logging() -> None:
    level_name = (
        os.getenv("KOGWISTAR_LOG_LEVEL")
        or os.getenv("LOG_LEVEL")
        or os.getenv("UVICORN_LOG_LEVEL")
        or "INFO"
    )
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            )
        )
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            if handler.level == logging.NOTSET or handler.level > level:
                handler.setLevel(level)

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "kogwistar.request",
        "workflow.trace",
        "workflow.runtime",
        "workflow.resolver",
    ):
        logging.getLogger(name).setLevel(level)

    logger.info("logging configured: level=%s handlers=%d", level_name, len(root.handlers))


def _install_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        started_at = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 500)
            return response
        finally:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            sink = request_logger if request_logger.handlers or request_logger.propagate else logger
            client = getattr(request, "client", None)
            client_host = getattr(client, "host", "-") or "-"
            client_port = getattr(client, "port", "-") or "-"
            sink.info(
                '%s:%s - "%s %s" %s (%dms)',
                client_host,
                client_port,
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )

try:
    from kogwistar.server.auth.db import get_session
    from kogwistar.server.auth.oidc import OIDCClient
    from kogwistar.server.auth.provider_config import (
        load_oidc_provider_configs_from_env,
    )
    from kogwistar.server.auth.router import router as auth_router
    from kogwistar.server.auth.seeding import seed_auth_data
    from kogwistar.server.auth.service import AuthService
except ModuleNotFoundError as exc:
    missing_name = str(getattr(exc, "name", "") or "")
    missing_text = str(exc)
    if (
        missing_name == "sqlalchemy"
        or missing_name.startswith("sqlalchemy.")
        or missing_text == "sqlalchemy"
        or missing_text.startswith("sqlalchemy.")
    ):
        get_session = None  # type: ignore[assignment]
        OIDCClient = None  # type: ignore[assignment]
        load_oidc_provider_configs_from_env = None  # type: ignore[assignment]
        seed_auth_data = None  # type: ignore[assignment]
        AuthService = None  # type: ignore[assignment]
        auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
    else:
        raise

mcp_app = mcp.http_app(path="/mcp")

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    _configure_console_logging()
    if (
        get_session is not None
        and AuthService is not None
        and seed_auth_data is not None
    ):
        auth_engine_resource.get()
        seed_auth_data(get_session())
        auth_mode = getattr(app.state, "auth_mode", None)
        if auth_mode is None:
            auth_mode = (
                "oidc"
                if getattr(app.state, "auth_service", None) is not None
                else (os.getenv("AUTH_MODE") or "oidc")
            )
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
        if getattr(app.state, "oidc_clients", None) is None:
            app.state.oidc_clients = {}
        if getattr(app.state, "oidc_provider_configs", None) is None:
            app.state.oidc_provider_configs = {}
        if getattr(app.state, "oidc_default_provider", None) is None:
            app.state.oidc_default_provider = None
        if (
            auth_mode != "dev"
            and OIDCClient is not None
            and load_oidc_provider_configs_from_env is not None
        ):
            default_provider, provider_configs = load_oidc_provider_configs_from_env()
            if provider_configs:
                app.state.oidc_provider_configs = provider_configs
                app.state.oidc_default_provider = default_provider
                app.state.oidc_clients = {
                    name: OIDCClient(
                        client_id=config.client_id,
                        client_secret=config.client_secret,
                        discovery_url=config.discovery_url,
                        redirect_uri=config.redirect_uri,
                        issuer=config.issuer,
                        scopes=config.scopes,
                    )
                    for name, config in provider_configs.items()
                    if config.allowed
                }

    async with mcp_app.lifespan(app):
        yield

app = FastAPI(title="KnowledgeEngine + MCP + Admin", lifespan=combined_lifespan)
set_auth_app(app)
_install_request_logging(app)

origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MCPRoleMiddleware)
app.add_middleware(JWTProtectMiddleware)
app.add_middleware(DevStreamGuardMiddleware)
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
app.include_router(
    create_syscall_router(
        get_service=lambda: chat_service.get(),
        require_role=require_role,
        require_namespace=require_namespace,
        conversation_namespace=NameSpace.CONVERSATION.value,
        workflow_namespaces={NameSpace.CONVERSATION.value, NameSpace.WORKFLOW.value},
        get_user_id=get_current_user_id,
    )
)
app.include_router(auth_router)

class DevTokenInp(BaseModel):
    username: str = "dev"
    role: str = "ro"
    ns: list[str] | str = "docs"
    capabilities: list[str] | str | None = None

    @field_validator("ns", mode="before")
    @classmethod
    def _normalize_ns(cls, value):
        allowed = {item.value for item in NameSpace}
        if value is None or value == "":
            return "docs"
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            parts = [str(item).strip() for item in value if str(item).strip()]
        else:
            return value

        for item in parts:
            if item not in allowed:
                raise ValueError(
                    f"ns must be one or more of {sorted(allowed)}, got {item!r}"
                )
        return parts[0] if len(parts) == 1 else parts

    @field_validator("capabilities", mode="before")
    @classmethod
    def _normalize_caps(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            parts = [str(item).strip() for item in value if str(item).strip()]
        else:
            return value
        return parts[0] if len(parts) == 1 else parts

@app.post("/auth/dev-token")
async def dev_token(request: Request):
    inp = DevTokenInp.model_validate((await request.json()))
    if inp.role not in ROLE_ORDER:
        raise HTTPException(400, f"role must be one of {list(ROLE_ORDER)}")
    payload = {
        "sub": inp.username,
        "ns": inp.ns,
        "role": inp.role,
        "capabilities": inp.capabilities,
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
        
        if a:=getattr(service, attr, None):
            _add(a)
        
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
                raise

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
    except RuntimeError:
        logger.warning("Chat service unavailable while collecting designer capabilities")
        service = None

    if service and hasattr(service, "workflow_catalog_ops"):
        try:
            catalog_ops = service.workflow_catalog_ops()
            for op_def in catalog_ops:
                op_name = op_def.get("op")
                if op_name:
                    builtin_ops.add(str(op_name))
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to inspect workflow catalog ops for designer capabilities: %s",
                exc,
            )

    resolver_candidates = _designer_resolver_candidates(service)
    if not resolver_candidates:
        # Fallback: use the default conversation resolver so ops are exposed even
        # when ChatRunService doesn't surface resolver attributes.
        try:
            from kogwistar.conversation.resolvers import default_resolver

            resolver_candidates = [default_resolver]
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Default resolver unavailable while collecting designer capabilities: %s",
                exc,
            )
            resolver_candidates = []

    for resolver in resolver_candidates:
        resolver_found = True
        try:
            ops = getattr(resolver, "ops", None)
            if ops is not None:
                builtin_ops.update(str(op) for op in ops)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to inspect resolver ops for designer capabilities: %s", exc
            )
        try:
            nested_ops.update(
                str(op) for op in (getattr(resolver, "nested_ops", set()) or set())
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to inspect resolver nested ops for designer capabilities: %s",
                exc,
            )
        try:
            sandboxed = getattr(resolver, "sandboxed_ops", None)
            if sandboxed is not None:
                sandbox_support = True
                sandboxed_ops.update(str(op) for op in sandboxed)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to inspect resolver sandboxed ops for designer capabilities: %s",
                exc,
            )
        try:
            describe_state = getattr(resolver, "describe_state", None)
            if callable(describe_state):
                state_schema.update(
                    {str(k): str(v) for k, v in (describe_state() or {}).items()}
                )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to inspect resolver state schema for designer capabilities: %s",
                exc,
            )
        try:
            if getattr(resolver, "_sandbox", None) is not None:
                sandbox_runtime_configured = True
                sandbox_support = True
        except AttributeError as exc:
            logger.warning(
                "Failed to inspect resolver sandbox runtime for designer capabilities: %s",
                exc,
            )

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
    require_capability("workflow.design.inspect")

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
        "pg_schema_base": storage_settings.pg_schema_base
        if storage_settings.backend == "pg"
        else None,
    }

# DELETE /admin/doc/{doc_id}  (non-MCP utility)
@app.delete("/admin/doc/{doc_id}")
def admin_delete_doc(doc_id: str):
    require_role("rw")
    eng = engine.get()
    try:
        node_ids = eng.read.node_ids_by_doc(doc_id)
        edge_ids = eng.read.edge_ids_by_doc(doc_id)
    except Exception as exc:
        logger.exception("Failed to enumerate graph rows for admin delete: doc_id=%s", doc_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to inspect document {doc_id!r} before deletion",
        ) from exc

    def _delete_step(step_name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            logger.exception(
                "Admin document delete failed at step '%s': doc_id=%s",
                step_name,
                doc_id,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document {doc_id!r} during {step_name}",
            ) from exc

    _delete_step(
        "edge_endpoints_delete",
        lambda: eng.backend.edge_endpoints_delete(where={"doc_id": doc_id}),
    )
    _delete_step(
        "node_docs_delete",
        lambda: eng.backend.node_docs_delete(where={"doc_id": doc_id}),
    )

    if edge_ids:
        _delete_step("edge_refs_delete", lambda: eng.backend.edge_refs_delete(ids=edge_ids))
        _delete_step("edge_delete", lambda: eng.backend.edge_delete(ids=edge_ids))
    else:
        _delete_step(
            "edge_delete",
            lambda: eng.backend.edge_delete(where={"doc_id": doc_id}),
        )

    if node_ids:
        _delete_step("node_refs_delete", lambda: eng.backend.node_refs_delete(ids=node_ids))
        _delete_step("node_delete", lambda: eng.backend.node_delete(ids=node_ids))
    else:
        _delete_step(
            "node_delete",
            lambda: eng.backend.node_delete(where={"doc_id": doc_id}),
        )

    _delete_step(
        "document_delete",
        lambda: eng.backend.document_delete(where={"doc_id": doc_id}),
    )

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
    insertion_method: Optional[str] = None,  # NEW
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
    payload = to_cytoscape(
        use_engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method
    )
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

    payload = to_d3_force(
        use_engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method
    )

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
    graph_type: Optional[str] = None,  # NEW: knowledge|conversation|workflow|wisdom
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

    payload = to_d3_force(
        use_engine, doc_id=doc_id, mode=mode, insertion_method=insertion_method
    )
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
                            "end_char": (
                                10**9 if p["end_char"] == -1 else p["end_char"]
                            ),
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
    return DocumentGraphValidationResult(
        ok=ok, node_errors=node_errors, edge_errors=edge_errors
    )

class GraphUpsertLLMIn(BaseModel):
    """
    Strict LLM-conformant upsert:
    - nodes / edges must follow LLM* models (including REQUIRED references with spans).
    - endpoints may use 'nn:*' / 'ne:*' temp ids that resolve in-batch.
    - references may use document alias token ::DOC:: in URLs; we’ll de-alias to doc_id.
    """

    doc_id: str = Field(..., description="Document id to scope persistence")
    content: Optional[str] = Field(
        None, description="If provided and doc is new, store this as document content"
    )
    doc_type: str = Field("text", description="Document type")
    insertion_method: str = Field(
        "api_upsert", description="Provenance tag copied into each ReferenceSession"
    )
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list, description="LLMNode['llm']-shaped dicts"
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list, description="LLMEdge['llm']-shaped dicts"
    )

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
    eng = engine.get()
    # 1) Ensure the document row exists (idempotent) but metadata are all missing if created in this fun
    if inp.content is not None:
        eng.write.add_document(
            Document(
                id=inp.doc_id,
                content=inp.content,
                type=inp.doc_type,
                metadata={},  # if you want
                embeddings=None,  # REQUIRED by Pydantic because Field(...)
                source_map=None,  # REQUIRED by Pydantic because Field(...)
                domain_id=None,
                processed=False,
            )
        )
    else:
        # create a placeholder if completely missing so refs can anchor to doc_id
        if not eng.extract.fetch_document_text(inp.doc_id):
            eng.write.add_document(
                Document(
                    id=inp.doc_id,
                    content=inp.content,
                    type=inp.doc_type,
                    metadata={},  # if you want
                    embeddings=None,  # REQUIRED by Pydantic because Field(...)
                    source_map=None,  # REQUIRED by Pydantic because Field(...)
                    domain_id=None,
                    processed=False,
                )
            )

    # 2) Validate to your LLM models (references REQUIRED by GraphEntityBase)
    #    This matches your extractor path which later calls FromLLMSlice.
    try:
        parsed = LLMGraphExtraction.FromLLMSlice(
            {
                "nodes": inp.nodes,
                "edges": inp.edges,
            },
            insertion_method=inp.insertion_method,
        )
    except Exception as _e:
        llm_like = LLMGraphExtraction.model_validate(
            {
                "nodes": inp.nodes,
                "edges": inp.edges,
            }
        )

        # 3) Copy insertion_method into every ReferenceSession (backend-only field),
        #    exactly like kg_extract does, so refs carry provenance.
        parsed = LLMGraphExtraction.FromLLMSlice(
            llm_like, insertion_method=inp.insertion_method
        )

    # 4) Persist using your first-class persistence path (allocates nn:/ne:, topo-sorts, enforces endpoints)
    persisted = eng.persist.persist_graph_extraction(
        document=Document(
            id=inp.doc_id,
            content=inp.content or eng.extract.fetch_document_text(inp.doc_id) or "",
            type=inp.doc_type,
            embeddings=None,
            source_map=None,
            metadata=None,
            domain_id=None,
            processed=None,
        ),
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
    content: str  # json string

class DocumentUpsertResult(BaseModel):
    status: str

@app.post("/api/document")
def document_upsert(inp: DocumentUpsert, response_model=DocumentUpsertResult):
    eng = engine.get()
    if inp.doc_type == "text":
        doc = Document(
            id=inp.doc_id,
            content=inp.content,
            type=inp.doc_type,
            metadata={},  # if you want
            embeddings=None,  # REQUIRED by Pydantic because Field(...)
            source_map=None,  # REQUIRED by Pydantic because Field(...)
            domain_id=None,
            processed=False,
        )
    elif inp.doc_type == "ocr":
        ocr_doc_dict = json.loads(inp.content)
        doc = Document.from_ocr(
            id=inp.doc_id, ocr_content=ocr_doc_dict, type=inp.doc_type
        )
    else:
        raise Exception(f"Unrecognised doc type{inp.doc_type}")

    if doc.metadata is None:
        doc.metadata = {"insertion_method": inp.insertion_method}
    else:
        doc.metadata["insertion_method"] = inp.insertion_method
    eng.write.add_document(doc)
    return DocumentUpsertResult(status="ok")

@app.post("/api/document.upsert_tree", response_model=DocumentGraphUpsertResult)
def document_upsert_tree(payload: DocumentGraphUpsert):
    """Persist a newly parsed graph extraction for an existing document.

    This endpoint accepts a graph extraction payload that represents nodes and
    edges before backend persistence. It is intended for extraction ingest, not
    for importing an already canonical backend graph with arbitrary stable IDs.

    ID contract:
    - This route accepts backend-shaped nodes/edges; the LLM-shaped `local_id`
      contract is honored upstream before conversion, not by this schema.
    - Node endpoints may reference same-batch `nn:<slug>` tokens, resolvable
      aliases, canonical UUIDs, or the compatibility node-label fallback.
    - Edge endpoints may reference same-batch `ne:<slug>` tokens, resolvable
      aliases, or canonical UUIDs.
    - Object `id` fields are more permissive than endpoint fields on this route.
    - The canonical ingestion table lives in
      `kogwistar/docs/ARD-postgresql-inclusive.md`.
    """
    eng = engine.get()
    try:
        res = eng.persist_document_graph_extraction(
            doc_id=payload.doc_id,
            parsed=GraphExtractionWithIDs(
                nodes=[
                    Node.model_validate(n.model_dump(field_mode="backend"))
                    for n in payload.nodes
                ],
                edges=[
                    Edge.model_validate(e.model_dump(field_mode="backend"))
                    for e in payload.edges
                ],
            ),
        )
    except Exception as e:
        raise internal_http_error(e)
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

    content: Optional[str] = Field(
        None, description="If provided and doc is new, store this as document content"
    )
    insertion_method: str = Field(
        "api_upsert", description="Provenance tag copied into each ReferenceSession"
    )
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list, description="PureNode-shaped dicts"
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list, description="PureEdge-shaped dicts"
    )
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
        {
            "request": request,
            "doc_id": doc_id,
            "mode": mode,
            "insertion_method": insertion_method,
        },
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
        {
            "request": request,
            "doc_id": doc_id,
            "mode": mode,
            "insertion_method": insertion_method,
        },
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
        {
            "request": request,
            "doc_id": doc_id,
            "mode": mode,
            "insertion_method": insertion_method,
        },
    )

from pydantic import BaseModel
class IndexingItem(BaseModel):
    node_id: str
    canonical_title: str
    keywords: List[str]
    aliases: List[str]
    provision: str
    doc_id: Optional[str]

@app.post("/api/add_index_entries")
def add_index_entries(payload: AddIndexEntriesInput):
    engine.get().search_index.upsert_entries(payload.index)
    return {"ok": True}

@app.get("/api/search_index_hybrid")
def search_index_hybrid(q: str, limit: int = 10, resolve_node: bool = False):
    return engine.get().search_index.search_hybrid(
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

    _configure_console_logging()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "28110"))
    uvicorn.run(
        "kogwistar.server_mcp_with_admin:app", host=host, port=port
    )


if __name__ == "__main__":
    main()
