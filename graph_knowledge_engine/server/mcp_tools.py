from __future__ import annotations

import functools
import json
import os
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    TypeVar,
)

from fastapi import HTTPException

try:
    from fastmcp import FastMCP
    from fastmcp.tools.function_tool import FunctionTool
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Knowledge MCP support requires the optional 'server' extra. "
        "Install with: pip install 'kogwistar[server]'"
    ) from exc
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from graph_knowledge_engine import shortids
from graph_knowledge_engine.engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationVerdict,
    Document,
    Edge,
    LLMGraphExtraction,
    Node,
    PureGraph,
)
from graph_knowledge_engine.id_provider import stable_id
from graph_knowledge_engine.ingester import PagewiseSummaryIngestor
from graph_knowledge_engine.server.auth_middleware import (
    NameSpace,
    Role,
    _decode_role_from_headers,
    _normalize_namespaces,
    current_role,
    get_current_namespaces,
    get_current_role,
    get_current_subject,
    get_current_user_id,
    require_role,
    require_workflow_access,
)
from graph_knowledge_engine.server.chat_mcp import (
    build_conversation_mcp,
    build_workflow_mcp,
)
from graph_knowledge_engine.server.resources import (
    engine,
    gq,
    wisdom_engine,
    wisdom_gq,
)
from graph_knowledge_engine.strategies.proposer import VectorProposer
from graph_knowledge_engine.visualization.graph_viz import to_cytoscape, to_d3_force

TOOL_ROLES: dict[str, set[str]] = {}
TOOL_NAMESPACE: dict[str, set[str]] = {}
P = ParamSpec("P")
R = TypeVar("R")


def tool_roles(roles: set[Role] | Role) -> Callable[[Callable[P, R]], Callable[P, R]]:
    allowed = {roles} if isinstance(roles, Role) else set(roles)

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        name = getattr(fn, "name", None) or getattr(fn, "__name__", None)
        if name is None:
            raise Exception("name not found")
        TOOL_ROLES[name] = {r.value for r in allowed}
        original_fn = fn

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            user_role = get_current_role()
            if user_role not in allowed:
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden: role {user_role} not permitted call this tool",
                )
            return original_fn(*args, **kwargs)

        fn = wrapper
        return fn

    return deco


class MCPRoleMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        token = current_role.set(_decode_role_from_headers(scope))
        started = {}
        body_chunks: list[bytes] = []

        async def _send(message):
            if message["type"] == "http.response.start":
                started["message"] = message
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"") or b""
                more = message.get("more_body", False)
                body_chunks.append(chunk)
                if not more:
                    raw = b"".join(body_chunks)
                    if raw.startswith(b"event: message\r\n"):
                        raw_lines = raw.split(b"\r\n")
                        try:
                            _, json_str = raw_lines[1].decode("utf-8").split("data: ")
                            prefix = "data: "
                            data = json.loads(json_str)
                            changed = False
                            res = data.get("result")
                            if isinstance(res, dict) and isinstance(
                                res.get("tools"), list
                            ):
                                res["tools"] = _filter_tool_list(res["tools"])
                                changed = True
                            elif isinstance(data.get("tools"), list):
                                data["tools"] = _filter_tool_list(data["tools"])
                                changed = True
                            if changed:
                                raw_lines[1] = (prefix + json.dumps(data)).encode(
                                    "utf-8"
                                )
                                raw = b"\r\n".join(raw_lines)
                        except Exception:
                            pass

                    await send(
                        started.get(
                            "message",
                            {
                                "type": "http.response.start",
                                "status": 200,
                                "headers": [],
                            },
                        )
                    )
                    await send(
                        {"type": "http.response.body", "body": raw, "more_body": False}
                    )
            else:
                await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            current_role.reset(token)


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
    nss = get_current_namespaces()
    out = []
    for item in lst:
        name = (
            getattr(item, "name", None)
            if type(item) is FunctionTool
            else None or item.get("name") or item.get("tool") or ""
        )
        if name:
            tool_roles_value = {Role.RO.value}
            tool_namespace = {NameSpace.DOCS.value}
            for candidate in _tool_name_candidates(str(name)):
                tool_roles_value = TOOL_ROLES.get(candidate, tool_roles_value)
                tool_namespace = TOOL_NAMESPACE.get(candidate, tool_namespace)
            if role in tool_roles_value and (
                "*" in nss or not tool_namespace.isdisjoint(nss)
            ):
                out.append(item)
    return out


try:
    from fastmcp.server.middleware import Middleware  # type: ignore
except Exception:  # pragma: no cover

    class Middleware:  # type: ignore[no-redef]
        pass


def _tool_allowed(tool_name: str, *, role: str, namespaces: set[str]) -> bool:
    allowed_roles = {Role.RO.value}
    allowed_namespaces = {NameSpace.DOCS.value}
    for candidate in _tool_name_candidates(str(tool_name)):
        allowed_roles = TOOL_ROLES.get(candidate, allowed_roles)
        allowed_namespaces = TOOL_NAMESPACE.get(candidate, allowed_namespaces)
    if role not in allowed_roles:
        return False
    if "*" in namespaces:
        return True
    return not allowed_namespaces.isdisjoint(namespaces)


class RbacMiddleware(Middleware):
    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        role = get_current_role()
        nss = get_current_namespaces()
        out = []
        for t in list(tools):
            name = getattr(t, "name", None) or (
                t.get("name") if isinstance(t, dict) else None
            )
            if not name or _tool_allowed(str(name), role=role, namespaces=nss):
                out.append(t)
        return out

    async def on_call_tool(self, context, call_next):
        role = get_current_role()
        nss = get_current_namespaces()
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
        if tool_name and not _tool_allowed(str(tool_name), role=role, namespaces=nss):
            raise HTTPException(
                status_code=403,
                detail=f"Tool '{tool_name}' not permitted for role '{role}' in namespaces {nss}",
            )
        return await call_next(context)


def require_ns(
    expected: set[NameSpace] | NameSpace,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    allowed = _normalize_namespaces(expected)

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        name = getattr(fn, "name", None) or getattr(fn, "__name__", None)
        if name is None:
            raise Exception("name not found")
        TOOL_NAMESPACE[name] = allowed
        original_fn = fn

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            actuals = get_current_namespaces()
            if "*" not in actuals and allowed.isdisjoint(actuals):
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden: namespaces {actuals} cannot call this tool (requires one of {allowed})",
                )
            return original_fn(*args, **kwargs)

        fn = wrapper
        return fn

    return deco


mcp = FastMCP("KnowledgeEngine + MCP + Admin", middleware=[RbacMiddleware()])


def _server_chat_service():
    from graph_knowledge_engine import server_mcp_with_admin as server

    return server.chat_service.get()


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
    eids = gq.get().find_edges(
        relation=relation,
        src_label_contains=src_label_contains,
        tgt_label_contains=tgt_label_contains,
        doc_id=doc_id,
    )
    return FindEdgesOut(edges=eids)


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_neighbors(rid: str, doc_id: Optional[str] = None) -> NeighborsOut:
    nb = gq.get().neighbors(rid, doc_id=doc_id)
    return NeighborsOut(nodes=sorted(nb["nodes"]), edges=sorted(nb["edges"]))


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_k_hop(start_ids: List[str], k: int = 1, doc_id: Optional[str] = None) -> KHopOut:
    layers = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"]))
        for L in gq.get().k_hop(start_ids, k=k, doc_id=doc_id)
    ]
    return KHopOut(layers=layers)


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_shortest_path(
    src_id: str, dst_id: str, doc_id: Optional[str] = None, max_depth: int = 8
) -> ShortestPathOut:
    return ShortestPathOut(
        path=gq.get().shortest_path(src_id, dst_id, doc_id=doc_id, max_depth=max_depth)
    )


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_semantic_seed_then_expand_text(
    text: str, top_k: int = 5, hops: int = 1, doc_ids: Optional[str] = None
) -> SeedExpandOut:
    doc_ids_coerced = None
    if doc_ids:
        doc_ids_coerced = (
            shortids.s2l_id(doc_ids)
            if type(doc_ids is str)
            else [shortids.s2l_id(i) for i in doc_ids]
        )
    out = gq.get().semantic_seed_then_expand_text(
        text, top_k=top_k, hops=hops, doc_ids=doc_ids_coerced
    )
    layers = [
        {
            "nodes": [shortids.l2s_doc(n) for n in L["nodes"]],
            "edges": [shortids.l2s_doc(n) for n in L["edges"]],
        }
        for L in out["layers"]
    ]
    layers2 = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"])) for L in layers
    ]
    return SeedExpandOut(
        seeds=[shortids.l2s_doc(i) for i in out["seeds"]], layers=layers2
    )


class DocParseIn(Document["dto"]):
    id: Optional[str]  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    content: str
    type: str = "text"


class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: List[str]
    summary_node_id: Optional[str]


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    require_role("rw")
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        raise RuntimeError(
            "doc_parse requires optional dependency group 'gemini'. Install with: pip install 'kogwistar[gemini]'"
        ) from e
    ingester_llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro"),
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    ingester = PagewiseSummaryIngestor(
        engine=engine, llm=ingester_llm, cache_dir=str(os.path.join(".", ".llm_cache"))
    )
    res: dict = ingester.ingest_document(document=doc)
    return DocParseOut(
        doc_id=doc.id,
        chunk_ids=res.get("chunk_ids"),
        summary_node_id=res.get("final_node_id"),
    )


class KGExtractIn(BaseModel):
    id: Optional[str]
    mode: str = "skip-if-exists"


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
    eng = engine.get()
    if type(file_name) is str:
        filenames = [file_name]
    elif type(file_name) is list:
        filenames = file_name
    else:
        filenames = []
    docs = eng.backend.document_get(ids=filenames)
    to_return = [{"file_name": i, "id": shortids.l2s_id(i)} for i in docs["ids"]]
    return DocIdsOut(id_mapping=to_return)


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def store_document(inp: DocParseIn):
    require_role("rw")
    eng = engine.get()
    doc = Document(id=inp.id, content=inp.content, type=inp.type)
    eng.write.add_document(doc)
    return DocStoreOut.model_validate({"success": True})


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    require_role("rw")
    eng = engine.get()
    content = eng.extract.fetch_document_text(inp.id)
    if not content:
        raise ValueError(f"Document '{inp.id}' not found; run store_document first.")
    from joblib import Memory

    location = os.path.join(".", ".kg_extract")
    os.makedirs(location, exist_ok=True)
    memory = Memory(location=location)

    @memory.cache()
    def get_reparsed_extraction(content):
        extracted = eng.extract.cached_extract_graph_with_llm(content=content)
        parsed_llm: LLMGraphExtraction["llm"] = extracted["parsed"]
        parsed = LLMGraphExtraction.FromLLMSlice(
            parsed_llm, insertion_method="llm_graph_extraction"
        )
        eng.persist.preflight_validate(parsed, inp.id)
        return parsed

    parsed = get_reparsed_extraction(content)
    persisted = eng.persist.persist_graph_extraction(
        document=Document(id=inp.id, content=content, type="text"),
        parsed=parsed,
        mode=inp.mode,
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
def kg_viz_cytoscape_json(
    doc_id: Optional[str] = None, mode: str = "reify"
) -> CytoscapeOut:
    payload = to_cytoscape(engine, doc_id=doc_id, mode=mode)
    return CytoscapeOut.model_validate(payload)


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_viz_d3_json(doc_id: Optional[str] = None, mode: str = "reify") -> D3Out:
    payload = to_d3_force(engine, doc_id=doc_id, mode=mode)
    return D3Out.model_validate(payload)


class LoadPersistedIn(BaseModel):
    doc_ids: List[str]
    insertion_method: Optional[str] = None
    sid: bool = True


class LoadPersistedOut(BaseModel):
    node_ids: List[str]
    edge_ids: List[str]


def _load_persisted_graph(
    doc_ids: List[str], insertion_method: Optional[str] = None
) -> LoadPersistedOut:
    eng = engine.get()
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    for did in doc_ids:
        if insertion_method:
            node_ids.update(
                eng.nodes_by_doc(did, where={"insertion_method": insertion_method})
            )
            edge_ids.update(
                eng.edges_by_doc(did, where={"insertion_method": insertion_method})
            )
        else:
            node_ids.update(eng.node_ids_by_doc(did))
            edge_ids.update(eng.edge_ids_by_doc(did))
    return LoadPersistedOut(node_ids=sorted(node_ids), edge_ids=sorted(edge_ids))


@tool_roles({Role.RO, Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_load_persisted(inp: LoadPersistedIn) -> LoadPersistedOut:
    all_persisted = _load_persisted_graph(
        inp.doc_ids, insertion_method=getattr(inp, "insertion_method", None)
    )
    if inp.sid:
        all_persisted.node_ids = sorted(
            shortids.l2s_id(nid) for nid in all_persisted.node_ids
        )
        all_persisted.edge_ids = sorted(
            shortids.l2s_id(eid) for eid in all_persisted.edge_ids
        )
    return all_persisted


class CrossDocAdjIn(BaseModel):
    doc_ids: List[str]
    kind: Literal["node", "edge", "any"] = "any"
    insertion_method: Optional[str] = None
    max_pairs_per_bucket: int = 50
    commit: bool = False
    scope: Literal["cross-doc", "within-doc"] = "cross-doc"
    strict_crossdoc: bool = True


class CrossDocAdjItem(BaseModel):
    left: str
    right: str
    left_kind: Literal["entity", "relationship"]
    right_kind: Literal["entity", "relationship"]
    same_entity: Optional[bool]
    confidence: Optional[float] = None
    reason: Optional[str] = None
    canonical_id: Optional[str] = None


class CrossDocAdjOut(BaseModel):
    question_key: str
    total_pairs: int
    positives: int
    negatives: int
    abstain: int
    committed_ids: List[str]
    results: List[CrossDocAdjItem]


def _fetch_nodes(ids: List[str]) -> List[Node]:
    eng = engine.get()
    if hasattr(eng, "get_nodes"):
        return eng.get_nodes(ids)
    got = eng.backend.node_get(ids=ids, include=["documents"])
    return [Node.model_validate_json(j) for j in (got.get("documents") or [])]


def _fetch_edges(ids: List[str]) -> List[Edge]:
    eng = engine.get()
    if hasattr(eng, "get_edges"):
        return eng.get_edges(ids)
    got = eng.backend.edge_get(ids=ids, include=["documents"])
    return [Edge.model_validate_json(j) for j in (got.get("documents") or [])]


def _primary_doc_of(n: Node) -> Optional[str]:
    if getattr(n, "doc_id", None):
        return n.doc_id
    for r in n.mentions or []:
        did = getattr(r, "doc_id", None)
        if did:
            return did
    return None


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _sigtext(n_or_e: Any) -> Optional[str]:
    props = getattr(n_or_e, "properties", None) or {}
    st = props.get("signature_text")
    return st if isinstance(st, str) else None


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_crossdoc_adjudicate_anykind(inp: CrossDocAdjIn) -> CrossDocAdjOut:
    require_role("rw")

    def _pairable(di: Optional[str], dj: Optional[str]) -> bool:
        if inp.scope == "cross-doc":
            if inp.strict_crossdoc and (not di or not dj):
                return False
            return bool(di) and bool(dj) and (di != dj)
        if inp.strict_crossdoc and (not di or not dj):
            return False
        return bool(di) and bool(dj) and (di != dj)

    loaded = _load_persisted_graph(inp.doc_ids, insertion_method=inp.insertion_method)
    node_objs: List[Node] = _fetch_nodes(loaded.node_ids) if loaded.node_ids else []
    edge_objs: List[Edge] = _fetch_edges(loaded.edge_ids) if loaded.edge_ids else []

    nodes_by_key: Dict[Tuple[str, str], List[Tuple[Node, Optional[str]]]] = {}
    for n in node_objs:
        nodes_by_key.setdefault((n.type, _norm(n.label)), []).append(
            (n, _primary_doc_of(n))
        )

    edges_by_sig: Dict[str, List[Tuple[Edge, Optional[str]]]] = {}
    edges_by_rel_label: Dict[Tuple[str, str], List[Tuple[Edge, Optional[str]]]] = {}
    for e in edge_objs:
        did = _primary_doc_of(e)
        st = _sigtext(e)
        if st:
            edges_by_sig.setdefault(st, []).append((e, did))
        else:
            edges_by_rel_label.setdefault(
                (e.relation or "", _norm(e.label)), []
            ).append((e, did))

    pairs: List[Tuple[Any, Any]] = []

    def _cap_pairing(items: List[Tuple[Any, Optional[str]]]):
        made = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                di, dj = items[i][1], items[j][1]
                if not di or not dj or di == dj:
                    continue
                if not _pairable(di, dj):
                    continue
                pairs.append((items[i][0], items[j][0]))
                made += 1
                if made >= inp.max_pairs_per_bucket:
                    return

    if inp.kind in ("node", "any"):
        for items in nodes_by_key.values():
            if len(items) >= 2:
                _cap_pairing(items)

    if inp.kind in ("edge", "any"):
        for items in edges_by_sig.values():
            if len(items) >= 2:
                _cap_pairing(items)
        for items in edges_by_rel_label.values():
            if len(items) >= 2:
                _cap_pairing(items)

    if inp.kind == "any":
        nodes_by_sig: Dict[str, List[Tuple[Node, Optional[str]]]] = {}
        for n in node_objs:
            st = _sigtext(n)
            if st:
                nodes_by_sig.setdefault(st, []).append((n, _primary_doc_of(n)))

        for st, n_items in nodes_by_sig.items():
            e_items = edges_by_sig.get(st) or []
            if not e_items or not n_items:
                continue
            made = 0
            for n, dn in n_items:
                for e, de in e_items:
                    if dn and de and dn != de:
                        pairs.append((n, e))
                        made += 1
                        if made >= inp.max_pairs_per_bucket:
                            break
                if made >= inp.max_pairs_per_bucket:
                    break

        edge_label_map: Dict[str, List[Tuple[Edge, Optional[str]]]] = {}
        for e in edge_objs:
            edge_label_map.setdefault(_norm(e.label), []).append(
                (e, _primary_doc_of(e))
            )
        for n in node_objs:
            e_items = edge_label_map.get(_norm(n.label))
            if not e_items:
                continue
            made = 0
            dn = _primary_doc_of(n)
            for e, de in e_items:
                if dn and de and dn != de:
                    pairs.append((n, e))
                    made += 1
                    if made >= inp.max_pairs_per_bucket:
                        break

    if not pairs:
        return CrossDocAdjOut(
            question_key=str(AdjudicationQuestionCode.SAME_ENTITY.value),
            total_pairs=0,
            positives=0,
            negatives=0,
            abstain=0,
            committed_ids=[],
            results=[],
        )

    eng = engine.get()
    adjudications, qkey = eng.batch_adjudicate_merges(
        pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY
    )

    def _kind(o: Any) -> Literal["entity", "relationship"]:
        return (
            "relationship"
            if isinstance(o, Edge) or getattr(o, "relation", None)
            else "entity"
        )

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
                if lkind == rkind:
                    canonical_id = eng.commit_merge(left, right, verdict)
                else:
                    canonical_id = eng.commit_any_kind(left, right, verdict)
                if canonical_id:
                    committed.append(str(canonical_id))
            results.append(
                CrossDocAdjItem(
                    left=left.id,
                    right=right.id,
                    left_kind=lkind,
                    right_kind=rkind,
                    same_entity=True,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                    canonical_id=canonical_id,
                )
            )
        elif verdict.same_entity is False:
            neg += 1
            results.append(
                CrossDocAdjItem(
                    left=left.id,
                    right=right.id,
                    left_kind=lkind,
                    right_kind=rkind,
                    same_entity=False,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            )
        else:
            abst += 1
            results.append(
                CrossDocAdjItem(
                    left=left.id,
                    right=right.id,
                    left_kind=lkind,
                    right_kind=rkind,
                    same_entity=None,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            )

    return CrossDocAdjOut(
        question_key=qkey,
        total_pairs=len(pairs),
        positives=pos,
        negatives=neg,
        abstain=abst,
        committed_ids=committed,
        results=results,
    )


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
    where: Optional[str | dict] = None


@tool_roles({Role.RO, Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def propose_vector(inp: ProposeVectorIn) -> ProposeOut:
    eng = engine.get()
    prop = VectorProposer(eng)
    pairs = prop.generate_merge_candidates(
        engine=eng,
        new_node=inp.new_node_ids,
        new_edge=inp.new_edge_ids,
        top_k=inp.top_k,
        allowed_docs=inp.allowed_docs,
        anchor_doc_id=inp.anchor_doc_id,
        cross_doc_only=inp.cross_doc_only,
        anchor_only=inp.anchor_only,
        score_mode=inp.score_mode,
        max_distance=inp.max_distance,
        min_similarity=inp.min_similarity,
        include_edges=inp.include_edges,
        where=inp.where,
    )
    out = []
    for _pair_ids, (l, r, _score) in pairs.items():
        out.append(
            ProposePair(
                left_id=getattr(l, "id", ""),
                left_kind="edge" if isinstance(l, Edge) else "node",
                right_id=getattr(r, "id", ""),
                right_kind="edge" if isinstance(r, Edge) else "node",
            )
        )
    return ProposeOut(pairs=out)


def _ids_matching_where(
    kind: Literal["node", "edge"], where: Dict[str, Any]
) -> Set[str]:
    eng = engine.get()
    if not where:
        return set()
    if kind == "node":
        res = eng.backend.node_get(where=where)
    else:
        res = eng.backend.edge_get(where=where)
    return set(res.get("ids") or [])


class ProposeBruteForceIn(BaseModel):
    PairableNodeTypes: ClassVar[List[Literal["node", "edge", "any"]]] = [
        "node",
        "edge",
        "any",
    ]
    pair_kind: str = "any_any"
    allowed_docs: Optional[List[str]] = None
    anchor_doc_id: Optional[str] = None
    cross_doc_only: bool = False
    anchor_only: bool = True
    limit_per_bucket: Optional[int] = 200
    where: Optional[dict] = None


@tool_roles({Role.RO, Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_propose_bruteforce(inp: ProposeBruteForceIn) -> ProposeOut:
    eng = engine.get()
    proposer = VectorProposer(eng)
    raw_pairs = proposer.propose_any_kind_any_doc(
        engine=eng,
        pair_kind=inp.pair_kind,
        allowed_docs=inp.allowed_docs,
        anchor_doc_id=inp.anchor_doc_id,
        cross_doc_only=inp.cross_doc_only,
        anchor_only=inp.anchor_only,
        limit_per_bucket=inp.limit_per_bucket,
    )
    if not inp.where:
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

    node_ok = _ids_matching_where("node", inp.where)
    edge_ok = _ids_matching_where("edge", inp.where)

    def _passes_where(obj: Node | Edge) -> bool:
        if isinstance(obj, Edge):
            return (not edge_ok) or (obj.id in edge_ok)
        return (not node_ok) or (obj.id in node_ok)

    filtered = []
    for l, r in raw_pairs:
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


class AdjPairsIn(BaseModel):
    pairs: List[ProposePair]
    commit: bool = False


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def commit_merge(inp: CrossDocAdjOut):
    require_role("rw")
    eng = engine.get()
    committed = []
    for pairs in inp.results:
        left, right = eng.get_nodes(pairs.left)[0], eng.get_nodes(pairs.right)[0]
        lkind, rkind, same_entity = pairs.left_kind, pairs.right_kind, pairs.same_entity
        if same_entity:
            verdict = AdjudicationVerdict(
                same_entity=pairs.same_entity,
                confidence=pairs.confidence,
                reason=pairs.reason,
                canonical_entity_id=None,
            )
            if lkind == rkind == "node":
                canonical_id = eng.commit_merge(left, right, verdict)
            else:
                canonical_id = eng.commit_any_kind(left, right, verdict)
            verdict.canonical_entity_id = canonical_id
            if canonical_id:
                committed.append(str(canonical_id))


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def adjudicate_pairs(inp: AdjPairsIn) -> CrossDocAdjOut:
    if inp.commit:
        require_role("rw")
    eng = engine.get()
    pairs: List[Tuple[Node | Edge, Node | Edge]] = [None] * len(inp.pairs)  # type: ignore
    for i, pair_info in enumerate(inp.pairs):

        def fetch_any(id, kind):
            if kind == "node":
                return _fetch_nodes([id])
            if kind == "edge":
                return _fetch_edges([id])
            return []

        l: Node | Edge = fetch_any(pair_info.left_id, pair_info.left_kind)
        r: Node | Edge = fetch_any(pair_info.right_id, pair_info.right_kind)
        pairs[i] = (l[0], r[0])

    adjudications, qkey = eng.batch_adjudicate_merges(
        pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY
    )

    def _kind(o: Any) -> Literal["entity", "relationship"]:
        return (
            "relationship"
            if isinstance(o, Edge) or getattr(o, "relation", None)
            else "entity"
        )

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
                if lkind == rkind:
                    canonical_id = eng.commit_merge(left, right, verdict)
                else:
                    canonical_id = eng.commit_any_kind(left, right, verdict)
                if canonical_id:
                    committed.append(str(canonical_id))
            results.append(
                CrossDocAdjItem(
                    left=left.id,
                    right=right.id,
                    left_kind=lkind,
                    right_kind=rkind,
                    same_entity=True,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                    canonical_id=canonical_id,
                )
            )
        elif verdict.same_entity is False:
            neg += 1
            results.append(
                CrossDocAdjItem(
                    left=left.id,
                    right=right.id,
                    left_kind=lkind,
                    right_kind=rkind,
                    same_entity=False,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            )
        else:
            abst += 1
            results.append(
                CrossDocAdjItem(
                    left=left.id,
                    right=right.id,
                    left_kind=lkind,
                    right_kind=rkind,
                    same_entity=None,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            )

    return CrossDocAdjOut(
        question_key=qkey,
        total_pairs=len(pairs),
        positives=pos,
        negatives=neg,
        abstain=abst,
        committed_ids=committed,
        results=results,
    )


class KGUpsertIn(BaseModel):
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


@tool_roles({Role.RW})
@require_ns(NameSpace.WISDOM)
@mcp.tool(name="wisdom.kg_upsert_graph")
def kg_upsert_graph_wisdom(inp: KGUpsertIn) -> GraphUpsertOut:
    inp.insertion_method = inp.insertion_method or "wisdom_runtime"
    pure_graph = PureGraph.model_validate(dict(nodes=inp.nodes, edges=inp.edges))
    return GraphUpsertOut.model_validate(
        wisdom_engine.get().persist_graph(
            parsed=pure_graph,
            session_id="wisdom:"
            + str(stable_id("wisdom_graph", str(pure_graph.model_dump_json()))),
        )
    )


@tool_roles({Role.RO, Role.RW})
@require_ns(NameSpace.WISDOM)
@mcp.tool(name="wisdom.semantic_seed_then_expand")
def wisdom_semantic_seed_then_expand(text: str, top_k: int = 10, hops: int = 2):
    return wisdom_gq.get().semantic_seed_then_expand_text(text, top_k=top_k, hops=hops)


conversation_mcp = build_conversation_mcp(
    get_service=_server_chat_service,
    tool_roles=tool_roles,
    require_ns=require_ns,
    role_ro=Role.RO,
    role_rw=Role.RW,
    ns_conversation=NameSpace.CONVERSATION,
)
workflow_mcp = build_workflow_mcp(
    get_service=_server_chat_service,
    tool_roles=tool_roles,
    require_ns=require_ns,
    role_ro=Role.RO,
    role_rw=Role.RW,
    ns_workflow=NameSpace.WORKFLOW,
    get_subject=get_current_subject,
    get_user_id=get_current_user_id,
    require_workflow_access=require_workflow_access,
)
mcp.mount(conversation_mcp)
mcp.mount(workflow_mcp)

__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
    and name
    not in {
        "functools",
        "json",
        "os",
        "product",
        "Any",
        "Callable",
        "ClassVar",
        "Dict",
        "List",
        "Literal",
        "Optional",
        "Set",
        "Tuple",
        "FunctionTool",
        "Receive",
        "Scope",
        "Send",
    }
]
