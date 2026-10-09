from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)

from fastapi import HTTPException
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from kogwistar import shortids
from kogwistar.engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationVerdict,
    Document,
    Edge,
    LLMGraphExtraction,
    Node,
    PureGraph,
)
from kogwistar.id_provider import stable_id
from kogwistar.ingester import PagewiseSummaryIngestor
from kogwistar.server.auth_middleware import (
    NameSpace,
    Role,
    _decode_role_from_headers,
    _normalize_namespaces,
    current_role,
    get_app_jwt_settings,
    get_current_namespaces,
    get_current_role,
    get_current_subject,
    get_current_user_id,
    require_role,
    require_workflow_access,
    reset_current_role,
    set_current_role,
)
from kogwistar.server.chat_mcp import (
    build_conversation_mcp,
    build_workflow_mcp,
)
from kogwistar.server.mcp_registry import McpRegistry
from kogwistar.strategies.proposer import PairKind
from kogwistar.server.resources import (
    engine,
    gq,
    wisdom_engine,
    wisdom_gq,
)
from kogwistar.strategies.proposer import VectorProposer
from kogwistar.visualization.graph_viz import to_cytoscape, to_d3_force

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

        token = set_current_role(
            _decode_role_from_headers(scope, get_app_jwt_settings(scope.get("app")))
        )
        path = str(scope.get("path") or "")
        if not path.startswith("/mcp"):
            try:
                await self.app(scope, receive, send)
            finally:
                reset_current_role(token)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_role(token)


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
        name = getattr(item, "name", None) or (
            item.get("name") or item.get("tool") or ""
            if isinstance(item, dict)
            else ""
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


mcp = McpRegistry("KnowledgeEngine + MCP + Admin", filter_tools=True)


def _server_chat_service():
    import kogwistar.server_mcp_with_admin as server

    return server.chat_service.get()


class FindEdgesOut(BaseModel):
    edges: list[str]


class NeighborsOut(BaseModel):
    nodes: list[str]
    edges: list[str]


class KHopLayer(BaseModel):
    nodes: list[str]
    edges: list[str]


class KHopOut(BaseModel):
    layers: list[KHopLayer]


class ShortestPathOut(BaseModel):
    path: list[str]


class SeedExpandOut(BaseModel):
    seeds: list[str]
    layers: list[KHopLayer]


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_find_edges(
    relation: str | None = None,
    src_label_contains: str | None = None,
    tgt_label_contains: str | None = None,
    doc_id: str | None = None,
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
def kg_neighbors(rid: str, doc_id: str | None = None) -> NeighborsOut:
    nb = gq.get().neighbors(rid, doc_id=doc_id)
    return NeighborsOut(nodes=sorted(nb["nodes"]), edges=sorted(nb["edges"]))


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_k_hop(start_ids: list[str], k: int = 1, doc_id: str | None = None) -> KHopOut:
    layers = [
        KHopLayer(nodes=sorted(L["nodes"]), edges=sorted(L["edges"]))
        for L in gq.get().k_hop(start_ids, k=k, doc_id=doc_id)
    ]
    return KHopOut(layers=layers)


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_shortest_path(
    src_id: str, dst_id: str, doc_id: str | None = None, max_depth: int = 8
) -> ShortestPathOut:
    return ShortestPathOut(
        path=gq.get().shortest_path(src_id, dst_id, doc_id=doc_id, max_depth=max_depth)
    )


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_semantic_seed_then_expand_text(
    text: str, top_k: int = 5, hops: int = 1, doc_ids: str | None = None
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


class DocParseIn(BaseModel):
    id: str | None = None
    content: str
    type: str = "text"


class DocParseOut(BaseModel):
    doc_id: str
    chunk_ids: list[str]
    summary_node_id: str | None


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def doc_parse(inp: DocParseIn) -> DocParseOut:
    require_role("rw")
    doc = Document(
        id=inp.id or str(stable_id("document", inp.content)),
        content=inp.content,
        type=inp.type,
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
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
        engine=engine.get(),
        llm=ingester_llm,
        cache_dir=str(os.path.join(".", ".llm_cache")),
    )
    res: dict[str, Any] = ingester.ingest_document(document=doc)
    return DocParseOut(
        doc_id=str(doc.id),
        chunk_ids=list(res.get("chunk_ids") or []),
        summary_node_id=res.get("final_node_id"),
    )


class KGExtractIn(BaseModel):
    id: str | None
    mode: str = "skip-if-exists"


class KGExtractOut(BaseModel):
    doc_id: str
    node_ids: list[str]
    edge_ids: list[str]
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
    docs = cast(dict[str, Any], eng.backend.document_get(ids=filenames))
    to_return = [
        {"file_name": i, "id": shortids.l2s_id(i)}
        for i in docs.get("ids", [])
    ]
    return DocIdsOut(id_mapping=to_return)


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def store_document(inp: DocParseIn):
    require_role("rw")
    eng = engine.get()
    doc = Document(
        id=inp.id or str(stable_id("document", inp.content)),
        content=inp.content,
        type=inp.type,
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
    eng.write.add_document(doc)
    return DocStoreOut.model_validate({"success": True})


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_extract(inp: KGExtractIn) -> KGExtractOut:
    require_role("rw")
    eng = engine.get()
    document_id = str(inp.id or "")
    if not document_id:
        raise ValueError("Document id is required")
    content = eng.extract.fetch_document_text(document_id)
    if not content:
        raise ValueError(f"Document '{inp.id}' not found; run store_document first.")
    from ..utils.cache_backend import Memory

    location = os.path.join(".", ".kg_extract")
    os.makedirs(location, exist_ok=True)
    memory = Memory(location=location)

    @memory.cache()
    def get_reparsed_extraction(content):
        extracted = cast(
            dict[str, Any],
            eng.extract.cached_extract_graph_with_llm(content=content),
        )
        parsed_llm: LLMGraphExtraction = extracted["parsed"]
        parsed = LLMGraphExtraction.FromLLMSlice(
            parsed_llm, insertion_method="llm_graph_extraction"
        )
        eng.persist.preflight_validate(parsed, document_id)
        return parsed

    parsed = get_reparsed_extraction(content)
    persisted: dict[str, Any] = cast(
        dict[str, Any],
        eng.persist.persist_graph_extraction(
            document=Document(
                id=document_id,
                content=content,
                type="text",
                metadata={},
                domain_id=None,
                processed=False,
                embeddings=None,
                source_map=None,
            ),
            parsed=parsed,
            mode=inp.mode,
        ),
    )
    node_ids = [str(item) for item in persisted.get("node_ids", [])]
    edge_ids = [str(item) for item in persisted.get("edge_ids", [])]
    return KGExtractOut(
        doc_id=document_id,
        node_ids=node_ids,
        edge_ids=edge_ids,
        nodes_added=int(persisted.get("nodes_added", len(node_ids)) or 0),
        edges_added=int(persisted.get("edges_added", len(edge_ids)) or 0),
    )


class CytoscapeOut(BaseModel):
    elements: list[dict]
    mode: str
    doc_id: str | None


class D3Out(BaseModel):
    nodes: list[dict]
    links: list[dict]
    mode: str
    doc_id: str | None


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_viz_cytoscape_json(
    doc_id: str | None = None, mode: str = "reify"
) -> CytoscapeOut:
    payload = to_cytoscape(engine, doc_id=doc_id, mode=mode)
    return CytoscapeOut.model_validate(payload)


@tool_roles({Role.RO, Role.RW})
@mcp.tool()
def kg_viz_d3_json(doc_id: str | None = None, mode: str = "reify") -> D3Out:
    payload = to_d3_force(engine, doc_id=doc_id, mode=mode)
    return D3Out.model_validate(payload)


class LoadPersistedIn(BaseModel):
    doc_ids: list[str]
    insertion_method: str | None = None
    sid: bool = True


class LoadPersistedOut(BaseModel):
    node_ids: list[str]
    edge_ids: list[str]


def _load_persisted_graph(
    doc_ids: list[str], insertion_method: str | None = None
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
    doc_ids: list[str]
    kind: Literal["node", "edge", "any"] = "any"
    insertion_method: str | None = None
    max_pairs_per_bucket: int = 50
    commit: bool = False
    scope: Literal["cross-doc", "within-doc"] = "cross-doc"
    strict_crossdoc: bool = True


class CrossDocAdjItem(BaseModel):
    left: str
    right: str
    left_kind: Literal["entity", "relationship"]
    right_kind: Literal["entity", "relationship"]
    same_entity: bool | None = None
    confidence: float | None = None
    reason: str | None = None
    canonical_id: str | None = None


class CrossDocAdjOut(BaseModel):
    question_key: str
    total_pairs: int
    positives: int
    negatives: int
    abstain: int
    committed_ids: list[str]
    results: list[CrossDocAdjItem]


def _fetch_nodes(ids: list[str]) -> list[Node]:
    eng = engine.get()
    if hasattr(eng, "get_nodes"):
        return eng.get_nodes(ids)
    got = cast(dict[str, Any], eng.backend.node_get(ids=ids, include=["documents"]))
    return [Node.model_validate_json(j) for j in (got.get("documents") or [])]


def _fetch_edges(ids: list[str]) -> list[Edge]:
    eng = engine.get()
    if hasattr(eng, "get_edges"):
        return eng.get_edges(ids)
    got = cast(dict[str, Any], eng.backend.edge_get(ids=ids, include=["documents"]))
    return [Edge.model_validate_json(j) for j in (got.get("documents") or [])]


def _primary_doc_of(n: Node | Edge) -> str | None:
    if getattr(n, "doc_id", None):
        return n.doc_id
    for r in n.mentions or []:
        did = getattr(r, "doc_id", None)
        if did:
            return did
    return None


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _sigtext(n_or_e: Any) -> str | None:
    props = getattr(n_or_e, "properties", None) or {}
    st = props.get("signature_text")
    return st if isinstance(st, str) else None


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_crossdoc_adjudicate_anykind(inp: CrossDocAdjIn) -> CrossDocAdjOut:
    require_role("rw")

    def _pairable(di: str | None, dj: str | None) -> bool:
        if inp.scope == "cross-doc":
            if inp.strict_crossdoc and (not di or not dj):
                return False
            return bool(di) and bool(dj) and (di != dj)
        if inp.strict_crossdoc and (not di or not dj):
            return False
        return bool(di) and bool(dj) and (di != dj)

    loaded = _load_persisted_graph(inp.doc_ids, insertion_method=inp.insertion_method)
    node_objs: list[Node] = _fetch_nodes(loaded.node_ids) if loaded.node_ids else []
    edge_objs: list[Edge] = _fetch_edges(loaded.edge_ids) if loaded.edge_ids else []

    nodes_by_key: dict[tuple[str, str], list[tuple[Node, str | None]]] = {}
    for n in node_objs:
        nodes_by_key.setdefault((n.type, _norm(n.label)), []).append(
            (n, _primary_doc_of(n))
        )

    edges_by_sig: dict[str, list[tuple[Edge, str | None]]] = {}
    edges_by_rel_label: dict[tuple[str, str], list[tuple[Edge, str | None]]] = {}
    for e in edge_objs:
        did = _primary_doc_of(e)
        st = _sigtext(e)
        if st:
            edges_by_sig.setdefault(st, []).append((e, did))
        else:
            edges_by_rel_label.setdefault(
                (e.relation or "", _norm(e.label)), []
            ).append((e, did))

    pairs: list[tuple[Any, Any]] = []

    def _cap_pairing(items: list[tuple[Any, str | None]]):
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
        nodes_by_sig: dict[str, list[tuple[Node, str | None]]] = {}
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

        edge_label_map: dict[str, list[tuple[Edge, str | None]]] = {}
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
    if all(isinstance(left, Node) and isinstance(right, Node) for left, right in pairs):
        node_pairs = cast(list[tuple[Node, Node]], pairs)
        adjudications, qkey = eng.batch_adjudicate_merges(
            node_pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY
        )
    else:
        # The batch API is intentionally node-only. Cross-kind pairs use the
        # engine's typed single-pair boundary instead of being cast to nodes.
        adjudications = [eng.adjudicate_merge(left, right) for left, right in pairs]
        qkey = str(AdjudicationQuestionCode.SAME_ENTITY.value)

    def _kind(o: Any) -> Literal["entity", "relationship"]:
        return (
            "relationship"
            if isinstance(o, Edge) or getattr(o, "relation", None)
            else "entity"
        )

    results: list[CrossDocAdjItem] = []
    pos = neg = abst = 0
    committed: list[str] = []
    for (left, right), out in zip(pairs, adjudications):
        verdict = AdjudicationVerdict.model_validate(
            getattr(out, "verdict", out)
        )
        lkind = _kind(left)
        rkind = _kind(right)
        if verdict.same_entity is True:
            pos += 1
            canonical_id = None
            if inp.commit:
                if isinstance(left, Node) and isinstance(right, Node):
                    canonical_id = eng.commit_merge(left, right, verdict)
                else:
                    canonical_id = eng.commit_any_kind(
                        eng.adjudicate.target_from_node(left)
                        if isinstance(left, Node)
                        else eng.adjudicate.target_from_edge(left),
                        eng.adjudicate.target_from_node(right)
                        if isinstance(right, Node)
                        else eng.adjudicate.target_from_edge(right),
                        verdict,
                    )
                if canonical_id:
                    committed.append(str(canonical_id))
            results.append(
                CrossDocAdjItem(
                    left=str(left.id),
                    right=str(right.id),
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
                    left=str(left.id),
                    right=str(right.id),
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
                    left=str(left.id),
                    right=str(right.id),
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
    pairs: list[ProposePair]


class ProposeVectorIn(BaseModel):
    new_node_ids: list[str] | None = None
    new_edge_ids: list[str] | None = None
    top_k: int = 10
    score_mode: Literal["distance", "similarity"] = "distance"
    max_distance: float = 0.35
    min_similarity: float = 0.65
    include_edges: bool = True
    allowed_docs: list[str] | None = None
    anchor_doc_id: str | None = None
    cross_doc_only: bool = False
    anchor_only: bool = True
    where: str | dict[str, Any] | None = None


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
        where=inp.where if isinstance(inp.where, dict) else None,
    )
    out = []
    for l, r, _score in pairs.values():
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
    kind: Literal["node", "edge"], where: dict[str, Any]
) -> set[str]:
    eng = engine.get()
    if not where:
        return set()
    if kind == "node":
        res = cast(dict[str, Any], eng.backend.node_get(where=where))
    else:
        res = cast(dict[str, Any], eng.backend.edge_get(where=where))
    return set(res.get("ids") or [])


class ProposeBruteForceIn(BaseModel):
    PairableNodeTypes: ClassVar[list[Literal["node", "edge", "any"]]] = [
        "node",
        "edge",
        "any",
    ]
    pair_kind: str = "any_any"
    allowed_docs: list[str] | None = None
    anchor_doc_id: str | None = None
    cross_doc_only: bool = False
    anchor_only: bool = True
    limit_per_bucket: int | None = 200
    where: dict | None = None


@tool_roles({Role.RO, Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def kg_propose_bruteforce(inp: ProposeBruteForceIn) -> ProposeOut:
    eng = engine.get()
    proposer = VectorProposer(eng)
    raw_pairs = proposer.propose_any_kind_any_doc(
        engine=eng,
        pair_kind=cast(PairKind, inp.pair_kind),
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
    pairs: list[ProposePair]
    commit: bool = False


@tool_roles({Role.RW})
@require_ns(NameSpace.DOCS)
@mcp.tool()
def commit_merge(inp: CrossDocAdjOut):
    require_role("rw")
    eng = engine.get()
    committed = []
    for pairs in inp.results:
        left = (
            _fetch_nodes([pairs.left])[0]
            if pairs.left_kind == "entity"
            else _fetch_edges([pairs.left])[0]
        )
        right = (
            _fetch_nodes([pairs.right])[0]
            if pairs.right_kind == "entity"
            else _fetch_edges([pairs.right])[0]
        )
        lkind, rkind, same_entity = pairs.left_kind, pairs.right_kind, pairs.same_entity
        if same_entity:
            verdict = AdjudicationVerdict(
                same_entity=bool(pairs.same_entity),
                confidence=float(pairs.confidence or 0.0),
                reason=str(pairs.reason or ""),
                canonical_entity_id=None,
            )
            if isinstance(left, Node) and isinstance(right, Node):
                canonical_id = eng.commit_merge(left, right, verdict)
            else:
                canonical_id = eng.commit_any_kind(
                    eng.adjudicate.target_from_node(left)
                    if isinstance(left, Node)
                    else eng.adjudicate.target_from_edge(left),
                    eng.adjudicate.target_from_node(right)
                    if isinstance(right, Node)
                    else eng.adjudicate.target_from_edge(right),
                    verdict,
                )
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
    pairs: list[tuple[Node | Edge, Node | Edge]] = []
    for i, pair_info in enumerate(inp.pairs):

        def fetch_any(identifier: str, kind: Literal["node", "edge"]) -> Node | Edge:
            if kind == "node":
                found = _fetch_nodes([identifier])
            else:
                found = _fetch_edges([identifier])
            if not found:
                raise ValueError(f"Unknown {kind} id: {identifier}")
            return found[0]

        l = fetch_any(pair_info.left_id, pair_info.left_kind)
        r = fetch_any(pair_info.right_id, pair_info.right_kind)
        pairs.append((l, r))

    if all(isinstance(left, Node) and isinstance(right, Node) for left, right in pairs):
        node_pairs = cast(list[tuple[Node, Node]], pairs)
        adjudications, qkey = eng.batch_adjudicate_merges(
            node_pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY
        )
    else:
        adjudications = [eng.adjudicate_merge(left, right) for left, right in pairs]
        qkey = str(AdjudicationQuestionCode.SAME_ENTITY.value)

    def _kind(o: Any) -> Literal["entity", "relationship"]:
        return (
            "relationship"
            if isinstance(o, Edge) or getattr(o, "relation", None)
            else "entity"
        )

    results: list[CrossDocAdjItem] = []
    pos = neg = abst = 0
    committed: list[str] = []
    for (left, right), out in zip(pairs, adjudications):
        verdict = AdjudicationVerdict.model_validate(
            getattr(out, "verdict", out)
        )
        lkind = _kind(left)
        rkind = _kind(right)
        if verdict.same_entity is True:
            pos += 1
            canonical_id = None
            if inp.commit:
                if isinstance(left, Node) and isinstance(right, Node):
                    canonical_id = eng.commit_merge(left, right, verdict)
                else:
                    canonical_id = eng.commit_any_kind(
                        eng.adjudicate.target_from_node(left)
                        if isinstance(left, Node)
                        else eng.adjudicate.target_from_edge(left),
                        eng.adjudicate.target_from_node(right)
                        if isinstance(right, Node)
                        else eng.adjudicate.target_from_edge(right),
                        verdict,
                    )
                if canonical_id:
                    committed.append(str(canonical_id))
            results.append(
                CrossDocAdjItem(
                    left=str(left.id),
                    right=str(right.id),
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
                    left=str(left.id),
                    right=str(right.id),
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
                    left=str(left.id),
                    right=str(right.id),
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
    content: str | None = Field(
        None, description="If provided and doc is new, store this as document content"
    )
    insertion_method: str = Field(
        "api_upsert", description="Provenance tag copied into each ReferenceSession"
    )
    nodes: list[dict[str, Any]] = Field(
        default_factory=list, description="PureNode-shaped dicts"
    )
    edges: list[dict[str, Any]] = Field(
        default_factory=list, description="PureEdge-shaped dicts"
    )


class GraphUpsertOut(BaseModel):
    node_ids: list[str]
    edge_ids: list[str]
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
