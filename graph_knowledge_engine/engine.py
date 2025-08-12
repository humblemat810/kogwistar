"""_summary_

Raises:
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
    
sample usage:
from graph_knowledge_engine.strategies import candidate_proposals as CP
from graph_knowledge_engine.strategies import adjudicators as AJ
from graph_knowledge_engine.strategies import merge_policies as MP
from graph_knowledge_engine.strategies import verifiers as VF
from graph_knowledge_engine.engine import GraphKnowledgeEngine

engine = GraphKnowledgeEngine(
    persist_directory="./chroma_db",
    candidate_generator=CP.hybrid,         # or CP.by_vector_similarity
    adjudicator=AJ.llm_pair,               # or AJ.rule_first_token / AJ.llm_batch in your batch path
    merge_policy=MP.prefer_existing_canonical,
    verifier=VF.ensemble_default,          # or VF.coverage_only / VF.strict_with_min_span
)
"""


from typing import List, Optional, Dict, Any, Tuple, TypeAlias

import chromadb
from .graph_query import GraphQuery
from chromadb import Client
from chromadb.config import Settings
from .models import (
    Node,
    Edge,
    Document,
    Domain,
    ReferenceSession,
    LLMGraphExtraction,
    LLMNode,
    LLMEdge,
    AdjudicationTarget,
    AdjudicationCandidate,
    AdjudicationQuestionCode,
    QUESTION_KEY,
    QUESTION_DESC,
    AdjudicationVerdict,
    LLMMergeAdjudication,
)
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import uuid
from joblib import Memory
import json, re, uuid

from .models import ReferenceSession, MentionVerification, LLMGraphExtraction, LLMNode, LLMEdge, Node, Edge, Document
from typing import (Callable, Optional, Tuple, Any, Dict, Iterable, Sequence,
                    List, Type, TypeVar, Union)
import math
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.utils import embedding_functions
# Optional: RapidFuzz
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Optional: Azure embeddings (only if you set env for embeddings)
try:
    from langchain_openai import AzureOpenAIEmbeddings
    _HAS_AZURE_EMB = True
except Exception:
    _HAS_AZURE_EMB = False

NodeOrEdge: TypeAlias =  Node | Edge
T = TypeVar("T", Node, Edge)
from graphlib import TopologicalSorter
import uuid, json
def _fmt_ref_short(r: dict) -> str:
    # expects a dict (already model_dump()'d) ReferenceSession
    pg = ""
    if r.get("start_page") is not None and r.get("end_page") is not None:
        if r["start_page"] == r["end_page"]:
            pg = f"p{r['start_page']}"
        else:
            pg = f"p{r['start_page']}-{r['end_page']}"
    span = ""
    if r.get("start_char") is not None and r.get("end_char") is not None:
        span = f":{r['start_char']}-{r['end_char']}"
    url = r.get("document_page_url") or r.get("collection_page_url") or ""
    snip = r.get("snippet") or ""
    snip = (snip[:60] + "…") if len(snip) > 60 else snip
    return f"{pg}{span} @{url}  “{snip}”".strip()
def _is_nn(x: str) -> bool: return isinstance(x, str) and x.startswith("nn:")
def _is_ne(x: str) -> bool: return isinstance(x, str) and x.startswith("ne:")
def _merge_refs(old_refs_json: str | None, new_refs):
    old = []
    if old_refs_json:
        try: old = json.loads(old_refs_json)
        except Exception: old = []
    # de-dup by (document_page_url, start_page, start_char, end_page, end_char)
    def key(r):
        return (
            r.get("document_page_url"),
            r.get("start_page"), r.get("start_char"),
            r.get("end_page"), r.get("end_char")
        )
    seen = {key(r): r for r in old}
    for r in (new_refs or []):
        seen[key(r.model_dump())] = r.model_dump()
    merged = list(seen.values())
    return merged, json.dumps(merged)
def _alloc_real_ids(parsed):
    """Map explicit nn:/ne: → fresh UUIDs; rewrite in-place."""
    nn2id, ne2id = {}, {}
    def map_id(x: str) -> str:
        if _is_nn(x): nn2id.setdefault(x, str(uuid.uuid4())); return nn2id[x]
        if _is_ne(x): ne2id.setdefault(x, str(uuid.uuid4())); return ne2id[x]
        return x

    for n in parsed.nodes or []:
        if n.id: n.id = map_id(n.id)
    for e in parsed.edges or []:
        if e.id: e.id = map_id(e.id)
        e.source_ids = [map_id(x) for x in (e.source_ids or [])]
        e.target_ids = [map_id(x) for x in (e.target_ids or [])]
        if hasattr(e, "source_edge_ids"):
            e.source_edge_ids = [map_id(x) for x in (e.source_edge_ids or [])]
        if hasattr(e, "target_edge_ids"):
            e.target_edge_ids = [map_id(x) for x in (e.target_edge_ids or [])]
    return nn2id, ne2id
def chroma_docs_to_pydantic(objs: dict, model_cls: Type[T]) -> List[T]:
    """
    Convert Chroma get()/query() results to a list of Pydantic models.
    
    `objs` is the dict returned by collection.get()/query(), 
    `model_cls` is Node or Edge.
    """
    docs = objs.get("documents") or []
    # query() returns [[...]] for documents, so flatten
    if docs and isinstance(docs[0], list):
        docs = docs[0]
    return [model_cls.model_validate_json(doc) for doc in docs]
def _normalize_chroma_result(objs: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Normalize Chroma get()/query() outputs into parallel lists:
    ids: List[str], documents: List[str], metadatas: List[dict]
    """
    ids = objs.get("ids") or []
    docs = objs.get("documents") or []
    metas = objs.get("metadatas") or []

    # query() returns nested lists; flatten the first level if present
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if docs and isinstance(docs[0], list):
        docs = docs[0]
    if metas and isinstance(metas[0], list):
        metas = metas[0]

    # Make lengths match (Chroma can omit metadatas if not requested)
    if not metas:
        metas = [{} for _ in docs]

    return ids, docs, metas

def chroma_to_models(objs: Dict[str, Any], model_cls: Type[T]) -> List[T]:
    """
    Convert Chroma results to a list of Pydantic models (Node/Edge).
    """
    _, docs, _ = _normalize_chroma_result(objs)
    return [model_cls.model_validate_json(doc) for doc in docs]

def chroma_to_models_with_meta(objs: Dict[str, Any], model_cls: Type[T]) -> List[Tuple[str, T, Dict[str, Any]]]:
    """
    Convert Chroma results to (id, model, metadata) tuples.
    """
    ids, docs, metas = _normalize_chroma_result(objs)
    out: List[Tuple[str, T, Dict[str, Any]]] = []
    for rid, doc, meta in zip(ids, docs, metas):
        out.append((rid, model_cls.model_validate_json(doc), meta or {}))
    return out
_DOC_URL = "document/{doc_id}"

def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

def _json_or_none(v):
    return None if v is None else json.dumps(v)


def _extract_doc_ids_from_refs(refs) -> list[str]:
    out = []
    for r in refs or []:
        did = getattr(r, "doc_id", None)
        if not did and getattr(r, "document_page_url", None):
            m = re.search(r"document/([A-Za-z0-9\-]+)", r.document_page_url)
            if m:
                did = m.group(1)
        if did:
            out.append(did)
    # unique + stable order
    return sorted(dict.fromkeys(out))
def _node_doc_and_meta(n: "Node") -> tuple[str, dict]:
    """Return (documents_string, metadata_dict) for Chroma."""
    doc = n.model_dump_json()
    meta = _strip_none({
        "doc_id": getattr(n, "doc_id", None),
        "label": n.label,
        "type": n.type,
        "summary": n.summary,
        "domain_id": n.domain_id,
        "canonical_entity_id": getattr(n, "canonical_entity_id", None),
        "properties": _json_or_none(getattr(n, "properties", None)),
        "references": _json_or_none([r.model_dump() for r in (n.references or [])]),
        # add any other flat, filterable fields you rely on
    })
    return doc, meta

def _edge_doc_and_meta(e: "Edge") -> tuple[str, dict]:
    """Return (documents_string, metadata_dict) for Chroma."""
    doc = e.model_dump_json()
    meta = _strip_none({
        "doc_id": getattr(e, "doc_id", None),
        "relation": e.relation,
        "source_ids": _json_or_none(e.source_ids),
        "target_ids": _json_or_none(e.target_ids),
        "type": e.type,
        "summary": e.summary,
        "domain_id": e.domain_id,
        "canonical_entity_id": getattr(e, "canonical_entity_id", None),
        "properties": _json_or_none(getattr(e, "properties", None)),
        "references": _json_or_none([r.model_dump() for r in (e.references or [])]),
    })
    return doc, meta

def _default_verification(note: str = "fallback span") -> MentionVerification:
    return MentionVerification(method="heuristic", is_verified=False, score=None, notes=note)

def _default_ref(doc_id: str, snippet: Optional[str] = None) -> ReferenceSession:
    return ReferenceSession(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=_DOC_URL.format(doc_id=doc_id),
        start_page=1, end_page=1, start_char=0, end_char=0,
        snippet=snippet or None,
        verification=_default_verification()
    )

def _ensure_ref_span(ref: ReferenceSession, doc_id: str) -> ReferenceSession:
    # Make sure URLs point at this doc and spans are complete
    r = ref.model_copy(deep=True)
    if not r.collection_page_url:
        r.collection_page_url = f"document_collection/{doc_id}"
    if not r.document_page_url or str(doc_id) not in r.document_page_url:
        r.document_page_url = _DOC_URL.format(doc_id=doc_id)
    # Fill span if missing/bad
    if r.end_page < r.start_page:
        r.end_page = r.start_page
    if r.start_page == r.end_page and r.end_char < r.start_char:
        r.end_char = r.start_char
    if r.start_page is None or r.end_page is None:
        r.start_page, r.end_page = 1, 1
    if r.start_char is None or r.end_char is None:
        r.start_char, r.end_char = 0, 0
    # Default verification if absent
    if r.verification is None:
        r.verification = _default_verification("no explicit verification from LLM")
    return r

def _normalize_refs(refs: Optional[List[ReferenceSession]], doc_id: str, fallback_snippet: Optional[str]) -> List[ReferenceSession]:
    if not refs or len(refs) == 0:
        return [_default_ref(doc_id, snippet=fallback_snippet)]
    return [_ensure_ref_span(ref, doc_id) for ref in refs]
import re, uuid

_UUID_RE = re.compile(r"^[0-9a-fA-F\-]{36}$")

def _is_uuid(x: str | None) -> bool:
    return bool(x and _UUID_RE.match(x))

def _is_alias(x: str | None) -> bool:
    # Accept session aliases N\d+, E\d+ and base62 N~..., E~...
    return bool(x) and (x.startswith("N") or x.startswith("E"))

def _is_new_node(x: str | None) -> bool:
    return bool(x) and x.startswith("nn:")

def _is_new_edge(x: str | None) -> bool:
    return bool(x) and x.startswith("ne:")
# Simple on-disk cache dir (optional)


def build_aliases(node_ids, edge_ids):
    node_aliases = {rid: f"N{i}" for i, rid in enumerate(node_ids, start=1)}
    edge_aliases = {rid: f"E{i}" for i, rid in enumerate(edge_ids, start=1)}
    alias_for_real = {**node_aliases, **edge_aliases}
    real_for_alias = {v: k for k, v in alias_for_real.items()}
    return alias_for_real, real_for_alias

def aliasify_graph(nodes, edges, alias_for_real):
    """Return shallow copies with ids replaced by aliases for prompt."""
    def a(rid): return alias_for_real.get(rid, rid)  # fallback: leave as-is
    aliased_nodes = [
        {
            "id": a(n["id"]),
            "label": n["label"],
            "type": n["type"],
            "summary": n.get("summary",""),
            # include minimal fields the LLM needs
        }
        for n in nodes
    ]
    aliased_edges = [
        {
            "id": a(e["id"]),
            "relation": e["relation"],
            "source_ids": [a(s) for s in e.get("source_ids", [])],
            "target_ids": [a(t) for t in e.get("target_ids", [])],
        }
        for e in edges
    ]
    return aliased_nodes, aliased_edges

def de_alias_ids(llm_result, real_for_alias):
    """Translate LLM aliases back to real UUIDs in-place."""
    def r(a): return real_for_alias.get(a, a)
    for n in llm_result.nodes:
        if n.id: n.id = r(n.id)
    for e in llm_result.edges:
        if e.id: e.id = r(e.id)
        e.source_ids = [r(x) for x in e.source_ids]
        e.target_ids = [r(x) for x in e.target_ids]
    return llm_result
from dataclasses import dataclass, field

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def uuid_to_base62(u: str) -> str:
    n = int(u.replace("-", ""), 16)
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 62)
        out.append(ALPHABET[r])
    return "".join(reversed(out))

def base62_to_uuid(s: str) -> str:
    n = 0
    for ch in s:
        n = n * 62 + ALPHABET.index(ch)
    hex128 = f"{n:032x}"
    return f"{hex128[0:8]}-{hex128[8:12]}-{hex128[12:16]}-{hex128[16:20]}-{hex128[20:]}"

@dataclass
class AliasBook:
    """Stable per-session alias book. Append-only for cache friendliness."""
    next_n: int = 1
    next_e: int = 1
    real_to_alias: dict = field(default_factory=dict)  # real_id -> alias "N#"/"E#"
    alias_to_real: dict = field(default_factory=dict)  # alias -> real_id

    def alias_for_node(self, real_id: str) -> str:
        a = self.real_to_alias.get(real_id)
        if a:
            return a
        a = f"N{self.next_n}"
        self.next_n += 1
        self.real_to_alias[real_id] = a
        self.alias_to_real[a] = real_id
        return a

    def alias_for_edge(self, real_id: str) -> str:
        a = self.real_to_alias.get(real_id)
        if a:
            return a
        a = f"E{self.next_e}"
        self.next_e += 1
        self.real_to_alias[real_id] = a
        self.alias_to_real[a] = real_id
        return a

    def assign_for_sets(self, node_ids: list[str], edge_ids: list[str]):
        for rid in node_ids:
            self.alias_for_node(rid)
        for rid in edge_ids:
            self.alias_for_edge(rid)

    def legend_delta(self, node_ids: list[str], edge_ids: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return only (real_id, alias) pairs that are NEW since last turn."""
        new_nodes, new_edges = [], []
        for rid in node_ids:
            if rid not in self.real_to_alias:
                new_nodes.append((rid, self.alias_for_node(rid)))
        for rid in edge_ids:
            if rid not in self.real_to_alias:
                new_edges.append((rid, self.alias_for_edge(rid)))
        return new_nodes, new_edges

# strategy toggle
# "session_alias" -> N#/E# with session-stable AliasBook (+ delta legend)
# "base62"        -> N~<22ch> / E~<22ch> (no legend, fully deterministic)
ID_STRATEGY = "session_alias"  # or "base62"
_DOC_ALIAS = "::DOC::"  # short, token-friendly
class GraphKnowledgeEngine:
    """High-level orchestration for extracting, storing, and adjudicating knowledge graph data."""

    #--------------------
    # Puhlic Interface
    #--------------------
    
    def node_ids_by_doc(self, doc_id: str) -> List[str]:
        # got = self.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        # metas = got.get("metadatas") or []
        # return [m["node_id"] for m in metas if m and "node_id" in m]
        return self._nodes_by_doc(doc_id)

    def edge_ids_by_doc(self, doc_id: str) -> List[str]:
        # eps = self.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        # metas = eps.get("metadatas") or []
        # # dedupe across endpoints
        # return list({m["edge_id"] for m in metas if m and "edge_id" in m})
        return self._edge_ids_by_doc(doc_id)

    @property
    def embedding_function(self):
        return self._ef
    @embedding_function.setter
    def embedding_function(self, val):
        self._ef = val
        
        
    
    def get_nodes(self, ids: Sequence[str]) -> List[Node]:
        if not ids: return []
        got = self.node_collection.get(ids=list(ids), include=["documents"])
        docs = got.get("documents") or []
        return [Node.model_validate_json(d) for d in docs]

    def get_edges(self, ids: Sequence[str]) -> List[Edge]:
        if not ids: return []
        got = self.edge_collection.get(ids=list(ids), include=["documents"])
        docs = got.get("documents") or []
        return [Edge.model_validate_json(d) for d in docs]

    def all_nodes_for_doc(self, doc_id: str) -> List[Node]:
        return self.get_nodes(self._nodes_by_doc(doc_id))

    def all_edges_for_doc(self, doc_id: str) -> List[Edge]:
        return self.get_edges(self._edge_ids_by_doc(doc_id))
    
    
    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _default_ref(doc_id, snippet):
        return _default_ref(doc_id, snippet)
        pass
    @staticmethod
    def chroma_sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop keys whose values are None. ChromaDB metadata rejects None values."""
        return _strip_none(metadata) #{k: v for k, v in metadata.items() if v is not None}
    @staticmethod
    def _strip_none(d: dict):
        return _strip_none(d)
    @staticmethod
    def _json_or_none(obj: Any) -> Optional[str]:
        return json.dumps(obj) if obj is not None else None
    def _exists_node(self, rid: str) -> bool:
        g = self.node_collection.get(ids=[rid])
        return (g.get("ids") or [None])[0] == rid

    def _exists_edge(self, rid: str) -> bool:
        g = self.edge_collection.get(ids=[rid])
        return (g.get("ids") or [None])[0] == rid

    def _exists_any(self, rid: str) -> bool:
        return self._exists_node(rid) or self._exists_edge(rid)
    def _select_doc_context(self, doc_id: str, max_nodes: int = 200, max_edges: int = 400):
        nodes = self.node_collection.get(where={"doc_id": doc_id}, include=["documents"])
        edges = self.edge_collection.get(where={"doc_id": doc_id}, include=["documents"])

        node_items = []
        for i, (nid, ndoc) in enumerate(zip(nodes.get("ids", []) or [], nodes.get("documents", []) or [])):
            if i >= max_nodes:
                break
            n = Node.model_validate_json(ndoc)
            node_items.append({"id": nid, "label": n.label, "type": n.type, "summary": n.summary})

        edge_items = []
        for i, (eid, edoc) in enumerate(zip(edges.get("ids", []) or [], edges.get("documents", []) or [])):
            if i >= max_edges:
                break
            e = Edge.model_validate_json(edoc)
            edge_items.append({"id": eid, "relation": e.relation, "source_ids": e.source_ids or [], "target_ids": e.target_ids or []})

        return node_items, edge_items

    def _preflight_validate(self, parsed: LLMGraphExtraction, doc_id: str):
        """Ensure every endpoint refers to a real id (in-batch or already in DB)."""
        self._resolve_llm_ids(doc_id, parsed)  # allocate/resolve first

        batch_node_ids = {n.id for n in parsed.nodes}
        batch_edge_ids = {e.id for e in parsed.edges}

        need_nodes, need_edges = set(), set()
        for e in parsed.edges:
            need_nodes.update(e.source_ids or [])
            need_nodes.update(e.target_ids or [])
            if getattr(e, "source_edge_ids", None):
                need_edges.update(e.source_edge_ids)
            if getattr(e, "target_edge_ids", None):
                need_edges.update(e.target_edge_ids)

        # Remove the ones we are about to write
        need_nodes -= batch_node_ids
        need_edges -= batch_edge_ids

        missing_nodes, missing_edges = set(), set()
        if need_nodes:
            got = set(self.node_collection.get(ids=list(need_nodes)).get("ids") or [])
            missing_nodes = need_nodes - got
        if need_edges:
            got = set(self.edge_collection.get(ids=list(need_edges)).get("ids") or [])
            missing_edges = need_edges - got

        if missing_nodes or missing_edges:
            raise ValueError(f"Dangling references: nodes={sorted(missing_nodes)}, edges={sorted(missing_edges)}")

        return batch_node_ids, batch_edge_ids
    def _assert_endpoints_exist(self, edge: Edge):
        need_nodes = set((edge.source_ids or []) + (edge.target_ids or []))
        if need_nodes:
            got = set(self.node_collection.get(ids=list(need_nodes)).get("ids") or [])
            if got != need_nodes:
                raise ValueError(f"Missing node endpoints: {sorted(need_nodes - got)}")

        for attr in ("source_edge_ids", "target_edge_ids"):
            ids = getattr(edge, attr, None) or []
            if ids:
                got = set(self.edge_collection.get(ids=ids).get("ids") or [])
                if got != set(ids):
                    raise ValueError(f"Missing edge endpoints in {attr}: {sorted(set(ids)-got)}")
    def _build_deps(self, parsed):
        """Dependencies only among *new* objects. Existing ids create no deps."""
        ts = TopologicalSorter()
        id2kind, id2obj = {}, {}

        # register new nodes
        for n in parsed.nodes or []:
            rid = n.id or str(uuid.uuid4())
            id2kind[rid], id2obj[rid] = "node", n
            # if it already exists, no need to sort it; but we keep it to merge refs if provided
            ts.add(rid)

        # index for quick membership
        new_ids = set(id2obj.keys())

        # register edges with deps on *new* endpoints present in this batch
        def deps_for_edge(e):
            deps = set()
            for x in (e.source_ids or []) + (e.target_ids or []):
                if x in new_ids and not self._exists_any(x):
                    deps.add(x)
            for x in getattr(e, "source_edge_ids", []) or []:
                if x in new_ids and not self._exists_any(x):
                    deps.add(x)
            for x in getattr(e, "target_edge_ids", []) or []:
                if x in new_ids and not self._exists_any(x):
                    deps.add(x)
            return deps

        for e in parsed.edges or []:
            rid = e.id or str(uuid.uuid4())
            id2kind[rid], id2obj[rid] = "edge", e
            ts.add(rid, *deps_for_edge(e))

        order = list(ts.static_order())  # raises if cycle
        return order, id2kind, id2obj

    def ingest_with_toposort(self, parsed, *, doc_id: str):
        """
        Single entrypoint:
        1) map nn:/ne: → UUIDs
        2) topologically sort by deps among new items
        3) apply per-kind rules:
            - existing node: merge refs
            - new node: add
            - existing edge: merge refs / (optional) update endpoints
            - new edge: add
        """
        _ = _alloc_real_ids(parsed)  # rewrite in place
        order, id2kind, id2obj = self._build_deps(parsed)

        nodes_added = edges_added = 0
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]

            if kind == "node":
                ln = obj
                # existing?
                if self._exists_node(rid):
                    # merge refs only if present
                    if ln.references:
                        prior = self.node_collection.get(ids=[rid], include=["documents", "metadatas"])
                        prior_meta = (prior.get("metadatas") or [None])[0] or {}
                        merged_list, merged_json = _merge_refs(prior_meta.get("references"), ln.references)
                        # update JSON & metadata refs
                        n = Node.model_validate_json(prior["documents"][0])
                        n.references = [ReferenceSession(**r) for r in merged_list]
                        self.node_collection.update(
                            ids=[rid],
                            documents=[n.model_dump_json()],
                            metadatas=[{
                                **{k:v for k,v in prior_meta.items() if v is not None},
                                "references": merged_json
                            }]
                        )
                        # sync node_docs index
                        self._index_node_docs(n)
                    continue

                # new node
                refs = _normalize_refs(ln.references, doc_id, ln.summary)
                node = Node(
                    id=rid, label=ln.label, type=ln.type, summary=ln.summary,
                    domain_id=ln.domain_id, canonical_entity_id=ln.canonical_entity_id,
                    properties=ln.properties, references=refs, doc_id=doc_id,
                )
                self.add_node(node, doc_id=doc_id)
                nodes_added += 1

            else:  # edge
                le = obj
                if self._exists_edge(rid):
                    # merge refs; optionally reconcile endpoints if you permit LLM to edit them
                    if le.references:
                        prior = self.edge_collection.get(ids=[rid], include=["documents", "metadatas"])
                        prior_meta = (prior.get("metadatas") or [None])[0] or {}
                        merged_list, merged_json = _merge_refs(prior_meta.get("references"), le.references)
                        e = Edge.model_validate_json(prior["documents"][0])
                        e.references = [ReferenceSession(**r) for r in merged_list]
                        self.edge_collection.update(
                            ids=[rid],
                            documents=[e.model_dump_json()],
                            metadatas=[{
                                **{k:v for k,v in prior_meta.items() if v is not None},
                                "references": merged_json
                            }]
                        )
                    continue

                # new edge
                refs = _normalize_refs(le.references, doc_id, le.summary)
                edge = Edge(
                    id=rid, label=le.label, type=le.type, summary=le.summary,
                    domain_id=le.domain_id, canonical_entity_id=le.canonical_entity_id,
                    properties=le.properties, references=refs, relation=le.relation,
                    source_ids=le.source_ids, target_ids=le.target_ids,
                    source_edge_ids=getattr(le, "source_edge_ids", []) or [],
                    target_edge_ids=getattr(le, "target_edge_ids", []) or [],
                    doc_id=doc_id,
                )
                self.add_edge(edge, doc_id=doc_id)
                edges_added += 1

        return {"nodes_added": nodes_added, "edges_added": edges_added}
    def _resolve_llm_ids(self, doc_id: str, parsed: LLMGraphExtraction) -> None:
        """
        In-place:
        - allocate UUIDs for all new nodes (nn:*) and new edges (ne:*),
        - de-alias existing N*/E* to UUIDs,
        - resolve edge endpoints (node + edge).
        """
        # alias book → real ids
        book = self._alias_book(doc_id)
        alias_to_real = book.alias_to_real

        def de_alias(x: str) -> str:
            if not x:
                return x
            if _is_uuid(x):
                return x
            return alias_to_real.get(x, x)

        # First pass: nodes → IDs
        nn2uuid: dict[str, str] = {}
        for n in parsed.nodes:
            tok = n.id or getattr(n, "local_id", None)
            if _is_new_node(tok):
                rid = nn2uuid.get(tok) or str(uuid.uuid4())
                nn2uuid[tok] = rid
                n.id = rid
            elif tok:
                n.id = de_alias(tok)
            else:
                n.id = str(uuid.uuid4())

        # Second pass: edges → IDs
        ne2uuid: dict[str, str] = {}
        for e in parsed.edges:
            tok = e.id
            if (not tok) or _is_new_edge(tok):
                rid = ne2uuid.get(tok or "") or str(uuid.uuid4())
                if tok:
                    ne2uuid[tok] = rid
                e.id = rid
            else:
                e.id = de_alias(tok)

        # Third pass: endpoints
        def _res(xs: list[str] | None, kind: str) -> list[str] | None:
            if not xs:
                return None
            out: list[str] = []
            for x in xs:
                if kind == "node":
                    if _is_new_node(x):
                        rid = nn2uuid.get(x)
                        if not rid:
                            raise ValueError(f"Unknown temp node id: {x}")
                        out.append(rid)
                    elif _is_alias(x) or _is_uuid(x):
                        out.append(de_alias(x))
                    else:
                        # last-resort: label match within this parsed batch
                        key = (x or "").strip().lower()
                        rid = next((n.id for n in parsed.nodes if (n.label or "").strip().lower() == key), None)
                        if not rid:
                            raise ValueError(f"Unresolvable node endpoint token: {x}")
                        out.append(rid)
                else:  # kind == "edge"
                    if _is_new_edge(x):
                        rid = ne2uuid.get(x)
                        if not rid:
                            raise ValueError(f"Unknown temp edge id: {x}")
                        out.append(rid)
                    elif _is_alias(x) or _is_uuid(x):
                        out.append(de_alias(x))
                    else:
                        raise ValueError(f"Unresolvable edge endpoint token: {x}")
            return out

        for e in parsed.edges:
            e.source_ids       = _res(e.source_ids, kind="node") or []
            e.target_ids       = _res(e.target_ids, kind="node") or []
            e.source_edge_ids  = _res(getattr(e, "source_edge_ids", None), kind="edge")
            e.target_edge_ids  = _res(getattr(e, "target_edge_ids", None), kind="edge")
    def _aliasify_for_prompt(self, doc_id: str, ctx_nodes: list[dict], ctx_edges: list[dict]):
        """Return aliased nodes/edges + prompt strings, using configured ID_STRATEGY."""
        if ID_STRATEGY == "base62":
            # Deterministic short IDs, no legend needed
            aliased_nodes = []
            for n in ctx_nodes:
                aliased_nodes.append({
                    "id": f"N~{uuid_to_base62(n['id'])}",
                    "label": n["label"], "type": n["type"], "summary": n.get("summary", "")
                })
            aliased_edges = []
            for e in ctx_edges:
                aliased_edges.append({
                    "id": f"E~{uuid_to_base62(e['id'])}",
                    "relation": e["relation"],
                    "source_ids": [f"N~{uuid_to_base62(s)}" for s in e.get("source_ids", [])],
                    "target_ids": [f"N~{uuid_to_base62(t)}" for t in e.get("target_ids", [])],
                })
            return aliased_nodes, aliased_edges, "Node aliases: (implicit base62)", "Edge aliases: (implicit base62)"

        # session_alias (cache-friendly with delta legend)
        book = self._alias_book(doc_id)
        node_ids = [n["id"] for n in ctx_nodes]
        edge_ids = [e["id"] for e in ctx_edges]
        # Only list *new* aliases to keep prompts stable
        new_nodes, new_edges = book.legend_delta(node_ids, edge_ids)

        def a_node(x): return book.real_to_alias[x]
        def a_edge(x): return book.real_to_alias[x]

        aliased_nodes = [{"id": a_node(n["id"]), "label": n["label"], "type": n["type"], "summary": n.get("summary", "")} for n in ctx_nodes]
        aliased_edges = [{"id": a_edge(e["id"]), "relation": e["relation"],
                        "source_ids": [a_node(s) for s in e.get("source_ids", [])],
                        "target_ids": [a_node(t) for t in e.get("target_ids", [])]} for e in ctx_edges]

        # Build tiny delta legend strings (only new aliases)
        if new_nodes:
            lines = [f"- {book.real_to_alias[rid]}: {next(n for n in ctx_nodes if n['id']==rid)['label']}" for rid, _ in new_nodes]
            nodes_str = "New node aliases:\n" + "\n".join(lines)
        else:
            nodes_str = "New node aliases: (none)"

        if new_edges:
            lines = []
            for rid, _ in new_edges:
                e = next(e for e in ctx_edges if e["id"] == rid)
                lines.append(f"- {book.real_to_alias[rid]}: {e['relation']}")
            edges_str = "New edge aliases:\n" + "\n".join(lines)
        else:
            edges_str = "New edge aliases: (none)"

        return aliased_nodes, aliased_edges, nodes_str, edges_str
    def _extract_graph_with_llm_aliases(self, content: str, alias_nodes_str: str, alias_edges_str: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are an expert knowledge graph extractor. "
            "Extract entities and relationships as nodes and edges in a hypergraph.\n\n"
            "Rules:\n"
            "1) When referring to existing items, use ONLY the given aliases.\n"
            "2) If creating new items, omit their id.\n"
            "3) Each node/edge MUST include at least one ReferenceSession with spans.\n"
            "4) Do not invent aliases; use the provided ones only."
            "5) Do NOT invent real UUIDs or external URLs.\n"
            "6) Each node/edge MUST include at least one ReferenceSession with spans."
            "- Always use the token '{_DOC_ALIAS}' to refer to the current document in every ReferenceSession.\n"
            "- For example: document_page_url='document/{_DOC_ALIAS}', "
            "collection_page_url='document_collection/{_DOC_ALIAS}'.\n"),
            ("human",
            "Aliases (delta for this turn):\n{alias_nodes}\n\n{alias_edges}\n\n"
            "Document chunk:\n{document}\n\n"
            "Return only the structured JSON for the schema.")
        ])
        chain = prompt | self.llm.with_structured_output(LLMGraphExtraction, include_raw=True)
        result = chain.invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content, "_DOC_ALIAS" : _DOC_ALIAS})
        return result.get("raw"), result.get("parsed"), result.get("parsing_error")
    def _de_alias_ids_in_result(self, doc_id: str, parsed: LLMGraphExtraction) -> LLMGraphExtraction:
        """Map aliases back to real UUIDs according to strategy."""
        if ID_STRATEGY == "base62":
            def r(s: str | None):
                if not s:
                    return s
                if s.startswith("N~"):
                    return base62_to_uuid(s[2:])
                if s.startswith("E~"):
                    return base62_to_uuid(s[2:])
                return s
        else:
            book = self._alias_book(doc_id)
            def r(s: str | None):
                if not s:
                    return s
                return book.alias_to_real.get(s, s)

        # mutate copy
        for n in parsed.nodes:
            if n.id: n.id = r(n.id)
        for e in parsed.edges:
            if e.id: e.id = r(e.id)
            e.source_ids = [r(x) for x in e.source_ids]
            e.target_ids = [r(x) for x in e.target_ids]
        return parsed

    def _alias_legend_strings(self, aliased_nodes, aliased_edges):
        """Tiny text blocks for the prompt, to guide the model to use aliases only."""
        if aliased_nodes:
            node_lines = [f"- {n['id']}: {n['label']} [{n['type']}] — {n.get('summary','')}" for n in aliased_nodes]
            nodes_str = "Node aliases:\n" + "\n".join(node_lines)
        else:
            nodes_str = "Node aliases: (none)"

        if aliased_edges:
            def fmt_ids(xs): return ", ".join(xs)
            edge_lines = [
                f"- {e['id']}: {e['relation']} — src[{fmt_ids(e.get('source_ids', []))}] → tgt[{fmt_ids(e.get('target_ids', []))}]"
                for e in aliased_edges
            ]
            edges_str = "Edge aliases:\n" + "\n".join(edge_lines)
        else:
            edges_str = "Edge aliases: (none)"

        return nodes_str, edges_str
    def _alias_book(self, key: str) -> AliasBook:
        if key not in self._alias_books:
            self._alias_books[key] = AliasBook()
        return self._alias_books[key]
    # ----------------------------
    # Init
    # ----------------------------
    
    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_function=None,
        proposer=None,
        adjudicator=None,            # callable(left: Node, right: Node) -> AdjudicationVerdict
        batch_adjudicator=None,      # callable(pairs) -> List[LLMMergeAdjudication]
        merge_policy=None,           # callable(left, right, verdict) -> str (canonical_id)
        verifier=None,               # callable(extracted, full_text, ref, **kw) -> ReferenceSession
    ):
        """
        embedding_function: callable(texts: List[str]) -> List[List[float]].
          If None, defaults to SentenceTransformerEmbeddingFunction with model:
          - default_st_model argument, or
          - ENV SENTENCE_TRANSFORMERS_MODEL, or
          - "all-MiniLM-L6-v2".
        """
        self.persist_directory = persist_directory
        self.query = GraphQuery(self)
        self.allow_cross_kind_adjudication = True  # can be set by user
        self.cross_kind_strategy = "reifies"       # "reifies" | "equivalent" (default "reifies")
        # to do- refractor via composition. protocol template in strategies.py, strategies helper in ./strategies/
        # strategies now are function objects
        from strategies import CompositeProposer, VectorProposer, PairAdjudicator, BatchAdjudicator, Verifier, PreferExistingCanonical
        from .strategies.adjudicators import LLMPairAdjudicatorImpl, LLMBatchAdjudicatorImpl
        from .strategies.verifiers import DefaultVerifier, VerifierConfig

        self.proposer = proposer or VectorProposer()
        self.pair_adjudicator: PairAdjudicator = adjudicator or LLMPairAdjudicatorImpl(self)
        self.batch_adjudicator: BatchAdjudicator = batch_adjudicator or LLMBatchAdjudicatorImpl(self)
        self.verifier: Verifier = verifier or DefaultVerifier(self, VerifierConfig(use_embeddings=False))
        self.merge_policy = merge_policy or PreferExistingCanonical()
        load_dotenv()
        
        self._alias_books: dict[str, AliasBook] = {}
        self._ef = embedding_function or embedding_functions.DefaultEmbeddingFunction()
        # Keep a 1-string convenience to reuse in cosine checks
        # convenience: single-string embed for verifiers
        def _embed_one(text: str):
            vecs = self._ef([text])  # DefaultEmbeddingFunction is callable(texts: List[str]) -> List[List[float]]
            return vecs[0] if vecs else None
        self._embed_one = _embed_one

        # 2) Chroma client + collections; inject embedder on vectorized collections
        self.chroma_client = Client(
            Settings(
                is_persistent=True,
                persist_directory=persist_directory or "./chroma_db",
                anonymized_telemetry=False,
            )
        )
        # IMPORTANT: pass embedding_function to vector collections
        self.node_collection = self.chroma_client.get_or_create_collection(
            "nodes", embedding_function=self._ef
        )
        self.edge_collection = self.chroma_client.get_or_create_collection(
            "edges", embedding_function=self._ef
        )
        self.edge_endpoints_collection = self.chroma_client.get_or_create_collection("edge_endpoints")
        self.document_collection = self.chroma_client.get_or_create_collection("documents")
        self.domain_collection = self.chroma_client.get_or_create_collection("domains")
        self.node_docs_collection = self.chroma_client.get_or_create_collection("node_docs")
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            cache=None,
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=12000,
            openai_api_type="azure",
        )
        self.embeddings: Optional[Callable[[str], Optional[List[float]]]] = None
        if _HAS_AZURE_EMB:
            emb_deployment = os.getenv("OPENAI_EMBED_DEPLOYMENT")
            emb_endpoint   = os.getenv("OPENAI_EMBED_ENDPOINT")
            emb_api_key    = os.getenv("OPENAI_API_KEY_GPT4_1") or os.getenv("OPENAI_API_KEY")
            emb_api_ver    = os.getenv("OPENAI_EMBED_API_VERSION", "2024-08-01-preview")
            if emb_deployment and emb_endpoint and emb_api_key:
                _emb = AzureOpenAIEmbeddings(
                    azure_deployment=emb_deployment,
                    openai_api_key=emb_api_key,
                    azure_endpoint=emb_endpoint,
                    openai_api_version=emb_api_ver,
                )
                def _embed_fn(text: str) -> Optional[List[float]]:
                    try:
                        v = _emb.embed_query(text)
                        return v
                    except Exception:
                        return None
                self.embeddings = _embed_fn
    # ----------------------------
    # Mention / citation verification
    # ----------------------------
    @staticmethod
    def _normalize(s: str) -> str:
        return " ".join((s or "").split()).strip().lower()
    

    def _alias_doc_in_prompt(self) -> str:
        # A tiny legend we show to the LLM so it knows the alias to use
        return f"Use '{_DOC_ALIAS}' whenever you need to reference the current document in ReferenceSession fields."

    def _dealias_one_ref(self, ref: ReferenceSession, real_doc_id: str) -> ReferenceSession:
        r = ref.model_copy(deep=True)
        # swap token in URLs
        if r.document_page_url and _DOC_ALIAS in r.document_page_url:
            r.document_page_url = r.document_page_url.replace(_DOC_ALIAS, real_doc_id)
        if r.collection_page_url and _DOC_ALIAS in r.collection_page_url:
            r.collection_page_url = r.collection_page_url.replace(_DOC_ALIAS, real_doc_id)
        # align explicit doc_id if model carries it
        if getattr(r, "doc_id", None) == _DOC_ALIAS or getattr(r, "doc_id", None) is None:
            r.doc_id = real_doc_id
        # basic span normalization (keeps your existing behavior)
        if getattr(r, "page", None) is not None:
            # if you split start/end page fields, keep your existing normalization here
            pass
        if r.start_char is not None and r.end_char is not None and r.end_char < r.start_char:
            r.end_char = r.start_char
        return r

    def _dealias_refs(self, refs: list[ReferenceSession] | None, real_doc_id: str, fallback_snip: str | None):
        if not refs or len(refs) == 0:
            # produce a default reference using the real doc id
            return [ReferenceSession(
                collection_page_url=f"document_collection/{real_doc_id}",
                document_page_url=f"document/{real_doc_id}",
                snippet=fallback_snip or None,
                doc_id=real_doc_id,
            )]
        return [self._dealias_one_ref(r, real_doc_id) for r in refs]
    @staticmethod
    def _slice_span(full_text: str, start: int, end: int) -> str:
        start = max(0, start or 0)
        end = max(start, end or start)
        end = min(len(full_text), end)
        return full_text[start:end]
    
    def _target_from_node(self, n: "Node") -> "AdjudicationTarget":
        return AdjudicationTarget(
            kind="node", id=n.id, label=n.label, type=n.type, summary=n.summary,
            domain_id=n.domain_id, canonical_entity_id=n.canonical_entity_id, properties=n.properties
        )

    def _target_from_edge(self, e: "Edge") -> "AdjudicationTarget":
        return AdjudicationTarget(
            kind="edge", id=e.id, label=e.label, type=e.type, summary=e.summary,
            relation=e.relation,
            source_ids=e.source_ids or [], target_ids=e.target_ids or [],
            source_edge_ids=e.source_edge_ids or [], target_edge_ids=e.target_edge_ids or [],
            domain_id=e.domain_id, canonical_entity_id=e.canonical_entity_id, properties=e.properties
        )
    def generate_cross_kind_candidates(self, scope_doc_id: str | None = None, limit_per_bucket: int = 50):
        """
        Propose node↔edge pairs where labels/summaries suggest reification.
        Heuristic: node.label or summary overlaps edge.label/summary or edge.relation text.
        """
        self.proposer.cross_kind_in_doc(engine = self, doc_id = scope_doc_id)
        
    def generate_merge_candidates_doc_brute_force(
        self,
        kind: str = "node",            # "node" or "edge"
        scope_doc_id: str | None = None,
        top_k: int = 200,
    ) -> list[AdjudicationCandidate]:
        """
        Offline candidate generator. For nodes: block by (type,label lower).
        For edges: block by (relation, sorted endpoints).
        """
        return self.proposer.same_kind_in_doc(self, doc_id = scope_doc_id, kind = kind)
        
    def _fetch_document_text(self, document_id: str) -> str:
        got = self.document_collection.get(ids=[document_id], include=["documents"])
        if got and got.get("documents"):
            return got["documents"][0] or ""
        # fallback: try lookup by where
        got = self.document_collection.get(where={"doc_id": document_id}, include=["documents"])
        if got and got.get("documents"):
            return got["documents"][0] or ""
        return ""

    @staticmethod
    def _cosine(u: List[float], v: List[float]) -> Optional[float]:
        if not u or not v or len(u) != len(v):
            return None
        dot = sum(a*b for a,b in zip(u,v))
        nu = math.sqrt(sum(a*a for a in u))
        nv = math.sqrt(sum(b*b for b in v))
        if nu == 0 or nv == 0:
            return None
        return dot / (nu * nv)

    def _score_rapidfuzz(self, a: str, b: str) -> Optional[float]:
        if not _HAS_RAPIDFUZZ:
            return None
        a, b = self._normalize(a), self._normalize(b)
        if not a or not b:
            return None
        # token_set_ratio is robust to word order/noise; scale 0..100 -> 0..1
        return float(fuzz.token_set_ratio(a, b)) / 100.0

    @staticmethod
    def _score_coverage(extracted: str, cited: str, min_ngram: int = 5) -> Optional[float]:
        """
        Percent of extracted characters covered by any n-gram (len>=min_ngram)
        that also appears in cited. Simple, fast lower bound for span support.
        """
        ex_norm = " ".join((extracted or "").split())
        ci_norm = " ".join((cited or "").split())
        if not ex_norm or not ci_norm:
            return None
        n = max(1, min_ngram)
        covered = [False] * len(ex_norm)
        # greedy n-gram cover: slide windows, mark matches
        for i in range(0, len(ex_norm) - n + 1):
            gram = ex_norm[i:i+n]
            if gram in ci_norm:
                for j in range(i, i+n):
                    covered[j] = True
        # expand matches (optional): if surrounded by covered, keep filling
        # (skip for speed; baseline is fine)
        total = len(ex_norm)
        hit = sum(1 for x in covered if x)
        return hit / total if total > 0 else None

    def _score_embedding(self, extracted: str, cited: str) -> float | None:
        u = self._embed_one(extracted or "")
        v = self._embed_one(cited or "")
        if u is None or v is None: 
            return None
        # cosine
        dot = sum(a*b for a, b in zip(u, v))
        nu = (sum(a*a for a in u)) ** 0.5
        nv = (sum(b*b for b in v)) ** 0.5
        return float((dot / (nu * nv))) if nu and nv else None

    def _ensemble(self, scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
        """Weighted average over available (non-None) scores."""
        num = 0.0
        den = 0.0
        for k, w in weights.items():
            s = scores.get(k)
            if s is None:
                continue
            num += w * s
            den += w
        if den == 0:
            return None
        return float(num / den)

    def _verify_one_reference(
        self,
        extracted_text: str,
        full_text: str,
        ref: ReferenceSession,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        threshold: float = 0.70,
    ) -> ReferenceSession:
        """
        Compute per-reference metrics & write a single ensemble MentionVerification
        back into the ReferenceSession (copy), preserving other fields.
        """
        span = self._slice_span(full_text, ref.start_char or 0, ref.end_char or 0)
        # fallbacks: if spans are empty, use entire doc (not ideal, but keeps pipeline moving)
        cited_text = span if span else full_text

        rf = self._score_rapidfuzz(extracted_text, cited_text)
        cv = self._score_coverage(extracted_text, cited_text, min_ngram=min_ngram)
        em = self._score_embedding(extracted_text, cited_text)

        score = self._ensemble({"rapidfuzz": rf, "coverage": cv, "embedding": em}, weights) or 0.0
        is_ok = score >= threshold

        out = ref.model_copy(deep=True)
        # compact, machine-readable notes for reuse later
        detail = {
            "rapidfuzz": rf,
            "coverage": cv,
            "embedding": em,
            "weights": weights,
            "threshold": threshold,
        }
        note = json.dumps(detail, separators=(",", ":"))
        if out.verification is None:
            out.verification = MentionVerification(
                method="ensemble",
                is_verified=is_ok,
                score=score,
                notes=note,
            )
        else:
            out.verification.method = "ensemble"
            out.verification.is_verified = is_ok
            out.verification.score = score
            out.verification.notes = note
        return out
    # ----------------------------
    # Chroma adders
    # ----------------------------
    def add_node(self, node: Node, doc_id: Optional[str] = None):
        node.doc_id = doc_id
        doc, meta = _node_doc_and_meta(node)
        self.node_collection.add(
            ids=[node.id],
            documents=[doc],
            embeddings=[node.embedding] if node.embedding else None,
            metadatas=[meta],
        )
        self._index_node_docs(node)
    def _fanout_endpoints_rows(self, edge: Edge, doc_id: str | None):

        def _maybe_doc_for_edge(eid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self.edge_collection.get(ids=[eid], include=["metadatas"])
            if meta.get("metadatas") and meta["metadatas"][0]:
                return meta["metadatas"][0].get("doc_id")
            return None

        rows = []

        def _per_node_doc(nid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self.node_collection.get(ids=[nid], include=["metadatas"])
            if meta.get("metadatas") and meta["metadatas"][0]:
                return meta["metadatas"][0].get("doc_id")
            return None

        rows = []
        # node endpoints
        for role, node_ids in (("src", edge.source_ids or []), ("tgt", edge.target_ids or [])):
            for nid in node_ids:
                r = {
                    "id": f"{edge.id}::{role}::node::{nid}",
                    "edge_id": edge.id,
                    "endpoint_id": nid,
                    "endpoint_type": "node",
                    "role": role,
                    "relation": edge.relation,
                }
                did = _per_node_doc(nid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})
        # edge endpoints (meta)
        for role, eids in (("src", getattr(edge, "source_edge_ids", []) or []),
                        ("tgt", getattr(edge, "target_edge_ids", []) or [])):
            for mid in eids:
                r = {
                    "id": f"{edge.id}::{role}::edge::{mid}",
                    "edge_id": edge.id,
                    "endpoint_id": mid,
                    "endpoint_type": "edge",
                    "role": role,
                    "relation": edge.relation,
                }
                did = _maybe_doc_for_edge(mid)
                if did is not None:
                    r["doc_id"] = did
                rows.append({k: v for k, v in r.items() if v is not None})

        # strip Nones just in case
        return [{k: v for k, v in r.items() if v is not None} for r in rows]
    def add_edge(self, edge: Edge, doc_id: Optional[str] = None):
        edge.doc_id = doc_id
        s_nodes, s_edges, t_nodes, t_edges = self._split_endpoints(edge.source_ids, edge.target_ids)
        edge.source_ids = s_nodes
        edge.source_edge_ids = getattr(edge, "source_edge_ids", []) or [] + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = getattr(edge, "target_edge_ids", []) or [] + t_edges
        edge.doc_id = doc_id
        # single-call safety for ad-hoc usage
        self._assert_endpoints_exist(edge)

        # receptive range counts
        node_endpoint_count = len(edge.source_ids or []) + len(edge.target_ids or [])
        edge_endpoint_count = len(getattr(edge, "source_edge_ids", []) or []) + len(getattr(edge, "target_edge_ids", []) or [])
        total_endpoint_count = node_endpoint_count + edge_endpoint_count
        
        # main edge row
        self.edge_collection.add(
            ids=[edge.id],
            documents=[edge.model_dump_json()],
            embeddings=[edge.embedding] if edge.embedding else None,
            metadatas=[_strip_none({
                "doc_id": doc_id,
                "relation": edge.relation,
                "source_ids": _json_or_none(edge.source_ids),
                "target_ids": _json_or_none(edge.target_ids),
                "source_edge_ids": _json_or_none(getattr(edge, "source_edge_ids", None)),
                "target_edge_ids": _json_or_none(getattr(edge, "target_edge_ids", None)),
                "type": edge.type,
                "summary": edge.summary,
                "domain_id": edge.domain_id,
                "canonical_entity_id": edge.canonical_entity_id,
                "properties": _json_or_none(edge.properties),
                "references": _json_or_none([r.model_dump() for r in (edge.references or [])]),
                "node_endpoint_count": node_endpoint_count,   # receptive range
                "edge_endpoint_count": edge_endpoint_count,
                "total_endpoint_count": total_endpoint_count,
            })],
        )

        # endpoints fan-out
        
        rows = self._fanout_endpoints_rows(edge, doc_id)
        if rows:
            ep_ids   = [r["id"] for r in rows]
            ep_docs  = [json.dumps(r) for r in rows]
            ep_metas = rows  # already sanitized (no None)
            self.edge_endpoints_collection.add(
                ids=ep_ids,
                documents=ep_docs,
                metadatas=ep_metas,
            )
    def add_document(self, document: Document):
        self.document_collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[_strip_none({
                "doc_id": document.id,  # <— critical
                "type": document.type,
                "metadata": _json_or_none(document.metadata),
                "domain_id": document.domain_id,
                "processed": document.processed,
            })],
        )

    def add_domain(self, domain: Domain):
        self.domain_collection.add(
            ids=[domain.id],
            documents=[domain.model_dump_json()],
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "name": domain.name,
                        "description": domain.description,
                    }
                )
            ],
        )
    def _index_node_docs(self, node: Node) -> list[str]:
        """Rebuild (node_id, doc_id) rows for this node and denormalize doc_ids on node metadata."""
        doc_ids = _extract_doc_ids_from_refs(node.references)

        # 1) Rebuild the (node_id, doc_id) index rows
        self.node_docs_collection.delete(where={"node_id": node.id})
        if doc_ids:
            ids, docs, metas = [], [], []
            for did in doc_ids:
                rid = f"{node.id}::{did}"
                row = {"id": rid, "node_id": node.id, "doc_id": did, "mention_count": 1}
                ids.append(rid)
                docs.append(json.dumps(row))
                metas.append(row)
            self.node_docs_collection.add(ids=ids, documents=docs, metadatas=metas)

        # 2) Denormalize onto node metadata (convenience only)
        #    Fetch current to avoid writing the same value repeatedly.
        current = self.node_collection.get(ids=[node.id], include=["metadatas"])
        cur_meta = (current.get("metadatas") or [None])[0] or {}
        new_doc_ids_json = json.dumps(doc_ids)
        if cur_meta.get("doc_ids") != new_doc_ids_json:
            self.node_collection.update(
                ids=[node.id],
                metadatas=[{"doc_ids": new_doc_ids_json}]
            )

        return doc_ids

    def _nodes_by_doc(self, doc_id: str) -> list[str]:
        rows = self.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return [m["node_id"] for m in (rows.get("metadatas") or [])]

    def _prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        """Remove references to doc_id from node; delete node_docs link; refresh denormalized meta."""
        got = self.node_collection.get(ids=[node_id], include=["documents", 'metadatas'])
        if not (got.get("documents") and got["documents"][0]):
            return False
        node = Node.model_validate_json(got["documents"][0])
        before = len(node.references or [])
        node.references = [r for r in (node.references or []) if r.doc_id != doc_id 
                           or   (bool(getattr(r, "document_page_url", None) 
                                 and (getattr(r, "document_page_url").rsplit('/',1)[-1] != doc_id)))]
        changed = len(node.references or []) != before
        if changed:
            # Update node JSON
            self.node_collection.update(ids=[node_id], documents=[node.model_dump_json()])
            # Drop (node,doc) link
            self.node_docs_collection.delete(where={"$and": [{"node_id": node_id}, {"doc_id": doc_id}]})
            # Refresh denormalized doc_ids meta
            self._index_node_docs(node)
        return changed
    # ----------------------------
    # helpers for rollback
    # ----------------------------
    def _edge_ids_by_doc(self, document_id: str) -> list[str]:
        eps = self.edge_endpoints_collection.get(where={"doc_id": document_id})
        if not eps["documents"]:
            return []
        return list({json.loads(doc)["edge_id"] for doc in eps["documents"]})

    def _delete_edges_by_ids(self, edge_ids: list[str]):
        if not edge_ids:
            return
        self.edge_collection.delete(ids=edge_ids)
        # also delete their endpoint rows
        # endpoints ids start with f"{edge_id}::"
        # we can delete via where on edge_id for simplicity:
        self.edge_endpoints_collection.delete(where={"edge_id": {"$in": edge_ids}})
    # ----------------------------
    # Vector queries
    # ----------------------------
    def vector_search_nodes(self, embedding: List[float], top_k: int = 5):
        return self.node_collection.query(query_embeddings=[embedding], n_results=top_k)

    def vector_search_edges(self, embedding: List[float], top_k: int = 5):
        return self.edge_collection.query(query_embeddings=[embedding], n_results=top_k)
    def _choose_anchor(self, node_ids: list[str]) -> str:
        """Pick a stable anchor for same_as rebalancing: prefer a node with a canonical id; else min UUID."""
        if not node_ids:
            raise ValueError("No nodes to anchor")
        nodes = self.node_collection.get(ids=node_ids, include=["documents"])
        for nid, ndoc in zip(nodes.get("ids") or [], nodes.get("documents") or []):
            n = Node.model_validate_json(ndoc)
            if n.canonical_entity_id:
                return nid
        return min(node_ids)  # stable fallback
    def _rebalance_same_as_edge(self, e: Edge, removed_node_id: str) -> tuple[bool, Edge | None]:
        """
        Remove removed_node_id from a same_as edge. If >=2 nodes remain, normalize to star form.
        Returns (deleted, updated_edge_or_None).
        """
        S = [x for x in (e.source_ids or []) + (e.target_ids or []) if x != removed_node_id]
        # dedupe while preserving order
        S = list(dict.fromkeys(S))
        if len(S) < 2:
            return True, None  # nothing meaningful left
        anchor = self._choose_anchor(S)
        rest = [x for x in S if x != anchor]
        e.source_ids = [anchor]
        e.target_ids = rest
        if not e.summary:
            e.summary = "Normalized same_as"
        return False, e
    
    def extract_graph_with_llm(self, *, content: str, alias_nodes_str = "[Empty]" , alias_edges_str = "[Empty]"):
        """Pure: run LLM + parse + alias resolution. No writes."""
        # (reuse your existing prompt + alias path)
        raw, parsed, error = self._extract_graph_with_llm_aliases(
            content, alias_nodes_str=alias_nodes_str, alias_edges_str=alias_edges_str
        )
        if error:
            raise ValueError(error)
        if not isinstance(parsed, LLMGraphExtraction):
            parsed = LLMGraphExtraction.model_validate(parsed)

        # resolve nn:/ne:/aliases -> UUIDs here
        # and run self._preflight_validate(parsed, doc_id) LATER (we don’t know doc_id yet)
        return {"raw": raw, "parsed": parsed}
    def persist_graph_extraction(
        self,
        *,
        document: Document,
        parsed: LLMGraphExtraction,
        mode: str = "append",   # "replace" | "append" | "skip-if-exists"
    ) -> dict:
        """
        Write nodes/edges/endpoints for `document.id`.
        Returns concrete ids written for idempotent tests.
        """
        doc_id = document.id

        # if replace, rollback prior doc content first
        if mode == "replace":
            self.rollback_document(doc_id)

        # ensure doc row exists (idempotent)
        self.add_document(document)

        

        # now validate (ensures no dangling refs to *other* docs)
        self._preflight_validate(parsed, doc_id)

        # persist and collect ids
        node_ids, edge_ids = [], []
        fallback_snip = (document.content[:160] + "…") if document.content else None

        for ln in parsed.nodes:
            ln.references = self._dealias_refs(ln.references, document.id, fallback_snip)
            # skip-if-exists mode
            if mode == "skip-if-exists":
                got = self.node_collection.get(ids=[ln.id])
                if got.get("ids"):  # already there
                    node_ids.append(ln.id)
                    continue

            n = Node(
                id=ln.id, label=ln.label, type=ln.type, summary=ln.summary,
                domain_id=ln.domain_id, canonical_entity_id=ln.canonical_entity_id,
                properties=ln.properties,
                references=_normalize_refs(ln.references, doc_id, fallback_snip),
                doc_id=doc_id,
            )
            self.add_node(n, doc_id=doc_id)
            node_ids.append(n.id)

        for le in parsed.edges:
            le.references = self._dealias_refs(le.references, document.id, fallback_snip)
            if mode == "skip-if-exists":
                got = self.edge_collection.get(ids=[le.id])
                if got.get("ids"):
                    edge_ids.append(le.id)
                    continue

            e = Edge(
                id=le.id, label=le.label, type=le.type, summary=le.summary,
                domain_id=le.domain_id, canonical_entity_id=le.canonical_entity_id,
                properties=le.properties,
                references=_normalize_refs(le.references, doc_id, fallback_snip),
                relation=le.relation,
                source_ids=le.source_ids, target_ids=le.target_ids,
                source_edge_ids=getattr(le, "source_edge_ids", None),
                target_edge_ids=getattr(le, "target_edge_ids", None),
                doc_id=doc_id,
            )
            self.add_edge(e, doc_id=doc_id)
            edge_ids.append(e.id)

        return {
            "document_id": doc_id,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "nodes_added": len(node_ids),
            "edges_added": len(edge_ids),
        }
    def ingest_document_with_llm(self, document: Document, *, mode: str = "append"):
        """Convenience: extract + persist. Still returns concrete ids written."""
        # add doc row now so fallback refs have URLs
        self.add_document(document)

        # build context & aliases as you already do, then:
        extracted = self.extract_graph_with_llm(content=document.content)
        parsed = extracted["parsed"]

        # de-alias against this doc scope & validate
        self._preflight_validate(parsed, document.id)

        return self.persist_graph_extraction(
            document=document,
            parsed=parsed,
            mode=mode,
        )
    def _extract_graph_with_llm(self, content: str, doc: Document) -> Tuple[Any, Optional[LLMGraphExtraction], Optional[str]]:
        """Call LLM for structured extractio without alias. Returns (raw, parsed, parsing_error)."""

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are an expert knowledge graph extractor. "
            "Given a document, extract entities and relationships as nodes and edges in a hypergraph.\n"
            "For each node/edge include: label, type ('entity' or 'relationship'), and a concise 'summary'.\n"
            "Each node/edge MUST include at least one ReferenceSession with start_page, end_page, start_char, end_char."
            "IMPORTANT:\n"
            "- Do NOT invent real UUIDs or external URLs.\n"
            "Each node/edge MUST include at least one ReferenceSession with spans."),
            ("human",
            "Document:\n{document}\n\n"
            "Document ID {document_id}\n\n"
            "Return only the structured JSON for the schema.")
        ])
        chain = prompt | self.llm.with_structured_output(LLMGraphExtraction, include_raw=True)
        result = chain.invoke({"doc_alias_note": self._alias_doc_in_prompt(), "document": content, "document_id": str(doc.id)})
        # result is a dict when include_raw=True
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        err = result.get("parsing_error") if isinstance(result, dict) else None
        return raw, parsed, err

    def _ingest_text_with_llm(self, *, doc_id: str, content: str, auto_adjudicate: bool):
        # context
        ctx_nodes, ctx_edges = self._select_doc_context(doc_id)
        aliased_nodes, aliased_edges, alias_nodes_str, alias_edges_str = self._aliasify_for_prompt(doc_id, ctx_nodes, ctx_edges)

        # extract
        raw, parsed, error = self._extract_graph_with_llm_aliases(content, alias_nodes_str, alias_edges_str)
        if error:
            raise ValueError(f"LLM parsing error: {error}")
        if not isinstance(parsed, LLMGraphExtraction):
            parsed = LLMGraphExtraction.model_validate(parsed)

        # de-alias
        parsed = self._de_alias_ids_in_result(doc_id, parsed)
        self._preflight_validate(parsed, doc_id)
        
        res = self.ingest_with_toposort( parsed, doc_id=doc_id)
        res['raw'] = raw
        # # persist
        # fallback_snip = (content[:160] + ("…" if len(content) > 160 else "")) if content else None
        # nodes_added = edges_added = 0
        # for ln in parsed.nodes:
        #     nid = ln.id or str(uuid.uuid4())
        #     refs = _normalize_refs(ln.references, doc_id, fallback_snip)
        #     node = Node(
        #         id=nid, label=ln.label, type=ln.type, summary=ln.summary,
        #         domain_id=ln.domain_id, canonical_entity_id=ln.canonical_entity_id,
        #         properties=ln.properties, references=refs, doc_id=doc_id
        #     )
        #     self.add_node(node, doc_id=doc_id)
        #     nodes_added += 1

        # for le in parsed.edges:
        #     edge = Edge(
        #         id=le.id, label=le.label, type=le.type, summary=le.summary,
        #         domain_id=le.domain_id, canonical_entity_id=le.canonical_entity_id,
        #         properties=le.properties,
        #         references=_normalize_refs(le.references, doc_id, fallback_snip),
        #         relation=le.relation,
        #         source_ids=le.source_ids, target_ids=le.target_ids,
        #         source_edge_ids=getattr(le, "source_edge_ids", None),
        #         target_edge_ids=getattr(le, "target_edge_ids", None),
        #         doc_id=doc_id
        #     )
        #     self.add_edge(edge, doc_id=doc_id)
        #     edges_added += 1

        # optional within-doc adjudication
        if auto_adjudicate:
            # naive label+type buckets; you can plug your candidate generator here
            data = self.node_collection.get(where={"doc_id": doc_id}, include=["documents"])
            buckets = {}
            for ndoc in (data.get("documents") or []):
                n = Node.model_validate_json(ndoc)
                buckets.setdefault((n.type, n.label.strip().lower()), []).append(n)
            pairs = []
            for _, bucket in buckets.items():
                if len(bucket) > 1:
                    for i in range(len(bucket)):
                        for j in range(i+1, len(bucket)):
                            pairs.append((bucket[i], bucket[j]))
            if pairs:
                verdicts, _ = self.batch_adjudicate_merges(pairs)
                for (left, right), out in zip(pairs, verdicts):
                    verdict = getattr(out, "verdict", out)
                    if verdict.same_entity:
                        self.commit_merge(left, right, verdict)

        return res


    def prune_node_from_edges(self, node_id: str):
        """
        Remove a node from all edges that reference it.
        - For normal edges: delete if one side becomes empty; else update endpoints.
        - For same_as hyperedges: if >=2 nodes remain, rebalance into star; else delete.
        Returns sets of edge IDs: {'deleted_edges': set, 'updated_edges': set}
        """
        eps = self.edge_endpoints_collection.get(
            where={"$and": [{"endpoint_id": node_id}, {"endpoint_type" : "node"}]}, include=["documents"]
            )
        if not eps["ids"]:
            return {"deleted_edges": set(), "updated_edges": set()}

        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps["documents"]})
        edges = self.edge_collection.get(ids=edge_ids, include=[ "documents", "metadatas"])

        removed_edge_ids: set[str] = set()
        updated_edge_ids: set[str] = set()

        for eid, edoc, meta in zip(edges.get("ids") or [], edges.get("documents") or [], edges.get("metadatas") or []):
            e = Edge.model_validate_json(edoc)
            relation = (meta or {}).get("relation") or e.relation  # be resilient to missing meta

            if relation == "same_as":
                # Special rebalancing for equality hyperedges
                edge_deleted, new_edge = self._rebalance_same_as_edge(e, removed_node_id=node_id)
                if edge_deleted:
                    self.edge_collection.delete(ids=[eid])
                    self.edge_endpoints_collection.delete(where={"edge_id": eid})
                    removed_edge_ids.add(eid)
                else:
                    # Update the edge
                    new_edge: Edge # help pylance
                    self.edge_collection.update(
                        ids=[eid],
                        documents=[new_edge.model_dump_json()],
                        metadatas=[_strip_none({
                            "doc_id": (meta or {}).get("doc_id"),
                            "relation": new_edge.relation,
                            "source_ids": _json_or_none(new_edge.source_ids),
                            "target_ids": _json_or_none(new_edge.target_ids),
                            "type": new_edge.type,
                            "summary": new_edge.summary,
                            "domain_id": new_edge.domain_id,
                            "canonical_entity_id": new_edge.canonical_entity_id,
                            "properties": _json_or_none(new_edge.properties),
                            "references": _json_or_none([ref.model_dump() for ref in (new_edge.references or [])]),
                        })],
                    )
                    # Rebuild endpoints from scratch to match new star form
                    self.edge_endpoints_collection.delete(where={"edge_id": eid})
                    ep_ids, ep_docs, ep_metas = [], [], []
                    for role, node_ids in (("src", new_edge.source_ids or []), ("tgt", new_edge.target_ids or [])):
                        for nid in node_ids:
                            ep_id = f"{eid}::{role}::{nid}"
                            # derive per-endpoint doc_id from node JSON if available
                            node_doc = self.node_collection.get(ids=[nid], include=["documents"])
                            per_doc_id = None
                            if node_doc.get("documents"):
                                try:
                                    n = Node.model_validate_json(node_doc["documents"][0])
                                    per_doc_id = getattr(n, "doc_id", None)
                                except Exception:
                                    per_doc_id = None
                            meta_ep = _strip_none({
                                "id": ep_id,
                                "edge_id": eid,
                                "node_id": nid,
                                "role": role,
                                "relation": new_edge.relation,
                                "doc_id": per_doc_id,
                            })
                            ep_ids.append(ep_id)
                            ep_docs.append(json.dumps(meta_ep))
                            ep_metas.append(meta_ep)
                    if ep_ids:
                        self.edge_endpoints_collection.add(ids=ep_ids, documents=ep_docs, metadatas=ep_metas)
                    updated_edge_ids.add(eid)
                continue

            # ---- Normal (non-same_as) edges ----
            new_src = [x for x in (e.source_ids or []) if x != node_id]
            new_tgt = [x for x in (e.target_ids or []) if x != node_id]

            if not new_src or not new_tgt:
                # delete the whole edge + its endpoint rows
                self.edge_collection.delete(ids=[eid])
                self.edge_endpoints_collection.delete(where={"edge_id": eid})
                removed_edge_ids.add(eid)
            else:
                e.source_ids, e.target_ids = new_src, new_tgt
                self.edge_collection.update(
                    ids=[eid],
                    documents=[e.model_dump_json()],
                    metadatas=[_strip_none({
                        "doc_id": (meta or {}).get("doc_id"),
                        "relation": e.relation,
                        "source_ids": _json_or_none(e.source_ids),
                        "target_ids": _json_or_none(e.target_ids),
                        "type": e.type,
                        "summary": e.summary,
                        "domain_id": e.domain_id,
                        "canonical_entity_id": e.canonical_entity_id,
                        "properties": _json_or_none(e.properties),
                        "references": _json_or_none([ref.model_dump() for ref in (e.references or [])]),
                    })],
                )
                # remove only the touched endpoint rows
                self.edge_endpoints_collection.delete(where={"$and": [{"edge_id": eid}, {"node_id": node_id}]})
                updated_edge_ids.add(eid)

        return {
            "deleted_edges": removed_edge_ids,
            "updated_edges": updated_edge_ids - removed_edge_ids,  # in case something was updated then later deleted
        }

    def rollback_document(self, document_id: str):
        # 1) nodes by flat doc_id

        node_ids = self._nodes_by_doc(document_id)

        # 2) prune each node from edges
        deleted_edges = set()
        updated_edges = set()
        deleted_edges_cnt = updated_edges_cnt = 0
        for nid in node_ids:
            res = self.prune_node_from_edges(nid)
            deleted_edges_cnt += len(res["deleted_edges"])
            updated_edges_cnt += len(res["updated_edges"])
            deleted_edges.update(res["deleted_edges"])
            updated_edges.update(res["updated_edges"])
        
        # 3) delete edges explicitly created by this document via endpoints table
        eps = self.edge_endpoints_collection.get(where={"doc_id": document_id})
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps.get("documents", [])})
        if edge_ids:
            self.edge_collection.delete(ids=edge_ids)
            self.edge_endpoints_collection.delete(where={"doc_id": document_id})
            deleted_edges.update(edge_ids)
        updated_edges = updated_edges - deleted_edges
        # 4) delete nodes and the document
        deleted_node_ids = []
        for nid in node_ids:
            # remove only refs for this doc
            self._prune_node_refs_for_doc(nid, document_id)
            # if node has no refs left, delete it
            got = self.node_collection.get(ids=[nid], include=["documents"])
            if got.get("documents") and got["documents"][0]:
                # n = Node.model_validate_json(got["documents"][0])
                if not json.loads(got["documents"][0]).get('references') : # none or empty list
                    self.node_collection.delete(ids=[nid])
                    deleted_node_ids.append(nid)
        doc_ids = set(self.document_collection.get(where={"doc_id": document_id})['ids'])
        self.document_collection.delete(where={"doc_id": document_id})
        doc_ids_after = set(self.document_collection.get(where={"doc_id": document_id})['ids'])
        return {
            "roll"
            "rolled_back_doc_id": doc_ids - doc_ids_after,
            "updated_edge_ids": list(updated_edges),
            "deleted_edge_ids": list(deleted_edges),
            "deleted_docs": len(doc_ids - doc_ids_after),
            "deleted_node_ids": deleted_node_ids,
            "deleted_nodes": len(node_ids),
            "deleted_edges": len(deleted_edges),
            "updated_edges": len(updated_edges),
        }

    def rollback_many_documents(self, document_ids: list[str]):
        totals = {"deleted_nodes": 0, "deleted_edges": 0, "updated_edges": 0, "deleted_docs": 0}
        for did in document_ids:
            res = self.rollback_document(did)
            totals["deleted_docs"] += 1
            totals["deleted_nodes"] += len(res["deleted_node_ids"])
            totals["deleted_edges"] += len(res["deleted_edge_ids"])
            totals["updated_edges"] += res["updated_edges"]
        return totals
    # ----------------------------
    # Adjudication (LLM-assisted merge decision)
    # ----------------------------
    def _fetch_target(self, t: AdjudicationTarget) -> Node | Edge:
        if t.kind == "node":
            got = self.node_collection.get(ids=[t.id], include=["documents"])
            if not got.get("documents"): raise ValueError(f"Node {t.id} not found")
            return Node.model_validate_json(got["documents"][0])
        got = self.edge_collection.get(ids=[t.id], include=["documents"])
        if not got.get("documents"): raise ValueError(f"Edge {t.id} not found")
        return Edge.model_validate_json(got["documents"][0])

    def commit_merge_target(self, left: AdjudicationTarget, right: AdjudicationTarget, verdict: AdjudicationVerdict) -> str:
        """Generalized merge: node↔node or edge↔edge. Returns canonical id."""
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")
        if left.kind != right.kind:
            raise ValueError("Cannot merge cross-kind (node vs edge).")

        # Decide canonical id
        canonical_id = verdict.canonical_entity_id
        if not canonical_id:
            # prefer any existing canonical; else new
            l = self._fetch_target(left)
            r = self._fetch_target(right)
            canonical_id = getattr(l, "canonical_entity_id", None) or getattr(r, "canonical_entity_id", None) or str(uuid.uuid4())

        if left.kind == "node":
            # --- Node merge: as you already do ---
            l = self._fetch_target(left)
            r = self._fetch_target(right)
            l.canonical_entity_id = r.canonical_entity_id = canonical_id
            # persist
            self.node_collection.update(
                ids=[l.id],
                documents=[l.model_dump_json()],
                metadatas=[_strip_none({
                    "doc_id": getattr(l, "doc_id", None),
                    "label": l.label, "type": l.type, "summary": l.summary,
                    "domain_id": l.domain_id, "canonical_entity_id": l.canonical_entity_id,
                    "properties": _json_or_none(l.properties),
                    "references": _json_or_none([ref.model_dump() for ref in (l.references or [])]),
                })],
            )
            self._index_node_docs(l)
            self.node_collection.update(
                ids=[r.id],
                documents=[r.model_dump_json()],
                metadatas=[_strip_none({
                    "doc_id": getattr(r, "doc_id", None),
                    "label": r.label, "type": r.type, "summary": r.summary,
                    "domain_id": r.domain_id, "canonical_entity_id": r.canonical_entity_id,
                    "properties": _json_or_none(r.properties),
                    "references": _json_or_none([ref.model_dump() for ref in (r.references or [])]),
                })],
            )
            self._index_node_docs(r)
            # record same_as (node↔node)
            left_ref = self._best_ref(l)
            right_ref = self._best_ref(r)
            same_as = Edge(
                id=str(uuid.uuid4()),
                label="same_as", type="relationship", summary=verdict.reason or "merge",
                relation="same_as",
                source_ids=[l.id], target_ids=[r.id],
                source_edge_ids=[], target_edge_ids=[],
                properties={"confidence": verdict.confidence},
                references=[left_ref, right_ref],
                doc_id="__adjudication__",
            )
            self.add_edge(same_as, doc_id=same_as.doc_id)
            return canonical_id

        # --- Edge merge: mirror the same pattern, but meta-edge same_as(edge, edge) ---
        le = self._fetch_target(left)
        re = self._fetch_target(right)
        le.canonical_entity_id = re.canonical_entity_id = canonical_id
        # persist edge updates
        self.edge_collection.update(
            ids=[le.id],
            documents=[le.model_dump_json()],
            metadatas=[_strip_none({
                "doc_id": getattr(le, "doc_id", None),
                "relation": le.relation,
                "source_ids": _json_or_none(le.source_ids),
                "target_ids": _json_or_none(le.target_ids),
                "type": le.type, "summary": le.summary,
                "domain_id": le.domain_id, "canonical_entity_id": le.canonical_entity_id,
                "properties": _json_or_none(le.properties),
                "references": _json_or_none([ref.model_dump() for ref in (le.references or [])]),
            })],
        )
        self.edge_collection.update(
            ids=[re.id],
            documents=[re.model_dump_json()],
            metadatas=[_strip_none({
                "doc_id": getattr(re, "doc_id", None),
                "relation": re.relation,
                "source_ids": _json_or_none(re.source_ids),
                "target_ids": _json_or_none(re.target_ids),
                "type": re.type, "summary": re.summary,
                "domain_id": re.domain_id, "canonical_entity_id": re.canonical_entity_id,
                "properties": _json_or_none(re.properties),
                "references": _json_or_none([ref.model_dump() for ref in (re.references or [])]),
            })],
        )
        # meta same_as: edge↔edge (use edge-endpoint lists)
        same_as_meta = Edge(
            id=str(uuid.uuid4()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "merge",
            relation="same_as",
            source_ids=[], target_ids=[],
            source_edge_ids=[le.id], target_edge_ids=[re.id],
            properties={"confidence": verdict.confidence},
            references=[],  # you can copy best refs from both edges if you desire
            doc_id="__adjudication__",
        )
        self.add_edge(same_as_meta, doc_id=same_as_meta.doc_id)
        return canonical_id

    def _primary_doc_id_from_node(self, n: Node) -> str | None:
        # Prefer explicit doc_id on the node JSON; else infer from references
        if getattr(n, "doc_id", None):
            return n.doc_id
        for r in (n.references or []):
            if r.document_page_url:
                m = re.search(r"document/([A-Za-z0-9\-]+)", r.document_page_url)
                if m:
                    return m.group(1)
        return None

    def _best_ref(self, n: Node | Edge) -> ReferenceSession:
        # support subclass of Node, that is including Edge indeed
        # Choose one ref to copy to the edge (earliest page, earliest char)
        if n.references:
            refs = sorted(
                n.references,
                key=lambda r: (getattr(r, "start_page", 10**9), getattr(r, "start_char", 10**9))
            )
            # Shallow copy + annotate verification that this ref is used for adjudication evidence
            ref = refs[0].model_copy(deep=True)
            if ref.verification is None:
                ref.verification = MentionVerification(method="heuristic", is_verified=True, score=0.5, notes="used as adjudication evidence")
            else:
                # don’t flip existing flags, just add note
                ref.verification.notes = (ref.verification.notes or "") + " | used as adjudication evidence"
            return ref
        # Fallback if somehow node has no refs (shouldn’t happen with your model)
        doc_id = self._primary_doc_id_from_node(n) or "unknown"
        return _default_ref(doc_id, snippet=n.summary if hasattr(n, "summary") else None)
    def add_edge_with_endpoint_docs(self, edge: Edge, endpoint_doc_ids: dict[str, str | None]):
        # Add the main edge row (neutral doc_id)
        self.edge_collection.add(
            ids=[edge.id],
            documents=[edge.model_dump_json()],
            embeddings=[edge.embedding] if edge.embedding else None,
            metadatas=[self.chroma_sanitize_metadata({
                "doc_id": getattr(edge, "doc_id", None),
                "relation": edge.relation,
                "source_ids": json.dumps(edge.source_ids),
                "target_ids": json.dumps(edge.target_ids),
                "type": edge.type,
                "summary": edge.summary,
                "domain_id": edge.domain_id,
                "canonical_entity_id": edge.canonical_entity_id,
                "properties": json.dumps(edge.properties) if edge.properties is not None else None,
                "references": json.dumps([ref.model_dump() for ref in (edge.references or [])]),
            })],
        )

        # Fan-out edge_endpoints; each endpoint gets the *node's* doc_id for rollback
        ep_ids, ep_docs, ep_metas = [], [], []
        for role, node_ids in (("src", edge.source_ids or []), ("tgt", edge.target_ids or [])):
            for nid in node_ids:
                eid = f"{edge.id}::{role}::{nid}"
                doc_id = endpoint_doc_ids.get(nid)
                meta_ep = self.chroma_sanitize_metadata({
                    "id": eid,
                    "edge_id": edge.id,
                    "node_id": nid,
                    "role": role,
                    "relation": edge.relation,
                    "doc_id": doc_id,  # <-- specific to that node's document
                })
                ep_ids.append(eid)
                ep_docs.append(json.dumps(meta_ep))
                ep_metas.append(meta_ep)

        if ep_ids:
            self.edge_endpoints_collection.add(
                ids=ep_ids,
                documents=ep_docs,
                metadatas=ep_metas,
            )
    def _classify_endpoint_id(self, rid: str) -> str:
        """Return 'node' or 'edge' by checking collections; raise if not found."""
        hit = self.node_collection.get(ids=[rid])
        if (hit.get("ids") or [None])[0] == rid:
            return "node"
        hit = self.edge_collection.get(ids=[rid])
        if (hit.get("ids") or [None])[0] == rid:
            return "edge"
        raise ValueError(f"Unknown endpoint id {rid!r} (not a node or edge)")
    def _split_endpoints(self, src_ids: list[str] | None, tgt_ids: list[str] | None):# -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        s_nodes, s_edges, t_nodes, t_edges = [], [], [], []
        for rid in (src_ids or []):
            (s_nodes if self._classify_endpoint_id(rid) == "node" else s_edges).append(rid)
        for rid in (tgt_ids or []):
            (t_nodes if self._classify_endpoint_id(rid) == "node" else t_edges).append(rid)
        return s_nodes, s_edges, t_nodes, t_edges
    def commit_merge(self, left: Node, right: Node, verdict: AdjudicationVerdict) -> str:
        canonical_id = self.merge_policy.commit_merge()
        return canonical_id
    def commit_any_kind(self, node_or_edge_l: AdjudicationTarget, node_or_edge_r: AdjudicationTarget,
                        verdict: AdjudicationVerdict) -> str:
        self.merge_policy.commit_any_kind(self, node_or_edge_l, node_or_edge_r,
                        verdict)
    def generate_merge_candidates(self, new_node: Node, top_k: int = 5, similarity_threshold: float = 0.85) -> List[Tuple[Node, Node]]:
        """
        Given a new node, find likely duplicates in Chroma for adjudication.
        Returns a list of (existing_node, new_node) pairs.
        """
        return self.proposer.for_new_node(self, new_node, top_k, similarity_threshold)
        
    
    def adjudicate_pair(self, left: AdjudicationTarget, right: AdjudicationTarget, question: str):
        if (left.kind != right.kind) and question != "node_edge_equivalence":
            raise ValueError("Cross-kind only allowed for 'node_edge_equivalence'")
        if not self.allow_cross_kind_adjudication and question == "node_edge_equivalence":
            raise ValueError("Cross-kind adjudication disabled")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a careful adjudicator. Decide if LEFT and RIGHT correspond as the question specifies.\n"
            "- If question == 'same_entity': same real-world entity (node↔node).\n"
            "- If question == 'same_relation': same logical relation instance (edge↔edge) including endpoints.\n"
            "- If question == 'node_edge_equivalence': determine if the NODE is a named reification/denotation "
            "  of the EDGE (i.e., the node represents that specific relation instance). "
            "Be conservative; return JSON verdict."),
            ("human", "Question: {question}\nLeft:\n{left}\nRight:\n{right}")
        ])
        chain = prompt | self.llm.with_structured_output(LLMMergeAdjudication)
        return chain.invoke({"question": question,
                            "left": left.model_dump(),
                            "right": right.model_dump()})
    def adjudicate_merge(self, left_node: Node | Edge, right_node: Node | Edge):
        # Back-compat wrapper if you still call with concrete models
        left = self._target_from_node(left_node) if isinstance(left_node, Node) else self._target_from_edge(left_node)
        right = self._target_from_node(right_node) if isinstance(right_node, Node) else self._target_from_edge(right_node)
        question = "same_entity" if left.kind == "node" else "same_relation"
        return self.adjudicate_pair(left, right, question=question)
    def batch_adjudicate_merges(
        self,
        pairs: List[Tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    ):
        if not pairs:
            return []

        qcode = AdjudicationQuestionCode(question_code)
        qkey = QUESTION_KEY[qcode]

        # --- helpers -------------------------------------------------------------
        def node_id(n):
            return getattr(n, "id", None) or n.model_dump().get("id")

        def node_kind(n):
            # prefer an explicit 'kind' (e.g., 'entity'|'relation') then fallback to type/classname
            return (
                getattr(n, "kind", None)
                or n.model_dump().get("kind")
                or getattr(n, "type", None)
                or n.model_dump().get("type")
                or n.__class__.__name__
            )

        def normalized_signature(left, right):
            """Normalized cache key for cross-type pairs, including qkey so different questions don't collide."""
            lid, rid = node_id(left), node_id(right)
            lkind, rkind = node_kind(left), node_kind(right)
            a, b = (lkind, str(lid)), (rkind, str(rid))
            key_pair = (a, b) if a <= b else (b, a)
            return key_pair + (qkey,)  # ((kind,id),(kind,id), qkey)

        # compact copy of node for LLM (keep it tiny but informative)
        def compact_payload(n):
            d = n.model_dump()
            out = {}
            # always include these:
            out["kind"] = node_kind(n)
            out["type"] = d.get("type")  # coarse class if present
            out["name"] = d.get("name") or d.get("label") or d.get("title")
            # optional small attrs whitelist if present
            attrs = {}
            for k in ("dob", "country", "ticker", "date", "role", "source"):
                if k in d and d[k] is not None:
                    attrs[k] = d[k]
            if attrs:
                out["attrs"] = attrs
            # relation/hyperedge signature if present
            if "signature" in d and d["signature"]:
                # expect [{"role": "...", "id": "...", ...}, ...]
                out["signature"] = d["signature"]
            return out

        # --- 1) pre-pass: cache lookup & collect unknowns -----------------------
        cache = {}  # key: ((kind,id),(kind,id), qkey) -> LLMMergeAdjudication
        known_by_index = {}
        unknown_indices = []
        unknown_pairs = []

        for idx, (left, right) in enumerate(pairs):
            k = normalized_signature(left, right)
            if k in cache:
                known_by_index[idx] = cache[k]
            else:
                unknown_indices.append(idx)
                unknown_pairs.append((left, right, k))

        if not unknown_pairs:
            ordered = [known_by_index[i] for i in range(len(pairs))]
            return ordered, qkey

        # --- 2) short-id aliasing over unique (kind,id) objects -----------------
        # collect unique objects by (kind,id)
        uniq_objs = []
        seen = set()
        for left, right, _ in unknown_pairs:
            for n in (left, right):
                tup = (node_kind(n), str(node_id(n)))
                if tup not in seen:
                    seen.add(tup)
                    uniq_objs.append(tup)

        alias_map = {obj: f"n{i}" for i, obj in enumerate(uniq_objs)}  # (kind,id) -> "nX"
        inv_alias = {v: obj for obj, v in alias_map.items()}           # "nX" -> (kind,id)

        # --- 3) build tiny LLM inputs (aliased ids + compact fields) ------------
        mapping_table = [
            {"code": int(code), "key": QUESTION_KEY[code], "description": QUESTION_DESC[code]}
            for code in AdjudicationQuestionCode
        ]

        adjudication_inputs = []
        for left, right, _ in unknown_pairs:
            l_key = (node_kind(left), str(node_id(left)))
            r_key = (node_kind(right), str(node_id(right)))
            adjudication_inputs.append({
                "left":  {"id": alias_map[l_key],  **compact_payload(left)},
                "right": {"id": alias_map[r_key], **compact_payload(right)},
                "cross_type": node_kind(left) != node_kind(right),
                "question_code": int(qcode),
            })

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You adjudicate candidate pairs, including cross-type pairs (entity↔relation, etc.). "
            "Use the mapping table to interpret question_code. "
            "Return only the structured JSON per schema. Use the short ids exactly."),
            ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}")
        ])
        from .models import BatchAdjudications
        chain = prompt | self.llm.with_structured_output(BatchAdjudications, include_raw= True)
        llm_results= chain.invoke({"mapping": mapping_table, "pairs": adjudication_inputs})
        raw = llm_results.get("raw") if isinstance(llm_results, dict) else None
        parsed: BatchAdjudications  = llm_results.get("parsed") if isinstance(llm_results, dict) else llm_results
        err = llm_results.get("parsing_error") if isinstance(llm_results, dict) else None
        # --- 4) de-alias result ids and cache by normalized key -----------------
        fixed_results = []
        for (left, right, key_sig), res in zip(unknown_pairs, parsed.merge_adjudications):
            # map "nX" back to (kind,id)
            left_alias = getattr(res, "left_id", None) or res.model_dump().get("left_id")
            right_alias = getattr(res, "right_id", None) or res.model_dump().get("right_id")
            l_kind, l_id = inv_alias.get(left_alias, (node_kind(left), str(node_id(left))))
            r_kind, r_id = inv_alias.get(right_alias, (node_kind(right), str(node_id(right))))

            # rebuild or mutate to set original ids
            if hasattr(res, "model_copy"):
                res = res.model_copy(update={"left_id": l_id, "right_id": r_id, "left_kind": l_kind, "right_kind": r_kind})
            else:
                setattr(res, "left_id", l_id); setattr(res, "right_id", r_id)
                setattr(res, "left_kind", l_kind); setattr(res, "right_kind", r_kind)

            cache[key_sig] = res
            fixed_results.append(res)

        # --- 5) stitch results back to original order ---------------------------
        ordered = [None] * len(pairs)
        for i, r in known_by_index.items():
            ordered[i] = r
        it = iter(fixed_results)
        for i in unknown_indices:
            ordered[i] = next(it)

        return ordered, qkey

    def add_page(self, *, document_id: str, page_text: str, page_number: int, auto_adjudicate: bool = True):
        """
        Ingest a single page of an existing document.
        - Reuses doc-scoped context with aliases (cheap tokens)
        - Stores nodes/edges tagged with doc_id
        - Auto-adjudicates within the document by default
        """
        if not page_text or not isinstance(page_text, str):
            return {"document_id": document_id, "nodes_added": 0, "edges_added": 0}

        # Optionally, you could attach a page-level Document record (virtual).
        # For now we just ingest content with doc_id and normalize refs as usual.
        res = self._ingest_text_with_llm(
            doc_id=document_id,
            content=page_text,
            auto_adjudicate=auto_adjudicate,
        )
        # If you want to force mention spans to that page when missing,
        # you could post-process the brand-new nodes/edges and ensure refs' start_page/end_page = page_number.
        return {"document_id": document_id, "page": page_number, **res}
    
    def verify_mentions_for_doc(
        self,
        document_id: str,
        *,
        source_text: Optional[str] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        update_edges: bool = True,
    ) -> Dict[str, int]:
        """
        Verify all references in nodes (and edges if update_edges=True) for a doc.
        Returns counts of updated items.
        """
        full_text = source_text if source_text is not None else self._fetch_document_text(document_id)
        upd_nodes = upd_edges = 0

        # Nodes
        got = self.node_collection.get(where={"doc_id": document_id}, include=["documents"])
        for nid, ndoc in zip(got.get("ids") or [], got.get("documents") or []):
            n = Node.model_validate_json(ndoc)
            # what text do we try to validate? prioritize summary, then label
            extracted = n.summary or n.label or ""
            if not (n.references and extracted):
                continue
            new_refs = [self.verify_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                        for r in n.references]
            n.references = new_refs
            doc, meta = _node_doc_and_meta(n)
            self.node_collection.update(ids=[nid], documents=[doc], metadatas=[meta])
            self._index_node_docs(n)
            upd_nodes += 1

        if update_edges:
            got = self.edge_collection.get(where={"doc_id": document_id}, include=["documents"])
            for eid, edoc in zip(got.get("ids") or [], got.get("documents") or []):
                e = Edge.model_validate_json(edoc)
                extracted = e.summary or e.label or e.relation or ""
                if not (e.references and extracted):
                    continue
                new_refs = [self.verify_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                            for r in e.references]
                new_refs = [self.verify_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                            for r in e.references]
                e.references = new_refs
                doc, meta = _edge_doc_and_meta(e)
                self.edge_collection.update(ids=[eid], documents=[doc], metadatas=[meta])
                upd_edges += 1

        return {"updated_nodes": upd_nodes, "updated_edges": upd_edges}

    def verify_mentions_for_items(
        self,
        items: List[Tuple[str, str]],  # list of ("node"|"edge", id)
        *,
        source_text_by_doc: Optional[Dict[str, str]] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
    ) -> Dict[str, int]:
        """
        Targeted verification for a mixed set of nodes/edges.
        source_text_by_doc lets you pass pre-fetched doc text keyed by doc_id.
        """
        upd_nodes = upd_edges = 0
        for kind, rid in items:
            if kind == "node":
                got = self.node_collection.get(ids=[rid], include=["documents", "metadatas"])
                if not got.get("documents"):
                    continue
                n = Node.model_validate_json(got["documents"][0])
                doc_id = (got["metadatas"][0] or {}).get("doc_id")
                full_text = (source_text_by_doc or {}).get(doc_id) or self._fetch_document_text(doc_id) if doc_id else ""
                extracted = n.summary or n.label or ""
                if not (n.references and extracted):
                    continue
                n.references = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                                for r in n.references]
                doc, meta = _node_doc_and_meta(n)
                self.node_collection.update(ids=[rid], documents=[doc], metadatas=[meta])
                self._index_node_docs(n)
                upd_nodes += 1
            elif kind == "edge":
                got = self.edge_collection.get(ids=[rid], include=["documents", "metadatas"])
                if not got.get("documents"):
                    continue
                e = Edge.model_validate_json(got["documents"][0])
                doc_id = (got["metadatas"][0] or {}).get("doc_id")
                full_text = (source_text_by_doc or {}).get(doc_id) or self._fetch_document_text(doc_id) if doc_id else ""
                extracted = e.summary or e.label or e.relation or ""
                if not (e.references and extracted):
                    continue
                e.references = [self._verify_one_reference(extracted, full_text, r, min_ngram=min_ngram, weights=weights, threshold=threshold)
                                for r in e.references]
                doc, meta = _edge_doc_and_meta(e)
                self.edge_collection.update(ids=[rid], documents=[doc], metadatas=[meta])
                upd_edges += 1
        return {"updated_nodes": upd_nodes, "updated_edges": upd_edges}
    # ----------------------------
    # Visualization 
    # ----------------------------
    def _load_node_map(self, ids: Iterable[str]) -> dict[str, dict]:
        """Return {id: {'label':..., 'type':..., 'summary':..., 'doc_ids': [...]}}, missing ids omitted."""
        ids = list(dict.fromkeys([i for i in ids if i]))  # dedupe/preserve order, drop falsy
        out: dict[str, dict] = {}
        if not ids:
            return out
        got = self.node_collection.get(ids=ids, include=["documents", "metadatas"])
        for nid, ndoc, meta in zip(got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []):
            if not nid: 
                continue
            try:
                n = Node.model_validate_json(ndoc)
                out[nid] = {
                    "label": n.label,
                    "type": n.type,
                    "summary": getattr(n, "summary", "") or "",
                    "doc_ids": json.loads((meta or {}).get("doc_ids") or "[]"),
                }
            except Exception:
                # fallback if pydantic fails
                out[nid] = {
                    "label": (meta or {}).get("label") or "(node)",
                    "type": (meta or {}).get("type") or "entity",
                    "summary": (meta or {}).get("summary") or "",
                    "doc_ids": json.loads((meta or {}).get("doc_ids") or "[]"),
                }
        return out

    def _load_edge_map(self, ids: Iterable[str]) -> dict[str, dict]:
        """Return {id: {'relation':..., 'source_ids':[...], 'target_ids':[...], 'label':..., 'summary':...}}."""
        ids = list(dict.fromkeys([i for i in ids if i]))
        out: dict[str, dict] = {}
        if not ids:
            return out
        got = self.edge_collection.get(ids=ids, include=["documents", "metadatas"])
        for eid, edoc, meta in zip(got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []):
            if not eid:
                continue
            try:
                e = Edge.model_validate_json(edoc)
                out[eid] = {
                    "label": e.label,
                    "relation": e.relation,
                    "summary": getattr(e, "summary", "") or "",
                    "source_ids": e.source_ids or [],
                    "target_ids": e.target_ids or [],
                    "source_edge_ids": getattr(e, "source_edge_ids", []) or [],
                    "target_edge_ids": getattr(e, "target_edge_ids", []) or [],
                }
            except Exception:
                # metadata fallback (source/target ids may be stored as JSON strings)
                def parse_ids(k):
                    v = (meta or {}).get(k)
                    try:
                        return json.loads(v) if isinstance(v, str) else (v or [])
                    except Exception:
                        return v or []
                out[eid] = {
                    "label": (meta or {}).get("label") or "(edge)",
                    "relation": (meta or {}).get("relation") or "",
                    "summary": (meta or {}).get("summary") or "",
                    "source_ids": parse_ids("source_ids"),
                    "target_ids": parse_ids("target_ids"),
                    "source_edge_ids": parse_ids("source_edge_ids"),
                    "target_edge_ids": parse_ids("target_edge_ids"),
                }
        return out

    def resolve_readable(
        self,
        *,
        node_ids: Optional[Iterable[str]] = None,
        edge_ids: Optional[Iterable[str]] = None,
        by_doc_id: Optional[str] = None,
        include_refs: bool = False,
    ) -> dict:
        """
        Structured, human-readable snapshot.
        - If by_doc_id is given, it overrides explicit node_ids/edge_ids (it pulls all linked).
        - include_refs: adds compact reference strings (can be heavy).
        Returns:
        {
            "nodes": [{"id":..., "label":..., "type":..., "summary":..., "doc_ids":[...] , "refs":[...] }],
            "edges": [{"id":..., "relation":..., "summary":..., "sources":[{"id":..., "kind":"node|edge", "label":...}], "targets":[...], "refs":[...]}]
        }
        """
        # 1) Fetch ids from a doc if requested
        if by_doc_id:
            # Pull all node_ids linked to the given doc_id from the fast index
            n_links = self.node_docs_collection.get(where={"doc_id": by_doc_id}, include=["documents"])
            node_ids = [json.loads(doc)["node_id"] for doc in (n_links.get("documents") or [])]

            # Pull edge_ids by scanning edge_endpoints for that doc_id
            e_links = self.edge_endpoints_collection.get(where={"doc_id": by_doc_id}, include=["documents"])
            edge_ids = list({
                json.loads(doc)["edge_id"] for doc in (e_links.get("documents") or [])
            })
        node_ids = list(dict.fromkeys(node_ids or []))
        edge_ids = list(dict.fromkeys(edge_ids or []))

        # 2) Build maps
        node_map = self._load_node_map(node_ids)
        edge_map = self._load_edge_map(edge_ids)

        # 3) If edges point to additional nodes/edges not explicitly requested, load them too
        extra_nodes, extra_edges = set(), set()
        for em in edge_map.values():
            extra_nodes.update(em.get("source_ids", []))
            extra_nodes.update(em.get("target_ids", []))
            extra_edges.update(em.get("source_edge_ids", []))
            extra_edges.update(em.get("target_edge_ids", []))
        # load missing
        missing_nodes = [i for i in extra_nodes if i not in node_map]
        missing_edges = [i for i in extra_edges if i not in edge_map]
        node_map.update(self._load_node_map(missing_nodes))
        edge_map.update(self._load_edge_map(missing_edges))

        # 4) Optionally load refs for nodes/edges
        node_out = []
        if node_map:
            got = self.node_collection.get(ids=list(node_map.keys()), include=["metadatas", "documents"])
            for nid, meta, ndoc in zip(got.get("ids") or [], got.get("metadatas") or [], got.get("documents") or []):
                m = node_map.get(nid, {})
                entry = {"id": nid, **m}
                if include_refs:
                    try:
                        n = Node.model_validate_json(ndoc)
                        entry["refs"] = [_fmt_ref_short(r.model_dump()) for r in (n.references or [])]
                    except Exception:
                        # try metadata path
                        refs = []
                        raw = (meta or {}).get("references")
                        if isinstance(raw, str):
                            try:
                                for r in json.loads(raw) or []:
                                    refs.append(_fmt_ref_short(r))
                            except Exception:
                                pass
                        entry["refs"] = refs
                node_out.append(entry)

        edge_out = []
        if edge_map:
            got = self.edge_collection.get(ids=list(edge_map.keys()), include=["metadatas", "documents"])
            for eid, meta, edoc in zip(got.get("ids") or [], got.get("metadatas") or [], got.get("documents") or []):
                m = edge_map.get(eid, {})
                # resolve endpoint labels
                def resolve_list(ids, kind_hint: str):
                    items = []
                    for rid in ids:
                        if rid in node_map:
                            items.append({"id": rid, "kind": "node", "label": node_map[rid]["label"]})
                        elif rid in edge_map:
                            items.append({"id": rid, "kind": "edge", "label": edge_map[rid]["label"]})
                        else:
                            items.append({"id": rid, "kind": kind_hint, "label": "(missing)"})
                    return items

                entry = {
                    "id": eid,
                    "relation": m.get("relation", ""),
                    "label": m.get("label", ""),
                    "summary": m.get("summary", ""),
                    "sources": resolve_list(m.get("source_ids", []), "node"),
                    "targets": resolve_list(m.get("target_ids", []), "node"),
                }
                # include edge-endpoint edges if you use them
                se = m.get("source_edge_ids") or []
                te = m.get("target_edge_ids") or []
                if se or te:
                    entry["source_edges"] = resolve_list(se, "edge")
                    entry["target_edges"] = resolve_list(te, "edge")

                if include_refs:
                    try:
                        e = Edge.model_validate_json(edoc)
                        entry["refs"] = [_fmt_ref_short(r.model_dump()) for r in (e.references or [])]
                    except Exception:
                        refs = []
                        raw = (meta or {}).get("references")
                        if isinstance(raw, str):
                            try:
                                for r in json.loads(raw) or []:
                                    refs.append(_fmt_ref_short(r))
                            except Exception:
                                pass
                        entry["refs"] = refs

                edge_out.append(entry)

        return {"nodes": node_out, "edges": edge_out}

    def pretty_print_graph(self, **kwargs) -> str:
        """
        Thin wrapper over resolve_readable() that renders a compact text block.
        kwargs are passed to resolve_readable (node_ids, edge_ids, by_doc_id, include_refs).
        """
        data = self.resolve_readable( **kwargs)
        lines = []
        if data["nodes"]:
            lines.append("Nodes:")
            for n in data["nodes"]:
                line = f"  • {n['id']}  [{n['type']}]  {n['label']}"
                if n.get("summary"): line += f" — {n['summary']}"
                if n.get("doc_ids"): line += f"  (docs: {', '.join(n['doc_ids'])})"
                lines.append(line)
                if kwargs.get("include_refs") and n.get("refs"):
                    for r in n["refs"]:
                        lines.append(f"     ↳ {r}")
        if data["edges"]:
            lines.append("Edges:")
            for e in data["edges"]:
                def fmt_endpoints(items):
                    return ", ".join([f"{i['label']}({i['id'][:8]})" for i in items])
                src = fmt_endpoints(e.get("sources", []))
                tgt = fmt_endpoints(e.get("targets", []))
                line = f"  → {e['id']}  [{e.get('relation','')}]  {src}  ->  {tgt}"
                if e.get("summary"): line += f" — {e['summary']}"
                lines.append(line)
                if e.get("source_edges") or e.get("target_edges"):
                    ss = fmt_endpoints(e.get("source_edges", []))
                    tt = fmt_endpoints(e.get("target_edges", []))
                    if ss: lines.append(f"     (source-edges: {ss})")
                    if tt: lines.append(f"     (target-edges: {tt})")
                if kwargs.get("include_refs") and e.get("refs"):
                    for r in e["refs"]:
                        lines.append(f"     ↳ {r}")
        return "\n".join(lines) or "(empty)"