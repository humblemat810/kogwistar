from __future__ import annotations
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
from dataclasses import dataclass, field
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
from typing import (Callable, Optional, Tuple, Any, Dict, Iterable, Sequence, Literal,
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
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_openai import AzureOpenAIEmbeddings
    _HAS_AZURE_EMB = True
except Exception:
    _HAS_AZURE_EMB = False

NodeOrEdge: TypeAlias =  Node | Edge
T = TypeVar("T", Node, Edge)
from graphlib import TopologicalSorter
def _refs_hash(refs) -> str:
    import hashlib, json
    return hashlib.sha1(json.dumps(
        [r.model_dump() for r in (refs or [])],
        sort_keys=True
    ).encode()).hexdigest()
import hashlib, json

def _refs_fingerprint(refs) -> str:
    # Normalize minimal fields that affect the index rows, order-sensitive
    payload = [
        {
            "doc_id": getattr(r, "doc_id", None),
            "method": getattr(getattr(r, "verification", None), "method", None),
            "is_verified": getattr(getattr(r, "verification", None), "is_verified", None),
            "score": getattr(getattr(r, "verification", None), "score", None),
            "sp": getattr(r, "start_page", None),
            "ep": getattr(r, "end_page", None),
            "sc": getattr(r, "start_char", None),
            "ec": getattr(r, "end_char", None),
            "snip": (getattr(r, "snippet", None) or "")[:64],  # cap; avoid huge digests
        }
        for r in (refs or [])
    ]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    # 128-bit BLAKE2b: fast + collision resistant enough for cache guards
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _safe_snippet(s: str | None, max_len: int = 200) -> str | None:
    if not s:
        return None
    s = s.strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s

def _ref_doc_id(ref) -> str | None:
    # Try explicit doc_id on the ReferenceSession if your model has it
    did = getattr(ref, "doc_id", None)
    if did:
        return did
    # Else try to extract from document_page_url like "document/<uuid>"
    url = getattr(ref, "document_page_url", None) or ""
    m = re.search(r"document/([A-Za-z0-9\-]+)", url)
    return m.group(1) if m else None

def _ref_insertion_method(ref) -> str:
    # If your ReferenceSession has 'insertion_method', use it
    m = getattr(ref, "insertion_method", None)
    if m:
        return str(m)
    # Fallbacks (optional): derive from verification.method or unknown
    ver = getattr(ref, "verification", None)
    if ver and getattr(ver, "method", None):
        return str(ver.method)
    return "unknown"
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

import json, re
from typing import Any, Iterable, List, Dict, Union, Optional

PageLike = Union[str, Dict[str, Any]]

def _split_pages_from_text(raw: str) -> List[Dict[str, Any]]:
    """
    Heuristics:
      - Prefer form-feed splits (\f) if present.
      - Else split on common page headers like 'Page 1', 'Page: 2', etc.
      - Else treat whole text as page 1.
    """
    if not raw:
        return []

    # 1) Form-feed first (common in OCR / PDFs)
    if "\f" in raw:
        parts = raw.split("\f")
        return [{"page_number": i+1, "text": p.strip()} for i, p in enumerate(parts) if p.strip()]

    # 2) Simple page header detection
    p = re.split(r"(?:^|\n)\s*Page[:\s]+(\d+)\s*(?:\n|$)", raw, flags=re.IGNORECASE)
    if len(p) > 1:
        out, i = [], 0
        # p looks like [prefix, num1, chunk1, num2, chunk2, ...]
        while i < len(p):
            if i == 0 and p[i].strip():
                out.append({"page_number": 1, "text": p[i].strip()})
                i += 1
                continue
            if i + 2 <= len(p)-1:
                try:
                    num = int(p[i+1])
                except Exception:
                    num = None
                txt = p[i+2].strip()
                if txt:
                    out.append({"page_number": num or (len(out)+1), "text": txt})
                i += 3
            else:
                break
        if out:
            return out

    # 3) Fallback: one page
    return [{"page_number": 1, "text": raw.strip()}]

def _coerce_pages(content_or_pages: Any, *, default_page_start: int = 1) -> List[Dict[str, Any]]:
    """
    Normalize many shapes to a list of {'page_number': int, 'text': str}.

    Accepts:
      - raw string (split by \f or headers → pages)
      - list[str] (each element is page text)
      - list[dict] with keys {'page'|'page_number', 'text'|'content'}
      - dict with a 'pages' field (any of the above inside)
      - JSON string of any of the above

    NEVER raises on shape; returns [] if nothing usable.
    """
    def as_page_dict(x: PageLike, idx0: int) -> Optional[Dict[str, Any]]:
        if isinstance(x, str):
            t = x.strip()
            if not t:
                return None
            return {"page_number": default_page_start + idx0, "text": t}
        if isinstance(x, dict):
            # map flexible keys
            num = x.get("page_number") or x.get("pdf_page_number")
            if num is None:
                num = x.get("page")
            try:
                num = int(num) if num is not None else (default_page_start + idx0)
            except Exception:
                num = default_page_start + idx0
            txt = x.get("text")
            if txt is None:
                txt = x.get("content")
            if isinstance(txt, str) and txt.strip():
                return {"page_number": num, "text": txt.strip()}
            return None
        return None

    # JSON string wrapper?
    if isinstance(content_or_pages, str):
        s = content_or_pages.strip()
        # if it looks like json pages, try parse; else split text
        if s and s[:1] in "[{" and s[-1:] in "]}" :
            try:
                parsed = json.loads(s)
                return _coerce_pages(parsed, default_page_start=default_page_start)
            except Exception:
                pass
        # treat as raw document text
        return _split_pages_from_text(s)

    # dict with 'pages'
    if isinstance(content_or_pages, dict):
        if "pages" in content_or_pages:
            pages = content_or_pages.get("pages") or []
            out: List[Dict[str, Any]] = []
            for i, item in enumerate(pages):
                row = as_page_dict(item, i)
                if row:
                    out.append(row)
            return out
        # or a single page-like dict
        row = as_page_dict(content_or_pages, 0)
        return [row] if row else []

    # list input
    if isinstance(content_or_pages, list):
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(content_or_pages):
            row = as_page_dict(item, i)
            if row:
                out.append(row)
        return out

    # unknown shape → empty (caller decides fallback)
    return []

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
    """Return (documents_string, metadata_dict) for Chroma. helper when inserting to backend db,"""
    """Extract and flatten certain fields that can be searched via collection """
    from pydantic import BaseModel, field_serializer
    doc = n.model_dump_json(field_mode = 'backend', exclude = ['embedding'])
    meta = _strip_none({
        "doc_id": getattr(n, "doc_id", None),
        "label": n.label,
        "type": n.type,
        "summary": n.summary,
        "domain_id": n.domain_id,
        "canonical_entity_id": getattr(n, "canonical_entity_id", None),
        "properties": _json_or_none(getattr(n, "properties", None)),
        "references": _json_or_none([r.model_dump(field_mode = 'backend') for r in (n.references or [])]),
        # add any other flat, filterable fields you rely on
    })
    return doc, meta

def _edge_doc_and_meta(e: "Edge") -> tuple[str, dict]:
    """Return (documents_string, metadata_dict) for Chroma."""
    doc = e.model_dump_json(field_mode = 'backend')
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
    if (not hasattr(r, 'verification') and r.__class__.__name__.endswith("LlmSlice")):  # llm slice no such field
        pass
    elif (hasattr(r, 'verification') and r.verification is None): # ok
        r.verification = _default_verification("no explicit verification from LLM")
    else: # ok defined
        pass
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

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import ollama
# ollama.embeddings(model='all-minilm:l6-v2', prompt='The sky is blue because of Rayleigh scattering')
import functools


class CustomEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name():
        return "default"
    def __init__(self, model_name: str = 'all-minilm:l6-v2'):
        import numpy as np
        _emb = functools.partial(ollama.embeddings, model=model_name)
        def ef(prompts: list[str]):
            
            res = []
            for p in prompts:
                r = _emb(prompt = p).embedding
                norm_val = np.linalg.norm(r)
                res.append(r / norm_val)
            
            return res
        self._emb = ef
        # Initialize your embedding model here
        # For example, if using Sentence Transformers:
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Implement your embedding logic here
        # This method should take a list of strings (documents)
        # and return a list of lists of floats (embeddings)

        # Example with a placeholder for your model's embedding logic:
        embeddings = self._emb(input)
        return embeddings
@dataclass
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
        
    def _infer_doc_id_from_ref(self, ref: ReferenceSession) -> Optional[str]:
        """Best-effort: prefer explicit ref.doc_id; else try to parse document_page_url like 'document/<id>'."""
        did = getattr(ref, "doc_id", None)
        if did:
            return did
        url = getattr(ref, "document_page_url", None) or ""
        # simple heuristic: last path token if present
        try:
            tail = url.strip("/").split("/")[-1]
            return tail or None
        except Exception:
            return None

    def extract_reference_contexts(
        self,
        node_or_id: Union[Node, str],  # also works if you pass an Edge or edge id
        *,
        window_chars: int = 120,
        max_contexts: Optional[int] = None,
        prefer_label_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build adjudication-ready reference snippets for a Node (or Edge).
        Strategy:
        - Prefer the stored ReferenceSession.snippet (cheap, already localized).
        - If document text is available, try to locate the snippet (or node label) in full text
            and expand with ±window_chars for richer context.
        - Always return provenance + verification info (doc_id, page spans, urls, etc.).
        """
        # 1) Materialize the object
        from .models import GraphEntityBase
        if isinstance(node_or_id, GraphEntityBase):
            obj = node_or_id
        else:
            # try node first, then edge
            got = self.node_collection.get(ids=[node_or_id], include=["documents"])
            doc_list = got.get("documents") or []
            if doc_list:
                obj = Node.model_validate_json(doc_list[0])
            else:
                got = self.edge_collection.get(ids=[node_or_id], include=["documents"])
                edoc_list = got.get("documents") or []
                if not edoc_list:
                    raise ValueError(f"Unknown node/edge id: {node_or_id}")
                obj = Edge.model_validate_json(edoc_list[0])

        label = getattr(obj, "label", None)
        refs = getattr(obj, "references", None) or []
        out: List[Dict[str, Any]] = []

        for ref in refs:
            # 2) locate the backing document
            doc_id = self._infer_doc_id_from_ref(ref)
            full = self._fetch_document_text(doc_id) if doc_id else None

            snippet = getattr(ref, "snippet", None)
            mention = snippet or (label or "")

            ctx_text = None
            span_start = None
            span_end = None

            if full:
                # Try exact snippet first (best anchor)
                idx = full.find(snippet) if snippet else -1
                # Fallback to label if allowed
                if idx < 0 and label and prefer_label_fallback:
                    idx = full.find(label)

                if idx >= 0:
                    # If we matched on snippet, use its length; else label length
                    length = len(snippet) if snippet else (len(label) if label else 0)
                    span_start = idx
                    span_end = idx + length
                    left = max(0, span_start - window_chars)
                    right = min(len(full), span_end + window_chars)
                    ctx_text = full[left:right]
                else:
                    # Full text present but we couldn't find anchor—still return snippet if present
                    ctx_text = snippet or None
            else:
                # No full text—just echo stored snippet if any
                ctx_text = snippet or None

            out.append({
                "doc_id": doc_id,
                "collection_page_url": getattr(ref, "collection_page_url", None),
                "document_page_url": getattr(ref, "document_page_url", None),
                "start_page": getattr(ref, "start_page", None),
                "end_page": getattr(ref, "end_page", None),
                "start_char": getattr(ref, "start_char", None),
                "end_char": getattr(ref, "end_char", None),
                "insertion_method": getattr(ref, "insertion_method", None),
                "verification": (ref.verification.model_dump() if getattr(ref, "verification", None) else None),
                "context": ctx_text,             # expanded context or stored snippet
                "mention": mention,              # what to highlight / quote in a prompt
                "loc_found": (span_start is not None),
                "loc_span": [span_start, span_end] if span_start is not None else None,
                # Always include the raw ref in case you need exact fields later
                "ref": ref.model_dump(),
            })

            if max_contexts and len(out) >= max_contexts:
                break

        return out
    
    def get_nodes(self, ids: Sequence[str]) -> List[Node]:
        if not ids: return []
        got = self.node_collection.get(ids=list(ids), include=["documents", "embeddings"])
        docs = got.get("documents") or []
        embs = got.get("embeddings") if got.get("embeddings") is not None else []
        return [Node.model_validate_json(d).model_copy(update={{"embedding": emb}}) for d, emb in zip(docs, embs)]

    def get_edges(self, ids: Sequence[str]) -> List[Edge]:
        if not ids: return []
        got = self.edge_collection.get(ids=list(ids), include=["documents", "embeddings"])
        docs = got.get("documents") or []
        embs = got.get("embeddings") if got.get("embeddings") is not None else []
        return [Edge.model_validate_json(d).model_copy(update={{"embedding": emb}}) for d, emb in zip(docs, embs)]

    def all_nodes_for_doc(self, doc_id: str) -> List[Node]:
        return self.get_nodes(self._nodes_by_doc(doc_id))

    def all_edges_for_doc(self, doc_id: str) -> List[Edge]:
        return self.get_edges(self._edge_ids_by_doc(doc_id))
    
    def _delete_edge_ref_rows(self, edge_id: str) -> None:
        # Deleting by where sometimes misses rows on some backends; prefer id list
        got = self.edge_refs_collection.get(where={"edge_id": edge_id}, include=[])
        ids = got.get("ids") or []
        if ids:
            self.edge_refs_collection.delete(ids=ids)

    def _delete_node_ref_rows(self, node_id: str) -> None:
        got = self.node_refs_collection.get(where={"node_id": node_id}, include=[])
        ids = got.get("ids") or []
        if ids:
            self.node_refs_collection.delete(ids=ids)
    
    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _default_ref(doc_id: str, snippet: Optional[str] = None) -> ReferenceSession:
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
                            documents=[n.model_dump_json(field_mode='backend')],
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
                            documents=[e.model_dump_json(field_mode='backend')],
                            metadatas=[{
                                **{k:v for k,v in prior_meta.items() if v is not None},
                                "references": merged_json
                            }]
                        )
                        self._maybe_reindex_edge_refs(e)
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
        return {
            "document_id": doc_id,
            "node_ids": nodes_added,
            "edge_ids": edges_added,
            "nodes_added": len(nodes_added),
            "edges_added": len(edges_added),
        }
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
            "You are an information extraction system that converts legal contracts into a knowledge graph.  "
            "Your task: extract ALL entities (nodes) and relationships (edges) from the text. "
            "Nodes should include at least: Parties, Obligations, Rights, Deliverables, Payment Terms, Termination Conditions, Confidentiality Clauses, Governing Law, Dates, and Penalties.  "
            "Edges should capture: (Party → Obligation), (Obligation → Condition), (Party → Right), (Obligation → Deliverable), (Clause → Governing Law).  "
            "Allow multiple edge between the same nodes. Allow hypergraph. Allow edge pointing to other edge. "
            "Allow same label but different content. "
            "Breakdown A is obligated to do work for B as A -> B : relation = do work for 100 dollar. You can create another edge. "
            "Build relationship triplets of SVO. "
            "Also pay attention to monetary terms, numbers. Be aware if they are definite, indefitite. Once off or recurrent. Keep a sharp eye one numbers. "
            "For any signatories with blank to sign, or signed. They are equally important to note."
            "Extract all nodes and edges from the following contract section.  "
            "Be exhaustive and granular:"
            "- Every obligation, right, condition, exception, penalty, deadline, and reference must become a separate node.  "
            "- Each clause and sub-clause should yield at least one node.  "
            "- Do not merge or summarize multiple obligations.  "
            "- Aim for at least 20 nodes per section, if possible."
            "Extract entities and terms as nodes, relationships edges in a hypergraph.\n\n"
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
        try:
            chain = prompt | self.llm.with_structured_output(LLMGraphExtraction['llm'], 
                                                             include_raw=True)
            steps = chain.steps
            realised_prmopt = steps[0].invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content, "_DOC_ALIAS" : _DOC_ALIAS})
            llm_raw = steps[1].invoke(realised_prmopt)
            result = steps[2].invoke(llm_raw)
            # result = chain.invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content, "_DOC_ALIAS" : _DOC_ALIAS})
        except Exception as e:
            raise
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
    
    @classmethod
    def _coerce_pages(cls, content_or_pages):# -> list[tuple[int, str]] | list[Any] | Any:
        """
        Normalize many shapes into a list[(page_no:int, text:str)].

        Accepts:
        - str (raw text)                      ->  [(1, text)] or split on \f if present
        - str (JSON)                          ->  tries json.loads:
                * ["page text", ...]
                * [{"page": 3, "text": "..."}, ...]
                * {"1": "text", "2": "text"} (mapping of page->text)
        - list[str]                           ->  [(1, item1), (2, item2), ...]
        - list[tuple[int, str]] or list[list] ->  as-is (page, text)
        - dict[int|str, str]                  ->  sorted by page key
        """
        import json, re

        # (A) Python containers first
        if isinstance(content_or_pages, dict):
            items = sorted(((int(k), v) for k, v in content_or_pages.items()), key=lambda x: x[0])
            return [(p, str(t or "")) for p, t in items]

        if isinstance(content_or_pages, (list, tuple)):
            if not content_or_pages:
                return []
            first = content_or_pages[0]
            # list of (page, text)
            if isinstance(first, (list, tuple)) and len(first) == 2:
                return [(int(p), str(t or "")) for p, t in content_or_pages]
            # list of dicts with page/text
            if isinstance(first, dict) and "text" in first:
                out = []
                for i, item in enumerate(content_or_pages, start=1):
                    p = int(item.get("page", i))
                    out.append((p, str(item.get("text", "") or "")))
                return out
            # list of plain page texts
            return [(i, str(t or "")) for i, t in enumerate(content_or_pages, start=1)]

        # (B) Strings: try JSON first
        if isinstance(content_or_pages, str):
            s = content_or_pages.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    loaded = json.loads(s)
                    return cls._coerce_pages(loaded)
                except Exception:
                    pass
            # raw text with optional form-feed page breaks
            if "\f" in s:
                parts = s.split("\f")
                return [(i, p) for i, p in enumerate(parts, start=1)]
            return [(1, s)]

        # fallback: one page
        return [(1, str(content_or_pages))]
    # ----------------------------
    # Init
    # ----------------------------
    
    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_function=None,
        proposer=None,               # callable(pairs) -> List[LLMMergeAdjudication]
        adjudicator=None,            # callable(left: Node, right: Node) -> AdjudicationVerdict
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
        from .strategies import CompositeProposer, VectorProposer, DefaultVerifier, PreferExistingCanonical, Adjudicator
        # from .strategies.adjudicators import LLMPairAdjudicatorImpl, LLMBatchAdjudicatorImpl
        from .strategies.verifiers import DefaultVerifier, VerifierConfig
        from .strategies.types import Verifier
        from graph_knowledge_engine.strategies import IAdjudicator
        self.proposer = proposer or VectorProposer(self)
        self.adjudicator : IAdjudicator = adjudicator or Adjudicator(self)
        # self.pair_adjudicator: PairAdjudicator = adjudicator or LLMPairAdjudicatorImpl(self)
        # self.batch_adjudicator: BatchAdjudicator = batch_adjudicator or LLMBatchAdjudicatorImpl(self)
        self.verifier: Verifier = verifier or DefaultVerifier(self, VerifierConfig(use_embeddings=False))
        self.merge_policy = merge_policy or PreferExistingCanonical(self)
        load_dotenv()
        # from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        # ef = ONNXMiniLM_L6_V2(preferred_providers=["CUDAExecutionProvider", ]) # 'TensorrtExecutionProvider'
        # try:
        #     ef('test')
        # except:
        #     ef = ONNXMiniLM_L6_V2(preferred_providers=['CPUExecutionProvider'])
        #     try:
        #         ef("test")
        #     except:
        #         pro = ef.ort.get_available_providers()[0]
        #         ef = ONNXMiniLM_L6_V2(preferred_providers=[pro])
        #         ef("test")
        self._alias_books: dict[str, AliasBook] = {}
        ef = CustomEmbeddingFunction()
            
        self._ef = ef#embedding_function or ef #embedding_functions.DefaultEmbeddingFunction()
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
            "nodes", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self.node_collection.get(limit = 1)
        self.node_collection.query(query_texts = ['hello world'])
        self.edge_collection = self.chroma_client.get_or_create_collection(
            "edges", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self.edge_endpoints_collection = self.chroma_client.get_or_create_collection("edge_endpoints", embedding_function=self._ef, 
                                metadata={"hnsw:space": "cosine"})
        self.document_collection = self.chroma_client.get_or_create_collection("documents", embedding_function=self._ef, 
                                metadata={"hnsw:space": "cosine"})
        self.domain_collection = self.chroma_client.get_or_create_collection("domains", embedding_function=self._ef, 
                                metadata={"hnsw:space": "cosine"})
        self.node_docs_collection = self.chroma_client.get_or_create_collection("node_docs", embedding_function=self._ef, 
                                metadata={"hnsw:space": "cosine"})
        self.node_refs_collection = self.chroma_client.get_or_create_collection("node_refs")
        self.edge_refs_collection = self.chroma_client.get_or_create_collection("edge_refs")
        # self.llm = AzureChatOpenAI(
        #     deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
        #     model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
        #     azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
        #     cache=None,
        #     openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
        #     api_version="2024-08-01-preview",
        #     model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
        #     temperature=0.1,
        #     max_tokens=30000,
        #     openai_api_type="azure",
        # )
        self.llm = ChatGoogleGenerativeAI(
                        #model = "gemini-2.5-pro-preview-05-06",
                        # model = "gemini-2.5-pro",
                        model = "gemini-2.5-pro",
                        # model = model_name,
                        #model="gemini-1.5-pro",
                        #model="gemini-2.0-pro",
                        #model="gemini-2.5-pro-preview-03-25",
                        temperature=0.1,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        
                        # other params...
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
        from joblib import Memory
        
        self.memory = Memory(location = os.path.join('.', '.kg_cache'))
        self._cached_extract_graph_with_llm = self.memory.cache(self.extract_graph_with_llm, ignore = ["self"])
    # ----------------------------
    # Mention / citation verification
    # ----------------------------
    @staticmethod
    def _node_doc_and_meta(n: "Node") -> tuple[str, dict]:
        return _node_doc_and_meta(n)
    @staticmethod
    def _edge_doc_and_meta(e: "Edge") -> tuple[str, dict]:
        return _edge_doc_and_meta(e)
    def _maybe_reindex_edge_refs(self, edge: Edge, *, force: bool = False) -> None:
        new_fp = _refs_fingerprint(edge.references or [])
        meta = self.edge_collection.get(ids=[edge.id], include=["metadatas"])
        old_fp = None
        if meta.get("metadatas") and meta["metadatas"][0]:
            old_fp = meta["metadatas"][0].get("edge_refs_fp")

        # Secondary guards: row-count & doc_id set must match
        got = self.edge_refs_collection.get(where={"edge_id": edge.id}, include=["documents"])
        current_rows = got.get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (edge.references or [])}
        count_ok = (len(current_rows) == len(edge.references or []))
        docset_ok = (current_doc_ids == expect_doc_ids)

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self.edge_collection.update(ids=[edge.id], metadatas=[{"edge_refs_fp": new_fp}])
            self._index_edge_refs(edge)

    def _maybe_reindex_node_refs(self, node: Node, *, force: bool = False) -> None:
        new_fp = _refs_fingerprint(node.references or [])
        meta = self.node_collection.get(ids=[node.id], include=["metadatas"])
        old_fp = None
        if meta.get("metadatas") and meta["metadatas"][0]:
            old_fp = meta["metadatas"][0].get("node_refs_fp")

        got = self.node_refs_collection.get(where={"node_id": node.id}, include=["documents"])
        current_rows = got.get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (node.references or [])}
        count_ok = (len(current_rows) == len(node.references or []))
        docset_ok = (current_doc_ids == expect_doc_ids)

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self.node_collection.update(ids=[node.id], metadatas=[{"node_refs_fp": new_fp}])
            self._index_node_refs(node)
    def _index_edge_refs(self, edge: Edge) -> None:
        self._delete_edge_ref_rows(edge.id)

        ids, docs, metas = [], [], []
        for i, ref in enumerate(edge.references or []):
            rid = f"{edge.id}::ref::{i}"
            did = getattr(ref, "doc_id", None)
            ver = getattr(ref, "verification", None)
            row = _strip_none({
                "id": rid,
                "edge_id": edge.id,
                "doc_id": did,
                "insertion_method" : getattr(ref, "insertion_method", None),
                "verification_method": getattr(ver, "method", None),
                "is_verified": getattr(ver, "is_verified", None),
                "verificication_score": getattr(ver, "score", None),
                "start_page": getattr(ref, "start_page", None),
                "end_page": getattr(ref, "end_page", None),
                "start_char": getattr(ref, "start_char", None),
                "end_char": getattr(ref, "end_char", None),
            })
            ids.append(rid); docs.append(json.dumps(row)); metas.append(row)

        if ids:
            self.edge_refs_collection.add(ids=ids, documents=docs, metadatas=metas)
    def _index_node_refs(self, node: Node) -> None:
        """
            create index for records, may flatten json into plain
        """
        self._delete_node_ref_rows(node.id)

        ids, docs, metas = [], [], []
        for i, ref in enumerate(node.references or []):
            rid = f"{node.id}::ref::{i}"
            did = getattr(ref, "doc_id", None)
            ver = getattr(ref, "verification", None)
            row = _strip_none({
                "id": rid,
                "node_id": node.id,
                "doc_id": did,
                "insertion_method" : getattr(ref, "insertion_method", None),
                "verification_method": getattr(ver, "method", None),
                "is_verified": getattr(ver, "is_verified", None),
                "verificication_score": getattr(ver, "score", None),
                "start_page": getattr(ref, "start_page", None),
                "end_page": getattr(ref, "end_page", None),
                "start_char": getattr(ref, "start_char", None),
                "end_char": getattr(ref, "end_char", None),
            })
            ids.append(rid); docs.append(json.dumps(row)); metas.append(row)

        if ids:
            self.node_refs_collection.add(ids=ids, documents=docs, metadatas=metas)
    
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
    
    def check_document_exist(self, document_id : str | list[str]):
        doc_ids = [document_id] if type(document_id) is str else document_id
        got = self.document_collection.get(ids=doc_ids, include=[])
        return set(got['ids']).union(set(document_id))
        
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


    # ----------------------------
    # Chroma adders
    # ----------------------------
    def add_node(self, node: Node, doc_id: Optional[str] = None):
        node.doc_id = doc_id  # may use engine.extract_reference_contexts
        doc, meta = _node_doc_and_meta(node)
        self.node_collection.add(
            ids=[node.id],
            documents=[doc],
            embeddings=[node.embedding] if node.embedding is not None else None,
            metadatas=[meta],
        )
        self._index_node_docs(node)
        self._maybe_reindex_node_refs(node)
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
        # if not (
        #     (edge.source_ids or edge.source_edge_ids)
        #     and (edge.target_ids or edge.target_edge_ids)
        # ):
        #     raise ValueError(
        #         f"Edge {edge.id} ({edge.label}) must have at least one source and one target"
        #     )
        
        # may use engine.extract_reference_contexts
        
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
            documents=[edge.model_dump_json(field_mode='backend', exclude = ['embedding'])],
            embeddings=[edge.embedding] if edge.embedding is not None else None,
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
        self._maybe_reindex_edge_refs(edge)
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

    def _nodes_by_doc(self, doc_id: str, insertion_method: Optional[str] = None) -> list[str]:
        if insertion_method:
            return self.ids_with_insertion_method(kind="node", insertion_method=insertion_method, doc_id=doc_id)
        # original behavior (fast if you have node_docs table; else scan)
        if hasattr(self, "node_docs_collection"):
            rows = self.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
            return sorted({m.get("node_id") for m in (rows.get("metadatas") or []) if m and m.get("node_id")})
        # slow fallback:
        got = self.node_collection.get(where={"doc_id": doc_id}, include=["ids"])
        return got.get("ids") or []

    def _edge_ids_by_doc(self, doc_id: str, insertion_method: Optional[str] = None) -> list[str]:
        if insertion_method:
            return self.ids_with_insertion_method(kind="edge", insertion_method=insertion_method, doc_id=doc_id)
        # original behavior via endpoints table:
        eps = self.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return sorted({m.get("edge_id") for m in (eps.get("metadatas") or []) if m and m.get("edge_id")})

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
            self.node_collection.update(ids=[node_id], documents=[node.model_dump_json(field_mode='backend')])
            # Drop (node,doc) link
            self.node_docs_collection.delete(where={"$and": [{"node_id": node_id}, {"doc_id": doc_id}]})
            # Refresh denormalized doc_ids meta
            self._index_node_docs(node)
        return changed
    def rebuild_edge_refs_for_doc(self, doc_id: str) -> int:
        # Prefer endpoints index to discover edges touching the doc quickly
        eps = self.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["documents"])
        edge_ids = list({json.loads(d)["edge_id"] for d in (eps.get("documents") or [])})
        if not edge_ids:
            return 0
        got = self.edge_collection.get(ids=edge_ids, include=["documents"])
        cnt = 0
        for js in (got.get("documents") or []):
            e = Edge.model_validate_json(js)
            self._index_edge_refs(e)
            cnt += 1
        return cnt

    def rebuild_all_edge_refs(self) -> int:
        got = self.edge_collection.get(include=["ids"])
        total = 0
        for eid in (got.get("ids") or []):
            edoc = self.edge_collection.get(ids=[eid], include=["documents"])
            if edoc.get("documents"):
                e = Edge.model_validate_json(edoc["documents"][0])
                self._index_edge_refs(e)
                total += 1
        return total
    def edges_by_doc(self, doc_id: str, where: Optional[dict] = None) -> list[str]:
        where = {"doc_id": doc_id} if not where else {
            "$and": [{"doc_id": doc_id}] + [{k:v} for k,v in where.items()]
        }
        rows = self.edge_refs_collection.get(where=where, include=["documents"])
        return list({json.loads(d)["edge_id"] for d in (rows.get("documents") or [])})

    def list_edges_with_ref_filter(self, doc_id: str, where: dict | None = None) -> list[Edge]:
        ids = self.edges_by_doc(doc_id, where)
        if not ids:
            return []
        got = self.edge_collection.get(ids=ids, include=["documents"])
        return [Edge.model_validate_json(js) for js in (got.get("documents") or [])]
    def nodes_by_ids(self, node_ids):
        return self.node_collection.get(node_ids)
    def edges_by_ids(self, edge_ids):
        return self.edge_collection.get(edge_ids)
    def nodes_by_doc(self, doc_id: str, *, where : Optional[dict] = None) -> list[str]:
        where = {"doc_id": doc_id} if not where else {
            "$and": [{"doc_id": doc_id}] + [{k:v} for k,v in where.items()]
        }
        rows = self.node_refs_collection.get(where=where, include=["documents"])
        return list({json.loads(d)["node_id"] for d in (rows.get("documents") or [])})

    def list_nodes_with_ref_filter(self, doc_id: str, *, where : Optional[dict] = None) -> list[Node]:
        ids = self.nodes_by_doc(doc_id, where = where)
        if not ids:
            return []
        got = self.node_collection.get(ids=ids, include=["documents"])
        return [Node.model_validate_json(js) for js in (got.get("documents") or [])]
    
    def rebuild_node_refs_for_doc(self, doc_id: str) -> int:
        # Use node_docs index if you have it; else fall back to where on nodes
        node_ids = []
        if hasattr(self, "node_docs_collection"):
            rows = self.node_docs_collection.get(where={"doc_id": doc_id}, include=["documents"])
            node_ids = list({json.loads(d)["node_id"] for d in (rows.get("documents") or [])})
        else:
            got = self.node_collection.get(where={"doc_id": doc_id}, include=["ids", "documents"])
            node_ids = list(got.get("ids") or [])

        if not node_ids:
            return 0

        got = self.node_collection.get(ids=node_ids, include=["documents"])
        cnt = 0
        for js in (got.get("documents") or []):
            n = Node.model_validate_json(js)
            self._index_node_refs(n)
            cnt += 1
        return cnt

    def rebuild_all_node_refs(self) -> int:
        got = self.node_collection.get(include=["ids"])
        total = 0
        for nid in (got.get("ids") or []):
            doc = self.node_collection.get(ids=[nid], include=["documents"])
            if doc.get("documents"):
                n = Node.model_validate_json(doc["documents"][0])
                self._index_node_refs(n)
                total += 1
        return total
    # ----------------------------
    # helpers for rollback
    # ----------------------------

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
    
    def extract_graph_with_llm(self, *, content: str, alias_nodes_str = "[Empty]" , alias_edges_str = "[Empty]", with_parsed = True):
        """Pure: run LLM + parse + alias resolution. No writes."""
        # (reuse your existing prompt + alias path)
        raw, parsed, error = self._extract_graph_with_llm_aliases(
            content, alias_nodes_str=alias_nodes_str, alias_edges_str=alias_edges_str
        )
        if error:
            raise ValueError(error)
        # if not isinstance(parsed, LLMGraphExtraction):
        #     dumped = parsed.model_dump(field_mode = 'backend')
            
            
        #     parsed = LLMGraphExtraction.model_validate(parsed, context={'insertion_method', 'llm_graph_extraction'})

        # resolve nn:/ne:/aliases -> UUIDs here
        # and run self._preflight_validate(parsed, doc_id) LATER (we don’t know doc_id yet)
        if with_parsed:
            return {"raw": raw, "parsed": parsed}
        else:
            return {"raw": raw}
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
        # self.ingest_with_toposort(parsed, doc_id = doc_id)

        

        # persist and collect ids
        node_ids, edge_ids = [], []
        fallback_snip = (document.content[:160] + "…") if document.content else None
        _ = _alloc_real_ids(parsed)  # rewrite in place
        order, id2kind, id2obj = self._build_deps(parsed)

        nodes_added = edges_added = 0
        nl = '\n'
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            # for ln in parsed.nodes:
            if kind == 'node':
                ln: Node = obj
                ln.references = self._dealias_refs(ln.references, document.id, fallback_snip)
                # skip-if-exists mode
                if mode == "skip-if-exists":
                    got = self.node_collection.get(ids=[ln.id])
                    if got.get("ids"):  # already there
                        node_ids.append(ln.id)
                        continue
                refs = [ReferenceSession.model_validate(i.model_dump(), context={'insertion_method': i.insertion_method or 'llm_graph_extraction'}) for i in ln.references]
                
                n = Node(
                    id=ln.id, label=ln.label, type=ln.type, summary=ln.summary,
                    domain_id=ln.domain_id, canonical_entity_id=ln.canonical_entity_id,
                    properties=ln.properties,
                    references= _normalize_refs(refs, doc_id, fallback_snip),
                    doc_id=doc_id,
                    # embedding=self._ef([f"{ln.label}: {ln.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(ln.id))}"])[0]
                )
                n.embedding = self._ef([f"{n.label}: {n.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(ln)[:1])}"])[0]
                self.add_node(n, doc_id=doc_id)
                node_ids.append(n.id)
            elif kind == 'edge':
            # for le in parsed.edges:
                le: Edge = obj
                le.references = self._dealias_refs(le.references, document.id, fallback_snip)
                if mode == "skip-if-exists":
                    got = self.edge_collection.get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                refs = [ReferenceSession.model_validate(i.model_dump(), context={'insertion_method': i.insertion_method or 'llm_graph_extraction'}) for i in le.references]
                e = Edge(
                    id=le.id, label=le.label, type=le.type, summary=le.summary,
                    domain_id=le.domain_id, canonical_entity_id=le.canonical_entity_id,
                    properties=le.properties,
                    references=_normalize_refs(refs, doc_id, fallback_snip),
                    relation=le.relation,
                    source_ids=le.source_ids, target_ids=le.target_ids,
                    source_edge_ids=getattr(le, "source_edge_ids", None),
                    target_edge_ids=getattr(le, "target_edge_ids", None),
                    doc_id=doc_id,
                    # embedding=self._ef([f"{le.label}: {le.summary}"])[0]
                )
                e.embedding = self._ef([f"{le.label}: {le.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(le)[:1])}"])[0]
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
                        documents=[new_edge.model_dump_json(field_mode='backend')],
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
                    self._index_edge_refs(new_edge)
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
                    documents=[e.model_dump_json(field_mode='backend')],
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
                self._index_edge_refs(e)

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
            self.edge_refs_collection.delete(where={"node_id": {"$in": edge_ids}})
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
                self.node_refs_collection.delete(where={"node_id": {"$in": node_ids}})
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
        canonical_id = self.merge_policy.commit_merge_target(left,right,verdict)
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
            documents=[edge.model_dump_json(field_mode='backend')],
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
        canonical_id = self.merge_policy.commit_merge(left, right, verdict)
        return canonical_id
    def commit_any_kind(self, node_or_edge_l: AdjudicationTarget, node_or_edge_r: AdjudicationTarget,
                        verdict: AdjudicationVerdict) -> str:
        self.merge_policy.commit_any_kind(node_or_edge_l, node_or_edge_r,
                        verdict)
    def generate_merge_candidates_doc_brute_force(
        self,
        kind: str = "node",
        scope_doc_id: Optional[str] = None,
        top_k: int = 200,
        *,
        # NEW optional knobs:
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
    ):
        """
        Back-compat:
        - If no new knobs are provided and scope_doc_id is set, behave exactly like before (same-doc only).
        - Otherwise, use the unified proposer with richer scoping.
        """
        if (allowed_docs is None and anchor_doc_id is None and not cross_doc_only and scope_doc_id):
            # legacy behavior (same doc only)
            return self.proposer.same_kind_in_doc(engine=self, doc_id=scope_doc_id, kind="node" if kind == "node" else "edge")

        pair_kind = "node_node" if kind == "node" else "edge_edge"
        return self.proposer.propose_any_kind_any_doc(
            engine=self,
            pair_kind=pair_kind,
            allowed_docs=allowed_docs,
            anchor_doc_id=anchor_doc_id or None,
            cross_doc_only=cross_doc_only,
            anchor_only=anchor_only,
            limit_per_bucket=top_k,
        )


    def generate_cross_kind_candidates(
        self,
        scope_doc_id: Optional[str] = None,
        limit_per_bucket: int = 200,
        *,
        # NEW optional knobs:
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
    ):
        """
        Back-compat:
        - Without new knobs and with scope_doc_id set, behave as before (same-doc only).
        - Otherwise use unified proposer in node↔edge mode.
        """
        if (allowed_docs is None and anchor_doc_id is None and not cross_doc_only and scope_doc_id):
            return self.proposer.cross_kind_in_doc(engine=self, doc_id=scope_doc_id, limit_per_bucket=limit_per_bucket)

        return self.proposer.propose_any_kind_any_doc(
            engine=self,
            pair_kind="node_edge",
            allowed_docs=allowed_docs,
            anchor_doc_id=anchor_doc_id,
            cross_doc_only=cross_doc_only,
            anchor_only=anchor_only,
            limit_per_bucket=limit_per_bucket,
        )


    def generate_merge_candidates(
        self,
        new_node: Union[Node, str, Sequence[Union[Node, str]]],
        top_k: int = 10,
        *,
        # NEW optional knobs (post-filtering on vector hits):
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
    ):
        out = self.proposer.generate_merge_candidates(engine=self,
                                                     new_node = new_node, top_k = top_k, 
                                                     allowed_docs=allowed_docs, 
                                                     anchor_doc_id=anchor_doc_id,
                                                     cross_doc_only=cross_doc_only,
                                                     anchor_only=anchor_only)
        return out
    
    def adjudicate_pair(self, left: AdjudicationTarget, right: AdjudicationTarget, question: str):
        return self.adjudicator.adjudicate_pair(left, right, question)
    def adjudicate_merge(self, left_node: Node | Edge, right_node: Node | Edge):
        return self.adjudicator.adjudicate_merge(left_node, right_node)
    def batch_adjudicate_merges(
        self,
        pairs: List[Tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    ):
        if not pairs:
            return []
        return self.adjudicator.batch_adjudicate_merges(pairs, question_code)
        

    def add_page(self, *, document_id: str, page_text: str | List[str] | Dict[str, Any], page_number: int | None = None, auto_adjudicate: bool = True):
        """
        Ingest a single page of an existing document.
        - Reuses doc-scoped context with aliases (cheap tokens)
        - Stores nodes/edges tagged with doc_id
        - Auto-adjudicates within the document by default
        """
    
        pages = _coerce_pages(
            {"pages": [{"page_number": page_number, "text": page_text}]}
            if isinstance(page_text, str)
            else page_text  # already a list/dict form
        )
        if not pages:
            return {"document_id": document_id, "nodes_added": 0, "edges_added": 0}

        total_nodes = total_edges = 0
        raw_by_page = []
        for pg in pages:
            res = self._ingest_text_with_llm(
                doc_id=document_id,
                content=pg["text"],
                auto_adjudicate=auto_adjudicate,
            )
            total_nodes += res["nodes_added"]
            total_edges += res["edges_added"]
            raw_by_page.append({"page": pg["page_number"], "raw": res.get("raw")})

        return {
            "document_id": document_id,
            "nodes_added": total_nodes,
            "edges_added": total_edges,
            "raw_by_page": raw_by_page,
        }
    
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
        to_return = self.verifier.verify_mentions_for_doc(document_id, source_text = source_text,
                                                          min_ngram = min_ngram,
                                                          threshold = threshold,
                                                          weights=weights,
                                                          update_edges = update_edges
                                                          )
        return to_return
    def ids_with_insertion_method(
        self,
        *,
        kind: str,                           # "node" | "edge"
        insertion_method: str,
        ids: Optional[Iterable[str]] = None, # optionally restrict to this set
        doc_id: Optional[str] = None,        # optionally restrict to a document
    ) -> list[str]:
        """
        Return distinct node_ids (or edge_ids) that have at least one ReferenceSession
        with insertion_method == <value> (and optionally within doc_id).
        Fast path uses the ref index; fallback scans the primary collection.
        """
        assert kind in ("node", "edge"), f"kind must be 'node' or 'edge', got {kind!r}"

        # Choose index & key
        if kind == "node":
            idx = getattr(self, "node_refs_collection", None)
            key = "node_id"
            primary = self.node_collection
            model_cls = Node
        else:
            idx = getattr(self, "edge_refs_collection", None)
            key = "edge_id"
            primary = self.edge_collection
            model_cls = Edge

        # ---------- Fast path: indexed rows ----------
        if idx is not None:
            where = {"insertion_method": insertion_method}
            if doc_id:
                where["doc_id"] = doc_id
            if ids:
                where[key] = {"$in": list(ids)}
            rows = idx.get(where=where, include=["metadatas"])
            picked = {m.get(key) for m in (rows.get("metadatas") or []) if m and m.get(key)}
            return sorted(picked)

        # ---------- Fallback: scan primary JSON ----------
        # (only used if you didn’t create the index collection)
        if ids:
            got = primary.get(ids=list(ids), include=["documents"])
            documents = got.get("documents") or []
            entity_ids = got.get("ids") or []
        else:
            got = primary.get(include=["ids", "documents"])
            documents = got.get("documents") or []
            entity_ids = got.get("ids") or []

        keep: set[str] = set()
        for eid, blob in zip(entity_ids, documents):
            ent = model_cls.model_validate_json(blob)
            for ref in (ent.references or []):
                im = getattr(ref, "insertion_method", None)
                if im == insertion_method and (not doc_id or _ref_doc_id(ref) == doc_id):
                    keep.add(eid)
                    break
        return sorted(keep)
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
        return self.verifier._verify_one_reference(
                                extracted_text=extracted_text,
                                full_text=full_text,
                                ref=ref,
                                
                                min_ngram=min_ngram,
                                weights = weights,
                                threshold = threshold,
                            ) 
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
        
        to_return = self.verifier.verify_mentions_for_items(items,source_text_by_doc=source_text_by_doc,
                                                            min_ngram=min_ngram, threshold = threshold, weights = weights)
        return to_return
    
