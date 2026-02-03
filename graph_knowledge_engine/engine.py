from __future__ import annotations
from types import MethodType

from langchain_core.language_models import BaseChatModel
import pathlib
import time
from . import models
from .models import AddTurnResult, MetaFromLastSummary, RetrievalResult, WorkflowCheckpointNode, WorkflowNode, WorkflowStepExecNode
from .engine_sqlite import EngineSQLite

if True:
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
from .conversation_context import ContextItem, ContextSources, ConversationContextView, DroppedItem, ContextRenderer, Role
import numpy as np
import ast
from typing import Iterator, List, Optional, Dict, Any, Self, Tuple, TypeAlias, cast
from .typing_interfaces import CollectionLike, EmbeddingFunctionLike
from dataclasses import dataclass, field
from .graph_query import GraphQuery
from chromadb import Client
from chromadb.config import Settings
from graph_knowledge_engine.extraction import BaseDocValidator
from .models import (
    Node,
    Edge,
    ConversationNode,
    ConversationEdge,
    Document,
    Domain,
    Grounding,
    PureChromaNode,
    PureChromaEdge,
    PureGraph,
    Span,
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
    ConversationNodeMetadata,
    ConversationAIResponse,
    FilteringResponse, 
    FilteringResult
)
from .cdc.change_event import EntityRef, Op, EntityRefModel
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import uuid
from joblib import Memory
import re
from functools import wraps
from .models import Span, MentionVerification, LLMGraphExtraction, LLMNode, LLMEdge, Node, Edge, Document, GraphExtractionWithIDs
from typing import (Callable, Optional, Tuple, Any, Dict, Iterable, Sequence, Literal,
                    List, Type, TypeVar, Union)
import math
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from chromadb.utils import embedding_functions
from graphlib import TopologicalSorter
import hashlib
from datetime import datetime, timezone
from graph_knowledge_engine.cdc.change_bus import ChangeBus, FastAPIChangeSink
from graph_knowledge_engine.cdc.change_event import ChangeEvent
from graph_knowledge_engine.cdc.oplog import OplogWriter

from pydantic import BaseModel

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
PageLike = Union[str, Dict[str, Any]]
EngineType = Literal ['knowledge', 'conversation', 'workflow']
NodeOrEdge: TypeAlias =  Node | Edge
T = TypeVar("T", Node, Edge)
# TT= TypeVar("TT", Type[Node], Type[Edge])
TNode = TypeVar("TNode", bound=Node)
TEdge = TypeVar("TEdge", bound=Edge)
AnyNode=Union[Node, ConversationNode, WorkflowNode, WorkflowStepExecNode, WorkflowStepExecNode]
TAnyNode = TypeVar("TAnyNode", bound=AnyNode)

def _refs_hash(refs) -> str:
    
    return hashlib.sha1(json.dumps(
        [r.model_dump() for r in (refs or [])],
        sort_keys=True
    ).encode()).hexdigest()
from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")    
def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_json_dict(doc: Any) -> dict:
    if isinstance(doc, dict):
        return doc
    if not isinstance(doc, str):
        return {}
    try:
        x = json.loads(doc)
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}

def _merge_meta(base_meta: dict | None, patch: dict) -> dict:
    base_meta = base_meta or {}
    # flat merge
    return {**base_meta, **patch}

def _is_tombstoned(meta: dict | None) -> bool:
    meta = meta or {}
    return str(meta.get("lifecycle_status") or "active") == "tombstoned"
def _str_or_none(to_str):
    if to_str is None:
        return to_str
    else:
        return str(to_str)
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
            "snip": (getattr(r, "excerpt", None) or "")[:64],  # cap; avoid huge digests
        }
        for r in (refs or [])
    ]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    # 128-bit BLAKE2b: fast + collision resistant enough for cache guards
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


F = TypeVar("F", bound=Callable[..., Any])

def conversation_only(fn: F) -> F:
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "kg_graph_type", None) != "conversation":
            raise RuntimeError(
                f"{fn.__qualname__} requires a conversation engine "
                f"(got {getattr(self, 'kg_graph_type', None)!r})"
            )
        return fn(self, *args, **kwargs)
    return wrapper  # type: ignore
def _safe_excerpt(s: str | None, max_len: int = 200) -> str | None:
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
def _node_doc_and_meta(n: Union["Node", "PureChromaNode"]) -> tuple[str, dict]:
    """Return (documents_string, metadata_dict) for Chroma. helper when inserting to backend db,"""
    """Extract and flatten certain fields that can be searched via collection """
    doc = n.model_dump_json(field_mode = 'backend', exclude = ['embedding', 'metadata'])
    meta = n.metadata # user custom metadata will be overwritten by system metadata
    meta.update({
        "doc_id": getattr(n, "doc_id", None),
        "label": n.label,
        "type": n.type,
        "summary": n.summary,
        "domain_id": n.domain_id,
        "canonical_entity_id": getattr(n, "canonical_entity_id", None),
        "properties": _json_or_none(getattr(n, "properties", None)),
        
    })
    meta.update(n.get_extra_update())
    
    mentions = getattr(n, "mentions", None)
    if mentions is not None:
        meta["mentions"] = _json_or_none(
            [r.model_dump(field_mode="backend") for r in mentions]
        )
    meta = _strip_none(meta)
    return doc, meta

def _edge_doc_and_meta(e: Union["Edge", "PureChromaEdge"]) -> tuple[str, dict]:
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
    })
    # if hasattr(e,"mentions"):
    #     meta['mentions'] = _json_or_none([r.model_dump(field_mode = 'backend') for r in (e.mentions or [])])
    mentions = getattr(e, "mentions", None)
    if mentions is not None:
        meta["mentions"] = _json_or_none(
            [r.model_dump(field_mode="backend") for r in mentions]
        )
    meta = _strip_none(meta)
    return doc, meta

def _default_verification(note: str = "fallback span") -> MentionVerification:
    return MentionVerification(method="heuristic", is_verified=False, score=None, notes=note)


def _ensure_ref_span(ref: Span, doc_id: str) -> Span:
    # Make sure URLs point at this doc and spans are complete
    r = ref.model_copy(deep=True)
    if not r.collection_page_url:
        r.collection_page_url = f"document_collection/{doc_id}"
    if not r.document_page_url or str(doc_id) not in r.document_page_url:
        r.document_page_url = _DOC_URL.format(doc_id=doc_id)
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

def _normalize_mentions(mentions: Optional[List[Span]], doc_id: str) -> List[Span]:
    if not mentions or len(mentions) == 0:
        raise Exception("missing mentions")
    return [_ensure_ref_span(ref, doc_id) for ref in mentions]

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
def _update_record_lifecycle(
    *,
    collection: Any,
    record_id: str,
    lifecycle_patch: dict,
) -> bool:
    """
    Update one record's metadata (and optionally document metadata) with lifecycle fields.
    Returns True if updated, False if record not found.
    """
    got = collection.get(ids=[record_id], include=["documents", "metadatas"])
    ids = got.get("ids") or []
    if not ids:
        return False

    doc = (got.get("documents") or [None])[0]
    meta = (got.get("metadatas") or [None])[0]
    base = _safe_json_dict(doc)

    # merge lifecycle fields into both stored metadata and embedded doc.metadata (if you use it)
    base_meta = base.get("metadata") if isinstance(base.get("metadata"), dict) else {}
    new_doc_meta = _merge_meta(base_meta, lifecycle_patch)
    base["metadata"] = new_doc_meta

    new_meta = _merge_meta(meta if isinstance(meta, dict) else {}, lifecycle_patch)

    collection.update(
        ids=[record_id],
        documents=[json.dumps(base, ensure_ascii=False)],
        metadatas=[new_meta],
    )
    return True
def tombstone_record(
    *,
    collection: Any,
    record_id: str,
    reason: str | None = None,
    deleted_by: str | None = None,
    deleted_at: str | None = None,
) -> bool:
    """
    Soft-delete a record in-place (tombstone). Keeps the record addressable by ID.
    """
    patch = {
        "lifecycle_status": "tombstoned",
        "redirect_to_id": None,
        "deleted_at": deleted_at or _utc_now_iso(),
    }
    if reason:
        patch["delete_reason"] = reason
    if deleted_by:
        patch["deleted_by"] = deleted_by

    return _update_record_lifecycle(collection=collection, record_id=record_id, lifecycle_patch=patch)
def tombstone_records(
    *,
    collection: Any,
    record_ids: Sequence[str],
    reason: str | None = None,
    deleted_by: str | None = None,
    deleted_at: str | None = None,
) -> int:
    """
    Tombstone many records. Returns count updated.
    """
    if not record_ids:
        return 0

    got = collection.get(ids=list(record_ids), include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []

    patch = {
        "lifecycle_status": "tombstoned",
        "redirect_to_id": None,
        "deleted_at": deleted_at or _utc_now_iso(),
    }
    if reason:
        patch["delete_reason"] = reason
    if deleted_by:
        patch["deleted_by"] = deleted_by

    out_ids, out_docs, out_metas = [], [], []
    for rid, doc, meta in zip(ids, docs, metas):
        base = _safe_json_dict(doc)
        base_meta = base.get("metadata") if isinstance(base.get("metadata"), dict) else {}
        base["metadata"] = _merge_meta(base_meta, patch)

        new_meta = _merge_meta(meta if isinstance(meta, dict) else {}, patch)

        out_ids.append(str(rid))
        out_docs.append(json.dumps(base, ensure_ascii=False))
        out_metas.append(new_meta)

    if out_ids:
        collection.update(ids=out_ids, documents=out_docs, metadatas=out_metas)
    return len(out_ids)    
def redirect_record(
    *,
    collection: Any,
    from_id: str,
    to_id: str,
    reason: str | None = None,
    deleted_by: str | None = None,
    deleted_at: str | None = None,
) -> bool:
    """
    Mark `from_id` as tombstoned and redirect it to `to_id`.
    The `to_id` record is not modified here (that’s a separate 'merge' concern).
    """
    if from_id == to_id:
        # no-op: cannot redirect to itself
        return False

    patch = {
        "lifecycle_status": "tombstoned",
        "redirect_to_id": str(to_id),
        "deleted_at": deleted_at or _utc_now_iso(),
    }
    if reason:
        patch["delete_reason"] = reason
    if deleted_by:
        patch["deleted_by"] = deleted_by

    return _update_record_lifecycle(collection=collection, record_id=from_id, lifecycle_patch=patch)
def redirect_records(
    *,
    collection: Any,
    redirects: Sequence[tuple[str, str]],  # (from_id, to_id)
    reason: str | None = None,
    deleted_by: str | None = None,
    deleted_at: str | None = None,
) -> int:
    """
    Redirect many records. Returns count updated.
    """
    if not redirects:
        return 0

    from_ids = [a for a, _ in redirects]
    to_map = {a: b for a, b in redirects if a != b}

    got = collection.get(ids=from_ids, include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []

    base_patch = {
        "lifecycle_status": "tombstoned",
        "deleted_at": deleted_at or _utc_now_iso(),
    }
    if reason:
        base_patch["delete_reason"] = reason
    if deleted_by:
        base_patch["deleted_by"] = deleted_by

    out_ids, out_docs, out_metas = [], [], []
    for rid, doc, meta in zip(ids, docs, metas):
        sid = str(rid)
        to_id = to_map.get(sid)
        if not to_id:
            continue

        patch = {**base_patch, "redirect_to_id": str(to_id)}

        base = _safe_json_dict(doc)
        base_meta = base.get("metadata") if isinstance(base.get("metadata"), dict) else {}
        base["metadata"] = _merge_meta(base_meta, patch)

        new_meta = _merge_meta(meta if isinstance(meta, dict) else {}, patch)

        out_ids.append(sid)
        out_docs.append(json.dumps(base, ensure_ascii=False))
        out_metas.append(new_meta)

    if out_ids:
        collection.update(ids=out_ids, documents=out_docs, metadatas=out_metas)
    return len(out_ids)
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



def candiate_filtering_callback(llm: BaseChatModel, conversation_content, 
                                cand_node_list_str, cand_edge_list_str, 
                                candidate_node_ids: list[str], candidate_edge_ids: list[str], context_text):
    # candidate_node_ids = [i.id for i in candidates.nodes]
    # candidate_edge_ids = [i.id for i in candidates.edges]
    max_retry = 3
    err_messages = []
    for _retry in range(max_retry):
        
        filter_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant filtering knowledge graph nodes."),
            ("human", f"User Input: {conversation_content}\n\n" + 
                        (f"Context: {context_text}\n\n" if context_text else "") +
                        f"Candidate Nodes:\n{cand_node_list_str}\n\n"
                        f"Candidate Edges:\n{cand_edge_list_str}\n\n"
                        "Return a JSON list of IDs for nodes and edges that are RELEVANT to the user input. "
                        # "Example: ['id1', 'id2']. Return empty list if none."
                        )
        ]+ err_messages)
    # Simple invocation (can optimize with structured output)
        chain = filter_prompt | llm.with_structured_output(FilteringResponse, include_raw = True)
        resp: dict | BaseModel = chain.invoke({})
        if isinstance(resp, BaseModel):
            raise Exception("Unreachable")
        
        if err := resp.get("parsing_error"):
            err_messages.append( ("system", f"error: {str(err)}"))
            continue
        else:
            resp2: dict = resp
        from typing import cast
        parsed: BaseModel | None
        if parsed := resp2.get("parsed"):
            parsed2: FilteringResponse = cast(FilteringResponse, parsed)
            not_node_candidate = set(parsed2.relevant_ids.node_ids).difference(set(candidate_node_ids))
            not_edge_candidate = set(parsed2.relevant_ids.edge_ids).difference(set(candidate_edge_ids))
            # set(parsed2.relevant_ids).issubset(set(candidate_ids ))
            if not_node_candidate or not_edge_candidate:
                if not_node_candidate:
                    err_messages.append(("system", str(Exception(f"Non candidates ids returned {not_node_candidate}"))))
                if not_edge_candidate:
                    err_messages.append(("system", str(Exception(f"Non candidates ids returned {not_node_candidate}"))))
                continue
                # raise Exception(f"Non candidates ids returned {not_candidate}")
            else:
                
                return FilteringResult(node_ids = parsed2.relevant_ids.node_ids, edge_ids = parsed2.relevant_ids.edge_ids), parsed2.reasoning
        else:
            raise Exception("Unreachable")
    raise Exception("Exhaused all models")
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


import functools
from typing import Any, Callable, cast, Sequence

import numpy as np
from numpy.typing import NDArray

# If you're using Chroma's base classes:
from chromadb.utils.embedding_functions import EmbeddingFunction



class CustomEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, model_name: str = "all-minilm:l6-v2"):
        import ollama

        _raw = functools.partial(ollama.embeddings, model=model_name)

        def ef(prompts: Sequence[str]) -> Embeddings:
            res: Embeddings = []
            for p in prompts:
                # Boundary: ollama types are weak -> cast once.
                out = ollama.embeddings(model=model_name,prompt=p)

                vec_any = cast(Any, out).embedding  # "embedding" is usually list[float] or ndarray
                # Normalize to ndarray[float] for math
                r = np.asarray(vec_any, dtype=float)

                norm_val = float(np.linalg.norm(r))
                if norm_val == 0.0:
                    res.append(r.tolist())
                else:
                    res.append((r / norm_val).tolist())
            return res

        self._emb: Callable[[Sequence[str]], Embeddings] = ef

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return self._emb(documents_or_texts)



@dataclass
class GraphKnowledgeEngine:
    """
    The **Base Abstraction** for the knowledge graph/graph database.

    This engine manages the lifecycle of **provenance-heavy primitives** (`Node`, `Edge`).
    Unlike typical graph databases, every primitive here carries rich metadata about its origin
    (source document, span, verification status).

    Key responsibilities:
    - Persisting nodes and edges with full provenance.
    - Managing extensions like `ConversationNode` and `WorkflowNode` (via subclasses).
    - Providing low-level to high-level APIs for extraction, storage, and adjudication.

    Methods are generally arranged from low-level generic helpers to task-specific calls.
    High-level orchestration for extracting, storing, and adjudicating knowledge graph data.
    """

    #--------------------
    # Puhlic Interface
    #--------------------
    def _filter_items_by_resolve_mode(self, items: list[T], resolve_mode: str) -> list[T]:
        if resolve_mode == "include_tombstones":
            return items
        if resolve_mode in ("active_only", "redirect"):
            return [x for x in items if ((getattr(x, "metadata", {}) or {}).get("lifecycle_status") or "active")=="active"]
        return items    
    def _resolve_redirect_chain(
        self,
        *,
        initial_items: list[T],
        fetch_by_ids: Callable[[Sequence[str]], list[T]],
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ) -> list[T]:
        if resolve_mode != "redirect":
            return initial_items

        resolved: dict[str, T] = {}
        visited: set[str] = set()
        frontier: list[T] = list(initial_items)

        while frontier:
            next_frontier_ids: set[str] = set()

            for item in frontier:
                item_id = str(item.id)
                if item_id in visited:
                    continue
                visited.add(item_id)

                meta = getattr(item, "metadata", {}) or {}
                status = meta.get("lifecycle_status")
                redirect_to = meta.get("redirect_to_id")

                if status == "tombstoned" or redirect_to:
                    next_frontier_ids.add(str(redirect_to))
                else:
                    resolved[item_id] = item

            if not next_frontier_ids:
                break

            frontier = fetch_by_ids(list(next_frontier_ids))

        # In redirect mode: never return tombstoned items
        return [x for x in resolved.values() if ((getattr(x, "metadata", {}) or {}).get("lifecycle_status") or "active")=="active"]

    def node_ids_by_doc(self, doc_id: str) -> List[str]:
        
        return self._nodes_by_doc(doc_id)

    def edge_ids_by_doc(self, doc_id: str) -> List[str]:
       
        return self._edge_ids_by_doc(doc_id)

    @property
    def embedding_function(self):
        return self._ef
    @embedding_function.setter
    def embedding_function(self, val):
        self._ef = val
        
    def _infer_doc_id_from_ref(self, ref: Span) -> Optional[str]:
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
        node_or_id: Union[Node | Edge, str],  # also works if you pass an Edge or edge id
        *,
        window_chars: int = 120,
        max_contexts: Optional[int] = None,
        prefer_label_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build adjudication-ready reference excerpts for a Node (or Edge).
        Strategy:
        - Prefer the stored ReferenceSession.excerpt (cheap, already localized).
        - If document text is available, try to locate the excerpt (or node label) in full text
            and expand with ±window_chars for richer context.
        - Always return provenance + verification info (doc_id, page spans, urls, etc.).
        """
        # 1) Materialize the object
        from .models import GraphEntityRefBase
        if isinstance(node_or_id, GraphEntityRefBase):
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
        doc_cache = {}
        
        for ref in refs:
            # 2) locate the backing document
            doc_id = self._infer_doc_id_from_ref(ref)
            pages = doc_cache.get(doc_id)
            full_doc = None
            if not pages:
                full_doc = self._fetch_document_text(doc_id) if doc_id else None
                pages = self._coerce_pages(full_doc)
                doc_cache[doc_id] = pages
            excerpt = getattr(ref, "excerpt", None)
            mention = excerpt or (label or "")

            ctx_text = excerpt or None
            span_start = None
            span_end = None
            # if ref.start_page == ref.end_page:
            #     pages[ref.start_page]
            # for p in range(ref.start_page, ref.end_page):
            
            def _coerce_to_referencable_text(text_or_ast_str):
                
                try:
                    return ('\n'.join((i['text'] for i in ast.literal_eval(text_or_ast_str)['OCR_text_clusters'])))
                except Exception as e :
                    return text_or_ast_str
                    
                if type(text_or_ast_str) is str:
                    return text_or_ast_str
                else:
                    return ast.literal_eval(text_or_ast_str)
                pass
            if full_doc: # has doc in db
                page_relevant = {p[0]: p[1] for p in pages if (p[0] >= ref.start_page and  p[0] <= ref.end_page )}
                if ref.start_page and ref.end_page:
                    if ref.start_page == ref.end_page:
                        ctx_text = ""
                        if ref.start_char and ref.end_char:
                            try:
                                page_split_page = _coerce_to_referencable_text(page_relevant[ref.start_page])
                            except:
                                raise
                            ctx_text = _coerce_to_referencable_text(page_relevant[ref.start_page])[max(ref.start_char-window_chars, 0): ref.end_char+window_chars]
                    else:
                        ctx_text = ""
                        ctx_text += _coerce_to_referencable_text(page_relevant[ref.start_page])[max(ref.start_char-window_chars, 0):]
                        for p_num in range(ref.start_page+1,ref.end_page):
                            ctx_text += _coerce_to_referencable_text(page_relevant[ref.start_page])[:]
                            ctx_text += '\n\f'
                        ctx_text += _coerce_to_referencable_text(page_relevant[ref.start_page])[: ref.end_char+window_chars]
                    if len(ctx_text) == 0:
                        raise Exception("Context empty")
                if ctx_text is None:
                    # Try exact excerpt first (best anchor)
                    idx = full_doc.find(excerpt) if excerpt else -1
                    # Fallback to label if allowed
                    if idx < 0 and label and prefer_label_fallback:
                        idx = full_doc.find(label)

                    if idx >= 0:
                        # If we matched on excerpt, use its length; else label length
                        length = len(excerpt) if excerpt else (len(label) if label else 0)
                        span_start = idx
                        span_end = idx + length
                        left = max(0, span_start - window_chars)
                        right = min(len(full_doc), span_end + window_chars)
                        ctx_text = full_doc[left:right]
            

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
                "context": ctx_text,             # expanded context or stored excerpt
                "mention": mention,              # what to highlight / quote in a prompt
                "loc_found": (span_start is not None),
                "loc_span": [span_start, span_end] if span_start is not None else None,
                # Always include the raw ref in case you need exact fields later
                "ref": ref.model_dump(),
            })

            if max_contexts and len(out) >= max_contexts:
                break

        return out
    ## update only allow redirect and soft delete  and delete node 
    def tombstone_node(self, node_id: str, **kw) -> bool:
        return tombstone_record(collection=self.node_collection, record_id=node_id, **kw)

    def redirect_node(self, from_id: str, to_id: str, **kw) -> bool:
        return redirect_record(collection=self.node_collection, from_id=from_id, to_id=to_id, **kw)

    def tombstone_edge(self, edge_id: str, **kw) -> bool:
        return tombstone_record(collection=self.edge_collection, record_id=edge_id, **kw)

    def redirect_edge(self, from_id: str, to_id: str, **kw) -> bool:
        return redirect_record(collection=self.edge_collection, from_id=from_id, to_id=to_id, **kw)
    ## get
    def get_nodes(
        self,
        ids: Sequence[str] | None = None,
        node_type: Type[Node] | None = None,
        include: None | list[str] = None,
        where = None,
        limit : None | int = 200,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> List[Node]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"] 

        if not node_type:
            node_type = ConversationNode if self.kg_graph_type == "conversation" else Node
        
        # IMPORTANT: ID fetch must NOT filter out tombstones, or redirect cannot start.
        got = self.node_collection.get(
            ids=ids,
            include=include,
            where = where,
            limit = limit
        )
        nodes = self.nodes_from_single_or_id_query_result(got, node_type=node_type)

        # Follow redirects (may require fetching tombstoned targets too)
        nodes = self._resolve_redirect_chain(
            initial_items=nodes,
            resolve_mode=resolve_mode,
            fetch_by_ids=lambda redirect_ids: self.get_nodes(
                redirect_ids,
                node_type=node_type,
                resolve_mode=resolve_mode,  # allow chain traversal
            ),
        )

        # Apply final mode filter (active_only/redirect hide tombstones)
        nodes = self._filter_items_by_resolve_mode(nodes, resolve_mode)

        return nodes
    
    def query_nodes(self,*args, query = None, 
            query_embeddings = None,  
            include=["documents", "embeddings", "metadatas"],
            node_type: Type[Node] = Node, **kwargs):
        if query_embeddings is not None:
            if query is not None:
                raise Exception("either query or query embedding but not both specified.")
        else:
            if query is not None:
                query_embeddings = self._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")
        
        got = self.node_collection.query(query_embeddings=query_embeddings, *args, 
                                        include=include, **kwargs)
        
        return self.nodes_from_query_result(got, node_type = node_type)
    def query_edges(self,*args, query = None, query_embeddings = None, 
                    include=["documents", "embeddings", "metadatas"],
                    edge_type: Type[Edge] = Edge, **kwargs):
        if query_embeddings is not None:
            pass
        else:
            if query is not None:
                query_embeddings = self._iterative_defensive_emb(query)
            else:
                raise ValueError("either query or query embeddings must be specified")
        
        got = self.edge_collection.query(query_embeddings=query_embeddings, *args, 
                                        include=include, **kwargs)
        
        return self.edges_from_query_result(got, edge_type=edge_type)
    def nodes_from_single_or_id_query_result(
            self,
            got,
            node_type: Type[TNode] = Node,
        ) -> list[TNode]:
        docs: list[str] = cast(list[str], got.get("documents"))
        
        if docs is None:
            raise Exception("Missing docs")
        
        embs = got.get("embeddings")
        if embs is None:
            raise Exception("Missing Embeddings")

        embs = cast(list[list[float]], embs)
        metadatas = cast(list[dict[str, Any]], got.get("metadatas"))
        if metadatas is None:
            raise Exception("Missing Metadatas")
        
        res = []
        from . import models
        for d, emb, metadata in zip(docs, embs, metadatas):
            if type(emb) is list:
                pass
            elif type(emb) is np.ndarray:
                emb = emb.tolist()
            json_d = json.loads(d)
            entity_type = metadata.get('entity_type')
            override_node_type = None
            _class_name = metadata.get("_class_name")
            if _class_name:
                node_cls = getattr(models, _class_name)
                if node_cls:
                    override_node_type = node_cls
            if not override_node_type:
                if entity_type == "workflow_checkpoint":
                    if self.kg_graph_type == "workflow":
                        override_node_type = WorkflowCheckpointNode
            
                
            json_d.update({"embedding": emb, "metadata": metadata})
            res.append((override_node_type or node_type).model_validate(json_d))
        return res
    def edges_from_single_or_id_query_result(self, got, edge_type: Type[Edge] = Edge, include = None):
        if include is None:
            include = ['documents', 'metadatas', 'embeddings']
        docs: list[str] = cast(list[str], got.get("documents"))
        ids = got['ids']
        if docs is None:
            if "documents" in include:
                raise Exception("Missing docs")
        
        embs = got.get("embeddings")
        if embs is None:
            raise Exception("Missing Embeddings")

        embs = cast(list[list[float]], embs)
        metadatas = cast(list[dict[str, Any]], got.get("metadatas"))
        if metadatas is None:
            raise Exception("Missing Metadatas")
        
        res = []
        import numpy as np
        for d, emb, metadata in zip(docs, embs, metadatas):
            if type(emb) is list:
                pass
            elif type(emb) is np.ndarray:
                emb = emb.tolist()
            json_d = json.loads(d)
            json_d.update({"embedding": emb, "metadata": metadata})
            entity_type = metadata.get('entity_type')
            override_edge_type = None
            _class_name = metadata.get("_class_name")
            if _class_name:
                edge_cls = getattr(models, _class_name)
                if edge_cls:
                    override_edge_type = edge_cls
   
            res.append((override_edge_type or edge_type).model_validate(json_d))
        return res
    def nodes_from_query_result(self, gots, node_type: Type[Node] = Node):
        res = []
        for i_q in range(len(gots['ids'])):
            n_doc = len(gots["ids"][i_q])
            for ids, docs, embs, metadatas in zip(
                                             gots.get("ids"),
                                             gots.get("documents") if gots.get("documents") is not None else  [[]]*n_doc, 
                                             gots.get("embeddings") if gots.get("embeddings") is not None else [[]]*n_doc, 
                                             gots.get("metadatas") if gots.get("metadatas") is not None else [[]]*n_doc):
                docs: list[str] = cast(list[str], docs)
                got = {"documents": docs, "embeddings": embs, "metadatas": metadatas}
                single_res = self.nodes_from_single_or_id_query_result(got, node_type = node_type)
                res.append(single_res)
        return res
    def edges_from_query_result(self, gots, edge_type: Type[Edge] = Edge):
        res = []
        for i_q in range(len(gots['ids'])):
            n_doc = len(gots["ids"][i_q])
            for ids, docs, embs, metadatas in zip(gots.get("ids"), 
                                             gots.get("documents") if gots.get("documents") is not None else  [[]]*n_doc, 
                                             gots.get("embeddings") if gots.get("embeddings") is not None else [[]]*n_doc, 
                                             gots.get("metadatas") if gots.get("metadatas") is not None else [[]]*n_doc):
                docs: list[str] = cast(list[str], docs)
                got = {"ids": ids, "documents": docs, "embeddings": embs, "metadatas": metadatas}
                single_res = self.edges_from_single_or_id_query_result(got, edge_type = edge_type)
                res.append(single_res)
        return res
    def _where_update_from_resolve_mode(self, resolve_mode : Literal["active_only" , "redirect", "include_tombstones"]):
        match resolve_mode:
            case "active_only":
                return {"lifecycle_status":"active"}
            case "redirect":
                return {}
            case "include_tombstones":
                return {}
        

    def get_edges(
        self,
        ids: Sequence[str] | None = None,
        edge_type: Type[Edge] | None = None,
        where = None,
        limit : int | None = 400,
        include: None | list[str] = None,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> List[Edge]:
        if include is None:
            include = ["documents", "embeddings", "metadatas"]
        if not edge_type:
            edge_type = ConversationEdge if self.kg_graph_type == "conversation" else Edge

        # IMPORTANT: ID fetch must NOT filter out tombstones, or redirect cannot start.
        got = self.edge_collection.get(
            ids=ids,
            include=include,
            where = where,
            limit = limit
        )
        edges = self.edges_from_single_or_id_query_result(got, edge_type=edge_type, include = include)

        # Follow redirects (may require fetching tombstoned targets too)
        edges = self._resolve_redirect_chain(
            initial_items=edges,
            resolve_mode=resolve_mode,
            fetch_by_ids=lambda redirect_ids: self.get_edges(
                redirect_ids,
                edge_type=edge_type,
                resolve_mode=resolve_mode,  # allow chain traversal
            ),
        )

        # Apply final mode filter (active_only/redirect hide tombstones)
        edges = self._filter_items_by_resolve_mode(edges, resolve_mode)

        return edges

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
    # @staticmethod
    # def _default_ref(doc_id: str, excerpt: Optional[str] = None) -> Span:
    #     return _default_ref(doc_id, excerpt)
    #     pass
    @staticmethod
    def chroma_sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop keys whose values are None. ChromaDB metadata rejects None values."""
        return _strip_none(metadata) #{k: v for k, v in metadata.items() if v is not None}
    # @staticmethod
    # def _strip_none(d: dict):
    #     return _strip_none(d)
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

    def _preflight_validate(self, parsed: LLMGraphExtraction|PureGraph|GraphExtractionWithIDs, alias_key: str, alias_book: AliasBook | None = None):
        """Ensure every endpoint refers to a real id (in-batch or already in DB)."""
        self._resolve_llm_ids(alias_key, parsed, alias_book = alias_book)  # allocate/resolve first

        batch_node_ids = {n.id for n in parsed.nodes}
        batch_edge_ids = {e.id for e in parsed.edges}

        need_nodes, need_edges = set(), set()
        for e in parsed.edges:
            need_nodes.update(e.source_ids or [])
            need_nodes.update(e.target_ids or [])
            if getattr(e, "source_edge_ids", None):
                need_edges.update(e.source_edge_ids or [])
            if getattr(e, "target_edge_ids", None):
                need_edges.update(e.target_edge_ids or [])

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
    def _assert_endpoints_exist(self, edge: Edge | PureChromaEdge):
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
        node_ids_added = set()
        edge_ids_added = set()
        nodes_added = edges_added = 0
        span_validator: BaseDocValidator = self.get_span_validator_of_doc_type(doc_id=doc_id)
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]

            if kind == "node":
                ln: Node = obj
                # existing?
                if self._exists_node(rid):
                    # merge refs only if present
                    if ln.mentions:
                        prior = self.node_collection.get(ids=[rid], include=["documents", "metadatas"])
                        prior_meta = (prior.get("metadatas") or [None])[0] or {}
                        prior_mentions = cast(str, prior_meta.get("mentions"))
                        merged_list, merged_json = _merge_refs(prior_mentions, ln.mentions)
                        # update JSON & metadata refs
                        doc = prior["documents"]
                        if not doc:
                            raise Exception("missing documents")
                        n = Node.model_validate_json(doc[0])
                        groundings = Grounding(spans = [Span.model_validate(r) for r in merged_list])
                        for sp in groundings.spans:
                            span_validator.validate_span(span = sp)
                        n.mentions = [groundings]
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
                ln.mentions = self._dealias_span(ln.mentions, doc_id)
                node = ln
                self.add_node(node, doc_id=doc_id)
                nodes_added += 1
                node_ids_added.add(node.id)
            else:  # edge
                le: Edge = obj
                if self._exists_edge(rid):
                    # merge refs; optionally reconcile endpoints if you permit LLM to edit them
                    if le.mentions:
                        prior = self.edge_collection.get(ids=[rid], include=["documents", "metadatas"])
                        prior_meta = (prior.get("metadatas") or [None])[0] or {}
                        prior_mentions = cast(str, prior_meta.get("mentions"))
                        merged_list, merged_json = _merge_refs(prior_mentions, le.mentions)
                        doc = prior["documents"]
                        if not doc:
                            raise Exception("missing documents")
                        e = Edge.model_validate_json(doc[0])
                        groundings = Grounding(spans = [Span.model_validate(r) for r in merged_list])
                        for sp in groundings.spans:
                            span_validator.validate_span(span = sp)
                        e.mentions = [groundings]
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
                edge = le
                self.add_edge(edge, doc_id=doc_id)
                edge_ids_added.add(edge.id)
                edges_added += 1
        return {
            "document_id": doc_id,
            "node_ids": nodes_added,
            "edge_ids": edges_added,
            "nodes_added": len(node_ids_added),
            "edges_added": len(edge_ids_added),
        }
        # return {"nodes_added": nodes_added, "edges_added": edges_added}
    def _resolve_llm_ids(self, doc_id: str, parsed: LLMGraphExtraction | PureGraph | GraphExtractionWithIDs, alias_book: AliasBook | None = None) -> None:
            """
            In-place:
            - allocate UUIDs for all new nodes (nn:*) and new edges (ne:*),
            - de-alias existing N*/E* to UUIDs,
            - resolve edge endpoints (node + edge).
            the id can either be, a real id, token indicating a new node that require new id, or a short id that need dealiasing
            """
            # alias book → real ids
            if alias_book is None:
                # create a book if no book provided
                book = self._alias_book(doc_id)
            else:
                book = alias_book
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
                # Pull a token in a way that keeps the type `str | None`
                token = n.id or cast(str | None, getattr(n, "local_id", None))

                if token is None:
                    n.id = str(uuid.uuid4())
                    continue

                if token == "":
                    n.id = str(uuid.uuid4())
                    continue

                if _is_new_node(token):
                    rid = nn2uuid.get(token)
                    if rid is None:
                        rid = str(uuid.uuid4())
                        nn2uuid[token] = rid
                    n.id = rid
                else:
                    n.id = de_alias(token)
            # Second pass: edges → IDs
            ne2uuid: dict[str, str] = {}
            for e in parsed.edges:
                tok = getattr(e, 'local_id', None) or e.id
                if (not tok) or _is_new_edge(tok) or _is_new_edge(getattr(e, 'local_id', None)):
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
    def _extract_graph_with_llm_aliases(self, content: str, alias_nodes_str: str, alias_edges_str: str, instruction_for_node_edge_contents_parsing_inclusion:  None | str = None,
                                        last_iteration_result : dict | None = None):
        if instruction_for_node_edge_contents_parsing_inclusion is None:
            instruction_for_node_edge_contents_parsing_inclusion = ("Nodes should include at least: Parties, Obligations, Rights, Deliverables, Payment Terms, Termination Conditions, Confidentiality Clauses, Governing Law, Dates, and Penalties.  "
            "Edges should capture: (Party → Obligation), (Obligation → Condition), (Party → Right), (Obligation → Deliverable), (Clause → Governing Law).  ")
        template_messages = [
            ("system",
            "You are an expert knowledge graph extractor. "
            "You are an information extraction system that converts legal contracts into a knowledge graph.  "
            "Your task: extract ALL entities (nodes) and relationships (edges) from the text. "
            f"{instruction_for_node_edge_contents_parsing_inclusion}"
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
            "- Important!: Span.excerpt MUST agree with corresponding zero-indexed start to end index. Direct identical text and no paraphrased is allowed. "
            "e.g. if snippet world from 'hello world!', word is 6 to 11 index. content[6:11] => 'world' in python indexing sense"
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
            "Document:\n```{document}```\n\n"
            "Return only the structured JSON for the schema.")
        ]
        to_append = []
        from pydantic import BaseModel
        if last_iteration_result and last_iteration_result.get("error"):
            last_parsed: BaseModel = last_iteration_result.get('parsed') # type: ignore
            last_error :str = str(last_iteration_result.get("error"))
            to_append.append(
                ("system", 
                    f"last answer has error \n\n Last attempt: ```{last_parsed.model_dump()}```\n\n"
                    f"error from last attempt: ```{last_error}```"
                 )
            )
            template_messages.extend(to_append)
            # ([("system", f"last answer has error \n\n Last attemp: ```{last_iteration_result.get('parsed').model_dump()} ``` ")]  else []
        prompt = ChatPromptTemplate.from_messages(template_messages)
        try:
            chain = prompt | self.llm.with_structured_output(LLMGraphExtraction['llm'], 
                                                             include_raw=True)
            from langchain_core.runnables import Runnable
            steps : list[Runnable] = chain.steps # type: ignore   langchain internal step is not exposed
            realised_prmopt = steps[0].invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content, "_DOC_ALIAS" : _DOC_ALIAS})
            llm_raw = steps[1].invoke(realised_prmopt)
            result = steps[2].invoke(llm_raw)
            # result = chain.invoke({"alias_nodes": alias_nodes_str, "alias_edges": alias_edges_str, "document": content, "_DOC_ALIAS" : _DOC_ALIAS})
        except Exception as e:
            raise e
        return result.get("raw"), result.get("parsed"), result.get("parsing_error")
    def _de_alias_ids_in_result(self, doc_id: str, parsed: LLMGraphExtraction) -> LLMGraphExtraction:
        """Map aliases back to real UUIDs according to strategy."""
        if ID_STRATEGY == "base62":
            def r(s: str): # type: ignore
                if not s:
                    raise ValueError("s cannot be None or Falsy")
                    return s
                if s.startswith("N~"):
                    return base62_to_uuid(s[2:])
                if s.startswith("E~"):
                    return base62_to_uuid(s[2:])
                return s
        else:
            book = self._alias_book(doc_id)
            def r(s: str):
                if not s:
                    raise ValueError("s cannot be None or Falsy")
                    return s
                return book.alias_to_real.get(s, s)

        # mutate copy
        for n in parsed.nodes:
            if n.id: 
                n.id = r(n.id)
        for e in parsed.edges:
            if e.id: 
                e.id = r(e.id)
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
            first: list = content_or_pages[0]
            # list of (page, text)
            if isinstance(first, (list, tuple)) and len(first) == 2:
                return [(int(p), str(t or "")) for p, t in content_or_pages]
            # list of dicts with page/text
            if isinstance(first, dict) and "text" in first:
                out = []
                for i, item in enumerate(content_or_pages, start=1):
                    p = int(item.get("page", i))
                    out.append((p, str(item.get("text", '\n'.join(i['text'] for i in item.get("OCR_text_clusters", ""))) or "")))
                return out
            # list of dict of plain page texts
            if 'pdf_page_num' in first[0]:
                try:
                    return [(t['pdf_page_num'], str(t or "")) for t in content_or_pages]
                except KeyError:
                    raise Exception("Value inconsistency, each page is a dict, some have 'pdf_page_num' but some do not")
            # fallback, each page simply a tuple of page number with stringified iter item if it is other iterables
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
        embedding_function : EmbeddingFunctionLike| None=None,
        embedding_cache_path: str | None = None,
        proposer=None,               # callable(pairs) -> List[LLMMergeAdjudication]
        adjudicator=None,            # callable(left: Node, right: Node) -> AdjudicationVerdict
        merge_policy=None,           # callable(left, right, verdict) -> str (canonical_id)
        verifier=None,               # callable(extracted, full_text, ref, **kw) -> ReferenceSession
        kg_graph_type : EngineType = 'knowledge',
        debug_dir: pathlib.Path | None = None
    ):
        """
        embedding_function: callable(texts: List[str]) -> List[List[float]].
          If None, defaults to SentenceTransformerEmbeddingFunction with model:
          - default_st_model argument, or
          - ENV SENTENCE_TRANSFORMERS_MODEL, or
          - "all-MiniLM-L6-v2".
        """
        self.meta_sqlite = EngineSQLite(pathlib.Path(persist_directory or "./chroma_db"), 'meta.sqlite')
        self.meta_sqlite.ensure_initialized()
        self.changes = ChangeBus()
        if cdc_publish_endpoint := os.environ.get('CDC_PUBLISH_ENDPOINT'):
            self.changes.add_sink(FastAPIChangeSink(cdc_publish_endpoint))
        
        # from .debug_producer import DebugEventProducer
        # self._debug_producer = DebugEventProducer("http://127.0.0.1:8000")
        self._oplog = None
        if debug_dir is not None:
            self._oplog = OplogWriter(debug_dir / "changes.jsonl", fsync=False)
        self.tool_call_id_factory: Callable[[],str] | None = None
        self.kg_graph_type = kg_graph_type
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
        ef = embedding_function or CustomEmbeddingFunction()
        self.embedding_length_limit = 512
        self._ef : EmbeddingFunctionLike = ef#embedding_function or ef #embedding_functions.DefaultEmbeddingFunction()
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
        from threading import Lock
        self.collection_lock={"node": Lock(), "edge": Lock()}
        
        self.node_index_collection = self.chroma_client.get_or_create_collection(
            "nodes_index", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self.node_collection = self.chroma_client.get_or_create_collection(
            "nodes", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self.node_collection.get(limit = 1)
        with self.get_collection_lock('node')[0]:
            self.node_collection
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
                        model = "gemini-3-flash-preview",
                        # model = "gemini-3-flash-preview",
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
                _emb = AzureOpenAIEmbeddings( # type: ignore
                    azure_deployment=emb_deployment,
                    openai_api_key=emb_api_key, # type: ignore
                    azure_endpoint=emb_endpoint,
                    openai_api_version=emb_api_ver, # type: ignore
                )
                def _embed_fn(text: str) -> Optional[List[float]]:
                    try:
                        v = _emb.embed_query(text)
                        return v
                    except Exception:
                        return None
                self.embeddings = _embed_fn
        
        self.memory = Memory(location = os.path.join('.', '.kg_cache'))
        self._cached_extract_graph_with_llm = self.memory.cache(self.extract_graph_with_llm, ignore = ["self"])
        self.cached_embed: Optional[Callable[[str], Iterable[float]]] = None
        if embedding_cache_path:
            self.embedding_cache = Memory(location = embedding_cache_path)
            def cached_embed(query, model_name):
                
                return self._iterative_defensive_emb(query)
            from functools import partial
            
            cached_embed = partial(cached(self.embedding_cache, cached_embed), model_name = self._ef.name())
            # MethodType
            # a.foo = MethodType(foo, a)
            self.cached_embed = cast(Callable[[str], Iterable[float]], cached_embed)
    def _emit_change(self, *, op: Op, entity: EntityRefModel, payload: object, run_id: str | None = None, step_id: str | None = None) -> None:
        seq = self.changes.next_seq()
        ev = ChangeEvent(
            seq=seq,
            op=op,
            ts_unix_ms=int(time.time() * 1000),
            entity=entity.model_dump_entity_ref(),
            payload=payload,
            run_id=run_id,
            step_id=step_id,
        )
        self.changes.emit(ev)
        if self._oplog:
            self._oplog.append(ev)

    def iterative_defensive_emb(self, emb_text0):
        if self.cached_embed:
            return self.cached_embed(emb_text0)
        else:
            return self._iterative_defensive_emb(emb_text0)
        
        
    # ... existing methods ...
    @staticmethod
    def _node_doc_and_meta(n: "Node") -> tuple[str, dict]:
        return _node_doc_and_meta(n)
    @staticmethod
    def _edge_doc_and_meta(e: "Edge") -> tuple[str, dict]:
        return _edge_doc_and_meta(e)
    def _maybe_reindex_edge_refs(self, edge: Edge, *, force: bool = False) -> None:
        
        new_fp = _refs_fingerprint(edge.mentions or [])
        meta = self.edge_collection.get(ids=[edge.id], include=["metadatas"])
        old_fp = None
        metadatas = meta.get("metadatas")
        if metadatas and metadatas[0]:
            old_fp = metadatas[0].get("edge_refs_fp")

        # Secondary guards: row-count & doc_id set must match
        got = self.edge_refs_collection.get(where={"edge_id": edge.id}, include=["documents"])
        current_rows = got.get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (edge.mentions or [])}
        count_ok = (len(current_rows) == len(edge.mentions or []))
        docset_ok = (current_doc_ids == expect_doc_ids)

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self.edge_collection.update(ids=[edge.id], metadatas=[{"edge_refs_fp": new_fp}])
            self._index_edge_refs(edge)

    def _maybe_reindex_node_refs(self, node: Node, *, force: bool = False) -> None:
        new_fp = _refs_fingerprint(node.mentions or [])
        meta = self.node_collection.get(ids=[node.id], include=["metadatas"])
        old_fp = None
        metadatas = meta.get("metadatas")
        if metadatas and metadatas[0]:
            old_fp = metadatas[0].get("node_refs_fp")

        got = self.node_refs_collection.get(where={"node_id": node.id}, include=["documents"])
        current_rows = got.get("documents") or []
        current_doc_ids = {json.loads(d).get("doc_id") for d in current_rows}
        expect_doc_ids = {getattr(r, "doc_id", None) for r in (node.mentions or [])}
        count_ok = (len(current_rows) == len(node.mentions or []))
        docset_ok = (current_doc_ids == expect_doc_ids)

        if force or (new_fp != old_fp) or (not count_ok) or (not docset_ok):
            self.node_collection.update(ids=[node.id], metadatas=[{"node_refs_fp": new_fp}])
            self._index_node_refs(node)
    def _index_edge_refs(self, edge: Edge) -> list[str]:
        self._delete_edge_ref_rows(edge.id)

        ids, docs, metas = [], [], []
        for i, ref in enumerate(edge.mentions or []):
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
            self.edge_refs_collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=[self._iterative_defensive_emb(str(d)) for d in docs])
        return ids
    def _index_node_refs(self, node: Node)-> list[str]:
        """
            create index for records, may flatten json into plain
        """
        self._delete_node_ref_rows(node.id)

        ids, docs, metas = [], [], []
        for i, ref in enumerate(node.mentions or []):
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
            self.node_refs_collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=[self._iterative_defensive_emb(str(d)) for d in docs])
        return ids
    def _alias_doc_in_prompt(self) -> str:
        # A tiny legend we show to the LLM so it knows the alias to use
        return f"Use '{_DOC_ALIAS}' whenever you need to reference the current document in ReferenceSession fields."
    def _delias_one_span(self, span: Span, real_doc_id: str) -> Span:
        span = span.model_copy(deep=True)
        # swap token in URLs
        if span.document_page_url and _DOC_ALIAS in span.document_page_url:
            span.document_page_url = span.document_page_url.replace(_DOC_ALIAS, real_doc_id)
        if span.collection_page_url and _DOC_ALIAS in span.collection_page_url:
            span.collection_page_url = span.collection_page_url.replace(_DOC_ALIAS, real_doc_id)
        # align explicit doc_id if model carries it
        if getattr(span, "doc_id", None) == _DOC_ALIAS or getattr(span, "doc_id", None) is None:
            span.doc_id = real_doc_id
        # basic span normalization (keeps your existing behavior)
        if getattr(span, "page", None) is not None:
            # if you split start/end page fields, keep your existing normalization here
            pass
        if span.start_char is not None and span.end_char is not None and span.end_char < span.start_char:
            span.end_char = span.start_char
        return span
    def _dealias_one_grounding(self, grounding: Grounding, real_doc_id: str) -> Grounding:
        out : list[Span]= []
        for span in grounding.spans:
            out.append(self._delias_one_span(span, real_doc_id ))
        GroundingOrGroundingSlice: Type = type(grounding) # more readable local variable for readability
        return GroundingOrGroundingSlice.model_validate({"spans": out})

    def _dealias_span(self, groundings: List[Grounding] | None, real_doc_id: str
                      ):
        if not groundings or len(groundings) == 0:
            # produce a default reference using the real doc id
            raise ValueError("No reference to dealias")
            # return [Span(
            #     collection_page_url=f"document_collection/{real_doc_id}",
            #     document_page_url=f"document/{real_doc_id}",
            #     excerpt=fallback_snip or None,
            #     doc_id=real_doc_id,
            # )]
        return [self._dealias_one_grounding(r, real_doc_id) for r in groundings]
    
    
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
            docs = got.get("documents")
            if docs:
                return docs[0] or ""
            else:
                raise Exception("document lost")
        # fallback: try lookup by where
        got = self.document_collection.get(where={"doc_id": document_id}, include=["documents"])
        if got and got.get("documents"):
            docs = got.get("documents")
            if docs:
                return docs[0] or ""
            else:
                raise Exception("document lost")
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
    def add_pure_node(self, node: PureChromaNode):
        # node.doc_id = doc_id
        doc, meta = _node_doc_and_meta(node)
        if meta.get("doc_id"):
            meta.pop('doc_id') 
        self.node_collection.add(
            ids=[node.id],
            documents=[doc],
            embeddings=[node.embedding] if node.embedding is not None else [self._iterative_defensive_emb(str(doc))],
            metadatas=[meta],
        )
    def add_pure_edge(self, edge: PureChromaEdge):
        s_nodes, s_edges, t_nodes, t_edges = self._split_endpoints(edge.source_ids, edge.target_ids)
        edge.source_ids = s_nodes
        edge.source_edge_ids = getattr(edge, "source_edge_ids", []) or [] + s_edges
        edge.target_ids = t_nodes
        edge.target_edge_ids = getattr(edge, "target_edge_ids", []) or [] + t_edges
        # edge.doc_id = doc_id
        # single-call safety for ad-hoc usage
        self._assert_endpoints_exist(edge)

        # receptive range counts
        node_endpoint_count = len(edge.source_ids or []) + len(edge.target_ids or [])
        edge_endpoint_count = len(getattr(edge, "source_edge_ids", []) or []) + len(getattr(edge, "target_edge_ids", []) or [])
        total_endpoint_count = node_endpoint_count + edge_endpoint_count
        doc = edge.model_dump_json(field_mode='backend', exclude = ['embedding'])
        # main edge row
        self.edge_collection.add(
            ids=[edge.id],
            documents=[doc],
            embeddings=[edge.embedding] if edge.embedding is not None else [self._iterative_defensive_emb(str(doc))],
            metadatas=[_strip_none({
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
                "node_endpoint_count": node_endpoint_count,   # receptive range
                "edge_endpoint_count": edge_endpoint_count,
                "total_endpoint_count": total_endpoint_count,
            })],
        )
    def get_collection_lock(self, collection_name):
        if collection_name not in self.collection_lock:
            raise ValueError(f"{collection_name} has not implemented lock")
        if collection_name == "node":
            col = self.node_collection
        if collection_name == "edge":
            col = self.edge_collection
        return (self.collection_lock[collection_name], col)
    @conversation_only
    def max_node_seq_present(self, conversation_id):
        return self.meta_sqlite.next_user_seq(conversation_id)
    @conversation_only
    def get_last_seq_node(self, conversation_id, buffer = 5):
        
        pass
    def add_node(self, node: Node, doc_id: Optional[str] = None):
        if doc_id is not None:
            node.doc_id = doc_id  # may use engine.extract_reference_contexts
        if self.kg_graph_type == "conversation":
            node_conv: ConversationNode = cast(ConversationNode, node)
            try:
                conv_id = node_conv.conversation_id
            except:
                conv_id = node_conv.metadata["conversation_id"]
            
            if conv_id is None:
                raise Exception('conv id required')
            self._get_last_seq_node(conv_id)
            seq = self.meta_sqlite.next_user_seq(conv_id)
            node.metadata['seq'] = seq
        doc, meta = _node_doc_and_meta(node)
        if node.embedding is None:
            node.embedding = self._iterative_defensive_emb(doc)
        meta['_class_name'] = type(node).__name__
        
        # lock, col = self.get_collection_lock('node')
        # with lock:
        
        self.node_collection.add(
            ids=[node.safe_get_id()],
            documents=[doc],
            embeddings=[node.embedding] if node.embedding is not None else [self._iterative_defensive_emb(str(doc))],
            metadatas=[meta],
        )
    
        # self.get_nodes([node.id], type(node))
        # node_cls = getattr(models, type(node).__name__)
        # self.get_nodes([node.id], node_cls)
        self._index_node_docs(node)
        self._maybe_reindex_node_refs(node)
        self._emit_change(
            op="node.upsert",
            entity=EntityRefModel(kind="node", id=node.safe_get_id(), 
                    kg_graph_type=self.kg_graph_type,
                    url=self.persist_directory),
            payload=node.to_jsonable() if hasattr(node, "to_jsonable") else node.model_dump(exclude=["embedding"]),
        )
    def _entity_is_conversation(self, node: Node | Edge):
        return type(node) in [ConversationEdge, ConversationNode]
    def _fanout_endpoints_rows(self, edge: Edge, doc_id: str | None):
        """Convert a multi endpoint to multiple rows
        
        Self-edge to endpoint node one
        
        Self-edge to endpoint node two
        
        Self-edge to endpoint node other edge one etc.
        
        """
        def _maybe_doc_for_edge(eid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self.edge_collection.get(ids=[eid], include=["metadatas"])
            metadata= meta.get("metadatas")
            if metadata and metadata[0]:
                if type(metadata[0].get("doc_id")) is str:
                    return str(metadata[0].get("doc_id"))
                else:
                    if self._entity_is_conversation(edge):
                        pass
                    else:
                        raise Exception("doc_id is not string")
                    # now have to allow conversation to have no prove
                    
            return None

        rows = []

        def _per_node_doc(nid: str) -> str | None:
            if doc_id is not None:
                return doc_id
            meta = self.node_collection.get(ids=[nid], include=["metadatas"])
            metadata= meta.get("metadatas")
            if metadata and metadata[0]:
                if type(metadata[0].get("doc_id")) is str:
                    return str(metadata[0].get("doc_id"))
                else:
                    if self._entity_is_conversation(edge):
                        pass
                    else:
                        raise Exception("doc_id is not string")
                    # now have to allow conversation to have no prove
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

        # may use engine.extract_reference_contexts
        if doc_id is not None:
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
        doc = edge.model_dump_json(field_mode='backend', exclude = ['embedding'])
        if edge.embedding is None:
            edge.embedding = self._iterative_defensive_emb(str(doc))
        # main edge row
        # from chromadb.base_types import Metadata
        doc = edge.model_dump_json(field_mode='backend', exclude = ['embedding'])
        base_metadata : list[dict[str,Any]]= [_strip_none({
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
                "references": _json_or_none([r.model_dump() for r in (edge.mentions or [])]),
                "node_endpoint_count": node_endpoint_count,   # receptive range
                "edge_endpoint_count": edge_endpoint_count,
                "total_endpoint_count": total_endpoint_count,
            })]
        if self.kg_graph_type == "conversation":
            base_metadata[0].update(_strip_none({
                'char_distance_from_last_summary': edge.metadata.get("char_distance_from_last_summary"), 
                'turn_distance_from_last_summary': edge.metadata.get("turn_distance_from_last_summary")
                }))
        if self.kg_graph_type == "workflow":
            from .models import WorkflowEdge
            edge = cast(WorkflowEdge, edge)
            edge_metadata = edge.metadata
            base_metadata[0].update(_strip_none({
                            "entity_type": edge_metadata.get("entity_type"),
                            "workflow_id": edge_metadata.get("workflow_id"),
                            "wf_priority": edge_metadata.get("wf_priority") or edge_metadata.get("priority"),
                            "wf_is_default": edge_metadata.get("wf_is_default") or edge_metadata.get("is_default"),
                            "wf_predicate": edge_metadata.get("wf_predicate") or edge_metadata.get("predicate"),
                            "wf_multiplicity": edge_metadata.get("wf_multiplicity") or edge_metadata.get("multiplicity"),
                }))
        self.edge_collection.add(
            ids=[edge.id],
            documents=[str(doc)],
            embeddings=[edge.embedding] if edge.embedding is not None else [self._iterative_defensive_emb(str(doc))],
            metadatas=base_metadata,
        )
        self._maybe_reindex_edge_refs(edge)
        # endpoints fan-out
        
        rows = self._fanout_endpoints_rows(edge, doc_id)
        if rows:
            ep_ids   = [r["id"] for r in rows]
            ep_docs  = [json.dumps(r) for r in rows]
            ep_metas: list[dict] = rows  # already sanitized (no None)
            self.edge_endpoints_collection.add(
                ids=ep_ids,
                documents=ep_docs,
                metadatas=ep_metas,
                embeddings=[self._iterative_defensive_emb(str(d)) for d in ep_docs]
            )
        
        self._emit_change(
            op="edge.upsert",
            entity=EntityRefModel(kind="edge", id=edge.safe_get_id(), 
                    kg_graph_type=self.kg_graph_type,
                    url=self.persist_directory),
            payload=edge.to_jsonable() if hasattr(edge, "to_jsonable") else edge.model_dump(exclude=["embedding"]),
        )
    def add_document(self, document: Document):
        if document.embeddings is None:
            document.embeddings = self._iterative_defensive_emb(str(document.content))
        self.document_collection.add(
            ids=[document.id],
            documents=[str(document.content)],
            embeddings = [cast(Sequence[float], document.embeddings)] if document.embeddings is not None else [self._iterative_defensive_emb(str(document.content))],
            metadatas=[_strip_none({
                "doc_id": document.id,  # <— critical
                "type": document.type,
                "metadata": _json_or_none(document.metadata),
                "domain_id": document.domain_id,
                "processed": document.processed,
            })],
        )
        
        self._emit_change(
            op="doc.upsert",
            entity=EntityRefModel(kind="doc_node", id=document.id, 
                    kg_graph_type=self.kg_graph_type,
                    url=self.persist_directory),
            payload=document.to_jsonable() if hasattr(document, "to_jsonable") else document.model_dump(exclude=['embedding']),
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
            embeddings=[self._iterative_defensive_emb(str(domain.model_dump_json()))]
        )
    def _index_node_docs(self, node: Node) -> list[str]:
        """Rebuild (node_id, doc_id) rows for this node and denormalize doc_ids on node metadata."""
        doc_ids = _extract_doc_ids_from_refs(node.mentions)

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
            self.node_docs_collection.add(ids=ids, documents=docs, metadatas=metas, embeddings = [self._iterative_defensive_emb(d) for d in docs])

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
            result = set()
            for m in (rows.get("metadatas") or []):
                if m and m.get("node_id"):
                    result.add(m.get("node_id"))
            return sorted(result)
        # slow fallback:
        got = self.node_collection.get(where={"doc_id": doc_id})
        return got.get("ids") or []

    def _edge_ids_by_doc(self, doc_id: str, insertion_method: Optional[str] = None) -> list[str]:
        if insertion_method:
            return self.ids_with_insertion_method(kind="edge", insertion_method=insertion_method, doc_id=doc_id)
        # original behavior via endpoints table:
        eps = self.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        result = set()
        for m in (eps.get("metadatas") or []):
            if m and m.get("edge_id"):
                result.add(m.get("edge_id"))
        return sorted(result)
        # return sorted({m.get("edge_id") for m in (eps.get("metadatas") or []) if m and m.get("edge_id")})

    def _prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool:
        """Remove references to doc_id from node; delete node_docs link; refresh denormalized meta."""
        got = self.node_collection.get(ids=[node_id], include=["documents", 'metadatas'])
        docs = got.get("documents")
        if not (docs and docs[0]):
            return False
        node = Node.model_validate_json(docs[0])
        before = len(node.mentions or [])
        # node.mentions = [r for r in (node.mentions or []) if r.doc_id != doc_id 
        #                    or   (bool(getattr(r, "document_page_url", None) 
        #                          and (getattr(r, "document_page_url").rsplit('/',1)[-1] != doc_id)))]
        for groundings in node.mentions:
            filtered_spans: list[Span] = [span for span in groundings.spans if span.doc_id != doc_id]
            groundings.spans = filtered_spans
            
                
        changed = len(node.mentions or []) != before
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
        got = self.edge_collection.get()
        total = 0
        for eid in (got.get("ids") or []):
            edges = self.edge_collection.get(ids=[eid], include=["documents"])
            if edge_docs:=edges.get("documents"):
                e = Edge.model_validate_json(edge_docs[0])
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
            got = self.node_collection.get(where={"doc_id": doc_id}, include=["documents"])
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
        got = self.node_collection.get()
        total = 0
        for nid in (got.get("ids") or []):
            doc = self.node_collection.get(ids=[nid], include=["documents"])
            if nod_docs := doc.get("documents"):
                n = Node.model_validate_json(nod_docs[0])
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
        self.edge_endpoints_collection.delete(where=cast(dict[str, Any], {"edge_id": {"$in": edge_ids}}))
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

    def persist_graph(self,
        *,
        parsed: PureGraph, 
        session_id: str,
        mode = None):
        self._preflight_validate(parsed, alias_key=session_id)
        node_ids, edge_ids = [], []
        _ = _alloc_real_ids(parsed)
        order, id2kind, id2obj = self._build_deps(parsed)
        nl = '\n'
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            # for ln in parsed.nodes:
            if kind == 'node':
                ln: Node = obj
                
                # skip-if-exists mode
                if mode == "skip-if-exists":
                    got = self.node_collection.get(ids=[ln.id])
                    if got.get("ids"):  # already there
                        node_ids.append(ln.id)
                        continue
                
                n = PureChromaNode(
                    id=ln.id, label=ln.label, type=ln.type, summary=ln.summary,
                    domain_id=ln.domain_id, canonical_entity_id=ln.canonical_entity_id,
                    properties=ln.properties,
                    doc_id = None,
                    embedding = None,
                    metadata = {}
                )
                emb_text = f"{n.label}: {n.summary}"
                if emb_text is None:
                    emb_text = f"{n.label}: {n.summary}"
                n.embedding = self._ef([emb_text])[0]
                self.add_pure_node(n)
                node_ids.append(n.id)
            elif kind == 'edge':
            # for le in parsed.edges:
                le: Edge = obj
                
                if mode == "skip-if-exists":
                    got = self.edge_collection.get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                e = PureChromaEdge(
                    id=le.id, label=le.label, type=le.type, summary=le.summary,
                    domain_id=le.domain_id, canonical_entity_id=le.canonical_entity_id,
                    properties=le.properties,
                    relation=le.relation,
                    source_ids=le.source_ids, target_ids=le.target_ids,
                    source_edge_ids=getattr(le, "source_edge_ids", None),
                    target_edge_ids=getattr(le, "target_edge_ids", None),
                    doc_id=None,
                    embedding = None,
                    metadata = {},
                    # embedding=self._ef([f"{le.label}: {le.summary}"])[0]
                )
                e.embedding = self._ef([f"{le.label}: {le.summary}"])[0]
                self.add_pure_edge(e)
                edge_ids.append(e.id)

        return {
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "nodes_added": len(node_ids),
            "edges_added": len(edge_ids),
        }

    def persist_graph_extraction(
        self,
        *,
        document: Document,
        parsed: LLMGraphExtraction,
        mode: str = "append",   # "replace" | "append" | "skip-if-exists"
        assign_real_id_in_place = True
    ) -> dict:
        """
        the external user use case is to send the whole extraction via mcp as a copy and therefore
        by default no deep copy is made.
        Write nodes/edges/endpoints for `document.id`.
        Returns concrete ids written for idempotent tests.
        node that side effect parsed is modified in place for efficiency 
        if assign_real_id_in_place is False, a parsed deep copy is made
        """
        doc_id = document.id
        if not assign_real_id_in_place:
            parsed = parsed.model_copy(deep=True)
        # if replace, rollback prior doc content first
        if mode == "replace":
            self.rollback_document(doc_id)

        # ensure doc row exists (idempotent)
        self.add_document(document)
        # now validate (ensures no dangling refs to *other* docs)
        self._preflight_validate(parsed, doc_id)
        # self.ingest_with_toposort(parsed, doc_id = doc_id)

        # diagnose code: set([j for i in parsed.edges for j in i.target_ids]).union(set([j for i in parsed.edges for j in i.source_ids])) <= set([i.id for i in parsed.edges]).union(set([i.id for i in parsed.nodes]))

        # persist and collect ids
        node_ids, edge_ids = [], []
        # fallback_snip = (document.content[:160] + "…") if document.content else None
        # _alloc_real_ids duplicated logic in Self::_resolve_llm_ids        
        _ = _alloc_real_ids(parsed)  # rewrite in place
        order, id2kind, id2obj = self._build_deps(parsed)
        span_validator: BaseDocValidator = self.get_span_validator_of_doc_type(document = document)
        nodes_added = edges_added = 0
        nl = '\n'
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            # for ln in parsed.nodes:
            if kind == 'node':
                ln: Node = Node.model_validate(obj.model_dump(), context={'insertion_method': 'llm_graph_extraction'})
                ln.mentions = self._dealias_span(ln.mentions, document.id)
                # skip-if-exists mode
                if mode == "skip-if-exists":
                    got = self.node_collection.get(ids=[ln.id])
                    if got.get("ids"):  # already there
                        node_ids.append(ln.id)
                        continue

                for g in ln.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(doc_id = doc_id, span = sp, engine = self, doc=document)
                        if result['correctness'] != True:
                            raise Exception(f"Incorrect span occur in grounding {str(g)} span {str(sp)}")
                n = ln.model_copy(deep=True)
                emb_text = f"{n.label}: {n.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(ln)[:1])}"
                if emb_text is None:
                    emb_text = f"{n.label}: {n.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(ln)[:1])}"
                n.embedding = self._ef([emb_text])[0]
                self.add_node(n, doc_id=doc_id)
                node_ids.append(n.id)
            elif kind == 'edge':
            # for le in parsed.edges:
                le: Edge = Edge.model_validate(obj.model_dump(), context={'insertion_method': 'llm_graph_extraction'})
                le.mentions = self._dealias_span(le.mentions, document.id)
                if mode == "skip-if-exists":
                    got = self.edge_collection.get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                # refs = [Span.model_validate(i.model_dump(field_mode = 'backend'), context={'insertion_method': i.insertion_method or 'llm_graph_extraction'}) for i in le.mentions]
                
                for g in le.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(doc_id = doc_id, span = sp, engine = self, doc=document)
                        if result['correctness'] != True:
                            raise Exception(f"Incorrect span occur in grounding {str(g)} span {str(sp)}")
                e = le.model_copy(deep=True)
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
    def extract_graph_with_llm(self, *, content: str, doc_type: str, alias_nodes_str = "[Empty]" , alias_edges_str = "[Empty]", with_parsed = True, 
                               instruction_for_node_edge_contents_parsing_inclusion: None| str = None, validate = True, autofix : bool | str= True,
                               last_iteration_result = None):
        """Pure: run LLM + parse + alias resolution. No writes.
        last_iteration_result dict with 3 fields of 'raw' 'parsed' and 'error'. Falsy on initialization
        """
        # (reuse your existing prompt + alias path)
        raw, parsed, error = self._extract_graph_with_llm_aliases(
            content, alias_nodes_str=alias_nodes_str, alias_edges_str=alias_edges_str,
            instruction_for_node_edge_contents_parsing_inclusion = instruction_for_node_edge_contents_parsing_inclusion,
            last_iteration_result = last_iteration_result
        )
        if error:
            raise ValueError(error)
        validation_error_group = []
        if validate:
            # prevent the case LLM hallucinated real UUID in the response that is not any existing node, internal structure intact invariant
            
            temp_alias_book = AliasBook()
            parsed_copy = parsed.model_copy(deep= True)
            # with open (os.path.join("manual_cache", "temp.json"), 'w') as f:
            #     f.write(parsed_copy.model_dump_json())
            # with open (os.path.join("manual_cache", "temp.json"), 'r') as f:
            #     import json
            #     data = json.load(f)
            #     parsed_copy.model_validate(data)
            self._preflight_validate(parsed_copy, "", alias_book = temp_alias_book)
            
            if not (set([j for i in parsed_copy.edges for j in i.target_ids]).union(
                set([j for i in parsed_copy.edges for j in i.source_ids])) <= set([i.id for i in parsed_copy.edges]).union(
                    set([i.id for i in parsed_copy.nodes]))):
                raise Exception("LLM error, new uuid hallucinated")
            span_validator: BaseDocValidator = self.get_span_validator_of_doc_type(doc_type = doc_type)
            dummy_doc = Document(content=content,
                   type=doc_type, metadata={}, domain_id = None, processed = False, embeddings = None, source_map = None)
            # validation_error_group = []
            pre_parse_nodes_or_edges: list [Node | Edge] = parsed.nodes + parsed.edges
            for i, node_or_edge in enumerate(parsed_copy.nodes + parsed_copy.edges):
                node_or_edge : Node | Edge
                for g in node_or_edge.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(doc = dummy_doc, span = sp )
                        if result['correctness'] == True:
                            pass
                        else:
                            if autofix:
                                if autofix == True:
                                    fix_result = span_validator.fix_span(doc = dummy_doc, span = sp, nodes_edges = parsed_copy.nodes + parsed_copy.edges)
                                    result = fix_result
                                else:
                                    raise NotImplementedError("string method options not iplemented")
                            if result['correctness'] == False:
                                pre_parsed_node_or_edge: Node | Edge = pre_parse_nodes_or_edges[i]
                                validation_error_group.append(f"Error found for {pre_parsed_node_or_edge.model_dump()}: {str(result)}")
                            
        # resolve nn:/ne:/aliases -> UUIDs here
        # and run self._preflight_validate(parsed, doc_id) LATER (we don’t know doc_id yet)
        if with_parsed:
            return {"raw": raw, "parsed": parsed, "error": validation_error_group or None}
        else:
            return {"raw": raw, "error": validation_error_group or None}
        
    def get_document(self, doc_id: str):

        doc_get_result = self.document_collection.get(doc_id)
        if len(doc_get_result['ids']) == 0:
            raise ValueError(f"no document found for doc id = {doc_id}")
        metadatas = doc_get_result["metadatas"]
        
        
        docs = doc_get_result["documents"]
        
        if docs is None or metadatas is None:
            raise ValueError('Invalid documnet metadata')
        metadata : dict= cast(dict, metadatas[0])
        
        
        doc = Document(id = doc_get_result['ids'][0],
                       content = docs[0],
                       metadata = metadata,
                       domain_id = _str_or_none(metadata.get('domain_id')),
                       type = metadata['type'],
                       processed = metadata['processed'],
                       embeddings = None,
                       source_map = None
                       )
        return doc
    from typing import overload, Literal, Any


    def get_span_validator_of_doc_type(self, *, doc_id: str| None= None, 
                                       doc_type: str | None = None, 
                                       document: Document| None=None) -> BaseDocValidator:
        """infer doc type from either doc_id, type_type or document and return corresponding span validator"""
        if (doc_id is not None) + (doc_type is not  None) + (document is not None) == 1:
            pass
        else:
            raise ValueError("Must only specify one of doc_id, doc_type or document")
        from graph_knowledge_engine.extraction import PlainTextDocSpanValidator, OcrDocSpanValidator
        if doc_type is None:
            if doc_id is not None:
                document = self.get_document(doc_id)
            if document:
                doc_type = document.type
            else:
                raise ValueError("Unreachable")
                
        
        if doc_type == 'text':
            return PlainTextDocSpanValidator()
        if doc_type == "ocr_document":
            return OcrDocSpanValidator()
        raise ValueError(f"No validator associated with document type {doc_type}")
    def persist_document_graph_extraction(
        self,
        *,
        doc_id,
        parsed: GraphExtractionWithIDs|LLMGraphExtraction,
        mode: str = "append",   # "replace" | "append" | "skip-if-exists"
    ) -> dict:
        """
        Write nodes/edges/endpoints for `document.id`.
        Returns concrete ids written for idempotent tests.
        """

        # now validate (ensures no dangling refs to *other* docs)
        self._preflight_validate(parsed, doc_id)
        # self.ingest_with_toposort(parsed, doc_id = doc_id)

        # persist and collect ids
        node_ids, edge_ids = [], []
        _ = _alloc_real_ids(parsed)  # rewrite in place
        order, id2kind, id2obj = self._build_deps(parsed)
        
        span_validator: BaseDocValidator = self.get_span_validator_of_doc_type(doc_id=doc_id)
        nodes_added = edges_added = 0
        nl = '\n'
        for rid in order:
            kind, obj = id2kind[rid], id2obj[rid]
            # for ln in parsed.nodes:
            if kind == 'node':
                ln: Node = obj
                ln.mentions = self._dealias_span(ln.mentions, doc_id)
                # skip-if-exists mode
                for g in ln.mentions:
                    for sp in g.spans:
                        span_validator.validate_span(doc_id = doc_id, span = sp)
                if mode == "skip-if-exists":
                    got = self.node_collection.get(ids=[ln.id])
                    if got.get("ids"):  # already there
                        node_ids.append(ln.id)
                        continue
                
                        
                n = ln

                emb_text = f"{n.label}: {n.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(ln)[:1])}"
                n.embedding = self._ef([emb_text])[0]
                self.add_node(n, doc_id=doc_id)
                node_ids.append(n.id)
            elif kind == 'edge':
            # for le in parsed.edges:
                le: Edge = obj
                le.mentions = self._dealias_span(le.mentions, doc_id)
                for g in le.mentions:
                    for sp in g.spans:
                        span_validator.validate_span(doc_id = doc_id, span = sp)
                if mode == "skip-if-exists":
                    got = self.edge_collection.get(ids=[le.id])
                    if got.get("ids"):
                        edge_ids.append(le.id)
                        continue
                
                e = le

                emb_text = f"{le.label}: {le.summary} : {nl.join(i['context'] for i in self.extract_reference_contexts(le)[:1])}"
                e.embedding = self._ef([emb_text])[0]
                self.add_edge(e, doc_id=doc_id)
                edge_ids.append(e.id)

        return {
            "document_id": doc_id,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "nodes_added": len(node_ids),
            "edges_added": len(edge_ids),
        }

    def rollback_document_extraction(
        self,
        doc_id: str,
        extraction_method: Literal['llm_graph_extraction', 'document_ingestion'],
    ) -> dict:
        """
        Remove all references contributed by (doc_id, extraction_method).
        - Keeps entities that still have refs from other docs/methods; re-saves them.
        - Deletes entities that lose all refs.
        - Cleans (node_id, doc_id) rows and edge_endpoints rows for this doc.
        - Does NOT try to infer cascading deletes beyond the entity itself.

        Returns summary counts.
        """
        from .models import Node, Edge, Span  # adjust import if needed
        import json

        summary = {
            "doc_id": doc_id,
            "method": extraction_method,
            "updated_nodes": 0,
            "updated_edges": 0,
            "deleted_nodes": 0,
            "deleted_edges": 0,
            "deleted_node_refs": 0,
            "deleted_edge_refs": 0,
            "deleted_node_doc_rows": 0,
            "deleted_edge_endpoints": 0,
        }

        # ---- helpers -------------------------------------------------------------
        def _load_many(col, ids):
            if not ids:
                return {}
            got = col.get(ids=list(ids), include=["documents"])
            docs = got.get("documents") or []
            ids_out = got.get("ids") or []
            out = {}
            for i, mj in enumerate(docs):
                try:
                    d = json.loads(mj)
                except Exception:
                    try:
                        d = (Node if col is self.node_collection else Edge).model_validate_json(mj).model_dump()
                    except Exception:
                        d = None
                if d is not None and i < len(ids_out):
                    out[ids_out[i]] = d
            return out

        def _save_node(d: dict):
            # keep existing metadata (doc_id, etc.) and update document JSON
            nid = d["id"]
            prior = self.node_collection.get(ids=[nid], include=["metadatas"])
            meta = (prior.get("metadatas") or [None])[0] or {}
            meta = dict(meta)  # shallow copy
            # write
            self.node_collection.update(
                ids=[nid],
                documents=[json.dumps(d, ensure_ascii=False)],
                metadatas=[meta],
            )
            # re-index node_docs for this node
            try:
                self._index_node_docs(Node.model_validate(d))
            except Exception:
                pass

        def _save_edge(d: dict):
            eid = d["id"]
            prior = self.edge_collection.get(ids=[eid], include=["metadatas"])
            meta = (prior.get("metadatas") or [None])[0] or {}
            meta = dict(meta)
            self.edge_collection.update(
                ids=[eid],
                documents=[json.dumps(d, ensure_ascii=False)],
                metadatas=[meta],
            )

        # ---- 1) candidate ids for this doc --------------------------------------
        # nodes (prefer node_docs index; fallback to scanning metadatas)
        node_ids = set()
        try:
            # node_docs rows have {"node_id": ..., "doc_id": ...}
            nd = self.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
            for m in (nd.get("metadatas") or []):
                if m and m.get("node_id"):
                    node_ids.add(m["node_id"])
            summary["deleted_node_doc_rows"] = len(nd.get("ids") or [])
        except Exception:
            # fallback: query node collection by doc_id metadata if present
            try:
                q = self.node_collection.get(where={"doc_id": doc_id})
                for nid in (q.get("ids") or []):
                    node_ids.add(nid)
            except Exception:
                pass

        # edges: derive by edge_endpoints rows filtered by doc_id
        edge_ids = set()
        try:
            ee = self.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
            for m in (ee.get("metadatas") or []):
                if m and m.get("edge_id"):
                    edge_ids.add(m["edge_id"])
            summary["deleted_edge_endpoints"] = len(ee.get("ids") or [])
        except Exception:
            # fallback: direct edge_collection metadata
            try:
                q = self.edge_collection.get(where={"doc_id": doc_id})
                for eid in (q.get("ids") or []):
                    edge_ids.add(eid)
            except Exception:
                pass

        # ---- 2) load, filter references, and persist or delete -------------------
        # Nodes
        nodes_map = _load_many(self.node_collection, node_ids)
        for nid, d in nodes_map.items():
            refs = d.get("references") or []
            keep = []
            removed = 0
            for r in refs:
                if r and (r.get("doc_id") == doc_id) and (r.get("insertion_method") == extraction_method):
                    removed += 1
                else:
                    keep.append(r)
            if removed:
                summary["deleted_node_refs"] += removed
                if keep:
                    d["references"] = keep
                    _save_node(d)
                    summary["updated_nodes"] += 1
                else:
                    # no refs left → delete node, node_docs rows, and (optionally) any edges’ endpoints for this doc if tied
                    try:
                        self.node_collection.delete(ids=[nid])
                    except Exception:
                        pass
                    try:
                        self.node_docs_collection.delete(where={"node_id": nid, "doc_id": doc_id})
                    except Exception:
                        pass
                    summary["deleted_nodes"] += 1
            else:
                # Still remove node_docs rows for this doc (this document’s contribution) if present
                try:
                    self.node_docs_collection.delete(where={"node_id": nid, "doc_id": doc_id})
                except Exception:
                    pass

        # Edges
        edges_map = _load_many(self.edge_collection, edge_ids)
        for eid, d in edges_map.items():
            refs = d.get("references") or []
            keep = []
            removed = 0
            for r in refs:
                if r and (r.get("doc_id") == doc_id) and (r.get("insertion_method") == extraction_method):
                    removed += 1
                else:
                    keep.append(r)

            # Always drop this doc’s endpoints rows; they are doc-scoped fanout rows
            try:
                self.edge_endpoints_collection.delete(where={"edge_id": eid, "doc_id": doc_id})
            except Exception:
                pass

            if removed:
                summary["deleted_edge_refs"] += removed
                if keep:
                    d["references"] = keep
                    _save_edge(d)
                    summary["updated_edges"] += 1
                else:
                    # No refs remain → delete edge entirely (and any leftover endpoints just in case)
                    try:
                        self.edge_collection.delete(ids=[eid])
                    except Exception:
                        pass
                    try:
                        self.edge_endpoints_collection.delete(where={"edge_id": eid})
                    except Exception:
                        pass
                    summary["deleted_edges"] += 1

        # ---- 3) final: delete node_docs and endpoints rows for this doc ----------
        # (Safe to repeat; collections ignore missing)
        try:
            self.node_docs_collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass
        try:
            self.edge_endpoints_collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass

        return summary         
    def ingest_document_with_llm(self, document: Document, *, mode: str="append",
                                 instruction_for_node_edge_contents_parsing_inclusion=None,
                                 raw_with_parsed = None):
        """Convenience: extract + persist. Still returns concrete ids written."""
        if raw_with_parsed is None:
            raw_with_parsed = {}
        # add doc row now so fallback refs have URLs
        self.add_document(document)

        # build context & aliases as you already do, then:
        extracted = self.extract_graph_with_llm(content=str(document.content),
                                doc_type=document.type,
                                instruction_for_node_edge_contents_parsing_inclusion=instruction_for_node_edge_contents_parsing_inclusion,
                                last_iteration_result=raw_with_parsed)
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
        parsed: LLMGraphExtraction = result.get("parsed") if isinstance(result, dict) else result # type: ignore
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
                verdicts, _ = self.batch_adjudicate_merges(pairs) # type: ignore verdict never falsy if pairs is truey
                verdicts: list[AdjudicationVerdict]
                for (left, right), out in zip(pairs, verdicts):
                    verdict: AdjudicationVerdict = getattr(out, "verdict", out)
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
        if eps_doc:= eps["documents"]:
            pass
        else:
            raise Exception("Document loss")
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps_doc})
        edges = self.edge_collection.get(ids=edge_ids, include=[ "documents", "metadatas"])

        removed_edge_ids: set[str] = set()
        updated_edge_ids: set[str] = set()

        for eid, edoc, meta in zip(edges.get("ids") or [], edges.get("documents") or [], edges.get("metadatas") or []):
            e = Edge.model_validate_json(edoc)
            relation = (meta or {}).get("relation") or e.relation  # be resilient to missing meta

            if relation == "same_as":
                # Special rebalancing for equality hyperedges
                new_edge: Edge | None
                edge_deleted, new_edge = self._rebalance_same_as_edge(e, removed_node_id=node_id)
                if edge_deleted or (new_edge is None):
                    self.edge_collection.delete(ids=[eid])
                    self.edge_endpoints_collection.delete(where={"edge_id": eid})
                    removed_edge_ids.add(eid)
                else:
                    # Update the edge
                     # help pylance
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
                            "references": _json_or_none([ref.model_dump() for ref in (new_edge.mentions or [])]),
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
                            if node_doc is None:
                                raise Exception(f"node_doc for {nid} is lost")
                            per_doc_id = None
                            if node_doc_doc:= node_doc.get("documents"):
                                try:
                                    n = Node.model_validate_json(node_doc_doc[0])
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
                        self.edge_endpoints_collection.add(ids=ep_ids, documents=ep_docs, metadatas=ep_metas, 
                                                           embeddings=[self._iterative_defensive_emb(d) for d in ep_docs])
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
                        "references": _json_or_none([ref.model_dump() for ref in (e.mentions or [])]),
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
        eps_doc = eps.get("documents", [])
        if eps_doc is None:
            raise Exception(f"edge endpoint collection lost for document id {document_id}")
        edge_ids = list({json.loads(doc)["edge_id"] for doc in eps_doc})
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
            if docs := got.get("documents"):
                if docs[0]:
                # n = Node.model_validate_json(got["documents"][0])
                    if not json.loads(docs[0]).get('references') : # none or empty list
                        self.node_collection.delete(ids=[nid])
                        deleted_node_ids.append(nid)
                    self.node_refs_collection.delete(where=cast(dict[str, Any], {"node_id": {"$in": node_ids}}))
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
            if docs := got.get("documents"): 
                return Node.model_validate_json(docs[0])
            else:
                raise ValueError(f"Node {t.id} not found")
        else: # edge
            got = self.edge_collection.get(ids=[t.id], include=["documents"])
            if docs:=got.get("documents"): 
                return Edge.model_validate_json(docs[0])
            else:
                raise ValueError(f"Edge {t.id} not found")
            

    def commit_merge_target(self, left: AdjudicationTarget, right: AdjudicationTarget, verdict: AdjudicationVerdict) -> str:
        canonical_id = self.merge_policy.commit_merge_target(left,right,verdict)
        return canonical_id

    
    def add_edge_with_endpoint_docs(self, edge: Edge, endpoint_doc_ids: dict[str, str | None]):
        # Add the main edge row (neutral doc_id)
        doc = edge.model_dump_json(field_mode='backend')
        self.edge_collection.add(
            ids=[edge.id],
            documents=[],
            embeddings=[edge.embedding] if edge.embedding else [self._iterative_defensive_emb(doc)],
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
                "references": json.dumps([ref.model_dump() for ref in (edge.mentions or [])]),
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
                embeddings=[self._iterative_defensive_emb(d) for d in ep_docs]
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
    def _split_endpoints(self, src_ids: list[str] | None, tgt_ids: list[str] | None)-> tuple[list[Any], list[Any], list[Any], list[Any]]:# -> tuple[list[Any], list[Any], list[Any], list[Any]]:
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
        return self.merge_policy.commit_any_kind(node_or_edge_l, node_or_edge_r,
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
                                                     anchor_only=anchor_only,
                                                     new_edge = [])
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
        idx: CollectionLike | None
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
            where: dict[str, Any] = {"insertion_method": insertion_method}
            if doc_id:
                where["doc_id"] = doc_id
            if ids:
                where[key]= {"$in": list(ids)}
            rows = idx.get(where=where, include=["metadatas"])
            picked = {str(m.get(key)) for m in (rows.get("metadatas") or []) if m and m.get(key)}
            return sorted(picked)

        # ---------- Fallback: scan primary JSON ----------
        # (only used if you didn’t create the index collection)
        if ids:
            got = primary.get(ids=list(ids), include=["documents"])
            documents = got.get("documents") or []
            entity_ids = got.get("ids") or []
        else:
            got = primary.get(include=["documents"])
            documents = got.get("documents") or []
            entity_ids = got.get("ids") or []

        keep: set[str] = set()
        for eid, blob in zip(entity_ids, documents):
            ent = model_cls.model_validate_json(blob)
            for ref in (ent.mentions or []):
                im = getattr(ref, "insertion_method", None)
                if im == insertion_method and (not doc_id or _ref_doc_id(ref) == doc_id):
                    keep.add(eid)
                    break
        return sorted(keep)
    def _verify_one_reference(
        self,
        extracted_text: str,
        full_text: str,
        ref: Span,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        threshold: float = 0.70,
    ) -> Span:
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

    # ----------------------------
    # Conversation Abstraction
    # ----------------------------
    @conversation_only
    def create_conversation(self, user_id, conv_id = None, node_id: str | None | uuid.UUID = None) -> tuple[str, str]:
        from .conversation_orchestrator import get_id_for_conversation_turn
        if self.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        """Create a new conversation thread ID and reserve it with a start node."""
        new_index = -1
        conv_id = conv_id or str(uuid.uuid4())
        node_id = node_id  or get_id_for_conversation_turn(ConversationNode.id_kind, user_id, 
                                            conv_id, "Start of conversation", str(new_index), "system", "conversation_summary", in_conv=True)
        # Create a start node to reserve the ID and anchor the conversation
        start_node = ConversationNode(
            id=str(node_id),
            user_id = user_id,
            label="conversation start",
            type="entity",
            summary="Start of conversation",
            role="system",
            turn_index=-1,
            conversation_id=conv_id,
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])], # No mentions for start node
            properties={"status": "active"},
            metadata={"level_from_root": 0, 
                      "entity_type": "conversation_start", 
                      "turn_index": -1,
                      "char_distance_from_last_summary": 0, 
                      "turn_distance_from_last_summary" : -1,
                      "in_conversation_chain": True},
            domain_id=None,
            canonical_entity_id=None,
            level_from_root = 0,
            doc_id = None,
            embedding = None
            
        )
        
        # We need to bypass the validator that might complain about empty mentions for GraphEntityRefBase
        # Ideally ConversationNode should relax this, but if not, we provide a dummy or handle it.
        # Since GraphEntityRefBase has a validator _require_non_empty_refs, we might need a dummy mention.
        # Let's create a dummy system span.
        dummy_span = Span.from_dummy_for_conversation()
        start_node.mentions = [Grounding(spans=[dummy_span])]
        self.add_node(start_node)
        
        return conv_id, node_id
    
    @conversation_only
    def _get_last_seq_node(self, conversation_id, min_seq = None):
        if min_seq is None:
            min_seq = self.max_node_seq_present(conversation_id)
        if self.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        got = self.node_collection.get(
            where={"$and":[
                {"conversation_id": conversation_id},                 
                ]
                + [{"seq": {"$gte": min_seq or 0}}]
            },
            include=["documents", "metadatas", "embeddings"]
        )
        if not got["ids"]:
            return None
        
        # reconstruct
        nodes: list[ConversationNode] = self.nodes_from_single_or_id_query_result(got, node_type=ConversationNode)
        # nodes2 = list(filter(lambda x: x.metadata.get("entity_type") in tail_search_includes, nodes))
        # if not nodes2:
        #     return None
        
        # # Sort by turn_index
        nodes.sort(key=lambda n: n.metadata.get("seq") or -1)
        return nodes[-1]
    @conversation_only
    def _get_conversation_tail(self, conversation_id: str, 
                               min_turn_index : int | None = None,
                               tail_search_includes: list[str] = ["conversation_start", "conversation_turn","conversation_summary", "assistant_turn"]) -> Optional[ConversationNode]:
        """Find the last node in the conversation (leaf of 'next_turn')."""
        # Simplistic: query all nodes for this conv, sort by turn_index desc
        # Optimization: Store tail ID in a separate 'conversations' metadata collection if needed.
        # For now, we scan.
        if self.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        got = self.node_collection.get(
            where={"$and":[
                {"conversation_id": conversation_id}, 
                {"in_conversation_chain": True}
                ]
                   + ([{"turn_index": {"$gte": min_turn_index}}]
                      if min_turn_index else [])
                      },
            include=["documents", "metadatas", "embeddings"]
        )
        if not got["ids"]:
            return None
        
        # reconstruct
        nodes: list[ConversationNode] = self.nodes_from_single_or_id_query_result(got, node_type=ConversationNode)
        nodes2 = list(filter(lambda x: x.metadata.get("entity_type") in tail_search_includes, nodes))
        if not nodes2:
            return None
        
        # Sort by turn_index
        nodes2.sort(key=lambda n: n.turn_index or -1)
        return nodes2[-1]
    def _iterative_defensive_emb(self, emb_text0):
        success = False
        
        idx = self.embedding_length_limit
        embedding = None
        cnt = 0
        while not success:
            cnt += 1
            if cnt >= 10:
                break
            emb_text = emb_text0[:idx] + ("..." if idx < len(emb_text0)-1 else "")
            try:
                embedding = self._ef([emb_text])[0]
                success = True
                break
            except:
                idx //= 2
        while success:
            cnt += 1
            if cnt >= 13:
                break
            emb_text = emb_text0[:idx] + ("..." if idx < len(emb_text0)-1 else "")
            try:
                embedding = self._ef([emb_text])[0]
                if idx >= len(emb_text0):
                    break
                idx = int(idx * 1.6)
            except:
                success = False
        if embedding is None:
            raise Exception("cannot get embedding after most defensive embedding strategy.")
        return embedding
    @conversation_only
    def last_summary_of_node(engine, node: ConversationNode):
            return engine.get_nodes(where = {"$and" : [
                    {"conversation_id": node.conversation_id} , 
                    {"turn_index": node.turn_index - node.metadata['turn_distance_from_last_summary']}]}, node_type = ConversationNode)
    @conversation_only
    def add_conversation_turn(self, user_id: str, conversation_id: str, turn_id: str, mem_id: 
                            str, role: str, content: str, 
                            ref_knowledge_engine: GraphKnowledgeEngine, 
                            filtering_callback: Callable[..., tuple[FilteringResult | RetrievalResult, str]] = candiate_filtering_callback,
                            max_retrieval_level: int = 2, summary_char_threshold = 12000,
                            prev_turn_meta_summary : MetaFromLastSummary = MetaFromLastSummary(0,0), 
                            add_turn_only = None) -> AddTurnResult:
        """Stable facade: delegate to the KGE-native conversation orchestrator.

        The orchestration policy (retrieve/filter/pin/answer/summarize) lives outside engine.py.
        The engine keeps storage/mutation primitives and a stable public API.
        """
        orch = self._get_orchestrator(ref_knowledge_engine=ref_knowledge_engine)
        orch.tool_runner.tool_call_id_factory
        return orch.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            mem_id=mem_id,
            role=role,
            content=content,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
            summary_char_threshold=summary_char_threshold,
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only = add_turn_only
        )

    # -----------------
    # Orchestrator hook
    # -----------------
    def _get_orchestrator(self, *, ref_knowledge_engine: "GraphKnowledgeEngine"):
        """Lazily construct an orchestrator.

        NOTE: we key the cache by (id(ref_knowledge_engine), id(self.llm)) so callers can swap
        knowledge engines or LLMs in tests without global side effects.
        """
        key = (id(ref_knowledge_engine), id(self.llm))
        cache = getattr(self, "_orch_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_orch_cache", cache)
        if key in cache:
            return cache[key]

        from .conversation_orchestrator import ConversationOrchestrator

        orch = ConversationOrchestrator(
            conversation_engine=self,
            ref_knowledge_engine=ref_knowledge_engine,
            llm=self.llm,
        )
        cache[key] = orch
        return orch
    @conversation_only    
    def get_conversation(self, conversation_id):
        pass
    @conversation_only
    def get_system_prompt(self, conversation_id: str) -> str:
        return "You are a helpful assistant. Answer the user using the conversation and any provided evidence."
    @conversation_only    
    def get_response_model(self, conversation_id) -> Type[BaseModel]:

        return ConversationAIResponse
    

    @conversation_only
    def get_conversation_view(
        self,
        *,
        conversation_id: str,
        user_id: str | None = None,
        purpose: str = "answer",
        budget_tokens: int = 6000,
        tail_turns: int = 8,
        include_summaries: bool = True,
        include_memory_context: bool = True,
        include_pinned_kg_refs: bool = True,
    ):
        

        tokenizer = ApproxTokenizer()

        sources = ContextSources(
            conversation_engine=self,
            tail_turns=tail_turns,
            include_summaries=include_summaries,
            include_memory_context=include_memory_context,
            include_pinned_kg_refs=include_pinned_kg_refs,
        )
        items: list[ContextItem] = sources.gather(conversation_id=conversation_id, purpose=purpose)

        # Add system prompt as pinned item
        sys = self.get_system_prompt(conversation_id)
        items.insert(
            0,
            ContextItem(
                role="system",
                kind="system_prompt",
                text=str(sys or ""),
                node_id=None,
                priority=0,
                pinned=True,
                max_tokens=900,
                source="system",
            ),
        )

        # Price items
        priced: list[ContextItem] = []
        for it in items:
            cost = tokenizer.count_tokens(it.text or "")
            priced.append(ContextItem(**{**it.__dict__, "token_cost": cost}))

        # Packing policy
        pinned_non_turn = [i for i in priced if i.pinned and i.kind != "tail_turn"]
        tail_turn_items = [i for i in priced if i.kind == "tail_turn"]

        pinned_non_turn.sort(key=lambda x: x.priority)
        tail_turn_items.sort(key=lambda x: x.priority)  # newest first (lowest priority)

        kept: list[ContextItem] = []
        dropped: list[DroppedItem] = []
        used = 0

        def _try_add(it: ContextItem) -> bool:
            nonlocal used
            if used + it.token_cost <= budget_tokens:
                kept.append(it)
                used += it.token_cost
                return True

            # compress if allowed
            if it.max_tokens is not None and it.max_tokens < it.token_cost:
                new_text = it.text[: max(1, it.max_tokens * 4)]  # cheap truncation placeholder
                new_cost = tokenizer.count_tokens(new_text)
                if used + new_cost <= budget_tokens:
                    kept.append(ContextItem(**{**it.__dict__, "text": new_text, "token_cost": new_cost}))
                    used += new_cost
                    dropped.append(DroppedItem(kind=it.kind, node_id=it.node_id, reason="compressed", token_cost=it.token_cost))
                    return True

            dropped.append(DroppedItem(kind=it.kind, node_id=it.node_id, reason="over_budget", token_cost=it.token_cost))
            return False

        for it in pinned_non_turn:
            _try_add(it)

        for it in tail_turn_items:
            _try_add(it)

        # Restore chronological order for turns
        non_turn_kept = [i for i in kept if i.kind != "tail_turn"]
        turn_kept: list[ContextItem] = [i for i in kept if i.kind == "tail_turn"]
        turn_kept.sort(key=lambda x: int((x.extra or {}).get("turn_index", 10**9)))
        kept = non_turn_kept + turn_kept

        renderer = ContextRenderer()
        messages = renderer.render(kept, purpose=purpose)

        included_node_ids = tuple(sorted({i.node_id for i in kept if i.node_id}))
        included_edge_ids = tuple(sorted({e for i in kept for e in (i.edge_ids or ())}))
        included_pointer_ids = tuple(sorted({p for i in kept for p in (i.pointer_ids or ()) if p}))

        head_summary_ids = tuple(i.node_id for i in kept if i.kind == "head_summary" and i.node_id)
        tail_turn_ids_out = tuple(i.node_id for i in kept if i.kind == "tail_turn" and i.node_id)
        active_memory_context_ids = tuple(i.node_id for i in kept if i.kind == "memory_context" and i.node_id)
        pinned_kg_ref_ids = tuple(i.node_id for i in kept if i.kind == "pinned_kg_ref" and i.node_id)

        view = ConversationContextView(
            conversation_id=conversation_id,
            purpose=purpose,
            messages=tuple(messages),
            token_budget=budget_tokens,
            tokens_used=used,
            items=tuple(kept),
            dropped=tuple(dropped),
            included_node_ids=included_node_ids,
            included_edge_ids=included_edge_ids,
            included_pointer_ids=included_pointer_ids,
            head_summary_ids=head_summary_ids,
            tail_turn_ids=tail_turn_ids_out,
            active_memory_context_ids=active_memory_context_ids,
            pinned_kg_ref_ids=pinned_kg_ref_ids,
        )
        # view.assert_valid()
        return view
    @conversation_only
    def get_ai_conversation_response(self, conversation_id, ref_knowledge_engine, model_names = None)->ConversationAIResponse:
        """Answer-only entry point.

        This stays as a stable public method, but delegates orchestration to the
        ConversationOrchestrator so policy/control-flow can evolve outside engine.py.
        """
        orch = self._get_orchestrator(ref_knowledge_engine=ref_knowledge_engine)
        return orch.answer_only(conversation_id=conversation_id, model_names=model_names)
    def get_llm(self, model_name = None) -> BaseChatModel:
        # will implement other model names later
        return self.llm
        
class ApproxTokenizer:
    def count_tokens(self, text: str) -> int:
        # very cheap approximation; stable for budget enforcement
        return max(1, len(text) // 4)
    
