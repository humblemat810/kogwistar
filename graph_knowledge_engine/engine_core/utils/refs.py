from __future__ import annotations

import json
import re
from typing import Optional, List, Any, TYPE_CHECKING

from ..models import Grounding, MentionVerification, Span
from .metadata import json_or_none, strip_none

if TYPE_CHECKING:
    from ..models import Edge, Node, PureChromaEdge, PureChromaNode

_DOC_URL = "document/{doc_id}"


def safe_excerpt(s: str | None, max_len: int = 200) -> str | None:
    if not s:
        return None
    s = s.strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "..."
    return s


def ref_doc_id(ref) -> str | None:
    did = getattr(ref, "doc_id", None)
    if did:
        return did
    url = getattr(ref, "document_page_url", None) or ""
    m = re.search(r"document/([A-Za-z0-9\-]+)", url)
    return m.group(1) if m else None


def ref_insertion_method(ref) -> str:
    m = getattr(ref, "insertion_method", None)
    if m:
        return str(m)
    ver = getattr(ref, "verification", None)
    if ver and getattr(ver, "method", None):
        return str(ver.method)
    return "unknown"


def default_verification(note: str = "fallback span") -> MentionVerification:
    return MentionVerification(method="heuristic", is_verified=False, score=None, notes=note)


def ensure_ref_span(ref: Span, doc_id: str) -> Span:
    r = ref.model_copy(deep=True)
    if not r.collection_page_url:
        r.collection_page_url = f"document_collection/{doc_id}"
    if not r.document_page_url or str(doc_id) not in r.document_page_url:
        r.document_page_url = _DOC_URL.format(doc_id=doc_id)
    if r.start_char is None or r.end_char is None:
        r.start_char, r.end_char = 0, 0
    if (not hasattr(r, "verification") and r.__class__.__name__.endswith("LlmSlice")):
        pass
    elif hasattr(r, "verification") and r.verification is None:
        r.verification = default_verification("no explicit verification from LLM")
    return r


def normalize_mentions(mentions: Optional[List[Span]], doc_id: str) -> List[Span]:
    if not mentions or len(mentions) == 0:
        raise Exception("missing mentions")
    return [ensure_ref_span(ref, doc_id) for ref in mentions]


def extract_doc_ids_from_refs(refs: list[Span] | list[Grounding]) -> list[str]:
    out = []
    for r in refs or []:
        if type(r) is Grounding:
            for span in r.spans:
                did = getattr(span, "doc_id", None)
                if not did and getattr(span, "document_page_url", None):
                    m = re.search(r"document/([A-Za-z0-9\-]+)", span.document_page_url)
                    if m:
                        did = m.group(1)
                if did:
                    out.append(did)
        elif type(r) is Span:
            did = getattr(r, "doc_id", None)
            if not did and getattr(r, "document_page_url", None):
                m = re.search(r"document/([A-Za-z0-9\-]+)", r.document_page_url)
                if m:
                    did = m.group(1)
            if did:
                out.append(did)
        else:
            raise ValueError(f"ref type of type {type(r)} is unsupported")
    return sorted(dict.fromkeys(out))


def select_best_grounding(entity: "Node | Edge") -> Grounding:
    mentions = list(getattr(entity, "mentions", None) or [])
    if mentions:
        def _sort_key(grounding: Grounding) -> tuple[int, int]:
            span = grounding.spans[0] if grounding.spans else None
            page = getattr(span, "page_number", None)
            if page is None:
                page = getattr(span, "start_page", None)
            start_char = getattr(span, "start_char", None) if span is not None else None
            return (
                int(page) if page is not None else 10**9,
                int(start_char) if start_char is not None else 10**9,
            )

        selected = min(mentions, key=_sort_key).model_copy(deep=True)
        for span in selected.spans:
            if span.verification is None:
                span.verification = MentionVerification(
                    method="heuristic",
                    is_verified=True,
                    score=0.5,
                    notes="adjudication evidence",
                )
                continue
            note = span.verification.notes or ""
            if "adjudication evidence" not in note:
                span.verification.notes = f"{note} | adjudication evidence" if note else "adjudication evidence"
        return selected

    doc_id = getattr(entity, "doc_id", None) or "unknown"
    excerpt = safe_excerpt(getattr(entity, "summary", None) or getattr(entity, "label", None) or "") or ""
    span = Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=_DOC_URL.format(doc_id=doc_id),
        doc_id=doc_id,
        insertion_method="heuristic",
        page_number=1,
        start_char=0,
        end_char=len(excerpt),
        excerpt=excerpt,
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=default_verification("adjudication evidence fallback"),
    )
    return Grounding(spans=[span])


def node_doc_and_meta(n: "Node | PureChromaNode") -> tuple[str, dict]:
    doc = n.model_dump_json(field_mode="backend", exclude=["embedding", "metadata"])
    meta = n.metadata
    meta.update(
        {
            "doc_id": getattr(n, "doc_id", None),
            "label": n.label,
            "type": n.type,
            "summary": n.summary,
            "domain_id": n.domain_id,
            "canonical_entity_id": getattr(n, "canonical_entity_id", None),
            "properties": json_or_none(getattr(n, "properties", None)),
        }
    )
    meta.update(n.get_extra_update())

    mentions = getattr(n, "mentions", None)
    if mentions is not None:
        meta["mentions"] = json_or_none([r.model_dump(field_mode="backend") for r in mentions])
    meta = strip_none(meta)
    return doc, meta


def edge_doc_and_meta(e: "Edge | PureChromaEdge") -> tuple[str, dict]:
    doc = e.model_dump_json(field_mode="backend")
    meta = strip_none(
        {
            "doc_id": getattr(e, "doc_id", None),
            "relation": e.relation,
            "source_ids": json_or_none(e.source_ids),
            "target_ids": json_or_none(e.target_ids),
            "type": e.type,
            "summary": e.summary,
            "domain_id": e.domain_id,
            "canonical_entity_id": getattr(e, "canonical_entity_id", None),
            "properties": json_or_none(getattr(e, "properties", None)),
        }
    )
    mentions = getattr(e, "mentions", None)
    if mentions is not None:
        meta["mentions"] = json_or_none([r.model_dump(field_mode="backend") for r in mentions])
    meta = strip_none(meta)
    return doc, meta


def merge_refs(old_refs_json: str | None, new_refs):
    old = []
    if old_refs_json:
        try:
            old = json.loads(old_refs_json)
        except Exception:
            old = []

    def key(r):
        return (
            r.get("document_page_url"),
            r.get("start_page"),
            r.get("start_char"),
            r.get("end_page"),
            r.get("end_char"),
        )

    seen = {key(r): r for r in old}
    for r in (new_refs or []):
        r2 = r.model_dump(field_mode="backend") if hasattr(r, "model_dump") else r
        seen[key(r2)] = r2
    merged = list(seen.values())
    return merged, json.dumps(merged)


def backend_update_record_lifecycle(*, backend: Any, kind: str, record_id: str, lifecycle_patch: dict, safe_json_dict_fn, merge_meta_fn) -> bool:
    get_fn = getattr(backend, f"{kind}_get", None)
    upd_fn = getattr(backend, f"{kind}_update", None)
    if get_fn is None or upd_fn is None:
        raise AttributeError(f"backend missing {kind}_get/{kind}_update")
    got = get_fn(ids=[record_id], include=["documents", "metadatas", "embeddings"])
    ids = got.get("ids") or []
    if not ids:
        return False

    doc = (got.get("documents") or [None])[0]
    meta = (got.get("metadatas") or [None])[0]
    emb = got.get("embeddings")
    embedding = (emb if emb is not None else [None])[0]
    base = safe_json_dict_fn(doc)

    base_meta = base.get("metadata") if isinstance(base.get("metadata"), dict) else {}
    base["metadata"] = merge_meta_fn(base_meta, lifecycle_patch)

    new_meta = merge_meta_fn(meta if isinstance(meta, dict) else {}, lifecycle_patch)
    upd_fn(
        ids=[record_id],
        documents=[json.dumps(base, ensure_ascii=False)],
        metadatas=[new_meta],
        embeddings=[embedding],
    )
    return True
