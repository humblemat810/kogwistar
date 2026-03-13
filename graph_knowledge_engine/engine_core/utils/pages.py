from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

PageLike = Union[str, Dict[str, Any]]
T = TypeVar("T")


def split_pages_from_text(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []

    if "\f" in raw:
        parts = raw.split("\f")
        return [
            {"page_number": i + 1, "text": p.strip()}
            for i, p in enumerate(parts)
            if p.strip()
        ]

    p = re.split(r"(?:^|\n)\s*Page[:\s]+(\d+)\s*(?:\n|$)", raw, flags=re.IGNORECASE)
    if len(p) > 1:
        out: list[dict[str, Any]] = []
        i = 0
        while i < len(p):
            if i == 0 and p[i].strip():
                out.append({"page_number": 1, "text": p[i].strip()})
                i += 1
                continue
            if i + 2 <= len(p) - 1:
                try:
                    num = int(p[i + 1])
                except Exception:
                    num = None
                txt = p[i + 2].strip()
                if txt:
                    out.append({"page_number": num or (len(out) + 1), "text": txt})
                i += 3
            else:
                break
        if out:
            return out

    return [{"page_number": 1, "text": raw.strip()}]


def coerce_pages(
    content_or_pages: Any, *, default_page_start: int = 1
) -> List[Dict[str, Any]]:
    def as_page_dict(x: PageLike, idx0: int) -> Optional[Dict[str, Any]]:
        if isinstance(x, str):
            t = x.strip()
            if not t:
                return None
            return {"page_number": default_page_start + idx0, "text": t}
        if isinstance(x, dict):
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

    if isinstance(content_or_pages, str):
        s = content_or_pages.strip()
        if s and s[:1] in "[{" and s[-1:] in "]}":
            try:
                parsed = json.loads(s)
                return coerce_pages(parsed, default_page_start=default_page_start)
            except Exception:
                pass
        return split_pages_from_text(s)

    if isinstance(content_or_pages, dict):
        if "pages" in content_or_pages:
            pages = content_or_pages.get("pages") or []
            out: List[Dict[str, Any]] = []
            for i, item in enumerate(pages):
                row = as_page_dict(item, i)
                if row:
                    out.append(row)
            return out
        row = as_page_dict(content_or_pages, 0)
        return [row] if row else []

    if isinstance(content_or_pages, list):
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(content_or_pages):
            row = as_page_dict(item, i)
            if row:
                out.append(row)
        return out

    return []


def chroma_docs_to_pydantic(objs: dict, model_cls: Type[T]) -> List[T]:
    docs = objs.get("documents") or []
    if docs and isinstance(docs[0], list):
        docs = docs[0]
    return [model_cls.model_validate_json(doc) for doc in docs]


def normalize_chroma_result(
    objs: Dict[str, Any],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids = objs.get("ids") or []
    docs = objs.get("documents") or []
    metas = objs.get("metadatas") or []

    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if docs and isinstance(docs[0], list):
        docs = docs[0]
    if metas and isinstance(metas[0], list):
        metas = metas[0]

    if not metas:
        metas = [{} for _ in docs]

    return ids, docs, metas


def chroma_to_models(objs: Dict[str, Any], model_cls: Type[T]) -> List[T]:
    _, docs, _ = normalize_chroma_result(objs)
    return [model_cls.model_validate_json(doc) for doc in docs]


def chroma_to_models_with_meta(
    objs: Dict[str, Any], model_cls: Type[T]
) -> List[Tuple[str, T, Dict[str, Any]]]:
    ids, docs, metas = normalize_chroma_result(objs)
    out: List[Tuple[str, T, Dict[str, Any]]] = []
    for rid, doc, meta in zip(ids, docs, metas):
        out.append((rid, model_cls.model_validate_json(doc), meta or {}))
    return out
