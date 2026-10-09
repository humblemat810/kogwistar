from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, TypeAlias, TypeVar

Metadata: TypeAlias = dict[str, Any]
MetadataPatch: TypeAlias = Mapping[str, object]
_T = TypeVar("_T")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dict(doc: object) -> Metadata:
    if isinstance(doc, dict):
        return dict(doc)
    if not isinstance(doc, str):
        return {}
    try:
        x = json.loads(doc)
        return dict(x) if isinstance(x, dict) else {}
    except Exception:
        return {}


def merge_meta(base_meta: Mapping[str, object] | None, patch: MetadataPatch) -> Metadata:
    return {**(base_meta or {}), **patch}


def is_tombstoned(meta: Mapping[str, object] | None) -> bool:
    meta = meta or {}
    return str(meta.get("lifecycle_status") or "active") == "tombstoned"


def str_or_none(to_str: object | None) -> str | None:
    if to_str is None:
        return to_str
    return str(to_str)


def refs_fingerprint(refs: Iterable[object] | None) -> str:
    payload = [
        {
            "doc_id": getattr(r, "doc_id", None),
            "method": getattr(getattr(r, "verification", None), "method", None),
            "is_verified": getattr(
                getattr(r, "verification", None), "is_verified", None
            ),
            "score": getattr(getattr(r, "verification", None), "score", None),
            "sp": getattr(r, "start_page", None),
            "ep": getattr(r, "end_page", None),
            "sc": getattr(r, "start_char", None),
            "ec": getattr(r, "end_char", None),
            "snip": (getattr(r, "excerpt", None) or "")[:64],
        }
        for r in (refs or [])
    ]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def strip_none(d: Mapping[str, _T]) -> dict[str, _T]:
    return {k: v for k, v in d.items() if v is not None}


def json_or_none(v: object) -> str | None:
    return None if v is None else json.dumps(v)
