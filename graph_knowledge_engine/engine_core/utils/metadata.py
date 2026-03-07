from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dict(doc: Any) -> dict:
    if isinstance(doc, dict):
        return doc
    if not isinstance(doc, str):
        return {}
    try:
        x = json.loads(doc)
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}


def merge_meta(base_meta: dict | None, patch: dict) -> dict:
    base_meta = base_meta or {}
    return {**base_meta, **patch}


def is_tombstoned(meta: dict | None) -> bool:
    meta = meta or {}
    return str(meta.get("lifecycle_status") or "active") == "tombstoned"


def str_or_none(to_str):
    if to_str is None:
        return to_str
    return str(to_str)


def refs_fingerprint(refs) -> str:
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
            "snip": (getattr(r, "excerpt", None) or "")[:64],
        }
        for r in (refs or [])
    ]
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def json_or_none(v):
    return None if v is None else json.dumps(v)
