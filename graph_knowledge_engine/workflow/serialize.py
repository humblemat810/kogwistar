from __future__ import annotations

import json
import hashlib
from typing import Any, Dict

Json = Any


def stable_json_dumps(obj: Json) -> str:
    """
    Deterministic JSON encoding for replay.
    Raises TypeError if not serializable.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def try_serialize_with_ref(obj: Json) -> str:
    """
    Serialize to JSON if possible; otherwise store a small ref object.

    This keeps replay/history possible without forcing everything to be natively JSON.
    """
    try:
        return stable_json_dumps(obj)
    except TypeError:
        # Fallback: store a compact ref payload with a hash of repr.
        # You can later upgrade this to real blob storage.
        rep = repr(obj).encode("utf-8")
        h = hashlib.sha256(rep).hexdigest()
        ref = {"_ref_type": "repr_sha256", "sha256": h, "repr": repr(obj)[:2000]}
        return stable_json_dumps(ref)