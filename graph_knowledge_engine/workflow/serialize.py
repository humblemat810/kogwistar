from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping

try:
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore

Json = Any


def stable_json_dumps(obj: Json) -> str:
    """Deterministic JSON encoding for replay/persistence."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ref_obj(obj: Any) -> dict:
    rep = repr(obj).encode("utf-8")
    h = hashlib.sha256(rep).hexdigest()
    return {"_ref_type": "repr_sha256", "sha256": h, "repr": repr(obj)[:2000]}


def to_jsonable(obj: Any) -> Json:
    """Convert arbitrary objects into a JSON-compatible Python structure.

    Goals:
    - The returned value must be safe to pass to json.dumps.
    - Prefer structural serialization (dict/list primitives) over opaque strings.
    - For unknown objects, fall back to a small reference payload.

    This is the foundation for *real* checkpointing/replay: checkpoints should
    store a JSON structure, not a JSON string of a repr.
    """
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # mappings
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # iterables
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # pydantic models
    if BaseModel is not None and isinstance(obj, BaseModel):  # type: ignore[arg-type]
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            return _ref_obj(obj)

    # dataclasses
    if dataclasses.is_dataclass(obj):
        try:
            return to_jsonable(dataclasses.asdict(obj))
        except Exception:
            return _ref_obj(obj)

    # best-effort: objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj))
        except Exception:
            return _ref_obj(obj)

    return _ref_obj(obj)


def try_serialize_with_ref(obj: Json) -> str:
    """Serialize to a JSON string.

    - First converts to a JSON-compatible structure via to_jsonable.
    - Always returns a JSON string.

    Note: checkpoints should store *structured* JSON (the output of to_jsonable)
    as well, but several existing models store JSON as strings (state_json,
    result_json), so this remains useful.
    """
    return stable_json_dumps(to_jsonable(obj))
