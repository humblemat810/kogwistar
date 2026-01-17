# id_provider.py
from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
from typing import Callable

UuidFn = Callable[[], uuid.UUID]

_uuid_fn_var: ContextVar[UuidFn] = ContextVar("_uuid_fn_var", default=uuid.uuid4)

PROJECT_NS = uuid.uuid5(uuid.NAMESPACE_URL, "graph-knowledge-engine")

def stable_id(kind: str, *parts: str) -> uuid.UUID:
    key = json.dumps([kind, *parts], separators=(",", ":"), ensure_ascii=False)
    return uuid.uuid5(PROJECT_NS, key)

def new_event_id() -> str:
    return uuid.uuid4()

def new_id_str() -> uuid.UUID:
    # IMPORTANT: call .get() INSIDE the function (do not capture it at import/definition time)
    return _uuid_fn_var.get()()

def set_uuid_fn(fn: UuidFn):
    # returns a token so you can restore previous context safely
    return _uuid_fn_var.set(fn)

def reset_uuid_fn(token) -> None:
    _uuid_fn_var.reset(token)
