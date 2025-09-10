# shortids.py
from __future__ import annotations
import json, hashlib, pathlib, re
from typing import Any, Iterable
from contextvars import ContextVar

# Run-id handling (prototype: run_id == raw JWT)
run_id_ctx: ContextVar[str] = ContextVar("run_id", default="anonymous")

from contextlib import contextmanager

def token_to_run_id(jwt_token: str) -> str: return jwt_token

@contextmanager
def run_id_scope(token: str):
    tok = run_id_ctx.set(token_to_run_id(token))
    try:
        yield
    finally:
        run_id_ctx.reset(tok)


def set_current_token(jwt_token: str) -> None: run_id_ctx.set(token_to_run_id(jwt_token))

class ShortIdMapper:
    SHORT_PREFIX = "<sid>"
    SHORT_RE     = re.compile(r"^<sid>[0-9]+$")

    # Graph-focused id fields
    SCALAR_ID_KEYS: tuple[str, ...] = (
        "id", "doc_id", "node_id", "edge_id", "edge_endpoint_id"
    )
    LIST_ID_KEYS:   tuple[str, ...] = (
        "source_ids", "target_ids", "source_edge_ids", "target_edge_ids"
    )

    def __init__(self, run_id: str, root_dir: str = "./.shortids"):
        self.run_id = run_id
        self.root = pathlib.Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self.obj_max_depth: int = 1  # shallow by default (top-level only)

    # --- persistence ---
    def _file(self) -> pathlib.Path:
        h = hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()[:32]
        return self.root / f"{h}.json"

    def _load(self) -> dict:
        p = self._file()
        if p.exists():
            try:
                return json.loads(p.read_text("utf-8"))
            except Exception:
                pass
        return {"next": 1, "l2s": {}, "s2l": {}}

    def _save(self) -> None:
        self._file().write_text(json.dumps(self.state, ensure_ascii=False), "utf-8")

    # --- knobs ---
    def set_obj_max_depth(self, depth: int) -> None:
        self.obj_max_depth = max(0, int(depth))

    def set_id_keys(self, scalars: Iterable[str] | None = None, lists: Iterable[str] | None = None) -> None:
        if scalars is not None:
            self.SCALAR_ID_KEYS = tuple(scalars)
        if lists is not None:
            self.LIST_ID_KEYS = tuple(lists)

    # --- id primitives ---
    def _alloc_short_for(self, long_id: str) -> str:
        st = self.state
        if long_id in st["l2s"]:
            return st["l2s"][long_id]
        sid = f"{self.SHORT_PREFIX}{st['next']}"
        st["next"] += 1
        st["l2s"][long_id] = sid
        st["s2l"][sid] = long_id
        self._save()
        return sid

    def l2s_id(self, in_id: str) -> str:
        """Server→User: if already <sid>…, keep; else allocate/return <sid>…"""
        if not isinstance(in_id, str):
            return in_id
        if self.SHORT_RE.fullmatch(in_id):
            return in_id
        # treat ANY other string as a long id in these fields
        return self._alloc_short_for(in_id)

    def s2l_id(self, in_id: str) -> str:
        """User→Server: ONLY accept <sid>…; anything else is rejected in id fields."""
        if not isinstance(in_id, str):
            return in_id
        if not self.SHORT_RE.fullmatch(in_id):
            raise ValueError("Only <sid>… is accepted in id fields.")
        long_id = self.state["s2l"].get(in_id)
        if not long_id:
            raise ValueError(f"Unknown short id '{in_id}' for this run.")
        return long_id

    # --- depth-limited object walkers (targeted keys only) ---
    def _walk_ids_l2s(self, obj: Any, depth: int) -> Any:
        if depth < 0:
            return obj
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in self.SCALAR_ID_KEYS:
                    out[k] = self._val_l2s(v)
                elif k in self.LIST_ID_KEYS:
                    out[k] = self._list_l2s(v)
                else:
                    out[k] = self._walk_ids_l2s(v, depth - 1) if depth > 0 else v
            return out
        if isinstance(obj, list):
            return [self._walk_ids_l2s(v, depth) for v in obj] if depth > 0 else obj
        return obj

    def _walk_ids_s2l(self, obj: Any, depth: int) -> Any:
        if depth < 0:
            return obj
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in self.SCALAR_ID_KEYS:
                    out[k] = self._val_s2l(v)
                elif k in self.LIST_ID_KEYS:
                    out[k] = self._list_s2l(v)
                else:
                    out[k] = self._walk_ids_s2l(v, depth - 1) if depth > 0 else v
            return out
        if isinstance(obj, list):
            return [self._walk_ids_s2l(v, depth) for v in obj] if depth > 0 else obj
        return obj

    def _val_l2s(self, v: Any) -> Any:
        if isinstance(v, str): return self.l2s_id(v)
        if isinstance(v, list): return [self._val_l2s(x) for x in v]
        return v

    def _list_l2s(self, v: Any) -> Any:
        if isinstance(v, list): return [self._val_l2s(x) for x in v]
        return v

    def _val_s2l(self, v: Any) -> Any:
        if isinstance(v, str): return self.s2l_id(v)
        if isinstance(v, list): return [self._val_s2l(x) for x in v]
        return v

    def _list_s2l(self, v: Any) -> Any:
        if isinstance(v, list): return [self._val_s2l(x) for x in v]
        return v

    # --- doc (JSON string) helpers: only targeted keys are touched ---
    def l2s_doc(self, in_doc_str: str) -> str:
        if not isinstance(in_doc_str, str): return in_doc_str
        try:
            data = json.loads(in_doc_str)
        except Exception:
            return in_doc_str  # not JSON: don't touch
        data2 = self._walk_ids_l2s(data, self.obj_max_depth - 1)
        return json.dumps(data2, ensure_ascii=False)

    def s2l_doc(self, in_doc_str: str) -> str:
        if not isinstance(in_doc_str, str): return in_doc_str
        try:
            data = json.loads(in_doc_str)
        except Exception:
            return in_doc_str  # not JSON: don't touch
        data2 = self._walk_ids_s2l(data, self.obj_max_depth - 1)
        return json.dumps(data2, ensure_ascii=False)

    # --- plain objects (dict/list) ---
    def l2s_obj(self, in_obj: Any) -> Any:
        if hasattr(in_obj, "model_dump"): in_obj = in_obj.model_dump()
        return self._walk_ids_l2s(in_obj, self.obj_max_depth - 1)

    def s2l_obj(self, in_obj: Any) -> Any:
        if hasattr(in_obj, "model_dump"): in_obj = in_obj.model_dump()
        return self._walk_ids_s2l(in_obj, self.obj_max_depth - 1)

# Per-run registry + required top-level API
_MAPPERS: dict[str, ShortIdMapper] = {}
def _mapper_for_current_run() -> ShortIdMapper:
    rid = run_id_ctx.get()
    m = _MAPPERS.get(rid)
    if not m:
        m = ShortIdMapper(rid)
        _MAPPERS[rid] = m
    return m

def set_shortid_obj_depth(depth: int) -> None: _mapper_for_current_run().set_obj_max_depth(depth)
def set_shortid_keys(scalars: Iterable[str] | None = None, lists: Iterable[str] | None = None) -> None:
    _mapper_for_current_run().set_id_keys(scalars, lists)

# === required function signatures ===
def s2l_doc(in_doc_str): return _mapper_for_current_run().s2l_doc(in_doc_str)
def l2s_doc(in_doc_str): return _mapper_for_current_run().l2s_doc(in_doc_str)
def l2s_id(in_id):       return _mapper_for_current_run().l2s_id(in_id)
def s2l_id(in_id):       return _mapper_for_current_run().s2l_id(in_id)
def s2l_obj(in_obj):     return _mapper_for_current_run().s2l_obj(in_obj)
def l2s_obj(in_obj):     return _mapper_for_current_run().l2s_obj(in_obj)
