# shortids.py
from __future__ import annotations
import json
import hashlib
import pathlib
import re
from typing import Any, Iterable
from contextvars import ContextVar

from ._rust_bridge import (
    RustParityError,
    contract_implementation_mode,
    json_contract_compatible,
    short_id_transform as _rust_short_id_transform,
)

# Run-id handling (prototype: run_id == raw JWT)
run_id_ctx: ContextVar[str] = ContextVar("run_id", default="anonymous")

from contextlib import contextmanager


def token_to_run_id(jwt_token: str) -> str:
    return jwt_token


@contextmanager
def run_id_scope(token: str):
    tok = run_id_ctx.set(token_to_run_id(token))
    try:
        yield
    finally:
        run_id_ctx.reset(tok)


def set_current_token(jwt_token: str) -> None:
    run_id_ctx.set(token_to_run_id(jwt_token))


class ShortIdMapper:
    SHORT_PREFIX = "<sid>"
    SHORT_RE = re.compile(r"^<sid>[0-9]+$")

    # Graph-focused id fields
    SCALAR_ID_KEYS: tuple[str, ...] = (
        "id",
        "doc_id",
        "node_id",
        "edge_id",
        "edge_endpoint_id",
    )
    LIST_ID_KEYS: tuple[str, ...] = (
        "source_ids",
        "target_ids",
        "source_edge_ids",
        "target_edge_ids",
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

    def _native_payload(
        self,
        value: Any,
        direction: str,
        state: dict | None = None,
        *,
        primitive: bool = False,
    ) -> dict:
        return {
            "state": self.state if state is None else state,
            "input": value,
            "direction": direction,
            "depth": self.obj_max_depth - 1,
            "scalar_keys": list(self.SCALAR_ID_KEYS),
            "list_keys": list(self.LIST_ID_KEYS),
            "primitive": primitive,
        }

    def _native_transform(
        self, value: Any, direction: str, *, primitive: bool = False
    ) -> Any | None:
        """JSON-only native transform. File persistence remains Python-owned."""
        if contract_implementation_mode() != "rust" or not json_contract_compatible(value):
            return None
        try:
            result = _rust_short_id_transform(
                payload=self._native_payload(
                    value, direction, primitive=primitive
                ),
                python_value=None,
            )
        except ValueError as exc:
            # Preserve legacy public exception class/message; code remains usable
            # for callers that opt into machine-readable native diagnostics.
            error = ValueError(str(exc))
            error.code = getattr(exc, "code", None)
            raise error from None
        self.state = result["state"]
        self._save()
        return result["value"]

    def _shadow_compare(
        self,
        value: Any,
        direction: str,
        before: dict,
        python_value: Any,
        *,
        primitive: bool = False,
    ) -> Any:
        if contract_implementation_mode() != "shadow" or not json_contract_compatible(value):
            return python_value
        native = _rust_short_id_transform(
            payload=self._native_payload(
                value, direction, before, primitive=primitive
            ),
            python_value=None,
        )
        expected = {"state": self.state, "value": python_value}
        if native != expected:
            raise RustParityError(
                "Rust parity mismatch for short_id_transform: "
                f"python={expected!r}, rust={native!r}"
            )
        return python_value

    # --- knobs ---
    def set_obj_max_depth(self, depth: int) -> None:
        self.obj_max_depth = max(0, int(depth))

    def set_id_keys(
        self, scalars: Iterable[str] | None = None, lists: Iterable[str] | None = None
    ) -> None:
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
        native = self._native_transform(in_id, "l2s", primitive=True)
        if native is not None:
            return native
        before = json.loads(json.dumps(self.state, ensure_ascii=False))
        output = self._alloc_short_for(in_id)
        return self._shadow_compare(in_id, "l2s", before, output, primitive=True)

    def s2l_id(self, in_id: str) -> str:
        """User→Server: ONLY accept <sid>…; anything else is rejected in id fields."""
        if not isinstance(in_id, str):
            return in_id
        if not self.SHORT_RE.fullmatch(in_id):
            raise ValueError("Only <sid>… is accepted in id fields.")
        native = self._native_transform(in_id, "s2l", primitive=True)
        if native is not None:
            return native
        before = json.loads(json.dumps(self.state, ensure_ascii=False))
        long_id = self.state["s2l"].get(in_id)
        if not long_id:
            raise ValueError(f"Unknown short id '{in_id}' for this run.")
        return self._shadow_compare(in_id, "s2l", before, long_id, primitive=True)

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
        if isinstance(v, str):
            return self.l2s_id(v)
        if isinstance(v, list):
            return [self._val_l2s(x) for x in v]
        return v

    def _list_l2s(self, v: Any) -> Any:
        if isinstance(v, list):
            return [self._val_l2s(x) for x in v]
        return v

    def _val_s2l(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.s2l_id(v)
        if isinstance(v, list):
            return [self._val_s2l(x) for x in v]
        return v

    def _list_s2l(self, v: Any) -> Any:
        if isinstance(v, list):
            return [self._val_s2l(x) for x in v]
        return v

    # --- doc (JSON string) helpers: only targeted keys are touched ---
    def l2s_doc(self, in_doc_str: str) -> str:
        if not isinstance(in_doc_str, str):
            return in_doc_str
        try:
            data = json.loads(in_doc_str)
        except Exception:
            return in_doc_str  # not JSON: don't touch
        before = json.loads(json.dumps(self.state, ensure_ascii=False))
        data2 = self._native_transform(data, "l2s")
        if data2 is None:
            data2 = self._walk_ids_l2s(data, self.obj_max_depth - 1)
            data2 = self._shadow_compare(data, "l2s", before, data2)
        return json.dumps(data2, ensure_ascii=False)

    def s2l_doc(self, in_doc_str: str) -> str:
        if not isinstance(in_doc_str, str):
            return in_doc_str
        try:
            data = json.loads(in_doc_str)
        except Exception:
            return in_doc_str  # not JSON: don't touch
        before = json.loads(json.dumps(self.state, ensure_ascii=False))
        data2 = self._native_transform(data, "s2l")
        if data2 is None:
            data2 = self._walk_ids_s2l(data, self.obj_max_depth - 1)
            data2 = self._shadow_compare(data, "s2l", before, data2)
        return json.dumps(data2, ensure_ascii=False)

    # --- plain objects (dict/list) ---
    def l2s_obj(self, in_obj: Any) -> Any:
        if hasattr(in_obj, "model_dump"):
            in_obj = in_obj.model_dump()
        before = json.loads(json.dumps(self.state, ensure_ascii=False))
        output = self._native_transform(in_obj, "l2s")
        if output is not None:
            return output
        output = self._walk_ids_l2s(in_obj, self.obj_max_depth - 1)
        return self._shadow_compare(in_obj, "l2s", before, output)

    def s2l_obj(self, in_obj: Any) -> Any:
        if hasattr(in_obj, "model_dump"):
            in_obj = in_obj.model_dump()
        before = json.loads(json.dumps(self.state, ensure_ascii=False))
        output = self._native_transform(in_obj, "s2l")
        if output is not None:
            return output
        output = self._walk_ids_s2l(in_obj, self.obj_max_depth - 1)
        return self._shadow_compare(in_obj, "s2l", before, output)


# Per-run registry + required top-level API
_MAPPERS: dict[str, ShortIdMapper] = {}


def _mapper_for_current_run() -> ShortIdMapper:
    rid = run_id_ctx.get()
    m = _MAPPERS.get(rid)
    if not m:
        m = ShortIdMapper(rid)
        _MAPPERS[rid] = m
    return m


def set_shortid_obj_depth(depth: int) -> None:
    _mapper_for_current_run().set_obj_max_depth(depth)


def set_shortid_keys(
    scalars: Iterable[str] | None = None, lists: Iterable[str] | None = None
) -> None:
    _mapper_for_current_run().set_id_keys(scalars, lists)


# === required function signatures ===
def s2l_doc(in_doc_str):
    return _mapper_for_current_run().s2l_doc(in_doc_str)


def l2s_doc(in_doc_str):
    return _mapper_for_current_run().l2s_doc(in_doc_str)


def l2s_id(in_id):
    return _mapper_for_current_run().l2s_id(in_id)


def s2l_id(in_id):
    return _mapper_for_current_run().s2l_id(in_id)


def s2l_obj(in_obj):
    return _mapper_for_current_run().s2l_obj(in_obj)


def l2s_obj(in_obj):
    return _mapper_for_current_run().l2s_obj(in_obj)
