from __future__ import annotations

from importlib import import_module
import json
import math
import os
import sys
from types import ModuleType
from typing import Any
import uuid


class RustExtensionUnavailableError(RuntimeError):
    """Raised when a Rust-backed mode is selected without the native extension."""


class RustParityError(RuntimeError):
    """Raised when Python and Rust contract implementations disagree."""


def store_sqlite(*, path: str | os.PathLike[str], operation: dict[str, Any]) -> Any:
    """Execute stable Phase-3 SQLite JSON ABI against actual database path.

    This is an explicit test/integration bridge. It does not select an authority
    mode and cannot route `EngineSQLite` or `KOGWISTAR_IMPL_META_STORE`.
    """
    engine_sqlite = sys.modules.get("kogwistar.engine_core.engine_sqlite")
    active_connection = (
        getattr(engine_sqlite, "get_active_sqlite_conn", lambda: None)()
        if engine_sqlite is not None
        else None
    )
    if active_connection is not None:
        error = RustParityError(
            "Rust SQLite bridge cannot open a second connection inside an active "
            "Python SQLite transaction"
        )
        error.code = "KOGWISTAR_STORE_ACTIVE_PYTHON_TRANSACTION"
        raise error
    try:
        extension = _load_extension()
        return json.loads(
            extension.store_sqlite_json(
                json.dumps(
                    {"path": os.fspath(path), "operation": operation},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
    except RustParityError:
        raise
    except Exception as exc:
        error = RustParityError(f"Rust SQLite store operation failed: {exc}")
        code = getattr(exc, "code", None)
        if code is not None:
            setattr(error, "code", code)
        raise error from exc


def _postgres_dsn_for_rust(dsn: str | None) -> str:
    """Resolve configured DSN and remove SQLAlchemy driver suffix for tokio-postgres."""
    resolved = dsn
    if not resolved:
        for env_name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL"):
            resolved = os.getenv(env_name)
            if resolved:
                break
    if not resolved:
        raise ValueError("PostgreSQL DSN required (GKE_PG_DSN, PG_DSN, DATABASE_URL)")
    scheme, separator, rest = resolved.partition("://")
    if separator and scheme.lower().startswith("postgresql+"):
        return f"postgresql://{rest}"
    return resolved


def store_postgres(
    *, dsn: str | None = None, schema: str, operation: dict[str, Any]
) -> Any:
    """Execute Phase-3 PostgreSQL JSON ABI through native Tokio/Postgres store.

    This explicit integration bridge does not select an implementation mode or
    route Python's public Postgres backend. SQLAlchemy DSNs are normalized in
    memory only; errors intentionally do not echo connection credentials.
    """
    try:
        extension = _load_extension()
        return json.loads(
            extension.store_postgres_json(
                json.dumps(
                    {
                        "dsn": _postgres_dsn_for_rust(dsn),
                        "schema": schema,
                        "operation": operation,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
    except Exception as exc:
        error = RustParityError("Rust PostgreSQL store operation failed")
        code = getattr(exc, "code", None)
        if code is not None:
            setattr(error, "code", code)
        raise error from exc


_MISSING_PYTHON_RESULT = object()


def contract_implementation_mode() -> str:
    mode = os.getenv(
        "KOGWISTAR_IMPL_CONTRACTS", os.getenv("KOGWISTAR_IMPL_MODE", "python")
    ).strip().lower()
    if mode not in {"python", "shadow", "rust"}:
        raise ValueError(
            "KOGWISTAR_IMPL_CONTRACTS must be one of: python, shadow, rust; "
            f"got {mode!r}"
        )
    return mode


def runtime_implementation_mode() -> str:
    mode = os.getenv(
        "KOGWISTAR_IMPL_RUNTIME", os.getenv("KOGWISTAR_IMPL_MODE", "python")
    ).strip().lower()
    if mode not in {"python", "shadow", "rust"}:
        raise ValueError(
            "KOGWISTAR_IMPL_RUNTIME must be one of: python, shadow, rust; "
            f"got {mode!r}"
        )
    return mode


def _store_implementation_mode(*, env_name: str) -> str:
    mode = os.getenv(env_name, os.getenv("KOGWISTAR_IMPL_MODE", "python")).strip().lower()
    if mode not in {"python", "shadow", "rust"}:
        raise ValueError(
            f"{env_name} must be one of: python, shadow, rust; got {mode!r}"
        )
    return mode


def graph_store_implementation_mode() -> str:
    """Internal Phase-2 graph-store selector; public backend remains Python-owned."""
    return _store_implementation_mode(env_name="KOGWISTAR_IMPL_GRAPH_STORE")


def meta_store_implementation_mode() -> str:
    """Internal Phase-2 meta-store selector; public meta-store remains Python-owned."""
    return _store_implementation_mode(env_name="KOGWISTAR_IMPL_META_STORE")


def _load_extension() -> ModuleType:
    try:
        return import_module("kogwistar._rust")
    except ImportError as exc:
        raise RustExtensionUnavailableError(
            "Rust contract mode requires the kogwistar._rust extension. "
            "Install the 'rust' extra and run 'maturin develop'."
        ) from exc


def _select(*, operation: str, python_value: Any, rust_value: Any, mode: str) -> Any:
    if mode == "shadow":
        if rust_value != python_value:
            raise RustParityError(
                f"Rust parity mismatch for {operation}: "
                f"python={python_value!r}, rust={rust_value!r}"
            )
        return python_value
    return rust_value


def _store_values_equal(python_value: Any, rust_value: Any) -> bool:
    """Compare JSON-only store reads without erasing list order or shape."""
    if isinstance(python_value, float) and isinstance(rust_value, float):
        return math.isclose(python_value, rust_value, rel_tol=1e-6, abs_tol=1e-6)
    if type(python_value) is not type(rust_value):
        return False
    if isinstance(python_value, list):
        return len(python_value) == len(rust_value) and all(
            _store_values_equal(left, right)
            for left, right in zip(python_value, rust_value, strict=True)
        )
    if isinstance(python_value, dict):
        return python_value.keys() == rust_value.keys() and all(
            _store_values_equal(python_value[key], rust_value[key]) for key in python_value
        )
    return python_value == rust_value


def store_memory_read(
    *,
    snapshot: dict[str, Any],
    operation: dict[str, Any],
    python_value: Any = _MISSING_PYTHON_RESULT,
    store: str = "graph",
) -> Any:
    """Inspect isolated native store built from immutable JSON snapshot.

    Caller owns Python read and passes its already-computed result for shadow
    comparison. This deliberately never receives backend/meta objects, so native
    execution cannot mutate authoritative Python state or recurse into its oracle.
    """
    if store == "graph":
        mode = graph_store_implementation_mode()
    elif store == "meta":
        mode = meta_store_implementation_mode()
    else:
        raise ValueError(f"store must be 'graph' or 'meta', got {store!r}")
    if mode == "python":
        if python_value is _MISSING_PYTHON_RESULT:
            raise RuntimeError("Python store mode requires caller-computed Python result")
        return python_value
    extension = _load_extension()
    rust_value = json.loads(
        extension.store_memory_read_json(
            json.dumps(
                {"snapshot": snapshot, "operation": operation},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    )
    if mode == "rust":
        return rust_value
    if python_value is _MISSING_PYTHON_RESULT:
        raise RuntimeError("Shadow store mode requires caller-computed Python result")
    if not _store_values_equal(python_value, rust_value):
        raise RustParityError(
            f"Rust parity mismatch for {store}_store_memory_read: "
            f"python={python_value!r}, rust={rust_value!r}"
        )
    return python_value


def metadata_filter_json_contract_compatible(value: Any) -> bool:
    """Whether a value retains its Python filter semantics through JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(metadata_filter_json_contract_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and metadata_filter_json_contract_compatible(item)
            for key, item in value.items()
        )
    return False


def contract_stable_id(
    *, kind: str, parts: tuple[Any, ...], python_value: uuid.UUID
) -> uuid.UUID:
    mode = contract_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    payload_json = json.dumps(
        [kind, *parts], ensure_ascii=False, separators=(",", ":")
    )
    rust_value = uuid.UUID(str(extension.stable_id_json(payload_json)))
    return _select(
        operation="stable_id",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def contract_canonical_json(*, value: Any, python_value: str) -> str:
    mode = contract_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    payload_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    rust_value = str(extension.canonical_json(payload_json))
    return _select(
        operation="canonical_json",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def contract_evidence_pack_digest_hash(
    *, value: dict[str, Any], python_value: str
) -> str:
    mode = contract_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    payload_json = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    rust_value = str(extension.evidence_pack_digest_hash(payload_json))
    return _select(
        operation="evidence_pack_digest_hash",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def contract_metadata_filter_matches(
    *, metadata: dict[str, Any], where: Any, python_value: bool | None = None
) -> bool:
    """Run native metadata filtering only after Python oracle has completed.

    Keeping `python_value` caller-provided prevents the Rust implementation from
    re-entering `_matches_where` while shadow mode is comparing the two paths.
    """
    mode = contract_implementation_mode()
    if mode == "python":
        if python_value is None:
            raise RuntimeError("Python metadata-filter mode requires Python result")
        return python_value
    extension = _load_extension()
    payload_json = json.dumps(
        {"metadata": metadata, "where": where},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rust_value = bool(extension.metadata_filter_matches(payload_json))
    if mode == "rust":
        return rust_value
    if python_value is None:
        raise RuntimeError("Shadow metadata-filter mode requires Python result")
    return _select(
        operation="metadata_filter_matches",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def json_contract_compatible(value: Any) -> bool:
    """Whether a value can cross this Phase-1 JSON-only native boundary."""
    return metadata_filter_json_contract_compatible(value)


def short_id_transform(*, payload: dict[str, Any], python_value: Any | None = None) -> Any:
    """Select native JSON short-ID transform; callers retain persistence ownership."""
    mode = contract_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    rust_value = json.loads(
        extension.short_id_transform(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    )
    if mode == "rust":
        return rust_value
    if python_value is None:
        return rust_value
    return _select(
        operation="short_id_transform",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def runtime_apply_state_update(
    *, payload: dict[str, Any], python_value: Any | None = None
) -> Any:
    """Select native JSON state fold. Python caller owns object mutation."""
    mode = runtime_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    rust_value = json.loads(
        extension.apply_state_update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    )
    if mode == "rust":
        return rust_value
    if python_value is None:
        return rust_value
    return _select(
        operation="runtime_state_update",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def runtime_workflow_may_reach_join(
    *,
    node_ids: list[str],
    edges: list[tuple[str, str]],
    join_ids: list[str],
    python_value: dict[str, int] | None = None,
) -> dict[str, int] | None:
    """Select native static join lineage; caller owns Python oracle execution."""
    mode = runtime_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    payload_json = json.dumps(
        {"node_ids": node_ids, "edges": edges, "join_ids": join_ids},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    raw_value = json.loads(extension.workflow_may_reach_join(payload_json))
    rust_value = {
        str(node_id): sum(1 << int(bit) for bit in bits)
        for node_id, bits in raw_value.items()
    }
    if mode == "rust":
        return rust_value
    if python_value is None:
        raise RuntimeError("Shadow workflow-lineage mode requires Python result")
    return _select(
        operation="runtime_workflow_may_reach_join",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def runtime_workflow_terminal_reachable(
    *,
    node_ids: list[str],
    edges: list[tuple[str, str]],
    start_node_id: str,
    terminal_ids: list[str],
    python_value: bool | None = None,
) -> bool | None:
    """Select native terminal reachability; callers retain public errors."""
    mode = runtime_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    payload_json = json.dumps(
        {
            "node_ids": node_ids,
            "edges": edges,
            "join_ids": [],
            "start_node_id": start_node_id,
            "terminal_ids": terminal_ids,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rust_value = bool(extension.workflow_terminal_reachable(payload_json))
    if mode == "rust":
        return rust_value
    if python_value is None:
        raise RuntimeError("Shadow workflow-terminal mode requires Python result")
    return _select(
        operation="runtime_workflow_terminal_reachable",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def contract_canonical_entity_event(*, payload: dict[str, Any], python_value: str) -> str:
    """Internal entity-event contract adapter; no second public Python model."""
    mode = contract_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    rust_value = str(
        extension.canonical_entity_event(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    )
    return _select(
        operation="canonical_entity_event",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )


def contract_replay_entity_events(*, payload: dict[str, Any], python_value: str) -> str:
    """Internal event replay contract adapter; no event-store ownership change."""
    mode = contract_implementation_mode()
    if mode == "python":
        return python_value
    extension = _load_extension()
    rust_value = str(
        extension.replay_entity_events(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    )
    return _select(
        operation="replay_entity_events",
        python_value=python_value,
        rust_value=rust_value,
        mode=mode,
    )
