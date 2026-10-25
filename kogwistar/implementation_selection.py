from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Literal, ParamSpec, TypeAlias, TypeVar, cast

from kogwistar.json_types import JsonObject

_IMPLEMENTATION_ENV_NAMES = (
    "KOGWISTAR_IMPL_MODE",
    "KOGWISTAR_IMPL_CONTRACTS",
    "KOGWISTAR_IMPL_META_STORE",
    "KOGWISTAR_IMPL_GRAPH_STORE",
    "KOGWISTAR_IMPL_RUNTIME",
    "KOGWISTAR_IMPL_SERVER",
    "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
)
ImplementationSelectionSnapshot: TypeAlias = tuple[tuple[str, str | None], ...]
TransactionAuthority: TypeAlias = Literal["python", "rust"]

_selection_snapshot: ContextVar[ImplementationSelectionSnapshot | None] = (
    ContextVar("kogwistar_implementation_selection", default=None)
)
_transaction_authority: ContextVar[TransactionAuthority | None] = ContextVar(
    "kogwistar_transaction_authority", default=None
)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _owner_implementation_snapshot(owner: object) -> ImplementationSelectionSnapshot | None:
    """Resolve a construction snapshot from engines and their subsystem proxies."""
    snapshot = getattr(owner, "_implementation_selection_snapshot", None)
    if snapshot is not None:
        return cast(ImplementationSelectionSnapshot, snapshot)
    delegated_owner = getattr(owner, "_e", None)
    delegated_snapshot = getattr(
        delegated_owner, "_implementation_selection_snapshot", None
    )
    if delegated_snapshot is None:
        return None
    return cast(ImplementationSelectionSnapshot, delegated_snapshot)


def implementation_env(name: str, default: str | None = None) -> str | None:
    """Read selector configuration from the active run snapshot, if present."""
    snapshot = _selection_snapshot.get()
    if snapshot is None:
        return os.getenv(name, default)
    for key, value in snapshot:
        if key == name:
            return default if value is None else value
    return default


def capture_implementation_selection() -> ImplementationSelectionSnapshot:
    """Capture all implementation selectors as one immutable configuration."""
    return tuple((name, os.getenv(name)) for name in _IMPLEMENTATION_ENV_NAMES)


def transaction_authority_for_selection(
    snapshot: ImplementationSelectionSnapshot,
) -> TransactionAuthority:
    """Derive the transaction owner from the frozen implementation choice.

    Rust owns a transaction only when the metadata or coordinated PostgreSQL
    authority is explicitly Rust. All other configurations retain the legacy
    Python owner. This is an internal decision, not caller-provided payload.
    """
    selected = dict(snapshot)
    if (
        selected.get("KOGWISTAR_IMPL_META_STORE") == "rust"
        or selected.get("KOGWISTAR_IMPL_POSTGRES_AUTHORITY") == "rust"
    ):
        return "rust"
    return "python"


def active_transaction_authority() -> TransactionAuthority | None:
    """Return the owner of the current unit of work, if one is active."""
    return _transaction_authority.get()


@contextmanager
def transaction_authority_scope(authority: TransactionAuthority) -> Iterator[None]:
    """Bind a transaction to the run selection and prevent owner switching.

    A frozen implementation selection is authoritative for the whole run.  The
    explicit check here prevents a lower-level caller from bypassing that
    selection by entering a different authority scope directly.  Calls made
    without a selection snapshot retain the legacy behavior.
    """
    selected_snapshot = _selection_snapshot.get()
    if selected_snapshot is not None:
        expected = transaction_authority_for_selection(selected_snapshot)
        if authority != expected:
            raise RuntimeError(
                "transaction authority does not match frozen implementation selection"
            )
    active = _transaction_authority.get()
    if active is not None:
        if active != authority:
            raise RuntimeError(
                "nested transaction uses a different implementation authority"
            )
        yield
        return
    token = _transaction_authority.set(authority)
    try:
        yield
    finally:
        _transaction_authority.reset(token)


def resolve_implementation_selection(
    *owners: object,
) -> ImplementationSelectionSnapshot:
    """Return one selector snapshot for cooperating components.

    Components without a snapshot retain the legacy behavior and inherit a
    fresh capture. Components that already captured different selectors fail
    closed instead of running one operation with split authorities.
    """

    snapshots = tuple(
        snapshot
        for owner in owners
        if (snapshot := getattr(owner, "_implementation_selection_snapshot", None))
        is not None
    )
    if snapshots and any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        raise ValueError(
            "cooperating components use different implementation selections"
        )
    return snapshots[0] if snapshots else capture_implementation_selection()


def postgres_authority_selectors_aligned(
    *, meta_mode: str, graph_mode: str, authority_mode: str
) -> bool:
    """Return whether PostgreSQL selectors name one transaction authority.

    The graph and metadata selectors are not independent implementation
    choices for PostgreSQL: both participate in the same event/graph write
    boundary.  Keeping this predicate separate from environment parsing lets
    construction-time validation and trace generation use the same invariant.
    """
    return meta_mode == graph_mode == authority_mode


def implementation_selection_metadata(
    snapshot: ImplementationSelectionSnapshot,
) -> JsonObject:
    """Project raw and effective selectors to a versioned, non-secret trace.

    Raw values preserve the operator configuration for compatibility. Effective
    values make fallback explicit: an unset selector inherits the global mode,
    except PostgreSQL authority, whose safe default is Python. This prevents a
    trace from making an unset field look like an unverified Rust path.
    """

    raw = {name: value for name, value in snapshot}
    global_mode = raw.get("KOGWISTAR_IMPL_MODE") or "python"
    effective = {
        "KOGWISTAR_IMPL_MODE": global_mode,
        "KOGWISTAR_IMPL_CONTRACTS": raw.get("KOGWISTAR_IMPL_CONTRACTS") or global_mode,
        "KOGWISTAR_IMPL_META_STORE": raw.get("KOGWISTAR_IMPL_META_STORE") or global_mode,
        "KOGWISTAR_IMPL_GRAPH_STORE": raw.get("KOGWISTAR_IMPL_GRAPH_STORE") or global_mode,
        "KOGWISTAR_IMPL_RUNTIME": raw.get("KOGWISTAR_IMPL_RUNTIME") or global_mode,
        "KOGWISTAR_IMPL_SERVER": raw.get("KOGWISTAR_IMPL_SERVER") or global_mode,
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY": raw.get(
            "KOGWISTAR_IMPL_POSTGRES_AUTHORITY"
        )
        or "python",
    }
    postgres_coherent = not (
        effective["KOGWISTAR_IMPL_POSTGRES_AUTHORITY"] == "rust"
        and (
            effective["KOGWISTAR_IMPL_META_STORE"] != "rust"
            or effective["KOGWISTAR_IMPL_GRAPH_STORE"] != "rust"
        )
    )
    postgres_selector_alignment = postgres_authority_selectors_aligned(
        meta_mode=effective["KOGWISTAR_IMPL_META_STORE"],
        graph_mode=effective["KOGWISTAR_IMPL_GRAPH_STORE"],
        authority_mode=effective["KOGWISTAR_IMPL_POSTGRES_AUTHORITY"],
    )
    return {
        "schema_version": 1,
        "source": "engine_construction",
        "selectors": raw,
        "effective_selectors": effective,
        "invariants": {
            "postgres_transaction_coherent": postgres_coherent,
            "postgres_selector_alignment": postgres_selector_alignment,
        },
    }


@contextmanager
def implementation_selection_scope(
    snapshot: ImplementationSelectionSnapshot | None = None,
) -> Iterator[None]:
    """Freeze selectors for a run and reject nested authority changes."""
    active_snapshot = _selection_snapshot.get()
    if active_snapshot is not None:
        if snapshot is not None and snapshot != active_snapshot:
            raise RuntimeError(
                "nested workflow runtime has a different implementation selection"
            )
        yield
        return

    token = _selection_snapshot.set(snapshot or capture_implementation_selection())
    try:
        yield
    finally:
        _selection_snapshot.reset(token)


def implementation_selection_bound(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Bind selector configuration across a sync or async execution boundary."""
    if inspect.iscoroutinefunction(inspect.unwrap(function)):
        async_function = cast(Callable[_P, Awaitable[object]], function)

        @wraps(function)
        async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> object:
            owner = args[0] if args else None
            snapshot = _owner_implementation_snapshot(owner)
            with implementation_selection_scope(snapshot):
                return await async_function(*args, **kwargs)

        return cast(Callable[_P, _R], async_wrapper)

    @wraps(function)
    def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        owner = args[0] if args else None
        snapshot = _owner_implementation_snapshot(owner)
        with implementation_selection_scope(snapshot):
            return function(*args, **kwargs)

    return sync_wrapper
