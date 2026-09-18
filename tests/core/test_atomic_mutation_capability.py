from __future__ import annotations

from kogwistar.engine_core.storage_backend import (
    AtomicMutationCapability,
    NoopUnitOfWork,
    get_atomic_mutation_capability,
)


def test_atomic_mutation_capability_is_explicit() -> None:
    atomic = AtomicMutationCapability(mode="atomic", reason="transaction")
    eventual = AtomicMutationCapability(mode="eventual", reason="reconcile")

    assert atomic.supports_atomic_mutation is True
    assert eventual.supports_atomic_mutation is False


def test_noop_uow_cannot_be_used_for_atomic_replacement() -> None:
    capability = get_atomic_mutation_capability(NoopUnitOfWork())

    assert capability.mode == "none"
    assert capability.supports_atomic_mutation is False


def test_backend_declaration_is_read_without_backend_type_inspection() -> None:
    class Backend:
        atomic_mutation_capability = AtomicMutationCapability(
            mode="eventual",
            reason="single writer only",
        )

    capability = get_atomic_mutation_capability(Backend())

    assert capability.mode == "eventual"
    assert capability.reason == "single writer only"


def test_async_backend_does_not_claim_sync_atomicity() -> None:
    class AsyncBackend:
        _is_async_engine = True
        atomic_mutation_capability = AtomicMutationCapability(
            mode="atomic",
            reason="async transaction",
        )

    capability = get_atomic_mutation_capability(AsyncBackend())

    assert capability.mode == "none"
    assert capability.supports_atomic_mutation is False
