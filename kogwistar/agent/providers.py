"""Deterministic provider registry for optional agent plugins."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from kogwistar.engine_core.embedding_profile import NamedProjectionStore

from .catalog import CatalogEntry


PROVIDER_LIFECYCLE_NAMESPACE = "agent_provider_lifecycle"


class ProviderCollisionError(ValueError):
    """Raised when a provider-qualified identity would be overwritten."""


class ProviderOwnershipError(ValueError):
    """Raised when a disposer tries to remove another registration."""


class ProviderInactiveError(RuntimeError):
    """Raised when work from a retired provider lifecycle tries to commit."""


@runtime_checkable
class DiscoveryProvider(Protocol):
    """Read-only provider contract shared by skill/MCP/doc discovery."""

    provider_id: str
    provider_version: str

    def descriptors(self) -> list[Mapping[str, Any]]: ...


@runtime_checkable
class ModelProvider(Protocol):
    provider_id: str

    def complete(self, prompt: str, context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class ToolProvider(Protocol):
    provider_id: str

    def invoke(self, arguments: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SkillProvider(Protocol):
    provider_id: str

    def descriptors(self) -> list[Mapping[str, Any]]: ...

    def load(self, provider_local_id: str) -> str: ...


@runtime_checkable
class MemoryProvider(Protocol):
    provider_id: str

    def search(self, query: str, **kwargs: Any) -> list[Mapping[str, Any]]: ...


@runtime_checkable
class CompressorProvider(Protocol):
    provider_id: str

    def compress(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def normalize_descriptor(
    descriptor: Mapping[str, Any], *, provider_id: str, provider_version: str = "v1"
) -> CatalogEntry:
    """Normalize provider data without granting invocation authority."""

    data = dict(descriptor)
    data.setdefault("provider_id", provider_id)
    data.setdefault("provider_version", provider_version)
    data.setdefault("provider_local_id", data.get("logical_id") or data.get("name"))
    data.setdefault("logical_id", f"{provider_id}:{data['provider_local_id']}")
    data.setdefault("source_fingerprint", "provider-unknown")
    return CatalogEntry.model_validate(data)


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    provider_id: str
    version: str = "v1"
    fingerprint: str = ""

    def __post_init__(self) -> None:
        provider_id = self.provider_id.strip().lower()
        version = self.version.strip()
        if not provider_id or not version:
            raise ValueError("provider_id and version must be non-empty")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "version", version)

    @property
    def qualified_id(self) -> str:
        return f"{self.provider_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    identity: ProviderIdentity
    provider: Any
    registration_fingerprint: str
    generation: int = 1
    failure_mode: Literal["fail_closed", "isolate"] = "fail_closed"

    @property
    def lifecycle_token(self) -> str:
        """Durable token identifying one provider registration incarnation."""

        return f"{self.generation}:{self.registration_fingerprint}"


class ProviderCleanupToken(str):
    """String-compatible unload identity for one provider incarnation."""

    provider_version: str
    generation: int
    lifecycle_token: str

    def __new__(cls, registration: ProviderRegistration) -> "ProviderCleanupToken":
        value = str.__new__(cls, registration.identity.provider_id)
        value.provider_version = registration.identity.version
        value.generation = registration.generation
        value.lifecycle_token = registration.lifecycle_token
        return value


class ProviderRegistry:
    """Ordered registry with explicit collision and disposal semantics."""

    def __init__(self, *, metadata: NamedProjectionStore | None = None) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}
        self._generations: dict[str, int] = {}
        self._metadata = metadata
        self._lock = RLock()

    def _lifecycle_row(self, key: str) -> dict[str, Any] | None:
        if self._metadata is None:
            return None
        return self._metadata.get_named_projection(PROVIDER_LIFECYCLE_NAMESPACE, key)

    def _write_lifecycle(
        self,
        key: str,
        *,
        generation: int,
        fingerprint: str,
        status: str,
        expected: dict[str, Any] | None,
    ) -> None:
        if self._metadata is None:
            return
        payload = {
            "provider_key": key,
            "generation": int(generation),
            "registration_fingerprint": fingerprint,
            "status": status,
        }
        expected_a = (
            int(expected["last_authoritative_seq"])
            if expected is not None
            else None
        )
        expected_m = (
            int(expected["last_materialized_seq"])
            if expected is not None
            else None
        )
        if not self._metadata.compare_and_swap_named_projection(
            PROVIDER_LIFECYCLE_NAMESPACE,
            key,
            payload,
            expected_last_authoritative_seq=expected_a,
            expected_last_materialized_seq=expected_m,
            last_authoritative_seq=(
                int(expected["last_authoritative_seq"]) + 1
                if expected is not None
                else int(generation)
            ),
            last_materialized_seq=(
                int(expected["last_materialized_seq"]) + 1
                if expected is not None
                else int(generation)
            ),
            projection_schema_version=1,
            materialization_status=status,
        ):
            raise ProviderCollisionError(f"provider lifecycle changed concurrently: {key}")

    def lifecycle_guard_update(self, registration: ProviderRegistration) -> dict[str, Any] | None:
        """Return a same-store CAS guard for one active registration."""

        with self._lock:
            row = self._lifecycle_row(registration.identity.qualified_id)
            if self._metadata is None:
                return None
            payload = (row or {}).get("payload") or {}
            if (
                row is None
                or str(payload.get("status")) != "active"
                or int(payload.get("generation", -1)) != registration.generation
                or str(payload.get("registration_fingerprint"))
                != registration.registration_fingerprint
            ):
                raise ProviderInactiveError(
                    "durable provider lifecycle is no longer active: "
                    f"{registration.identity.qualified_id}@{registration.generation}"
                )
            expected_a = int(row.get("last_authoritative_seq", 0))
            expected_m = int(row.get("last_materialized_seq", 0))
            return {
                "namespace": PROVIDER_LIFECYCLE_NAMESPACE,
                "key": registration.identity.qualified_id,
                "payload": dict(payload),
                "expected_last_authoritative_seq": expected_a,
                "expected_last_materialized_seq": expected_m,
                "last_authoritative_seq": expected_a,
                "last_materialized_seq": expected_m,
                "projection_schema_version": int(row.get("projection_schema_version", 1)),
                "materialization_status": "active",
            }

    @staticmethod
    def _fingerprint(provider: Any, explicit: str | None) -> str:
        if explicit:
            return str(explicit)
        name = f"{type(provider).__module__}.{type(provider).__qualname__}"
        return sha256(name.encode("utf-8")).hexdigest()

    def register(
        self,
        *,
        provider_id: str,
        provider: Any,
        version: str = "v1",
        fingerprint: str | None = None,
        failure_mode: Literal["fail_closed", "isolate"] = "fail_closed",
    ) -> ProviderRegistration:
        identity = ProviderIdentity(provider_id=provider_id, version=version)
        key = identity.qualified_id
        registration_fingerprint = self._fingerprint(provider, fingerprint)
        with self._lock:
            existing = self._registrations.get(key)
            if existing is not None:
                raise ProviderCollisionError(
                    f"provider identity already registered: {key}"
                )
            durable_row = self._lifecycle_row(key)
            durable_payload = (durable_row or {}).get("payload") or {}
            generation = max(
                self._generations.get(key, 0),
                int(durable_payload.get("generation", 0) or 0),
            ) + 1
            registration = ProviderRegistration(
                identity=identity,
                provider=provider,
                registration_fingerprint=registration_fingerprint,
                generation=generation,
                failure_mode=failure_mode,
            )
            self._write_lifecycle(
                key,
                generation=generation,
                fingerprint=registration_fingerprint,
                status="active",
                expected=durable_row,
            )
            # Publish process-local state only after durable CAS succeeds.
            self._registrations[key] = registration
            self._generations[key] = generation
            return registration

    def get(self, provider_id: str, version: str = "v1") -> ProviderRegistration:
        key = ProviderIdentity(provider_id, version).qualified_id
        with self._lock:
            try:
                return self._registrations[key]
            except KeyError as exc:
                raise KeyError(f"unknown provider identity: {key}") from exc

    def list(self) -> tuple[ProviderRegistration, ...]:
        with self._lock:
            return tuple(self._registrations.values())

    def dispose(self, registration: ProviderRegistration) -> None:
        key = registration.identity.qualified_id
        with self._lock:
            current = self._registrations.get(key)
            if current is not registration:
                raise ProviderOwnershipError(f"registration is not active owner: {key}")
            self._write_lifecycle(
                key,
                generation=registration.generation,
                fingerprint=registration.registration_fingerprint,
                status="retired",
                expected=self._lifecycle_row(key),
            )
            self._registrations.pop(key)
        close = getattr(registration.provider, "close", None)
        if callable(close):
            close()

    def run_if_active(
        self,
        registration: ProviderRegistration,
        operation: Callable[[], Any],
    ) -> Any:
        """Commit provider work only while exact registration remains active."""

        key = registration.identity.qualified_id
        with self._lock:
            current = self._registrations.get(key)
            if current is not registration or current.generation != registration.generation:
                raise ProviderInactiveError(
                    f"provider lifecycle is no longer active: {key}@{registration.generation}"
                )
            if self._metadata is not None:
                durable = self._lifecycle_row(key)
                payload = (durable or {}).get("payload") or {}
                if (
                    durable is None
                    or str(payload.get("status")) != "active"
                    or int(payload.get("generation", -1)) != registration.generation
                    or str(payload.get("registration_fingerprint"))
                    != registration.registration_fingerprint
                ):
                    raise ProviderInactiveError(
                        f"durable provider lifecycle is no longer active: {key}@{registration.generation}"
                    )
            # Keep the lifecycle lock across the final projection commit.  The
            # operation is already post-parse; this prevents unload from
            # racing a durable CAS and is the generic commit guard.
            return operation()

    def unload(
        self,
        provider_id: str,
        version: str = "v1",
        *,
        cleanup: Callable[[str], None] | None = None,
    ) -> None:
        """Unload provider, then let owners retract current projections."""
        registration = self.get(provider_id, version)
        self.dispose(registration)
        if cleanup is not None:
            cleanup(ProviderCleanupToken(registration))

    def discovery_descriptors(
        self, *, isolate_failures: bool | None = None
    ) -> tuple[CatalogEntry, ...]:
        """Read descriptors in order; optionally isolate one provider failure."""

        result: list[CatalogEntry] = []
        for registration in self.list():
            provider = registration.provider
            loader = getattr(provider, "descriptors", None)
            if not callable(loader):
                continue
            try:
                descriptors = loader()
            except Exception:
                isolated = (
                    registration.failure_mode == "isolate"
                    if isolate_failures is None
                    else isolate_failures
                )
                if isolated:
                    continue
                raise
            for descriptor in descriptors:
                result.append(
                    normalize_descriptor(
                        descriptor,
                        provider_id=registration.identity.provider_id,
                        provider_version=registration.identity.version,
                    )
                )
        return tuple(result)
