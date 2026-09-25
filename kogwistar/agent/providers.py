"""Deterministic provider registry for optional agent plugins."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from .catalog import CatalogEntry


class ProviderCollisionError(ValueError):
    """Raised when a provider-qualified identity would be overwritten."""


class ProviderOwnershipError(ValueError):
    """Raised when a disposer tries to remove another registration."""


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
    failure_mode: Literal["fail_closed", "isolate"] = "fail_closed"


class ProviderRegistry:
    """Ordered registry with explicit collision and disposal semantics."""

    def __init__(self) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}

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
        existing = self._registrations.get(key)
        if existing is not None:
            raise ProviderCollisionError(
                f"provider identity already registered: {key}"
            )
        registration = ProviderRegistration(
            identity=identity,
            provider=provider,
            registration_fingerprint=registration_fingerprint,
            failure_mode=failure_mode,
        )
        self._registrations[key] = registration
        return registration

    def get(self, provider_id: str, version: str = "v1") -> ProviderRegistration:
        key = ProviderIdentity(provider_id, version).qualified_id
        try:
            return self._registrations[key]
        except KeyError as exc:
            raise KeyError(f"unknown provider identity: {key}") from exc

    def list(self) -> tuple[ProviderRegistration, ...]:
        return tuple(self._registrations.values())

    def dispose(self, registration: ProviderRegistration) -> None:
        key = registration.identity.qualified_id
        current = self._registrations.get(key)
        if current is not registration:
            raise ProviderOwnershipError(f"registration is not active owner: {key}")
        self._registrations.pop(key)
        close = getattr(registration.provider, "close", None)
        if callable(close):
            close()

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
            cleanup(registration.identity.provider_id)

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
