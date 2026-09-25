"""Deterministic provider registry for optional agent plugins."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


class ProviderCollisionError(ValueError):
    """Raised when a provider-qualified identity would be overwritten."""


class ProviderOwnershipError(ValueError):
    """Raised when a disposer tries to remove another registration."""


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

    def unload(self, provider_id: str, version: str = "v1") -> None:
        self.dispose(self.get(provider_id, version))
