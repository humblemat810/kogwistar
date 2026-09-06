"""Backend-neutral embedding compatibility metadata.

Embedding vectors are only comparable inside one declared semantic space.  The
profile registry records that space beside the existing durable metadata, while
backend inspectors report whether the physical store is empty or already has
vectors.  The registry is operational metadata, not graph truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit


PROFILE_REGISTRY_NAMESPACE = "__kogwistar_embedding_profiles_v1__"
PROFILE_PROJECTION_SCHEMA_VERSION = 1


def endpoint_fingerprint(endpoint: str | None) -> str | None:
    """Return a stable, non-secret identity for an embedding endpoint."""

    if not endpoint:
        return None
    raw = str(endpoint).strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        # Drop credentials, query strings, and fragments before hashing.  A
        # deployment URL can contain credentials accidentally, and they must
        # never become part of durable profile metadata.
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port is not None else ""
        normalized = urlunsplit(
            (parsed.scheme.lower(), f"{host.lower()}{port}", parsed.path.rstrip("/"), "", "")
        )
    else:
        normalized = raw.rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingProfile:
    """Immutable semantic identity for one persisted vector space."""

    provider: str
    model: str
    dimension: int
    similarity_metric: str = "cosine"
    endpoint_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not str(self.provider).strip():
            raise ValueError("embedding profile provider must not be empty")
        if not str(self.model).strip():
            raise ValueError("embedding profile model must not be empty")
        if int(self.dimension) <= 0:
            raise ValueError("embedding profile dimension must be positive")
        if str(self.similarity_metric).lower() not in {"cosine", "l2", "ip"}:
            raise ValueError("embedding profile similarity_metric must be cosine, l2, or ip")

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": str(self.provider).strip().lower(),
            "model": str(self.model).strip(),
            "dimension": int(self.dimension),
            "similarity_metric": str(self.similarity_metric).strip().lower(),
            "endpoint_fingerprint": self.endpoint_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EmbeddingProfile":
        if not isinstance(value, Mapping):
            raise TypeError("embedding profile payload must be a mapping")
        return cls(
            provider=str(value.get("provider", "")),
            model=str(value.get("model", "")),
            dimension=int(value.get("dimension", 0)),
            similarity_metric=str(value.get("similarity_metric", "cosine")),
            endpoint_fingerprint=(
                str(value["endpoint_fingerprint"])
                if value.get("endpoint_fingerprint") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class EmbeddingStorageState:
    """Observed state of one physical vector storage scope."""

    backend_kind: str
    storage_scope: str
    persistent: bool
    vector_count: int
    details: tuple[str, ...] = ()

    @property
    def has_vectors(self) -> bool:
        return self.vector_count > 0


class EmbeddingProfileError(RuntimeError):
    """Base class for startup embedding compatibility failures."""


class EmbeddingProfileMismatchError(EmbeddingProfileError):
    """Raised when a physical store is bound to another embedding profile."""

    def __init__(
        self,
        *,
        storage_scope: str,
        configured: EmbeddingProfile,
        registered: EmbeddingProfile,
    ) -> None:
        self.storage_scope = storage_scope
        self.configured = configured
        self.registered = registered
        super().__init__(
            "embedding profile mismatch for "
            f"{storage_scope}: configured {configured.provider}/{configured.model}/"
            f"{configured.dimension}D ({configured.fingerprint[:12]}), but the store is bound to "
            f"{registered.provider}/{registered.model}/{registered.dimension}D "
            f"({registered.fingerprint[:12]}). Stop writers and migrate to an isolated store "
            "by replaying canonical state and re-embedding; existing vectors are not resized."
        )


class LegacyEmbeddingProfileError(EmbeddingProfileError):
    """Raised when vectors predate the profile registry and cannot be verified."""

    def __init__(self, *, state: EmbeddingStorageState, configured: EmbeddingProfile) -> None:
        self.state = state
        self.configured = configured
        super().__init__(
            "embedding profile is unbound for a non-empty persistent "
            f"{state.backend_kind} store ({state.storage_scope}); found "
            f"{state.vector_count} vectors but cannot prove they use "
            f"{configured.provider}/{configured.model}/{configured.dimension}D. "
            "Run the operator legacy-profile adoption command only after verifying the "
            "old configuration, or archive/replay into an isolated store."
        )


class CorruptEmbeddingProfileError(EmbeddingProfileError):
    """Raised when the durable registry record cannot be trusted."""


class EmbeddingStorageInspector(Protocol):
    """Optional backend capability for physical embedding compatibility checks."""

    def embedding_storage_scope(self) -> str: ...

    def inspect_embedding_storage(self) -> EmbeddingStorageState: ...


class NamedProjectionStore(Protocol):
    """Minimal durable metadata surface required by the profile registry."""

    def get_named_projection(self, namespace: str, key: str) -> dict[str, Any] | None: ...

    def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        **values: Any,
    ) -> bool: ...


def _profile_projection(profile: EmbeddingProfile, *, adopted: bool) -> dict[str, Any]:
    return {
        "profile_schema_version": PROFILE_PROJECTION_SCHEMA_VERSION,
        "embedding_profile": profile.as_dict(),
        "embedding_fingerprint": profile.fingerprint,
        "legacy_adopted": bool(adopted),
    }


class EmbeddingProfileRegistry:
    """Bind and validate one profile per physical vector storage scope."""

    def __init__(self, metadata: NamedProjectionStore) -> None:
        self._metadata = metadata

    def _key(self, inspector: EmbeddingStorageInspector) -> str:
        return str(inspector.embedding_storage_scope())

    def inspect(
        self,
        inspector: EmbeddingStorageInspector,
        *,
        configured: EmbeddingProfile | None = None,
    ) -> dict[str, Any]:
        scope = self._key(inspector)
        state = inspector.inspect_embedding_storage()
        projection = self._metadata.get_named_projection(PROFILE_REGISTRY_NAMESPACE, scope)
        result: dict[str, Any] = {
            "storage_scope": scope,
            "storage_state": {
                "backend_kind": state.backend_kind,
                "persistent": state.persistent,
                "vector_count": state.vector_count,
                "details": list(state.details),
            },
            "registered": None,
            "configured": configured.as_dict() if configured else None,
        }
        if projection is not None:
            payload = projection.get("payload") or {}
            self._validate_schema(scope, projection, payload)
            try:
                registered = EmbeddingProfile.from_mapping(payload["embedding_profile"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CorruptEmbeddingProfileError(
                    f"invalid embedding profile registry record for {scope}"
                ) from exc
            result["registered"] = {
                **registered.as_dict(),
                "fingerprint": registered.fingerprint,
                "legacy_adopted": bool(payload.get("legacy_adopted", False)),
            }
        return result

    def ensure_bound(
        self,
        inspector: EmbeddingStorageInspector,
        configured: EmbeddingProfile,
        *,
        allow_legacy_adoption: bool = False,
    ) -> EmbeddingProfile:
        scope = self._key(inspector)
        current = self._metadata.get_named_projection(PROFILE_REGISTRY_NAMESPACE, scope)
        if current is not None:
            return self._validate_current(scope, current, configured)

        state = inspector.inspect_embedding_storage()
        if state.persistent and state.has_vectors and not allow_legacy_adoption:
            raise LegacyEmbeddingProfileError(state=state, configured=configured)

        payload = _profile_projection(configured, adopted=allow_legacy_adoption and state.has_vectors)
        inserted = self._metadata.compare_and_swap_named_projection(
            PROFILE_REGISTRY_NAMESPACE,
            scope,
            payload,
            expected_last_authoritative_seq=None,
            expected_last_materialized_seq=None,
            last_authoritative_seq=0,
            last_materialized_seq=0,
            projection_schema_version=PROFILE_PROJECTION_SCHEMA_VERSION,
            materialization_status="bound",
        )
        if inserted:
            return configured

        winner = self._metadata.get_named_projection(PROFILE_REGISTRY_NAMESPACE, scope)
        if winner is None:
            raise CorruptEmbeddingProfileError(
                f"embedding profile binding disappeared during startup for {scope}"
            )
        return self._validate_current(scope, winner, configured)

    @staticmethod
    def _validate_current(
        scope: str,
        projection: Mapping[str, Any],
        configured: EmbeddingProfile,
    ) -> EmbeddingProfile:
        payload = projection.get("payload") or {}
        EmbeddingProfileRegistry._validate_schema(scope, projection, payload)
        try:
            registered = EmbeddingProfile.from_mapping(payload["embedding_profile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptEmbeddingProfileError(
                f"invalid embedding profile registry record for {scope}"
            ) from exc
        if registered.fingerprint != configured.fingerprint:
            raise EmbeddingProfileMismatchError(
                storage_scope=scope,
                configured=configured,
                registered=registered,
            )
        return registered

    @staticmethod
    def _validate_schema(
        scope: str,
        projection: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        try:
            projection_version = int(projection.get("projection_schema_version", -1))
            payload_version = int(payload.get("profile_schema_version", -1))
        except (TypeError, ValueError) as exc:
            raise CorruptEmbeddingProfileError(
                f"invalid embedding profile schema metadata for {scope}"
            ) from exc
        if projection_version != PROFILE_PROJECTION_SCHEMA_VERSION:
            raise CorruptEmbeddingProfileError(
                f"unsupported embedding profile projection schema for {scope}"
            )
        if payload_version != PROFILE_PROJECTION_SCHEMA_VERSION:
            raise CorruptEmbeddingProfileError(
                f"unsupported embedding profile payload schema for {scope}"
            )


__all__ = [
    "CorruptEmbeddingProfileError",
    "EmbeddingProfile",
    "EmbeddingProfileError",
    "EmbeddingProfileMismatchError",
    "EmbeddingProfileRegistry",
    "EmbeddingStorageInspector",
    "EmbeddingStorageState",
    "LegacyEmbeddingProfileError",
    "NamedProjectionStore",
    "PROFILE_PROJECTION_SCHEMA_VERSION",
    "PROFILE_REGISTRY_NAMESPACE",
    "endpoint_fingerprint",
]
