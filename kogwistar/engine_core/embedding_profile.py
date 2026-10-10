"""Backend-neutral embedding compatibility metadata.

Embedding vectors are only comparable inside one declared semantic space.  The
profile registry records that space beside the existing durable metadata, while
backend inspectors report whether the physical store is empty or already has
vectors.  The registry is operational metadata, not graph truth.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.runtime.checkpointed_projection import ProjectionPayload

PROFILE_REGISTRY_NAMESPACE = "__kogwistar_embedding_profiles_v1__"
PROFILE_PROJECTION_SCHEMA_VERSION = 1


def _json_object(value: JsonValue | None, *, field: str) -> JsonObject:
    """Narrow a persisted JSON value while keeping corruption fail-closed."""

    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _json_int(value: JsonValue | None, *, field: str, default: int | None = None) -> int | None:
    """Convert the scalar integer forms accepted by legacy profile records."""

    if value is None:
        return default
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise TypeError(f"{field} must be an integer-compatible JSON scalar")


def _json_text(value: JsonValue | None, *, field: str, default: str = "") -> str:
    """Read a persisted string field without allowing nested JSON values."""

    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


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
    max_sequence_length: int | None = None
    crop_token_budget: int | None = None
    tokenizer_fingerprint: str | None = None
    crop_policy: str | None = None
    embedding_kind: str | None = None
    model_revision: str | None = None
    preprocessing_fingerprint: str | None = None
    max_image_patches: int | None = None

    def __post_init__(self) -> None:
        if not str(self.provider).strip():
            raise ValueError("embedding profile provider must not be empty")
        if not str(self.model).strip():
            raise ValueError("embedding profile model must not be empty")
        if int(self.dimension) <= 0:
            raise ValueError("embedding profile dimension must be positive")
        if str(self.similarity_metric).lower() not in {"cosine", "l2", "ip"}:
            raise ValueError("embedding profile similarity_metric must be cosine, l2, or ip")
        if self.max_sequence_length is not None and int(self.max_sequence_length) <= 0:
            raise ValueError("embedding profile max_sequence_length must be positive")
        if self.crop_token_budget is not None and int(self.crop_token_budget) <= 0:
            raise ValueError("embedding profile crop_token_budget must be positive")
        if (
            self.max_sequence_length is not None
            and self.crop_token_budget is not None
            and int(self.crop_token_budget) > int(self.max_sequence_length)
        ):
            raise ValueError("embedding profile crop_token_budget cannot exceed max_sequence_length")
        if self.crop_policy is not None and not str(self.crop_policy).strip():
            raise ValueError("embedding profile crop_policy must not be empty")
        if self.embedding_kind is not None and str(self.embedding_kind).lower() not in {
            "single_vector",
            "dense",
            "late_interaction",
        }:
            raise ValueError(
                "embedding profile embedding_kind must be single_vector, dense, or late_interaction"
            )
        if self.model_revision is not None and not str(self.model_revision).strip():
            raise ValueError("embedding profile model_revision must not be empty")
        if self.preprocessing_fingerprint is not None and not str(self.preprocessing_fingerprint).strip():
            raise ValueError("embedding profile preprocessing_fingerprint must not be empty")
        if self.max_image_patches is not None and int(self.max_image_patches) <= 0:
            raise ValueError("embedding profile max_image_patches must be positive")

    def as_dict(self) -> JsonObject:
        result = {
            "provider": str(self.provider).strip().lower(),
            "model": str(self.model).strip(),
            "dimension": int(self.dimension),
            "similarity_metric": str(self.similarity_metric).strip().lower(),
            "endpoint_fingerprint": self.endpoint_fingerprint,
        }
        optional = {
            "max_sequence_length": self.max_sequence_length,
            "crop_token_budget": self.crop_token_budget,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "crop_policy": self.crop_policy,
            "embedding_kind": self.embedding_kind,
            "model_revision": self.model_revision,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "max_image_patches": self.max_image_patches,
        }
        result.update({key: value for key, value in optional.items() if value is not None})
        return result

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue]) -> EmbeddingProfile:
        if not isinstance(value, Mapping):
            raise TypeError("embedding profile payload must be a mapping")
        return cls(
            provider=_json_text(value.get("provider"), field="provider"),
            model=_json_text(value.get("model"), field="model"),
            dimension=_json_int(value.get("dimension"), field="dimension", default=0) or 0,
            similarity_metric=_json_text(
                value.get("similarity_metric"), field="similarity_metric", default="cosine"
            ),
            endpoint_fingerprint=(
                _json_text(value.get("endpoint_fingerprint"), field="endpoint_fingerprint")
                if value.get("endpoint_fingerprint") is not None
                else None
            ),
            max_sequence_length=_json_int(
                value.get("max_sequence_length"), field="max_sequence_length"
            ),
            crop_token_budget=_json_int(value.get("crop_token_budget"), field="crop_token_budget"),
            tokenizer_fingerprint=(
                _json_text(value.get("tokenizer_fingerprint"), field="tokenizer_fingerprint")
                if value.get("tokenizer_fingerprint") is not None
                else None
            ),
            crop_policy=(
                _json_text(value.get("crop_policy"), field="crop_policy")
                if value.get("crop_policy") is not None
                else None
            ),
            embedding_kind=(
                _json_text(value.get("embedding_kind"), field="embedding_kind")
                if value.get("embedding_kind") is not None
                else None
            ),
            model_revision=(
                _json_text(value.get("model_revision"), field="model_revision")
                if value.get("model_revision") is not None
                else None
            ),
            preprocessing_fingerprint=(
                _json_text(value.get("preprocessing_fingerprint"), field="preprocessing_fingerprint")
                if value.get("preprocessing_fingerprint") is not None
                else None
            ),
            max_image_patches=_json_int(value.get("max_image_patches"), field="max_image_patches"),
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

    def embedding_storage_scope_aliases(self) -> tuple[str, ...]: ...


class NamedProjectionStore(Protocol):
    """Minimal durable metadata surface required by the profile registry."""

    def get_named_projection(self, namespace: str, key: str) -> ProjectionPayload | None: ...

    def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: ProjectionPayload,
        *,
        expected_last_authoritative_seq: int | None,
        expected_last_materialized_seq: int | None,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> bool: ...

    def compare_and_swap_named_projections(
        self, updates: list[ProjectionPayload]
    ) -> bool: ...

    def list_named_projections(self, namespace: str) -> list[ProjectionPayload]: ...


@runtime_checkable
class AsyncNamedProjectionStore(Protocol):
    """Native async counterpart for durable named-projection adapters.

    Implementations must await their backend operations directly.  This seam
    deliberately covers metadata projections only; it is not an async engine
    facade and does not bridge a synchronous engine behind an async API.
    """

    async def get_named_projection(
        self, namespace: str, key: str
    ) -> ProjectionPayload | None: ...

    async def compare_and_swap_named_projection(
        self,
        namespace: str,
        key: str,
        payload: ProjectionPayload,
        *,
        expected_last_authoritative_seq: int | None,
        expected_last_materialized_seq: int | None,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> bool: ...

    async def compare_and_swap_named_projections(
        self, updates: list[ProjectionPayload]
    ) -> bool: ...

    async def list_named_projections(
        self, namespace: str
    ) -> list[ProjectionPayload]: ...


def _profile_projection(profile: EmbeddingProfile, *, adopted: bool) -> ProjectionPayload:
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
    ) -> JsonObject:
        scope = self._key(inspector)
        state = inspector.inspect_embedding_storage()
        projection = self._metadata.get_named_projection(PROFILE_REGISTRY_NAMESPACE, scope)
        result: JsonObject = {
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
            payload = _json_object(projection.get("payload"), field="payload")
            self._validate_schema(scope, projection, payload)
            try:
                registered = EmbeddingProfile.from_mapping(
                    _json_object(payload.get("embedding_profile"), field="embedding_profile")
                )
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

        # Older releases keyed local Chroma stores by absolute path and
        # PostgreSQL stores by the full connection URL.  A copied store or a
        # rotated credential must retain its binding, but conflicting legacy
        # records must never be guessed through.
        aliases = tuple(getattr(inspector, "embedding_storage_scope_aliases", lambda: ())())
        list_projections = getattr(self._metadata, "list_named_projections", None)
        list_projections_fn = (
            cast(Callable[[str], list[ProjectionPayload]], list_projections)
            if callable(list_projections)
            else None
        )
        legacy_rows = (
            list_projections_fn(PROFILE_REGISTRY_NAMESPACE)
            if list_projections_fn is not None
            else [
                row
                for alias in aliases
                for row in [self._metadata.get_named_projection(PROFILE_REGISTRY_NAMESPACE, alias)]
                if row is not None
            ]
        )
        legacy_rows = [row for row in legacy_rows if str(row.get("key") or "") in aliases]
        if legacy_rows:
            profiles = [
                self._registered_profile(str(row.get("key") or ""), row)
                for row in legacy_rows
            ]
            if any(profile.fingerprint != configured.fingerprint for profile in profiles):
                raise EmbeddingProfileMismatchError(
                    storage_scope=scope,
                    configured=configured,
                    registered=profiles[0],
                )
            if len({profile.fingerprint for profile in profiles}) != 1:
                raise CorruptEmbeddingProfileError(
                    f"conflicting legacy embedding profile bindings for {scope}"
                )
            payload = _json_object(legacy_rows[0].get("payload"), field="payload")
            inserted = self._metadata.compare_and_swap_named_projection(
                PROFILE_REGISTRY_NAMESPACE,
                scope,
                payload,
                expected_last_authoritative_seq=None,
                expected_last_materialized_seq=None,
                last_authoritative_seq=0,
                last_materialized_seq=0,
                projection_schema_version=(
                    _json_int(
                        legacy_rows[0].get("projection_schema_version"),
                        field="projection_schema_version",
                        default=1,
                    )
                    or 1
                ),
                materialization_status="bound",
            )
            if inserted:
                return configured
            winner = self._metadata.get_named_projection(PROFILE_REGISTRY_NAMESPACE, scope)
            if winner is None:
                raise CorruptEmbeddingProfileError(
                    f"embedding profile migration disappeared during startup for {scope}"
                )
            return self._validate_current(scope, winner, configured)

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
    def _registered_profile(scope: str, projection: Mapping[str, JsonValue]) -> EmbeddingProfile:
        payload = _json_object(projection.get("payload"), field="payload")
        EmbeddingProfileRegistry._validate_schema(scope, projection, payload)
        try:
            return EmbeddingProfile.from_mapping(
                _json_object(payload.get("embedding_profile"), field="embedding_profile")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptEmbeddingProfileError(
                f"invalid embedding profile registry record for {scope}"
            ) from exc

    @staticmethod
    def _validate_current(
        scope: str,
        projection: Mapping[str, JsonValue],
        configured: EmbeddingProfile,
    ) -> EmbeddingProfile:
        payload = _json_object(projection.get("payload"), field="payload")
        EmbeddingProfileRegistry._validate_schema(scope, projection, payload)
        try:
            registered = EmbeddingProfile.from_mapping(
                _json_object(payload.get("embedding_profile"), field="embedding_profile")
            )
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
        projection: Mapping[str, JsonValue],
        payload: Mapping[str, JsonValue],
    ) -> None:
        try:
            projection_version = _json_int(
                projection.get("projection_schema_version"),
                field="projection_schema_version",
                default=-1,
            )
            payload_version = _json_int(
                payload.get("profile_schema_version"), field="profile_schema_version", default=-1
            )
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
    "PROFILE_PROJECTION_SCHEMA_VERSION",
    "PROFILE_REGISTRY_NAMESPACE",
    "AsyncNamedProjectionStore",
    "CorruptEmbeddingProfileError",
    "EmbeddingProfile",
    "EmbeddingProfileError",
    "EmbeddingProfileMismatchError",
    "EmbeddingProfileRegistry",
    "EmbeddingStorageInspector",
    "EmbeddingStorageState",
    "LegacyEmbeddingProfileError",
    "NamedProjectionStore",
    "endpoint_fingerprint",
]
