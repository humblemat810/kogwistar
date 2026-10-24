"""Shared Python representation of the versioned delegation authority contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

DELEGATION_AUTHORITY_SCHEMA_VERSION = 1


def attenuate_capabilities(
    parent: Iterable[object], requested: Iterable[object] | None = None
) -> tuple[str, ...]:
    """Return an equal-or-narrower capability set, rejecting expansion."""

    parent_capabilities = {str(value) for value in parent}
    requested_capabilities = (
        parent_capabilities
        if requested is None
        else {str(value) for value in requested}
    )
    missing = requested_capabilities - parent_capabilities
    if missing:
        raise PermissionError(
            "child requested capabilities absent from parent: "
            + ", ".join(sorted(missing))
        )
    return tuple(sorted(requested_capabilities))


@dataclass(frozen=True, slots=True)
class DelegationAuthorityV1:
    """Serializable authority snapshot shared at a workflow-language seam."""

    delegated_by: str
    effective_capabilities: tuple[str, ...]
    schema_version: int = DELEGATION_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.delegated_by.strip():
            raise ValueError("delegated_by must be a non-empty subject")
        if self.schema_version != DELEGATION_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("unsupported delegation authority schema version")

    def as_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "delegated_by": self.delegated_by,
            "effective_capabilities": list(self.effective_capabilities),
        }


def delegation_authority_payload(
    *,
    delegated_by: str,
    parent: Iterable[object],
    requested: Iterable[object] | None = None,
) -> Mapping[str, object]:
    """Build the canonical JSON-shaped authority payload after attenuation."""

    effective = attenuate_capabilities(parent, requested)
    return DelegationAuthorityV1(
        delegated_by=delegated_by,
        effective_capabilities=effective,
    ).as_payload()
