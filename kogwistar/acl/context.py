from __future__ import annotations

from dataclasses import asdict, dataclass

from kogwistar.server.auth_middleware import (
    get_current_acl_principal_id,
    get_current_principal_groups,
    get_security_scope,
)


@dataclass(frozen=True, slots=True)
class AclContext:
    principal_id: str
    principal_groups: tuple[str, ...] = ()
    security_scope: str | None = None
    acl_enabled: bool = True
    purpose: str = "user_visible"
    source_graph: str | None = None
    source_entity_id: str | None = None
    visibility: str | None = None
    owner_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def current_acl_context(
    *,
    acl_enabled: bool,
    purpose: str,
    source_graph: str | None = None,
    source_entity_id: str | None = None,
    visibility: str | None = None,
    owner_id: str | None = None,
) -> AclContext:
    return AclContext(
        principal_id=get_current_acl_principal_id(),
        principal_groups=get_current_principal_groups(),
        security_scope=str(get_security_scope() or "").strip().lower() or None,
        acl_enabled=bool(acl_enabled),
        purpose=str(purpose or "user_visible"),
        source_graph=source_graph,
        source_entity_id=source_entity_id,
        visibility=visibility,
        owner_id=owner_id,
    )


__all__ = ["AclContext", "current_acl_context"]
