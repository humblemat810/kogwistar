from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from kogwistar.server.auth_middleware import (
    claims_ctx,
    get_current_agent_id,
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_groups(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: Iterable[Any] = (raw,)
    elif isinstance(raw, Iterable):
        items = raw
    else:
        return ()
    out = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return tuple(dict.fromkeys(out))


def current_acl_context(
    *,
    acl_enabled: bool,
    purpose: str,
    source_graph: str | None = None,
    source_entity_id: str | None = None,
    visibility: str | None = None,
    owner_id: str | None = None,
) -> AclContext:
    claims = claims_ctx.get() or {}
    groups = _normalize_groups(
        claims.get("groups")
        or claims.get("group")
        or claims.get("roles")
        or claims.get("principal_groups")
    )
    return AclContext(
        principal_id=str(get_current_agent_id() or "system"),
        principal_groups=groups,
        security_scope=str(get_security_scope() or "").strip().lower() or None,
        acl_enabled=bool(acl_enabled),
        purpose=str(purpose or "user_visible"),
        source_graph=source_graph,
        source_entity_id=source_entity_id,
        visibility=visibility,
        owner_id=owner_id,
    )


__all__ = ["AclContext", "current_acl_context"]
