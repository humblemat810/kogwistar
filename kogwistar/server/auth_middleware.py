from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(*args, **kwargs):
        return False


from fastapi import HTTPException
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.types import Receive, Scope, Send

from kogwistar.shortids import run_id_scope

if TYPE_CHECKING:
    pass


load_dotenv()
logger = logging.getLogger(__name__)
_auth_app = None


def _runtime_env_jwt_settings() -> dict[str, str | None]:
    settings = {
        "alg": os.getenv("JWT_ALG", "HS256"),
        "secret": os.getenv("JWT_SECRET"),
        "iss": os.getenv("JWT_ISS"),
        "aud": os.getenv("JWT_AUD"),
    }
    _auth_probe(
        "jwt_env_snapshot",
        **_secret_fingerprint(settings.get("secret")),
        alg=settings.get("alg"),
        iss=settings.get("iss"),
        aud=settings.get("aud"),
    )
    return settings


def _resolve_stateful_app(app):
    current = app
    seen: set[int] = set()
    for _ in range(8):
        if current is None:
            return None
        ident = id(current)
        if ident in seen:
            return None
        seen.add(ident)
        if getattr(current, "state", None) is not None:
            return current
        current = getattr(current, "app", None)
    return None


def get_app_jwt_settings(app=None) -> dict[str, str | None]:
    target = _resolve_stateful_app(app if app is not None else _auth_app)
    state = getattr(target, "state", None)
    if state is None:
        return _runtime_env_jwt_settings()
    settings = getattr(state, "auth_jwt_settings", None)
    if not isinstance(settings, dict):
        settings = _runtime_env_jwt_settings()
        state.auth_jwt_settings = settings
        _auth_probe(
            "jwt_app_settings_frozen",
            app_type=type(target).__name__ if target is not None else None,
            **_secret_fingerprint(settings.get("secret")),
            alg=settings.get("alg"),
            iss=settings.get("iss"),
            aud=settings.get("aud"),
        )
    return settings

# --- JWT config (env-driven) ---
def get_jwt_alg() -> str:
    return str(os.getenv("JWT_ALG", "HS256"))


def get_jwt_secret() -> str | None:
    return os.getenv("JWT_SECRET")


def get_jwt_iss() -> str | None:
    return os.getenv("JWT_ISS")


def get_jwt_aud() -> str | None:
    return os.getenv("JWT_AUD")


# Backward-compatible module exports. Do not use for runtime decisions.
JWT_ALG = get_jwt_alg()
JWT_SECRET = get_jwt_secret()
JWT_ISS = get_jwt_iss()
JWT_AUD = get_jwt_aud()
PROTECTED_PREFIXES = tuple(
    (os.getenv("JWT_PROTECTED_PATHS") or "/mcp,/admin").split(",")
)


class Role(str, Enum):
    RO = "ro"
    RW = "rw"


class NameSpace(str, Enum):
    DOCS = "docs"
    CONVERSATION = "conversation"
    WORKFLOW = "workflow"
    WISDOM = "wisdom"


# Simple two-role lattice
ROLE_ORDER = {"ro": 0, "rw": 1}  # read-only < read-write
DEFAULT_ROLE = "ro"
DEFAULT_NAMESPACE = NameSpace.DOCS.value
DEFAULT_STORAGE_NAMESPACE = "default"
DEFAULT_EXECUTION_NAMESPACE = "default"
DEFAULT_SECURITY_SCOPE = "public"
DEFAULT_CAPABILITIES: frozenset[str] = frozenset()


# Context var to expose claims in any handler/tool
_existing_claims_ctx = globals().get("claims_ctx")
if isinstance(_existing_claims_ctx, contextvars.ContextVar):
    claims_ctx = _existing_claims_ctx
else:
    claims_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
        "claims", default=None
    )

_existing_current_role = globals().get("current_role")
if isinstance(_existing_current_role, ContextVar):
    current_role = _existing_current_role
else:
    current_role: ContextVar[str] = ContextVar("current_role", default=Role.RO.value)


def _extract_bearer(scope: Scope) -> str | None:
    headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
    auth = headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def _auth_debug_probe_enabled() -> bool:
    return str(os.getenv("KOGWISTAR_AUTH_DEBUG_PROBE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _auth_debug_log_path() -> Path | None:
    raw = str(os.getenv("KOGWISTAR_AUTH_DEBUG_LOG", "")).strip()
    if not raw:
        return None
    return Path(raw)


def _append_auth_debug_log(record: dict) -> None:
    path = _auth_debug_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
    except Exception:  # noqa: BLE001
        logger.exception("auth probe write failed")


def _auth_probe(event: str, **extra) -> None:
    if not _auth_debug_probe_enabled():
        return
    payload = {
        "event": event,
        "pid": os.getpid(),
        "thread": threading.get_ident(),
        **extra,
    }
    try:
        stack = traceback.extract_stack(limit=12)
        payload["caller"] = {
            "file": stack[-2].filename if len(stack) >= 2 else None,
            "line": stack[-2].lineno if len(stack) >= 2 else None,
            "fn": stack[-2].name if len(stack) >= 2 else None,
        }
        payload["stack"] = [
            {
                "file": frame.filename,
                "line": frame.lineno,
                "fn": frame.name,
            }
            for frame in stack[-8:]
        ]
    except Exception:  # noqa: BLE001
        payload["caller"] = {"file": None, "line": None, "fn": None}
        payload["stack"] = ["<stack unavailable>"]
    _append_auth_debug_log(payload)
    logger.warning(json.dumps(payload, ensure_ascii=False, default=str))


def _secret_fingerprint(secret: str | None) -> dict[str, object]:
    if not secret:
        return {"secret_present": False}
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return {
        "secret_present": True,
        "secret_len": len(secret),
        "secret_fp": digest[:12],
    }


def set_claims_ctx(claims: dict | None):
    token = claims_ctx.set(claims)
    _auth_probe("claims_set", claims=claims)
    return token


def reset_claims_ctx(token) -> None:
    claims_ctx.reset(token)
    _auth_probe("claims_reset", claims=claims_ctx.get())


def set_current_role(role: str):
    token = current_role.set(role)
    _auth_probe("current_role_set", role=role)
    return token


def reset_current_role(token) -> None:
    current_role.reset(token)
    _auth_probe("current_role_reset", role=current_role.get())


def verify_jwt(token: str, jwt_settings: dict[str, str | None] | None = None) -> dict:
    """
    Validates a JWT and returns claims. Works for HS256 or RS256 depending on env.
    - HS256: set JWT_ALG=HS256 and JWT_SECRET=<shared secret>
    - RS256: set JWT_ALG=RS256 and JWT_SECRET=<PEM public key string>
    Optionally set JWT_ISS and/or JWT_AUD for issuer/audience checks.
    """
    settings = jwt_settings or _runtime_env_jwt_settings()
    jwt_alg = str(settings.get("alg") or "HS256")
    jwt_secret = settings.get("secret")
    jwt_iss = settings.get("iss")
    jwt_aud = settings.get("aud")
    if not jwt_secret:
        raise HTTPException(status_code=500, detail="JWT secret is not configured")
    try:
        _auth_probe("jwt_verify_secret", **_secret_fingerprint(jwt_secret))
        options = {"verify_aud": bool(jwt_aud)}
        claims = jwt.decode(
            token,
            jwt_secret,
            algorithms=[jwt_alg],
            audience=jwt_aud,
            issuer=jwt_iss,
            options=options,
        )
        return claims
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def _decode_role_from_headers(
    scope: Scope, jwt_settings: dict[str, str | None] | None = None
) -> str:
    try:
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return Role.RO.value
        token = auth.split(" ", 1)[1]
        settings = jwt_settings or _runtime_env_jwt_settings()
        jwt_secret = settings.get("secret")
        if not jwt_secret:
            return Role.RO.value
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=[str(settings.get("alg") or "HS256")],
        )
        role = payload.get("role", Role.RO.value)
        return role if role in (Role.RO.value, Role.RW.value) else Role.RO.value
    except Exception:
        return Role.RO.value


class DevStreamGuardMiddleware:
    """Dev-only ASGI guard that downgrades expected stream disconnect noise."""

    def __init__(self, app):
        self.app = app
        self.enabled = str(os.getenv("DEV_STREAM_GUARD", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.auth_mode = str(os.getenv("AUTH_MODE", "")).strip().lower()
        raw_paths = os.getenv(
            "DEV_STREAM_PATH_PREFIXES", "/api/runs/,/api/workflow/runs/"
        )
        self.path_prefixes = tuple(p.strip() for p in raw_paths.split(",") if p.strip())

    def _applies(self, scope: Scope) -> bool:
        if scope.get("type") != "http":
            return False
        if not self.enabled:
            return False
        if self.auth_mode not in {"", "dev"}:
            return False
        path = str(scope.get("path") or "")
        if "/events" not in path:
            return False
        return any(path.startswith(prefix) for prefix in self.path_prefixes)

    @staticmethod
    def _expected_disconnect(exc: BaseException) -> bool:
        if isinstance(
            exc,
            (asyncio.CancelledError, BrokenPipeError, ConnectionResetError, OSError),
        ):
            return True
        return isinstance(exc, RuntimeError) and str(exc) == "No response returned."

    async def __call__(self, scope: Scope, receive, send):
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        except BaseException as exc:  # noqa: BLE001
            if self._expected_disconnect(exc):
                return
            raise


class JWTProtectMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # IMPORTANT: some /api endpoints enforce auth via require_role()/require_ns(),
        # which read from claims_ctx. So we must populate claims_ctx whenever a
        # bearer token is present, not only for PROTECTED_PREFIXES.
        path = str(scope.get("path") or "")
        is_protected = any(path.startswith(p) for p in PROTECTED_PREFIXES)
        token = _extract_bearer(scope)
        _auth_probe(
            "jwt_request_seen",
            path=path,
            has_token=bool(token),
            token_len=len(token or ""),
            token_head=(token or "")[:12],
            token_tail=(token or "")[-12:] if token else "",
            is_protected=is_protected,
        )

        if not token:
            if claims_ctx.get() is not None:
                await self.app(scope, receive, send)
                return
            if is_protected:
                await JSONResponse({"detail": "Missing bearer token"}, status_code=401)(
                    scope, receive, send
                )
                return
            await self.app(scope, receive, send)
            return

        jwt_settings = get_app_jwt_settings(scope.get("app"))

        try:
            claims = verify_jwt(token, jwt_settings)
        except HTTPException as e:
            # If endpoint isn't globally protected, treat invalid token as anonymous.
            # Handlers that require auth will still reject via require_role/ns.
            _auth_probe(
                "jwt_invalid",
                path=path,
                is_protected=is_protected,
                status_code=e.status_code,
                detail=str(e.detail),
                token_len=len(token),
                token_head=token[:12],
                token_tail=token[-12:],
            )
            if is_protected:
                await JSONResponse({"detail": e.detail}, status_code=e.status_code)(
                    scope, receive, send
                )
                return
            await self.app(scope, receive, send)
            return

        scope.setdefault("state", {})["claims"] = claims
        token_ = set_claims_ctx(claims)
        try:
            _auth_probe(
                "jwt_claims_set",
                path=path,
                claims=claims,
                current_role=current_role.get(),
            )
            with run_id_scope(token):
                await self.app(scope, receive, send)
        finally:
            reset_claims_ctx(token_)


def get_current_role() -> str:
    claims = claims_ctx.get() or {}
    role = str(claims.get("role") or "").strip().lower()
    if role in ROLE_ORDER:
        return role
    current = str(current_role.get() or "").strip().lower()
    if current in ROLE_ORDER:
        return current
    return DEFAULT_ROLE


def require_role(min_role: str = "ro"):
    user_role = get_current_role()
    if ROLE_ORDER.get(user_role, 0) < ROLE_ORDER.get(min_role, 0):
        _auth_probe(
            "require_role_denied",
            min_role=min_role,
            user_role=user_role,
            current_role=current_role.get(),
            claims=claims_ctx.get() or {},
        )
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: requires role '{min_role}', you have '{user_role}'",
        )


def get_current_namespaces() -> set[str]:
    claims = claims_ctx.get() or {}
    ns_val = claims.get("ns") or DEFAULT_NAMESPACE

    if isinstance(ns_val, list):
        raw_list = [str(x).lower() for x in ns_val]
    else:
        raw_list = [str(ns_val).lower()]

    allowed_values = {item.value for item in NameSpace} | {"*"}
    return {ns for ns in raw_list if ns in allowed_values}


def get_storage_namespace() -> str:
    claims = claims_ctx.get() or {}
    scope = str(claims.get("storage_ns") or "").strip().lower()
    if scope:
        return scope
    ns = sorted(get_current_namespaces())
    return ns[0] if ns else DEFAULT_STORAGE_NAMESPACE


def get_execution_namespace() -> str:
    claims = claims_ctx.get() or {}
    scope = str(claims.get("execution_ns") or "").strip().lower()
    if scope:
        return scope
    ns = sorted(get_current_namespaces())
    return ns[0] if ns else DEFAULT_EXECUTION_NAMESPACE


def get_security_scope() -> str:
    claims = claims_ctx.get() or {}
    scope = str(
        claims.get("security_scope")
        or claims.get("tenant")
        or claims.get("scope")
        or ""
    ).strip().lower()
    if scope and scope != "*":
        return scope
    parts = get_security_scope_parts()
    if parts["path"]:
        return parts["path"]
    ns = sorted(get_current_namespaces())
    return ns[0] if ns else DEFAULT_SECURITY_SCOPE


def get_security_scope_parts() -> dict[str, str]:
    claims = claims_ctx.get() or {}
    tenant = str(claims.get("tenant") or "").strip().lower()
    workspace = str(claims.get("workspace") or "").strip().lower()
    project = str(claims.get("project") or "").strip().lower()
    explicit = str(claims.get("security_scope") or "").strip().lower()
    if explicit and explicit != "*":
        return {
            "tenant": tenant,
            "workspace": workspace,
            "project": project,
            "path": explicit,
        }
    parts = [part for part in (tenant, workspace, project) if part]
    return {
        "tenant": tenant,
        "workspace": workspace,
        "project": project,
        "path": "/".join(parts),
    }


def describe_storage_security_mapping() -> dict[str, str]:
    return {
        "storage_namespace": get_storage_namespace(),
        "execution_namespace": get_execution_namespace(),
        "security_scope": get_security_scope(),
        "security_scope_path": get_security_scope_parts()["path"],
        "tenant": get_security_scope_parts()["tenant"],
        "workspace": get_security_scope_parts()["workspace"],
        "project": get_security_scope_parts()["project"],
    }


def get_current_capabilities() -> set[str]:
    claims = claims_ctx.get() or {}
    caps = claims.get("capabilities") or claims.get("caps") or []
    if isinstance(caps, str):
        raw_items = [caps]
    else:
        raw_items = list(caps)
    out = {
        str(item).strip().lower()
        for item in raw_items
        if str(item).strip() and str(item).strip() != "*"
    }
    return out


def has_explicit_capabilities_claim() -> bool:
    claims = claims_ctx.get() or {}
    return "capabilities" in claims or "caps" in claims


def get_current_subject() -> str | None:
    claims = claims_ctx.get() or {}
    sub = str(claims.get("sub") or "").strip()
    return sub or None


def get_current_user_id() -> str | None:
    claims = claims_ctx.get() or {}
    return claims.get("user_id")


def get_current_agent_id() -> str | None:
    claims = claims_ctx.get() or {}
    agent_id = str(
        claims.get("agent_id")
        or claims.get("actor_id")
        or claims.get("sub")
        or claims.get("user_id")
        or ""
    ).strip()
    return agent_id or None


def _normalize_namespaces(
    expected: set[NameSpace] | NameSpace | set[str] | str,
) -> set[str]:
    if isinstance(expected, set):
        raw_items = list(expected)
    else:
        raw_items = [expected]
    if not raw_items:
        raise ValueError("At least 1 namespace has to be specified")
    allowed = set()
    valid = {item.value for item in NameSpace}
    for item in raw_items:
        value = str(item.value if isinstance(item, NameSpace) else item).lower()
        if value not in valid:
            raise ValueError(f"Unknown namespace: {item!r}")
        allowed.add(value)
    return allowed


def require_namespace(expected: set[NameSpace] | NameSpace | set[str] | str):
    allowed = _normalize_namespaces(expected)
    actuals = get_current_namespaces()

    if "*" in actuals:
        return "*"

    for a in actuals:
        if a in allowed:
            return a

    raise HTTPException(
        status_code=403,
        detail=f"Forbidden: namespaces {actuals} do not permit access to {allowed}",
    )


def require_security_scope(expected: set[str] | str):
    if isinstance(expected, set):
        allowed = {str(item).lower() for item in expected if str(item).strip()}
    else:
        allowed = {str(expected).lower()}
    actual = get_security_scope()
    if actual in allowed or "*" in allowed:
        return actual
    raise HTTPException(
        status_code=403,
        detail=f"Forbidden: security scope '{actual}' does not permit access to {allowed}",
    )


def can_access_security_scope(
    target_scope: str | None,
    *,
    shared: bool = False,
) -> bool:
    actual = get_security_scope()
    target = str(target_scope or "").strip().lower()
    if shared:
        return True
    if not target or target == "*":
        return True
    return actual == target


def require_security_scope_access(
    target_scope: str | None,
    *,
    shared: bool = False,
    action: str = "access",
) -> str:
    actual = get_security_scope()
    target = str(target_scope or "").strip().lower()
    if can_access_security_scope(target, shared=shared):
        return actual
    raise HTTPException(
        status_code=403,
        detail=(
            f"Forbidden: security scope '{actual}' cannot {action} target scope "
            f"'{target or '*'}' without explicit sharing"
        ),
    )


def require_capability(expected: set[str] | str):
    if isinstance(expected, set):
        allowed = {str(item).strip().lower() for item in expected if str(item).strip()}
    else:
        allowed = {str(expected).strip().lower()}
    actual = get_current_capabilities()
    if "*" in allowed or not allowed.isdisjoint(actual):
        return next(iter(actual & allowed), None) if actual else None
    raise HTTPException(
        status_code=403,
        detail=f"Forbidden: capabilities {sorted(actual)} do not permit access to {sorted(allowed)}",
    )


def set_auth_app(app) -> None:
    global _auth_app
    _auth_app = app
    get_app_jwt_settings(app)


def require_workflow_access(workflow_id: str, required_role: str = "ro"):
    user_id = get_current_user_id()
    if not user_id:
        # For dev-token users, we might want to bypass or have a default
        return

    if _auth_app is None:
        raise RuntimeError("Auth app not configured")

    # Injected AuthService
    auth_service = _auth_app.state.auth_service
    if not auth_service.check_workflow_access(workflow_id, user_id, required_role):
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: user {user_id} does not have {required_role} access to workflow {workflow_id}",
        )


__all__ = [
    "JWT_ALG",
    "JWT_SECRET",
    "JWT_ISS",
    "JWT_AUD",
    "get_jwt_alg",
    "get_jwt_secret",
    "get_jwt_iss",
    "get_jwt_aud",
    "PROTECTED_PREFIXES",
    "Role",
    "NameSpace",
    "ROLE_ORDER",
    "DEFAULT_ROLE",
    "DEFAULT_NAMESPACE",
    "claims_ctx",
    "current_role",
    "_decode_role_from_headers",
    "verify_jwt",
    "DevStreamGuardMiddleware",
    "JWTProtectMiddleware",
    "get_current_role",
    "require_role",
    "get_current_namespaces",
    "get_storage_namespace",
    "get_execution_namespace",
    "get_security_scope",
    "get_security_scope_parts",
    "describe_storage_security_mapping",
    "get_current_subject",
    "get_current_user_id",
    "get_current_capabilities",
    "has_explicit_capabilities_claim",
    "get_current_agent_id",
    "require_namespace",
    "require_security_scope",
    "can_access_security_scope",
    "require_security_scope_access",
    "require_capability",
    "_normalize_namespaces",
    "set_auth_app",
    "require_workflow_access",
]
