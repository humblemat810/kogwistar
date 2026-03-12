from __future__ import annotations

import asyncio
import contextvars
import os
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Callable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Scope

from graph_knowledge_engine.shortids import run_id_scope
if TYPE_CHECKING:
    from graph_knowledge_engine.server.auth.service import AuthService


load_dotenv()

# --- JWT config (env-driven) ---
JWT_ALG = os.getenv("JWT_ALG", "HS256")  # HS256 (shared secret) or RS256 (public key)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")  # HS256 secret OR RS256 public key
JWT_ISS = os.getenv("JWT_ISS")  # optional issuer to check
JWT_AUD = os.getenv("JWT_AUD")  # optional audience to check
PROTECTED_PREFIXES = tuple((os.getenv("JWT_PROTECTED_PATHS") or "/mcp,/admin").split(","))


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


# Context var to expose claims in any handler/tool
claims_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("claims", default=None)
current_role: ContextVar[str] = ContextVar("current_role", default=Role.RO.value)


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def verify_jwt(token: str) -> dict:
    """
    Validates a JWT and returns claims. Works for HS256 or RS256 depending on env.
    - HS256: set JWT_ALG=HS256 and JWT_SECRET=<shared secret>
    - RS256: set JWT_ALG=RS256 and JWT_SECRET=<PEM public key string>
    Optionally set JWT_ISS and/or JWT_AUD for issuer/audience checks.
    """
    try:
        options = {"verify_aud": bool(JWT_AUD)}
        claims = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALG],
            audience=JWT_AUD,
            issuer=JWT_ISS,
            options=options,
        )
        return claims
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def _decode_role_from_headers(scope: Scope) -> str:
    try:
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return Role.RO.value
        token = auth.split(" ", 1)[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        role = payload.get("role", Role.RO.value)
        return role if role in (Role.RO.value, Role.RW.value) else Role.RO.value
    except Exception:
        return Role.RO.value




class DevStreamGuardMiddleware:
    """Dev-only ASGI guard that downgrades expected stream disconnect noise."""

    def __init__(self, app):
        self.app = app
        self.enabled = str(os.getenv("DEV_STREAM_GUARD", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.auth_mode = str(os.getenv("AUTH_MODE", "")).strip().lower()
        raw_paths = os.getenv("DEV_STREAM_PATH_PREFIXES", "/api/runs/,/api/workflow/runs/")
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
        if isinstance(exc, (asyncio.CancelledError, BrokenPipeError, ConnectionResetError, OSError)):
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


class JWTProtectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # IMPORTANT: some /api endpoints enforce auth via require_role()/require_ns(),
        # which read from claims_ctx. So we must populate claims_ctx whenever a
        # bearer token is present, not only for PROTECTED_PREFIXES.
        if claims_ctx.get() is not None:
            return await call_next(request)

        path = request.url.path
        is_protected = any(path.startswith(p) for p in PROTECTED_PREFIXES)
        token = _extract_bearer(request)

        if not token:
            if is_protected:
                return JSONResponse({"detail": "Missing bearer token"}, status_code=401)
            return await call_next(request)

        try:
            claims = verify_jwt(token)
        except HTTPException as e:
            # If endpoint isn't globally protected, treat invalid token as anonymous.
            # Handlers that require auth will still reject via require_role/ns.
            if is_protected:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code)
            return await call_next(request)

        request.state.claims = claims
        token_ = claims_ctx.set(claims)
        try:
            with run_id_scope(token):
                return await call_next(request)
        finally:
            claims_ctx.reset(token_)


def get_current_role() -> str:
    claims = claims_ctx.get() or {}
    return (claims.get("role") or DEFAULT_ROLE).lower()


def require_role(min_role: str = "ro"):
    user_role = get_current_role()
    if ROLE_ORDER.get(user_role, 0) < ROLE_ORDER.get(min_role, 0):
        raise HTTPException(status_code=403, detail=f"Forbidden: requires role '{min_role}', you have '{user_role}'")


def get_current_namespaces() -> set[str]:
    claims = claims_ctx.get() or {}
    ns_val = claims.get("ns") or DEFAULT_NAMESPACE

    if isinstance(ns_val, list):
        raw_list = [str(x).lower() for x in ns_val]
    else:
        raw_list = [str(ns_val).lower()]

    allowed_values = {item.value for item in NameSpace} | {"*"}
    return {ns for ns in raw_list if ns in allowed_values}


def get_current_subject() -> str | None:
    claims = claims_ctx.get() or {}
    sub = str(claims.get("sub") or "").strip()
    return sub or None


def get_current_user_id() -> str | None:
    claims = claims_ctx.get() or {}
    return claims.get("user_id")


def _normalize_namespaces(expected: set[NameSpace] | NameSpace | set[str] | str) -> set[str]:
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

    raise HTTPException(status_code=403, detail=f"Forbidden: namespaces {actuals} do not permit access to {allowed}")


_auth_app = None


def set_auth_app(app) -> None:
    global _auth_app
    _auth_app = app


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
    "get_current_subject",
    "get_current_user_id",
    "require_namespace",
    "_normalize_namespaces",
    "set_auth_app",
    "require_workflow_access",
]
