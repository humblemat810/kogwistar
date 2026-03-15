from __future__ import annotations
import os
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from .service import AuthService
from .oidc import OIDCClient

router = APIRouter(prefix="/api/auth", tags=["auth"])


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def _get_oidc_clients(request: Request) -> dict[str, OIDCClient]:
    return getattr(request.app.state, "oidc_clients", None) or {}


def _get_default_provider(request: Request) -> str | None:
    return getattr(request.app.state, "oidc_default_provider", None)


def _resolve_provider_name(request: Request, provider: str | None = None) -> str:
    clients = _get_oidc_clients(request)
    provider_name = provider or _get_default_provider(request)
    if not provider_name:
        raise HTTPException(status_code=400, detail="OIDC provider is not configured")
    if provider_name not in clients:
        raise HTTPException(
            status_code=400,
            detail=f"OIDC provider {provider_name!r} is not configured",
        )
    return provider_name


def _get_oidc_client_for_provider(
    request: Request, provider: str | None = None
) -> tuple[str, OIDCClient]:
    provider_name = _resolve_provider_name(request, provider)
    oidc = _get_oidc_clients(request).get(provider_name)
    if oidc is None:
        raise HTTPException(
            status_code=400,
            detail=f"OIDC provider {provider_name!r} is not configured",
        )
    return provider_name, oidc


def _get_auth_mode(request: Request) -> str:
    return (getattr(request.app.state, "auth_mode", None) or "oidc").lower()


def _mint_dev_token(auth_service: AuthService) -> str:
    email = os.getenv("DEV_AUTH_EMAIL", "dev@example.com")
    subject = os.getenv("DEV_AUTH_SUBJECT", "dev")
    display_name = os.getenv("DEV_AUTH_NAME", "Dev User")
    role = os.getenv("DEV_AUTH_ROLE", "ro")
    ns_raw = os.getenv("DEV_AUTH_NS")
    if ns_raw:
        ns = ns_raw.split(",") if "," in ns_raw else ns_raw
    else:
        # Default to all namespaces in dev mode to satisfy all apps/tests
        ns = ["docs", "conversation", "workflow", "wisdom"]
    user_id = auth_service.resolve_user_from_external(
        issuer="dev",
        subject=subject,
        email=email,
        display_name=display_name,
    )
    return auth_service.mint_token(user_id, role=role, ns=ns)


@router.get("/login")
async def login(
    request: Request,
    redirect_uri: str | None = None,
    provider: str | None = None,
):
    auth_mode = _get_auth_mode(request)

    # redirect_uri override is a dev-only convenience — reject it in prod
    if redirect_uri and auth_mode != "dev":
        raise HTTPException(
            status_code=400,
            detail="redirect_uri override is only allowed when AUTH_MODE=dev",
        )

    if auth_mode == "dev":
        auth_service = get_auth_service(request)
        ui_url = redirect_uri or os.getenv("UI_URL", "/")
        token = _mint_dev_token(auth_service)
        return RedirectResponse(f"{ui_url}?token={token}")

    provider_name, oidc = _get_oidc_client_for_provider(request, provider)
    state = OIDCClient.generate_state()
    pkce = OIDCClient.generate_pkce()
    nonce = OIDCClient.generate_nonce()

    auth_url = await oidc.get_auth_url(state, pkce["challenge"], nonce)

    response = RedirectResponse(auth_url)
    response.set_cookie("auth_state", state, httponly=True, max_age=600)
    response.set_cookie(
        "auth_pkce_verifier", pkce["verifier"], httponly=True, max_age=600
    )
    response.set_cookie("auth_nonce", nonce, httponly=True, max_age=600)
    response.set_cookie("auth_provider", provider_name, httponly=True, max_age=600)
    return response


@router.get("/callback")
async def callback(
    request: Request,
    state: str,
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    auth_mode = _get_auth_mode(request)
    if auth_mode == "dev":
        raise HTTPException(
            status_code=400, detail="OIDC callback disabled (AUTH_MODE=dev)"
        )

    if error:
        raise HTTPException(
            status_code=400, detail=f"OIDC Error: {error} - {error_description}"
        )

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    auth_service = get_auth_service(request)
    stored_state = request.cookies.get("auth_state")
    stored_verifier = request.cookies.get("auth_pkce_verifier")
    stored_nonce = request.cookies.get("auth_nonce")
    stored_provider = request.cookies.get("auth_provider")

    if not stored_state or state != stored_state:
        raise HTTPException(status_code=400, detail="Invalid state")
    if not stored_verifier:
        raise HTTPException(status_code=400, detail="Missing PKCE verifier")
    if not stored_nonce:
        raise HTTPException(status_code=400, detail="Missing nonce")
    if not stored_provider:
        raise HTTPException(status_code=400, detail="Missing provider")

    _, oidc = _get_oidc_client_for_provider(request, stored_provider)

    tokens = await oidc.exchange_code(code, stored_verifier)
    id_token = tokens.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Missing id_token")

    try:
        id_claims = await oidc.validate_id_token(id_token, nonce=stored_nonce)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    userinfo = await oidc.get_userinfo(tokens["access_token"])
    if userinfo.get("sub") != id_claims["sub"]:
        raise HTTPException(
            status_code=401,
            detail="userinfo subject does not match validated id_token subject",
        )

    id_email = id_claims.get("email")
    userinfo_email = userinfo.get("email")
    if id_email and userinfo_email and id_email != userinfo_email:
        raise HTTPException(
            status_code=401,
            detail="userinfo email does not match validated id_token email",
        )
    email = id_email or userinfo_email
    if not email:
        raise HTTPException(status_code=400, detail="OIDC identity missing email")
    display_name = id_claims.get("name") or userinfo.get("name")

    user_id = auth_service.resolve_user_from_external(
        issuer=id_claims.get("iss") or oidc.discovery_url,
        subject=id_claims["sub"],
        email=email,
        display_name=display_name,
    )

    app_token = auth_service.mint_token(user_id)

    ui_url = os.getenv("UI_URL", "/")
    response = RedirectResponse(f"{ui_url}?token={app_token}")
    response.delete_cookie("auth_state")
    response.delete_cookie("auth_pkce_verifier")
    response.delete_cookie("auth_nonce")
    response.delete_cookie("auth_provider")
    return response


@router.get("/me")
async def me(request: Request, auth_service: AuthService = Depends(get_auth_service)):
    claims = getattr(request.state, "claims", None)
    if not claims or "user_id" not in claims:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = auth_service.get_user(claims["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {**user, "role": claims.get("role"), "ns": claims.get("ns")}


@router.post("/logout")
async def logout():
    return {"ok": True}
