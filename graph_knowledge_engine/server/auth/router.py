from __future__ import annotations
import os
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from .service import AuthService
from .oidc import OIDCClient

router = APIRouter(prefix="/api/auth", tags=["auth"])

def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service

def get_oidc_client(request: Request) -> OIDCClient:
    oidc = getattr(request.app.state, "oidc_client", None)
    if oidc is None:
        raise HTTPException(status_code=400, detail="OIDC is disabled (AUTH_MODE=dev)")
    return oidc

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

    oidc = get_oidc_client(request)
    state = OIDCClient.generate_state()
    pkce = OIDCClient.generate_pkce()
    
    auth_url = await oidc.get_auth_url(state, pkce["challenge"])
    
    response = RedirectResponse(auth_url)
    response.set_cookie("auth_state", state, httponly=True, max_age=600)
    response.set_cookie("auth_pkce_verifier", pkce["verifier"], httponly=True, max_age=600)
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
        raise HTTPException(status_code=400, detail="OIDC callback disabled (AUTH_MODE=dev)")

    if error:
        raise HTTPException(status_code=400, detail=f"OIDC Error: {error} - {error_description}")
        
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    oidc = get_oidc_client(request)
    auth_service = get_auth_service(request)
    stored_state = request.cookies.get("auth_state")
    stored_verifier = request.cookies.get("auth_pkce_verifier")
    
    if not stored_state or state != stored_state:
        raise HTTPException(status_code=400, detail="Invalid state")
    if not stored_verifier:
        raise HTTPException(status_code=400, detail="Missing PKCE verifier")
        
    tokens = await oidc.exchange_code(code, stored_verifier)
    userinfo = await oidc.get_userinfo(tokens["access_token"])
    
    user_id = auth_service.resolve_user_from_external(
        issuer=oidc.discovery_url,
        subject=userinfo["sub"],
        email=userinfo["email"],
        display_name=userinfo.get("name")
    )
    
    app_token = auth_service.mint_token(user_id)
    
    ui_url = os.getenv("UI_URL", "/")
    response = RedirectResponse(f"{ui_url}?token={app_token}")
    response.delete_cookie("auth_state")
    response.delete_cookie("auth_pkce_verifier")
    return response

@router.get("/me")
async def me(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    claims = getattr(request.state, "claims", None)
    if not claims or "user_id" not in claims:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = auth_service.get_user(claims["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {
        **user,
        "role": claims.get("role"),
        "ns": claims.get("ns")
    }

@router.post("/logout")
async def logout():
    return {"ok": True}
