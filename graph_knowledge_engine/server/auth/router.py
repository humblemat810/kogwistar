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
    return request.app.state.oidc_client

@router.get("/login")
async def login(
    request: Request,
    oidc: OIDCClient = Depends(get_oidc_client)
):
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
    code: str,
    state: str,
    oidc: OIDCClient = Depends(get_oidc_client),
    auth_service: AuthService = Depends(get_auth_service)
):
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
