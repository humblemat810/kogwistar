from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from graph_knowledge_engine.server.auth_middleware import verify_jwt
from graph_knowledge_engine.server_mcp_with_admin import app as backend_app


@asynccontextmanager
async def wrapper_lifespan(app: FastAPI):
    async with backend_app.router.lifespan_context(backend_app):
        yield


app = FastAPI(title="Manual OIDC Test Wrapper", lifespan=wrapper_lifespan)

_LATEST_AUTH_RESULT: dict[str, Any] | None = None


@app.get("/__test__/auth/success", response_class=HTMLResponse)
async def test_auth_success(request: Request, token: str | None = None):
    global _LATEST_AUTH_RESULT

    if not token:
        return HTMLResponse(
            "<html><body><h1>Missing token</h1>"
            "<p>The OIDC callback did not provide a token query parameter.</p>"
            "</body></html>",
            status_code=400,
        )

    claims = verify_jwt(token)
    auth_service = backend_app.state.auth_service
    user = auth_service.get_user(claims["user_id"]) if auth_service else None
    ns_value = claims.get("ns")
    if isinstance(ns_value, list):
        ns_display = ", ".join(str(item) for item in ns_value)
    else:
        ns_display = str(ns_value)

    _LATEST_AUTH_RESULT = {
        "token": token,
        "claims": claims,
        "user": user,
        "email": (user or {}).get("email"),
        "role": claims.get("role"),
        "ns": claims.get("ns"),
    }

    html = f"""
    <html>
      <body>
        <h1>OIDC login complete</h1>
        <p>Backend callback succeeded and the test app captured the token.</p>
        <ul>
          <li><strong>Email:</strong> {(user or {}).get("email") or "unknown"}</li>
          <li><strong>Role:</strong> {claims.get("role") or "unknown"}</li>
          <li><strong>Namespaces:</strong> {ns_display or "unknown"}</li>
        </ul>
        <p>You can return to the terminal. The pytest check will continue.</p>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/__test__/auth/result")
async def test_auth_result():
    if _LATEST_AUTH_RESULT is None:
        return JSONResponse({"ready": False}, status_code=200)
    return JSONResponse({"ready": True, **_LATEST_AUTH_RESULT}, status_code=200)


app.mount("/", backend_app)
