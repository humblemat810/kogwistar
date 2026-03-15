from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field


class OIDCProviderConfig(BaseModel):
    name: str = Field(..., description="Stable provider key, e.g. google or azure")
    discovery_url: str
    redirect_uri: str
    issuer: str | None = Field(
        default=None,
        description="Optional explicit issuer override; if omitted, use discovery issuer",
    )
    client_id: str
    client_secret: str = ""
    scopes: list[str] = Field(default_factory=lambda: ["openid", "email", "profile"])
    required_email: bool = True
    allowed: bool = True
    claim_map: dict[str, str] = Field(
        default_factory=lambda: {
            "subject": "sub",
            "email": "email",
            "name": "name",
        }
    )


def oidc_provider_config_skeleton() -> dict[str, Any]:
    return {
        "default_provider": "google",
        "providers": {
            "google": {
                "name": "google",
                "discovery_url": "https://accounts.google.com/.well-known/openid-configuration",
                "redirect_uri": "https://your-app.example.com/api/auth/callback",
                "issuer": "https://accounts.google.com",
                "client_id": "google-client-id",
                "client_secret": "google-client-secret",
                "scopes": ["openid", "email", "profile"],
                "required_email": True,
                "allowed": True,
            },
            "azure": {
                "name": "azure",
                "discovery_url": "https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration",
                "redirect_uri": "https://your-app.example.com/api/auth/callback",
                "issuer": "https://login.microsoftonline.com/<tenant-id>/v2.0",
                "client_id": "azure-client-id",
                "client_secret": "azure-client-secret",
                "scopes": ["openid", "email", "profile"],
                "required_email": True,
                "allowed": True,
            },
            "cognito": {
                "name": "cognito",
                "discovery_url": "https://cognito-idp.<region>.amazonaws.com/<user-pool-id>/.well-known/openid-configuration",
                "redirect_uri": "https://your-app.example.com/api/auth/callback",
                "issuer": "https://cognito-idp.<region>.amazonaws.com/<user-pool-id>",
                "client_id": "cognito-client-id",
                "client_secret": "cognito-client-secret",
                "scopes": ["openid", "email", "profile"],
                "required_email": True,
                "allowed": True,
            },
        },
    }


def load_oidc_provider_configs_from_env() -> tuple[str | None, dict[str, OIDCProviderConfig]]:
    raw = os.getenv("OIDC_PROVIDERS_JSON", "").strip()
    if not raw:
        return None, {}

    data = json.loads(raw)
    default_provider = data.get("default_provider")
    providers_raw = data.get("providers") or {}
    providers = {
        str(name): OIDCProviderConfig.model_validate({**cfg, "name": cfg.get("name", name)})
        for name, cfg in providers_raw.items()
    }
    if default_provider is None and len(providers) == 1:
        default_provider = next(iter(providers))
    if default_provider is not None and default_provider not in providers:
        raise ValueError(
            f"default_provider {default_provider!r} is not present in OIDC_PROVIDERS_JSON.providers"
        )
    return default_provider, providers
