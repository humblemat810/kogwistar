from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Mapping
from urllib.parse import urlencode
from typing import cast

import httpx
import jwt as pyjwt
from jwt import InvalidTokenError

from kogwistar.json_types import JsonValue


JsonObject = dict[str, JsonValue]


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError("OIDC response must be a JSON object")
    return {str(key): cast(JsonValue, item) for key, item in value.items()}


class OIDCClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        discovery_url: str,
        redirect_uri: str,
        *,
        issuer: str | None = None,
        scopes: list[str] | None = None,
        clock_skew_seconds: int = 60,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.discovery_url = discovery_url
        self.redirect_uri = redirect_uri
        self.issuer = issuer
        self.scopes = scopes or ["openid", "email", "profile"]
        self.clock_skew_seconds = clock_skew_seconds
        self._config: JsonObject | None = None
        self._jwk_client: pyjwt.PyJWKClient | None = None

    async def _ensure_config(self):
        if not self._config:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self.discovery_url)
                resp.raise_for_status()
                self._config = _json_object(resp.json())

    def _config_string(self, key: str) -> str:
        if self._config is None:
            raise RuntimeError("OIDC discovery configuration is unavailable")
        value = self._config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"OIDC discovery field {key!r} must be a non-empty string")
        return value

    async def _ensure_jwk_client(self):
        await self._ensure_config()
        assert self._config is not None
        if self._jwk_client is None:
            self._jwk_client = pyjwt.PyJWKClient(self._config_string("jwks_uri"))

    @staticmethod
    def generate_pkce() -> dict[str, str]:
        verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        return {"verifier": verifier, "challenge": challenge}

    @staticmethod
    def generate_state() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_nonce() -> str:
        return secrets.token_urlsafe(32)

    async def get_auth_url(self, state: str, code_challenge: str, nonce: str) -> str:
        await self._ensure_config()
        assert self._config is not None
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "redirect_uri": self.redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "nonce": nonce,
        }
        query = urlencode(params)
        return f"{self._config_string('authorization_endpoint')}?{query}"

    async def exchange_code(self, code: str, code_verifier: str) -> JsonObject:
        await self._ensure_config()
        assert self._config is not None
        data = {
            "client_id": self.client_id,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        async with httpx.AsyncClient() as client:
            resp = await client.post(self._config_string("token_endpoint"), data=data)
            resp.raise_for_status()
            return _json_object(resp.json())

    async def get_userinfo(self, access_token: str) -> JsonObject:
        await self._ensure_config()
        assert self._config is not None
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self._config_string("userinfo_endpoint"), headers=headers
            )
            resp.raise_for_status()
            return _json_object(resp.json())

    async def validate_id_token(self, id_token: str, *, nonce: str) -> JsonObject:
        await self._ensure_jwk_client()
        assert self._config is not None
        assert self._jwk_client is not None

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(id_token).key
            alg = pyjwt.get_unverified_header(id_token).get("alg")
            if not alg:
                raise ValueError("id_token missing alg")
            claims = pyjwt.decode(
                id_token,
                signing_key,
                algorithms=[alg],
                audience=self.client_id,
                issuer=self.issuer or self._config.get("issuer"),
                leeway=self.clock_skew_seconds,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except InvalidTokenError as exc:
            raise ValueError(f"Invalid id_token: {exc}") from exc

        token_nonce = claims.get("nonce")
        if not token_nonce:
            raise ValueError("id_token missing nonce")
        if token_nonce != nonce:
            raise ValueError("Invalid nonce")

        audiences = claims.get("aud")
        if isinstance(audiences, list) and len(audiences) > 1:
            azp = claims.get("azp")
            if azp != self.client_id:
                raise ValueError("id_token azp does not match client_id")

        return cast(JsonObject, claims)
