from __future__ import annotations
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import jwt
from sqlalchemy.orm import Session
from .repository import AuthRepository

class AuthService:
    def __init__(
        self,
        session: Session,
        jwt_secret: str,
        jwt_alg: str = "HS256",
        jwt_iss: Optional[str] = None,
        jwt_aud: Optional[str] = None
    ):
        self.repo = AuthRepository(session)
        self.jwt_secret = jwt_secret
        self.jwt_alg = jwt_alg
        self.jwt_iss = jwt_iss
        self.jwt_aud = jwt_aud

    def resolve_user_from_external(
        self,
        issuer: str,
        subject: str,
        email: str,
        display_name: Optional[str] = None,
        default_role: str = "ro",
        default_ns: str = "docs"
    ) -> str:
        # 1. Check if identity exists
        identity = self.repo.get_external_identity(issuer, subject)
        if identity:
            self.repo.update_last_login(identity.user_id)
            return identity.user_id

        # 2. Check if user exists by email (to link)
        user = self.repo.get_user_by_email(email)
        if not user:
            # 3. Create new user
            user_id = str(uuid.uuid4())
            user = self.repo.upsert_user(
                user_id, 
                email, 
                display_name,
                global_role=default_role,
                global_ns=default_ns
            )
        
        # 4. Link identity
        self.repo.link_external_identity(user.user_id, issuer, subject, email)
        self.repo.update_last_login(user.user_id)
        return user.user_id

    def mint_token(self, user_id: str, role: Optional[str] = None, ns: Any = None) -> str:
        user = self.repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Explicit token parameters should override persisted defaults.
        final_role = role if role is not None else (user.global_role or "ro")
        final_ns = ns if ns is not None else (user.global_ns or "docs")
        
        # Handle string list in global_ns (e.g. "docs,workflow")
        if isinstance(final_ns, str) and "," in final_ns:
            final_ns = [x.strip() for x in final_ns.split(",")]

        payload = {
            "sub": user.email,
            "user_id": user.user_id,
            "role": final_role,
            "ns": final_ns,
            "iat": int(time.time()),
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=4)).timestamp()),
            "iss": self.jwt_iss or "local",
            "aud": self.jwt_aud,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_alg)

    def check_workflow_access(self, workflow_id: str, user_id: str, required_role: str = "ro") -> bool:
        acl = self.repo.get_workflow_acl(workflow_id, user_id)
        if not acl:
            return False
        
        role_order = {"ro": 0, "rw": 1}
        return role_order.get(acl.role, 0) >= role_order.get(required_role, 0)
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self.repo.get_user(user_id)
        if not user:
            return None
        return {
            "user_id": user.user_id,
            "email": user.email,
            "display_name": user.display_name,
            "is_active": user.is_active,
        }
