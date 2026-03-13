from __future__ import annotations
import json
import os
from sqlalchemy.orm import Session
from .repository import AuthRepository


def seed_auth_data(session: Session, seed_json: str | None = None):
    repo = AuthRepository(session)

    if not seed_json:
        # Check for seed data in env var
        seed_json = os.getenv("DEV_AUTH_SEED_JSON")
        if not seed_json:
            # Check for seed data in file
            seed_path = os.getenv("DEV_AUTH_SEED_PATH", "auth_seed.json")
            if os.path.exists(seed_path):
                with open(seed_path, "r") as f:
                    seed_json = f.read()

    if not seed_json:
        # Default seed for dev if nothing else provided
        seed_data = [
            {
                "user_id": "dev-user-id",
                "email": "dev@example.com",
                "display_name": "Dev User",
                "global_role": "rw",
                "global_ns": "docs,conversation,workflow,wisdom",
                "identities": [{"issuer": "dev", "subject": "dev"}],
            }
        ]
    else:
        try:
            seed_data = json.loads(seed_json)
        except Exception as e:
            print(f"Error parsing auth seed JSON: {e}")
            return

    for entry in seed_data:
        user_id = entry.get("user_id")
        email = entry.get("email")
        if not user_id or not email:
            continue

        repo.upsert_user(
            user_id=user_id,
            email=email,
            display_name=entry.get("display_name"),
            global_role=entry.get("global_role"),
            global_ns=entry.get("global_ns"),
        )

        # Seed identities
        for ident in entry.get("identities", []):
            issuer = ident.get("issuer")
            subject = ident.get("subject")
            if issuer and subject:
                existing = repo.get_external_identity(issuer, subject)
                if not existing:
                    repo.link_external_identity(user_id, issuer, subject, email)

        # Seed Workflow ACLs
        for acl in entry.get("workflow_acls", []):
            wf_id = acl.get("workflow_id")
            role = acl.get("role")
            if wf_id and role:
                repo.set_workflow_acl(wf_id, user_id, role)
