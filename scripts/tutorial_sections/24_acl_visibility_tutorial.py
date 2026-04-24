# %% [markdown]
# # 24 ACL Visibility and Auditing
# Notebook-style companion for the ACL / visibility tutorial.
#
# This file shows one small auth-backed app, one protected read route, and one
# write route guarded by role scope.

# %%
import os

os.environ["JWT_ALG"] = "HS256"
os.environ["JWT_SECRET"] = "tutorial-acl-secret"
os.environ["JWT_ISS"] = "tutorial"

import jwt
from fastapi import FastAPI
from fastapi.testclient import TestClient

from _helpers import banner, show
from kogwistar.server.auth_middleware import (
    JWTProtectMiddleware,
    get_current_role,
    get_current_subject,
    require_role,
    set_auth_app,
)

# %%
banner("Build small auth app.")

app = FastAPI()
set_auth_app(app)
app.add_middleware(JWTProtectMiddleware)


@app.get("/probe")
def probe():
    return {
        "role": get_current_role(),
        "subject": get_current_subject(),
    }


@app.get("/write")
def write_probe():
    require_role("rw")
    return {
        "ok": True,
        "role": get_current_role(),
        "subject": get_current_subject(),
    }


# %%
def mint_token(*, role: str, sub: str) -> str:
    return jwt.encode(
        {"sub": sub, "role": role, "ns": "workflow", "iss": "tutorial"},
        "tutorial-acl-secret",
        algorithm="HS256",
    )


ro_token = mint_token(role="ro", sub="acl-reader")
rw_token = mint_token(role="rw", sub="acl-editor")

with TestClient(app) as client:
    ro_probe = client.get("/probe", headers={"Authorization": f"Bearer {ro_token}"})
    ro_write = client.get("/write", headers={"Authorization": f"Bearer {ro_token}"})
    rw_write = client.get("/write", headers={"Authorization": f"Bearer {rw_token}"})

show(
    "acl visibility result",
    {
        "ro_probe": ro_probe.json(),
        "ro_write_status": ro_write.status_code,
        "ro_write_body": ro_write.json(),
        "rw_write_status": rw_write.status_code,
        "rw_write_body": rw_write.json(),
    },
)

# %%
# Invariant:
# - read path sees current role and subject
# - write path denies ro token
# - write path allows rw token
