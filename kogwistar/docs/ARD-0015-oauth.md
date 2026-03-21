
# Architecture Requirement Document (ARD)
## Real Authentication for Workflow Designer and MCP/REST Server

### 1. Goal
Add real authentication and authorization for the workflow designer and MCP/REST server while keeping the current architecture intact.

Key constraints:
- Python server remains the trust boundary
- Vite remains a UI/dev proxy only
- Must work with SQLite + Chroma today
- Must remain portable to PostgreSQL later
- Reuse existing JWT/RBAC seams instead of replacing them

---

### 2. Non‑Goals
The following are explicitly out of scope:

- Event sourcing auth/session state
- Migrating the entire engine to an ORM
- Making Vite an auth authority
- Requiring PostgreSQL for auth
- Redesigning workflow history or runtime state

Auth/session state should remain simple mutable relational records.

---

### 3. Current System Reality

The server already contains:

- JWT middleware
- role + namespace checks
- bearer token authorization
- `/auth/dev-token` endpoint for development tokens
- `/designer/capabilities` endpoint used by the workflow UI

Storage modes already supported:

- SQLite + Chroma
- PostgreSQL + pgvector

Docker compose already supports optional runtime stacks via profiles.

---

### 4. Target Architecture

Browser (React Flow UI)
        |
        | Login Redirect
        v
Python Backend (Auth + Authorization Boundary)
        |
        | OIDC Authorization Code + PKCE
        v
OIDC Provider (Keycloak/Auth0/Azure/etc)

Responsibilities:

OIDC provider
- authenticates user

Python backend
- validates login result
- maps identity to internal user
- issues application JWT
- enforces RBAC

Frontend
- UI only
- sends bearer token to backend

---

### 5. Auth Mode

Use:

OIDC Authorization Code + PKCE

But retain:

`/auth/dev-token`

for development fallback.

This allows the frontend and authorization layer to be built before the OIDC provider is integrated.

---

### 6. Token Model

The server already expects Bearer JWT tokens, so keep that model.

Flow:

1. User logs in through OIDC
2. Python receives callback
3. Python maps external identity
4. Python issues application JWT
5. Frontend sends Authorization header

Authorization: Bearer <token>

This integrates cleanly with the existing middleware.

---

### 7. Storage Design

Use SQLAlchemy ORM only for auth metadata.

Supported databases:
- SQLite
- PostgreSQL

Tables:

users
- user_id
- email
- display_name
- is_active
- created_at
- last_login_at

external_identities
- issuer
- subject
- user_id
- email

Composite key:
(issuer, subject)

workflow_acl
- workflow_id
- user_id
- role

Composite key:
(workflow_id, user_id)

Optional:

auth_audit_events
Append-only security audit log.

---

### 8. API Endpoints

New endpoints:

GET  /api/auth/login  
GET  /api/auth/callback  
GET  /api/auth/me  
POST /api/auth/logout  

/api/auth/login
- generate PKCE verifier
- generate state
- redirect to provider

/api/auth/callback
- verify state
- exchange authorization code
- validate identity
- create internal user mapping
- mint application JWT

/api/auth/me
Return current user information.

Example:

{
  "user_id": "u123",
  "email": "user@example.com",
  "display_name": "Example User",
  "role": "rw"
}

---

### 9. Authorization Rules

Claims in application JWT:
- sub
- user_id
- role
- namespace access

Role model:

ro → read designer/runtime  
rw → mutate workflows / run executions  

Authorization checks must be enforced in:

- workflow design APIs
- workflow run APIs
- MCP tool execution
- designer capability endpoint

---

### 10. Frontend Flow

Startup sequence:

App Start  
→ GET /api/auth/me  

401 → show login button  

200 → load:
- /designer/capabilities
- /api/workflows

Frontend must never trust client‑side identity.

Identity comes exclusively from backend JWT validation.

---

### 11. Python Module Layout

Recommended structure:

server/
  auth/
    models.py
    db.py
    repository.py
    service.py
    oidc.py
    router.py

Responsibilities:

models.py  
SQLAlchemy ORM models

db.py  
engine + session factory

repository.py  
database queries

service.py  
user/session/ACL logic

oidc.py  
OIDC login + PKCE flow

router.py  
FastAPI endpoints

---

### 12. Integration Points

Key files to modify:

- server_mcp_with_admin.py
- runtime_api.py
- bootstrap.py

Important change:

runtime APIs must stop trusting client supplied user_id.

User identity must come from JWT claims.

---

### 13. Phased Implementation

Phase 0
Prep work

- SQLAlchemy auth tables
- /api/auth/me
- keep /auth/dev-token

Phase 1
Internal authorization

- workflow ACL
- derive user from JWT
- enforce access checks

Phase 2
OIDC login

- /api/auth/login
- /api/auth/callback
- PKCE support

Phase 3
Dev OIDC container

Add docker compose profile:
auth

---

### 14. Acceptance Criteria

The system is complete when:

- dev-token flow still works
- /api/auth/me works
- unauthorized workflow access is blocked
- workflow runs cannot spoof user_id
- auth tables work on SQLite
- auth tables work on PostgreSQL
- OIDC login can mint application JWT

---

### 15. Summary

Implement provider‑agnostic OIDC authentication terminating in the Python backend, issuing application JWT tokens compatible with the existing RBAC system while storing minimal user and ACL metadata via SQLAlchemy that works on both SQLite and PostgreSQL.
