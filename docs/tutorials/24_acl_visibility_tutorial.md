# 24 ACL Visibility and Auditing

Audience: Security / platform developers  
Time: 15-20 minutes

Companion notebook: [`scripts/tutorial_sections/24_acl_visibility_tutorial.py`](../../scripts/tutorial_sections/24_acl_visibility_tutorial.py)

## What You Will Build

A tiny auth-backed app with one read route and one role-gated write route.

## What You Will Learn

This tutorial shows how Kogwistar separates authoritative truth from what different roles can see.
You will read the visibility model, inspect audit surfaces, and understand how ACL, namespaces, and repair views fit together.

## Why This Matters

ACL is not just deny/allow.
In this repo, ACL controls:

- which nodes and edges are visible
- which projections are safe to show
- which repairs need audit trail
- which operator views are read-only projections, not truth

Without that split, visibility leaks into execution and audit becomes ambiguous.

## Core Concepts

- `truth`: durable graph / event history
- `visibility`: what a subject may read from that truth
- `projection`: a view built from truth for a scope
- `audit`: immutable trace of allow / deny / revoke / repair
- `repair`: privileged correction path that still leaves evidence

## Read This First

- [Visibility / Viewing / Auditing](../visibility_viewing_auditing.md)
- [Leakage Prevention with Model Slicing](./16_leakage_prevention_with_model_slicing.md)
- [OAuth / ACL design notes](../ARD-0015-oauth.md)

## Run or Inspect

Run the notebook companion:

```bash
python scripts/tutorial_sections/24_acl_visibility_tutorial.py
```

## What To Inspect

Look for these surfaces:

- `read` view: safe inspection
- `operator` view: broader internal projection
- `audit` view: immutable event trail
- `repair` view: privileged corrective action trail

The important rule: views are projections, not second truth.

## Demo Shape

The repo does not hide ACL in one giant black box.
Instead, visibility is spread across:

- role and namespace checks
- LLM slicing and leakage prevention
- audit event records
- repair / replay flows
- operator projections

That means you can trace a denial or allow decision back to source scope and source action.

## Code Example

Use visibility docs first, then inspect the relevant surfaces:

```bash
python -m pydoc kogwistar.server.auth_middleware
```

```python
from kogwistar.server.auth_middleware import JWTProtectMiddleware, set_auth_app
from fastapi import FastAPI

app = FastAPI()
set_auth_app(app)
app.add_middleware(JWTProtectMiddleware)
```

The point is not the exact helper above. The point is:

- scope before read
- project before show
- audit every allow/deny/repair

## Inspect The Result

After you read the linked docs, confirm:

- denied access is visible as evidence
- repair actions still emit audit trail
- operator views do not mutate truth
- visibility rules stay deterministic across reads

## Invariant Demonstrated

ACL is a visibility layer over durable truth, not a replacement for truth.

## Next Tutorial

Continue with [25 Async Runtime Basics](./25_async_runtime_tutorial.md) or return to [Runtime Ladder Overview](./runtime-ladder-overview.md).
