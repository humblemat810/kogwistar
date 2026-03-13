# 13 How to Test This Repo

Audience: Advanced / contributor
Time: 20-25 minutes

## What You Will Build

You will build a contributor mental model for how this repo defends behavior: invariant-driven tests, smoke scripts, parity checks, and replay assertions.

## Why This Matters

The project is broad enough that weak testing would make every architectural claim suspect. This page explains how contributors should add coverage without flattening the design.

## Run or Inspect

- Inspect `tests/core/test_tutorial_ladder_smoke.py` for script-backed tutorial checkpoints.
- Inspect `tests/core/test_tutorial_docs_integrity.py` for docs and learning-path integrity.
- Inspect parity and replay-oriented suites under `tests/kg_conversation/` and `tests/outbox/`.

## Inspect The Result

- Notice the preference for invariant assertions over brittle snapshot prose.
- Notice that backend differences are tested through parity harnesses, not ignored.
- Notice that replay, provenance, and context-snapshot behaviors have dedicated tests because they are architectural claims.

## Invariant Demonstrated

Tests are treated as preserved structure across changes. The repo uses tests to lock contracts, not just to chase temporary CI green.

## Next Tutorial

Continue to [14 Architecture Deep Dive](./14_architecture_deep_dive.md).
