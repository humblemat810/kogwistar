# Architecture Requirements Document (ARD)
## CI Marking Strategy for Fake vs Real Engine Coverage

**Status:** Accepted
**Date:** 2026-03-21
**Owner:** Maintainers

---

## 1. Purpose

This document defines the test-marking strategy for coverage that can run with either a fake engine path or a real engine path.

The goal is to keep pull-request CI fast, deterministic, and stable while preserving broader parity coverage in the full matrix.

---

## 2. Marking Policy

- `ci` is reserved for the cheapest deterministic path:
  - fake backend
  - fake or constant embedding
  - no external provider dependency
- `ci_full` is reserved for broader parity coverage:
  - real engine and real backend behavior
  - provider-backed or real embeddings
  - matrix coverage that is valuable but not required for PR CI
- When a test file covers both fake and real variants, only the fake variant should be marked `ci`.
- If the behavior is required in PR CI, keep or add a fake-backed path rather than promoting the real path into `ci`.
- Real-engine-only tests should not be marked `ci` just because the behavior matters.
- Do not assign both `ci` and `ci_full` to the same case.

---

## 3. Repository Rule

Use the existing pattern already present in the test suite as the reference:

- fake-backed parametrized cases get `pytest.mark.ci`
- real-backed parametrized cases stay unmarked or move to `pytest.mark.ci_full`
- mixed tests make the CI decision at the parameter level, not by inheriting a broad file-level `ci` mark

This rule applies especially to engine fixtures and backend matrices built from `backend_kind` and `embedding_kind`.

---

## 4. Guardrails

- New tests that depend on external providers should not silently enter `ci`.
- New tests that exercise real engine behavior should default to `ci_full` unless a fake-backed equivalent is provided.
- When adding a new backend matrix, the fake case must be the only PR-CI case unless a small deterministic slice is explicitly intended.
- If a test is meant to prove real-engine parity, it belongs in the broader matrix even when the LLM/provider calls are stubbed.

---

## 5. Acceptance Criteria

- `pytest -m ci` selects only fake-backed, deterministic coverage for mixed backend or embedding tests.
- `pytest -m ci_full` selects the real-engine and broader parity coverage.
- Existing and future mixed-matrix tests follow the same mark placement.
- No real-engine-only test is accidentally promoted into PR CI.

---

## 6. Assumptions

- `ci` means pull-request CI.
- `ci_full` means broader post-merge or full-matrix coverage.
- Deterministic fake coverage should be preferred whenever the same behavioral signal can be asserted without the real engine.
