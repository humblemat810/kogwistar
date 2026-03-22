# Architecture Requirements Document (ARD)
## Lightweight Test Suite Selection and Import Safety

**Status:** Draft  
**Date:** 2026-03-22  
**Owner:** Maintainers

---

## 1. Purpose

This document defines the lightweight test-suite path for the repository.

The goal is to make a fast, dependency-light pytest selection that can run without SQLAlchemy, FastAPI, websocket client libraries, Postgres containers, or other optional integration dependencies.

---

## 2. Problem Statement

The current test collection model can import heavy optional dependencies at module scope before marker filtering happens.

That creates two problems:

1. `pytest -m ...` cannot reliably select a light subset if some collected files import optional packages immediately.
2. Folder-based marker inference is too coarse to represent dependency strategy.

In particular, `core` currently does not mean "mocked and import-safe."

---

## 3. Target Taxonomy

The suite should separate scope from dependency strategy.

### Scope markers
- `unit`: pure logic with minimal fixture usage
- `integration`: multiple components working together
- `e2e`: full-path behavior
- `ci`: fast deterministic default suite
- `ci_full`: broader parity and system coverage

### Dependency-strategy markers
- `real`: requires real backend, database, server, container, or network-facing dependency

### Rule
- `ci` remains the lightweight deterministic selector as defined by the accepted CI-marking strategy.
- `real` is the heavy selector.
- `core` should not be used as a loose synonym for lightweight unless it is explicitly redefined.

---

## 4. Selection Policy

The lightweight command must select only tests that are safe without optional runtime dependencies.

Recommended command:

```bash
pytest -m ci
```

Optional broader lightweight command:

```bash
pytest -m "ci or unit"
```

The selection must exclude tests that depend on:

- `sqlalchemy`
- `fastapi`
- `websocket-client`
- Postgres containers
- provider-backed or network-backed integrations

---

## 5. Collection Policy

Marker selection alone is not sufficient if optional imports happen at module scope.

To preserve lightweight collection:

- Move optional imports inside fixtures or test bodies when feasible.
- Use `pytest.importorskip(...)` only when an entire file is genuinely optional.
- Split mixed files when a subset of tests is lightweight and another subset requires heavy dependencies.
- Avoid making `skip` inside a test body the primary defense against import-time failures.

The rule is simple:

- if the file is meant to be lightweight, it must be import-safe in a lightweight environment;
- if the file is heavy, it must be excluded from the lightweight marker set.

---

## 6. Repository Guidance

The repository already contains tests that validate optional-dependency boundaries.

This document formalizes the next step:

- keep the lightweight path aligned with `ci`
- keep heavy backend behavior under `real` / `ci_full`
- avoid broad directory inference for dependency strategy

The current `ci` / `ci_full` split should continue to describe execution scope, and the lightweight `ci` path should stay import-safe in environments that only install the fast dependencies.

---

## 7. Acceptance Criteria

- `pytest -m ci` runs without collection-time import errors in an environment that only has the lightweight dependency set.
- Heavy tests remain available under `ci_full` or `real`.
- Module-level optional imports no longer break the lightweight suite.
- The marker taxonomy is explicit enough that a test file is not promoted to lightweight status just because it sits in a core-looking directory.

---

## 8. Implementation Sequence

1. Keep the `ci` lightweight path explicit and import-safe.
2. Reclassify lightweight tests explicitly.
3. Remove or narrow folder-based inference for lightweight selection.
4. Convert heavy module-level imports to lazy imports or file-level `importorskip` where appropriate.
5. Verify `pytest -m ci` in a minimal environment.
