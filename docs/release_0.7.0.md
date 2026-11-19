# Kogwistar 0.7.0

## Dual-language runtime and graph capabilities

- Add the Rust-native runtime, graph API, and storage paths while retaining
  Python-authority and mixed Python/Rust execution modes.
- Define cross-language contracts and parity fixtures for graph models,
  workflow/runtime behavior, persistence, MCP/HTTP operations, and tracing.
- Add backend coverage for in-memory, SQLite-backed projections, Chroma vector
  projection, and PostgreSQL/pgvector, with backend-specific durability rules.
- Add benchmark and verification tooling for native, Python, and mixed-stack
  execution paths.

## PyPy 3.11 package compatibility

- Publish a pure-Python `py3-none-any` wheel with `Requires-Python >=3.11` for
  Python-authority PyPy 3.11 installs.
- Keep MCP available by default except on PyPy for Windows, where its
  `pywin32` dependency has no compatible wheel. Installing the explicit
  `[server]` extra still requests MCP and may fail on that platform.
- Keep `python-jose` available on PyPy for Windows but omit its optional
  `cryptography` extra, which has no compatible PyPy/Windows wheel. HMAC JWT
  behavior remains available; crypto-dependent asymmetric JWT and OIDC
  operations on that target are not claimed.
- Keep MCP enabled in Linux PyPy CI and on CPython platforms.

## Verification

- Published from tag `v0.7.0`. GitHub Actions release run
  [43706093143](https://github.com/humblemat810/kogwistar/actions/runs/43706093143)
  passed release verification, artifact audit, trusted publication, and all
  wheel/source-distribution jobs.
- A clean PyPy 3.11.13/7.3.20 environment installed `kogwistar==0.7.0` and
  passed the pure-wheel import smoke from `site-packages`. This verifies the
  Python-authority compatibility profile only; it does not claim native Rust
  extension support on PyPy or full Rust/Python behavior parity.

