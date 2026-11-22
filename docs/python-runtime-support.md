# Python Runtime Support

Kogwistar maintains separate runtime profiles because the native Rust path and
the Python-authority compatibility path have different interpreter contracts.

## Supported profiles

| Runtime | Support level | Scope |
| --- | --- | --- |
| CPython 3.12+ | Supported | Normal package installation, native Rust wheels, and the standard CI matrix |
| PyPy 3.11 | Supported compatibility profile | Provider-free CI suite with Python authorities and `KOGWISTAR_IMPL_MODE=python` |
| PyPy 3.12 | Experimental | Best-effort native-extension and ABI diagnostics; not a release gate |

## PyPy 3.11 checkout setup

Run these commands with a PyPy 3.11 interpreter:

```bash
pypy3.11 scripts/pypy311_profile.py --import-package
pypy3.11 scripts/run_pypy311_ci.py
```

The PyPy 3.11 profile intentionally keeps native extensions, NumPy, Chroma,
pgvector, Torch, and GPU extras outside its provider-free dependency set. The
profile uses the Python implementation authority and selects DiskCache for its
cache backend.

The profile dependencies are listed in
[`requirements-pypy-3.11-experimental.txt`](../requirements-pypy-3.11-experimental.txt).
The filename retains its historical `experimental` suffix for compatibility
with existing CI and local scripts; the PyPy 3.11 compatibility profile itself
is now supported.

## Published package metadata

PyPI `kogwistar==0.7.0` declares `Requires-Python >=3.11` and publishes a pure
Python wheel that installs on PyPy 3.11. The clean PyPy 3.11.13/7.3.20 install
and import smoke passed from the new environment's `site-packages`.

This package profile uses Python authorities and does not imply PyPy support
for native Rust extensions, MCP on Windows, or every optional server/auth
extra. The PyPy 3.11 dependency markers intentionally omit `mcp` and
`python-jose[cryptography]` on Windows.

## Verified downstream paths

The current downstream parser evidence uses `graph-knowledge-doc-parser==0.2.21`
and the published `kogwistar==0.7.2` package. These are scoped consumer checks,
not a claim of complete parser or backend parity:

| Path | Verified cell | Evidence boundary |
| --- | --- | --- |
| Python authority | Windows CPython 3.14.3; in-memory engines; deterministic fake parser | Ingestion succeeded and one exported node was readable from the knowledge engine. No external DB, vector backend, or live LLM. |
| Mixed | Windows CPython 3.14.6; Python parser/runtime with Rust SQLite metadata authority | Fake-backend ingestion passed; a separate deterministic Chroma smoke completed. This does not verify Rust graph authority. |
| Rust-native graph | Windows CPython 3.14.3 consumer; Rust HTTP service; PostgreSQL/pgvector 16 | One deterministic source produced one canonical node, then Rust visualization read it back. No multi-node/edge, live LLM, or vector-inference parity claim. |

Kogwistar 0.7.0 was published from tag `v0.7.0`. GitHub Actions release run
`43706093143` passed release verification, artifact audit, trusted publication,
and all wheel/source-distribution jobs. A fresh PyPy 3.11.13/7.3.20 virtual
environment installed `kogwistar==0.7.0` and 36 dependencies from PyPI;
`scripts/pypy311_pure_wheel_smoke.py --expected-version 0.7.0` confirmed the
installed module path, version, and `Node` import. This verifies package
metadata and the Python-authority compatibility profile only, not whole-stack
Rust/Python parity or the optional features excluded above.

PyPI `kogwistar==0.7.1` declares `Requires-Python >=3.11` and preserves the
Python-authority profile and PyPy compatibility constraints from `0.7.0`.
Kogwistar 0.7.1 was published from tag `v0.7.1` and adds the ADR-015
six-capability gate fixes.

PyPI `kogwistar==0.7.2` preserves the same runtime profiles and updates the
published downstream compatibility pin used by the current parser evidence.
Kogwistar 0.7.2 was published from tag `v0.7.2`.

