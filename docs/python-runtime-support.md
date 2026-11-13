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

## Package metadata note

The current published package metadata declares `Requires-Python >=3.12`.
Therefore, PyPy 3.11 support is currently documented and tested from a source
checkout. Lowering the package requirement so that pip accepts PyPy 3.11 from
PyPI should be included in a future package release.

## Verified downstream paths

The current downstream parser evidence uses `graph-knowledge-doc-parser==0.2.17`
and the published `kogwistar==0.6.15` package. These are scoped consumer checks,
not a claim of complete parser or backend parity:

| Path | Verified cell | Evidence boundary |
| --- | --- | --- |
| Python authority | Windows CPython 3.14.3; in-memory engines; deterministic fake parser | Ingestion succeeded and one exported node was readable from the knowledge engine. No external DB, vector backend, or live LLM. |
| Mixed | Windows CPython 3.14.6; Python parser/runtime with Rust SQLite metadata authority | Fake-backend ingestion passed; a separate deterministic Chroma smoke completed. This does not verify Rust graph authority. |
| Rust-native graph | Windows CPython 3.14.3 consumer; Rust HTTP service; PostgreSQL/pgvector 16 | One deterministic source produced one canonical node, then Rust visualization read it back. No multi-node/edge, live LLM, or vector-inference parity claim. |

The 0.6.16 release note describes an unpublished candidate and its PyPy
dependency-marker changes. Do not treat that candidate as a PyPI installation
result until the release workflow publishes it and a clean PyPy 3.11.13/7.3.20
install/import check passes. Current published PyPI evidence remains 0.6.15.
