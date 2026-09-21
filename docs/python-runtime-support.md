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
