# Kogwistar 0.6.16

## PyPy 3.11 package compatibility

- Publish a pure-Python `py3-none-any` wheel with `Requires-Python >=3.11` for
  Python-authority PyPy 3.11 installs.
- Keep MCP available by default except on PyPy for Windows, where its
  `pywin32` dependency has no compatible wheel. Installing the explicit
  `[server]` extra still requests MCP and may fail on that platform.
- Keep `python-jose` available on PyPy for Windows but omit its optional
  `cryptography` extra, which has no compatible PyPy/Windows wheel. This
  excludes cryptography-dependent asymmetric JWT and OIDC operations there;
  HMAC JWT behavior remains available. Server/OIDC support on that target is
  not claimed.
- Keep MCP enabled in Linux PyPy CI and on CPython platforms.

## Verification

- Candidate-only. Do not tag or publish until this commit is merged and the
  PyPI release workflow passes, including the pinned PyPy 3.11.13/7.3.20
  wheel-build, dependency-install, and import smoke.
- The current local PyPy 3.11.17 Linux wheel/install smoke is supporting
  evidence only; the published PyPI install remains the release gate.
