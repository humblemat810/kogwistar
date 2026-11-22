# Kogwistar 0.7.2

## Rust agent surface

- Add the initial Rust-native `kogwistar-agent` composition surface.
- Preserve the thin-harness boundary: authority is injected at dispatch and the
  runtime remains the execution authority.
- Add focused tests for capability attenuation, binding validation, authority
  context, runtime delegation, budget seeding, control-point validation,
  delegated child state, provider lifecycle, and hook gating.

## Package support

- Keep `Requires-Python >=3.11`, CPython native wheels, and the PyPy 3.11
  Python-authority compatibility profile unchanged.
