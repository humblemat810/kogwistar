# ADR-015 Library Release Target

## Release claim

`Kogwistar 0.2.5` is release-ready as a **single-VM / bounded-workload Python
library distribution with a native Rust extension**. This is the fixed ADR-015
library target. It is not a claim that Kogwistar operates a production service or
that Rust is a downstream deployment's default durable writer.

## Acceptance criteria

All criteria apply to one identified candidate wheel.

- A versioned, platform-tagged native wheel builds from the current source and
  passes package metadata validation.
- A clean Linux consumer installation imports the public package and native
  extension, and `pip check` passes.
- The deterministic consumer UAT proves public Python/Rust selection, Rust raw
  writer closure, rollback, and Rust -> Python -> Rust persisted SQLite
  compatibility through fresh processes.
- The four-layer Linux harness passes core, parser, sink, and reference
  application groups against the same candidate wheel.
- Current local capability/runtime/server gates pass without turning a local
  result into a downstream authority promotion.
- Public Python API and Python rollback selection remain available.

## Current candidate evidence

- source digest: `c74f0f0a47f8e09b05fe58019ab266243ab2d4ab694b427025fde0f1eac26412`
- wheel: `kogwistar-0.2.5-cp312-abi3-manylinux_2_34_x86_64.whl`
- wheel SHA-256: `1a2ec45b38be99c789fae9ddcf5aabcb7f1afc5354f4903c491c7c500b437a6f`
- build report: `.codex/wheelhouse-adr015-0.2.5-pyo3.29-local/build-report.json`
- VM consumer UAT: `.codex/wheelhouse-adr015-0.2.5-pyo3.29-local/adr015-consumer-uat-vm.json`
- four-layer report: `.codex/adr015-0.2.5-pyo3.29-local-feature.json`
- Phase 3/4/5 local gate reports:
  `.codex/adr015-phase3-capability-gate-current.json`,
  `.codex/adr015-phase4-runtime-gate-current.json`, and
  `.codex/adr015-phase5-server-gate-current.json`

## Explicitly outside this release claim

- real customer traffic;
- HA, hyperscale, or fleet operations;
- downstream deployment canaries or a Rust default switch;
- marking PostgreSQL, runtime, or server capabilities
  `rust_cutover_ready: true`;
- automatic Git commit, push, GitHub CI dispatch, or PyPI publication.

Those are separate downstream deployment or maintainer operations. The
`rust_cutover_ready` flags stay false until an adopter supplies its own canary
evidence; this is a safe default, not an unchecked library-release item.

## Maintainer handoff

Before publishing, a maintainer reviews the diff, creates a commit, pushes it,
and verifies GitHub CI for that new commit SHA. Publishing the wheel to PyPI is a
separate explicit action. Do not use an older GitHub run as evidence for a dirty
or uncommitted candidate.

`.github/workflows/pypi-release.yml` is the release path. Run it manually from
the signed `v<version>` tag with `publish=false` to build/download audited wheel
and source-distribution artifacts. Only rerun it with `publish=true` after review
and GitHub environment approval. The workflow verifies that the version is not
already on PyPI and publishes through OIDC Trusted Publishing; it never requires
or stores a PyPI API key.

Before first use, configure PyPI project's **Trusted Publishers** entry with the
GitHub owner `humblemat810`, repository `kogwistar`, workflow filename
`pypi-release.yml`, and GitHub environment `pypi`. Protect that environment with
required reviewers.

### Manual fallback

If OIDC publishing is unavailable, first run the same tagged workflow with
`publish=false` and download its audited artifacts. In an empty directory, verify
them again with `python -m twine check dist/*`, then run
`python -m twine upload dist/*`. Twine prompts for a project-scoped PyPI token;
use username `__token__` and enter the token only at that prompt. Never place a
token in the repository, workflow YAML, shell history, or artifact directory.

The source distribution uses the Maturin build backend, so a source install builds
the native extension instead of silently producing a pure-Python fallback. The
release workflow rebuilds the sdist through standard PEP 517 before publishing.
The manual path still requires the same version/tag/PyPI-availability checks as
the workflow.

## Related documents

- `ADR-015-incremental-rust-port.md`: migration and downstream authority policy.
- `ADR-015-implementation-status.md`: implementation and evidence status.
- `ADR-015-test-harness.md`: reproducible wheel, UAT, and four-layer commands.
- `ADR-015-release-review-0.2.5.md`: publication record and post-release hardening backlog.
