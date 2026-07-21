# ADR-015 Library Release Target

## Release claim

`Kogwistar 0.2.4` is release-ready as a **single-VM / bounded-workload Python
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

- source digest: `6a5f99e1ce7aa6b1882aee891daf32ecd29e5fe20610354ab54ea5592f40a6a9`
- wheel: `kogwistar-0.2.4-cp312-abi3-manylinux_2_34_x86_64.whl`
- wheel SHA-256: `e08a55eeec499337df5edca5d943b78ab4f18251be7dbe0ce1ee67043140d247`
- build report: `.codex/wheelhouse-adr015-release-current/build-report.json`
- VM consumer UAT: `.codex/adr015-consumer-uat-current-vm.json`
- four-layer report: `.codex/adr015-release-current-feature.json`
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

## Related documents

- `ADR-015-incremental-rust-port.md`: migration and downstream authority policy.
- `ADR-015-implementation-status.md`: implementation and evidence status.
- `ADR-015-test-harness.md`: reproducible wheel, UAT, and four-layer commands.
