# ADR-015 Library Release Target

## Current Release Snapshot (2026-09-28)

The fixed `0.2.5` target and evidence below are historical; they are not the
current package version or evidence for later source revisions. The current
`pyproject.toml` version is `0.6.1`.

- Tag `v0.6.1` points to `6066d1161d3520e61c33bf2c5b8dcb428e21a345` and is an
  annotated, unsigned tag, created by explicit maintainer choice.
- PyPI Release workflow `36417569358` passed version verification, all four
  wheel builds/smokes, sdist build, and artifact audit.
- At this snapshot, its `publish` job is waiting for approval in the `pypi`
  GitHub environment. PyPI publication is therefore **not yet verified**.

Do not claim `0.6.1` is published until that workflow succeeds and the PyPI
release metadata confirms it. See the [tagged release workflow](https://github.com/humblemat810/kogwistar/actions/runs/36417569358).

### Downstream dependency snapshot

This is a repository inspection snapshot, not a compatibility certification.
Update each downstream pin only after the release is installable and that
consumer's tests pass.

| Consumer | Observed dependency form | Status / next action |
| --- | --- | --- |
| `kogwistar-agent-suite` | `kogwistar>=0.6.1` | Constraint updated; installability still depends on publishing 0.6.1. |
| `kg_doc_parser` | PR #16 merged as `8b336e6`; dependency pins Kogwistar release commit `6066d11` | Rebased parser changes passed CI on CPython 3.12/3.13/3.14 and PyPy 3.11 before merge. Direct Git pin does not depend on PyPI publication. |
| `kogwistar-llm-wiki` | Local editable `kogwistar` path via `uv` | No published-version pin to bump. Verify nested checkout commit and run consumer tests when syncing core. |
| `kogwistar-agent` | Direct Git dependency pinned to commit `3a9dfb8` | Deliberate immutable source pin; upgrade only with downstream compatibility validation. |
| `kogwistar-obsidian-sink` | Direct Git dependency pinned to commit `ae24e16` | Deliberate immutable source pin; upgrade only with downstream compatibility validation. |
| `kogwistar-tokensafe` | `kogwistar>=0.5.0` | Range admits 0.6.1; no forced bump without compatibility evidence. |

The local `kg_doc_parser` checkout inspected on this machine is on a different,
stale feature branch and contains uncommitted changes. Its separate
`chore/kogwistar-0.6.1` worktree was also created from stale local `origin/main`;
neither represents the current GitHub `main` or merged PR #16. Preserve both;
PR #16 was rebased from a fresh checkout based on remote main before merge.

## Historical 0.2.5 Release Claim (Superseded)

The following claim describes the historical 0.2.5 candidate only. It does not
establish release readiness for 0.6.1 or any later version:

`Kogwistar 0.2.5` was release-ready as a **single-VM / bounded-workload Python
library distribution with a native Rust extension**. It did not claim that
Kogwistar operated a production service or that Rust was a downstream
deployment's default durable writer.

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

## Historical 0.2.5 Candidate Evidence

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

## Explicitly Outside the 0.2.5 Release Claim

- real customer traffic;
- HA, hyperscale, or fleet operations;
- downstream deployment canaries or a Rust default switch;
- marking PostgreSQL, runtime, or server capabilities
  `rust_cutover_ready: true`;
- automatic Git commit, push, GitHub CI dispatch, or PyPI publication.

Those are separate downstream deployment or maintainer operations. The
`rust_cutover_ready` flags stay false until an adopter supplies its own canary
evidence; this is a safe default, not an unchecked library-release item.

## Release Workflow and Maintainer Handoff

Before tagging, a maintainer reviews the diff, merges the release version, and
verifies CI for the merged commit. Do not use an older GitHub run as evidence
for a different source SHA.

Pushing a `v<version>` tag starts `.github/workflows/pypi-release.yml`. The
workflow verifies package/tag version and PyPI availability, builds and tests
platform wheels plus an sdist, audits the exact artifact set, then publishes
through OIDC Trusted Publishing after the `pypi` environment's required review.
It never requires or stores a PyPI API key. The workflow does not verify tag
cryptographic signatures; `v0.6.1` is annotated but unsigned.

Manual `workflow_dispatch` must target a matching `v<version>` tag. With
`publish=false`, it builds and audits without publishing; with `publish=true`, it
also publishes after environment approval. This differs from the normal tag-push
path, which proceeds to the protected publish job automatically.

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
