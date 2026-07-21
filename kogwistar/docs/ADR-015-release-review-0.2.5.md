# ADR-015 Release Review: Kogwistar 0.2.5

Status: post-release hardening backlog  
Published: 2026-07-22  
Release tag: `v0.2.5` -> `46d57847df36d91d2f6bb9c825bea079e0d66caa`

## Released evidence

`Kogwistar 0.2.5` was published to PyPI through GitHub OIDC Trusted
Publishing. The publish workflow run
`https://github.com/humblemat810/kogwistar/actions/runs/29846282814` completed
successfully after protected `pypi` environment approval.

Published, non-yanked files:

- `kogwistar-0.2.5-cp312-abi3-manylinux_2_34_x86_64.whl`
- `kogwistar-0.2.5-cp312-abi3-win_amd64.whl`
- `kogwistar-0.2.5-cp312-abi3-macosx_10_12_x86_64.whl`
- `kogwistar-0.2.5-cp312-abi3-macosx_11_0_arm64.whl`
- `kogwistar-0.2.5.tar.gz`

The release workflow rebuilt each platform wheel, rebuilt the source
distribution through standard PEP 517, ran native wheel smoke tests, and passed
combined artifact metadata audit. A clean CPython 3.13 Windows installation
from PyPI passed `pip check`, public package import, native `_rust` import, and
the `1.0.0` Rust contract smoke.

The ADR-015 library claim remains bounded: single VM / bounded workload,
fresh-process Rust -> Python -> Rust SQLite persistence compatibility, and the
four-layer Linux feature harness. It is not a claim of HA, live mixed-owner
SQLite operation, downstream Rust default authority, or customer production
traffic.

## P0: reconcile default branch with the released source

At publication time, the default branch was `main` at
`7b5517e47eec92c7b160178e07bf884178f2192f` and still declared package version
`0.2.4`. The released tag is based on `migration/rust` and declares `0.2.5`.
The released side is 25 commits ahead of `main`; `main` also has two commits
not present in the release tag.

Do not merge `migration/rust` into `main` blindly and do not rewrite either
history. Create a reviewed release-integration PR that explicitly resolves the
two histories. Before merge it must:

1. preserve `0.2.5` as the released package version;
2. include the current release workflow, including `macos-15-intel`, `macos-15`,
   and the isolated Python metadata-audit environment;
3. run CI against the proposed `main` result; and
4. state whether the two `main`-only commits are retained, superseded, or
   reconciled.

Until this is done, `main` is not a valid source reference for the published
0.2.5 release.

## P0: synchronize the default-branch release workflow

The tagged workflow was corrected before publication because GitHub retired the
`macos-13` Intel runner and deprecated `macos-14`. The old runner left a dry-run
job permanently queued. The tagged workflow now uses `macos-15-intel` for
Intel and `macos-15` for Apple Silicon.

The first audit job also exposed a host dependency drift: Twine 6.2 running
with Ubuntu's `packaging 24.0` rejected the valid `License-File` metadata field.
The same artifact passed with `packaging 26.2`. The release workflow now uses
`actions/setup-python@v5` with Python 3.12 and explicitly upgrades `pip`,
`twine`, and `packaging` in both audit and publish validation jobs.

Carry these exact workflow changes into the release-integration PR. A future
release started from `main` must not revive either failure.

## P1: harden cached Rust SQLite cross-owner visibility

### Observation

The first GitHub CI attempt for release commit `a5e6929` had one failure only
on CPython 3.12:

```text
test_rust_sqlite_then_python_initializes_reads_writes_aliases_and_cursors
replay cursor 2 exceeds latest event sequence 1
```

The sequence is Rust event append (sequence 1), Python append (sequence 2),
then Rust strict cursor advance to 2. The second CI attempt on the exact same
commit passed all Python 3.12, 3.13, and 3.14 jobs. An exact released-wheel
Linux CPython 3.12 same-process stress probe passed 500 repetitions. Therefore
the failure is not proven deterministic, but it is not dismissed or skipped.

The persistent Rust SQLite connection/cache was introduced after this
differential test. It is the primary investigation target. The cache currently
keys reuse by path and retains an open connection; a replaced file at the same
path and a committed external Python writer both require explicit freshness
semantics.

### Required hardening slice

1. Define and test the cache freshness invariant: when no Rust external
   transaction is active, a Rust operation following a committed Python writer
   must observe the newest committed SQLite state.
2. Evaluate `PRAGMA data_version` as the external-writer signal. If it changes
   while the cached handle is idle, refresh/reopen the Rust connection before
   the next operation. Do not refresh an explicitly owned Rust transaction.
3. Replace path-existence-only reuse with a safe file-replacement strategy, or
   explicitly evict an idle handle before a recreated database can be reused.
   The implementation must work on both POSIX unlink semantics and Windows file
   locking semantics.
4. Add deterministic Linux CPython 3.12 regression coverage for repeated
   Rust -> Python -> Rust handoff, cache/path recreation, and a deliberately
   retained cached handle. Run it in one process and in fresh processes.
5. Re-run the persistent SQLite performance benchmark. The cache hardening may
   not silently discard the recorded single-writer performance benefit; record
   the new latency, throughput, RSS, and state-digest evidence.

This is post-release hardening, not authorization to promote live mixed SQLite
ownership. The 0.2.5 library claim deliberately excludes it.

## P1: make tag creation one-way

`v0.2.5` was moved before publication while release-only workflow defects were
fixed. No GitHub Release or PyPI file existed at those moves, so publication
integrity was preserved; nevertheless, release tags should normally be
immutable.

Add a branch-dispatchable release preflight that exercises the same build,
platform-wheel, sdist, and metadata-audit path with publishing disabled. It
must not require a version tag. The final tag is then created once, after
preflight succeeds, and the tagged workflow is used only for final audited
publish.

## P2: persist release evidence outside local `.codex`

The final workflow artifacts are retained by GitHub, but ADR evidence currently
also names local `.codex` paths. Add a versioned, committed release manifest or
GitHub Release attachment index containing:

- tag commit and candidate-source digest;
- final publish workflow run URL;
- PyPI filenames and SHA-256 digests;
- four-layer and consumer-UAT report identities; and
- the bounded release claim and exclusions.

This keeps a maintainer-facing evidence index available after local temporary
directories are removed.

## P2: future platform and release presentation work

- Linux ARM64 and Windows ARM64 wheels were not published in 0.2.5. Add them
  only with native smoke and consumer-UAT evidence; do not imply support before
  those artifacts exist.
- Create a GitHub Release for `v0.2.5` with concise release notes and a link to
  the PyPI project and final workflow run.
- Keep PyPI Trusted Publishing OIDC-only. Do not add a PyPI API token to the
  repository, workflow, artifact, or shell history.

## Completion criteria for this review

This review is complete when the release-integration PR is merged and green,
the default-branch workflow matches the published workflow, the SQLite cache
hardening slice has deterministic regression and performance evidence, and a
versioned evidence manifest is available outside `.codex`.
