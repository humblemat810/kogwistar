# Rust Build And Artifact-Size Guide

This guide applies to the Rust workspace in `kogwistar/rust/`. The workspace
contains multiple crates for contracts, engine, stores, runtime, API, and the
Python binding. Large local Cargo build directories are expected during active
development and do not, by themselves, indicate a bloated release binary or a
runtime memory leak.

## Why `target/` Gets Large

Cargo stores intermediate artifacts as well as final binaries. A workspace
build may retain:

| Artifact | Why it uses disk |
| --- | --- |
| `debug/deps/` | Compiled dependencies and crate libraries for development and tests |
| `debug/incremental/` | Compiler caches that make later development builds faster |
| `debug/build/` | Build-script outputs and generated/native artifacts |
| Test executables | Separate linked binaries for test targets and feature combinations |
| `release/` | A separate optimized set of dependencies and binaries |
| Other target directories | Separate copies for another target triple or `CARGO_TARGET_DIR` |

Debug symbols and incremental state are frequent large contributors. Building
multiple workspace crates, test targets, feature sets, or target triples adds
more artifacts. The checked-out tree may also contain both `rust/target/` and
`rust/target-chroma/`; these are ignored build output directories, and artifacts
in one are not reused by the other. Keep one target directory for ordinary
work unless a separate build is intentional.

A multi-gigabyte `target/` directory can therefore be normal. It is not the
size of the installed package or executable, and it does not imply equivalent
RAM use when the program runs. There is no fixed "normal" size: the number of
crates, test targets, enabled features, toolchains, target triples, and profiles
all affect it. Investigate if disk usage grows unexpectedly across repeated
builds, duplicate target directories accumulate, or the filesystem is under
pressure.

## Inspect Usage (PowerShell)

Run from the LLM-Wiki repository root. This reports Rust-local directories
named `target*` (including any intentionally isolated benchmark targets):

```powershell
Get-ChildItem .\kogwistar\rust -Directory |
  Where-Object { $_.Name -like 'target*' } |
  ForEach-Object {
    $bytes = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
      Measure-Object -Property Length -Sum).Sum
    [pscustomobject]@{
      Directory = $_.FullName
      GiB = [math]::Round(($bytes / 1GB), 2)
    }
  } | Sort-Object GiB -Descending
```

Inspect the main Cargo output by profile (`debug`, `release`, and any custom
profiles):

```powershell
Get-ChildItem .\kogwistar\rust\target -Directory |
  ForEach-Object {
    $bytes = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
      Measure-Object -Property Length -Sum).Sum
    [pscustomobject]@{ Profile = $_.Name; GiB = [math]::Round(($bytes / 1GB), 2) }
  } | Sort-Object GiB -Descending
```

If another process sets `CARGO_TARGET_DIR`, inspect that path too. Different
target directories intentionally duplicate compiled dependencies; do not assume
the active Cargo command is using `rust/target/` without checking its
environment. In PowerShell, inspect the current process value with:

```powershell
$env:CARGO_TARGET_DIR
```

Cargo configuration can also redirect output. Inspect workspace and user Cargo
configuration if the observed path is unexpected:

```powershell
Get-Content .\kogwistar\rust\.cargo\config.toml -ErrorAction SilentlyContinue
Get-Content "$env:USERPROFILE\.cargo\config.toml" -ErrorAction SilentlyContinue
```

On Linux or macOS, inspect sizes without deleting anything:

```sh
du -sh kogwistar/rust/target* 2>/dev/null
du -h --max-depth=2 kogwistar/rust/target 2>/dev/null | sort -h | tail -25
```

## Build And Test From The Workspace

The workspace manifest pins Rust 1.97.1 through `rust-toolchain.toml`. Run Cargo
commands from `kogwistar/rust/`, or pass the manifest path explicitly:

```powershell
Set-Location .\kogwistar\rust
cargo check --workspace --locked
cargo test --workspace --all-targets --locked
cargo build --release --workspace --locked
```

Prefer `cargo check` while iterating on type-level changes, then run focused
crate or test targets before the full workspace suite. Use the same target
directory across runs to benefit from incremental compilation. Do not routinely
run `cargo clean` before each build; it discards useful caches and makes the
next build slower.

## Reduce Disk Use Safely

First identify which profile or target directory is large. Then choose the
narrowest cleanup that meets the need:

```powershell
# From kogwistar/rust: remove only development-profile artifacts.
cargo clean --profile dev

# Remove all artifacts for this workspace (the next build recompiles them).
cargo clean
```

Before removing a separate directory such as `rust/target-chroma/`, confirm no
active process is building into it and that it contains only generated Cargo
artifacts. Never remove source directories or a target directory whose purpose
is unknown. The repository ignores `/rust/target*`, but ignore rules do not
identify whether a directory is currently in use.

## Optional Smaller Development/Test Artifacts

Cargo's default development and test profiles retain substantial debug
information and use incremental compilation. If disk capacity is more
important than full debugger detail, profile overrides can reduce debug data.
For a local-only experiment, add this to `rust/Cargo.toml`:

```toml
[profile.dev]
debug = 1
incremental = true

[profile.test]
debug = 1
incremental = true
```

`debug = 1` retains line-table information but omits much of the full debug
data. Debuggers may provide less local-variable/value inspection. Keep
incremental compilation enabled initially so normal edit-build cycles remain
responsive. Measure disk use and representative rebuild time before deciding
whether to keep an override; do not add it globally just because a target
directory is large. Turning off incremental compilation or all debug
information can save more space but may make rebuilds slower and debugging less
useful. Prefer a developer-local Cargo config or command-specific profile when
the change is only for one machine; commit a workspace-wide profile change only
when the team has measured and accepted the trade-off.

## Release And CI Builds

Release artifacts are separate from development/test artifacts and are
optimized for execution, not fast compilation. CI and release builds should use
the pinned toolchain, locked dependencies, and the repository's existing
workflow/profile choices. Do not change shared Cargo profiles solely to reduce
one developer's local disk use. If CI cache size or build duration is the
problem, measure cache keys, target triples, features, and profiles separately
before changing cache policy.

## Interpreting Large Directories

- Large `debug/incremental/`: compiler cache; deleting it trades space for
  slower subsequent builds.
- Large `debug/deps/`: dependency/test artifacts; multiple features, test
  binaries, and toolchain/target variants can duplicate these.
- Large `release/`: optimized artifacts; it is distinct from `debug/`.
- Several `target*` directories: isolated/custom build outputs; verify their
  owners, active processes, and `CARGO_TARGET_DIR` usage before cleanup. A
  directory ignored by Git may still be in active use.

The practical default for this workspace is: keep the pinned toolchain, reuse a
single target directory for ordinary builds, retain incremental compilation,
inspect before cleaning, and use reduced debug information only when disk
pressure justifies the debugging trade-off. A large `target/` is a build-cache
and artifact-management issue; it is not, by itself, evidence of an oversized
runtime, a memory leak, or a broken Rust build.
