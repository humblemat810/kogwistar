# ADR-015 compatibility test harness

Updated: 2026-07-20

The persisted test harness is:

- `scripts/rust_port_compat.py`: suite selection, process groups, sharding,
  timing history, resume, and candidate reports.
- `scripts/rust_port_container_compat.py`: immutable source staging, cached test
  image construction, ordinary-container workers, cleanup, and report merge.
- `scripts/adr015_container.Dockerfile`: inspectable dual-venv image recipe.
- `scripts/adr015_container_worker.sh`: inspectable container entry point.
- `scripts/adr015_native_path.py`: inspectable native-extension locator used by
  container staging; no inline Python command is embedded in shell.
- `scripts/adr015_pytest_bootstrap.py`: inspectable import-provenance check and
  `pytest.main()` entry point.
- `scripts/adr015_candidate_identity.py`: inspectable candidate import,
  interpreter, ABI, native-extension, and environment identity probe.
- `scripts/adr015_source_identity.py`: shared candidate-source hash used by
  wheel construction and four-layer evidence; source drift fails closed.
- `scripts/rust_port_build_wheel.py`: host orchestrator for a current-source
  Linux candidate wheel.
- `scripts/adr015_build_host_native.py` and
  `scripts/adr015_native_extension_smoke.py`: host-native exact-Python-ABI
  builder and smoke gate for focused Windows/local tests.
- `scripts/adr015_wheel_builder.Dockerfile` and
  `scripts/adr015_build_wheel.sh`: inspectable pinned Rust/maturin build image
  and entry point; no inline build script is generated.
- `scripts/rust_port_test_compare.py`: identity- and coverage-checked serial
  versus parallel comparison.

No executable test logic is stored in `.codex` or generated temporary scripts.
No inline shell, Python, or Dockerfile is hidden inside orchestrator code.
`.codex` contains local raw reports only. Canonical aggregate evidence is in
`contracts/benchmarks/adr015-test-parallelism-current-windows.json`.

The host-native builder atomically refreshes the ignored source-tree extension,
writes source/wheel/extension digests, and verifies the `transaction_id` store
ABI before pytest. This prevents a stale local `.pyd` from impersonating current
Rust source during host gates.

The container recipe uses pinned `uv` for environment installation, with an
inspectable BuildKit package cache. This is required for future image rebuilds:
the 2026-07-20 current-candidate rebuild completed in 46.9 seconds with `uv`;
the prior equivalent `pip` rebuild took 632.6 seconds. This is a local cache and
network measurement, not a semantic performance claim.

## Profiles

- `feature`: `(ci or regression) and not slow`, excluding manual, real-LLM,
  Ollama, and legacy tests.
- `regression`: `regression and not slow` with the same external exclusions.
- `milestone`: explicit `ci or ci_full`, excluding `slow`, manual, and live-model
  coverage. Core and parser remain file-isolated for release evidence.

Real provider constructor and OCR/model-selection coverage in kg-doc-parser is
marked `llm_real`; it is outside ADR-015 contract CI until fully faked.

## Development cadence

Work in bounded semantic slices. Each slice has its focused regression gate
(including every applicable fake, Chroma, and PostgreSQL backend), plus static
checks. Accumulate a small independent batch—normally three to five slices—then
run Cargo/Python parity and the host `ci_full` selector. Build a Linux wheel and
run all four container layers only after the source candidate is frozen.

Count slices by independent semantics, never by changed files or tests. For
example, Rust DTO checks, the matching PyO3 boundary checks, and their golden
fixture are one contract-hardening slice; they do not justify a batch gate by
themselves.

An ABI, durable-store transaction, public API, or authority-boundary change
skips the batch allowance: run its proportional cross-boundary gate immediately.
This keeps iteration fast without allowing a focused green test to substitute
for a release candidate.

## Parallel policy

Default execution uses three ordinary Docker containers and zero pytest-xdist
workers. Existing pytest process boundaries remain the semantic unit:

- core feature/regression can use xdist explicitly, but controlled evidence
  found two workers slower than serial;
- parser always runs one file per process because module-global SQLite logging
  and provider state are not xdist-safe;
- sink runs one process;
- application keeps its 13 isolated groups and distributes them among ordinary
  containers.

Groups use deterministic longest-processing-time scheduling when timing history
is supplied. Every merged report verifies one candidate identity, one plan,
complete shard indexes, and exactly-once group coverage. Interrupted or failed
workers are cleaned up by container name. Each container receives a private
writable workspace and `/tmp`; source staging is read-only and includes current
tracked-dirty plus non-ignored untracked source. Secret `.env` is never copied.

The `Rust Port Compatibility` GitHub workflow builds one current Linux wheel,
then runs both `feature` and `regression` through this three-container harness.
Its release dispatch runs `milestone` through the same harness. `pytest-xdist`
is intentionally omitted from all three workflow paths: it remains an explicit
experiment only, because controlled core evidence measured it slower.

## Commands

Build current candidate first:

```powershell
.venv\Scripts\python.exe scripts\rust_port_build_wheel.py `
  --output .codex\wheelhouse-adr015-current
```

Fast four-layer validation:

```powershell
.venv\Scripts\python.exe scripts\rust_port_container_compat.py `
  --wheel <candidate.whl> --profile feature --shards 3 `
  --report .codex\adr015-feature-sharded.json
```

Regression-only validation:

```powershell
.venv\Scripts\python.exe scripts\rust_port_container_compat.py `
  --wheel <candidate.whl> --profile regression --shards 3 `
  --report .codex\adr015-regression-sharded.json
```

### Resume an interrupted Docker run

Reuse the exact prior command and report path, adding `--resume`:

```powershell
.venv\Scripts\python.exe scripts\rust_port_container_compat.py `
  --wheel <same-candidate.whl> --profile milestone --shards 3 `
  --resume --report .codex\adr015-milestone-sharded.json
```

Resume is fail-closed. The runner verifies candidate, source, ABI, capability
mode, and harness identity before reusing anything. It reruns only unfinished or
failed groups; previously successful groups are retained. Do not reuse a report
after changing source, rebuilding a different wheel, changing capability modes,
or editing the harness: start a new report instead. A merged `passed` report
also proves all shard indexes and exactly-once group coverage.

Explicit xdist experiment, not the default:

```powershell
.venv\Scripts\python.exe scripts\rust_port_container_compat.py `
  --wheel <candidate.whl> --profile feature --layer core --shards 1 `
  --pytest-workers 2 --report .codex\adr015-core-xdist.json
```

Release rehearsal keeps xdist off and uses container/file isolation:

```powershell
.venv\Scripts\python.exe scripts\rust_port_container_compat.py `
  --wheel <candidate.whl> --profile milestone --shards 3 `
  --report .codex\adr015-milestone-sharded.json
```

## Controlled results

Under the same candidate, harness, profile, and exact group coverage:

- core serial: 124.53 seconds;
- core xdist with two workers: 172.12 seconds, 0.724x speedup (a slowdown);
- application one container: 251.12 seconds;
- application three containers: 176.23 seconds, 1.425x speedup.

Therefore container sharding is enabled by default; xdist remains available but
opt-in. Historical four-layer serial reports took about 74 minutes. A sharded
four-layer feature run passed in about eight minutes, but dependency and harness
identity differed, so that figure is contextual rather than controlled evidence.
