from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable, Final

try:
    # Direct script execution places this file's directory on sys.path.
    # The package fallback keeps importlib-based test loading supported.
    from adr015_source_identity import candidate_source_fingerprint  # type: ignore[import-not-found]
except ModuleNotFoundError:
    from scripts.adr015_source_identity import candidate_source_fingerprint


ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST_PATH: Final = ROOT / "contracts" / "rust-port-v1.json"
_SERVER_CONFIGURATION: Final = "KOGWISTAR_IMPL_SERVER"
_IGNORED_TEST_TREE_NAMES: Final = {"__pycache__", ".pytest_cache", "_tmp"}
_IDENTITY_DIAGNOSTIC_PATH_FIELDS: Final = {
    "consumer_python_executable",
    "consumer_resolved_package_file",
    "layer_interpreters",
    "python_executable",
    "resolved_package_file",
    "rust_extension_file",
}
_FAILURE_OUTPUT_LIMIT: Final = 64 * 1024


def _failure_output(value: str) -> str:
    """Keep enough pytest output to diagnose a failed isolated command."""
    if len(value) <= _FAILURE_OUTPUT_LIMIT:
        return value
    header = f"[... output truncated to {_FAILURE_OUTPUT_LIMIT} characters ...]\n"
    return header + value[-(_FAILURE_OUTPUT_LIMIT - len(header)) :]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_fingerprint(identity: dict[str, object]) -> str:
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_identity_fingerprint(identity: dict[str, object]) -> str:
    """Hash stable candidate/runtime identity, excluding job-local absolute paths."""
    stable = {
        key: value
        for key, value in identity.items()
        if key not in _IDENTITY_DIAGNOSTIC_PATH_FIELDS and key != "identity_sha256"
    }
    return _identity_fingerprint(stable)


def _candidate_source_fingerprint() -> str:
    """Hash executable candidate source and contracts, excluding build outputs."""
    return candidate_source_fingerprint(ROOT)[0]


def _suite_test_files(root: Path) -> list[Path]:
    """List test sources without traversing generated test artifacts."""
    if not root.is_dir():
        return []
    files: list[Path] = []
    for current, directories, names in os.walk(root):
        directories[:] = [
            name
            for name in directories
            if not name.startswith(".")
            and name not in _IGNORED_TEST_TREE_NAMES
            and not name.startswith("_engine_tmp")
        ]
        directory = Path(current)
        files.extend(
            directory / name
            for name in names
            if name.startswith("test") and name.endswith(".py")
        )
    return files


def _verification_harness_fingerprint(application_root: Path) -> str:
    """Hash suite-selection code/config so resume cannot cross harness changes."""
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "adr015_candidate_identity.py",
        ROOT / "scripts" / "adr015_source_identity.py",
        ROOT / "scripts" / "adr015_container_worker.sh",
        ROOT / "scripts" / "adr015_native_path.py",
        ROOT / "scripts" / "adr015_pytest_bootstrap.py",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "pytest.ini",
        ROOT / "tests" / "conftest.py",
        application_root / "pytest.ini",
        application_root / "pyproject.toml",
        application_root / ".env.example",
        application_root / ".vscode" / "launch.json",
        application_root / "tests" / "conftest.py",
        application_root / "kg-doc-parser" / "pytest.ini",
        application_root / "kg-doc-parser" / "pyproject.toml",
        application_root / "kg-doc-parser" / "tests" / "conftest.py",
        application_root / "kogwistar-obsidian-sink" / "pytest.ini",
        application_root / "kogwistar-obsidian-sink" / "pyproject.toml",
        application_root / "kogwistar-obsidian-sink" / "tests" / "conftest.py",
    ]
    for suite_root in (
        ROOT / "tests",
        application_root / "tests",
        application_root / "kg-doc-parser" / "tests",
        application_root / "kogwistar-obsidian-sink" / "tests",
    ):
        files.extend(_suite_test_files(suite_root))
    digest = hashlib.sha256()
    for path in sorted(path for path in files if path.is_file()):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_report(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _resolve_application_root(raw: str | None, manifest: dict) -> Path:
    configured = raw or os.getenv(
        manifest["reference_application"]["root_environment_variable"]
    )
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = (
            ROOT / manifest["reference_application"]["default_relative_root"]
        ).resolve()
    if not root.is_dir():
        raise SystemExit(f"reference application root does not exist: {root}")
    return root


def _pythonpath(application_root: Path) -> str:
    paths = [
        ROOT,
        application_root / "src",
        application_root / "kg-doc-parser",
        application_root / "kogwistar-obsidian-sink",
    ]
    return os.pathsep.join(str(path) for path in paths)


def _capability_configurations(manifest: dict) -> tuple[str, ...]:
    """Return ordered coarse implementation configuration keys from ADR-015."""
    configurations: list[str] = []
    for ownership in manifest.get("capability_ownership", []):
        configuration = ownership.get("configuration")
        if isinstance(configuration, str) and configuration not in configurations:
            configurations.append(configuration)
    if not configurations:
        raise ValueError("manifest has no capability ownership configurations")
    return tuple(configurations)


def _capability_mode_choices(manifest: dict, configuration: str) -> tuple[str, ...]:
    modes = tuple(manifest["implementation_modes"])
    if configuration == _SERVER_CONFIGURATION:
        modes = tuple(mode for mode in modes if mode in {"python", "rust"})
    if not modes:
        raise ValueError(f"manifest has no valid modes for {configuration}")
    return modes


def _capability_argument_name(configuration: str) -> str:
    suffix = configuration.removeprefix("KOGWISTAR_IMPL_").lower()
    return f"--{suffix.replace('_', '-')}-mode"


def _capability_modes(
    *, manifest: dict, implementation_mode: str, overrides: dict[str, str | None]
) -> dict[str, str]:
    """Resolve legacy global mode plus explicit coarse capability overrides.

    `shadow` remains valid for the old global flag.  The server has no shadow
    implementation, so its explicit configuration remains Python unless the
    caller supplies its valid Python/Rust override.
    """
    modes: dict[str, str] = {}
    for configuration in _capability_configurations(manifest):
        mode = implementation_mode
        if (
            configuration == _SERVER_CONFIGURATION
            and mode not in _capability_mode_choices(manifest, configuration)
        ):
            mode = "python"
        override = overrides.get(configuration)
        if override is not None:
            mode = override
        valid_modes = _capability_mode_choices(manifest, configuration)
        if mode not in valid_modes:
            choices = ", ".join(valid_modes)
            raise ValueError(f"{configuration} must be one of: {choices}; got {mode!r}")
        modes[configuration] = mode
    return modes


def _active_writers(manifest: dict, capability_modes: dict[str, str]) -> dict[str, str]:
    """Resolve ADR-015's current versus target writer for every capability."""
    writers: dict[str, str] = {}
    for ownership in manifest.get("capability_ownership", []):
        capability = ownership.get("capability")
        configuration = ownership.get("configuration")
        if not isinstance(capability, str) or not isinstance(configuration, str):
            continue
        mode = capability_modes[configuration]
        # Readiness must be explicit. Missing metadata can never promote an
        # authoritative writer during a strangler migration.
        rust_cutover_ready = ownership.get("rust_cutover_ready", False)
        writer_key = (
            "target_authoritative_writer"
            if mode == "rust" and rust_cutover_ready is True
            else "current_authoritative_writer"
        )
        writer = ownership.get(writer_key)
        if not isinstance(writer, str):
            raise ValueError(f"manifest has no {writer_key} for {capability}")
        writers[capability] = writer
    return writers


def _environment(
    application_root: Path,
    mode: str,
    capability_modes: dict[str, str],
) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath(application_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["KOGWISTAR_IMPL_MODE"] = mode
    env.update(capability_modes)
    env.setdefault("KOGWISTAR_LOG_LEVEL", "WARNING")
    env.setdefault("LOG_LEVEL", "WARNING")
    env.setdefault("UVICORN_LOG_LEVEL", "WARNING")
    env.setdefault(
        "PYTEST_ADDOPTS",
        " ".join(
            [
                "--log-cli-level=WARNING",
                "--log-level=WARNING",
                "--log-disable=workflow.trace",
                "--log-disable=workflow.telemetry",
                "--log-disable=urllib3",
                "--log-disable=urllib3.connectionpool",
                "--show-capture=no",
            ]
        ),
    )
    return env


def _identity(
    *,
    python: Path,
    application_root: Path,
    mode: str,
    capability_modes: dict[str, str],
    active_writers: dict[str, str],
    backend: str,
    storage_root: str,
    commit_provenance: dict[str, str | None] | None = None,
) -> dict[str, object]:
    expected = (ROOT / "kogwistar" / "__init__.py").resolve()
    result = subprocess.run(
        [
            str(python),
            "-P",
            str(ROOT / "scripts" / "adr015_candidate_identity.py"),
            str(expected),
        ],
        cwd=ROOT,
        env=_environment(application_root, mode, capability_modes),
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(result.stdout.strip().splitlines()[-1])
    resolved = Path(str(identity["resolved_package_file"]))
    if resolved != expected:
        raise SystemExit(
            f"candidate import mismatch: expected {expected}, resolved {resolved}"
        )

    detected_commits = {
        "core_commit": _git_commit(ROOT),
        "application_commit": _git_commit(application_root),
        "parser_commit": _git_commit(application_root / "kg-doc-parser"),
        "sink_commit": _git_commit(application_root / "kogwistar-obsidian-sink"),
        "application_core_pin_commit": _git_commit(application_root / "kogwistar"),
    }
    commits = _resolved_commit_provenance(detected_commits, commit_provenance)
    identity.update(
        {
            "implementation_mode": mode,
            "capability_modes": capability_modes,
            "active_writers": active_writers,
            "backend": backend,
            "storage_root": storage_root,
            "candidate_source_sha256": _candidate_source_fingerprint(),
            "verification_harness_sha256": _verification_harness_fingerprint(
                application_root
            ),
            **commits,
        }
    )
    return identity


def _resolved_commit_provenance(
    detected: dict[str, str | None], supplied: dict[str, str | None] | None
) -> dict[str, str | None]:
    """Prefer immutable-stage provenance supplied by its host orchestrator."""
    return {key: (supplied or {}).get(key) or value for key, value in detected.items()}


def _layer_paths(manifest: dict, application_root: Path) -> dict[str, Path]:
    roots = {"candidate": ROOT, "application": application_root}
    return {
        item["name"]: (roots[item["relative_to"]] / item["path"]).resolve()
        for item in manifest["suite_layers"]
    }


def _default_consumer_python(application_root: Path, fallback: Path) -> Path:
    candidates = [
        application_root / ".venv" / "Scripts" / "python.exe",
        application_root / ".venv" / "bin" / "python",
    ]
    return next((path for path in candidates if path.is_file()), fallback)


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    """Make executable paths absolute while preserving venv symlinks."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _pytest_args(values: list[str]) -> list[str]:
    """Remove argparse's option terminator before forwarding to pytest."""
    return values[1:] if values[:1] == ["--"] else values


def _marker_expression(layer: str, profile: str) -> str:
    exclusions = "not manual and not llm_real and not legacy and not requires_ollama"
    if profile == "feature":
        if layer == "sink":
            # Sink marker debt: retain its deterministic unmarked suite until
            # the consumer classifies feature/regression tests explicitly.
            return (
                "not slow and not manual and not integration and not longrun "
                "and not external and not llm_real and not requires_ollama"
            )
        return f"(ci or regression) and not slow and {exclusions}"
    if profile == "regression":
        return f"regression and not slow and {exclusions}"
    if profile == "milestone":
        if layer in {"core", "parser"}:
            return (
                "(ci or ci_full) and not manual and not llm_real "
                "and not legacy and not requires_ollama"
            )
        if layer == "application":
            return (
                "(ci or ci_full) and not manual and not llm_real "
                "and not legacy and not requires_ollama"
            )
        if layer == "sink":
            # The sink does not classify its deterministic tests with ci markers.
            return (
                "not manual and not integration and not longrun and not external "
                "and not llm_real and not requires_ollama"
            )
        raise ValueError(f"unknown suite layer: {layer}")
    raise ValueError(f"unknown test profile: {profile}")


def _target_group_key(tests_path: Path, targets: list[Path]) -> str:
    """Return a stable, workspace-independent identity for one pytest group."""
    values: list[str] = []
    for target in targets:
        try:
            values.append(target.relative_to(tests_path).as_posix() or ".")
        except ValueError:
            values.append(target.as_posix())
    return "|".join(values)


def _shard_assignments(
    group_keys: list[str],
    *,
    shard_count: int,
    shard_offset: int = 0,
    timing_estimates: dict[str, float] | None = None,
) -> list[int]:
    """Assign every group once using deterministic longest-processing-time order."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    estimates = timing_estimates or {}
    weighted = [
        (index, key, max(float(estimates.get(key, 1.0)), 0.0))
        for index, key in enumerate(group_keys)
    ]
    weighted.sort(key=lambda item: (-item[2], item[1], item[0]))
    loads = [0.0] * shard_count
    assignments = [0] * len(group_keys)
    for index, _key, duration in weighted:
        shard = min(
            range(shard_count),
            key=lambda value: (
                loads[value],
                (value - shard_offset) % shard_count,
            ),
        )
        assignments[index] = shard
        loads[shard] += duration
    return assignments


def _timing_history(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Read median group wall times; timing data never enters candidate identity."""
    samples: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"timing history does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        layers = payload.get("layers", []) if isinstance(payload, dict) else []
        for layer in layers:
            if not isinstance(layer, dict) or not isinstance(layer.get("name"), str):
                continue
            layer_samples = samples.setdefault(layer["name"], {})
            for run in layer.get("runs", []):
                if not isinstance(run, dict):
                    continue
                key = run.get("group_key")
                duration = run.get("duration_seconds")
                if isinstance(key, str) and isinstance(duration, (int, float)):
                    layer_samples.setdefault(key, []).append(float(duration))
    return {
        layer: {key: statistics.median(values) for key, values in groups.items()}
        for layer, groups in samples.items()
    }


# Full core coverage mixes embedded Chroma clients, subprocess Chroma servers,
# testcontainers, MCP/ASGI servers, and context-global auth state.  On Windows,
# even small cross-file batches can leak native handles or global clients into
# the next file.  One file per process is the release evidence boundary.
_FILE_ISOLATED_RELEASE_LAYERS: Final = frozenset({"core", "parser"})
_ALWAYS_FILE_ISOLATED_LAYERS: Final = frozenset({"parser"})
_XDIST_SAFE_LAYERS: Final = frozenset({"core"})
_RELEASE_GROUP_SIZE: Final = 1
_LAYER_SHARD_OFFSETS: Final = {
    "core": 0,
    "parser": 1,
    "sink": 2,
    "application": 0,
}

def _target_groups(*, layer: str, tests_path: Path, profile: str) -> list[list[Path]]:
    """Return deterministic pytest process boundaries for one suite layer."""
    if layer in _ALWAYS_FILE_ISOLATED_LAYERS or (
        layer in _FILE_ISOLATED_RELEASE_LAYERS and profile == "milestone"
    ):
        tests = sorted(
            (path for path in tests_path.rglob("test*.py") if path.is_file()),
            key=lambda path: path.relative_to(tests_path).as_posix(),
        )
        return [
            tests[index : index + _RELEASE_GROUP_SIZE]
            for index in range(0, len(tests), _RELEASE_GROUP_SIZE)
        ]
    if layer != "application":
        return [[tests_path]]

    # Large Windows application runs can exhaust process handles in Chroma
    # before late tests initialize TLS. Separate processes preserve exact
    # collection while releasing handles between directory groups.
    root_tests = sorted(tests_path.glob("test*.py"))
    grouped = [
        [tests_path / name]
        for name in ("integration", "smoke")
        if (tests_path / name).is_dir()
    ]
    unit_tests = sorted((tests_path / "unit").rglob("test*.py"))
    unit_groups = [
        unit_tests[index : index + 5] for index in range(0, len(unit_tests), 5)
    ]
    return ([root_tests] if root_tests else []) + grouped + unit_groups


def _command_succeeded(*, layer: str, profile: str, returncode: int) -> bool:
    # File-isolated release commands legitimately return pytest code 5
    # when every item in that file is outside ci/ci_full. Coverage is proven by
    # `_target_groups`; the raw code remains in the report for auditability.
    return (
        returncode == 0
        or (profile == "regression" and returncode == 5)
        or (layer in _ALWAYS_FILE_ISOLATED_LAYERS and returncode == 5)
        or (
            layer in _FILE_ISOLATED_RELEASE_LAYERS
            and profile == "milestone"
            and returncode == 5
        )
    )


def _reusable_run(
    *,
    command: list[str],
    prior_runs: list[dict[str, object]],
    layer: str,
    profile: str,
) -> dict[str, object] | None:
    for run in prior_runs:
        if run.get("command") != command:
            continue
        returncode = run.get("returncode")
        if isinstance(returncode, int) and _command_succeeded(
            layer=layer, profile=profile, returncode=returncode
        ):
            return {**run, "reused": True}
    return None


def _run_layer(
    *,
    python: Path,
    application_root: Path,
    mode: str,
    capability_modes: dict[str, str],
    active_writers: dict[str, str],
    layer: str,
    tests_path: Path,
    cwd: Path,
    profile: str,
    extra_pytest_args: list[str],
    dry_run: bool,
    identity_fingerprint: str,
    shard_index: int = 0,
    shard_count: int = 1,
    pytest_workers: int = 0,
    timing_estimates: dict[str, float] | None = None,
    prior_runs: list[dict[str, object]] | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    expression = _marker_expression(layer, profile)
    bootstrap = ROOT / "scripts" / "adr015_pytest_bootstrap.py"
    expected_package = (ROOT / "kogwistar" / "__init__.py").resolve()
    target_groups = _target_groups(layer=layer, tests_path=tests_path, profile=profile)

    group_keys = [_target_group_key(tests_path, targets) for targets in target_groups]
    assignments = _shard_assignments(
        group_keys,
        shard_count=shard_count,
        shard_offset=_LAYER_SHARD_OFFSETS[layer] % shard_count,
        timing_estimates=timing_estimates,
    )
    selected_indexes = [
        index for index, shard in enumerate(assignments) if shard == shard_index
    ]
    effective_pytest_workers = (
        pytest_workers if layer in _XDIST_SAFE_LAYERS and profile != "milestone" else 0
    )
    xdist_args = (
        ["-n", str(pytest_workers), "--dist", "loadfile", "--max-worker-restart", "0"]
        if effective_pytest_workers > 0
        else []
    )
    all_commands = [
        [
            str(python),
            "-P",
            str(bootstrap),
            str(expected_package),
            "--",
            *(str(target) for target in targets),
            "-m",
            expression,
            "-q",
            *xdist_args,
            *extra_pytest_args,
        ]
        for targets in target_groups
    ]
    commands = [all_commands[index] for index in selected_indexes]
    selected_metadata = [
        {
            "group_index": index,
            "group_key": group_keys[index],
            "estimated_duration_seconds": float(
                (timing_estimates or {}).get(group_keys[index], 1.0)
            ),
        }
        for index in selected_indexes
    ]
    record: dict[str, object] = {
        "name": layer,
        "interpreter": str(python),
        "tests_path": str(tests_path),
        "cwd": str(cwd),
        "marker_expression": expression,
        "test_profile": profile,
        "commands": commands,
        "target_group_count": len(target_groups),
        "selected_group_indexes": selected_indexes,
        "group_assignments": assignments,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "pytest_workers": effective_pytest_workers,
        "requested_pytest_workers": pytest_workers,
        "candidate_identity_sha256": identity_fingerprint,
        "capability_modes": capability_modes,
        "active_writers": active_writers,
    }
    for command in commands:
        print(f"[{layer}] {' '.join(command)}", flush=True)
    if dry_run:
        record.update(
            {
                "status": "planned",
                "returncode": None,
                "duration_seconds": 0.0,
                "runs": [],
            }
        )
        return record
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    returncode = 0
    for command, metadata in zip(commands, selected_metadata, strict=True):
        reusable = _reusable_run(
            command=command,
            prior_runs=prior_runs or [],
            layer=layer,
            profile=profile,
        )
        if reusable is not None:
            runs.append({**reusable, **metadata})
            record["runs"] = runs
            if progress is not None:
                progress(record)
            print(f"[{layer}] reused passed command", flush=True)
            continue
        run_started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=_environment(application_root, mode, capability_modes),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        run_record = {
            "command": command,
            "returncode": result.returncode,
            "duration_seconds": round(time.perf_counter() - run_started, 6),
            **metadata,
        }
        if result.returncode != 0:
            run_record.update(
                {
                    "stdout": _failure_output(result.stdout),
                    "stderr": _failure_output(result.stderr),
                }
            )
        runs.append(run_record)
        record["runs"] = runs
        if progress is not None:
            progress(record)
        if not _command_succeeded(
            layer=layer, profile=profile, returncode=result.returncode
        ):
            returncode = result.returncode
            break
    record.update(
        {
            "status": "passed" if returncode == 0 else "failed",
            "returncode": returncode,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "runs": runs,
        }
    )
    return record


def _parse_args(manifest: dict | None = None) -> argparse.Namespace:
    if manifest is None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Run ADR-015 compatibility suites against one core candidate."
    )
    parser.add_argument("--application-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--consumer-python",
        help="Interpreter for parser, sink, and application layers. "
        "Default: reference application's .venv, then --python.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        choices=("core", "parser", "sink", "application"),
        help="Run one layer; repeat for multiple. Default: all four.",
    )
    parser.add_argument(
        "--implementation-mode",
        choices=("python", "shadow", "rust"),
        default="python",
        help="Legacy default for every capability; use capability flags to override.",
    )
    for configuration in _capability_configurations(manifest):
        parser.add_argument(
            _capability_argument_name(configuration),
            dest=f"capability_mode_{configuration}",
            choices=_capability_mode_choices(manifest, configuration),
            help=f"Override {configuration} for this run.",
        )
    parser.add_argument("--backend", default="unspecified")
    parser.add_argument("--storage-root", default="unspecified")
    for name in (
        "core",
        "application",
        "parser",
        "sink",
        "application-core-pin",
    ):
        parser.add_argument(
            f"--{name}-commit",
            help="Commit provenance supplied by an outer immutable-stage harness.",
        )
    parser.add_argument(
        "--profile",
        choices=("feature", "regression", "milestone"),
        default="feature",
        help="feature=(ci or regression) without slow; regression=regression "
        "without slow; milestone=full ADR compatibility rehearsal.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Deprecated alias for --profile milestone.",
    )
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse passed layers from --report when candidate identity is unchanged.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically divide each layer's existing process groups.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to execute.",
    )
    parser.add_argument(
        "--pytest-workers",
        type=int,
        default=0,
        help="pytest-xdist workers inside each selected pytest process; 0 disables.",
    )
    parser.add_argument(
        "--timing-history",
        action="append",
        type=Path,
        default=[],
        help="Prior report used only for median LPT shard balancing; repeatable.",
    )
    parser.add_argument("--report")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    args = _parse_args(manifest)
    profile = "milestone" if args.release else args.profile
    pytest_args = _pytest_args(args.pytest_args)
    if args.release and args.profile != "feature":
        raise SystemExit("--release cannot be combined with an explicit --profile")
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    if args.pytest_workers < 0:
        raise SystemExit("--pytest-workers cannot be negative")
    application_root = _resolve_application_root(args.application_root, manifest)
    python = _absolute_path_without_symlink_resolution(Path(args.python))
    if not python.is_file():
        raise SystemExit(f"python interpreter does not exist: {python}")
    consumer_python = (
        _absolute_path_without_symlink_resolution(Path(args.consumer_python))
        if args.consumer_python
        else _default_consumer_python(application_root, python)
    )
    if not consumer_python.is_file():
        raise SystemExit(
            f"consumer python interpreter does not exist: {consumer_python}"
        )

    report_path = Path(args.report).expanduser().resolve() if args.report else None
    started_at = _utc_now()
    capability_modes = _capability_modes(
        manifest=manifest,
        implementation_mode=args.implementation_mode,
        overrides={
            configuration: getattr(args, f"capability_mode_{configuration}")
            for configuration in _capability_configurations(manifest)
        },
    )
    active_writers = _active_writers(manifest, capability_modes)
    identity = _identity(
        python=python,
        application_root=application_root,
        mode=args.implementation_mode,
        capability_modes=capability_modes,
        active_writers=active_writers,
        backend=args.backend,
        storage_root=args.storage_root,
        commit_provenance={
            "core_commit": args.core_commit,
            "application_commit": args.application_commit,
            "parser_commit": args.parser_commit,
            "sink_commit": args.sink_commit,
            "application_core_pin_commit": args.application_core_pin_commit,
        },
    )
    consumer_identity = _identity(
        python=consumer_python,
        application_root=application_root,
        mode=args.implementation_mode,
        capability_modes=capability_modes,
        active_writers=active_writers,
        backend=args.backend,
        storage_root=args.storage_root,
        commit_provenance={
            "core_commit": args.core_commit,
            "application_commit": args.application_commit,
            "parser_commit": args.parser_commit,
            "sink_commit": args.sink_commit,
            "application_core_pin_commit": args.application_core_pin_commit,
        },
    )
    identity["layer_interpreters"] = {
        "core": str(python),
        "parser": str(consumer_python),
        "sink": str(consumer_python),
        "application": str(consumer_python),
    }
    identity["consumer_resolved_package_file"] = consumer_identity[
        "resolved_package_file"
    ]
    identity["consumer_rust_extension_version"] = consumer_identity[
        "rust_extension_version"
    ]
    identity["consumer_rust_contract_version"] = consumer_identity[
        "rust_contract_version"
    ]
    identity["consumer_python_version"] = consumer_identity["python_version"]
    identity["consumer_python_implementation"] = consumer_identity[
        "python_implementation"
    ]
    identity["consumer_python_abi"] = consumer_identity["python_abi"]
    identity["consumer_python_environment_sha256"] = consumer_identity[
        "python_environment_sha256"
    ]
    identity["consumer_python_executable"] = consumer_identity["python_executable"]
    identity["contract_version"] = manifest["contract_version"]
    identity["test_profile"] = profile
    fingerprint = _candidate_identity_fingerprint(identity)
    identity["identity_sha256"] = fingerprint
    timing_history = _timing_history(
        [path.expanduser().resolve() for path in args.timing_history]
    )
    print(json.dumps(identity, indent=2, sort_keys=True), flush=True)

    report: dict[str, object]
    if args.resume and report_path is not None and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prior_candidate = report.get("candidate")
        if (
            not isinstance(prior_candidate, dict)
            or prior_candidate.get("identity_sha256") != fingerprint
        ):
            raise SystemExit("cannot resume: report candidate identity has changed")
        prior_layers = report.get("layers")
        if not isinstance(prior_layers, list):
            raise SystemExit("cannot resume: report layers are invalid")
        report.update(
            {
                "candidate": identity,
                "finished_at_utc": None,
                "status": "running",
            }
        )
    else:
        report = {
            "report_version": 1,
            "contract_version": manifest["contract_version"],
            "manifest": str(MANIFEST_PATH),
            "candidate": identity,
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "status": "running",
            "layers": [],
        }
    report["execution"] = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pytest_workers": args.pytest_workers,
        "timing_history": [str(path) for path in args.timing_history],
    }
    _write_report(report_path, report)

    if args.identity_only:
        report.update({"status": "identity-only", "finished_at_utc": _utc_now()})
        _write_report(report_path, report)
        return 0

    layers = args.layer or ["core", "parser", "sink", "application"]
    paths = _layer_paths(manifest, application_root)
    layer_cwds = {
        "core": ROOT,
        "parser": application_root,
        "sink": application_root / "kogwistar-obsidian-sink",
        "application": application_root,
    }
    for layer in layers:
        layer_records = report["layers"]
        assert isinstance(layer_records, list)
        prior_pass = next(
            (
                item
                for item in layer_records
                if isinstance(item, dict)
                and item.get("name") == layer
                and item.get("status") == "passed"
                and item.get("candidate_identity_sha256") == fingerprint
                and item.get("shard_index") == args.shard_index
                and item.get("shard_count") == args.shard_count
                and item.get("requested_pytest_workers", item.get("pytest_workers"))
                == args.pytest_workers
            ),
            None,
        )
        if args.resume and prior_pass is not None:
            print(f"[{layer}] reused passed layer from report", flush=True)
            continue
        prior_record = next(
            (
                item
                for item in layer_records
                if isinstance(item, dict)
                and item.get("name") == layer
                and item.get("candidate_identity_sha256") == fingerprint
            ),
            None,
        )
        layer_records[:] = [
            item
            for item in layer_records
            if not isinstance(item, dict) or item.get("name") != layer
        ]
        tests_path = paths[layer]
        if not tests_path.is_dir():
            raise SystemExit(f"{layer} tests path does not exist: {tests_path}")

        def persist_progress(record: dict[str, object]) -> None:
            layer_records[:] = [
                item
                for item in layer_records
                if not isinstance(item, dict) or item.get("name") != layer
            ]
            layer_records.append(record)
            _write_report(report_path, report)

        layer_record = _run_layer(
            python=python if layer == "core" else consumer_python,
            application_root=application_root,
            mode=args.implementation_mode,
            capability_modes=capability_modes,
            active_writers=active_writers,
            layer=layer,
            tests_path=tests_path,
            cwd=layer_cwds[layer],
            profile=profile,
            extra_pytest_args=pytest_args,
            dry_run=args.dry_run,
            identity_fingerprint=fingerprint,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            pytest_workers=args.pytest_workers,
            timing_estimates=timing_history.get(layer, {}),
            prior_runs=(
                prior_record.get("runs", [])
                if args.resume and isinstance(prior_record, dict)
                else []
            ),
            progress=persist_progress,
        )
        persist_progress(layer_record)
        _write_report(report_path, report)
        returncode = layer_record["returncode"]
        if isinstance(returncode, int) and returncode != 0:
            report.update({"status": "failed", "finished_at_utc": _utc_now()})
            _write_report(report_path, report)
            return returncode
    report.update(
        {
            "status": "planned" if args.dry_run else "passed",
            "finished_at_utc": _utc_now(),
        }
    )
    _write_report(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
