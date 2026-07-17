from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Final


ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST_PATH: Final = ROOT / "contracts" / "rust-port-v1.json"
_CANDIDATE_SUFFIXES: Final = {".json", ".lock", ".py", ".rs", ".toml"}
_SERVER_CONFIGURATION: Final = "KOGWISTAR_IMPL_SERVER"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_fingerprint(identity: dict[str, object]) -> str:
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_source_fingerprint() -> str:
    """Hash executable candidate source, contracts, and built native artifact."""
    files: set[Path] = {ROOT / "pyproject.toml"}
    for root in (ROOT / "kogwistar", ROOT / "rust" / "crates", ROOT / "contracts"):
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _CANDIDATE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    for path in (ROOT / "rust" / "Cargo.toml", ROOT / "rust" / "Cargo.lock"):
        if path.is_file():
            files.add(path)
    native = ROOT / "kogwistar" / "_rust.pyd"
    if native.is_file():
        files.add(native)

    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verification_harness_fingerprint(application_root: Path) -> str:
    """Hash suite-selection code/config so resume cannot cross harness changes."""
    files = [
        Path(__file__).resolve(),
        ROOT / "pytest.ini",
        ROOT / "tests" / "conftest.py",
        application_root / "pytest.ini",
        application_root / "pyproject.toml",
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
        if suite_root.is_dir():
            files.extend(suite_root.rglob("test*.py"))
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
        if (
            isinstance(configuration, str)
            and configuration not in configurations
        ):
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


def _active_writers(
    manifest: dict, capability_modes: dict[str, str]
) -> dict[str, str]:
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
) -> dict[str, object]:
    script = r"""
import importlib.util
import json
import pathlib
import kogwistar

spec = importlib.util.find_spec("kogwistar._rust")
extension = None if spec is None else __import__("kogwistar._rust", fromlist=["*"])
print(json.dumps({
    "resolved_package_file": str(pathlib.Path(kogwistar.__file__).resolve()),
    "package_version": getattr(kogwistar, "__version__", None),
    "rust_extension_file": None if spec is None else spec.origin,
    "rust_extension_version": None if extension is None else getattr(extension, "__version__", None),
    "rust_contract_version": None if extension is None else getattr(extension, "CONTRACT_VERSION", None),
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-P", "-c", script],
        cwd=ROOT,
        env=_environment(application_root, mode, capability_modes),
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(result.stdout.strip().splitlines()[-1])
    resolved = Path(str(identity["resolved_package_file"]))
    expected = (ROOT / "kogwistar" / "__init__.py").resolve()
    if resolved != expected:
        raise SystemExit(
            "candidate import mismatch: "
            f"expected {expected}, resolved {resolved}"
        )

    nested = {
        "application_commit": _git_commit(application_root),
        "parser_commit": _git_commit(application_root / "kg-doc-parser"),
        "sink_commit": _git_commit(application_root / "kogwistar-obsidian-sink"),
        "application_core_pin_commit": _git_commit(application_root / "kogwistar"),
    }
    identity.update(
        {
            "implementation_mode": mode,
            "capability_modes": capability_modes,
            "active_writers": active_writers,
            "core_commit": _git_commit(ROOT),
            "backend": backend,
            "storage_root": storage_root,
            "candidate_source_sha256": _candidate_source_fingerprint(),
            "verification_harness_sha256": _verification_harness_fingerprint(
                application_root
            ),
            **nested,
        }
    )
    return identity


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
    return next((path.resolve() for path in candidates if path.is_file()), fallback)


def _marker_expression(layer: str, profile: str) -> str:
    exclusions = (
        "not manual and not llm_real and not legacy and not requires_ollama"
    )
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


# Full core coverage mixes embedded Chroma clients, subprocess Chroma servers,
# testcontainers, MCP/ASGI servers, and context-global auth state.  On Windows,
# even small cross-file batches can leak native handles or global clients into
# the next file.  One file per process is the release evidence boundary.
_FILE_ISOLATED_RELEASE_LAYERS: Final = frozenset({"core", "parser"})
_RELEASE_GROUP_SIZE: Final = 1


def _target_groups(
    *, layer: str, tests_path: Path, profile: str
) -> list[list[Path]]:
    """Return deterministic pytest process boundaries for one suite layer."""
    if layer in _FILE_ISOLATED_RELEASE_LAYERS and profile == "milestone":
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
    return returncode == 0 or (profile == "regression" and returncode == 5) or (
        layer in _FILE_ISOLATED_RELEASE_LAYERS
        and profile == "milestone"
        and returncode == 5
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
    prior_runs: list[dict[str, object]] | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    expression = _marker_expression(layer, profile)
    bootstrap = (
        "import pathlib, pytest, kogwistar; "
        f"expected=pathlib.Path({str((ROOT / 'kogwistar' / '__init__.py').resolve())!r}); "
        "resolved=pathlib.Path(kogwistar.__file__).resolve(); "
        "assert resolved == expected, "
        "f'candidate import mismatch before pytest: expected {expected}, resolved {resolved}'; "
        "raise SystemExit(pytest.console_main())"
    )
    target_groups = _target_groups(
        layer=layer, tests_path=tests_path, profile=profile
    )

    commands = [
        [
            str(python),
            "-P",
            "-c",
            bootstrap,
            *(str(target) for target in targets),
            "-m",
            expression,
            "-q",
            *extra_pytest_args,
        ]
        for targets in target_groups
    ]
    record: dict[str, object] = {
        "name": layer,
        "interpreter": str(python),
        "tests_path": str(tests_path),
        "cwd": str(cwd),
        "marker_expression": expression,
        "test_profile": profile,
        "commands": commands,
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
    for command in commands:
        reusable = _reusable_run(
            command=command,
            prior_runs=prior_runs or [],
            layer=layer,
            profile=profile,
        )
        if reusable is not None:
            runs.append(reusable)
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
        )
        runs.append(
            {
                "command": command,
                "returncode": result.returncode,
                "duration_seconds": round(time.perf_counter() - run_started, 6),
            }
        )
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
    parser.add_argument("--report")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    args = _parse_args(manifest)
    profile = "milestone" if args.release else args.profile
    if args.release and args.profile != "feature":
        raise SystemExit("--release cannot be combined with an explicit --profile")
    application_root = _resolve_application_root(args.application_root, manifest)
    python = Path(args.python).expanduser().resolve()
    if not python.is_file():
        raise SystemExit(f"python interpreter does not exist: {python}")
    consumer_python = (
        Path(args.consumer_python).expanduser().resolve()
        if args.consumer_python
        else _default_consumer_python(application_root, python)
    )
    if not consumer_python.is_file():
        raise SystemExit(f"consumer python interpreter does not exist: {consumer_python}")

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
    )
    consumer_identity = _identity(
        python=consumer_python,
        application_root=application_root,
        mode=args.implementation_mode,
        capability_modes=capability_modes,
        active_writers=active_writers,
        backend=args.backend,
        storage_root=args.storage_root,
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
    identity["contract_version"] = manifest["contract_version"]
    identity["test_profile"] = profile
    fingerprint = _identity_fingerprint(identity)
    identity["identity_sha256"] = fingerprint
    print(json.dumps(identity, indent=2, sort_keys=True), flush=True)

    report: dict[str, object]
    if args.resume and report_path is not None and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prior_candidate = report.get("candidate")
        if not isinstance(prior_candidate, dict) or prior_candidate.get(
            "identity_sha256"
        ) != fingerprint:
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
    _write_report(report_path, report)

    if args.identity_only:
        report.update(
            {"status": "identity-only", "finished_at_utc": _utc_now()}
        )
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
            extra_pytest_args=args.pytest_args,
            dry_run=args.dry_run,
            identity_fingerprint=fingerprint,
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
