"""Build and verify the current-host ADR-015 PyO3 extension in source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

try:
    from adr015_source_identity import candidate_source_fingerprint
except ModuleNotFoundError:  # imported as a repository module in unit tests
    from scripts.adr015_source_identity import candidate_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rust" / "crates" / "kogwistar-python" / "Cargo.toml"
SMOKE = ROOT / "scripts" / "adr015_native_extension_smoke.py"
DEFAULT_OUTPUT = ROOT / ".codex" / "host-native-wheel"
DEFAULT_REPORT = ROOT / ".codex" / "adr015-host-native-report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _wheel_native_member(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith("kogwistar/_rust")
            and Path(name).suffix.lower() in {".pyd", ".so", ".dylib"}
        ]
    if len(members) != 1:
        raise RuntimeError(f"expected one native extension in {wheel.name}, got {members!r}")
    return members[0]


def _install_native_from_wheel(wheel: Path) -> Path:
    member = _wheel_native_member(wheel)
    target = ROOT / member
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        with archive.open(member) as source:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=target.parent, suffix=target.suffix
            ) as temporary:
                shutil.copyfileobj(source, temporary)
                temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target.resolve()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def main() -> int:
    args = _args()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {python}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    built = _run(
        [
            str(python),
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(MANIFEST),
            "--interpreter",
            str(python),
            "--out",
            str(output),
        ]
    )
    if built.returncode:
        sys.stderr.write(built.stdout)
        sys.stderr.write(built.stderr)
        return built.returncode
    wheels = sorted(output.glob("kogwistar-*.whl"), key=lambda path: path.stat().st_mtime_ns)
    if not wheels:
        raise SystemExit("maturin completed without producing a wheel")
    wheel = wheels[-1]
    extension = _install_native_from_wheel(wheel)
    smoke = _run([str(python), str(SMOKE), "--expected-extension", str(extension)])
    if smoke.returncode:
        sys.stderr.write(smoke.stdout)
        sys.stderr.write(smoke.stderr)
        return smoke.returncode
    source_sha256, source_file_count = candidate_source_fingerprint(ROOT)
    report = {
        "schema": "adr015-host-native/v1",
        "candidate_source_sha256": source_sha256,
        "candidate_source_file_count": source_file_count,
        "python_executable": str(python),
        "wheel": str(wheel),
        "wheel_sha256": _sha256(wheel),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "smoke": json.loads(smoke.stdout),
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
