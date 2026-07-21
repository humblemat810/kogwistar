"""Print installed candidate/runtime identity for ADR-015 compatibility runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig

import kogwistar


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("expected candidate package path")
    expected = Path(args[0]).resolve()
    resolved = Path(kogwistar.__file__).resolve()
    if resolved != expected:
        raise SystemExit(
            f"candidate import mismatch: expected {expected}, resolved {resolved}"
        )
    spec = importlib.util.find_spec("kogwistar._rust")
    extension = (
        None if spec is None else __import__("kogwistar._rust", fromlist=["*"])
    )
    packages = sorted(
        f"{(distribution.metadata.get('Name') or '').lower()}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    print(
        json.dumps(
            {
                "resolved_package_file": str(resolved),
                "package_version": getattr(kogwistar, "__version__", None),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_abi": sysconfig.get_config_var("SOABI"),
                "python_executable": sys.executable,
                "python_environment_sha256": hashlib.sha256(
                    "\n".join(packages).encode()
                ).hexdigest(),
                "rust_extension_file": None if spec is None else spec.origin,
                "rust_extension_version": None
                if extension is None
                else getattr(extension, "__version__", None),
                "rust_contract_version": None
                if extension is None
                else getattr(extension, "CONTRACT_VERSION", None),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
