"""Smoke-test an installed native wheel and its packaged Rust server."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, json.load(response)


def main() -> None:
    import kogwistar
    import kogwistar._rust as native

    package_file = Path(kogwistar.__file__).resolve()
    native_file = Path(native.__file__).resolve()
    assert package_file.is_relative_to(Path(sys.prefix))
    assert native_file.is_relative_to(Path(sys.prefix))
    assert native.CONTRACT_VERSION >= 1
    assert hasattr(native, "api_run_server")
    assert any(
        item.name == "kogwistar-rust-server"
        for item in importlib.metadata.entry_points(group="console_scripts")
    )

    executable = Path(sys.executable).with_name("kogwistar-rust-server")
    environment = {
        **os.environ,
        "KOGWISTAR_IMPL_SERVER": "rust",
        "KOGWISTAR_SERVER_HOST": "127.0.0.1",
        "KOGWISTAR_SERVER_PORT": "18087",
        "KOGWISTAR_META_SQLITE_PATH": "/tmp/server.sqlite",
        "AUTH_MODE": "dev",
        "JWT_SECRET": "dev-secret",
    }
    process = subprocess.Popen(
        [str(executable)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"server exited {process.returncode}: {stdout}\n{stderr}"
                )
            try:
                status, health = _get("http://127.0.0.1:18087/health")
                assert status == 200
                assert health["implementation"]["mode"] == "rust"
                assert health["implementation"]["server_cutover_ready"] is False
                break
            except Exception as error:
                last_error = error
                time.sleep(0.1)
        else:
            raise RuntimeError(f"server did not become healthy: {last_error}")
        print(
            json.dumps(
                {
                    "package": str(package_file),
                    "extension": str(native_file),
                    "entrypoint": str(executable),
                    "health": health,
                },
                sort_keys=True,
            )
        )
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.returncode not in {0, -signal.SIGINT}:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"server shutdown failed {process.returncode}: {stdout}\n{stderr}"
            )


if __name__ == "__main__":
    main()
