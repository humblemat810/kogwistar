from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    chroma_cli = os.environ.get("CHROMA_CLI") or shutil.which("chroma")
    if not chroma_cli:
        raise SystemExit(
            "Could not find the `chroma` CLI on PATH. Install chromadb's CLI or "
            "set CHROMA_CLI to the executable path."
        )

    host = os.environ.get("CHROMA_HOST", "127.0.0.1")
    port = os.environ.get("CHROMA_PORT", "8000")
    persist_dir = Path(os.environ.get("CHROMA_PERSIST_DIR", ".gke-data/chroma"))
    persist_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.environ.get("CHROMA_LOG_DIR", ".gke-data/chroma-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"chroma-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    env = os.environ.copy()
    env.setdefault("RUST_LOG", "debug")
    env.setdefault("CHROMA_LOG_LEVEL", "debug")

    cmd = [
        chroma_cli,
        "run",
        "--path",
        str(persist_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]

    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
        try:
            return int(proc.wait(timeout=10))
        except subprocess.TimeoutExpired:
            proc.kill()
            return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
