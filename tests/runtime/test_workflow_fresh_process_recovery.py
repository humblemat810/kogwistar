from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core, pytest.mark.runtime, pytest.mark.ci_full, pytest.mark.slow]


def test_persistent_workflow_recovers_from_fresh_process_after_checkpoint(
    tmp_path: Path,
) -> None:
    pytest.importorskip("chromadb")
    worker = Path(__file__).with_name("_fresh_process_recovery_worker.py")
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(repo_root), env.get("PYTHONPATH", "")) if path
    )

    stderr_path = tmp_path / "start-worker.stderr"
    with stderr_path.open("w", encoding="utf-8") as stderr_log:
        interrupted = subprocess.Popen(
            [sys.executable, str(worker), str(tmp_path), "start"],
            cwd=repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
            text=True,
        )
        marker = tmp_path / "gate-entered"
        deadline = time.monotonic() + 90.0
        while not marker.exists() and time.monotonic() < deadline:
            if interrupted.poll() is not None:
                stderr_log.flush()
                pytest.fail(
                    "workflow worker exited before checkpoint gate: "
                    f"{stderr_path.read_text(encoding='utf-8')}"
                )
            time.sleep(0.1)
        assert marker.exists(), "worker did not reach the post-checkpoint gate"

        interrupted.kill()
        interrupted.wait(timeout=30)
    assert interrupted.returncode != 0
    time.sleep(1.1)  # Let the dead process's ownership lease expire before takeover.

    resumed = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), "resume"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr

    result = json.loads((tmp_path / "resume-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["state"]["durable_start_completed"] is True
    assert result["state"]["gate_recovered"] is True
    assert result["state"]["workflow_completed"] is True
    assert (tmp_path / "start-count.txt").read_text(encoding="utf-8") == "1"
