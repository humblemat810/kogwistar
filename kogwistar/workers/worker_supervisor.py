from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _write_text(path: Optional[str], content: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _spawn_worker(
    *, python_exe: str, worker_module: str, args: list[str]
) -> subprocess.Popen:
    # Use -m so it works when installed as a package.
    cmd = [python_exe, "-m", worker_module] + args
    return subprocess.Popen(cmd)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform supervisor for the index job worker process"
    )
    parser.add_argument("--persist-directory", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--tick-interval-ms", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-jobs-per-tick", type=int, default=200)
    parser.add_argument("--max-inflight", type=int, default=50)
    parser.add_argument("--lease-seconds", type=int, default=60)
    parser.add_argument(
        "--pidfile", default="", help="Write worker PID here (for tests/ops)"
    )
    parser.add_argument(
        "--restart-delay-ms",
        type=int,
        default=100,
        help="Delay before restarting worker after exit",
    )
    parser.add_argument("--phase1-enable-index-jobs", action="store_true")
    parser.add_argument(
        "--worker-module",
        default="kogwistar.workers.index_job_worker",
        help="Module path to run as worker",
    )
    args = parser.parse_args(argv)

    stop = {"flag": False}

    def _handle(_signum, _frame):
        stop["flag"] = True

    try:
        signal.signal(signal.SIGTERM, _handle)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGINT, _handle)
    except Exception:
        pass

    worker_args = [
        "--persist-directory",
        args.persist_directory,
        "--namespace",
        args.namespace,
        "--tick-interval-ms",
        str(args.tick_interval_ms),
        "--batch-size",
        str(args.batch_size),
        "--max-jobs-per-tick",
        str(args.max_jobs_per_tick),
        "--max-inflight",
        str(args.max_inflight),
        "--lease-seconds",
        str(args.lease_seconds),
    ]
    if args.phase1_enable_index_jobs:
        worker_args.append("--phase1-enable-index-jobs")

    python_exe = sys.executable
    proc: Optional[subprocess.Popen] = None

    while not stop["flag"]:
        if proc is None or proc.poll() is not None:
            # (re)start worker
            proc = _spawn_worker(
                python_exe=python_exe,
                worker_module=args.worker_module,
                args=worker_args,
            )
            _write_text(args.pidfile, str(proc.pid))
            # Small delay to avoid tight respawn loops
            time.sleep(max(0.01, args.restart_delay_ms / 1000.0))

        # sleep a bit; keep loop responsive to stop signals
        time.sleep(0.05)

    # graceful shutdown: terminate worker then wait
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
