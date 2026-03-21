from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests




@dataclass(frozen=True, slots=True)
class ReplayStats:
    sent: int = 0
    skipped: int = 0
    failed: int = 0


def _iter_jsonl_events(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Minimal JSONL reader that ignores non-JSON lines.
    Works even if your oplog has a header line.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # header or corrupted line — skip
                continue


def replay_oplog_to_bridge(
    *,
    oplog_path: Path,
    bridge_url: str,
    since_seq: int = 0,
    until_seq: Optional[int] = None,
    kg_graph_type: Optional[str] = None,
    max_events: Optional[int] = None,
    sleep_ms: int = 0,
    timeout_s: float = 1.0,
    stop_on_error: bool = False,
    verbose: bool = False,
) -> ReplayStats:
    """
    Reads ChangeEvent JSON objects from oplog JSONL and POSTs them to bridge /ingest.
    Intended for test/dev only.
    """
    bridge_url = bridge_url.rstrip("/")
    ingest_url = f"{bridge_url}/ingest"

    session = requests.Session()
    stats = ReplayStats()

    sent = skipped = failed = 0
    emitted = 0

    for ev in _iter_jsonl_events(oplog_path):
        # tolerate different schemas
        try:
            seq = int(ev.get("seq", -1))
        except Exception:
            seq = -1

        if seq < 0:
            skipped += 1
            if verbose:
                print(f"[skip] no seq: {ev.keys()}", file=sys.stderr)
            continue

        if seq <= since_seq:
            skipped += 1
            continue

        if until_seq is not None and seq > until_seq:
            break

        if kg_graph_type is not None:
            ent = ev.get("entity") or {}
            if ent.get("kg_graph_type") != kg_graph_type:
                skipped += 1
                continue

        try:
            r = session.post(ingest_url, json=ev, timeout=timeout_s)
            if r.status_code >= 400:
                failed += 1
                if verbose:
                    print(
                        f"[fail] seq={seq} status={r.status_code} body={r.text[:200]}",
                        file=sys.stderr,
                    )
                if stop_on_error:
                    break
            else:
                sent += 1
                emitted += 1
                if verbose and (sent <= 5 or sent % 1000 == 0):
                    print(f"[sent] seq={seq} op={ev.get('op')}")
        except Exception as e:
            failed += 1
            if verbose:
                print(f"[fail] seq={seq} err={e}", file=sys.stderr)
            if stop_on_error:
                break

        if max_events is not None and emitted >= max_events:
            break

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    return ReplayStats(sent=sent, skipped=skipped, failed=failed)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replay oplog JSONL into FastAPI debug bridge /ingest."
    )
    ap.add_argument("--oplog", required=True, help="Path to oplog JSONL file")
    ap.add_argument(
        "--bridge", required=True, help="Bridge base URL, e.g. http://127.0.0.1:8787"
    )
    ap.add_argument(
        "--since-seq", type=int, default=0, help="Send events with seq > since-seq"
    )
    ap.add_argument(
        "--until-seq", type=int, default=None, help="Stop after this seq (inclusive)"
    )
    ap.add_argument(
        "--kg-graph-type", default=None, help="Filter by entity.kg_graph_type"
    )
    ap.add_argument(
        "--max-events", type=int, default=None, help="Stop after sending this many"
    )
    ap.add_argument(
        "--sleep-ms", type=int, default=0, help="Delay between events (throttle)"
    )
    ap.add_argument("--timeout-s", type=float, default=1.0, help="HTTP timeout")
    ap.add_argument(
        "--stop-on-error", action="store_true", help="Stop on first failed POST"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    stats = replay_oplog_to_bridge(
        oplog_path=Path(args.oplog),
        bridge_url=args.bridge,
        since_seq=args.since_seq,
        until_seq=args.until_seq,
        kg_graph_type=args.kg_graph_type,
        max_events=args.max_events,
        sleep_ms=args.sleep_ms,
        timeout_s=args.timeout_s,
        stop_on_error=args.stop_on_error,
        verbose=args.verbose,
    )

    print(f"sent={stats.sent} skipped={stats.skipped} failed={stats.failed}")
    return 0 if stats.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""
python replay_oplog_to_bridge.py \
  --oplog ./your.oplog.jsonl \
  --bridge http://127.0.0.1:8787 \
  --since-seq 0 \
  --sleep-ms 2 \
  --kg-graph-type conversation \
  -v

"""
