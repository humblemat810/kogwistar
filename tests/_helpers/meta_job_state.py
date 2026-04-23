from __future__ import annotations

import sqlite3
import time
from typing import Any


def set_index_job_state(
    meta_sqlite: Any,
    txn: Any,
    *,
    job_id: str,
    status: str,
    lease_until: int | float | None = None,
    next_run_at: int | float | None = None,
    updated_at: int | float | None = None,
) -> None:
    now = int(time.time())
    updated_at_int = int(updated_at if updated_at is not None else now)
    lease_until_val = None if lease_until is None else int(lease_until)
    next_run_at_val = None if next_run_at is None else int(next_run_at)

    if isinstance(txn, sqlite3.Connection):
        txn.execute(
            "UPDATE index_jobs SET status=?, lease_until=?, next_run_at=?, updated_at=? WHERE job_id=?",
            (status, lease_until_val, next_run_at_val, updated_at_int, job_id),
        )
        return

    job = txn.state.index_jobs.get(str(job_id))
    if job is None:
        return
    job.status = status
    job.lease_until = lease_until_val
    job.next_run_at = next_run_at_val
    job.updated_at = updated_at_int
