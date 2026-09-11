from __future__ import annotations

from pathlib import Path

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore


pytestmark = pytest.mark.core


def _enqueue(meta: object) -> None:
    meta.enqueue_index_job(
        job_id="candidate-job",
        namespace="maintenance",
        entity_kind="maintenance",
        entity_id="doc-1",
        index_kind="parse",
        op="UPSERT",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_first_live_claim_wins_and_result_is_immutable(backend: str, tmp_path: Path) -> None:
    meta = InMemoryMetaStore() if backend == "memory" else EngineSQLite(tmp_path / "meta")
    if backend == "sqlite":
        meta.ensure_initialized()
    _enqueue(meta)
    claim = meta.claim_index_jobs(limit=1, lease_seconds=30, namespace="maintenance")[0]
    winner = meta.accept_index_job_result(
        claim.job_id,
        claim_token=claim.claim_token or "",
        result_json='{"winner":1}',
        result_sha256="digest-1",
    )
    assert winner["status"] == "accepted"
    loser = meta.accept_index_job_result(
        claim.job_id,
        claim_token="stale-worker",
        result_json='{"winner":2}',
        result_sha256="digest-2",
    )
    assert loser["status"] == "existing"
    assert loser["result_json"] == '{"winner":1}'
