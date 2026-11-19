from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "durable_store_capability_gate.py"


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("durable_store_capability_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capability_gate_is_persisted_and_covers_every_pending_store_owner() -> None:
    gate = _gate()

    assert set(gate.CAPABILITY_GROUPS) == {
        "postgres-sequence-event-log",
        "projections-snapshots-run-registry",
        "queues-leases-lanes",
        "graph-pgvector",
    }
    assert all(tests for tests in gate.CAPABILITY_GROUPS.values())
    assert any(
        "test_rust_store_postgres_differential.py" in test
        for test in gate.CAPABILITY_GROUPS["postgres-sequence-event-log"]
    )
    assert any(
        "test_postgres_projection_valid_invalid_fixture_matches_python_rust" in test
        for test in gate.CAPABILITY_GROUPS["projections-snapshots-run-registry"]
    )
    assert any(
        "test_postgres_projection_resume_rejects_stale_checkpoint_after_restart"
        in test
        for test in gate.CAPABILITY_GROUPS["projections-snapshots-run-registry"]
    )
    assert any(
        "test_rust_store_lane_messages_differential.py" in test
        for test in gate.CAPABILITY_GROUPS["queues-leases-lanes"]
    )
    assert any(
        "test_rust_store_graph_mutation_differential.py" in test
        for test in gate.CAPABILITY_GROUPS["graph-pgvector"]
    )
    graph_tests = gate.CAPABILITY_GROUPS["graph-pgvector"]
    assert any("test_python_node_and_edge_payloads_round_trip" in test for test in graph_tests)
    assert any("test_graph_mutation_scope_vector_contract_and_fault_rollback" in test for test in graph_tests)
    assert any("test_graph_mutation_fresh_rust_schema_and_validation_rollback" in test for test in graph_tests)
    assert any("test_rust_postgres_graph_read_tools_match_python_over_shared_projection" in test for test in graph_tests)
    assert any("test_rust_native_promotion_and_python_recovery_are_both_available" in test for test in graph_tests)


def test_capability_gate_reuses_live_postgres_skip_rejection() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    store_gate = (ROOT / "scripts" / "durable_store_gate.py").read_text(encoding="utf-8")

    assert "live_postgres=True" in source
    assert "not-promoted-local-evidence-only" in source
    assert "store_gate._run_group" in source
    assert 'command.extend(("-k", "postgres"))' not in store_gate
    assert 'counts["skipped"] == 0' in store_gate
