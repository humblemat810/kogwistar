from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import re

import pytest


pytestmark = [pytest.mark.ci, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "contracts" / "rust-port-v1.json"
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
CONSUMER_IMPORT_INVENTORY_PATH = (
    ROOT / "contracts" / "inventory" / "consumer-imports-v1.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_ci_full_uses_postgres_image_with_pgvector_extension() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'GKE_TEST_PG_IMAGE: "pgvector/pgvector:pg16"' in workflow
    assert "ci_full and not slow and not manual and not llm_real and not requires_ollama" in workflow


def test_ci_uses_persisted_inspectable_python_smoke_scripts() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python scripts/adr015_package_import_smoke.py" in workflow
    assert "python scripts/adr015_native_wheel_smoke.py --wheelhouse wheelhouse" in workflow
    assert "python -c" not in workflow
    assert "python -P -c" not in workflow


def test_rust_port_manifest_has_required_authority_and_suite_gates() -> None:
    manifest = _manifest()

    assert manifest["manifest_version"] == 1
    assert manifest["status"] == "frozen-for-port"
    assert manifest["implementation_modes"] == ["python", "shadow", "rust"]
    assert [item["name"] for item in manifest["suite_layers"]] == [
        "core",
        "parser",
        "sink",
        "application",
    ]
    assert manifest["storage_contracts"]["automatic_startup_mutations"] is False
    assert manifest["storage_contracts"]["chroma_single_writer_preserved"] is True
    assert manifest["storage_contracts"]["postgres_pgvector_same_transaction"] is True
    assert {
        "resolved_package_file",
        "package_version",
        "implementation_mode",
        "rust_extension_version",
        "core_commit",
        "application_commit",
        "parser_commit",
        "sink_commit",
        "backend",
        "storage_root",
        "candidate_source_sha256",
    } <= set(
        manifest["reference_application"]["candidate_identity_fields"]
    )
    evidence = manifest["phase_0_evidence"]
    for relative_path in [
        evidence["consumer_import_inventory"],
        evidence["benchmark_baseline"],
        *evidence["golden_fixtures"],
    ]:
        assert (ROOT / relative_path).is_file(), relative_path
    assert set(evidence["ci_jobs"]) == {
        "rust",
        "native-wheel-smoke",
        "rust-port-four-layer-compat",
    }
    assert all(
        len(commit) == 40
        for commit in manifest["reference_application"]["pinned_commits"].values()
    )


def test_passing_performance_and_public_routing_unlock_sqlite_meta_store() -> None:
    manifest = _manifest()
    persistent = manifest["current_persistent_evidence"]
    runtime = manifest["runtime_serving_evidence"]

    persistent_report = json.loads(
        (ROOT / persistent["benchmark_report"]).read_text(encoding="utf-8")
    )
    assert persistent["gate_status"] == "passed"
    assert persistent["cutover_use"] == "sqlite-meta-authority-verified"
    assert persistent_report["gate"]["status"] == "passed"
    sqlite_owner = next(
        item
        for item in manifest["capability_ownership"]
        if item["capability"] == "sqlite-meta"
    )
    assert sqlite_owner["rust_cutover_ready"] is True
    runtime_report = json.loads((ROOT / runtime["benchmark_report"]).read_text(encoding="utf-8"))
    assert runtime_report["gate"]["status"] == "passed"
    assert runtime_report["history_multiplier"] == 100


def test_test_parallelism_evidence_keeps_xdist_opt_in() -> None:
    manifest = _manifest()
    evidence = manifest["test_parallelism_evidence"]
    report = json.loads((ROOT / evidence["benchmark_report"]).read_text())

    assert evidence["gate_status"] == report["status"] == "passed"
    assert evidence["container_default"] == {
        "shards": 3,
        "pytest_xdist_workers": 0,
    }
    assert report["results"]["application_container_shards"]["speedup"] > 1
    assert report["results"]["core_pytest_xdist"]["speedup"] < 1


def test_four_layer_ci_uses_persisted_three_container_fast_profiles() -> None:
    workflow = (ROOT / ".github" / "workflows" / "rust-port-compat.yml").read_text(
        encoding="utf-8"
    )

    assert "scripts/rust_port_build_wheel.py" in workflow
    assert workflow.count("scripts/rust_port_container_compat.py") == 2
    assert 'for PROFILE in feature regression; do' in workflow
    assert '--profile "$PROFILE"' in workflow
    assert workflow.count("--shards 3") == 2
    assert "--pytest-workers" not in workflow


def test_server_cutover_ledger_count_matches_rust_frozen_route_inventory() -> None:
    rust_api = (
        ROOT / "rust" / "crates" / "kogwistar-api" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")
    pending = re.search(
        r"pub const PENDING_SERVER_CUTOVER_ROUTES:.*?= &\[(.*?)\];",
        rust_api,
        flags=re.DOTALL,
    )
    assert pending is not None
    pending_count = len(re.findall(r'"(?:DELETE|GET|POST)",', pending.group(1)))
    status = (ROOT / "kogwistar" / "docs" / "ADR-015-implementation-status.md").read_text(
        encoding="utf-8"
    )

    assert pending_count == 16
    assert f"the {pending_count} currently frozen" in status


def test_versioned_server_and_syscall_deferrals_match_rust_frozen_inventory() -> None:
    manifest = _manifest()
    deferrals = manifest["server_operation_deferrals"]
    assert deferrals["contract_version"] == 1
    assert deferrals["status"] == "versioned-intentionally-deferred"
    assert deferrals["owner"] == "python-rollback-deployment"

    rust_api = (
        ROOT / "rust" / "crates" / "kogwistar-api" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")
    pending_routes = re.search(
        r"pub const PENDING_SERVER_CUTOVER_ROUTES:.*?= &\[(.*?)\];",
        rust_api,
        flags=re.DOTALL,
    )
    pending_syscalls = re.search(
        r"pub const PENDING_SYSCALL_CUTOVER_OPS:.*?= &\[(.*?)\];",
        rust_api,
        flags=re.DOTALL,
    )
    assert pending_routes is not None
    assert pending_syscalls is not None
    rust_routes = set(
        re.findall(r'"([A-Z]+)"\s*,\s*"([^"]+)"', pending_routes.group(1))
    )
    rust_syscalls = set(re.findall(r'"([a-z_]+)"', pending_syscalls.group(1)))
    manifest_routes = {
        (operation["method"], operation["path"])
        for group in deferrals["route_groups"]
        for operation in group["operations"]
    }
    manifest_syscalls = set(deferrals["syscall_group"]["operations"])

    assert len(manifest_routes) == sum(
        len(group["operations"]) for group in deferrals["route_groups"]
    )
    assert manifest_routes == rust_routes
    assert manifest_syscalls == rust_syscalls
    for group in [*deferrals["route_groups"], deferrals["syscall_group"]]:
        assert group["owner"] == "python"
        assert group["status"] == "intentionally-deferred"
        assert group["required_authority"]
        assert group["exit_evidence"]


def test_each_authoritative_capability_has_owner_and_rollback() -> None:
    manifest = _manifest()
    capabilities = manifest["capability_ownership"]
    expected = {
        "deterministic-contracts",
        "sqlite-meta",
        "postgres-sequence-event-log",
        "projections-snapshots-run-registry",
        "queues-leases-lanes",
        "graph-pgvector",
        "chroma-adapter",
        "workflow-runtime",
        "server-rest-sse-mcp-cli",
    }

    assert {item["capability"] for item in capabilities} == expected
    for item in capabilities:
        assert item["configuration"].startswith("KOGWISTAR_IMPL_")
        assert item["current_authoritative_writer"]
        assert item["target_authoritative_writer"]
        assert isinstance(item.get("rust_cutover_ready"), bool)
        assert item["rollback"]


def test_rust_port_manifest_preserves_consumer_event_envelopes() -> None:
    manifest = _manifest()
    events = {item["type"]: item for item in manifest["event_contracts"]}

    assert {"entity.upsert", "entity.tombstone", "entity_event"} <= events.keys()
    assert set(events["entity.tombstone"]["accepted_aliases"]) == {
        "entity.delete",
        "entity.remove",
    }
    assert events["entity_event"]["idempotency_key"] == "event_id"


def test_consumer_import_classifications_use_approved_statuses() -> None:
    manifest = _manifest()
    approved = {
        "preserved",
        "adapter-backed",
        "deprecated",
        "intentionally-deferred",
    }
    entries = manifest["consumer_import_classifications"]

    assert entries
    assert all(item["status"] in approved for item in entries)
    keys = [(item["module"], item["symbol"]) for item in entries]
    assert len(keys) == len(set(keys))


def test_committed_consumer_import_inventory_is_fully_classified() -> None:
    inventory = json.loads(
        CONSUMER_IMPORT_INVENTORY_PATH.read_text(encoding="utf-8")
    )

    assert inventory["contract_version"] == _manifest()["contract_version"]
    assert inventory["summary"]["records"] > 0
    assert inventory["summary"]["unclassified"] == 0
    assert all(
        details["commit"] for details in inventory["repositories"].values()
    )


def test_committed_non_openapi_goldens_have_no_drift() -> None:
    from scripts.rust_port_goldens import GOLDEN_ROOT, _encoded, golden_payloads

    for name, payload in golden_payloads(include_openapi=False).items():
        assert (GOLDEN_ROOT / name).read_text(encoding="utf-8") == _encoded(payload)


def test_committed_database_ddl_baseline_has_no_drift() -> None:
    from scripts.rust_port_schema_baseline import OUTPUT, _encoded, build_baseline

    baseline = build_baseline()
    assert baseline["sqlite"]["objects"]
    assert baseline["postgresql"]["statements"]
    assert OUTPUT.read_text(encoding="utf-8") == _encoded(baseline)


def test_grounded_parser_golden_validates_and_preserves_offsets() -> None:
    from kogwistar.engine_core.models import Document, Edge, Node

    fixture = json.loads(
        (ROOT / "contracts" / "golden" / "parser-grounded-output.json").read_text(
            encoding="utf-8"
        )
    )
    document = Document.model_validate(fixture["document"])
    nodes = [Node.model_validate(item) for item in fixture["nodes"]]
    edges = [Edge.model_validate(item) for item in fixture["edges"]]

    assert [node.mentions[0].spans[0].excerpt for node in nodes] == ["Alpha", "Beta"]
    assert document.content[0:5] == nodes[0].mentions[0].spans[0].excerpt
    assert document.content[12:16] == nodes[1].mentions[0].spans[0].excerpt
    assert document.content[6:11] == edges[0].mentions[0].spans[0].excerpt


def test_consumer_projection_goldens_cover_scope_delete_rebuild_and_drift() -> None:
    scoped = json.loads(
        (ROOT / "contracts" / "golden" / "scoped-graph-snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    vault = json.loads(
        (ROOT / "contracts" / "golden" / "vault-projection.json").read_text(
            encoding="utf-8"
        )
    )

    assert all(
        entity["metadata"]["workspace_id"] == scoped["workspace_id"]
        and entity["metadata"]["graph_space"] == scoped["graph_space"]
        for entity in [*scoped["nodes"], *scoped["edges"]]
    )
    assert vault["expected"]["deleted_paths"]
    assert vault["expected"]["full_rebuild_equals_incremental"] is True
    assert vault["expected"]["drift"] == []


@pytest.mark.ci_full
def test_committed_openapi_golden_has_no_drift() -> None:
    from scripts.rust_port_goldens import GOLDEN_ROOT, _encoded, golden_payloads

    openapi = golden_payloads(include_openapi=True)["openapi.json"]
    assert (GOLDEN_ROOT / "openapi.json").read_text(encoding="utf-8") == _encoded(
        openapi
    )


def test_compat_report_identity_fingerprint_is_canonical() -> None:
    from scripts.rust_port_compat import _identity_fingerprint

    left = {"mode": "shadow", "commits": {"core": "abc", "app": "def"}}
    right = {"commits": {"app": "def", "core": "abc"}, "mode": "shadow"}
    expected = hashlib.sha256(
        json.dumps(
            left, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    assert _identity_fingerprint(left) == expected
    assert _identity_fingerprint(right) == expected


def test_compat_candidate_source_fingerprint_is_stable() -> None:
    from scripts.rust_port_compat import _candidate_source_fingerprint

    first = _candidate_source_fingerprint()
    assert len(first) == 64
    assert _candidate_source_fingerprint() == first


def test_consumer_inventory_scans_and_classifies_imports(tmp_path: Path) -> None:
    from scripts.rust_port_inventory import _classification_index, _scan_file

    source = tmp_path / "consumer.py"
    source.write_text(
        "import kogwistar.runtime\n"
        "from kogwistar import Node\n"
        "from kogwistar.runtime import WorkflowRuntime\n",
        encoding="utf-8",
    )
    manifest = _manifest()
    index = _classification_index(manifest)

    assert _scan_file(source) == [
        ("kogwistar.runtime", "<module>", 1),
        ("kogwistar", "Node", 2),
        ("kogwistar.runtime", "WorkflowRuntime", 3),
    ]
    assert index[("kogwistar", "Node")] == "preserved"
    assert index[("kogwistar.runtime", "WorkflowRuntime")] == "preserved"


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        (item["module"], symbol)
        for item in _manifest()["python_facade"]["modules"]
        for symbol in item["symbols"]
    ],
)
def test_frozen_python_facade_symbols_are_importable(
    module_name: str, symbol: str
) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"missing frozen contract: {module_name}.{symbol}"
