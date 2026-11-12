from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci, pytest.mark.core]


def _mixed_evidence_is_executable(
    evidence_id: str,
    checklist_statuses: dict[str, str],
    fine_rows: dict[str, list[tuple[str, str, str]]],
) -> bool:
    gate_status = checklist_statuses.get(evidence_id)
    has_passing_fine_evidence = any(
        status == "PASS" and evidence.strip()
        for _, status, evidence in fine_rows.get(evidence_id, [])
    )
    return gate_status == "PASS" or (
        gate_status == "IN PROGRESS" and has_passing_fine_evidence
    )


def _capability_link_errors(
    crosswalk_capabilities: list[dict],
    matrix_capabilities: list[dict],
    checklist_rows: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    crosswalk_ids = [item.get("id") for item in crosswalk_capabilities]
    matrix_ids = [item.get("id") for item in matrix_capabilities]
    if len(crosswalk_ids) != len(set(crosswalk_ids)):
        errors.append("duplicate crosswalk capability id")
    if len(matrix_ids) != len(set(matrix_ids)):
        errors.append("duplicate support-matrix capability id")
    if set(crosswalk_ids) != set(matrix_ids):
        errors.append("missing or unexpected capability evidence link")

    for capability in matrix_capabilities:
        capability_id = capability.get("id", "<missing>")
        for lane in ("python", "rust", "mixed"):
            decision = capability.get(lane)
            if not isinstance(decision, dict):
                errors.append(f"{capability_id}/{lane} has no evidence decision")
                continue
            evidence_id = decision.get("evidence")
            if not isinstance(evidence_id, str) or not re.fullmatch(
                r"DL-[A-Z0-9-]+", evidence_id
            ):
                errors.append(f"{capability_id}/{lane} has non-gate evidence")
            elif evidence_id not in checklist_rows:
                errors.append(f"{capability_id}/{lane} has stale evidence {evidence_id}")
            elif (
                lane in {"rust", "mixed"}
                and decision.get("status") in {"enabled", "opt-in"}
                and "advertisement" in checklist_rows[evidence_id].lower()
                and "does not make" in checklist_rows[evidence_id].lower()
                and "compatible" in checklist_rows[evidence_id].lower()
            ):
                errors.append(
                    f"{capability_id}/{lane} cites route-advertisement-only evidence"
                )
    return errors


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "contracts" / "rust-port-v1.json"
DUAL_LANGUAGE_MANIFEST_PATH = (
    ROOT / "contracts" / "dual-language-capabilities-v1.json"
)
SUPPORT_MATRIX_PATH = ROOT / "contracts" / "dual-language-support-matrix-v1.json"
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
RUST_PORT_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "rust-port-compat.yml"
RUST_API_BUILD_SCRIPT_PATH = (
    ROOT / "rust" / "crates" / "kogwistar-api" / "build.rs"
)
CONSUMER_IMPORT_INVENTORY_PATH = (
    ROOT / "contracts" / "inventory" / "consumer-imports-v1.json"
)
DUAL_LANGUAGE_CHECKLIST_PATH = (
    ROOT / "kogwistar" / "docs" / "ADR-022-dual-language-capability-checklist.md"
)
DUAL_LANGUAGE_WORKLIST_PATH = (
    ROOT / "kogwistar" / "docs" / "ADR-022-dual-language-capability-worklist.md"
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _markdown_table_rows(path: Path, row_prefix: str) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(row_prefix):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _python_declarations(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    declarations: set[str] = set()

    def add_assignment_targets(target: ast.expr, prefix: str) -> None:
        if isinstance(target, ast.Name):
            declarations.add(f"{prefix}.{target.id}" if prefix else target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_assignment_targets(element, prefix)

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for statement in body:
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{statement.name}" if prefix else statement.name
                declarations.add(name)
                visit(statement.body, name)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    add_assignment_targets(target, prefix)
            elif isinstance(statement, ast.AnnAssign):
                add_assignment_targets(statement.target, prefix)

    visit(tree.body)
    return declarations


def _python_references_from_marker(value: str) -> list[tuple[str, str]]:
    path_pattern = re.compile(r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.py)#")
    matches = list(path_pattern.finditer(value))
    references: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        symbols = value[match.end() : end].strip("+;")
        parent = ""
        for symbol in symbols.split("+"):
            symbol = symbol.strip()
            if not symbol:
                continue
            if "." in symbol:
                parent = symbol.rpartition(".")[0]
                declaration = symbol
            else:
                declaration = f"{parent}.{symbol}" if parent else symbol
            references.append((match.group("path"), declaration))
    return references


def test_rust_python_parity_markers_resolve_to_python_declarations() -> None:
    marker_pattern = re.compile(r"PYTHON_PARITY_V1:.*?python=([^;]+)")
    marker_count = 0
    for rust_file in (ROOT / "rust").rglob("*.rs"):
        for line_number, line in enumerate(
            rust_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = marker_pattern.search(line)
            if match is None:
                continue
            marker_count += 1
            references = _python_references_from_marker(match.group(1))
            assert references, f"{rust_file}:{line_number}: missing Python symbol"
            declarations_by_path: dict[str, set[str]] = {}
            for relative_path, declaration in references:
                python_file = ROOT / relative_path
                assert python_file.is_file(), (
                    f"{rust_file}:{line_number}: missing Python reference file {relative_path}"
                )
                declarations = declarations_by_path.setdefault(
                    relative_path, _python_declarations(python_file)
                )
                matches = (
                    {declaration}
                    if declaration in declarations
                    else {
                        candidate
                        for candidate in declarations
                        if candidate.endswith(f".{declaration}")
                    }
                )
                assert len(matches) == 1, (
                    f"{rust_file}:{line_number}: {relative_path} has no unique declaration "
                    f"for {declaration}: {sorted(matches)}"
                )
    assert marker_count > 0, "expected checked-in PYTHON_PARITY_V1 markers"


def test_python_parity_marker_parser_expands_grouped_declarations() -> None:
    marker = (
        "kogwistar/server/chat_mcp.py#build_workflow_mcp.workflow_service_get"
        "+workflow_service_list+workflow_service_events"
        "+kogwistar/server/chat_service.py#ChatRunService.get_service"
        "+list_services+list_service_events"
    )
    assert _python_references_from_marker(marker) == [
        ("kogwistar/server/chat_mcp.py", "build_workflow_mcp.workflow_service_get"),
        ("kogwistar/server/chat_mcp.py", "build_workflow_mcp.workflow_service_list"),
        ("kogwistar/server/chat_mcp.py", "build_workflow_mcp.workflow_service_events"),
        ("kogwistar/server/chat_service.py", "ChatRunService.get_service"),
        ("kogwistar/server/chat_service.py", "ChatRunService.list_services"),
        ("kogwistar/server/chat_service.py", "ChatRunService.list_service_events"),
    ]


def test_dual_language_coarse_status_matches_all_mapped_fine_tasks() -> None:
    coarse_rows = _markdown_table_rows(DUAL_LANGUAGE_CHECKLIST_PATH, "| DL-")
    fine_rows = _markdown_table_rows(DUAL_LANGUAGE_WORKLIST_PATH, "| FW-")
    coarse_status = {
        row[0]: next(
            (status for status in ("IN PROGRESS", "NOT STARTED", "PASS", "FAIL", "BLOCKED")
             if row[1] == status or row[1].startswith(f"{status} ")),
            row[1].split()[0],
        )
        for row in coarse_rows
    }
    fine_by_coarse: dict[str, list[str]] = {}
    fine_ids: set[str] = set()
    parents: dict[str, tuple[str, list[str]]] = {}
    for row in fine_rows:
        fine_id, coarse_id, status = row[:3]
        assert fine_id not in fine_ids, f"duplicate fine tasklet ID {fine_id}"
        fine_ids.add(fine_id)
        assert coarse_id in coarse_status, f"{fine_id} references unknown coarse item {coarse_id}"
        if status == "PARENT":
            match = re.fullmatch(r"Parent scope; children: (FW-[A-Za-z0-9,-]+)", row[3])
            assert match is not None, f"{fine_id} must enumerate its non-counted leaf children"
            child_ids = match.group(1).split(",")
            assert child_ids and len(set(child_ids)) == len(child_ids)
            parents[fine_id] = (coarse_id, child_ids)
            continue
        assert status in {"NOT STARTED", "IN PROGRESS", "PASS", "FAIL", "BLOCKED"}
        if status in {"NOT STARTED", "IN PROGRESS"}:
            task_words = row[3].split()
            assert len(task_words) <= 21, (
                f"{fine_id} exceeds the 21-word open-tasklet description limit "
                f"({len(task_words)} words); keep one concise acceptance outcome"
            )
        fine_by_coarse.setdefault(coarse_id, []).append(status)

    rows_by_id = {row[0]: row for row in fine_rows}

    def descendant_leaves(parent_id: str, ancestry: frozenset[str] = frozenset()) -> set[str]:
        assert parent_id not in ancestry, f"parent hierarchy contains a cycle at {parent_id}"
        parent_coarse_id, child_ids = parents[parent_id]
        leaves: set[str] = set()
        for child_id in child_ids:
            child = rows_by_id.get(child_id)
            assert child is not None, f"{parent_id} references missing child {child_id}"
            assert child[1] == parent_coarse_id, f"{child_id} must stay under {parent_coarse_id}"
            if child[2] == "PARENT":
                leaves.update(descendant_leaves(child_id, ancestry | {parent_id}))
            else:
                leaves.add(child_id)
        assert len(leaves) >= 2, f"{parent_id} must group at least two descendant tasklets"
        return leaves

    for parent_id in parents:
        descendant_leaves(parent_id)

    open_coarse = {
        coarse_id
        for coarse_id, status in coarse_status.items()
        if status in {"IN PROGRESS", "NOT STARTED", "FAIL", "BLOCKED"}
    }
    assert open_coarse <= fine_by_coarse.keys(), (
        "every open coarse gate must have at least one fine task: "
        f"{sorted(open_coarse - fine_by_coarse.keys())}"
    )
    for coarse_id, statuses in fine_by_coarse.items():
        all_fine_pass = all(status == "PASS" for status in statuses)
        coarse_pass = coarse_status[coarse_id] == "PASS"
        assert coarse_pass == all_fine_pass, (
            f"{coarse_id}: coarse status {coarse_status[coarse_id]!r} must match "
            f"mapped fine statuses {statuses!r}"
        )

    worklist = DUAL_LANGUAGE_WORKLIST_PATH.read_text(encoding="utf-8")
    coarse_snapshot = re.search(
        r"\*\*Coarse checklist:\*\* \*\*(\d+)/(\d+) PASS \(([\d.]+)%\)\*\*\.",
        worklist,
    )
    fine_snapshot = re.search(
        r"\*\*Fine worklist:\*\* \*\*(\d+)/(\d+) PASS \(([\d.]+)%\)\*\*; "
        r"\*\*(\d+) IN PROGRESS\*\*, \*\*(\d+) NOT STARTED\*\*, "
        r"\*\*(\d+) BLOCKED\*\*\.",
        worklist,
    )
    assert coarse_snapshot is not None
    assert fine_snapshot is not None
    coarse_pass_count = sum(status == "PASS" for status in coarse_status.values())
    fine_statuses = [row[2] for row in fine_rows if row[2] != "PARENT"]
    fine_pass_count = sum(status == "PASS" for status in fine_statuses)
    assert tuple(map(int, coarse_snapshot.groups()[:2])) == (
        coarse_pass_count,
        len(coarse_status),
    )
    assert float(coarse_snapshot.group(3)) == round(
        100 * coarse_pass_count / len(coarse_status), 1
    )
    assert tuple(map(int, fine_snapshot.groups()[:2])) == (
        fine_pass_count,
        len(fine_statuses),
    )
    assert float(fine_snapshot.group(3)) == round(
        100 * fine_pass_count / len(fine_statuses), 1
    )
    assert tuple(map(int, fine_snapshot.groups()[3:])) == (
        sum(status == "IN PROGRESS" for status in fine_statuses),
        sum(status == "NOT STARTED" for status in fine_statuses),
        sum(status == "BLOCKED" for status in fine_statuses),
    )


def test_native_operation_backend_evidence_matrix_is_complete() -> None:
    inventory = json.loads(
        (
            ROOT
            / "contracts"
            / "golden"
            / "runtime-native-operation-inventory-v1.json"
        ).read_text(encoding="utf-8")
    )
    worklist_rows = _markdown_table_rows(DUAL_LANGUAGE_WORKLIST_PATH, "| FW-")
    rows_by_id = {row[0]: row for row in worklist_rows}
    operation_suffix = {
        "noop": "NOOP",
        "increment-counter": "COUNTER",
        "emit-result": "RESULT",
        "agent.event.request": "REQUEST",
        "agent.event.receive": "RECEIVE",
        "agent.event.ack": "ACK",
    }
    assert {item["operation"] for item in inventory["operations"]} == set(
        operation_suffix
    )
    test_by_backend_and_operation = {
        ("SQLITE", "NOOP"): "sqlite_native_noop_result_is_durable_through_runtime_route",
        ("SQLITE", "COUNTER"): "sqlite_native_runtime_route_uses_shared_result_reducer",
        ("SQLITE", "RESULT"): "sqlite_native_runtime_route_uses_shared_result_reducer",
        ("SQLITE", "REQUEST"): "sqlite_native_provider_event_request_persists_semantic_events",
        ("SQLITE", "RECEIVE"): "sqlite_native_provider_event_request_persists_semantic_events",
        ("SQLITE", "ACK"): "sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically",
        ("02A", "NOOP"): "postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available",
        ("02A", "COUNTER"): "postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available",
        ("02A", "RESULT"): "postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available",
        ("02A", "REQUEST"): "postgres_provider_event_ack_persists_python_causal_links_atomically",
        ("02A", "RECEIVE"): "postgres_provider_event_ack_persists_python_causal_links_atomically",
        ("02A", "ACK"): "postgres_provider_event_ack_persists_python_causal_links_atomically",
        ("02B", "NOOP"): "in_memory_inline_native_noop_completes_claimed_run_atomically",
        ("02B", "COUNTER"): "in_memory_inline_native_counter_matches_shared_effect_atomically",
        ("02B", "RESULT"): "in_memory_inline_native_result_is_persisted_and_completes_run",
        ("02B", "REQUEST"): "in_memory_native_provider_request_persists_semantic_events_and_keeps_deltas_transient",
        ("02B", "RECEIVE"): "in_memory_native_receive_claims_visible_event_and_routes_it",
        ("02B", "ACK"): "provider_event_acknowledgement_and_run_trace_are_atomic_and_idempotent",
    }
    rust_test_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "rust" / "crates").rglob("*.rs")
    )

    def leaf_evidence(task: list[str]) -> list[list[str]]:
        if task[2] != "PARENT":
            return [task]
        children = re.search(r"Parent scope; children: (FW-[A-Za-z0-9,-]+)", task[3])
        assert children is not None, f"{task[0]} has no declared children"
        return [
            leaf
            for child_id in children.group(1).split(",")
            for leaf in leaf_evidence(rows_by_id[child_id])
        ]

    for item in inventory["operations"]:
        suffix = operation_suffix[item["operation"]]
        for backend in ("SQLITE", "02A", "02B"):
            prefix = (
                "FW-13-02-SQLITE"
                if backend == "SQLITE"
                else f"FW-13-{backend}"
            )
            task_id = f"{prefix}-{suffix}"
            task = rows_by_id.get(task_id)
            assert task is not None, f"missing native matrix tasklet {task_id}"
            evidence_rows = leaf_evidence(task)
            assert evidence_rows and all(row[2] == "PASS" for row in evidence_rows), (
                f"native matrix tasklet {task_id} has unfinished leaves"
            )
            test_name = test_by_backend_and_operation[(backend, suffix)]
            assert test_name in rust_test_sources, (
                f"{task_id} references missing Rust test {test_name}"
            )
            assert any(test_name in " ".join(row[3:]) for row in evidence_rows), (
                f"{task_id} does not link the Rust test {test_name}"
            )

def test_dual_language_open_tasklets_use_one_bounded_work_unit() -> None:
    worklist = DUAL_LANGUAGE_WORKLIST_PATH.read_text(encoding="utf-8")
    assert "every counted leaf row is exactly one unit of work" in worklist
    assert "Target 20 minutes, hard cap 30 minutes" in worklist
    assert "| Active: <=30 min`" in worklist
    assert "The audited splits are FW-06-01 (seven durable resource families" in worklist
    assert "each child has one family/backend cell and one oracle" in worklist

    in_progress_rows = [
        line
        for line in worklist.splitlines()
        if line.startswith("| FW-") and "| IN PROGRESS |" in line
    ]
    required_fields = ("Invariant:", "Cell:", "Oracle:", "Check:", "Active:")
    for line in in_progress_rows:
        missing = [field for field in required_fields if field not in line]
        assert not missing, f"in-progress tasklet lacks bounded definition fields: {missing}"


def test_ci_full_uses_postgres_image_with_pgvector_extension() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'GKE_TEST_PG_IMAGE: "pgvector/pgvector:pg16"' in workflow
    assert "ci_full and not slow and not manual and not llm_real and not requires_ollama" in workflow


def test_rust_ci_keeps_the_native_api_dependency_closure_python_free() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "Keep the native API dependency closure Python-free" in workflow
    assert (
        'cargo tree --package kogwistar-api --edges normal --prefix none --all-features'
        in workflow
    )
    assert "(kogwistar-python|pyo3(-[^ ]+)?|pythonize|cpython|rust-cpython(-[^ ]+)?) v" in workflow


def test_rust_ci_provides_live_backends_for_native_interoperability() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "image: chromadb/chroma:1.5.9" in workflow
    assert "KOGWISTAR_TEST_CHROMA_URL: http://localhost:8000" in workflow
    assert "Wait for Chroma service" in workflow
    assert "curl --silent --fail http://127.0.0.1:8000/api/v2/heartbeat" in workflow
    assert "cargo test --workspace --all-targets --locked --" in workflow
    assert "--include-ignored" in workflow
    assert (
        "tests/server/test_rust_api_chroma_interop_live.py::"
        "test_python_chroma_projection_is_read_and_reconciled_by_rust_server"
    ) in workflow
    assert (
        "tests/server/test_rust_api_chroma_interop_live.py::"
        "test_python_stage1_is_promoted_by_rust_native_worker"
    ) in workflow
    assert "tests/server/test_rust_api_postgres_worker_interop_live.py::" in workflow
    assert "test_rust_native_promotion_and_python_recovery_are_both_available" in workflow
    assert "if tests != 3 or skipped:" in workflow


def test_ci_uses_persisted_inspectable_python_smoke_scripts() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python scripts/package_import_provenance_smoke.py" in workflow
    assert "python scripts/native_wheel_contract_smoke.py --wheelhouse wheelhouse" in workflow
    assert "python scripts/pypy_pydantic_core_smoke.py --metadata-version" in workflow
    assert "python scripts/pypy_ffi_symbol_audit.py" in workflow
    assert "python scripts/pypy_native_extension_dlopen.py" in workflow
    assert 'python "$GITHUB_WORKSPACE/scripts/pypy_native_extension_smoke.py"' in workflow
    assert (
        "python scripts/pypy_pydantic_core_smoke.py \\\n"
        '            --verify --expected-version "$version"'
    ) in workflow
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


def test_four_layer_ci_core_pin_matches_reference_application_contract() -> None:
    manifest = _manifest()
    workflow = RUST_PORT_WORKFLOW_PATH.read_text(encoding="utf-8")
    core_pin = manifest["reference_application"]["pinned_commits"]["core_pin"]

    assert f"ref: {core_pin}" in workflow
    assert core_pin == "3a9dfb8951175a35f7be9299208ac31d849c288f"


def test_server_cutover_ledger_count_matches_rust_frozen_route_inventory() -> None:
    deferrals = _manifest()["server_operation_deferrals"]
    pending_count = sum(
        len(group["operations"]) for group in deferrals["route_groups"]
    )
    status = (ROOT / "kogwistar" / "docs" / "ADR-015-implementation-status.md").read_text(
        encoding="utf-8"
    )

    assert pending_count == 1
    assert f"the {pending_count} currently frozen" in status


def test_versioned_server_and_syscall_deferrals_match_rust_frozen_inventory() -> None:
    manifest = _manifest()
    deferrals = manifest["server_operation_deferrals"]
    assert deferrals["contract_version"] == 1
    assert deferrals["status"] == "versioned-intentionally-deferred"
    assert deferrals["owner"] == "python-rollback-deployment"

    build_script = RUST_API_BUILD_SCRIPT_PATH.read_text(encoding="utf-8")
    manifest_routes = {
        (operation["method"], operation["path"])
        for group in deferrals["route_groups"]
        for operation in group["operations"]
    }
    manifest_syscalls = set(deferrals["syscall_group"]["operations"])

    assert len(manifest_routes) == sum(
        len(group["operations"]) for group in deferrals["route_groups"]
    )
    assert len(manifest_syscalls) == len(deferrals["syscall_group"]["operations"])
    assert 'join("../../../contracts/rust-port-v1.json")' in build_script
    assert '"server_operation_deferrals"' in build_script
    assert "PENDING_SERVER_CUTOVER_ROUTES" in build_script
    assert "PENDING_SYSCALL_CUTOVER_OPS" in build_script
    assert "duplicate deferred route" in build_script
    assert "duplicate deferred syscall operation" in build_script
    gate_path = ROOT / "scripts" / "rust_port_release_gate.py"
    gate_source = gate_path.read_text(encoding="utf-8")
    assert f"SERVER_DEFERRAL_ROUTE_COUNT_V1 = {len(manifest_routes)}" in gate_source
    assert f"SERVER_DEFERRAL_SYSCALL_COUNT_V1 = {len(manifest_syscalls)}" in gate_source
    for group in [*deferrals["route_groups"], deferrals["syscall_group"]]:
        assert group["owner"] == "python"
        assert group["status"] == "intentionally-deferred"
        assert group["required_authority"]
        assert group["exit_evidence"]


def test_dual_language_crosswalk_matches_frozen_ownership_and_rust_gaps() -> None:
    port_manifest = _manifest()
    crosswalk = json.loads(
        DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert crosswalk["schema_version"] == 1
    assert crosswalk["status"] == "inventory-in-progress"
    port_capabilities = {
        item["capability"]: item for item in port_manifest["capability_ownership"]
    }
    crosswalk_capabilities = {
        item["id"]: item for item in crosswalk["capabilities"]
    }
    assert crosswalk_capabilities.keys() == port_capabilities.keys()
    profiles = crosswalk["conformance_profiles"]
    for capability_id, item in crosswalk_capabilities.items():
        ownership = port_capabilities[capability_id]
        assert item["rust_port_manifest_id"] == capability_id
        assert item["conformance_profile"] == capability_id
        assert profiles[capability_id]
        assert all(isinstance(suite, str) and suite for suite in profiles[capability_id])
        expected_rust_coverage = (
            "python-only"
            if ownership["target_authoritative_writer"] == "python-adapter"
            else "cutover-ready"
            if ownership["rust_cutover_ready"]
            else "not-cutover-ready"
        )
        assert item["rust_coverage"] == expected_rust_coverage
        assert item["native_python_path"] == "not-verified"
        if expected_rust_coverage != "python-only":
            assert item["native_rust_path"] == "not-verified"

    deferrals = port_manifest["server_operation_deferrals"]
    expected_routes = {
        (operation["method"], operation["path"])
        for group in deferrals["route_groups"]
        for operation in group["operations"]
    }
    expected_syscalls = set(deferrals["syscall_group"]["operations"])
    gaps = crosswalk["rust_server_gaps"]
    assert {tuple(route) for route in gaps["pending_routes"]} == expected_routes
    assert set(gaps["pending_syscalls"]) == expected_syscalls


def test_capability_evidence_links_reject_missing_stale_duplicate_and_route_refs(
) -> None:
    crosswalk = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    checklist_rows = {
        row[0]: " | ".join(row[1:])
        for row in _markdown_table_rows(DUAL_LANGUAGE_CHECKLIST_PATH, "| DL-")
    }
    crosswalk_capabilities = crosswalk["capabilities"]
    matrix_capabilities = matrix["capabilities"]

    assert not _capability_link_errors(
        crosswalk_capabilities, matrix_capabilities, checklist_rows
    )

    assert "missing or unexpected capability evidence link" in _capability_link_errors(
        crosswalk_capabilities, matrix_capabilities[:-1], checklist_rows
    )
    assert "duplicate support-matrix capability id" in _capability_link_errors(
        crosswalk_capabilities,
        [*matrix_capabilities, matrix_capabilities[0]],
        checklist_rows,
    )
    assert "duplicate crosswalk capability id" in _capability_link_errors(
        [*crosswalk_capabilities, crosswalk_capabilities[0]],
        matrix_capabilities,
        checklist_rows,
    )

    stale_matrix = [dict(item) for item in matrix_capabilities]
    stale_matrix[0] = {
        **stale_matrix[0],
        "python": {"status": "enabled", "evidence": "DL-999"},
    }
    assert any(
        "stale evidence DL-999" in error
        for error in _capability_link_errors(
            crosswalk_capabilities, stale_matrix, checklist_rows
        )
    )

    route_matrix = [dict(item) for item in matrix_capabilities]
    route_matrix[0] = {
        **route_matrix[0],
        "rust": {"status": "opt-in", "evidence": "POST /api/workflow/runs"},
    }
    assert any(
        "has non-gate evidence" in error
        for error in _capability_link_errors(
            crosswalk_capabilities, route_matrix, checklist_rows
        )
    )

    advertisement_matrix = [dict(item) for item in matrix_capabilities]
    advertisement_matrix[-1] = {
        **advertisement_matrix[-1],
        "rust": {"status": "opt-in", "evidence": "DL-16N"},
    }
    assert any(
        "route-advertisement-only evidence" in error
        for error in _capability_link_errors(
            crosswalk_capabilities, advertisement_matrix, checklist_rows
        )
    )


def test_dual_language_surface_inventory_is_current() -> None:
    inventory_path = ROOT / "contracts" / "dual-language-surface-inventory-v1.json"
    generator = ROOT / "scripts" / "generate_dual_language_surface_inventory.py"
    crosswalk = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert (
        crosswalk["sources_of_truth"]["public_operation_inventory"]
        == "contracts/dual-language-surface-inventory-v1.json"
    )
    assert inventory["schema_version"] == 1
    assert inventory["status"] == "inventory-in-progress"
    assert inventory["counts"]["http_operations"] == len(inventory["http_operations"])
    assert inventory["counts"]["mcp_operations"] == len(inventory["mcp_operations"])
    http_operations = inventory["http_operations"]
    http_keys = [
        (operation["method"], operation["path"]) for operation in http_operations
    ]
    openapi = json.loads(
        (ROOT / "contracts" / "golden" / "openapi.json").read_text(encoding="utf-8")
    )
    expected_http = {
        (method.upper(), path)
        for path, path_item in openapi["paths"].items()
        for method in path_item
        if method.lower()
        in {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
    }
    assert len(http_keys) == len(set(http_keys))
    assert set(http_keys) == expected_http
    assert all(operation["python"] == "openapi" for operation in http_operations)

    route_map = json.loads(
        (ROOT / "contracts" / "http-route-tasklet-map-v1.json").read_text(
            encoding="utf-8"
        )
    )
    worklist = DUAL_LANGUAGE_WORKLIST_PATH.read_text(encoding="utf-8")
    mappings = route_map["operations"]
    mapping_keys = [(item["method"], item["path"]) for item in mappings]
    assert route_map["schema_version"] == 1
    assert len(mapping_keys) == len(set(mapping_keys))
    assert set(mapping_keys) == set(http_keys)
    tasklet_ids = [item["tasklet_id"] for item in mappings]
    assert len(tasklet_ids) == len(set(tasklet_ids))
    for tasklet_id in tasklet_ids:
        assert len(re.findall(rf"^\| {re.escape(tasklet_id)} \|", worklist, re.M)) == 1

    unsupported = [item for item in mappings if item.get("classification") == "unsupported"]
    assert len(unsupported) == 1
    assert unsupported[0]["tasklet_id"] == "FW-03-03B"
    supported_tasklet_ids = {
        item["tasklet_id"] for item in mappings if item not in unsupported
    }
    declared_route_tasklet_ids = set(
        re.findall(r"^\| (FW-03-03A-HTTP-(?:GET|POST|DELETE)-[0-9a-f]{8}) \|", worklist, re.M)
    )
    assert declared_route_tasklet_ids == supported_tasklet_ids
    rust_by_key = {
        (item["method"], item["path"]): item["rust"] for item in http_operations
    }
    for item in mappings:
        key = (item["method"], item["path"])
        if item.get("classification") == "unsupported":
            assert rust_by_key[key] == "pending"
        else:
            assert item["tasklet_id"].startswith("FW-03-03A-HTTP-")
            assert rust_by_key[key] != "pending"

    mcp_names = {item["name"] for item in inventory["mcp_operations"]}
    for prefix in ("FW-03-04A-MCP-", "FW-03-04B-MCP-"):
        expected_ids = {
            prefix + name.upper().replace(".", "-").replace("_", "-")
            for name in mcp_names
        }
        declared_ids = set(
            re.findall(rf"^\| ({re.escape(prefix)}[^ |]+) \|", worklist, re.M)
        )
        assert len(expected_ids) == len(mcp_names)
        assert declared_ids == expected_ids
        for tasklet_id in expected_ids:
            assert len(re.findall(rf"^\| {re.escape(tasklet_id)} \|", worklist, re.M)) == 1

    deferrals = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))[
        "server_operation_deferrals"
    ]
    deferred_routes = {
        (operation["method"], operation["path"])
        for group in deferrals["route_groups"]
        for operation in group["operations"]
    }
    cutover_source = (
        ROOT / "rust" / "crates" / "kogwistar-api" / "src" / "cutover.rs"
    ).read_text(encoding="utf-8")
    transport_block = re.search(
        r"pub const TRANSPORT_OWNED_HTTP_ROUTES:.*?= &\[(.*?)\];",
        cutover_source,
        flags=re.DOTALL,
    )
    assert transport_block is not None
    transport_routes = set(
        re.findall(r'\("([A-Z]+)",\s*"([^"]+)"\)', transport_block.group(1))
    )
    statuses = {operation["rust"] for operation in http_operations}
    assert statuses == {
        "application-route-in-cutover-inventory",
        "transport-owned",
        "pending",
    }
    assert {
        (operation["method"], operation["path"])
        for operation in http_operations
        if operation["rust"] == "pending"
    } == deferred_routes
    assert {
        (operation["method"], operation["path"])
        for operation in http_operations
        if operation["rust"] == "transport-owned"
    } == transport_routes
    assert {
        (operation["method"], operation["path"])
        for operation in http_operations
        if operation["rust"] == "application-route-in-cutover-inventory"
    } == expected_http - deferred_routes - transport_routes

    assert inventory["counts"]["pending_http_operations"] == sum(
        operation["rust"] == "pending" for operation in http_operations
    )
    assert inventory["counts"]["allowlisted_mcp_operations"] == sum(
        operation["rust"] == "allowlisted-subject-to-runtime-gates"
        for operation in inventory["mcp_operations"]
    )
    mcp_fixture = json.loads(
        (ROOT / "contracts" / "golden" / "mcp-tools.json").read_text(encoding="utf-8")
    )
    expected_mcp = {
        (surface, tool["name"])
        for surface, tools in mcp_fixture["surfaces"].items()
        for tool in tools
    }
    mcp_operations = inventory["mcp_operations"]
    mcp_keys = [
        (operation["surface"], operation["name"]) for operation in mcp_operations
    ]
    assert len(mcp_keys) == len(set(mcp_keys))
    assert set(mcp_keys) == expected_mcp
    assert all(operation["python"] == "frozen-catalog" for operation in mcp_operations)
    assert {operation["rust"] for operation in mcp_operations} <= {
        "allowlisted-subject-to-runtime-gates",
        "not-allowlisted",
    }
    syscall_catalogs = inventory["syscall_catalogs"]
    python_syscalls = syscall_catalogs["python"]["operations"]
    rust_syscalls = syscall_catalogs["rust"]["operations"]
    assert inventory["counts"]["python_syscalls"] == len(python_syscalls) == 11
    assert inventory["counts"]["rust_syscalls"] == len(rust_syscalls) == 11
    assert len(python_syscalls) == len(set(python_syscalls))
    assert len(rust_syscalls) == len(set(rust_syscalls))
    assert syscall_catalogs["python_only"] == []
    assert syscall_catalogs["rust_only"] == []
    deferred_syscalls = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))[
        "server_operation_deferrals"
    ]["syscall_group"]["operations"]
    assert set(deferred_syscalls) <= set(rust_syscalls)
    backend_syscalls = inventory["rust_backend_syscalls"]
    assert set(backend_syscalls) == {
        "postgresql",
        "sqlite-metadata",
        "sqlite-chroma",
        "in-memory",
    }
    for backend in backend_syscalls.values():
        dispatched = set(backend["dispatches"])
        unavailable = set(backend["unavailable"])
        assert dispatched.isdisjoint(unavailable)
        assert dispatched | unavailable == set(rust_syscalls)
        assert set(backend["cutover_deferred"]) <= set(deferred_syscalls)
        assert set(backend.get("explicit_unavailable", [])) <= unavailable
        source_path = ROOT / backend["source"].split("#", 1)[0]
        assert source_path.is_file()
        assert backend["dispatch_scope"] == (
            "source-level dispatch only; not behavioral parity evidence"
        )
    assert set(backend_syscalls["postgresql"]["dispatches"]) == set(rust_syscalls)
    assert set(backend_syscalls["sqlite-chroma"]["dispatches"]) == set(rust_syscalls)
    assert set(backend_syscalls["sqlite-metadata"]["dispatches"]) == {
        "spawn_process",
        "terminate_process",
        "checkpoint",
        "project_view",
        "resume",
        "request_approval",
    }
    assert backend_syscalls["sqlite-metadata"]["explicit_unavailable"] == [
        "mount_memory",
        "receive_message",
    ]
    assert backend_syscalls["in-memory"]["dispatches"] == [
        "create_conversation",
        "receive_message",
    ]

    result = subprocess.run(
        ["python", str(generator), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_dual_language_support_matrix_is_complete_and_conservative() -> None:
    crosswalk = json.loads(
        DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))

    assert matrix["schema_version"] == 1
    assert matrix["status"] == "evidence-gated"
    assert matrix["policy"]["unverified_is_not_enabled"] is True
    assert matrix["policy"]["python_and_rust_are_maintained"] is True
    assert matrix["policy"]["deprecation_requires_separate_adr"] is True

    expected = {item["id"] for item in crosswalk["capabilities"]}
    entries = {item["id"]: item for item in matrix["capabilities"]}
    ownership = {
        item["capability"]: item
        for item in json.loads(
            (ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8")
        )["capability_ownership"]
    }
    assert set(entries) == expected
    allowed = {"enabled", "opt-in", "disabled", "not-applicable"}
    for capability in entries.values():
        assert set(capability) == {"id", "python", "rust", "mixed"}
        for lane in ("python", "rust", "mixed"):
            decision = capability[lane]
            assert set(decision) == {"status", "evidence"}
            assert decision["status"] in allowed
            assert decision["evidence"].startswith("DL-")

    for item in crosswalk["capabilities"]:
        matrix_item = entries[item["id"]]
        if item["native_rust_path"] == "not-verified":
            assert matrix_item["rust"]["status"] != "enabled"

    checklist_path = ROOT / "kogwistar" / "docs" / "ADR-022-dual-language-capability-checklist.md"
    checklist_entries = {
        line.split("|", 2)[1].strip(): line.split("|", 3)[2].strip()
        for line in checklist_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("| DL-")
    }
    checklist_statuses = {
        evidence_id: status.split(" (", 1)[0]
        for evidence_id, status in checklist_entries.items()
    }
    worklist_rows = [
        line.split("|", 4)
        for line in DUAL_LANGUAGE_WORKLIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("| FW-")
    ]
    fine_statuses_by_gate: dict[str, list[tuple[str, str, str]]] = {}
    for columns in worklist_rows:
        if len(columns) >= 5:
            fine_statuses_by_gate.setdefault(columns[2].strip(), []).append(
                (columns[1].strip(), columns[3].strip(), columns[4].strip(" |"))
            )

    for capability in entries.values():
        for lane in ("python", "rust", "mixed"):
            decision = capability[lane]
            evidence_id = decision["evidence"]
            assert evidence_id in checklist_entries, (
                f"{capability['id']}/{lane} references missing evidence {evidence_id}"
            )
            if lane == "mixed" and decision["status"] == "opt-in":
                assert _mixed_evidence_is_executable(
                    evidence_id, checklist_statuses, fine_statuses_by_gate
                ), (
                    f"{capability['id']}/mixed is opt-in but {evidence_id} "
                    "has no passing executable fine-grained evidence"
                )
            if decision["status"] == "enabled":
                assert lane == "python"
                assert ownership[capability["id"]][
                    "current_authoritative_writer"
                ] == "python"


def test_mixed_opt_in_evidence_rejects_unstarted_or_unproven_gates() -> None:
    checklist = {
        "DL-14": "IN PROGRESS",
        "DL-15": "NOT STARTED",
    }
    worklist = {
        "DL-14": [("FW-14-01A", "PASS", "test_runtime.py::test_mixed_path")],
        "DL-15": [("FW-15-01", "NOT STARTED", "")],
    }

    assert _mixed_evidence_is_executable("DL-14", checklist, worklist)
    assert not _mixed_evidence_is_executable("DL-15", checklist, worklist)
    assert not _mixed_evidence_is_executable("DL-404", checklist, worklist)


def test_mixed_runtime_backend_combinations_are_frozen_to_passing_evidence() -> None:
    matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    profiles = matrix.get("mixed_execution_profiles")
    assert isinstance(profiles, list)
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["id"] == "rust-server-python-worker-v1"
    assert profile["status"] == "opt-in"
    assert set(profile["capabilities"]) == {"queues-leases-lanes", "workflow-runtime"}
    assert profile["worker_modes"] == ["sync", "async"]

    entries = {item["id"]: item for item in matrix["capabilities"]}
    for capability_id in profile["capabilities"]:
        assert entries[capability_id]["mixed"]["status"] == "opt-in"

    expected_cells = {
        (runtime, backend)
        for runtime in ("CPython 3.12", "CPython 3.13", "CPython 3.14")
        for backend in ("sqlite", "sqlite-chroma", "pg")
    } | {
        ("PyPy 3.11", "sqlite"),
        ("PyPy 3.11", "sqlite-chroma"),
        ("PyPy 3.11", "pg"),
    }
    cells = profile["cells"]
    assert {(cell["python_runtime"], cell["backend"]) for cell in cells} == expected_cells

    worklist = {
        line.split("|", 4)[1].strip(): line.split("|", 4)
        for line in DUAL_LANGUAGE_WORKLIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("| FW-")
    }
    for cell in cells:
        evidence_ids = cell["evidence"]
        assert evidence_ids
        for evidence_id in evidence_ids:
            columns = worklist[evidence_id]
            assert columns[2].strip() == "DL-14A"
            assert columns[3].strip() == "PASS"
            assert "test_python_worker_and_rust_native_step_interoperate_over_real_server" in columns[4]


def test_dual_language_checklist_ids_are_unique() -> None:
    checklist = (
        ROOT / "kogwistar" / "docs" / "ADR-022-dual-language-capability-checklist.md"
    ).read_text(encoding="utf-8")
    rows = [line for line in checklist.splitlines() if line.startswith("| DL-")]
    gate_pattern = re.compile(
        r"\|\s*(DL-[A-Z0-9-]+)\s*\|\s*"
        r"(PASS|IN PROGRESS|NOT STARTED|FAIL|BLOCKED)"
        r"(?:\s*\([^|]*\))?\s*\|"
    )
    gates = [gate_pattern.match(row) for row in rows]
    ids = [gate.group(1) for gate in gates if gate is not None]

    assert rows
    assert all(gates), "every checklist row must use a recognized status"
    assert len(ids) == len(set(ids)), "dual-language checklist IDs must be unique"


def test_dual_language_checklist_counts_match_current_rows_and_baseline() -> None:
    checklist_path = (
        ROOT / "kogwistar" / "docs" / "ADR-022-dual-language-capability-checklist.md"
    )
    checklist = checklist_path.read_text(encoding="utf-8")
    scope = checklist.split("## Scope Accounting", 1)[1].split(
        "## Design and inventory", 1
    )[0]
    row_pattern = re.compile(
        r"^\| (DL-[A-Z0-9-]+) \| (PASS|IN PROGRESS|NOT STARTED|FAIL|BLOCKED)"
        r"(?:\s*\([^|]*\))? \|",
        re.MULTILINE,
    )
    current = dict(row_pattern.findall(checklist))
    assert len(current) == len(row_pattern.findall(checklist))

    inventory = re.search(
        r"Current inventory: \*\*(\d+) items\*\* "
        r"\(\*\*(\d+) PASS, (\d+) IN PROGRESS;",
        scope,
    )
    assert inventory is not None
    assert tuple(map(int, inventory.groups())) == (
        len(current),
        sum(status == "PASS" for status in current.values()),
        sum(status == "IN PROGRESS" for status in current.values()),
    )

    baseline = subprocess.run(
        [
            "git",
            "show",
            "6130d04:kogwistar/docs/ADR-022-dual-language-capability-checklist.md",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    if baseline.returncode != 0:
        # Hosted CI uses a shallow checkout; current-row accounting still runs.
        return

    baseline_rows = dict(row_pattern.findall(baseline.stdout))
    assert len(baseline_rows) == 130
    assert set(baseline_rows) <= set(current)
    assert all(current[item_id] == "PASS" for item_id, status in baseline_rows.items() if status == "PASS")
    added = re.search(
        r"(\d+) later-added IDs are (\d+) PASS and (\d+) IN PROGRESS",
        scope,
    )
    assert added is not None
    added_rows = {item_id: status for item_id, status in current.items() if item_id not in baseline_rows}
    assert tuple(map(int, added.groups())) == (
        len(added_rows),
        sum(status == "PASS" for status in added_rows.values()),
        sum(status == "IN PROGRESS" for status in added_rows.values()),
    )


def test_unverified_runtime_matrix_lanes_are_not_enabled() -> None:
    matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    entries = {item["id"]: item for item in matrix["capabilities"]}

    for capability_id in ("chroma-adapter", "server-rest-sse-mcp-cli"):
        assert entries[capability_id]["python"]["status"] != "enabled"


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


def test_projection_snapshot_registry_capability_remains_explicitly_unsupported() -> None:
    capability_id = "projections-snapshots-run-registry"
    ownership = next(
        item
        for item in _manifest()["capability_ownership"]
        if item["capability"] == capability_id
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item for item in support_matrix["capabilities"] if item["id"] == capability_id
    )
    conformance = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    coverage = next(
        item for item in conformance["capabilities"] if item["id"] == capability_id
    )

    assert ownership["configuration"] == "KOGWISTAR_IMPL_META_STORE"
    assert ownership["current_authoritative_writer"] == "python"
    assert ownership["target_authoritative_writer"] == "rust"
    assert ownership["rust_cutover_ready"] is False
    assert support["python"]["status"] == "enabled"
    assert support["rust"]["status"] == "disabled"
    assert support["mixed"]["status"] == "disabled"
    assert coverage["rust_coverage"] == "not-cutover-ready"
    assert coverage["mixed_coverage"] == "not-inventoried"
    assert {
        "projection-rebuild-and-snapshot-parity",
        "restart-resume-and-stale-state",
    } <= set(conformance["conformance_profiles"]["projections-snapshots-run-registry"])


def test_queue_capability_is_opt_in_until_all_conformance_gates_pass() -> None:
    capability_id = "queues-leases-lanes"
    ownership = next(
        item
        for item in _manifest()["capability_ownership"]
        if item["capability"] == capability_id
    )
    selector = next(
        item
        for item in _manifest()["implementation_selectors"]
        if item["name"] == ownership["configuration"]
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item for item in support_matrix["capabilities"] if item["id"] == capability_id
    )
    conformance = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    coverage = next(
        item for item in conformance["capabilities"] if item["id"] == capability_id
    )

    assert ownership["configuration"] == "KOGWISTAR_IMPL_META_STORE"
    assert ownership["current_authoritative_writer"] == "python"
    assert ownership["target_authoritative_writer"] == "rust"
    assert ownership["rust_cutover_ready"] is False
    assert selector["default"] == "python"
    assert set(selector["accepted_modes"]) == {"python", "shadow", "rust"}
    assert support["python"]["status"] == "enabled"
    assert support["rust"]["status"] == "opt-in"
    assert support["mixed"]["status"] == "opt-in"
    assert coverage["rust_coverage"] == "not-cutover-ready"
    assert "full parity open" in coverage["mixed_coverage"]


def test_graph_pgvector_capability_is_opt_in_until_graph_conformance_passes() -> None:
    capability_id = "graph-pgvector"
    ownership = next(
        item
        for item in _manifest()["capability_ownership"]
        if item["capability"] == capability_id
    )
    selector = next(
        item
        for item in _manifest()["implementation_selectors"]
        if item["name"] == ownership["configuration"]
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item for item in support_matrix["capabilities"] if item["id"] == capability_id
    )
    conformance = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    coverage = next(
        item for item in conformance["capabilities"] if item["id"] == capability_id
    )

    assert ownership["configuration"] == "KOGWISTAR_IMPL_GRAPH_STORE"
    assert ownership["current_authoritative_writer"] == "python"
    assert ownership["target_authoritative_writer"] == "rust"
    assert ownership["rust_cutover_ready"] is False
    assert selector["default"] == "python"
    assert set(selector["accepted_modes"]) == {"python", "shadow", "rust"}
    assert support["python"]["status"] == "enabled"
    assert support["rust"]["status"] == "opt-in"
    assert support["mixed"]["status"] == "opt-in"
    assert coverage["rust_coverage"] == "not-cutover-ready"
    assert {
        "graph-acl-namespace-and-source-evidence-parity",
        "transaction-and-vector-index-reopen-parity",
    } <= set(conformance["conformance_profiles"][capability_id])


def test_chroma_adapter_remains_python_only_and_mixed_disabled() -> None:
    capability_id = "chroma-adapter"
    ownership = next(
        item
        for item in _manifest()["capability_ownership"]
        if item["capability"] == capability_id
    )
    selector = next(
        item
        for item in _manifest()["implementation_selectors"]
        if item["name"] == ownership["configuration"]
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item for item in support_matrix["capabilities"] if item["id"] == capability_id
    )
    conformance = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    coverage = next(
        item for item in conformance["capabilities"] if item["id"] == capability_id
    )

    assert ownership["configuration"] == "KOGWISTAR_IMPL_GRAPH_STORE"
    assert ownership["target_authoritative_writer"] == "python-adapter"
    assert ownership["rust_cutover_ready"] is False
    assert selector["default"] == "python"
    assert set(selector["accepted_modes"]) == {"python", "shadow", "rust"}
    assert support["python"]["status"] == "opt-in"
    assert support["rust"]["status"] == "not-applicable"
    assert support["mixed"]["status"] == "disabled"
    assert coverage["rust_coverage"] == "python-only"
    assert coverage["mixed_coverage"] == "not-inventoried"
    assert set(conformance["conformance_profiles"][capability_id]) == {
        "python-chroma-contract",
        "single-writer-and-isolated-shadow-read",
    }


def test_workflow_runtime_selector_keeps_python_default_and_rust_opt_in() -> None:
    capability_id = "workflow-runtime"
    ownership = next(
        item
        for item in _manifest()["capability_ownership"]
        if item["capability"] == capability_id
    )
    selector = next(
        item
        for item in _manifest()["implementation_selectors"]
        if item["name"] == ownership["configuration"]
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item for item in support_matrix["capabilities"] if item["id"] == capability_id
    )
    conformance = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    coverage = next(
        item for item in conformance["capabilities"] if item["id"] == capability_id
    )

    assert ownership["configuration"] == "KOGWISTAR_IMPL_RUNTIME"
    assert ownership["current_authoritative_writer"] == "python"
    assert ownership["target_authoritative_writer"] == "rust"
    assert ownership["rust_cutover_ready"] is False
    assert selector["default"] == "python"
    assert set(selector["accepted_modes"]) == {"python", "shadow", "rust"}
    assert support["python"]["status"] == "enabled"
    assert support["rust"]["status"] == "opt-in"
    assert support["mixed"]["status"] == "opt-in"
    assert coverage["rust_coverage"] == "not-cutover-ready"
    assert {
        "transition-route-and-checkpoint-parity",
        "worker-version-environment-and-replay-contract",
        "trace-provenance-and-cancellation",
    } <= set(conformance["conformance_profiles"][capability_id])


def test_rest_capability_keeps_deferred_route_explicit_and_cutover_gated() -> None:
    capability_id = "server-rest-sse-mcp-cli"
    ownership = next(
        item
        for item in _manifest()["capability_ownership"]
        if item["capability"] == capability_id
    )
    selector = next(
        item
        for item in _manifest()["implementation_selectors"]
        if item["name"] == ownership["configuration"]
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item for item in support_matrix["capabilities"] if item["id"] == capability_id
    )
    server_gaps = json.loads(DUAL_LANGUAGE_MANIFEST_PATH.read_text(encoding="utf-8"))[
        "rust_server_gaps"
    ]

    assert ownership["configuration"] == "KOGWISTAR_IMPL_SERVER"
    assert ownership["rust_cutover_ready"] is False
    assert selector["default"] == "python"
    assert set(selector["accepted_modes"]) == {"python", "rust"}
    assert all(support[mode]["status"] == "opt-in" for mode in ("python", "rust", "mixed"))
    assert server_gaps["status"] == "intentionally-deferred"
    assert ["DELETE", "/admin/doc/{doc_id}"] in server_gaps["pending_routes"]


def test_implementation_selector_manifest_matches_runtime_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar import _rust_bridge as bridge
    from kogwistar import implementation_selection as selection

    manifest = _manifest()
    selector_rows = manifest["implementation_selectors"]
    selectors = {item["name"]: item for item in selector_rows}
    assert len(selectors) == len(selector_rows)
    assert set(selectors) == set(selection._IMPLEMENTATION_ENV_NAMES)
    assert set(manifest["implementation_modes"]) == {"python", "shadow", "rust"}
    capability_selectors = {
        item["configuration"] for item in manifest["capability_ownership"]
    }
    assert capability_selectors <= set(selectors)
    assert all(
        item["owner"] == "kogwistar.implementation_selection"
        and item["default"] == "python"
        for item in selector_rows
    )
    for item in selector_rows:
        fallback = item["fallback_selector"]
        assert fallback is None or fallback in selectors
        assert set(item["accepted_modes"]) <= set(manifest["implementation_modes"])

    readers = {
        "KOGWISTAR_IMPL_CONTRACTS": bridge.contract_implementation_mode,
        "KOGWISTAR_IMPL_META_STORE": bridge.meta_store_implementation_mode,
        "KOGWISTAR_IMPL_GRAPH_STORE": bridge.graph_store_implementation_mode,
        "KOGWISTAR_IMPL_RUNTIME": bridge.runtime_implementation_mode,
        "KOGWISTAR_IMPL_SERVER": bridge.server_implementation_mode,
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY": bridge.postgres_authority_implementation_mode,
    }
    for env_name in selection._IMPLEMENTATION_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    for env_name, reader in readers.items():
        row = selectors[env_name]
        for mode in row["accepted_modes"]:
            for selector in selection._IMPLEMENTATION_ENV_NAMES:
                monkeypatch.delenv(selector, raising=False)
            monkeypatch.setenv(env_name, mode)
            assert reader() == mode, f"{env_name}={mode}"

        for selector in selection._IMPLEMENTATION_ENV_NAMES:
            monkeypatch.delenv(selector, raising=False)
        monkeypatch.setenv(env_name, "")
        with pytest.raises(ValueError):
            reader()

    for global_mode in manifest["implementation_modes"]:
        for selector in selection._IMPLEMENTATION_ENV_NAMES:
            monkeypatch.delenv(selector, raising=False)
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", global_mode)
        for env_name, reader in readers.items():
            row = selectors[env_name]
            if row["fallback_selector"] != "KOGWISTAR_IMPL_MODE":
                assert reader() == row["default"], f"{env_name} is independent"
            elif global_mode in row["accepted_modes"]:
                assert reader() == global_mode, f"{env_name} inherits {global_mode}"
            else:
                with pytest.raises(ValueError):
                    reader()

    for env_name, reader in readers.items():
        row = selectors[env_name]
        if row["fallback_selector"] != "KOGWISTAR_IMPL_MODE":
            continue
        for inherited_mode in manifest["implementation_modes"]:
            for explicit_mode in row["accepted_modes"]:
                for selector in selection._IMPLEMENTATION_ENV_NAMES:
                    monkeypatch.delenv(selector, raising=False)
                monkeypatch.setenv("KOGWISTAR_IMPL_MODE", inherited_mode)
                monkeypatch.setenv(env_name, explicit_mode)
                assert reader() == explicit_mode, f"{env_name} overrides {inherited_mode}"

    for selector in selection._IMPLEMENTATION_ENV_NAMES:
        monkeypatch.delenv(selector, raising=False)
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "")
    for env_name, reader in readers.items():
        if selectors[env_name]["fallback_selector"] == "KOGWISTAR_IMPL_MODE":
            with pytest.raises(ValueError):
                reader()
        else:
            assert reader() == selectors[env_name]["default"]

    for selector in selection._IMPLEMENTATION_ENV_NAMES:
        monkeypatch.delenv(selector, raising=False)
    assert all(reader() == "python" for reader in readers.values())


def test_implementation_fallback_inventory_has_owned_behavior_and_evidence() -> None:
    inventory_path = ROOT / "contracts" / "implementation-fallbacks-v1.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    entries = inventory["entries"]
    expected = {
        "explicit-python-selection": "explicit_mode_selection",
        "shadow-oracle-result": "shadow_oracle_return",
        "metadata-filter-non-json": "python_compatibility_semantics",
        "short-id-non-json": "python_compatibility_semantics",
        "runtime-state-non-json": "python_compatibility_semantics",
        "rust-extension-unavailable": "fail_closed_error",
    }

    assert inventory["contract_version"] == 1
    assert {item["id"]: item["class"] for item in entries} == expected
    for item in entries:
        assert item["owner"]
        assert item["owner_symbol"]
        assert item["behavior"]
        assert item["source_marker"]
        assert item["evidence"]
        assert (ROOT / item["site"]).is_file()
        owner_module = importlib.import_module(item["owner"])
        assert Path(owner_module.__file__).resolve() == (ROOT / item["site"]).resolve()
        assert item["source_marker"] in (ROOT / item["site"]).read_text(encoding="utf-8")
        owner = owner_module
        for component in item["owner_symbol"].split("."):
            owner = getattr(owner, component)
        assert callable(owner), f"{item['owner']}:{item['owner_symbol']}"
        for evidence in item["evidence"]:
            test_path, test_name = evidence.split("::", 1)
            test_source = (ROOT / test_path).read_text(encoding="utf-8")
            assert f"def {test_name.split('[')[0]}(" in test_source


def test_manifest_deterministic_contract_selector_modes_and_unsupported_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar import _rust_bridge as bridge

    manifest = _manifest()
    capability = next(
        item
        for item in manifest["capability_ownership"]
        if item["capability"] == "deterministic-contracts"
    )
    selector = next(
        item
        for item in manifest["implementation_selectors"]
        if item["name"] == capability["configuration"]
    )
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item
        for item in support_matrix["capabilities"]
        if item["id"] == "deterministic-contracts"
    )

    assert capability["rust_cutover_ready"] is True
    assert capability["current_authoritative_writer"] == "python"
    assert support["python"]["status"] == "enabled"
    assert support["rust"]["status"] == "opt-in"
    assert support["mixed"]["status"] == "opt-in"
    assert selector["accepted_modes"] == ["python", "shadow", "rust"]

    expected = '{"key":"value"}'

    class CanonicalJsonExtension:
        def canonical_json(self, _payload: str) -> str:
            return expected

    for mode in selector["accepted_modes"]:
        monkeypatch.setenv(capability["configuration"], mode)
        monkeypatch.setattr(
            bridge,
            "_load_extension",
            lambda: CanonicalJsonExtension(),
        )
        assert bridge.contract_canonical_json(
            value={"key": "value"}, python_value=expected
        ) == expected

    monkeypatch.setenv(capability["configuration"], "native")

    def native_must_not_load() -> None:
        pytest.fail("unsupported selector loaded native code")

    monkeypatch.setattr(bridge, "_load_extension", native_must_not_load)
    with pytest.raises(ValueError, match="KOGWISTAR_IMPL_CONTRACTS"):
        bridge.contract_canonical_json(
            value={"key": "value"}, python_value=expected
        )


def test_manifest_postgres_event_log_selector_rejects_split_authority_cells() -> None:
    from kogwistar.implementation_selection import postgres_authority_selectors_aligned

    manifest = _manifest()
    capability = next(
        item
        for item in manifest["capability_ownership"]
        if item["capability"] == "postgres-sequence-event-log"
    )
    selectors = {item["name"]: item for item in manifest["implementation_selectors"]}
    support_matrix = json.loads(SUPPORT_MATRIX_PATH.read_text(encoding="utf-8"))
    support = next(
        item
        for item in support_matrix["capabilities"]
        if item["id"] == "postgres-sequence-event-log"
    )

    assert capability["configuration"] == "KOGWISTAR_IMPL_META_STORE"
    assert capability["rust_cutover_ready"] is False
    assert support["python"]["status"] == "enabled"
    assert support["rust"]["status"] == "opt-in"
    assert support["mixed"]["status"] == "opt-in"
    meta_modes = selectors["KOGWISTAR_IMPL_META_STORE"]["accepted_modes"]
    graph_modes = selectors["KOGWISTAR_IMPL_GRAPH_STORE"]["accepted_modes"]
    authority_modes = selectors["KOGWISTAR_IMPL_POSTGRES_AUTHORITY"]["accepted_modes"]

    assert postgres_authority_selectors_aligned(
        meta_mode="python", graph_mode="python", authority_mode="python"
    )
    assert postgres_authority_selectors_aligned(
        meta_mode="rust", graph_mode="rust", authority_mode="rust"
    )
    for meta_mode in meta_modes:
        for graph_mode in graph_modes:
            for authority_mode in authority_modes:
                aligned = meta_mode == graph_mode == authority_mode
                assert (
                    postgres_authority_selectors_aligned(
                        meta_mode=meta_mode,
                        graph_mode=graph_mode,
                        authority_mode=authority_mode,
                    )
                    is aligned
                ), (meta_mode, graph_mode, authority_mode)


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
