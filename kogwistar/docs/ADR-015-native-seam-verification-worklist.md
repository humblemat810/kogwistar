# ADR-015 Native-Seam Verification Worklist

**Scope:** Verify claims that a selected Rust-native execution path did not cross into Python.
**Progress:** **5/5 PASS (100%)**. This is only this bounded workstream, not ADR-015 migration completion.
**Parent gate:** ADR-015 Phase 4 public runtime owner; does not promote `workflow-runtime` readiness.

## Fixed Tasklets

| ID | Status | Acceptance and evidence |
| --- | --- | --- |
| NSV-01 | PASS | Define attestation as response annotation, not independent process evidence. `native_seam.rs` documents this limit; ADR-022 records the evidence boundary. Evidence: commit `d76b2681`, `d335c683`; `test_dual_language_coarse_status_matches_all_mapped_fine_tasks` passed. |
| NSV-02 | PASS | Preserve any previously observed seam crossings; non-empty crossings must fail zero-seam validation. Evidence: `native_seam_attestation_preserves_observed_crossings`; all 3 `native_seam_attestation` Rust tests passed. |
| NSV-03 | PASS | For a packaged Rust server native workflow, durable run history must contain native step events and no `runtime.python_worker.*` or `executor_target=python-worker` event for that run. SQLite and SQLite+Chroma, external-worker and inline-server cells: **4 passed**. PostgreSQL cells skipped because no DSN was configured; this item claims only the four executed cells. |
| NSV-04 | PASS | Define accepted Python-worker handoff count from durable, Rust-validated `runtime.python_worker.result_submitted` events keyed by `(run_id, message_id, lease_claim_id)`; prepared/submitted pair counts once, distinct lease retries count separately, duplicates fail. The live mixed workflow checks expected counts: two Python steps when persistence is Python-owned, one when persistence is Rust-native. Evidence: `test_accepted_python_worker_handoffs_count_submissions_once`; `test_python_agentic_answer_workflow_commits_graph_effects_through_rust_storage` (**2 passed** in SQLite+Chroma, 2 PostgreSQL cases skipped without DSN). |
| NSV-05 | PASS | Paired proof: packaged Rust-server native runs yield zero accepted Python handoffs; mixed runs yield exact positive counts. Python E2E: `test_rust_native_workflow_runs_through_real_server_without_python_worker` (**4 passed**, 2 PostgreSQL cases skipped without DSN); combined focused runtime run (**7 passed, 4 skipped**). Rust trace assertion passed: `CARGO_PROFILE_TEST_CODEGEN_UNITS=255 cargo test --manifest-path rust/Cargo.toml -p kogwistar-api sqlite_run_service_handles_get_events_and_cancel --lib` (**1 passed**). The profile override produced a fresh test executable to avoid the externally locked default exe; `cargo check --manifest-path rust/Cargo.toml -p kogwistar-api --tests` also passed. |

## Closure Rules

- Keep these five IDs and denominator fixed. Do not add tasklets for implementation details.
- A newly discovered invariant requires explicit approval before adding an item.
- Passing this list proves only the named runtime worker seam contract for the tested SQLite and SQLite+Chroma paths; it does not prove all Rust API routes, PostgreSQL behavior, or whole-stack parity.
- Overall Rust cutover remains governed by `ADR-015-implementation-status.md` and `contracts/rust-port-v1.json`.
