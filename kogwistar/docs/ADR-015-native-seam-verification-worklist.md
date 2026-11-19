# ADR-015 Native-Seam Verification Worklist

**Scope:** Verify claims that a selected Rust-native execution path did not cross into Python.
**Progress:** **5/5 PASS (100%)**. This is only this bounded workstream, not ADR-015 migration completion.
**Parent gate:** ADR-015 Phase 4 public runtime owner; does not promote `workflow-runtime` readiness.

## Fixed Tasklets

| ID | Status | Acceptance and evidence |
| --- | --- | --- |
| NSV-01 | PASS | Define attestation as response annotation, not independent process evidence. `native_seam.rs` documents this limit; ADR-022 records the evidence boundary. Evidence: commit `d76b2681`, `d335c683`; `test_dual_language_coarse_status_matches_all_mapped_fine_tasks` passed. |
| NSV-02 | PASS | Preserve any previously observed seam crossings; non-empty crossings must fail zero-seam validation. Evidence: `native_seam_attestation_preserves_observed_crossings`; all 3 `native_seam_attestation` Rust tests passed. |
| NSV-03 | PASS | For packaged Rust-server native workflows, durable run history contains native step events and no `runtime.python_worker.*` or `executor_target=python-worker` event for that run. Full backend matrix: **7 passed, 0 skipped** across in-memory, SQLite, SQLite+Chroma, and PostgreSQL; external-worker and inline-server modes. Evidence: `tests/runtime/test_rust_runtime_authority.py::test_rust_native_workflow_runs_through_real_server_without_python_worker`; `.codex/adr015-rust-native-current/native-seam-full-backends.junit.xml`. |
| NSV-04 | PASS | Define accepted Python-worker handoff count from durable, Rust-validated `runtime.python_worker.result_submitted` events keyed by `(run_id, message_id, lease_claim_id)`; prepared/submitted pair counts once, distinct lease retries count separately, duplicates fail. Full mixed persistence matrix: **4 passed, 0 skipped** across SQLite+Chroma/PostgreSQL and Python-worker/Rust-native persistence. Expected counts remain exact: two Python steps when persistence is Python-owned, one when Rust-native. Evidence: `test_accepted_python_worker_handoffs_count_submissions_once`; `test_python_agentic_answer_workflow_commits_graph_effects_through_rust_storage`; `.codex/adr015-rust-native-current/native-seam-full-backends.junit.xml`. |
| NSV-05 | PASS | Paired proof: native Rust-server runs yield zero accepted Python handoffs; mixed runs yield exact positive counts. Combined focused runtime matrix: **11 passed, 0 skipped** across in-memory, SQLite, SQLite+Chroma, and PostgreSQL; both worker modes. Separate handoff-count unit test passed. Rust trace assertion passed: `CARGO_PROFILE_TEST_CODEGEN_UNITS=255 cargo test --manifest-path rust/Cargo.toml -p kogwistar-api sqlite_run_service_handles_get_events_and_cancel --lib` (**1 passed**). The profile override produced a fresh test executable to avoid the externally locked default exe; `cargo check --manifest-path rust/Cargo.toml -p kogwistar-api --tests` also passed. |

## Closure Rules

- Keep these five IDs and denominator fixed. Do not add tasklets for implementation details.
- A newly discovered invariant requires explicit approval before adding an item.
- Passing this list proves only the named runtime worker seam contract for the tested backend/mode cells; it does not prove all Rust API routes or whole-stack parity.
- Overall Rust cutover remains governed by `ADR-015-implementation-status.md` and `contracts/rust-port-v1.json`.
