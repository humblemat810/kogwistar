# ADR-015 Native-Seam Verification Worklist

**Scope:** Verify claims that a selected Rust-native execution path did not cross into Python.
**Progress:** **3/5 PASS (60%)**. This is only this bounded workstream, not ADR-015 migration completion.
**Parent gate:** ADR-015 Phase 4 public runtime owner; does not promote `workflow-runtime` readiness.

## Fixed Tasklets

| ID | Status | Acceptance and evidence |
| --- | --- | --- |
| NSV-01 | PASS | Define attestation as response annotation, not independent process evidence. `native_seam.rs` documents this limit; ADR-022 records the evidence boundary. Evidence: commit `d76b2681`, `d335c683`; `test_dual_language_coarse_status_matches_all_mapped_fine_tasks` passed. |
| NSV-02 | PASS | Preserve any previously observed seam crossings; non-empty crossings must fail zero-seam validation. Evidence: `native_seam_attestation_preserves_observed_crossings`; all 3 `native_seam_attestation` Rust tests passed. |
| NSV-03 | PASS | For a packaged Rust server native workflow, durable run history must contain native step events and no `runtime.python_worker.*` or `executor_target=python-worker` event for that run. SQLite and SQLite+Chroma, external-worker and inline-server cells: **4 passed**. PostgreSQL cells skipped because no DSN was configured; this item claims only the four executed cells. |
| NSV-04 | NOT STARTED | Add actual runtime-boundary accounting for Python-worker handoffs; derive counts from observed durable claim/result boundaries, not response annotations. Keep counts scoped to run/step and avoid double-counting prepare/submit events for one handoff. |
| NSV-05 | NOT STARTED | Add paired proof: native-only run reports zero measured crossings; mixed Python-worker/Rust-native run reports the expected nonzero count; missing, duplicated, or forged counters fail validation. Verify durable trace and API projection agree. |

## Closure Rules

- Keep these five IDs and denominator fixed. Do not add tasklets for implementation details.
- A newly discovered invariant requires explicit approval before adding an item.
- Passing this list proves only the named runtime seam contract; it does not prove all Rust API routes, all backends, or whole-stack parity.
- Overall Rust cutover remains governed by `ADR-015-implementation-status.md` and `contracts/rust-port-v1.json`.
