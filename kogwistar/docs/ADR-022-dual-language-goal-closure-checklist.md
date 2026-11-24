# ADR-022 Dual-Language Goal Closure Checklist

**Updated:** 2026-11-24  
**Overall status:** **NOT COMPLETE**. This checklist audits the full dual-language objective; it is separate from ADR-015's fixed capability-gate denominator and ADR-022's historical evidence inventories. Its rows are not added to either count.  
**Status summary:** **6 bounded criteria PASS; 1 PARTIAL; 2 OPEN.** PASS means only the stated bounded criterion is evidenced, not that the overall goal is complete.

**Fine execution tracking:** Remaining GOAL-06/08/09 work is tracked in [ADR-022 dual-language goal fine checklist](ADR-022-dual-language-goal-fine-checklist.md). Its denominator is independent from ADR-015 and the closed 177/911 evidence inventory.

## Closure Criteria

| ID | Criterion | Status | Evidence and exact remaining condition |
| --- | --- | --- | --- |
| GOAL-01 | Keep usable Python and Rust implementations, with explicit per-capability selection and no silent cross-language fallback. | PASS (bounded) | `contracts/rust-port-v1.json` keeps Python as current writer and Rust selection explicit; `ADR-022-dual-language-capability-paths.md` defines declared seams and fail-closed selection. Python-only and Rust-only boundary tests are mapped in ADR-015 REL-04. Scope is the tested routes, not every public operation. |
| GOAL-02 | Preserve safe authority, ACL/scope, transaction ownership, retry, persistence, and trace contracts at declared language seams. | PASS (bounded) | ADR-022 invariants 1-14 plus ADR-015's named capability clauses and REL-04C. Mixed traces test declared seams and transaction ownership; these do not cover every public API operation. |
| GOAL-03 | Provide at least one top-down Rust-native request path with no Python handoff, from server ingress through workflow execution to durable state. | PASS (bounded) | ADR-015 WFR-04 and REL-04B exercise Rust-native workflow/server paths; REL-05C measures the Rust-native HTTP graph trace with zero FFI calls. Current-checkout live pgvector rerun: `cargo test --manifest-path rust/Cargo.toml -p kogwistar-api --lib postgres_agentic_answer_workflow_executes_durably_without_python_worker -- --include-ignored --nocapture` (**1 passed**, 23.37s test; operation benchmark 10.47s), including authenticated conversation ingress, persisted run/turn/lane state, and no Python worker. This proves named workloads only, not all server routes or a global speedup. |
| GOAL-04 | Keep persisted state readable and writable across Python/Rust handoff without weakening rollback or canonical authority. | PASS (bounded) | ADR-015 REL-02C/02D and named SQLite/PostgreSQL/Chroma interoperability tests cover bidirectional reads/writes, reopen, and rollback for their fixtures. Unlisted schemas, historical versions, and backend combinations are not implied. |
| GOAL-05 | Exercise interchangeable memory, PostgreSQL/pgvector, and Chroma paths without claiming cross-store atomicity. | PASS (bounded) | ADR-015 backend gates and `chroma_rust_interop_worklist.md` show Rust/Python Chroma composition and handoff tests; PostgreSQL and memory have separate named gates. Chroma remains an eventual projection; its Python adapter is still explicitly Python-owned in `rust-port-v1`. This PASS establishes tested interoperability, not Rust ownership of that adapter. |
| GOAL-06 | Cover every public graph, storage, runtime, conversation, agent, REST, MCP, and CLI operation in the cross-language inventory, with explicit implementation and conformance status. | OPEN | `contracts/dual-language-surface-inventory-v1.json` counts 84 HTTP operations (75 application routes, 9 transport-owned), 62 MCP allowlisted operations, and 11 syscall names in both language catalogs. `test_public_http_and_mcp_tasklets_name_executable_tests` passes and fails closed when any HTTP/MCP tasklet lacks an executable selector; focused evidence validates inventory freshness, backend applicability, crosswalk, and evidence links. The four former syscall deferrals are closed by named selectors in `contracts/rust-port-v1.json`; pending route/syscall inventories are empty. Bounded in-memory cells include `workflow.run_resume` (suspended-run → REST resume → Rust-native continuation), `send_message` invalid-input errors, `terminate_process` result contract, checkpoint numeric-string normalization, `project_view` resource snapshot required shape, `invoke_tool` invalid-name errors, and `request_approval` branches. Resume tests: `in_memory_native_synchronous_resume_matches_shared_result_contract`, `in_memory_native_run_admission_is_authenticated_and_atomic`, `resume_records_transition_and_native_lane_atomically`; Python counterpart `tests/server/test_chat_server_api.py::test_resume_route_uses_python_defaults_and_checks_namespace`. Approval tests: Rust `in_memory_workflow_syscalls_use_native_run_handlers`; Python `tests/server/test_syscall_api.py::test_syscall_request_approval_branches`. Both consume `contracts/golden/syscall-request-approval-invalid-v1.json` and compare exact HTTP 400 response bodies for missing action and missing grant/revoke capability; unknown action also matches requested/default-reason behavior. Python checkpoint `test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq` and the same Rust in-memory syscall test consume `contracts/golden/syscall-checkpoint-step-seq-v1.json`, verifying string `"0"` normalizes to integer `0`. Project-view test pair (`test_syscall_project_view_matches_resource_snapshot_contract`, `in_memory_project_view_syscall_is_reachable_through_authenticated_http`) checks required resource fields and nested object keys from `workflow-resources-http-v1.json`; dynamic values are not asserted identical. Python `test_syscall_invoke_tool_list_transcript_returns_messages` and Rust `in_memory_invoke_builtin_tools_match_shared_contract_and_authority` consume `syscall-invoke-tool-errors-v1.json` and compare exact error status/body for missing name, blank name, and unknown tool. Terminate-process Python/Rust tests assert fields and accepted statuses from `run-cancel-v1.json`, without exact dynamic-value comparison. Python `test_syscall_send_message_invalid_inputs_match_shared_contract` and Rust `in_memory_send_message_admits_turn_and_run_atomically` consume `syscall-send-message-invalid-v1.json`, asserting exact invalid-input status/body for blank text and non-string idempotency key; Rust also asserts no run is admitted. These prove only named operations/backend pairs, not full inventory closure. GOAL-06 remains open until the full matrix below is executed for every applicable operation/backend pair or a passing fail-closed test proves exclusion. Backend support still varies explicitly.
| GOAL-07 | Verify native and mixed paths on the declared interpreter/platform support matrix. | PASS (bounded) | CI #860 at `f5fcc708dbae11134d154dc1d81a5475d25f610b`: required CPython 3.12-3.14 and PyPy 3.11 jobs passed; `rust-python-interop (pypy311, pypy-3.11-v7.3.20, true)` passed all three real-server worker/backend cases (SQLite, SQLite+Chroma, PostgreSQL), while Rust API server was built separately with no Python handoff in its native path. PyO3 SQLite-meta binding remains excluded and unsupported under PyPy; this PASS covers the declared HTTP worker seam, not in-process extension parity. PyPy 3.12 beta remains best-effort and failed independently, outside the declared support matrix. |
| GOAL-08 | Reconcile behavior across the full declared public surface, preserving Python compatibility and documenting any intentional differences. | OPEN | ADR-015's 22 clauses and 68 tasklets remain bounded capability evidence; ADR-022's 177/911 rows are not whole-stack parity. DLFC-16 is now PASS: CI-selected tests prove Python-only graph/MCP paths do not load Rust, Rust-native workflow ingress-to-persistence has no Python worker handoff across declared in-memory/SQLite/SQLite+Chroma/PostgreSQL paths, and the API dependency closure excludes PyO3. The new test matrix still has many operation/backend behavior families not enumerated and executed. Bounded resume and `request_approval` branches have adjacent Python/Rust coverage, but neither closes other runtime/backend cells. Close only after GOAL-06 inventory is reconciled, applicable matrix oracles pass, and each deliberate difference has an owner, scope, and compatibility statement. Do not reopen existing PASS items absent regression, invalid evidence, or approved contract change. |
| GOAL-09 | Keep the seam-free native option and evaluate its performance without promising unmeasured global superiority. | PARTIAL | REL-05 now has a current-checkout resource-matched Docker smoke at `1dcd1278cba081a6d5a698330d0df5ca57d3c959`: Python HTTP 56.71 ops/s vs Rust-native HTTP 584.35 ops/s on the same 100-record/300-operation SQLite trace, 2 CPU/2 GiB, two samples, equal final state digest, and zero FFI in both pure paths. `tests/core/test_mixed_endpoint_container_benchmark.py::test_endpoint_report_verifier_rejects_ffi_in_pure_mode` now enforces zero FFI in both pure paths; its Python-mode case exposed a missing verifier check. Rust-native won this measured trace; sample count/fixture scope is insufficient to close DLFC-18. Broader representative operations/backends and stable per-workload thresholds remain unverified; no universal Rust speed claim. Detailed evidence is in DLFC-18 and ignored raw output `.tmp/mixed-runtime-current-1dcd1278/`. |

## Pivoted Test-Case Status

One row per named test selector; repeated goal IDs make progress visible by case instead of hiding it in a composite goal row. This is a view of existing goal evidence, not a new completion denominator. Suite-level aggregate mentions are not expanded into invented cases. `PASS` means cited run is recorded as passing; `IN PROGRESS` means some mapped cases pass while coverage remains; `NOT STARTED` means no complete selector/evidence exists for that cell. Keep goal-level status in the summary above. Open work items without a test selector are not test cases and remain in the fine checklist, not this table.

| Goal | Test case / selector | Status | Evidence or gap |
| --- | --- | --- | --- |
| GOAL-03 | `postgres_agentic_answer_workflow_executes_durably_without_python_worker` | PASS | Live pgvector, 1 passed; authenticated ingress through durable run/turn/lane state, no Python worker. |
| GOAL-06 | `test_public_http_and_mcp_tasklets_name_executable_tests` | PASS | Manifest selector-mapping check passed; proves selectors exist, not behavior parity. |
| GOAL-06 | `test_dual_language_surface_inventory_is_current` | PASS | Inventory-to-route/MCP mapping passed. |
| GOAL-06 | `test_syscall_backend_ledger_matches_surface_inventory` | PASS | 11 syscalls × 4 backends resolved, 44 cells. |
| GOAL-06 | `test_capability_evidence_links_reject_missing_stale_duplicate_and_route_refs` | PASS | Evidence-link validation passed. |
| GOAL-06 | `test_adr015_capability_worklist_covers_fixed_incomplete_scope` | PASS | Worklist scope cross-check passed. |
| GOAL-06 | `test_each_capability_gate_clause_names_an_executable_test` | PASS | Gate-clause selectors verified. |
| GOAL-06 | `test_rust_python_parity_annotations_resolve_to_shared_executable_contracts` | PASS | Source/test/fixture references resolved. |
| GOAL-06 | `document_validate_graph_http_route_matches_shared_success_contract` | PASS | Rust shared HTTP success contract passed. |
| GOAL-06 | `document_graph_validation_missing_nodes_matches_python_error_detail` | PASS | Rust/Python invalid-input detail case passed. |
| GOAL-06 | `in_memory_native_synchronous_resume_matches_shared_result_contract` | PASS | Named Rust in-memory resume result case passed. |
| GOAL-06 | `in_memory_native_run_admission_is_authenticated_and_atomic` | PASS | Named admission/authority case passed. |
| GOAL-06 | `resume_records_transition_and_native_lane_atomically` | PASS | Durable resume transition case passed. |
| GOAL-06 | `test_resume_route_uses_python_defaults_and_checks_namespace` | PASS | Python counterpart passed. |
| GOAL-06 | `in_memory_workflow_syscalls_use_native_run_handlers` | PASS | Rust approval/syscall case passed. |
| GOAL-06 | `test_syscall_request_approval_branches` | PASS | Python approval branches and shared invalid fixture passed. |
| GOAL-06 | `test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq` | PASS | Python checkpoint normalization case passed. |
| GOAL-06 | `test_syscall_project_view_matches_resource_snapshot_contract` | PASS | Python resource-shape case passed. |
| GOAL-06 | `in_memory_project_view_syscall_is_reachable_through_authenticated_http` | PASS | Rust authenticated resource-shape case passed. |
| GOAL-06 | `test_syscall_invoke_tool_list_transcript_returns_messages` | PASS | Python invoke-tool success/error cases passed. |
| GOAL-06 | `in_memory_invoke_builtin_tools_match_shared_contract_and_authority` | PASS | Rust invoke-tool contract/authority case passed. |
| GOAL-06 | `test_syscall_send_message_invalid_inputs_match_shared_contract` | PASS | Python invalid-input/no-admission case passed. |
| GOAL-06 | `in_memory_send_message_admits_turn_and_run_atomically` | PASS | Rust invalid-input and atomic admission case passed. |
| GOAL-06 | `test_document_validate_graph_matches_shared_success_contract` | PASS | Python HTTP success contract passed. |
| GOAL-06 | `test_python_document_graph_validation_http_error_matches_shared_fixture` | PASS | Python HTTP error fixture case passed. |
| GOAL-06 | `test_server_health_matches_shared_contract` | PASS | Python `GET /health` checks success status and common health fields against `contracts/golden/health-v1.json`; focused pytest: 1 passed. |
| GOAL-06 | `health_route_matches_shared_python_contract` | PASS | Rust `GET /health` checks same shared fixture fields and types; focused Cargo test: 1 passed. |
| GOAL-06 | `test_conversation_list_is_user_scoped_and_has_stable_response_shape` | PASS | Python `GET /api/conversations` verifies owner filtering and exact response/row field sets against `conversation-list-http-v1.json`; focused pytest: 1 passed. |
| GOAL-06 | `sqlite_chroma_receive_message_returns_scoped_acl_filtered_transcript` | PASS | Rust route test also exercises `GET /api/conversations`; exact response/row field sets match `conversation-list-http-v1.json`; focused Cargo test: 1 passed. |
| GOAL-06 | `test_run_status_route_matches_shared_v1_contract` | PASS | Python `GET /api/runs/{run_id}` checks fixture-required fields, terminal/status relation, cancel flag, null result/error for seeded run, and timestamp types; focused pytest: 1 passed. |
| GOAL-06 | `sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic` | PASS | Rust successful-run status projection checks fixture-required fields, terminal/status relation, cancel flag, null error, and timestamp types; focused Cargo test: 1 passed. Its executed-workflow result is intentionally not compared to Python's directly seeded null result. |
| GOAL-06 | `test_workflow_design_empty_graph_matches_shared_contract` | PASS | Python `GET /api/workflow/design/{id}/graph` exactly matches `workflow-design-empty-graph-http-v1.json`; focused pytest: 1 passed. |
| GOAL-06 | `test_workflow_design_empty_history_matches_shared_contract` | PASS | Python `GET /api/workflow/design/{id}/history` exactly matches `workflow-design-empty-history-http-v1.json`; focused pytest: 1 passed. |
| GOAL-06 | `sqlite_design_history_folds_branch_and_repairs_projection` | PASS | Rust route checks both shared empty-graph and empty-history HTTP fixtures exactly; focused Cargo test: 1 passed. |
| GOAL-07 | `test_main_ci_matrix_uses_hosted_cpython_and_pypy_runtimes` | PASS | CI interpreter matrix contract passed. |
| GOAL-07 | `test_rust_ci_requires_all_pypy_mixed_worker_backends_without_skips` | PASS | Required mixed-worker backend selection passed. |
| GOAL-08 | `test_python_workflow_success_matches_rust_durable_successor_admission` | PASS | Fixed-trace success parity case passed. |
| GOAL-08 | `test_python_suspend_resume_matches_rust_durable_ordered_history` | PASS | Fixed-trace suspend/resume history case passed. |
| GOAL-08 | `test_python_workflow_failure_matches_rust_durable_state_and_step_event` | PASS | Failure state/event parity case passed. |
| GOAL-08 | `test_native_transaction_id_cannot_cross_contexts` | PASS | Transaction ownership invariant passed. |
| GOAL-09 | `test_endpoint_runner_compares_python_mixed_and_rust_http_modes` | PASS | Benchmark runner executes both modes; not a performance threshold result. |
| GOAL-09 | `test_endpoint_report_verifier_rejects_cross_mode_digest_mismatch` | PASS | Digest-mismatch rejection case passed. |
| GOAL-09 | `test_endpoint_report_verifier_rejects_ffi_in_pure_mode` | PASS | Both pure modes enforce zero FFI; workload remains a smoke, not representative gate closure. |
| GOAL-06 | `postgres_workflow_run_cancel_contract_enforces_scope_and_write_capability` | PASS | Live pgvector Rust test passed: nonterminal REST/MCP/syscall cancellation persists `run.cancelling`; denied calls are effect-free; terminal syscall retry adds no event. |
| GOAL-06 | `sqlite_syscall_spawn_and_terminate_use_authoritative_run_service` | PASS | Rust SQLite metadata syscall appends one `run.cancelling` event before cancellation; terminal retry appends none. Focused Rust test: 1 passed. |
| GOAL-06 | `sqlite_chroma_syscall_spawn_and_terminate_use_authoritative_run_service` | PASS | Rust SQLite+Chroma syscall test verifies one `run.cancelling` event and no extra event on terminal retry; live-Chroma focused test passed. |
| GOAL-06 | `test_run_lifecycle_syscalls_use_python_backend[in_memory]` | PASS | Python in-memory syscall persists `run.cancelling`; terminal retry leaves event history unchanged. Focused case passed. |
| GOAL-06 | `test_run_lifecycle_syscalls_use_python_backend[chroma]` | PASS | Python SQLite+Chroma syscall persists `run.cancelling`; terminal retry leaves event history unchanged. Focused case passed. |
| GOAL-06 | `test_conversation_syscalls_use_python_postgres_backend` | PASS | Python PostgreSQL syscall cancellation exposes `run.cancelling`; terminal retry leaves SSE history unchanged. Live testcontainers: 1 passed. |
| GOAL-06 | `in_memory_workflow_cancel_mcp_reaches_authorized_store_effect` | PASS | Focused rerun passed; authorized cancel emits `run.cancelling`, denied cancel is effect-free, terminal retry adds no event. |
| GOAL-06 | `test_mcp_workflow_runtime_submit_cancel` | PASS | Python MCP consumes `workflow-run-submit-mcp-v1.json`; forged caller `user_id` is replaced by fixture's authenticated owner in the persisted run. Python also checks cancellation persistence and duplicate-free terminal retry. Focused run: 1 passed. Workflow-access callback is stubbed; this case does not replace authorization coverage. |
| GOAL-06 | `sqlite_mcp_run_submit_persists_admission` | PASS | Rust SQLite MCP consumes `workflow-run-submit-mcp-v1.json`; forged caller `user_id` is replaced by fixture's authenticated owner in the persisted run. The test also verifies authorized cancellation persists `run.cancelling` and terminal retry adds no event. Focused run: 1 passed. This case does not replace authorization coverage. |
| GOAL-06 | `test_mcp_workflow_dead_letter_replay_dispatches_effect_once` | PASS | Python MCP dispatches scheduler resume once and reports no replay on retry. Focused run: 1 passed. Python scheduler is stubbed. |
| GOAL-06 | `sqlite_run_service_handles_get_events_and_cancel` | PASS | Rust SQLite routes compare `/api/workflow/catalog/ops` exactly with `workflow-catalog-ops-http-v1.json` and MCP dead-letter replay moves lane to `pending`; focused Rust test: 1 passed. |
| GOAL-06 | `test_workflow_catalog_ops_matches_shared_contract` | PASS | Python `GET /api/workflow/catalog/ops` response equals `workflow-catalog-ops-http-v1.json`; focused pytest: 1 passed. |
| GOAL-06 | `test_d3_bundle_route_matches_shared_rust_contract_and_escapes_inline_json` | PASS | Python `GET /viz/d3.bundle` consumes the shared fixture, checks embedded payload/metadata, and rejects unsafe script fragments; focused pytest: 1 passed. |
| GOAL-06 | `d3_bundle_route_matches_shared_python_contract` | PASS | Rust `GET /viz/d3.bundle` uses the same fixture and checks status, content type, payload escaping, and forbidden fragments; focused Cargo test: 1 passed. |

## Full-Parity Test Matrix

This matrix defines the required test classes for GOAL-06/08; it adds no new
goal or checklist denominator. Instantiate cases from every operation in
`contracts/dual-language-surface-inventory-v1.json`. Each operation's Python
and Rust selectors must be executable and recorded in the existing operation
tasklet rows. A selector's existence is only mapping evidence, not proof that
the assertion is sufficient or currently passing.

| Case family | Required oracle for each applicable operation |
| --- | --- |
| Contract success | Same versioned fixture/request through Python and Rust; compare normalized status, response fields, defaults, ordering, IDs, and error-free result. Ignore only declared nondeterministic fields. |
| Input validation | Missing, malformed, wrong-type, unknown, and extra fields; both paths reject consistently and produce no durable side effect. Include boundary values and empty collections where accepted. |
| Authorization | Unauthenticated, wrong role, missing capability, wrong namespace/workspace/graph-space, and ACL-hidden target; deny before protected reads or writes and assert state unchanged/no information leak. Apply only where the operation has an authority boundary. |
| Read semantics | Same seeded state; compare filtering, pagination/limits, stable ordering, hydration, tombstone/redirect behavior, provenance, and not-found semantics. Prove ACL filtering occurs before result limits. |
| Write semantics | Same request; compare canonical graph/run state and emitted event/trace fields. Verify atomic rollback on injected failure, duplicate/retry behavior, idempotency, and no partial side effects. Apply only to mutators. |
| Persistence lifecycle | For durable operations, close/reopen each supported backend and compare read-after-write, replay/recovery, sequence/version, and Python-written ↔ Rust-read plus Rust-written ↔ Python-read fixtures. |
| Backend applicability | Exercise every declared operation/backend pair. For unsupported pairs, assert explicit fail-closed behavior and no silent fallback; do not count dispatch presence as behavioral parity. Distinguish SQLite metadata + eventual Chroma projection from PostgreSQL and in-memory authority semantics. |
| Runtime/worker | Compare identical fixed-ID traces across Python and Rust: routing, state deltas, step/event order, retry/cancel/suspend/resume, terminal frontier, trace parentage, and restart/replay. Check seam crossing counts and transaction ownership. |
| Transport | For HTTP, MCP, and CLI surfaces, compare wire status/envelope, schemas, auth propagation, streaming/notification order where applicable, and malformed protocol behavior. Do not infer one transport from another. |
| Native isolation | For Rust-native claims, run ingress-to-persistence while Python worker/adaptor entry points are instrumented to fail; prove zero Python handoff. Separately prove Python-only mode does not import/call Rust. |
| Regression and CI selection | Run exact selectors in supported CPython 3.12–3.14 / PyPy 3.11 lanes as applicable, with live backend gates marked explicit. A locally passing selector not selected by CI is not a closed CI gate. |

## Executable Test Cases By Goal

Every coarse goal has executable test coverage. This map prevents a
goal from having only prose evidence; it does **not** imply all goal criteria
are closed. Run the named selectors before changing their status. GOAL-06 and
GOAL-08 remain broader matrix obligations, not single-test claims.

| Goal | Implemented executable cases | What they assert; remaining scope |
| --- | --- | --- |
| GOAL-01 | Python unit test `tests/core/test_python_only_graph_path.py::test_python_only_graph_path_does_not_load_rust`; Python integration test of Rust server `tests/runtime/test_rust_runtime_authority.py::test_rust_native_workflow_runs_through_real_server_without_python_worker`. | Explicit Python selection cannot enter Rust; selected Rust-native request reaches persistence without Python worker. Covers named graph/runtime paths, not all APIs. |
| GOAL-02 | Python tests exercising authority/seam behavior: `tests/core/test_sqlite_context_invariant.py::test_cross_implementation_acquire_race_grants_one_owner_only`, `::test_direct_native_python_overlap_fails_before_database_mutation`; `tests/core/test_rust_store_index_jobs_differential.py::test_sqlite_coalesce_claim_tokens_retry_tail_namespace_and_rollback`; `tests/mcp/test_chat_mcp_tools.py::test_mcp_workflow_run_submit_denies_missing_capabilities_before_admission`. | One SQLite owner, overlap denied before mutation, rollback/retry integrity, authorization denied before run admission. Other operations/backends remain subject to their matrix cells. |
| GOAL-03 | Rust-native unit tests: `tests::postgres_agentic_answer::postgres_agentic_answer_workflow_executes_durably_without_python_worker` and `tests::in_memory_provider_events::in_memory_inline_graph_v2_commits_graph_acl_index_and_lane_together`; Python integration test of Rust server `tests/runtime/test_rust_runtime_authority.py::test_rust_native_workflow_runs_through_real_server_without_python_worker`. | Authenticated ingress/native execution/durable effects and absence of Python-worker handoff for the tested backend paths. PostgreSQL case requires its live test gate. |
| GOAL-04 | Python integration tests exercising bidirectional Rust/Python persistence: `tests/core/test_rust_store_index_jobs_differential.py::test_sqlite_python_to_rust_and_rust_to_python_queue_contract`; `tests/core/test_rust_store_projection_differential.py::test_sqlite_named_projections_python_and_rust_interoperate`; live Chroma `tests/server/test_rust_api_chroma_interop_live.py::test_python_chroma_projection_is_read_and_reconciled_by_rust_server` and `::test_rust_graph_mutation_is_readable_by_python_over_chroma`; live PostgreSQL `tests/core/test_rust_store_index_jobs_differential.py::test_postgres_python_rust_queue_differential_live`. | Bidirectional read/write or reconciliation across named persisted contracts; backend-specific selectors are separate and do not imply cross-store atomicity. |
| GOAL-05 | Python tests: `tests/core/test_backend_contract.py::test_backend_contract_collection_crud_and_where` (fake/Chroma/PostgreSQL parameterized); `tests/core/test_backend_contract.py::test_in_memory_semantic_query_excludes_pending_embedding`; `tests/core/test_async_backend_invariants.py::test_async_pg_backend_invariants`; Rust/Python Chroma integration selectors listed under GOAL-04. | Backend contract behavior plus separate memory, PostgreSQL, and Chroma composition cases. Chroma eventual projection semantics remain distinct; no universal transaction promise. |
| GOAL-06 | Python manifest test `tests/core/test_rust_port_contract_manifest.py::test_public_http_and_mcp_tasklets_name_executable_tests`; Python/Rust HTTP contract pair: Python `tests/server/test_chat_server_api.py::test_document_validate_graph_matches_shared_success_contract`, Rust unit tests `tests::document_validation::document_validate_graph_http_route_matches_shared_success_contract` and `::document_graph_validation_missing_nodes_matches_python_error_detail`. | Fails closed for missing HTTP/MCP selectors and checks one shared HTTP success/error contract. Full 84 HTTP × 62 MCP × syscall/backend behavior matrix remains OPEN. |
| GOAL-07 | Python CI-configuration tests: `tests/core/test_pypy_ci_contract.py::test_main_ci_matrix_uses_hosted_cpython_and_pypy_runtimes`, `::test_rust_ci_requires_all_pypy_mixed_worker_backends_without_skips`; separate Rust/Python runtime interop jobs are named by those tests. | Support-matrix declaration and required interop lane selection are executable assertions. Actual runtime compatibility still requires successful CI evidence for each declared lane. |
| GOAL-08 | Python parity tests exercising Rust runtime/storage paths: `tests/runtime/test_runtime_parity_bridge_contract.py::test_python_workflow_success_matches_rust_durable_successor_admission`, `::test_python_suspend_resume_matches_rust_durable_ordered_history`, `::test_python_workflow_failure_matches_rust_durable_state_and_step_event`; `tests/core/test_sqlite_context_invariant.py::test_native_transaction_id_cannot_cross_contexts`. | Fixed-trace state/history parity, failure semantics, and transaction-context ownership for named cells. Full behavior/transport/backend matrix remains OPEN. |
| GOAL-09 | Python benchmark tests exercising Python and Rust HTTP modes: `tests/core/test_mixed_endpoint_container_benchmark.py::test_endpoint_runner_compares_python_mixed_and_rust_http_modes`, `::test_endpoint_report_verifier_rejects_cross_mode_digest_mismatch`, `::test_endpoint_report_verifier_rejects_ffi_in_pure_mode`. | Same trace/digest and zero-FFI checks for pure paths; smoke test guards methodology, not statistical performance closure. DLFC-18 remains IN PROGRESS pending representative samples and thresholds. |

These selectors are deliberately a minimum per-goal test index, not a new
checklist denominator. Existing OPEN/PARTIAL statuses remain unchanged until
their full stated exit conditions pass.

## Test-First Entry Cases For Open Goals

These are the first small executable cases to run or reuse before production
changes for each open coarse goal. They are not whole-goal passing evidence;
do not change a goal status until the wider acceptance matrix passes. Reuse
existing fixtures and helpers; add no new framework or checklist rows. When a
later slice selects a new operation/backend cell, add its focused failing test
before changing production code.

| Goal | First cases | Minimum assertion |
| --- | --- | --- |
| GOAL-06: public operation parity | Reuse Python `tests/server/test_chat_server_api.py::test_document_validate_graph_matches_shared_success_contract` and `tests/core/test_rest_error_contract.py::test_python_document_graph_validation_http_error_matches_shared_fixture`; Rust `tests::document_validation::document_validate_graph_http_route_matches_shared_success_contract` and `::document_graph_validation_missing_nodes_matches_python_error_detail`. | Shared success/error fixtures; compare wire status and stable body fields. For each newly selected mutation/authority cell, first add same-request Python/Rust test plus no-effect assertion on denial/failure. These starter cases do not cover all inventory operations. |
| GOAL-08: behavioral compatibility | Reuse `tests/core/test_rust_store_index_jobs_differential.py::test_sqlite_python_to_rust_and_rust_to_python_queue_contract`; `tests/runtime/test_runtime_parity_bridge_contract.py::test_python_workflow_success_matches_rust_durable_successor_admission` and `::test_python_suspend_resume_matches_rust_durable_ordered_history`. | Compare bidirectional persisted state and normalized ordered runtime history for fixed IDs; preserve transaction owner and assert no undeclared language handoff. Extend only when selecting another backend/behavior cell. |
| GOAL-09: seam-free performance | Reuse `tests/core/test_mixed_endpoint_container_benchmark.py::test_endpoint_runner_compares_python_mixed_and_rust_http_modes`, `::test_endpoint_report_verifier_rejects_cross_mode_digest_mismatch`, and new `::test_endpoint_report_verifier_rejects_ffi_in_pure_mode` (Python/Rust-native parameter cases). | Same trace, resource verification, equal final digest, and pure-path zero FFI. Before claiming the gate, add/run enough representative samples and apply declared per-workload thresholds; current two-sample smoke is insufficient. |

TDD order per goal: land these few focused test cases first, run them, then
change only code needed to pass. Build/live-backend integration comes after
unit cases; never infer full goal closure from these starter cases alone.

Selected verification 2026-11-24: Python goal-map selectors **27 passed**
across Python-only isolation, SQLite ownership/interop, memory backend,
inventory/support-matrix checks, runtime parity, and benchmark verification.
Rust-native server selector: **5 passed, 2 skipped** (both PostgreSQL cases
skipped because no DSN was configured). Rust API unit selectors: **2 passed**
for shared HTTP validation and native in-memory persistence. Live Chroma and
PostgreSQL interop selectors are implemented and mapped above, but were not
rerun in this batch. These are representative checks only; GOAL-06/08 remain
OPEN and GOAL-09 remains PARTIAL.

Completion rule: GOAL-06 requires every inventory operation × declared backend
pair to be `tested` or explicitly `unsupported` with a passing fail-closed
test. GOAL-08 requires all applicable matrix families to pass on both languages
and every intentional difference to name its owner, scope, and compatibility
contract. A passing tasklet cannot imply untested matrix families.

## Reading the Existing Counts

- ADR-015 **68/68 tasklets, 22/22 clauses, 8/8 Rust-targeted capability readiness** means its declared capability-gate suite passed. It does not close GOAL-06 through GOAL-09.
- ADR-022 **177/177 coarse rows and 911/911 fine rows** are mapped implementation/evidence inventories, not proof that every public operation or runtime combination is represented.
- `rust_cutover_ready=true` is an explicit selection control for the named capability profile. It is not proof of complete public-surface inventory or universal parity.
- `chroma-adapter` being intentionally Python-owned is distinct from Rust's tested Chroma GraphStore support. Keep this distinction visible; do not infer either Rust adapter ownership or lack of Chroma interoperability from one flag.
- Do not add rows to the established ADR-015 or ADR-022 denominators to make this audit look more complete. Update a row only when its stated evidence/remaining condition changes; add a new criterion only for a separately approved requirement.

## Next Closure Order

1. Reconcile GOAL-06 against `contracts/dual-language-surface-inventory-v1.json`, the Rust pending-route/syscall declarations, and actual Python/Rust handlers. For each of 84 HTTP, 62 MCP, and 11 syscall contracts, distinguish catalog/dispatch presence from backend behavior and authority evidence; close or explicitly exclude each operation/backend pair without changing completed gates. Current bounded evidence: SQLite+Chroma `receive_message` passes the live supported-path fixture cell; SQLite metadata `project_view` has only partial Rust/Python response-shape evidence. Neither closes broader syscall or public-surface gates. See the fine checklist for selectors and remaining cells.
2. Use the operation inventory to select a bounded GOAL-06/08 behavior cell. Add or strengthen an executable parity oracle, run both implementations on the same fixture, and change production code only for a reproduced contract gap.
3. Extend performance evidence only for representative high-value request families/backends; keep current benchmark claims scoped to their measured traces.

## Latest Bounded Slice Evidence

In-memory workflow cancellation is no longer deferred: Rust REST dispatch, role/scope and active-run capability checks, shared cancel response fields, terminal-repeat idempotency, and atomic recorded-transition/lane cleanup pass `in_memory_run_cancel_matches_shared_contract_and_authority` and `cancel_commits_runtime_transition_and_lane_cleanup_atomically`. Python MCP and in-memory-store comparison tests also pass. This closes only the named memory cancellation cell; GOAL-06 and GOAL-08 remain OPEN pending the full operation/backend matrix.

MCP transport is also exercised end to end: missing `workflow.run.write` returns an MCP tool error without changing run state; an authorized call reaches the Rust in-memory store and cancels the run (`in_memory_workflow_cancel_mcp_reaches_authorized_store_effect`; full API suite: 418 passed, 41 ignored).

The in-memory backend now dispatches all 11 catalogued syscall operations. This slice routes `spawn_process`, `terminate_process`, and `resume` through existing run handlers, and adds `checkpoint`, `project_view`, and `request_approval` using existing runtime-inspection, resource-snapshot, and approval helpers. Rust tests cover delegated spawn authorization, denied termination/checkpoint/project-view/approval without unauthorized effects, resume result envelope, checkpoint lookup, project-view scope, approval denial, authenticated Axum HTTP success/denial, and unchanged direct REST resume (`in_memory_workflow_syscalls_use_native_run_handlers`, `in_memory_resume_syscall_preserves_resume_envelope_and_result`, `in_memory_project_view_syscall_is_reachable_through_authenticated_http`, `in_memory_native_synchronous_resume_matches_shared_result_contract`). Five Python adjacent-contract tests passed for spawn, approval denial, resume, checkpoint access, and resource-snapshot authorization. The latest `request_approval` slice compares exact Python/Rust 400 status and body for missing action and missing grant/revoke capability using `contracts/golden/syscall-request-approval-invalid-v1.json`; both focused tests pass. Generated inventory confirms 11/11 dispatch presence and zero unavailable entries for this backend; this is not proof of every syscall/backend parity cell. GOAL-06/08 remain OPEN; overall checklist summary remains 6 bounded PASS, 1 PARTIAL, 2 OPEN.

