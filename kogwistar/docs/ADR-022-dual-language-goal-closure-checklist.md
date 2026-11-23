# ADR-022 Dual-Language Goal Closure Checklist

**Updated:** 2026-11-23  
**Overall status:** **NOT COMPLETE**. This checklist audits the full dual-language objective; it is separate from ADR-015's fixed capability-gate denominator and ADR-022's historical evidence inventories. Its rows are not added to either count.  
**Status summary:** **6 bounded criteria PASS; 1 PARTIAL; 2 OPEN.** PASS means only the stated bounded criterion is evidenced, not that the overall goal is complete.

## Closure Criteria

| ID | Criterion | Status | Evidence and exact remaining condition |
| --- | --- | --- | --- |
| GOAL-01 | Keep usable Python and Rust implementations, with explicit per-capability selection and no silent cross-language fallback. | PASS (bounded) | `contracts/rust-port-v1.json` keeps Python as current writer and Rust selection explicit; `ADR-022-dual-language-capability-paths.md` defines declared seams and fail-closed selection. Python-only and Rust-only boundary tests are mapped in ADR-015 REL-04. Scope is the tested routes, not every public operation. |
| GOAL-02 | Preserve safe authority, ACL/scope, transaction ownership, retry, persistence, and trace contracts at declared language seams. | PASS (bounded) | ADR-022 invariants 1-14 plus ADR-015's named capability clauses and REL-04C. Mixed traces test declared seams and transaction ownership; these do not cover every public API operation. |
| GOAL-03 | Provide at least one top-down Rust-native request path with no Python handoff, from server ingress through workflow execution to durable state. | PASS (bounded) | ADR-015 WFR-04 and REL-04B exercise Rust-native workflow/server paths; REL-05C measures the Rust-native HTTP graph trace with zero FFI calls. Current-checkout live pgvector rerun: `cargo test --manifest-path rust/Cargo.toml -p kogwistar-api --lib postgres_agentic_answer_workflow_executes_durably_without_python_worker -- --include-ignored --nocapture` (**1 passed**, 23.37s test; operation benchmark 10.47s), including authenticated conversation ingress, persisted run/turn/lane state, and no Python worker. This proves named workloads only, not all server routes or a global speedup. |
| GOAL-04 | Keep persisted state readable and writable across Python/Rust handoff without weakening rollback or canonical authority. | PASS (bounded) | ADR-015 REL-02C/02D and named SQLite/PostgreSQL/Chroma interoperability tests cover bidirectional reads/writes, reopen, and rollback for their fixtures. Unlisted schemas, historical versions, and backend combinations are not implied. |
| GOAL-05 | Exercise interchangeable memory, PostgreSQL/pgvector, and Chroma paths without claiming cross-store atomicity. | PASS (bounded) | ADR-015 backend gates and `chroma_rust_interop_worklist.md` show Rust/Python Chroma composition and handoff tests; PostgreSQL and memory have separate named gates. Chroma remains an eventual projection; its Python adapter is still explicitly Python-owned in `rust-port-v1`. This PASS establishes tested interoperability, not Rust ownership of that adapter. |
| GOAL-06 | Cover every public graph, storage, runtime, conversation, agent, REST, MCP, and CLI operation in the cross-language inventory, with explicit implementation and conformance status. | OPEN | `contracts/dual-language-surface-inventory-v1.json` counts 84 HTTP operations (75 application routes, 9 transport-owned), 62 MCP allowlisted operations, and 11 syscall names in both language catalogs. `test_public_http_and_mcp_tasklets_name_executable_tests` fails closed when any HTTP/MCP tasklet lacks an executable selector; focused evidence validates inventory freshness, backend applicability, crosswalk, and evidence links. The four former syscall deferrals are closed by named selectors in `contracts/rust-port-v1.json`; pending route/syscall inventories are empty. New bounded cell: in-memory `workflow.run_resume` exercises suspended-run → REST resume → Rust-native continuation and shared v1 response fields; MCP admission, role/scope/capability denial, and stale-token rejection are also tested (`in_memory_native_synchronous_resume_matches_shared_result_contract`, `in_memory_native_run_admission_is_authenticated_and_atomic`, `resume_records_transition_and_native_lane_atomically`; commits `6b6365ae`, `41c42009`). Python requires `rw`, workflow namespace, and either `workflow.run.write` or `approve_action`; Rust now matches. This is one backend/operation cell, not a Python-vs-Rust trace comparison or full inventory closure. GOAL-06 remains open until the full matrix below is executed for every applicable operation/backend pair or a passing fail-closed test proves exclusion. Backend support still varies explicitly.
| GOAL-07 | Verify native and mixed paths on the declared interpreter/platform support matrix. | PASS (bounded) | CI #860 at `f5fcc708dbae11134d154dc1d81a5475d25f610b`: required CPython 3.12-3.14 and PyPy 3.11 jobs passed; `rust-python-interop (pypy311, pypy-3.11-v7.3.20, true)` passed all three real-server worker/backend cases (SQLite, SQLite+Chroma, PostgreSQL), while Rust API server was built separately with no Python handoff in its native path. PyO3 SQLite-meta binding remains excluded and unsupported under PyPy; this PASS covers the declared HTTP worker seam, not in-process extension parity. PyPy 3.12 beta remains best-effort and failed independently, outside the declared support matrix. |
| GOAL-08 | Reconcile behavior across the full declared public surface, preserving Python compatibility and documenting any intentional differences. | OPEN | ADR-015's 22 clauses and 68 tasklets remain bounded capability evidence; ADR-022's 177/911 rows are not whole-stack parity. The new test matrix defines the required case families, but most have not yet been enumerated and executed across every operation/backend cell. The in-memory resume cell now follows Python's synchronous result shape and authorization policy, but does not compare identical Python/Rust event traces or close other runtime/backend cells. Close only after GOAL-06 inventory is reconciled, applicable matrix oracles pass, and each deliberate difference has an owner and compatibility statement. Do not reopen existing PASS items absent regression, invalid evidence, or approved contract change. |
| GOAL-09 | Keep the seam-free native option and evaluate its performance without promising unmeasured global superiority. | PARTIAL | REL-05 compares Python, mixed, and Rust-native HTTP for one resource-matched SQLite graph trace; Rust-native was fastest in that measured workload. The goal requires a seam-free option, not universal Rust speed guarantees. Broader representative request families and backends have no end-to-end performance evidence; retain only per-workload claims until measured. |

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

1. Reconcile GOAL-06 against `contracts/dual-language-surface-inventory-v1.json`, the Rust pending-route/syscall declarations, and actual Python/Rust handlers. For each of 84 HTTP, 62 MCP, and 11 syscall contracts, distinguish catalog/dispatch presence from backend behavior and authority evidence; close or explicitly exclude each operation/backend pair without changing completed gates.
2. Use the operation inventory to select a bounded GOAL-06/08 behavior cell. Add or strengthen an executable parity oracle, run both implementations on the same fixture, and change production code only for a reproduced contract gap.
3. Extend performance evidence only for representative high-value request families/backends; keep current benchmark claims scoped to their measured traces.

