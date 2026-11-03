# ADR-022 Remaining Worklist

**Updated:** 2026-11-03
**Purpose:** Track bounded work beneath the coarse gates in [ADR-022-dual-language-capability-checklist.md](ADR-022-dual-language-capability-checklist.md).

## Completion Snapshot

- **Coarse checklist:** **154/177 PASS (87.0%)**.
- **Fine worklist:** **7/265 PASS (2.6%)**; **0 IN PROGRESS**, **258 NOT STARTED**.
- Tasklet sizing rule: one counted leaf is one independently reviewable outcome, designed for one focused edit/test session (target: 30-90 minutes; split before starting if it is likely to exceed that). Each leaf has exactly one primary artifact or behavior, one acceptance check, and one evidence reference. A leaf may cover **one** operation family, **one** backend/transport/runtime cell, or **one** failure mode; it must not combine those axes. The fine denominator counts leaves only, never parent scopes. Preserve already-passed evidence and IDs; split only rows that violate this rule, and report any denominator change with the exact parent-to-child mapping. The fine denominator is an explicit expansion of previously bundled work, not newly discovered implementation progress; report coarse and fine percentages separately and do not infer schedule or nearness to completion from either percentage.

## How To Track

- This is the detailed execution list; the linked checklist remains the sole coarse progress and denominator source. Only counted leaf tasklets contribute to the fine percentage.
- The baseline is the 23 coarse IDs currently marked `IN PROGRESS`. Every coarse ID must map to fine work below. Fine IDs do not add to the 177-item coarse inventory.
- A parent scope groups related leaves and is not itself countable or completable. If an open row names multiple backends, operation families, runtime/platform cells, or independent failure modes, treat it as a parent and split it into leaves before work begins. Do not mark a parent `PASS` in place of its children.
- A fine task reaches `PASS` only when its stated acceptance evidence is recorded here or linked from its referenced test/report. Tests must exercise the named behavior, not merely compile the surrounding code.
- Close a coarse ID only when its original exit criteria and all mapped required fine tasks pass. Do not reopen a passed coarse ID without a reproduced regression or new required contract.
- Equal-size acceptance test: a reviewer can state the leaf's single expected result in one sentence, run its named focused check, and decide pass/fail without waiting for unrelated leaves. “All/every/each/across/matrix” wording is a split warning, not sufficient tasklet scope by itself. A matrix is split only where cells have independent evidence; do not multiply rows by files or dimensions that do not change the acceptance result.
- If a leaf proves unexpectedly broad, preserve its original ID as a non-counted parent, split only its demonstrated independent outcomes, and report the parent/child mapping and revised denominator. Never keep an oversized parent counted alongside its children.

**Status vocabulary:** `NOT STARTED`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

| Fine ID | Coarse ID | Status | Bounded task and acceptance evidence |
| --- | --- | --- | --- |
| FW-02-01 | DL-02 | PASS | The SQLite+Chroma authenticated HTTP protocol/authorization route contract is covered in `tests::sqlite_chroma_graph_topology_reads_are_scoped_and_acl_filtered`: successful canonical node resolution, missing-`read_graph` denial, fail-closed foreign-namespace handling, and metadata-only service rejection with `501 KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER` (**1 passed**). Paired Python ACL/fusion/fixture tests: **5 passed**. |
| FW-02-01B | DL-02 | PASS | The ignored live-Chroma regression `tests::sqlite_search_index_events_project_to_live_chroma_and_survive_reopen` now exercises authenticated `GET /api/search_index_hybrid`: a scoped public node resolves to its canonical document, a newer private ACL revision removes the hit, and missing `read_graph` returns `403`. Verified against disposable `chromadb/chroma:1.5.9` on Windows (**1 passed**); paired fake-endpoint scoped-route regression also passes (**1 passed**). |
| FW-02-02 | DL-02 | PASS | The Python DTO and Rust `SearchIndexUpsertV1` share `contracts/golden/search-index-entry-v1.json`; frozen OpenAPI still defines the request/response schema. Python authorization is covered by `tests/server/test_search_index_api_auth.py::test_add_index_entries_requires_write_role_and_keeps_payload`; Rust contract mapping is covered by `search_index_entries_match_python_golden_fixture`. Rust keeps this public route intentionally unavailable until the versioned authority gates pass: `pending_routes_fail_closed_for_every_rust_service` verifies it is not advertised and returns `501 KOGWISTAR_ROUTE_DEFERRED` across services. Python auth/contract, Rust contract, OpenAPI drift, and Rust deferral tests passed (6 checks across targeted runs). This closes the contract-and-deferral definition only, not public admission. |
| FW-02-02A | DL-02 | PASS | The authenticated Rust `POST /api/add_index_entries` path on SQLite+Chroma was live-tested against disposable Chroma 1.5.9: missing `write_graph` and mismatched storage namespace each return `403` without appending an event; an authorized request and exact retry return `200`, append one event, and project the changed vector metadata. Evidence: `tests::sqlite_search_index_events_project_to_live_chroma_and_survive_reopen` (**1 passed**, Windows); Python route contract remains covered by `tests/server/test_search_index_api_auth.py::test_add_index_entries_requires_write_role_and_keeps_payload`. |
| FW-02-02B | DL-02 | PASS | Authenticated Rust `POST /api/add_index_entries` was exercised against PostgreSQL/pgvector through the public HTTP router: an authorized request and exact retry both return `200`, with stable event admission. Evidence: `tests::postgres_hybrid_search_is_available_over_authenticated_http` (**1 passed**, live PostgreSQL); durable event replay after service reopen remains covered by `tests::postgres_search_index_events_project_to_pgvector_and_survive_reopen`. In-memory route authorization/idempotency also passes `tests::in_memory_search_index_upsert_is_authenticated_and_idempotent`. |
| FW-02-02C | DL-02 | NOT STARTED | Reconcile each deferred search-index syscall with the Python contract and record a paired pass or explicit unsupported result; do not enable deferred operations in this task. |
| FW-03-01 | DL-03 | NOT STARTED | Map every capability-crosswalk claim to one named Python evidence test and verify no claim lacks Python behavior evidence. |
| FW-03-01A | DL-03 | NOT STARTED | Map every capability-crosswalk claim to one named Rust evidence test or an explicit `not-verified`/unsupported record. |
| FW-03-01B | DL-03 | NOT STARTED | Add a drift test that rejects missing, stale, or duplicate capability-to-evidence links without treating route advertisement as evidence. |
| FW-03-03 | DL-03 | NOT STARTED | Reconcile frozen REST inventory method/path entries to one owner and one evidence reference each; report missing and stale mappings. |
| FW-03-03A | DL-03 | NOT STARTED | For each REST route already implemented in Rust, verify its success response against the Python contract using one paired test per route. |
| FW-03-03B | DL-03 | NOT STARTED | For each REST route intentionally unsupported in Rust, verify the explicit unsupported response and that no state change occurs. |
| FW-03-04 | DL-03 | NOT STARTED | Reconcile frozen MCP operation names and schemas to one implementation owner and one evidence reference each. |
| FW-03-04A | DL-03 | NOT STARTED | Verify Python/Rust request and response parity for each advertised MCP operation, one operation family per test slice. |
| FW-03-04B | DL-03 | NOT STARTED | Verify authorization denial and stable error semantics for each advertised MCP operation family without executing denied effects. |
| FW-03-02 | DL-03 | NOT STARTED | Run conformance tests for currently advertised REST routes and record exact pass/unsupported outcomes; keep untested paths `not-verified`. |
| FW-03-02A | DL-03 | NOT STARTED | Run conformance tests for currently advertised MCP operations and record exact pass/unsupported outcomes; keep untested paths `not-verified`. |
| FW-05-01 | DL-05 | NOT STARTED | Add paired Pydantic/Serde valid-input fixtures for remaining graph entity DTOs and compare canonical outputs. |
| FW-05-01A | DL-05 | NOT STARTED | Add paired invalid-input fixtures for remaining graph entity DTOs and verify both languages reject before effects. |
| FW-05-01B | DL-05 | NOT STARTED | Add paired persistence DTO fixtures for valid and invalid payloads, comparing canonical serialization and rejection behavior. |
| FW-05-01C | DL-05 | NOT STARTED | Add paired REST/MCP request and response DTO fixtures for valid and invalid payloads, comparing canonical serialization. |
| FW-05-03 | DL-05 | NOT STARTED | Complete Node and Edge base-family DTO parity for identity, common fields, lifecycle, and legacy defaults. |
| FW-05-03A | DL-05 | NOT STARTED | Complete hyperedge-family DTO parity for identity, participations/endpoints, and invalid endpoint rejection. |
| FW-05-03B | DL-05 | NOT STARTED | Verify graph mutation DTOs reject malformed entity batches before any document, event, or index effect. |
| FW-05-03C | DL-05 | NOT STARTED | Verify lifecycle and tombstone DTOs preserve redirect and endpoint fields through Python/Rust round trips. |
| FW-05-04 | DL-05 | NOT STARTED | Compare provider request/event DTO canonical serialization and stable errors across Python and Rust. |
| FW-05-04A | DL-05 | NOT STARTED | Compare persistence and transaction error-envelope DTOs, including stable error codes and details. |
| FW-05-04B | DL-05 | NOT STARTED | Compare REST/MCP error envelopes and identify/document intentionally Python-only payload shapes. |
| FW-05-02 | DL-05 | NOT STARTED | Verify identifier and source-span boundary cases with shared valid/invalid fixtures and matching offsets. |
| FW-05-02A | DL-05 | NOT STARTED | Verify additive schema-version evolution remains accepted in both languages for the declared compatible DTOs. |
| FW-05-02B | DL-05 | NOT STARTED | Verify unsupported schema versions and unknown fields fail consistently before effects. |
| FW-06-01 | DL-06 | NOT STARTED | Inventory canonical durable resources and their Python transaction owner, one resource family per evidence row. |
| FW-06-01A | DL-06 | NOT STARTED | Inventory Rust and mixed-mode write entry points and map each to its declared transaction owner. |
| FW-06-01B | DL-06 | NOT STARTED | Mark resources with explicitly weaker backend guarantees and verify those guarantees are stated in the backend contract. |
| FW-06-03 | DL-06 | NOT STARTED | Verify SQLite single-writer ownership for event journal and graph projection with a concurrent-writer test. |
| FW-06-03A | DL-06 | NOT STARTED | Verify PostgreSQL transaction ownership for graph, workflow/run, worker journal, ACL, and index-job intent with an overlapping-writer test. |
| FW-06-03B | DL-06 | NOT STARTED | Verify in-memory authority serialization/ownership for one conflicting write pair and document its process-local boundary. |
| FW-06-04 | DL-06 | NOT STARTED | Verify Chroma projection is eventual and SQLite metadata/events remain canonical when projection is unavailable. |
| FW-06-04A | DL-06 | NOT STARTED | Verify Chroma projection replay repairs a stale/missing vector without claiming Chroma multi-writer ACID. |
| FW-06-05 | DL-06 | NOT STARTED | Inject failure before canonical transaction commit and assert no authoritative partial write remains. |
| FW-06-05A | DL-06 | NOT STARTED | Inject failure after canonical commit but before projection/ack and assert replay converges without duplicate authoritative effects. |
| FW-06-05B | DL-06 | NOT STARTED | Inject failure during multi-entity atomic mutation and assert transaction rollback or explicit unsupported atomicity. |
| FW-06-02 | DL-06 | NOT STARTED | Reopen SQLite after injected failure and verify durable state and idempotent retry. |
| FW-06-02A | DL-06 | NOT STARTED | Reopen PostgreSQL after injected failure and verify durable state and idempotent retry. |
| FW-08-01 | DL-08 | NOT STARTED | Add a live PostgreSQL graph-neighbor read test for tenant/project scope and source-evidence authorization. |
| FW-08-01A | DL-08 | NOT STARTED | Add a live PostgreSQL graph-snapshot test for edge endpoint dependency filtering and malformed backend records. |
| FW-08-01B | DL-08 | NOT STARTED | Add a live PostgreSQL conflicting-scope record test and prove it is excluded from every returned shape. |
| FW-08-03 | DL-08 | NOT STARTED | Verify live SQLite+Chroma public graph reads revalidate canonical scope and ACL after vector/backend response. |
| FW-08-03A | DL-08 | NOT STARTED | Verify live SQLite+Chroma endpoint dependencies and stale-projection handling on graph reads. |
| FW-08-04 | DL-08 | NOT STARTED | Verify in-memory authenticated node reads match Python scope/ACL behavior for allowed and denied records. |
| FW-08-04A | DL-08 | NOT STARTED | Verify in-memory edge/adjacency reads fail closed for absent, malformed, or cross-scope endpoints. |
| FW-08-02 | DL-08 | NOT STARTED | Verify public read effects after an ACL revocation on each backend and assert unauthorized records are absent. |
| FW-08-02A | DL-08 | NOT STARTED | Verify public read effects with stale/missing canonical records and compare the supported backend result shapes. |
| FW-08-02B | DL-08 | NOT STARTED | Verify public read effects do not leak denied nodes through adjacency, snapshots, or vector-derived results. |
| FW-08C-01 | DL-08C | NOT STARTED | Verify vector retrieval excludes tombstoned entities and resolves redirects consistently in Python and Rust. |
| FW-08C-01A | DL-08C | NOT STARTED | Verify vector retrieval requires authorized source evidence and rejects stale/foreign evidence references. |
| FW-08C-01B | DL-08C | NOT STARTED | Run the lifecycle/evidence vector cases on PostgreSQL and record the exact result parity. |
| FW-08C-01C | DL-08C | NOT STARTED | Run the lifecycle/evidence vector cases on SQLite+Chroma and record the exact result parity. |
| FW-08C-01D | DL-08C | NOT STARTED | Run the lifecycle/evidence vector cases on in-memory and record the exact result parity. |
| FW-08C-03 | DL-08C | NOT STARTED | Verify lifecycle filtering occurs before top-k limit on each supported vector backend. |
| FW-08C-03A | DL-08C | NOT STARTED | Verify ACL-denied candidates are omitted without unauthorized backfill changing result limits. |
| FW-08C-03B | DL-08C | NOT STARTED | Verify normalized score meaning and ordering for each declared metric/backend pair. |
| FW-08C-04 | DL-08C | NOT STARTED | Verify tenant/project and graph-role scope on vector retrieval across the declared backend set. |
| FW-08C-04A | DL-08C | NOT STARTED | Verify source span/provenance and endpoint authorization on vector results. |
| FW-08C-04B | DL-08C | NOT STARTED | Verify the same scope and evidence constraints on hybrid retrieval results. |
| FW-08C-02 | DL-08C | NOT STARTED | Verify mutation admission scope parity for Python and Rust on PostgreSQL and in-memory. |
| FW-08C-02A | DL-08C | NOT STARTED | Verify mutation admission scope parity for SQLite+Chroma and record unsupported combinations explicitly. |
| FW-11-01 | DL-11 | NOT STARTED | Compare canonical trace fields for one successful runtime transition against a shared Python/Rust fixture. |
| FW-11-01A | DL-11 | NOT STARTED | Compare trace fields for retry and attempt increments against a shared Python/Rust fixture. |
| FW-11-01B | DL-11 | NOT STARTED | Compare trace fields for suspend/resume and terminal error transitions against shared fixtures. |
| FW-11-03 | DL-11 | NOT STARTED | Verify trace parentage, run/step identity, and sequence ordering for synchronous execution. |
| FW-11-03A | DL-11 | NOT STARTED | Verify trace parentage and event-to-step causal edges for asynchronous execution. |
| FW-11-03B | DL-11 | NOT STARTED | Verify retries and terminal failures retain correct parent span, attempt, and event identity. |
| FW-11-04 | DL-11 | NOT STARTED | Verify authenticated trace/run-event inspection on SQLite before and after reopen/replay. |
| FW-11-04A | DL-11 | NOT STARTED | Verify authenticated trace/run-event inspection on PostgreSQL before and after reopen/replay. |
| FW-11-04B | DL-11 | NOT STARTED | Verify legacy persisted trace rows with omitted optional fields remain readable and are not misreported. |
| FW-11-02 | DL-11 | NOT STARTED | Verify authenticated SQLite inspection returns correct trace parentage and event-to-step linkage. |
| FW-11-02A | DL-11 | NOT STARTED | Verify authenticated PostgreSQL inspection returns correct trace parentage and event-to-step linkage. |
| FW-12-01 | DL-12 | NOT STARTED | Verify explicit Python selector behavior for one REST operation family with Rust extension absent. |
| FW-12-01A | DL-12 | NOT STARTED | Verify explicit Python selector behavior for one MCP operation family with Rust extension absent. |
| FW-12-01B | DL-12 | NOT STARTED | Verify legacy selector defaults and configured graph-role selection remain unchanged in Python-only mode. |
| FW-12-03 | DL-12 | NOT STARTED | Verify Python-only REST ingress and graph mutation do not import or call Rust for one supported request path. |
| FW-12-03A | DL-12 | NOT STARTED | Verify unsupported/missing Rust extension does not alter Python-only REST behavior or trigger an implicit Rust load. |
| FW-12-04 | DL-12 | NOT STARTED | Verify Python-only MCP in-process authorization and response/error shape without Rust import. |
| FW-12-04A | DL-12 | NOT STARTED | Verify Python-only MCP HTTP transport authorization and response/error shape without Rust import. |
| FW-12-02 | DL-12 | NOT STARTED | Verify Python-only indexing emits durable job state through a public interface. |
| FW-12-02A | DL-12 | NOT STARTED | Verify Python-only retrieval and visualization read back the indexed result on supported backends. |
| FW-13-01 | DL-13 | NOT STARTED | Inventory advertised native operation kinds and map each kind to its Rust handler and shared parity fixture. |
| FW-13-01A | DL-13 | NOT STARTED | Verify one advertised native operation's output against the Python/shared fixture contract. |
| FW-13-01B | DL-13 | NOT STARTED | Verify unknown native operation kinds fail closed before durable or graph effects. |
| FW-13-03 | DL-13 | NOT STARTED | Verify versioned native dispatch and legacy routing defaults for valid operations. |
| FW-13-03A | DL-13 | NOT STARTED | Verify malformed operation payloads fail before effects. |
| FW-13-03B | DL-13 | NOT STARTED | Verify unavailable native capabilities fail explicitly before effects and do not silently fallback. |
| FW-13-04 | DL-13 | NOT STARTED | Verify native executor durable completion and final digest after SQLite service restart. |
| FW-13-04A | DL-13 | NOT STARTED | Verify native executor durable completion and final digest after PostgreSQL service restart. |
| FW-13-04B | DL-13 | NOT STARTED | Verify native executor error routing and recovery for one injected backend failure. |
| FW-13-02 | DL-13 | NOT STARTED | Run the native executor supported-operation matrix on SQLite and record tested cells. |
| FW-13-02A | DL-13 | NOT STARTED | Run the native executor supported-operation matrix on PostgreSQL and record tested cells. |
| FW-13-02B | DL-13 | NOT STARTED | Run the native executor supported-operation matrix on in-memory and record tested cells. |
| FW-14-01 | DL-14 | NOT STARTED | Freeze the mixed-language mode combinations currently claimed by the support matrix. |
| FW-14-01A | DL-14 | NOT STARTED | Run one paired mixed-mode operation test and verify scope fields at the Python/Rust seam. |
| FW-14-01B | DL-14 | NOT STARTED | Add drift protection that rejects an advertised mixed combination without an executable evidence reference. |
| FW-14-03 | DL-14 | NOT STARTED | Verify namespace, workspace, and graph-role preservation for one mixed graph mutation seam. |
| FW-14-03A | DL-14 | NOT STARTED | Verify ACL and source-provenance preservation for one mixed graph read/mutation seam. |
| FW-14-03B | DL-14 | NOT STARTED | Verify runtime/storage handoff preserves the same scope contract at one declared seam. |
| FW-14-04 | DL-14 | NOT STARTED | Verify mixed-mode crash/reopen recovery produces the expected canonical digest for one operation family. |
| FW-14-04A | DL-14 | NOT STARTED | Verify mixed-mode retry does not duplicate durable effects for one idempotent operation family. |
| FW-14-04B | DL-14 | NOT STARTED | Verify a non-idempotent mixed-mode retry fails closed or uses its declared idempotency key. |
| FW-14-02 | DL-14 | NOT STARTED | Add a mixed-mode crash-before-commit recovery test and verify final digest. |
| FW-14-02A | DL-14 | NOT STARTED | Add a mixed-mode crash-after-commit-before-ack recovery test and verify no duplicate effect. |
| FW-14-02B | DL-14 | NOT STARTED | Add a mixed-mode reopen/replay test and verify canonical final-state digest. |
| FW-14A-01 | DL-14A | NOT STARTED | Verify Rust-owned scheduler claims and receives a result from one synchronous Python worker. |
| FW-14A-01A | DL-14A | NOT STARTED | Verify Rust-owned scheduler claims and receives a result from one asynchronous Python worker. |
| FW-14A-01B | DL-14A | NOT STARTED | Verify worker loss is detected and the durable claim can be recovered after scheduler restart. |
| FW-14A-03 | DL-14A | NOT STARTED | Verify result run/step identity and correlation match the active claim before acceptance. |
| FW-14A-03A | DL-14A | NOT STARTED | Verify bounded JSON projection rejects malformed and oversized worker results before journal mutation. |
| FW-14A-03B | DL-14A | NOT STARTED | Verify cross-run and stale-lease results are rejected before graph mutation. |
| FW-14A-04 | DL-14A | NOT STARTED | Verify process-local Python dependencies are absent from durable run state. |
| FW-14A-04A | DL-14A | NOT STARTED | Verify process-local dependencies are re-resolved by run identity after scheduler restart. |
| FW-14A-04B | DL-14A | NOT STARTED | Verify Python objects/callback closures are never serialized into durable state or worker effects. |
| FW-14A-05 | DL-14A | NOT STARTED | Verify exact idempotent result replay on SQLite does not rerun callback or duplicate effects. |
| FW-14A-05A | DL-14A | NOT STARTED | Verify worker loss and lease mismatch recovery on SQLite. |
| FW-14A-05B | DL-14A | NOT STARTED | Verify exact idempotent result replay and crash recovery on PostgreSQL. |
| FW-14A-05C | DL-14A | NOT STARTED | Verify worker loss and lease mismatch recovery on PostgreSQL. |
| FW-14A-02 | DL-14A | NOT STARTED | Verify resolver rejection on SQLite keeps rejected results out of durable state. |
| FW-14A-02A | DL-14A | NOT STARTED | Verify resolver rejection on PostgreSQL keeps rejected results out of durable state. |
| FW-14A-02B | DL-14A | NOT STARTED | Verify bounded JSON projection and dependency isolation on one successful worker result. |
| FW-14C-01 | DL-14C | PASS | Verified exact environment admission, mismatch rejection, and compatible-recipient-first ordering while preserving generic fallback: `tests/runtime/test_workflow_version_admission.py` and focused `test_python_worker_checks_environment_before_numerical_callback`, `test_environment_pinned_resolver_advertises_compatible_claim_recipients`, `test_async_environment_pinned_resolver_advertises_compatible_claim_recipients`; Rust `cargo test -p kogwistar-api sqlite_environment_pinned_work_resumes_on_another_compatible_worker --lib` (1 passed). Combined focused Python run: 42 passed. |
| FW-14C-02 | DL-14C | PASS | Verified wrong-environment work fails before callback/journal mutation, compatible worker B resumes the persisted workflow after service reopen, and Python sync/async worker journals replay prepared effects after restart/retry without re-running callbacks: `test_worker_environment_mismatch_does_not_poison_execution_journal`, Rust `sqlite_environment_pinned_work_resumes_on_another_compatible_worker` (1 passed), `test_worker_journals_callback_before_send_and_replays_after_restart`, and `test_async_worker_journals_awaited_callback_before_response_retry`; combined focused Python run: 42 passed. |
| FW-14C-03 | DL-14C | NOT STARTED | Define a versioned worker identity-attestation payload binding environment identity to image/build and dependency-lock identity. |
| FW-14C-03A | DL-14C | NOT STARTED | Verify Python worker admission rejects missing, stale, and mismatched identity attestations before callback execution. |
| FW-14C-03B | DL-14C | NOT STARTED | Verify Rust scheduler admission rejects missing, stale, and mismatched identity attestations before scheduling/claim. |
| FW-14C-03C | DL-14C | NOT STARTED | Verify Python and Rust accept the same valid attestation fixture and reject the same invalid fixtures. |
| FW-14C-03D | DL-14C | NOT STARTED | Verify admitted worker identity remains bound to a persisted workflow across service restart/resume. |
| FW-15-01 | DL-15 | NOT STARTED | Reconcile selector names and precedence rules with the supported-mode manifest. |
| FW-15-01A | DL-15 | NOT STARTED | Reconcile legacy/default selector behavior with the supported-mode manifest. |
| FW-15-01B | DL-15 | NOT STARTED | Classify every fallback as intentional compatibility or an error and link the owner/evidence. |
| FW-15-03 | DL-15 | NOT STARTED | Verify selector behavior for one advertised backend and operation combination against the manifest. |
| FW-15-03A | DL-15 | NOT STARTED | Verify selector behavior for each remaining advertised backend/operation combination and report unsupported cells. |
| FW-15-03B | DL-15 | NOT STARTED | Add drift assertion that blocks a supported-mode manifest entry without a matching runtime selector. |
| FW-15-02 | DL-15 | NOT STARTED | Verify missing Rust extension fails before effects for one Rust-selected REST operation. |
| FW-15-02A | DL-15 | NOT STARTED | Verify unsupported backend fails before effects for one Rust-selected graph operation. |
| FW-15-02B | DL-15 | NOT STARTED | Verify missing-extension and unsupported-backend failures for the remaining Rust-selected operation classes. |
| FW-15C-01 | DL-15C | NOT STARTED | Verify transaction-owner guard on one workflow continuation write path. |
| FW-15C-01A | DL-15C | NOT STARTED | Verify transaction-owner guard on one callback state-changing path and reject cross-authority writes. |
| FW-15C-01B | DL-15C | NOT STARTED | Verify remaining state-changing call sites are inventoried and each maps to a negative guard test. |
| FW-15C-03 | DL-15C | NOT STARTED | Verify async adapter atomicity with transactional UoW. |
| FW-15C-03A | DL-15C | NOT STARTED | Verify async adapter without transactional UoW fails closed before partial effects. |
| FW-15C-04 | DL-15C | NOT STARTED | Verify nested framework invocation preserves selected transaction ownership. |
| FW-15C-04A | DL-15C | NOT STARTED | Verify framework-managed task/thread context propagation cannot bypass transaction owner guards. |
| FW-15C-04B | DL-15C | NOT STARTED | Verify retained framework handles cannot write after owner context ends. |
| FW-15C-02 | DL-15C | NOT STARTED | Verify async transactional backend commits or rolls back its declared atomic unit. |
| FW-15C-02A | DL-15C | NOT STARTED | Verify async non-transactional backend reports weaker semantics and rejects unsupported atomic operations. |
| FW-16-01 | DL-16 | NOT STARTED | Add derived Node-family round-trip coverage for exact type, wire tag, and extension fields. |
| FW-16-01A | DL-16 | NOT STARTED | Add derived Edge-family round-trip coverage for exact type, wire tag, endpoints, and lifecycle fields. |
| FW-16-01B | DL-16 | NOT STARTED | Verify malformed derived-family payloads reject before graph mutation. |
| FW-16-03 | DL-16 | NOT STARTED | Inventory production derived Node families and define their Python/Rust identity and field contracts. |
| FW-16-03A | DL-16 | NOT STARTED | Inventory production derived Edge families and define their Python/Rust identity, endpoint, and lifecycle contracts. |
| FW-16-03B | DL-16 | NOT STARTED | Verify extension fields are explicitly represented or intentionally excluded in every derived-family wire contract. |
| FW-16-04 | DL-16 | NOT STARTED | Verify shared canonical payload and digest parity for one derived family on the native path. |
| FW-16-04A | DL-16 | NOT STARTED | Verify recovery parity for one derived family on the native path. |
| FW-16-04B | DL-16 | NOT STARTED | Verify shared payload, digest, and invalid-input rejection for one derived family on the mixed path. |
| FW-16-02 | DL-16 | NOT STARTED | Verify shared payload and digest parity for derived Node families on advertised paths. |
| FW-16-02A | DL-16 | NOT STARTED | Verify shared payload and digest parity for derived Edge families on advertised paths. |
| FW-16-02B | DL-16 | NOT STARTED | Verify restart/recovery parity and invalid-payload rejection for derived families. |
| FW-16AC-01 | DL-16AC | NOT STARTED | Verify node ACL outcomes for public, owner, private, scope, and shared/group visibility against Python fixtures. |
| FW-16AC-01A | DL-16AC | NOT STARTED | Verify edge ACL outcomes for public, owner, private, scope, and shared/group visibility against Python fixtures. |
| FW-16AC-01B | DL-16AC | NOT STARTED | Verify latest-version selection and no-record denial for node and edge reads. |
| FW-16AC-01C | DL-16AC | NOT STARTED | Verify tombstone and redirect filtering for node and edge reads against Python. |
| FW-16AC-03 | DL-16AC | NOT STARTED | Verify node ACL decisions for each graph role and same-string cross-kind IDs. |
| FW-16AC-03A | DL-16AC | NOT STARTED | Verify edge ACL decisions for each graph role and same-string cross-kind IDs. |
| FW-16AC-03B | DL-16AC | NOT STARTED | Verify shared/group and latest-version decisions remain scoped to selected graph role. |
| FW-16AC-04 | DL-16AC | NOT STARTED | Verify source-evidence authorization and tombstone/redirect behavior on graph reads. |
| FW-16AC-04A | DL-16AC | NOT STARTED | Verify edge endpoint dependency filtering for authorized and denied endpoint cases. |
| FW-16AC-04B | DL-16AC | NOT STARTED | Verify stale Chroma projection is reconciled after backend restart and graph mutation. |
| FW-16AC-02 | DL-16AC | NOT STARTED | Verify mutation and graph-read ACL parity for same-string node/edge identifiers. |
| FW-16AC-02A | DL-16AC | NOT STARTED | Verify stale Chroma projection recovery after service restart using a fake endpoint. |
| FW-16AC-02B | DL-16AC | NOT STARTED | Verify stale Chroma projection recovery after service restart against live Chroma and label evidence type. |
| FW-16AM-01 | DL-16AM | NOT STARTED | Inventory public MCP operation names, schemas, and discovery metadata against Python. |
| FW-16AM-01A | DL-16AM | NOT STARTED | Verify paired request/response and stable error semantics for one advertised MCP operation family. |
| FW-16AM-01B | DL-16AM | NOT STARTED | Verify authorization behavior for each advertised MCP operation family and reject denied effects. |
| FW-16AM-03 | DL-16AM | NOT STARTED | Verify supported server-initiated event type and sequence ordering against Python. |
| FW-16AM-03A | DL-16AM | NOT STARTED | Verify event replay/resume behavior and explicit unsupported status for unimplemented event types. |
| FW-16AM-03B | DL-16AM | NOT STARTED | Verify transport negotiation for the declared MCP event transport. |
| FW-16AM-04 | DL-16AM | NOT STARTED | Verify public ingestion operation request/response and capability advertisement. |
| FW-16AM-04A | DL-16AM | NOT STARTED | Verify public parsing operation behavior and capability advertisement. |
| FW-16AM-04B | DL-16AM | NOT STARTED | Verify public indexing/backend operation behavior and capability advertisement. |
| FW-16AM-02 | DL-16AM | NOT STARTED | Verify one remaining server-initiated event family and its production transport behavior. |
| FW-16AM-02A | DL-16AM | NOT STARTED | Verify ingestion/parsing/indexing operation boundaries against the support matrix. |
| FW-16AM-02B | DL-16AM | NOT STARTED | Mark unexercised backend/transport combinations explicitly unsupported in discovery. |
| FW-16BQ-01 | DL-16BQ | NOT STARTED | Inventory advertised syscall and REST/MCP operations against Python contract and backend support declarations. |
| FW-16BQ-01A | DL-16BQ | NOT STARTED | Verify syscall request/response parity for SQLite+Chroma on one operation family. |
| FW-16BQ-01B | DL-16BQ | NOT STARTED | Verify syscall request/response parity for PostgreSQL on one operation family. |
| FW-16BQ-01C | DL-16BQ | NOT STARTED | Verify syscall request/response parity for in-memory on one operation family. |
| FW-16BQ-03 | DL-16BQ | NOT STARTED | Verify syscall authorization and namespace/scope semantics against Python for SQLite+Chroma. |
| FW-16BQ-03A | DL-16BQ | NOT STARTED | Verify syscall authorization and namespace/scope semantics against Python for PostgreSQL. |
| FW-16BQ-03B | DL-16BQ | NOT STARTED | Verify syscall authorization and namespace/scope semantics against Python for in-memory. |
| FW-16BQ-03C | DL-16BQ | NOT STARTED | Verify stable syscall error semantics against Python for each advertised backend. |
| FW-16BQ-04 | DL-16BQ | NOT STARTED | Verify native answer/provider event publication and claim on SQLite+Chroma without Python fallback. |
| FW-16BQ-04A | DL-16BQ | NOT STARTED | Verify native event handling, acknowledgment, causal trace, and replay on SQLite+Chroma. |
| FW-16BQ-04B | DL-16BQ | NOT STARTED | Verify native event publication, handling, acknowledgment, causal trace, and replay on PostgreSQL. |
| FW-16BQ-05 | DL-16BQ | NOT STARTED | Verify SQLite+Chroma restart/retry and canonical digest for one advertised syscall path. |
| FW-16BQ-05A | DL-16BQ | NOT STARTED | Verify PostgreSQL restart/retry and canonical digest for one advertised syscall path. |
| FW-16BQ-05B | DL-16BQ | NOT STARTED | Verify supported interpreter/runtime cells and list untested/unsupported combinations. |
| FW-16BQ-02 | DL-16BQ | NOT STARTED | Verify cross-backend ACL parity for one shared operation fixture. |
| FW-16BQ-02A | DL-16BQ | NOT STARTED | Verify cross-backend event/worker state parity for one shared operation fixture. |
| FW-16BQ-02B | DL-16BQ | NOT STARTED | Verify cross-backend final graph-state digest on the declared interpreter matrix. |
| FW-17-01 | DL-17 | NOT STARTED | Run and record required CPython 3.12 CI cells for the declared platform set. |
| FW-17-01A | DL-17 | NOT STARTED | Run and record required CPython 3.13/3.14 CI cells for the declared platform set. |
| FW-17-01B | DL-17 | NOT STARTED | Run and record required PyPy 3.11 cells; retain best-effort status where a cell is not required. |
| FW-17-01C | DL-17 | NOT STARTED | Verify Rust toolchain/platform CI cells and clearly distinguish required versus best-effort. |
| FW-17-03 | DL-17 | NOT STARTED | Verify published wheel metadata and dependency installation on CPython required cells. |
| FW-17-03A | DL-17 | NOT STARTED | Verify native extension loading on each supported CPython/platform cell. |
| FW-17-03B | DL-17 | NOT STARTED | Verify package installation/import behavior on the required PyPy 3.11 cell. |
| FW-17-04 | DL-17 | NOT STARTED | Run downstream pinned-consumer feature tests for the Python implementation path. |
| FW-17-04A | DL-17 | NOT STARTED | Run downstream pinned-consumer regression tests for the Rust/mixed implementation paths. |
| FW-17-04B | DL-17 | NOT STARTED | Reconcile CI, package metadata, and support documentation after downstream runs. |
| FW-17-02 | DL-17 | NOT STARTED | Run pinned-consumer feature tests against the declared package version. |
| FW-17-02A | DL-17 | NOT STARTED | Run pinned-consumer regression tests for each advertised implementation path. |
| FW-17-02B | DL-17 | NOT STARTED | Verify published metadata and documented support matrix match the tested consumer cells. |
| FW-18-01 | DL-18 | NOT STARTED | Freeze one deterministic balanced mixed read/write trace and expected read visibility. |
| FW-18-01A | DL-18 | NOT STARTED | Freeze one deterministic read-heavy trace and expected read visibility. |
| FW-18-01B | DL-18 | NOT STARTED | Verify final canonical digest parity for the balanced trace across measured implementations. |
| FW-18-03 | DL-18 | NOT STARTED | Verify seeded workload and expected per-read visibility at the small fixture size. |
| FW-18-03A | DL-18 | NOT STARTED | Verify seeded workload and expected per-read visibility at the larger fixture size. |
| FW-18-03B | DL-18 | NOT STARTED | Verify Python and Rust trace consumers agree on final digest for both fixture sizes. |
| FW-18-04 | DL-18 | NOT STARTED | Verify individual write commit boundaries and duplicate semantics on SQLite. |
| FW-18-04A | DL-18 | NOT STARTED | Verify commit, duplicate, and replay semantics on PostgreSQL. |
| FW-18-04B | DL-18 | NOT STARTED | Verify supported Chroma projection semantics without claiming vector-store transactionality. |
| FW-18-02 | DL-18 | NOT STARTED | Verify batch atomicity on a backend that declares transaction support. |
| FW-18-02A | DL-18 | NOT STARTED | Verify weaker/non-transactional backend fails or reports partial outcomes according to its contract. |
| FW-19-01 | DL-19 | NOT STARTED | Run fixed-resource PostgreSQL graph operation benchmark at the larger fixture size. |
| FW-19-01A | DL-19 | NOT STARTED | Run fixed-resource indexing benchmark at the larger fixture size. |
| FW-19-01B | DL-19 | NOT STARTED | Run fixed-resource ACL and agent request profiles at the larger fixture size. |
| FW-19-03 | DL-19 | NOT STARTED | Run PostgreSQL profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-03A | DL-19 | NOT STARTED | Run indexing profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-03B | DL-19 | NOT STARTED | Run ACL profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-03C | DL-19 | NOT STARTED | Run agent-request profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-04 | DL-19 | NOT STARTED | Add supported CPython/Rust runtime cells and verify canonical digest parity. |
| FW-19-04A | DL-19 | NOT STARTED | Add replay/recovery benchmark runs and record state digest and resource limits. |
| FW-19-04B | DL-19 | NOT STARTED | Report medians, tail latency, and sample variation for the added runtime/recovery cells. |
| FW-19-02 | DL-19 | NOT STARTED | Add one supported-runtime replay/recovery cell and verify digest parity. |
| FW-19-02A | DL-19 | NOT STARTED | Report median and tail latency for replay/recovery samples. |
| FW-20-01 | DL-20 | NOT STARTED | Attribute endpoint latency to transport and authentication/authorization stages on one representative request. |
| FW-20-01A | DL-20 | NOT STARTED | Attribute endpoint latency to graph storage and indexing stages on one representative request. |
| FW-20-01B | DL-20 | NOT STARTED | Attribute endpoint latency to worker transfer and aggregate those timings across representative sizes/backends. |
| FW-20-03 | DL-20 | NOT STARTED | Add ingress/transport stage timing and operation counter without changing request semantics. |
| FW-20-03A | DL-20 | NOT STARTED | Add authentication/authorization and graph-storage stage timing counters. |
| FW-20-03B | DL-20 | NOT STARTED | Add indexing and worker-transfer stage timing counters and verify attribution. |
| FW-20-04 | DL-20 | NOT STARTED | Repeat resource-matched Python versus mixed endpoint comparison and verify identical digest. |
| FW-20-04A | DL-20 | NOT STARTED | Repeat resource-matched Rust-native endpoint comparison and verify identical digest. |
| FW-20-04B | DL-20 | NOT STARTED | Report latency tails and sample variation for all three modes under matched resources. |
| FW-20-02 | DL-20 | NOT STARTED | Repeat endpoint comparison with correctness digest and enough samples for Python/mixed modes. |
| FW-20-02A | DL-20 | NOT STARTED | Repeat endpoint comparison with correctness digest and enough samples for Rust-native mode. |
| FW-20-02B | DL-20 | NOT STARTED | State performance conclusions only where confidence/variation evidence supports them. |
| FW-20A-01 | DL-20A | NOT STARTED | Join FFI and serialized-byte counters for one representative graph trace. |
| FW-20A-01A | DL-20A | NOT STARTED | Join worker-dispatch and transaction counters for one representative workflow/storage trace. |
| FW-20A-01B | DL-20A | NOT STARTED | Add the joined counters to representative indexing and MCP traces. |
| FW-20A-03 | DL-20A | NOT STARTED | Verify counters join to one trace/run identity and count no duplicate state effects. |
| FW-20A-03A | DL-20A | NOT STARTED | Verify ingress, worker, FFI, serialization, and transaction ownership fields correlate for one mixed trace. |
| FW-20A-04 | DL-20A | NOT STARTED | Add small/large batch-versus-item fixtures and verify equal final-state digest. |
| FW-20A-04A | DL-20A | NOT STARTED | Define regression thresholds for crossings, serialized bytes, and latency using measured baseline. |
| FW-20A-02 | DL-20A | NOT STARTED | Add batch-versus-per-item graph fixture and verify no duplicated state effects. |
| FW-20A-02A | DL-20A | NOT STARTED | Add batch-versus-per-item workflow fixture and verify crossing/serialization counts. |
| FW-20A-02B | DL-20A | NOT STARTED | Add regression assertion for threshold violations without comparing different operation semantics. |

## Current Snapshot

All currently open coarse IDs map to fine rows. The list has a defined tasklet-size contract, but the row-by-row sizing audit is **not complete**: rows that describe a whole inventory, multiple operation families, multiple backend/runtime cells, or multiple independent failure modes must be split before they are started. Therefore `5/265` is the current row-status ratio, **not yet a validated estimate of equal-sized tasklet completion**. Do not describe the fine list as uniformly sized until those rows have been audited and any parent-to-leaf denominator changes are recorded. DL-14C's two passed rows cover environment routing and retry/recovery, not worker identity attestation; FW-14C-03 through FW-14C-03D divide that remainder into contract, language-specific admission, shared fixture parity, and restart binding. CI checks mapping and status consistency; it cannot infer semantic exhaustiveness or effort parity from prose. Record any newly discovered requirement as a child of an existing coarse gate only when required by that gate's contract; otherwise pause and update scope explicitly. The completion snapshot at the top of this file is the sole fine-status/count summary; the linked coarse inventory remains authoritative for coarse status.
