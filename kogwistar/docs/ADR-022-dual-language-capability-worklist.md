# ADR-022 Remaining Worklist

**Updated:** 2026-11-04
**Purpose:** Track bounded work beneath the coarse gates in [ADR-022-dual-language-capability-checklist.md](ADR-022-dual-language-capability-checklist.md).

## Completion Snapshot

- **Coarse checklist:** **156/177 PASS (88.1%)**.
- **Fine worklist:** **91/740 PASS (12.3%)**; **0 IN PROGRESS**, **649 NOT STARTED**, **0 BLOCKED**.
- **Frozen inventory:** DL-03 contains 83 HTTP-route parity leaves, 62 MCP contract leaves, and 62 MCP authorization leaves. These remain one operation/contract per parity or authorization verdict; the HTTP path is not a license to bundle a whole route family. The PostgreSQL Rust-native answer smoke verifies provider-event publication/handling, and its transport test negotiates Streamable HTTP and inspects the structured MCP result. Do not add rows solely to improve a percentage.
- **Equal-size tasklet (S=1):** every counted leaf row is exactly one unit of work: one invariant, one target cell, one focused check, and one binary oracle. Equal size means the same acceptance boundary and active-work budget, not identical source-line count. Target 20 minutes, hard cap 30 minutes of active engineering; environment setup, CI/runtime waiting, and test execution time are excluded. A Python/Rust pair is one cell only when one parity oracle covers one behavior. Commit/rollback may jointly prove one transaction-atomicity invariant; unrelated outcomes are separate tasklets.
- **Tasklet definition:** before work starts, rewrite the row as `Invariant: <one observable behavior> | Cell: <one operation/path/backend/runtime/transport/fixture> | Oracle: <one expected result> | Check: <one focused command> | Active: <=30 min`. An open row that cannot be expressed this way is not a tasklet yet and must not move to `IN PROGRESS` until resized. Existing rows are acceptance targets, not permission to bundle several independent outcomes under one ID.
- **Equal-size boundary:** implementation is not an acceptance unit. “Implement route family,” “finish backend,” and “cover matrix” are not leaves. A leaf may name a matrix/inventory only when one generated completeness assertion yields one binary result without manual per-entry decisions. For ordinary route and MCP rows, one operation is the maximum cell; success-contract parity and authorization are separate cells. One row must not combine different endpoints, authorization policies, backend guarantees, or lifecycle behaviors merely because one test function happens to exercise them.
- **Tasklet resize rule:** before work starts, inspect the acceptance sentence for multiple independently decidable outcomes and identify its one oracle. If two parts can independently pass or fail, retain the original ID as an uncounted parent, create the minimum independently testable children, and recalculate the denominator immediately. Preserve the original requirement and evidence; never count a parent and its children, copy a PASS to an unverified child, or leave a stale denominator after a justified split. Do not split examples that share one oracle or add work outside the coarse gate. A short sentence, one test-function name, or a single CI job does not by itself prove equal sizing.
- **Single-artifact exception:** a generated inventory/manifest and its one completeness assertion may be one leaf only when generation and validation are a single commandable check. Manual decisions or independently failing operation/resource/capability families require separate children; the inventory/reconciliation parent remains uncounted.
- **Status and denominator:** only S=1 leaves count; `PARENT` rows never count or pass. Preserve existing IDs and PASS evidence unless the acceptance itself contains independently decidable required outcomes; then retain the original ID as a non-counted parent and replace it with the minimum independently verifiable S=1 children. The audited splits are FW-06-01 (seven durable resource families; no generated inventory artifact exists), FW-06-01B (three backend guarantees), and FW-06-01-RUN (two registry backend cells and two store-transaction cells); each child has one family/backend cell and one oracle. Current denominator: **740 counted leaves**. Checkpoint graph-node wire parity remains tracked under DL-16; do not duplicate it under DL-06. Do not add speculative rows or split for percentage optics. Coarse and fine percentages are raw status arithmetic, not schedule or completion estimates. A short description or passing aggregate suite is not evidence that a leaf is correctly sized.

## How To Track

- This is the detailed execution list; the linked checklist remains the sole coarse progress and denominator source. Only counted leaf tasklets contribute to the fine percentage.
- The baseline is every coarse ID currently marked `IN PROGRESS` in the linked checklist. Every such coarse ID must map to fine work below. Fine IDs do not add to the 177-item coarse inventory.
- A `PARENT` row groups tasklets or nested parent groups and is not countable or completable. CI validates that every parent resolves to at least two descendant leaf IDs under the same coarse gate and that the hierarchy has no missing references or cycles. No task may be worked or reported as complete through a parent row.
- A fine task reaches `PASS` only when its stated acceptance evidence is recorded here or linked from its referenced test/report. Tests must exercise the named behavior, not merely compile the surrounding code.
- Close a coarse ID only when its original exit criteria and all mapped required fine tasks pass. Do not reopen a passed coarse ID without a reproduced regression or new required contract.
- **Before starting:** write the tasklet in the four-field format and confirm the 30-minute active-work cap. **Before passing:** record the exact check and its result in the same row. If either gate fails, resize the item before work continues; do not mark it `IN PROGRESS` or `PASS` prematurely.
- Words such as `all`, `each`, `every`, `inventory`, `matrix`, and `reconcile` trigger a scope review, not an automatic split. Keep a row only if it yields one binary verdict for one cell; otherwise preserve its ID as an uncounted parent and enumerate the smallest independently passable children. Never count both parent and children, and do not split passed evidence without a reproduced regression.
- Do not create rows per file, field, or platform when they are implementation details of one invariant. Split only when the operation, permission, backend, runtime, transport, or lifecycle outcome can independently pass or fail.

**Status vocabulary:** `NOT STARTED`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

| Fine ID | Coarse ID | Status | S=1 tasklet definition and acceptance evidence (invariant, cell, oracle, focused check) |
| --- | --- | --- | --- |
| FW-02-01 | DL-02 | PASS | The SQLite+Chroma authenticated HTTP protocol/authorization route contract is covered in `tests::sqlite_chroma_graph_topology_reads_are_scoped_and_acl_filtered`: successful canonical node resolution, missing-`read_graph` denial, fail-closed foreign-namespace handling, and metadata-only service rejection with `501 KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER` (**1 passed**). Paired Python ACL/fusion/fixture tests: **5 passed**. |
| FW-02-01B | DL-02 | PASS | The ignored live-Chroma regression `tests::sqlite_search_index_events_project_to_live_chroma_and_survive_reopen` now exercises authenticated `GET /api/search_index_hybrid`: a scoped public node resolves to its canonical document, a newer private ACL revision removes the hit, and missing `read_graph` returns `403`. Verified against disposable `chromadb/chroma:1.5.9` on Windows (**1 passed**); paired fake-endpoint scoped-route regression also passes (**1 passed**). |
| FW-02-02 | DL-02 | PASS | The Python DTO and Rust `SearchIndexUpsertV1` share `contracts/golden/search-index-entry-v1.json`, and frozen OpenAPI defines the request/response schema. Python authorization is covered by `tests/server/test_search_index_api_auth.py::test_add_index_entries_requires_write_role_and_keeps_payload`; Rust mapping is covered by `search_index_entries_match_python_golden_fixture`. This row closes DTO/OpenAPI parity only; public route behavior is covered by FW-02-02A and FW-02-02B. The route is no longer globally deferred: configured implementations are enabled, while backend configurations lacking FTS/vector/embedding dependencies fail closed. `pending_routes_fail_closed_for_every_rust_service` continues to protect the remaining truly deferred route. |
| FW-02-02A | DL-02 | PASS | The authenticated Rust `POST /api/add_index_entries` path on SQLite+Chroma was live-tested against disposable Chroma 1.5.9: missing `write_graph` and mismatched storage namespace each return `403` without appending an event; an authorized request and exact retry return `200`, append one event, and project the changed vector metadata. Evidence: `tests::sqlite_search_index_events_project_to_live_chroma_and_survive_reopen` (**1 passed**, Windows); Python route contract remains covered by `tests/server/test_search_index_api_auth.py::test_add_index_entries_requires_write_role_and_keeps_payload`. |
| FW-02-02B | DL-02 | PASS | Authenticated Rust `POST /api/add_index_entries` was exercised against PostgreSQL/pgvector through the public HTTP router: an authorized request and exact retry both return `200`, with stable event admission. Evidence: `tests::postgres_hybrid_search_is_available_over_authenticated_http` (**1 passed**, live PostgreSQL); durable event replay after service reopen remains covered by `tests::postgres_search_index_events_project_to_pgvector_and_survive_reopen`. In-memory route authorization/idempotency is also covered by `tests::in_memory_search_index_hybrid_is_scoped_and_acl_filtered`. |
| FW-02-02C | DL-02 | PASS | In-memory hybrid GET uses shared fusion and canonical ACL filtering; missing dependencies expose neither index route nor durable event. Evidence: `in_memory_search_index_hybrid_is_scoped_and_acl_filtered` (**1 passed**), Rust API suite (**284 passed, 33 ignored**), Python search-index/ACL/auth/replay suite (**10 passed**), parity and checklist verification (**5 passed**). |
| FW-03-01 | DL-03 | PARENT | Parent scope; children: FW-03-01-DET,FW-03-01-SQLITE,FW-03-01-PG,FW-03-01-PROJ,FW-03-01-QUEUE,FW-03-01-GRAPH,FW-03-01-CHROMA,FW-03-01-WF,FW-03-01-SERVER |
| FW-03-01-DET | DL-03 | NOT STARTED | Link the deterministic-contracts capability claim to one Python behavior test. |
| FW-03-01-SQLITE | DL-03 | NOT STARTED | Link the sqlite-meta capability claim to one Python behavior test. |
| FW-03-01-PG | DL-03 | NOT STARTED | Link the postgres-sequence-event-log capability claim to one Python behavior test. |
| FW-03-01-PROJ | DL-03 | NOT STARTED | Link the projections-snapshots-run-registry capability claim to one Python behavior test. |
| FW-03-01-QUEUE | DL-03 | NOT STARTED | Link the queues-leases-lanes capability claim to one Python behavior test. |
| FW-03-01-GRAPH | DL-03 | NOT STARTED | Link the graph-pgvector capability claim to one Python behavior test. |
| FW-03-01-CHROMA | DL-03 | NOT STARTED | Link the chroma-adapter capability claim to one Python behavior test. |
| FW-03-01-WF | DL-03 | NOT STARTED | Link the workflow-runtime capability claim to one Python behavior test. |
| FW-03-01-SERVER | DL-03 | NOT STARTED | Link the server-rest-sse-mcp-cli capability claim to one Python behavior test. |
| FW-03-01A | DL-03 | PARENT | Parent scope; children: FW-03-01A-DET,FW-03-01A-SQLITE,FW-03-01A-PG,FW-03-01A-PROJ,FW-03-01A-QUEUE,FW-03-01A-GRAPH,FW-03-01A-CHROMA,FW-03-01A-WF,FW-03-01A-SERVER |
| FW-03-01A-DET | DL-03 | NOT STARTED | Link the deterministic-contracts capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-SQLITE | DL-03 | NOT STARTED | Link the sqlite-meta capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-PG | DL-03 | NOT STARTED | Link the postgres-sequence-event-log capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-PROJ | DL-03 | NOT STARTED | Link the projections-snapshots-run-registry capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-QUEUE | DL-03 | NOT STARTED | Link the queues-leases-lanes capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-GRAPH | DL-03 | NOT STARTED | Link the graph-pgvector capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-CHROMA | DL-03 | NOT STARTED | Link the chroma-adapter capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-WF | DL-03 | NOT STARTED | Link the workflow-runtime capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01A-SERVER | DL-03 | NOT STARTED | Link the server-rest-sse-mcp-cli capability claim to one Rust evidence test or explicit unsupported status. |
| FW-03-01B | DL-03 | NOT STARTED | Add a drift test that rejects missing, stale, or duplicate capability-to-evidence links without treating route advertisement as evidence. |
| FW-03-03 | DL-03 | PARENT | Parent scope; children: FW-03-03A |
| FW-03-03A | DL-03 | PARENT | Parent scope; children: FW-03-03A-HTTP-POST-7437a87e,FW-03-03A-HTTP-GET-3f7c2705,FW-03-03A-HTTP-GET-a8089377,FW-03-03A-HTTP-POST-dc297acc,FW-03-03A-HTTP-GET-d41dac18,FW-03-03A-HTTP-GET-e0dc6609,FW-03-03A-HTTP-POST-06ebeb79,FW-03-03A-HTTP-GET-89c079fb,FW-03-03A-HTTP-GET-b51ee1f1,FW-03-03A-HTTP-GET-7dad12d0,FW-03-03A-HTTP-POST-fe141198,FW-03-03A-HTTP-POST-8d28876a,FW-03-03A-HTTP-POST-46a01a68,FW-03-03A-HTTP-POST-6e72c43d,FW-03-03A-HTTP-POST-ff574d21,FW-03-03A-HTTP-GET-942f076b,FW-03-03A-HTTP-POST-8ecb139e,FW-03-03A-HTTP-GET-c01cb68b,FW-03-03A-HTTP-GET-cc1c09d6,FW-03-03A-HTTP-GET-03070335,FW-03-03A-HTTP-GET-ca3e4d2f,FW-03-03A-HTTP-GET-c4ec1841,FW-03-03A-HTTP-GET-350293cb,FW-03-03A-HTTP-POST-16cb26ff,FW-03-03A-HTTP-GET-40879ca4,FW-03-03A-HTTP-GET-d998dc57,FW-03-03A-HTTP-GET-104a818f,FW-03-03A-HTTP-GET-0276cc44,FW-03-03A-HTTP-GET-3fd6dc44,FW-03-03A-HTTP-POST-2fcd7bac,FW-03-03A-HTTP-GET-db651b56,FW-03-03A-HTTP-GET-8bff2a38,FW-03-03A-HTTP-GET-41cea2c6,FW-03-03A-HTTP-GET-ea066435,FW-03-03A-HTTP-GET-4a869379,FW-03-03A-HTTP-POST-9e09cfbb,FW-03-03A-HTTP-POST-3f800ae0,FW-03-03A-HTTP-GET-504d2035,FW-03-03A-HTTP-GET-c60d5d69,FW-03-03A-HTTP-POST-1d601433,FW-03-03A-HTTP-POST-896a4b9c,FW-03-03A-HTTP-DELETE-f853b4cf,FW-03-03A-HTTP-GET-97f82626,FW-03-03A-HTTP-GET-278647b6,FW-03-03A-HTTP-POST-bb56b3cb,FW-03-03A-HTTP-DELETE-6d2b9489,FW-03-03A-HTTP-POST-97e41d1e,FW-03-03A-HTTP-POST-37b8ab5c,FW-03-03A-HTTP-GET-dd837bed,FW-03-03A-HTTP-POST-40910f93,FW-03-03A-HTTP-GET-43d9f440,FW-03-03A-HTTP-GET-b459552c,FW-03-03A-HTTP-POST-a0f3234d,FW-03-03A-HTTP-GET-f86465af,FW-03-03A-HTTP-POST-8957ad0a,FW-03-03A-HTTP-GET-558a42e7,FW-03-03A-HTTP-GET-9e712d3a,FW-03-03A-HTTP-GET-ea750721,FW-03-03A-HTTP-GET-b9d21d15,FW-03-03A-HTTP-GET-dba824ff,FW-03-03A-HTTP-POST-e6b662c3,FW-03-03A-HTTP-GET-41946228,FW-03-03A-HTTP-GET-b5049323,FW-03-03A-HTTP-GET-8d1191ee,FW-03-03A-HTTP-GET-ea5861b7,FW-03-03A-HTTP-POST-90c5d2e7,FW-03-03A-HTTP-POST-811ebc9d,FW-03-03A-HTTP-GET-afa9f3a7,FW-03-03A-HTTP-POST-a98fb634,FW-03-03A-HTTP-POST-96633587,FW-03-03A-HTTP-GET-893e03e9,FW-03-03A-HTTP-POST-6d273898,FW-03-03A-HTTP-POST-232bd6ed,FW-03-03A-HTTP-POST-33c5e064,FW-03-03A-HTTP-GET-6bc448ee,FW-03-03A-HTTP-GET-00fa121f,FW-03-03A-HTTP-POST-1decd954,FW-03-03A-HTTP-GET-539b217e,FW-03-03A-HTTP-GET-8dae33ec,FW-03-03A-HTTP-GET-9cf87495,FW-03-03A-HTTP-GET-0a257751,FW-03-03A-HTTP-GET-64c0cb0d,FW-03-03A-HTTP-GET-562caaba |
| FW-03-03A-HTTP-POST-7437a87e | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/add_index_entries using paired route tests. |
| FW-03-03A-HTTP-GET-3f7c2705 | DL-03 | PASS | Python and Rust OIDC callbacks consume the short-lived `auth_return_to` cookie, revalidate the same local-path/exact-allowlist policy, preserve existing state/PKCE/nonce checks, append the token without dropping an existing query, and clear the cookie after success. Evidence: Rust `cargo test -p kogwistar-api oidc_ --lib` (**4 passed**, including `oidc_callback_validates_rs256_and_persists_identity_before_minting`); Python `pytest -p no:cacheprovider tests/auth/test_auth_router.py -q` (**16 passed**, including callback return-target behavior). |
| FW-03-03A-HTTP-GET-a8089377 | DL-03 | PASS | Python and Rust login accept local `return_to` paths, accept external URLs only by exact configured allowlist, reject credentials/network-path/unallowlisted redirects, and retain `return_to` for callback. Rust `cargo test -p kogwistar-api auth_return_targets_match_python_allowlist_and_reject_open_redirects --lib` (**1 passed**) and `oidc_ --lib` (**4 passed**); Python auth-router module (**16 passed**, including allowlisted and rejected external targets). |
| FW-03-03A-HTTP-POST-dc297acc | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/auth/logout using paired route tests. |
| FW-03-03A-HTTP-GET-d41dac18 | DL-03 | PASS | Python `/api/auth/me` and Rust now return stored user identity/activity fields plus verified-token role/namespace; Rust dev tokens persist identities and legacy signed dev tokens without `user_id` are resolved on first profile lookup. Both routes report an account deactivated after token issuance as inactive. Evidence: `pytest -p no:cacheprovider tests/auth/test_auth_router.py -q` (**16 passed**, including `test_auth_me_reports_current_persisted_activity`); `cargo test -p kogwistar-api dev_token_me_and_logout_share_verified_jwt_contract --lib` (**1 passed**, covering stored activity, legacy token compatibility, role/namespace, forged token denial, and logout). |
| FW-03-03A-HTTP-GET-e0dc6609 | DL-03 | PASS | Python and Rust `GET /api/conversations` return the same `{conversations:[{id,start_node_id,status,turn_count}]}` success shape and restrict results to the authenticated owner. Evidence: Python `pytest -p no:cacheprovider tests/server/test_chat_server_api.py::test_conversation_list_is_user_scoped_and_has_stable_response_shape -q` (**1 passed**), covering a signed `user_id`, spoofed request-body owner, and exclusion of another owner's conversation; Rust `cargo test -p kogwistar-api sqlite_chroma_receive_message_returns_scoped_acl_filtered_transcript --lib` (**1 passed**), covering the public router, response fields, exclusion of a different owner's public conversation, capability/namespace/identity denials, and ACL-filtered conversation data. |
| FW-03-03A-HTTP-POST-06ebeb79 | DL-03 | PASS | Python and Rust `POST /api/conversations` return the canonical conversation identifiers, authenticated owner, and zero-turn initial state while persisting exactly the requested start node. Evidence: Python `pytest -p no:cacheprovider tests/server/test_chat_server_api.py::test_conversation_create_uses_authenticated_owner_and_persists_start_node -q` (**1 passed**), including a spoofed body owner; Rust `cargo test -p kogwistar-api sqlite_chroma_conversation_create_and_turn_admission_are_canonical_and_atomic --lib` (**1 passed**), covering the public route, canonical graph persistence, and atomic admission. |
| FW-03-03A-HTTP-GET-89c079fb | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/conversations/{conversation_id} using paired route tests. |
| FW-03-03A-HTTP-GET-b51ee1f1 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/conversations/{conversation_id}/snapshots/latest using paired route tests. |
| FW-03-03A-HTTP-GET-7dad12d0 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/conversations/{conversation_id}/turns using paired route tests. |
| FW-03-03A-HTTP-POST-fe141198 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/conversations/{conversation_id}/turns:answer using paired route tests. |
| FW-03-03A-HTTP-POST-8d28876a | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/document using paired route tests. |
| FW-03-03A-HTTP-POST-46a01a68 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/document.upsert_tree using paired route tests. |
| FW-03-03A-HTTP-POST-6e72c43d | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/document.validate_graph using paired route tests. |
| FW-03-03A-HTTP-POST-ff574d21 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/graph/upsert using paired route tests. |
| FW-03-03A-HTTP-GET-942f076b | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id} using paired route tests. |
| FW-03-03A-HTTP-POST-8ecb139e | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/runs/{run_id}/cancel using paired route tests. |
| FW-03-03A-HTTP-GET-c01cb68b | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/checkpoints using paired route tests. |
| FW-03-03A-HTTP-GET-cc1c09d6 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/checkpoints/{step_seq} using paired route tests. |
| FW-03-03A-HTTP-GET-03070335 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/events using paired route tests. |
| FW-03-03A-HTTP-GET-ca3e4d2f | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/events/poll using paired route tests. |
| FW-03-03A-HTTP-GET-c4ec1841 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/evidence using paired route tests. |
| FW-03-03A-HTTP-GET-350293cb | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/replay using paired route tests. |
| FW-03-03A-HTTP-POST-16cb26ff | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/runs/{run_id}/resume using paired route tests. |
| FW-03-03A-HTTP-GET-40879ca4 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/resume-contract using paired route tests. |
| FW-03-03A-HTTP-GET-d998dc57 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/runs/{run_id}/steps using paired route tests. |
| FW-03-03A-HTTP-GET-104a818f | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/search_index_hybrid using paired route tests. |
| FW-03-03A-HTTP-GET-0276cc44 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/syscall/v1 using paired route tests. |
| FW-03-03A-HTTP-GET-3fd6dc44 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/syscall/v1/audit using paired route tests. |
| FW-03-03A-HTTP-POST-2fcd7bac | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/syscall/v1/{op} using paired route tests. |
| FW-03-03A-HTTP-GET-db651b56 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/viz/cytoscape.json using paired route tests. |
| FW-03-03A-HTTP-GET-8bff2a38 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/viz/d3.json using paired route tests. |
| FW-03-03A-HTTP-GET-41cea2c6 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/budget using paired route tests. |
| FW-03-03A-HTTP-GET-ea066435 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/budget/history using paired route tests. |
| FW-03-03A-HTTP-GET-4a869379 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/capabilities using paired route tests. |
| FW-03-03A-HTTP-POST-9e09cfbb | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/capabilities/approve using paired route tests. |
| FW-03-03A-HTTP-POST-3f800ae0 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/capabilities/revoke using paired route tests. |
| FW-03-03A-HTTP-GET-504d2035 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/catalog/ops using paired route tests. |
| FW-03-03A-HTTP-GET-c60d5d69 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/dead-letters using paired route tests. |
| FW-03-03A-HTTP-POST-1d601433 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/dead-letters/{run_id}/replay using paired route tests. |
| FW-03-03A-HTTP-POST-896a4b9c | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/design/{workflow_id}/edges using paired route tests. |
| FW-03-03A-HTTP-DELETE-f853b4cf | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP DELETE /api/workflow/design/{workflow_id}/edges/{edge_id} using paired route tests. |
| FW-03-03A-HTTP-GET-97f82626 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/design/{workflow_id}/graph using paired route tests. |
| FW-03-03A-HTTP-GET-278647b6 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/design/{workflow_id}/history using paired route tests. |
| FW-03-03A-HTTP-POST-bb56b3cb | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/design/{workflow_id}/nodes using paired route tests. |
| FW-03-03A-HTTP-DELETE-6d2b9489 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP DELETE /api/workflow/design/{workflow_id}/nodes/{node_id} using paired route tests. |
| FW-03-03A-HTTP-POST-97e41d1e | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/design/{workflow_id}/redo using paired route tests. |
| FW-03-03A-HTTP-POST-37b8ab5c | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/design/{workflow_id}/undo using paired route tests. |
| FW-03-03A-HTTP-GET-dd837bed | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/lane/progress using paired route tests. |
| FW-03-03A-HTTP-POST-40910f93 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/messages/repair-orphans using paired route tests. |
| FW-03-03A-HTTP-GET-43d9f440 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/operator/dashboard using paired route tests. |
| FW-03-03A-HTTP-GET-b459552c | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/resources using paired route tests. |
| FW-03-03A-HTTP-POST-a0f3234d | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/runs using paired route tests. |
| FW-03-03A-HTTP-GET-f86465af | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id} using paired route tests. |
| FW-03-03A-HTTP-POST-8957ad0a | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/runs/{run_id}/cancel using paired route tests. |
| FW-03-03A-HTTP-GET-558a42e7 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/checkpoints using paired route tests. |
| FW-03-03A-HTTP-GET-9e712d3a | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/checkpoints/{step_seq} using paired route tests. |
| FW-03-03A-HTTP-GET-ea750721 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/events using paired route tests. |
| FW-03-03A-HTTP-GET-b9d21d15 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/lineage using paired route tests. |
| FW-03-03A-HTTP-GET-dba824ff | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/replay using paired route tests. |
| FW-03-03A-HTTP-POST-e6b662c3 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/runs/{run_id}/resume using paired route tests. |
| FW-03-03A-HTTP-GET-41946228 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/resume-contract using paired route tests. |
| FW-03-03A-HTTP-GET-b5049323 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/runs/{run_id}/steps using paired route tests. |
| FW-03-03A-HTTP-GET-8d1191ee | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/scheduler/timeline using paired route tests. |
| FW-03-03A-HTTP-GET-ea5861b7 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/services using paired route tests. |
| FW-03-03A-HTTP-POST-90c5d2e7 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services using paired route tests. |
| FW-03-03A-HTTP-POST-811ebc9d | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services/repair using paired route tests. |
| FW-03-03A-HTTP-GET-afa9f3a7 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/services/{service_id} using paired route tests. |
| FW-03-03A-HTTP-POST-a98fb634 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services/{service_id}/disable using paired route tests. |
| FW-03-03A-HTTP-POST-96633587 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services/{service_id}/enable using paired route tests. |
| FW-03-03A-HTTP-GET-893e03e9 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/services/{service_id}/events using paired route tests. |
| FW-03-03A-HTTP-POST-6d273898 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services/{service_id}/heartbeat using paired route tests. |
| FW-03-03A-HTTP-POST-232bd6ed | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services/{service_id}/repair using paired route tests. |
| FW-03-03A-HTTP-POST-33c5e064 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/workflow/services/{service_id}/trigger using paired route tests. |
| FW-03-03A-HTTP-GET-6bc448ee | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/tools/audit using paired route tests. |
| FW-03-03A-HTTP-GET-00fa121f | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/workflow/visibility using paired route tests. |
| FW-03-03A-HTTP-POST-1decd954 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /auth/dev-token using paired route tests. |
| FW-03-03A-HTTP-GET-539b217e | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /designer/capabilities using paired route tests. |
| FW-03-03A-HTTP-GET-8dae33ec | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /health using paired route tests. |
| FW-03-03A-HTTP-GET-9cf87495 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /viz/cytoscape using paired route tests. |
| FW-03-03A-HTTP-GET-0a257751 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /viz/d3 using paired route tests. |
| FW-03-03A-HTTP-GET-64c0cb0d | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /viz/d3.bundle using paired route tests. |
| FW-03-03A-HTTP-GET-562caaba | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /viz/go using paired route tests. |
| FW-03-03B | DL-03 | NOT STARTED | Verify `DELETE /admin/doc/{doc_id}` returns the declared unsupported response and causes no state change. |
| FW-03-04 | DL-03 | PARENT | Parent scope; children: FW-03-04A,FW-03-04B |
| FW-03-04A | DL-03 | PARENT | Parent scope; children: FW-03-04A-MCP-CONVERSATION-ASK,FW-03-04A-MCP-CONVERSATION-CANCEL-RUN,FW-03-04A-MCP-CONVERSATION-CREATE,FW-03-04A-MCP-CONVERSATION-GET-TRANSCRIPT,FW-03-04A-MCP-CONVERSATION-RUN-STATUS,FW-03-04A-MCP-DOCUMENT-ID-FROM-FILE-NAME,FW-03-04A-MCP-KG-FIND-EDGES,FW-03-04A-MCP-KG-K-HOP,FW-03-04A-MCP-KG-LOAD-PERSISTED,FW-03-04A-MCP-KG-NEIGHBORS,FW-03-04A-MCP-KG-PROPOSE-BRUTEFORCE,FW-03-04A-MCP-KG-SEMANTIC-SEED-THEN-EXPAND-TEXT,FW-03-04A-MCP-KG-SHORTEST-PATH,FW-03-04A-MCP-KG-VIZ-CYTOSCAPE-JSON,FW-03-04A-MCP-KG-VIZ-D3-JSON,FW-03-04A-MCP-PROPOSE-VECTOR,FW-03-04A-MCP-WISDOM-KG-UPSERT-GRAPH,FW-03-04A-MCP-WISDOM-SEMANTIC-SEED-THEN-EXPAND,FW-03-04A-MCP-WORKFLOW-BLOCKED-RUNS,FW-03-04A-MCP-WORKFLOW-BUDGET-HISTORY,FW-03-04A-MCP-WORKFLOW-BUDGET-SNAPSHOT,FW-03-04A-MCP-WORKFLOW-CAPABILITIES-SNAPSHOT,FW-03-04A-MCP-WORKFLOW-CAPABILITY-APPROVE,FW-03-04A-MCP-WORKFLOW-CAPABILITY-REVOKE,FW-03-04A-MCP-WORKFLOW-DEAD-LETTER-REPLAY,FW-03-04A-MCP-WORKFLOW-DEAD-LETTERS,FW-03-04A-MCP-WORKFLOW-DESIGN-EDGE-DELETE,FW-03-04A-MCP-WORKFLOW-DESIGN-EDGE-UPSERT,FW-03-04A-MCP-WORKFLOW-DESIGN-HISTORY,FW-03-04A-MCP-WORKFLOW-DESIGN-NODE-DELETE,FW-03-04A-MCP-WORKFLOW-DESIGN-NODE-UPSERT,FW-03-04A-MCP-WORKFLOW-DESIGN-REDO,FW-03-04A-MCP-WORKFLOW-DESIGN-UNDO,FW-03-04A-MCP-WORKFLOW-MESSAGE-ORPHANS-REPAIR,FW-03-04A-MCP-WORKFLOW-OPERATOR-DASHBOARD,FW-03-04A-MCP-WORKFLOW-OPERATOR-INBOX,FW-03-04A-MCP-WORKFLOW-PROCESS-TABLE,FW-03-04A-MCP-WORKFLOW-PROCESS-TIMELINE,FW-03-04A-MCP-WORKFLOW-RUN-CANCEL,FW-03-04A-MCP-WORKFLOW-RUN-CHECKPOINT-GET,FW-03-04A-MCP-WORKFLOW-RUN-EVENTS,FW-03-04A-MCP-WORKFLOW-RUN-EVIDENCE,FW-03-04A-MCP-WORKFLOW-RUN-LINEAGE,FW-03-04A-MCP-WORKFLOW-RUN-REPLAY,FW-03-04A-MCP-WORKFLOW-RUN-RESUME,FW-03-04A-MCP-WORKFLOW-RUN-RESUME-CONTRACT,FW-03-04A-MCP-WORKFLOW-RUN-STATUS,FW-03-04A-MCP-WORKFLOW-RUN-STEPS,FW-03-04A-MCP-WORKFLOW-RUN-SUBMIT,FW-03-04A-MCP-WORKFLOW-SCHEDULER-TIMELINE,FW-03-04A-MCP-WORKFLOW-SERVICE-DECLARE,FW-03-04A-MCP-WORKFLOW-SERVICE-DISABLE,FW-03-04A-MCP-WORKFLOW-SERVICE-ENABLE,FW-03-04A-MCP-WORKFLOW-SERVICE-EVENTS,FW-03-04A-MCP-WORKFLOW-SERVICE-GET,FW-03-04A-MCP-WORKFLOW-SERVICE-HEARTBEAT,FW-03-04A-MCP-WORKFLOW-SERVICE-LIST,FW-03-04A-MCP-WORKFLOW-SERVICE-REPAIR,FW-03-04A-MCP-WORKFLOW-SERVICE-TRIGGER,FW-03-04A-MCP-WORKFLOW-SERVICES-REPAIR,FW-03-04A-MCP-WORKFLOW-TOOL-AUDIT,FW-03-04A-MCP-WORKFLOW-VISIBILITY-SNAPSHOT |
| FW-03-04A-MCP-CONVERSATION-ASK | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP conversation operation conversation.ask using one shared fixture. |
| FW-03-04A-MCP-CONVERSATION-CANCEL-RUN | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP conversation operation conversation.cancel_run using one shared fixture. |
| FW-03-04A-MCP-CONVERSATION-CREATE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP conversation operation conversation.create using one shared fixture. |
| FW-03-04A-MCP-CONVERSATION-GET-TRANSCRIPT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP conversation operation conversation.get_transcript using one shared fixture. |
| FW-03-04A-MCP-CONVERSATION-RUN-STATUS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP conversation operation conversation.run_status using one shared fixture. |
| FW-03-04A-MCP-DOCUMENT-ID-FROM-FILE-NAME | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation document_id_from_file_name using one shared fixture. |
| FW-03-04A-MCP-KG-FIND-EDGES | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_find_edges using one shared fixture. |
| FW-03-04A-MCP-KG-K-HOP | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_k_hop using one shared fixture. |
| FW-03-04A-MCP-KG-LOAD-PERSISTED | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_load_persisted using one shared fixture. |
| FW-03-04A-MCP-KG-NEIGHBORS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_neighbors using one shared fixture. |
| FW-03-04A-MCP-KG-PROPOSE-BRUTEFORCE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_propose_bruteforce using one shared fixture. |
| FW-03-04A-MCP-KG-SEMANTIC-SEED-THEN-EXPAND-TEXT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_semantic_seed_then_expand_text using one shared fixture. |
| FW-03-04A-MCP-KG-SHORTEST-PATH | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_shortest_path using one shared fixture. |
| FW-03-04A-MCP-KG-VIZ-CYTOSCAPE-JSON | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_viz_cytoscape_json using one shared fixture. |
| FW-03-04A-MCP-KG-VIZ-D3-JSON | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation kg_viz_d3_json using one shared fixture. |
| FW-03-04A-MCP-PROPOSE-VECTOR | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation propose_vector using one shared fixture. |
| FW-03-04A-MCP-WISDOM-KG-UPSERT-GRAPH | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation wisdom.kg_upsert_graph using one shared fixture. |
| FW-03-04A-MCP-WISDOM-SEMANTIC-SEED-THEN-EXPAND | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP root operation wisdom.semantic_seed_then_expand using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-BLOCKED-RUNS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.blocked_runs using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-BUDGET-HISTORY | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.budget_history using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-BUDGET-SNAPSHOT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.budget_snapshot using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-CAPABILITIES-SNAPSHOT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.capabilities_snapshot using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-CAPABILITY-APPROVE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.capability_approve using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-CAPABILITY-REVOKE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.capability_revoke using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DEAD-LETTER-REPLAY | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.dead_letter_replay using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DEAD-LETTERS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.dead_letters using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-EDGE-DELETE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_edge_delete using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-EDGE-UPSERT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_edge_upsert using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-HISTORY | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_history using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-NODE-DELETE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_node_delete using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-NODE-UPSERT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_node_upsert using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-REDO | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_redo using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-DESIGN-UNDO | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.design_undo using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-MESSAGE-ORPHANS-REPAIR | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.message_orphans_repair using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-OPERATOR-DASHBOARD | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.operator_dashboard using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-OPERATOR-INBOX | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.operator_inbox using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-PROCESS-TABLE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.process_table using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-PROCESS-TIMELINE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.process_timeline using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-CANCEL | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_cancel using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-CHECKPOINT-GET | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_checkpoint_get using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-EVENTS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_events using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-EVIDENCE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_evidence using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-LINEAGE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_lineage using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-REPLAY | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_replay using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-RESUME | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_resume using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-RESUME-CONTRACT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_resume_contract using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-STATUS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_status using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-STEPS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_steps using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-RUN-SUBMIT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.run_submit using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SCHEDULER-TIMELINE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.scheduler_timeline using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-DECLARE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_declare using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-DISABLE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_disable using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-ENABLE | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_enable using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-EVENTS | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_events using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-GET | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_get using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-HEARTBEAT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_heartbeat using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-LIST | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_list using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-REPAIR | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_repair using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICE-TRIGGER | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.service_trigger using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-SERVICES-REPAIR | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.services_repair using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-TOOL-AUDIT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.tool_audit using one shared fixture. |
| FW-03-04A-MCP-WORKFLOW-VISIBILITY-SNAPSHOT | DL-03 | NOT STARTED | Compare Python/Rust request and response contract for MCP workflow operation workflow.visibility_snapshot using one shared fixture. |
| FW-03-04B | DL-03 | PARENT | Parent scope; children: FW-03-04B-MCP-CONVERSATION-ASK,FW-03-04B-MCP-CONVERSATION-CANCEL-RUN,FW-03-04B-MCP-CONVERSATION-CREATE,FW-03-04B-MCP-CONVERSATION-GET-TRANSCRIPT,FW-03-04B-MCP-CONVERSATION-RUN-STATUS,FW-03-04B-MCP-DOCUMENT-ID-FROM-FILE-NAME,FW-03-04B-MCP-KG-FIND-EDGES,FW-03-04B-MCP-KG-K-HOP,FW-03-04B-MCP-KG-LOAD-PERSISTED,FW-03-04B-MCP-KG-NEIGHBORS,FW-03-04B-MCP-KG-PROPOSE-BRUTEFORCE,FW-03-04B-MCP-KG-SEMANTIC-SEED-THEN-EXPAND-TEXT,FW-03-04B-MCP-KG-SHORTEST-PATH,FW-03-04B-MCP-KG-VIZ-CYTOSCAPE-JSON,FW-03-04B-MCP-KG-VIZ-D3-JSON,FW-03-04B-MCP-PROPOSE-VECTOR,FW-03-04B-MCP-WISDOM-KG-UPSERT-GRAPH,FW-03-04B-MCP-WISDOM-SEMANTIC-SEED-THEN-EXPAND,FW-03-04B-MCP-WORKFLOW-BLOCKED-RUNS,FW-03-04B-MCP-WORKFLOW-BUDGET-HISTORY,FW-03-04B-MCP-WORKFLOW-BUDGET-SNAPSHOT,FW-03-04B-MCP-WORKFLOW-CAPABILITIES-SNAPSHOT,FW-03-04B-MCP-WORKFLOW-CAPABILITY-APPROVE,FW-03-04B-MCP-WORKFLOW-CAPABILITY-REVOKE,FW-03-04B-MCP-WORKFLOW-DEAD-LETTER-REPLAY,FW-03-04B-MCP-WORKFLOW-DEAD-LETTERS,FW-03-04B-MCP-WORKFLOW-DESIGN-EDGE-DELETE,FW-03-04B-MCP-WORKFLOW-DESIGN-EDGE-UPSERT,FW-03-04B-MCP-WORKFLOW-DESIGN-HISTORY,FW-03-04B-MCP-WORKFLOW-DESIGN-NODE-DELETE,FW-03-04B-MCP-WORKFLOW-DESIGN-NODE-UPSERT,FW-03-04B-MCP-WORKFLOW-DESIGN-REDO,FW-03-04B-MCP-WORKFLOW-DESIGN-UNDO,FW-03-04B-MCP-WORKFLOW-MESSAGE-ORPHANS-REPAIR,FW-03-04B-MCP-WORKFLOW-OPERATOR-DASHBOARD,FW-03-04B-MCP-WORKFLOW-OPERATOR-INBOX,FW-03-04B-MCP-WORKFLOW-PROCESS-TABLE,FW-03-04B-MCP-WORKFLOW-PROCESS-TIMELINE,FW-03-04B-MCP-WORKFLOW-RUN-CANCEL,FW-03-04B-MCP-WORKFLOW-RUN-CHECKPOINT-GET,FW-03-04B-MCP-WORKFLOW-RUN-EVENTS,FW-03-04B-MCP-WORKFLOW-RUN-EVIDENCE,FW-03-04B-MCP-WORKFLOW-RUN-LINEAGE,FW-03-04B-MCP-WORKFLOW-RUN-REPLAY,FW-03-04B-MCP-WORKFLOW-RUN-RESUME,FW-03-04B-MCP-WORKFLOW-RUN-RESUME-CONTRACT,FW-03-04B-MCP-WORKFLOW-RUN-STATUS,FW-03-04B-MCP-WORKFLOW-RUN-STEPS,FW-03-04B-MCP-WORKFLOW-RUN-SUBMIT,FW-03-04B-MCP-WORKFLOW-SCHEDULER-TIMELINE,FW-03-04B-MCP-WORKFLOW-SERVICE-DECLARE,FW-03-04B-MCP-WORKFLOW-SERVICE-DISABLE,FW-03-04B-MCP-WORKFLOW-SERVICE-ENABLE,FW-03-04B-MCP-WORKFLOW-SERVICE-EVENTS,FW-03-04B-MCP-WORKFLOW-SERVICE-GET,FW-03-04B-MCP-WORKFLOW-SERVICE-HEARTBEAT,FW-03-04B-MCP-WORKFLOW-SERVICE-LIST,FW-03-04B-MCP-WORKFLOW-SERVICE-REPAIR,FW-03-04B-MCP-WORKFLOW-SERVICE-TRIGGER,FW-03-04B-MCP-WORKFLOW-SERVICES-REPAIR,FW-03-04B-MCP-WORKFLOW-TOOL-AUDIT,FW-03-04B-MCP-WORKFLOW-VISIBILITY-SNAPSHOT |
| FW-03-04B-MCP-CONVERSATION-ASK | DL-03 | NOT STARTED | Verify MCP conversation operation conversation.ask applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-CONVERSATION-CANCEL-RUN | DL-03 | NOT STARTED | Verify MCP conversation operation conversation.cancel_run applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-CONVERSATION-CREATE | DL-03 | NOT STARTED | Verify MCP conversation operation conversation.create applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-CONVERSATION-GET-TRANSCRIPT | DL-03 | NOT STARTED | Verify MCP conversation operation conversation.get_transcript applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-CONVERSATION-RUN-STATUS | DL-03 | NOT STARTED | Verify MCP conversation operation conversation.run_status applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-DOCUMENT-ID-FROM-FILE-NAME | DL-03 | NOT STARTED | Verify MCP root operation document_id_from_file_name applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-FIND-EDGES | DL-03 | NOT STARTED | Verify MCP root operation kg_find_edges applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-K-HOP | DL-03 | NOT STARTED | Verify MCP root operation kg_k_hop applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-LOAD-PERSISTED | DL-03 | NOT STARTED | Verify MCP root operation kg_load_persisted applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-NEIGHBORS | DL-03 | NOT STARTED | Verify MCP root operation kg_neighbors applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-PROPOSE-BRUTEFORCE | DL-03 | NOT STARTED | Verify MCP root operation kg_propose_bruteforce applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-SEMANTIC-SEED-THEN-EXPAND-TEXT | DL-03 | NOT STARTED | Verify MCP root operation kg_semantic_seed_then_expand_text applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-SHORTEST-PATH | DL-03 | NOT STARTED | Verify MCP root operation kg_shortest_path applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-VIZ-CYTOSCAPE-JSON | DL-03 | NOT STARTED | Verify MCP root operation kg_viz_cytoscape_json applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-KG-VIZ-D3-JSON | DL-03 | NOT STARTED | Verify MCP root operation kg_viz_d3_json applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-PROPOSE-VECTOR | DL-03 | NOT STARTED | Verify MCP root operation propose_vector applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WISDOM-KG-UPSERT-GRAPH | DL-03 | NOT STARTED | Verify MCP root operation wisdom.kg_upsert_graph applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WISDOM-SEMANTIC-SEED-THEN-EXPAND | DL-03 | NOT STARTED | Verify MCP root operation wisdom.semantic_seed_then_expand applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-BLOCKED-RUNS | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.blocked_runs applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-BUDGET-HISTORY | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.budget_history applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-BUDGET-SNAPSHOT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.budget_snapshot applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-CAPABILITIES-SNAPSHOT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.capabilities_snapshot applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-CAPABILITY-APPROVE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.capability_approve applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-CAPABILITY-REVOKE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.capability_revoke applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DEAD-LETTER-REPLAY | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.dead_letter_replay applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DEAD-LETTERS | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.dead_letters applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-EDGE-DELETE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_edge_delete applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-EDGE-UPSERT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_edge_upsert applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-HISTORY | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_history applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-NODE-DELETE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_node_delete applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-NODE-UPSERT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_node_upsert applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-REDO | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_redo applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-DESIGN-UNDO | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.design_undo applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-MESSAGE-ORPHANS-REPAIR | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.message_orphans_repair applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-OPERATOR-DASHBOARD | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.operator_dashboard applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-OPERATOR-INBOX | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.operator_inbox applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-PROCESS-TABLE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.process_table applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-PROCESS-TIMELINE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.process_timeline applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-CANCEL | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_cancel applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-CHECKPOINT-GET | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_checkpoint_get applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-EVENTS | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_events applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-EVIDENCE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_evidence applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-LINEAGE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_lineage applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-REPLAY | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_replay applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-RESUME | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_resume applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-RESUME-CONTRACT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_resume_contract applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-STATUS | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_status applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-STEPS | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_steps applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-RUN-SUBMIT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.run_submit applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SCHEDULER-TIMELINE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.scheduler_timeline applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-DECLARE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_declare applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-DISABLE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_disable applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-ENABLE | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_enable applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-EVENTS | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_events applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-GET | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_get applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-HEARTBEAT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_heartbeat applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-LIST | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_list applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-REPAIR | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_repair applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICE-TRIGGER | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.service_trigger applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-SERVICES-REPAIR | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.services_repair applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-TOOL-AUDIT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.tool_audit applies its declared authorization policy and denies unauthorized effects. |
| FW-03-04B-MCP-WORKFLOW-VISIBILITY-SNAPSHOT | DL-03 | NOT STARTED | Verify MCP workflow operation workflow.visibility_snapshot applies its declared authorization policy and denies unauthorized effects. |
| FW-03-02 | DL-03 | NOT STARTED | Add one generated coverage assertion that every HTTP inventory operation maps to exactly one route tasklet or an explicit unsupported classification. |
| FW-03-02A | DL-03 | NOT STARTED | Add one generated coverage assertion that every MCP inventory operation maps to exactly one contract and authorization tasklet. |
| FW-05-01 | DL-05 | PARENT | Parent scope; children: FW-05-01-NODE,FW-05-01-EDGE,FW-05-01-HYPEREDGE |
| FW-05-01-NODE | DL-05 | NOT STARTED | Add paired valid Node DTO fixture; Python and Rust canonical JSON outputs must match exactly. |
| FW-05-01-EDGE | DL-05 | NOT STARTED | Add paired valid Edge DTO fixture; Python and Rust canonical JSON outputs must match exactly. |
| FW-05-01-HYPEREDGE | DL-05 | NOT STARTED | Add paired valid Hyperedge DTO fixture; Python and Rust canonical JSON outputs must match exactly. |
| FW-05-01A | DL-05 | PARENT | Parent scope; children: FW-05-01A-NODE,FW-05-01A-EDGE,FW-05-01A-HYPEREDGE |
| FW-05-01A-NODE | DL-05 | NOT STARTED | Add invalid Node DTO cases; both validators reject them before any graph effect. |
| FW-05-01A-EDGE | DL-05 | NOT STARTED | Add invalid Edge DTO cases; both validators reject them before any graph effect. |
| FW-05-01A-HYPEREDGE | DL-05 | NOT STARTED | Add invalid Hyperedge DTO cases; both validators reject them before any graph effect. |
| FW-05-01B | DL-05 | PARENT | Parent scope; children: FW-05-01B-VALID,FW-05-01B-INVALID |
| FW-05-01B-VALID | DL-05 | NOT STARTED | Add one valid persistence DTO fixture; Python and Rust canonical serialization must match. |
| FW-05-01B-INVALID | DL-05 | NOT STARTED | Add one invalid persistence DTO fixture; Python and Rust must reject before effects. |
| FW-05-01C | DL-05 | PARENT | Parent scope; children: FW-05-01C-REST,FW-05-01C-MCP |
| FW-05-01C-REST | DL-05 | NOT STARTED | Add one REST request/response DTO fixture; Python and Rust canonical payloads must match. |
| FW-05-01C-MCP | DL-05 | NOT STARTED | Add one MCP request/response DTO fixture; Python and Rust canonical payloads must match. |
| FW-05-03 | DL-05 | PARENT | Parent scope; children: FW-05-03-NODE,FW-05-03-EDGE |
| FW-05-03-NODE | DL-05 | NOT STARTED | Verify base Node identity, common fields, lifecycle, and legacy defaults round-trip identically. |
| FW-05-03-EDGE | DL-05 | NOT STARTED | Verify base Edge identity, common fields, lifecycle, and legacy defaults round-trip identically. |
| FW-05-03A | DL-05 | PARENT | Parent scope; children: FW-05-03A-IDENTITY,FW-05-03A-PARTICIPATION,FW-05-03A-ENDPOINT-REJECT |
| FW-05-03A-IDENTITY | DL-05 | NOT STARTED | Verify Python/Rust hyperedge identity serialization against one shared fixture. |
| FW-05-03A-PARTICIPATION | DL-05 | NOT STARTED | Verify Python/Rust hyperedge participation and endpoint serialization against one shared fixture. |
| FW-05-03A-ENDPOINT-REJECT | DL-05 | NOT STARTED | Verify malformed hyperedge endpoints are rejected before graph effects in both languages. |
| FW-05-03B | DL-05 | NOT STARTED | Verify graph mutation DTOs reject malformed entity batches before any document, event, or index effect. |
| FW-05-03C | DL-05 | PARENT | Parent scope; children: FW-05-03C-REDIRECT,FW-05-03C-ENDPOINT |
| FW-05-03C-REDIRECT | DL-05 | NOT STARTED | Verify Python/Rust lifecycle DTO round trips preserve `redirect_to_id` for one tombstoned entity fixture. |
| FW-05-03C-ENDPOINT | DL-05 | NOT STARTED | Verify Python/Rust edge lifecycle DTO round trips preserve canonical endpoint IDs for one edge fixture. |
| FW-05-04 | DL-05 | PARENT | Parent scope; children: FW-05-04-REQUEST,FW-05-04-EVENT |
| FW-05-04-REQUEST | DL-05 | NOT STARTED | Compare provider request DTO canonical serialization and stable validation errors. |
| FW-05-04-EVENT | DL-05 | NOT STARTED | Compare provider event DTO canonical serialization and stable validation errors. |
| FW-05-04A | DL-05 | PARENT | Parent scope; children: FW-05-04A-PERSISTENCE,FW-05-04A-TRANSACTION |
| FW-05-04A-PERSISTENCE | DL-05 | NOT STARTED | Compare persistence error code and detail fields for one shared error fixture. |
| FW-05-04A-TRANSACTION | DL-05 | NOT STARTED | Compare transaction error code and detail fields for one shared error fixture. |
| FW-05-04B | DL-05 | PARENT | Parent scope; children: FW-05-04B-REST,FW-05-04B-MCP |
| FW-05-04B-REST | DL-05 | NOT STARTED | Compare REST error envelope fields for one shared invalid-request fixture. |
| FW-05-04B-MCP | DL-05 | NOT STARTED | Compare MCP error envelope fields for one shared invalid-request fixture. |
| FW-05-02 | DL-05 | PARENT | Parent scope; children: FW-05-02-ID,FW-05-02-SPAN |
| FW-05-02-ID | DL-05 | NOT STARTED | Verify identifier validity boundaries with shared valid/invalid fixtures and matching Python/Rust acceptance. |
| FW-05-02-SPAN | DL-05 | PARENT | Parent scope; children: FW-05-02-SPAN-BOUNDS,FW-05-02-SPAN-OFFSETS |
| FW-05-02-SPAN-BOUNDS | DL-05 | NOT STARTED | Verify source-span start/end bounds against the shared valid/invalid fixtures in Python and Rust. |
| FW-05-02-SPAN-OFFSETS | DL-05 | NOT STARTED | Verify source-span offset validity against the shared valid/invalid fixtures in Python and Rust. |
| FW-05-02A | DL-05 | NOT STARTED | Verify additive schema-version evolution remains accepted in both languages for the declared compatible DTOs. |
| FW-05-02B | DL-05 | NOT STARTED | Verify unsupported schema versions and unknown fields fail consistently before effects. |
| FW-06-01 | DL-06 | PARENT | Parent scope; children: FW-06-01-GRAPH,FW-06-01-SOURCE,FW-06-01-ACL,FW-06-01-DESIGN,FW-06-01-RUN,FW-06-01-WORKER,FW-06-01-INDEX |
| FW-06-01-GRAPH | DL-06 | PASS | Invariant: graph-resource transaction ownership is explicit without claiming one transaction for whole-document ingestion or atomic Chroma writes. Cell: canonical graph entity add/projection across Python authority, Rust SQLite, Rust PostgreSQL, and SQLite+Chroma. Owners: Python `WriteSubsystem._add_node_impl`/edge path enters `Engine.uow`; `Engine._uow_impl` delegates to the configured metadata transaction and backend UoW (SQLite metadata commits via `EngineSQLite.transaction`; Python PostgreSQL metadata via `EnginePostgresMetaStore.transaction`). Rust SQLite `SqliteStore::apply_canonical_graph_mutation_batch` owns `immediate_transaction`; Rust PostgreSQL `PostgresStore::apply_graph_mutation(s)` owns `PostgresStore::transaction`. SQLite+Chroma keeps canonical events and durable index jobs in SQLite; Chroma is an eventual projection, not a transaction authority. The Python extraction loop calls per-entity writes, so no document-wide atomicity is claimed. Checks: Rust `canonical_graph_batch_commits_events_and_projection_jobs_together` (**1 passed**); Python facade/native boundary `test_postgres_graph_mutation_batch_uses_one_native_boundary` and live mixed-writer event ordering `test_python_and_rust_postgres_writers_serialize_shared_event_stream` (**2 passed**, pgvector/pg16). |
| FW-06-01-SOURCE | DL-06 | PARENT | Parent scope; children: FW-06-01-SOURCE-DOCUMENT,FW-06-01-SOURCE-MAP,FW-06-01-SOURCE-EVIDENCE |
| FW-06-01-SOURCE-DOCUMENT | DL-06 | PASS | Invariant: Rust SQLite+Chroma document upsert uses SQLite-owned persistence. Cell: authenticated Rust `/api/document` write. Oracle: authorized scoped projection survives store reopen; denied and cross-scope reads do not expose it. Owner: `SqliteStore::upsert_source_document_projection` owns `immediate_transaction`; Chroma is not authority for source text. Check: `tests::sqlite_chroma_public_document_upsert_is_scope_isolated_and_durable` (**1 passed**). This does not claim Python-path parity. |
| FW-06-01-SOURCE-MAP | DL-06 | PASS | Invariant: Python and Rust source-map serialization matches the shared fixture exactly. Cell: `source-map-v1.json`. Oracle: identical canonical projection and successful Rust wire round-trip. Checks: Python `test_python_source_map_projection_matches_shared_chunk_contract` plus Rust `source_map::tests::shared_source_map_fixture_round_trips` (**1 passed each**). |
| FW-06-01-SOURCE-EVIDENCE | DL-06 | PASS | Invariant: Python node/edge source spans survive Rust PostgreSQL graph mutation persistence. Cell: shared Python graph payload fixture through Rust Postgres. Oracle: spans round-trip through canonical events and projected documents without loss. Check: `test_python_node_and_edge_payloads_round_trip_through_rust_graph_mutation` (**1 passed**, live pgvector/pg16). |
| FW-06-01-ACL | DL-06 | PASS | Invariant: ACL truth remains graph-owned and follows each backend's actual atomicity contract. Cell: default ACL writes for Python and Rust SQLite/PostgreSQL graph paths. Oracle: PostgreSQL rolls back ACL and entity together; Chroma can repair event-first partial projection; Rust ACL events share canonical store commit. Checks: Python `test_acl_mid_write_failure_is_atomic_or_repairable_by_backend` (**2 passed**, pg/chroma); Rust `sqlite_chroma_public_graph_upsert_uses_shared_extraction_contract` (**1 passed**); live `postgres_wisdom_graph_upsert_is_scoped_atomic_and_acl_checked` (**1 passed**). |
| FW-06-01-DESIGN | DL-06 | PASS | Invariant: Python/Rust workflow-design snapshots, deltas, and named projection share the persisted schema and transaction contract. Cell: SQLite and PostgreSQL history stores. Owners: Python `EngineSQLite`/`EnginePostgresMetaStore` transactions; Rust `SqliteStore::put_workflow_design_delta` uses `immediate_transaction` and `PostgresStore::put_workflow_design_delta` uses the store transaction. Check: `test_sqlite_workflow_history_both_directions_pruning_and_atomic_batches` and `test_postgres_workflow_history_python_created_schema_and_atomic_batch` (**2 passed**, PostgreSQL live). |
| FW-06-01-RUN | DL-06 | PARENT | Parent scope; children: FW-06-01-RUN-REGISTRY-SQLITE,FW-06-01-RUN-REGISTRY-POSTGRES,FW-06-01-RUN-COMMIT-SQLITE,FW-06-01-RUN-COMMIT-POSTGRES |
| FW-06-01-RUN-REGISTRY-SQLITE | DL-06 | PASS | Invariant: Python/Rust run-registry writes share visibility and atomic admission. Cell: SQLite registry. Oracle: cross-runtime reads agree; failed batch has no partial rows. Check: `test_sqlite_python_rust_bidirectional_run_registry_and_atomic_batch` (**1 passed**). Active: <=30 min. |
| FW-06-01-RUN-REGISTRY-POSTGRES | DL-06 | PASS | Invariant: Python/Rust run-registry writes share visibility and atomic admission. Cell: PostgreSQL registry. Oracle: cross-runtime reads agree; failed batch has no partial rows. Check: `test_postgres_run_registry_bidirectional_isolated_and_atomic` (**1 passed**, live pgvector/pg16). Active: <=30 min. |
| FW-06-01-RUN-COMMIT-SQLITE | DL-06 | PASS | Invariant: Rust SQLite worker-result transition, checkpoint state, and acknowledgement share one transaction. Cell: Python worker adapter to Rust SQLite store. Oracle: injected failure rolls back all effects; exact retry commits once. Check: `test_claimed_worker_result_atomically_records_and_acks` and `test_fault_after_writes_rolls_back_result_checkpoint_and_ack` (**2 passed**). |
| FW-06-01-RUN-COMMIT-POSTGRES | DL-06 | PASS | Invariant: Rust PostgreSQL runtime transition and provider-event acknowledgement share one transaction. Cell: PostgreSQL recorded transition. Oracle: injected failure rolls back both acknowledgement and runtime state. Check: `tests::runtime::provider_event_ack_and_runtime_transition_share_one_transaction` (**1 passed**, live pgvector/pg16). |
| FW-06-01-WORKER | DL-06 | NOT STARTED | Record Python and Rust transaction owners for worker queues, leases, and durable handoffs. |
| FW-06-01-INDEX | DL-06 | NOT STARTED | Record Python and Rust transaction owners for embedding jobs and derived index projections. |
| FW-06-01A | DL-06 | PARENT | Parent scope; children: FW-06-01A-GRAPH,FW-06-01A-WORKFLOW,FW-06-01A-WORKER,FW-06-01A-ACL,FW-06-01A-INDEX |
| FW-06-01A-GRAPH | DL-06 | PARENT | Parent scope; children: FW-06-01A-GRAPH-SQLITE,FW-06-01A-GRAPH-POSTGRES,FW-06-01A-GRAPH-FACADE,FW-06-01A-GRAPH-MIXED-ORDER |
| FW-06-01A-GRAPH-SQLITE | DL-06 | PASS | Invariant: Rust SQLite graph events, projection, and explicit index intents use one native transaction owner. Cell: Rust `SqliteStore`, SQLite canonical graph batch. Oracle: the store delegates the whole batch to `immediate_transaction`, and events/projection/jobs commit together. Check: `SqliteStore::apply_canonical_graph_mutation_batch` plus `tests::graph_mutations::canonical_graph_batch_commits_events_and_projection_jobs_together` (**1 passed**). |
| FW-06-01A-GRAPH-POSTGRES | DL-06 | PASS | Invariant: Rust PostgreSQL graph event/projection/index-intent writes use one store-owned transaction. Cell: Rust `PostgresStore`, PostgreSQL atomic graph batch. Oracle: the store transaction owns all writes and an injected index-intent failure rolls back the complete batch. Check: `PostgresStore::apply_graph_mutations_with_index_jobs_atomic` plus live `tests::graph::graph_mutation_batch_rolls_back_when_index_intent_fails_when_dsn_available` (**1 passed**). |
| FW-06-01A-GRAPH-FACADE | DL-06 | PASS | Invariant: the Python graph facade in Rust-PostgreSQL mode dispatches an atomic graph batch through one native boundary. Cell: Python facade to Rust Postgres bridge, graph mutation batch. Oracle: `Write._rust_postgres_add` enters `meta.transaction`, whose `RustPostgresSession` owns begin/commit/rollback; one native batch call owns batch admission. Check: `tests/core/test_rust_postgres_session_facade.py::test_postgres_graph_mutation_batch_uses_one_native_boundary` (**1 passed**). |
| FW-06-01A-GRAPH-MIXED-ORDER | DL-06 | PASS | Invariant: independent Python and Rust graph writers serialize into one ordered PostgreSQL event stream. Cell: live mixed-language writers, shared PostgreSQL namespace. Oracle: committed events have unique ordered sequences and both projected entities persist. Check: `tests/core/test_rust_store_graph_mutation_differential.py::test_python_and_rust_postgres_writers_serialize_shared_event_stream` (**1 passed** against live pgvector/pg16). |
| FW-06-01A-WORKFLOW | DL-06 | PARENT | Parent scope; children: FW-06-01A-WORKFLOW-HISTORY-SQLITE,FW-06-01A-WORKFLOW-HISTORY-POSTGRES,FW-06-01A-WORKFLOW-RUN-SQLITE,FW-06-01A-WORKFLOW-RUN-POSTGRES,FW-06-01A-WORKFLOW-TRANSITION-POSTGRES |
| FW-06-01A-WORKFLOW-HISTORY-SQLITE | DL-06 | PASS | Invariant: Rust SQLite workflow-history writes preserve Python-compatible atomic batch and readback semantics. Cell: SQLite workflow-design history, Python/Rust differential. Oracle: identical delta history and pruning state in both directions. Owner: `SqliteStore::put_workflow_design_delta` uses `immediate_transaction`. Check: `tests/core/test_rust_store_workflow_history_differential.py::test_sqlite_workflow_history_both_directions_pruning_and_atomic_batches` (**1 passed**). |
| FW-06-01A-WORKFLOW-HISTORY-POSTGRES | DL-06 | PASS | Invariant: Rust PostgreSQL workflow-history writes preserve the Python schema and atomic batch contract. Cell: PostgreSQL workflow-design history, Python/Rust differential. Oracle: Python-created schema round-trips atomic history mutations through Rust and Python. Owner: `PostgresStore::put_workflow_design_delta` opens the store transaction. Check: `tests/core/test_rust_store_workflow_history_differential.py::test_postgres_workflow_history_python_created_schema_and_atomic_batch` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-WORKFLOW-RUN-SQLITE | DL-06 | PASS | Invariant: Rust SQLite run-registry writes preserve Python-compatible visibility and atomic batch semantics. Cell: SQLite run registry, Python/Rust differential. Oracle: both runtimes read each other's committed rows and atomic batches. Owner: `SqliteStore::create_server_run` uses `immediate_transaction`. Check: `tests/core/test_rust_store_run_registry_differential.py::test_sqlite_python_rust_bidirectional_run_registry_and_atomic_batch` (**1 passed**). |
| FW-06-01A-WORKFLOW-RUN-POSTGRES | DL-06 | PASS | Invariant: Rust PostgreSQL run-registry writes preserve Python-compatible visibility and atomic batch semantics. Cell: PostgreSQL run registry, Python/Rust differential. Oracle: both runtimes read each other's committed rows and atomic batches. Owner: `PostgresStore::create_server_run` opens the store transaction. Check: `tests/core/test_rust_store_run_registry_differential.py::test_postgres_run_registry_bidirectional_isolated_and_atomic` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-WORKFLOW-TRANSITION-POSTGRES | DL-06 | PASS | Invariant: provider-event acknowledgement and recorded runtime transition share the PostgreSQL store-owned transaction. Cell: Rust PostgreSQL runtime transition with claimed provider event. Oracle: injected failure rolls back both acknowledgement and transition effects. Owner: `PostgresStore::apply_recorded_runtime_transition` opens the UoW transaction. Check: `tests::runtime::provider_event_ack_and_runtime_transition_share_one_transaction` (**1 passed** against live PostgreSQL). |
| FW-06-01A-WORKER | DL-06 | PARENT | Parent scope; children: FW-06-01A-WORKER-SQLITE-RUST,FW-06-01A-WORKER-SQLITE-ADAPTER |
| FW-06-01A-WORKER-SQLITE-RUST | DL-06 | PASS | Invariant: a Rust SQLite worker effect and exact lane acknowledgement commit atomically. Cell: Rust-native `SqliteStore` worker-result handoff. Oracle: the recorded result and acknowledgement both commit, or neither does; exact retry is idempotent. Check: `tests::runtime::worker_failure_routing_is_atomic_handled_terminal_and_retry_safe` (**1 passed**). |
| FW-06-01A-WORKER-SQLITE-ADAPTER | DL-06 | PASS | Invariant: the Python worker-result adapter delegates one claimed result to the Rust-owned SQLite transaction. Cell: Python adapter to Rust store, SQLite worker handoff. Oracle: transition, checkpoint, and acknowledgement commit once and exact retry adds no event. Check: `tests/runtime/test_rust_runtime_worker_handoff.py::test_claimed_worker_result_atomically_records_and_acks` (**1 passed**). |
| FW-06-01A-ACL | DL-06 | PARENT | Parent scope; children: FW-06-01A-ACL-SQLITE-CHROMA,FW-06-01A-ACL-POSTGRES,FW-06-01A-ACL-PYTHON-RUST |
| FW-06-01A-ACL-SQLITE-CHROMA | DL-06 | PASS | Invariant: host-derived default ACL records and index intents share the canonical SQLite transaction. Cell: Rust SQLite+Chroma wisdom graph upsert. Oracle: graph entities and ACL records commit as canonical events before eventual Chroma projection. Check: `tests::sqlite_chroma_wisdom_upsert_uses_canonical_events_and_acl_before_projection` (**1 passed**). |
| FW-06-01A-ACL-POSTGRES | DL-06 | PASS | Invariant: host-derived default ACL mutations share the PostgreSQL graph-upsert transaction. Cell: Rust PostgreSQL wisdom graph upsert, isolated schema. Oracle: accepted entities receive canonical ACL records, while unauthorized or invalid writes have no accepted graph effect. Check: `tests::postgres_wisdom_graph_upsert_is_scoped_atomic_and_acl_checked` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-ACL-PYTHON-RUST | DL-06 | PASS | Invariant: Python `record_acl` writes its ACL node and supersedes edge under one Rust PostgreSQL transaction. Cell: live PostgreSQL mixed Python ACL facade, second-version update. Oracle: both mutations use one native transaction ID; injected post-edge index failure rolls both back and preserves the original ACL node. Check: `tests/core/test_rust_postgres_transaction_session.py::test_python_acl_supersession_uses_one_rust_postgres_transaction` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-INDEX | DL-06 | PARENT | Parent scope; children: FW-06-01A-INDEX-SYNC-NODE,FW-06-01A-INDEX-SYNC-EDGE,FW-06-01A-INDEX-ASYNC-NODE,FW-06-01A-INDEX-ASYNC-EDGE |
| FW-06-01A-INDEX-SYNC-NODE | DL-06 | PASS | Invariant: Rust-authority PostgreSQL node event, Stage-1 projection, and embedding job intent roll back together. Cell: synchronous two-stage node admission. Oracle: injected enqueue failure leaves no target event, projection, or job. Check: `tests/pg_sql/test_two_stage_postgres_projection.py::test_postgres_two_stage_rust_admission_rolls_back_index_job_with_graph[node]` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-INDEX-SYNC-EDGE | DL-06 | PASS | Invariant: Rust-authority PostgreSQL edge event, Stage-1 projection, and embedding job intent roll back together. Cell: synchronous two-stage edge admission. Oracle: injected enqueue failure leaves no target event, projection, or job. Check: `tests/pg_sql/test_two_stage_postgres_projection.py::test_postgres_two_stage_rust_admission_rolls_back_index_job_with_graph[edge]` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-INDEX-ASYNC-NODE | DL-06 | PASS | Invariant: Rust-authority PostgreSQL async node event, Stage-1 projection, and embedding job intent roll back together. Cell: async two-stage node admission across worker-thread calls. Oracle: injected enqueue failure leaves no target event, projection, or job. Check: `tests/pg_sql/test_two_stage_postgres_projection.py::test_postgres_two_stage_async_rust_admission_rolls_back_index_job[node]` (**1 passed**, live pgvector/pg16). |
| FW-06-01A-INDEX-ASYNC-EDGE | DL-06 | PASS | Invariant: Rust-authority PostgreSQL async edge event, Stage-1 projection, and embedding job intent roll back together. Cell: async two-stage edge admission across worker-thread calls. Oracle: injected enqueue failure leaves no target event, projection, or job. Check: `tests/pg_sql/test_two_stage_postgres_projection.py::test_postgres_two_stage_async_rust_admission_rolls_back_index_job[edge]` (**1 passed**, live pgvector/pg16). |
| FW-06-01B | DL-06 | PARENT | Parent scope; children: FW-06-01B-POSTGRES,FW-06-01B-SQLITE-CHROMA,FW-06-01B-MEMORY |
| FW-06-01B-POSTGRES | DL-06 | PASS | ADR-022 states PostgreSQL atomically commits canonical events, graph projections, and explicit index intents; `graph_mutation_batch_rolls_back_when_index_intent_fails_when_dsn_available` passed against disposable pgvector/pg16 (**1 passed**). |
| FW-06-01B-SQLITE-CHROMA | DL-06 | PASS | ADR-022/ADR-018 define canonical SQLite events and Chroma as an eventual projection; `sqlite_chroma_graph_mutation_commits_canonical_events_and_eventual_jobs` verifies failure, reopen, and retry (**1 passed**, fake HTTP Chroma endpoint; not live Chroma). |
| FW-06-01B-MEMORY | DL-06 | PASS | `InMemoryStore` documents volatile state and process-local handles; `cloned_handles_share_volatile_state_but_new_store_starts_empty` verifies clone visibility and fresh-store isolation (**1 passed**). |
| FW-06-03 | DL-06 | PASS | `concurrent_event_appends_have_one_ordered_sqlite_authority` verifies unique ordered canonical events under eight writers; Chroma remains eventual (**1 passed**). |
| FW-06-03A | DL-06 | PARENT | Parent scope; children: FW-06-03A-GRAPH,FW-06-03A-RUN,FW-06-03A-WORKER,FW-06-03A-ACL,FW-06-03A-INDEX |
| FW-06-03A-GRAPH | DL-06 | PASS | Live `postgres_concurrent_graph_mutations_share_one_transaction_order_when_dsn_available` uses separate pools and verifies contiguous events plus final projection (**1 passed**). |
| FW-06-03A-RUN | DL-06 | PASS | Two independent PostgreSQL pools raced start transitions for one run; exactly one committed and the stale-sequence loser conflicted. The single event, run row, and reconstructed state agree. Evidence: `tests::runtime::postgres_concurrent_run_starts_have_one_transaction_owner_when_dsn_available` (live pgvector/PostgreSQL, **1 passed**). |
| FW-06-03A-WORKER | DL-06 | PASS | Two PostgreSQL pools raced to claim one pending lane message; exactly one worker received it, and the durable `claimed_by` matched. Evidence: `tests::runtime::postgres_concurrent_lane_claims_have_one_worker_owner_when_dsn_available` (live pgvector/PostgreSQL, **1 passed**). |
| FW-06-03A-ACL | DL-06 | PASS | Two PostgreSQL pools concurrently replaced one ACL record; ordered canonical events and ACL lookup agree on the final policy. Evidence: `tests::graph::postgres_concurrent_acl_mutations_keep_event_and_truth_projection_order_when_dsn_available` (live pgvector/PostgreSQL, **1 passed**). |
| FW-06-03A-INDEX | DL-06 | PASS | Two pools concurrently committed graph updates with the same index-job coalesce key; one pending intent remained and matched the final event/projection. Evidence: `tests::graph::postgres_concurrent_graph_writes_keep_coalesced_index_intent_current_when_dsn_available` (live pgvector/PostgreSQL, **1 passed**). |
| FW-06-03B | DL-06 | PASS | `conflicting_graph_writes_are_serialized_within_process` verifies ordered events and matching final projection; `InMemoryStore` docs state the process-local boundary (**1 passed**). |
| FW-06-04 | DL-06 | PASS | `sqlite_chroma_graph_mutation_commits_canonical_events_and_eventual_jobs` injects Chroma failure while canonical SQLite events/jobs remain readable (**1 passed**). |
| FW-06-04A | DL-06 | PASS | The same test restarts and retries projection until all jobs finish; it asserts repeated identical Chroma upserts (**1 passed**). |
| FW-06-05 | DL-06 | PASS | `canonical_graph_batch_rolls_back_when_stage1_metadata_projection_fails` injects pre-commit failure and asserts no event, job, or partial projection (**1 passed**). |
| FW-06-05A | DL-06 | PASS | The SQLite+Chroma integration test injects post-commit projection/ack failures, then retries without duplicate events (**1 passed**). |
| FW-06-05B | DL-06 | PASS | `canonical_graph_batch_rolls_back_event_when_projection_job_insert_fails` fails the second entity's intent and asserts full batch rollback (**1 passed**). |
| FW-06-02 | DL-06 | PARENT | Parent scope; children: FW-06-02-REOPEN,FW-06-02-RETRY |
| FW-06-02-REOPEN | DL-06 | PASS | A committed graph event/projection survived reopen and a later injected Stage-1 write failure; after another reopen the failed event/projection were absent. Evidence: `tests::graph_mutations::committed_canonical_graph_state_survives_reopen_after_injected_batch_failure` (**1 passed**). |
| FW-06-02-RETRY | DL-06 | PASS | Exact canonical graph-batch retry after reopen returned the original event sequence, kept one event and the same job IDs/count, and preserved the projection sequence. Evidence: `tests::graph_mutations::canonical_graph_retry_after_reopen_does_not_duplicate_events_or_index_intents` (**1 passed**). |
| FW-06-02A | DL-06 | PARENT | Parent scope; children: FW-06-02A-REOPEN,FW-06-02A-RETRY |
| FW-06-02A-REOPEN | DL-06 | PASS | Invariant: a previously committed graph event/projection survives a later failed mutation, and the failed entity leaves no durable effect. Cell: Rust PostgreSQL store, fresh schema, with the store reopened after failure. Oracle: replay contains exactly the committed event and projection, while the failed ID is absent. Check: `tests::graph::committed_graph_state_survives_reopen_after_injected_graph_failure_when_dsn_available` (**1 passed** against live PostgreSQL; active work estimated at 20 minutes). |
| FW-06-02A-RETRY | DL-06 | PASS | Invariant: retrying an identical graph mutation and index intent after reopening returns the original event sequence without duplicate events or jobs; conflicting reuse remains rejected. Cell: Rust PostgreSQL store, live database, exact batch before/after store reopen. Oracle: retry reports no new mutation, event sequence and job identity/count remain unchanged, conflicting payload fails. Check: `tests::graph::graph_mutation_retry_requires_identical_effect_when_dsn_available` (**1 passed** against live PostgreSQL; active work estimated at 20 minutes). |
| FW-08-01 | DL-08 | NOT STARTED | Add a live PostgreSQL graph-neighbor read test for tenant/project scope and source-evidence authorization. |
| FW-08-01A | DL-08 | NOT STARTED | Add a live PostgreSQL graph-snapshot test for edge endpoint dependency filtering and malformed backend records. |
| FW-08-01B | DL-08 | PARENT | Parent scope; children: FW-08-01B-NEIGHBOR,FW-08-01B-SNAPSHOT,FW-08-01B-EDGE-DEPENDENCY,FW-08-01B-VECTOR |
| FW-08-01B-NEIGHBOR | DL-08 | NOT STARTED | Verify a conflicting-scope record is omitted from live PostgreSQL neighbor results. |
| FW-08-01B-SNAPSHOT | DL-08 | NOT STARTED | Verify a conflicting-scope node is omitted from live PostgreSQL graph snapshots. |
| FW-08-01B-EDGE-DEPENDENCY | DL-08 | NOT STARTED | Verify PostgreSQL snapshots omit edges whose endpoint has conflicting scope. |
| FW-08-01B-VECTOR | DL-08 | NOT STARTED | Verify a conflicting-scope record is omitted from live PostgreSQL vector results. |
| FW-08-03 | DL-08 | NOT STARTED | Verify live SQLite+Chroma public graph reads revalidate canonical scope and ACL after vector/backend response. |
| FW-08-03A | DL-08 | NOT STARTED | Verify live SQLite+Chroma endpoint dependencies and stale-projection handling on graph reads. |
| FW-08-04 | DL-08 | NOT STARTED | Verify in-memory authenticated node reads match Python scope/ACL behavior for allowed and denied records. |
| FW-08-04A | DL-08 | NOT STARTED | Verify in-memory edge/adjacency reads fail closed for absent, malformed, or cross-scope endpoints. |
| FW-08-02 | DL-08 | PARENT | Parent scope; children: FW-08-02-PG,FW-08-02-SQLITE-CHROMA,FW-08-02-MEMORY |
| FW-08-02-PG | DL-08 | NOT STARTED | After ACL revocation, verify PostgreSQL public reads omit the revoked record. |
| FW-08-02-SQLITE-CHROMA | DL-08 | NOT STARTED | After ACL revocation, verify SQLite+Chroma public reads omit the revoked record. |
| FW-08-02-MEMORY | DL-08 | NOT STARTED | After ACL revocation, verify in-memory public reads omit the revoked record. |
| FW-08-02A | DL-08 | NOT STARTED | Verify public read effects with stale/missing canonical records and compare the supported backend result shapes. |
| FW-08-02B | DL-08 | NOT STARTED | Verify public read effects do not leak denied nodes through adjacency, snapshots, or vector-derived results. |
| FW-08C-01 | DL-08C | PARENT | Parent scope; children: FW-08C-01-TOMBSTONE,FW-08C-01-REDIRECT |
| FW-08C-01-TOMBSTONE | DL-08C | NOT STARTED | Verify Python and Rust vector retrieval excludes tombstoned entities. |
| FW-08C-01-REDIRECT | DL-08C | NOT STARTED | Verify Python and Rust vector retrieval resolves redirects to the canonical target. |
| FW-08C-01A | DL-08C | NOT STARTED | Verify vector retrieval requires authorized source evidence and rejects stale/foreign evidence references. |
| FW-08C-01B | DL-08C | NOT STARTED | Run the lifecycle/evidence vector cases on PostgreSQL and record the exact result parity. |
| FW-08C-01C | DL-08C | NOT STARTED | Run the lifecycle/evidence vector cases on SQLite+Chroma and record the exact result parity. |
| FW-08C-01D | DL-08C | NOT STARTED | Run the lifecycle/evidence vector cases on in-memory and record the exact result parity. |
| FW-08C-03 | DL-08C | PARENT | Parent scope; children: FW-08C-03-PG,FW-08C-03-SQLITE-CHROMA,FW-08C-03-MEMORY |
| FW-08C-03-PG | DL-08C | NOT STARTED | Verify PostgreSQL filters lifecycle state before applying the vector top-k limit. |
| FW-08C-03-SQLITE-CHROMA | DL-08C | NOT STARTED | Verify SQLite+Chroma filters lifecycle state before applying the vector top-k limit. |
| FW-08C-03-MEMORY | DL-08C | NOT STARTED | Verify in-memory retrieval filters lifecycle state before applying the vector top-k limit. |
| FW-08C-03A | DL-08C | NOT STARTED | Verify ACL-denied candidates are omitted without unauthorized backfill changing result limits. |
| FW-08C-03B | DL-08C | PARENT | Parent scope; children: FW-08C-03B-PG,FW-08C-03B-SQLITE-CHROMA,FW-08C-03B-MEMORY |
| FW-08C-03B-PG | DL-08C | NOT STARTED | Verify normalized score meaning and ordering for supported metrics on PostgreSQL. |
| FW-08C-03B-SQLITE-CHROMA | DL-08C | NOT STARTED | Verify normalized score meaning and ordering for supported metrics on SQLite+Chroma. |
| FW-08C-03B-MEMORY | DL-08C | NOT STARTED | Verify normalized score meaning and ordering for supported metrics in-memory. |
| FW-08C-04 | DL-08C | PARENT | Parent scope; children: FW-08C-04-PG,FW-08C-04-SQLITE-CHROMA,FW-08C-04-MEMORY |
| FW-08C-04-PG | DL-08C | NOT STARTED | Verify tenant, project, and graph-role scope on PostgreSQL vector retrieval. |
| FW-08C-04-SQLITE-CHROMA | DL-08C | NOT STARTED | Verify tenant, project, and graph-role scope on SQLite+Chroma vector retrieval. |
| FW-08C-04-MEMORY | DL-08C | NOT STARTED | Verify tenant, project, and graph-role scope on in-memory vector retrieval. |
| FW-08C-04A | DL-08C | NOT STARTED | Verify source span/provenance and endpoint authorization on vector results. |
| FW-08C-04B | DL-08C | NOT STARTED | Verify the same scope and evidence constraints on hybrid retrieval results. |
| FW-08C-02 | DL-08C | PARENT | Parent scope; children: FW-08C-02-PG,FW-08C-02-MEMORY |
| FW-08C-02-PG | DL-08C | NOT STARTED | Verify Python/Rust vector mutation admission scope parity on PostgreSQL. |
| FW-08C-02-MEMORY | DL-08C | NOT STARTED | Verify Python/Rust vector mutation admission scope parity in-memory. |
| FW-08C-02A | DL-08C | NOT STARTED | Verify mutation admission scope parity for SQLite+Chroma and record unsupported combinations explicitly. |
| FW-11-01 | DL-11 | NOT STARTED | Compare canonical trace fields for one successful runtime transition against a shared Python/Rust fixture. |
| FW-11-01A | DL-11 | NOT STARTED | Compare trace fields for retry and attempt increments against a shared Python/Rust fixture. |
| FW-11-01B | DL-11 | PARENT | Parent scope; children: FW-11-01B-RESUME,FW-11-01B-ERROR |
| FW-11-01B-RESUME | DL-11 | NOT STARTED | Compare trace fields for one suspend/resume transition against its shared fixture. |
| FW-11-01B-ERROR | DL-11 | NOT STARTED | Compare trace fields for one terminal error transition against its shared fixture. |
| FW-11-03 | DL-11 | PARENT | Parent scope; children: FW-11-03-PARENTAGE,FW-11-03-RUN-STEP,FW-11-03-SEQUENCE |
| FW-11-03-PARENTAGE | DL-11 | NOT STARTED | Verify parent-span linkage for one synchronous runtime transition. |
| FW-11-03-RUN-STEP | DL-11 | NOT STARTED | Verify run and step identity fields for one synchronous runtime transition. |
| FW-11-03-SEQUENCE | DL-11 | NOT STARTED | Verify sequence ordering for one synchronous runtime transition. |
| FW-11-03A | DL-11 | PARENT | Parent scope; children: FW-11-03A-PARENTAGE,FW-11-03A-CAUSAL-EDGE |
| FW-11-03A-PARENTAGE | DL-11 | NOT STARTED | Verify parent-span linkage for one asynchronous runtime transition. |
| FW-11-03A-CAUSAL-EDGE | DL-11 | NOT STARTED | Verify one asynchronous event-to-step causal edge. |
| FW-11-03B | DL-11 | PARENT | Parent scope; children: FW-11-03B-RETRY,FW-11-03B-TERMINAL |
| FW-11-03B-RETRY | DL-11 | NOT STARTED | Verify retry trace preserves parent span, incremented attempt, and event identity. |
| FW-11-03B-TERMINAL | DL-11 | NOT STARTED | Verify terminal-failure trace preserves parent span, final attempt, and event identity. |
| FW-11-04 | DL-11 | NOT STARTED | Verify authenticated trace/run-event inspection on SQLite before and after reopen/replay. |
| FW-11-04A | DL-11 | NOT STARTED | Verify authenticated trace/run-event inspection on PostgreSQL before and after reopen/replay. |
| FW-11-04B | DL-11 | NOT STARTED | Verify legacy persisted trace rows with omitted optional fields remain readable and are not misreported. |
| FW-11-02 | DL-11 | NOT STARTED | Verify authenticated SQLite inspection returns correct trace parentage and event-to-step linkage. |
| FW-11-02A | DL-11 | NOT STARTED | Verify authenticated PostgreSQL inspection returns correct trace parentage and event-to-step linkage. |
| FW-12-01 | DL-12 | NOT STARTED | Verify explicit Python selector behavior for one REST operation family with Rust extension absent. |
| FW-12-01A | DL-12 | NOT STARTED | Verify explicit Python selector behavior for one MCP operation family with Rust extension absent. |
| FW-12-01B | DL-12 | NOT STARTED | Verify legacy selector defaults and configured graph-role selection remain unchanged in Python-only mode. |
| FW-12-03 | DL-12 | PARENT | Parent scope; children: FW-12-03-INGRESS,FW-12-03-MUTATION |
| FW-12-03-INGRESS | DL-12 | NOT STARTED | Verify one Python-only REST ingress request completes without importing or calling Rust. |
| FW-12-03-MUTATION | DL-12 | NOT STARTED | Verify one Python-only graph mutation completes without importing or calling Rust. |
| FW-12-03A | DL-12 | NOT STARTED | Verify unsupported/missing Rust extension does not alter Python-only REST behavior or trigger an implicit Rust load. |
| FW-12-04 | DL-12 | NOT STARTED | Verify Python-only MCP in-process authorization and response/error shape without Rust import. |
| FW-12-04A | DL-12 | NOT STARTED | Verify Python-only MCP HTTP transport authorization and response/error shape without Rust import. |
| FW-12-02 | DL-12 | NOT STARTED | Verify Python-only indexing emits durable job state through a public interface. |
| FW-12-02A | DL-12 | PARENT | Parent scope; children: FW-12-02A-RETRIEVAL,FW-12-02A-VISUALIZATION |
| FW-12-02A-RETRIEVAL | DL-12 | NOT STARTED | Verify Python-only retrieval reads back one indexed result on a supported backend. |
| FW-12-02A-VISUALIZATION | DL-12 | NOT STARTED | Verify Python-only visualization reads back one indexed result on a supported backend. |
| FW-13-01 | DL-13 | NOT STARTED | Generate a native-operation inventory and one drift assertion for handler/version/fixture mappings. |
| FW-13-01A | DL-13 | NOT STARTED | Verify one advertised native operation's output against the Python/shared fixture contract. |
| FW-13-01B | DL-13 | PASS | Unknown, unsupported-version, and malformed native operations are rejected before run admission: the API returns `422` and leaves neither a server run nor a runtime lane message. Evidence: `tests::sqlite_native_runtime_route_uses_shared_result_reducer` (**1 passed** via `cargo test --manifest-path rust/Cargo.toml -p kogwistar-api sqlite_native_runtime_route_uses_shared_result_reducer --lib`). |
| FW-13-03 | DL-13 | PARENT | Parent scope; children: FW-13-03-VERSIONED,FW-13-03-LEGACY |
| FW-13-03-VERSIONED | DL-13 | PASS | Explicit `v2` on a registered custom native operation is preserved through runtime claim, native execution, and durable completion; the final transition records the `v2` handler result. Evidence: `tests::sqlite_native_runtime_route_uses_shared_result_reducer` (**1 passed**). |
| FW-13-03-LEGACY | DL-13 | PASS | An omitted operation version defaults to `v1` and yields the exact shared legacy effect without changing routing semantics. Evidence: `tests::native_registry_legacy_request_uses_shared_v1_effect_fixture` against `runtime-operation-legacy-version-v1.json` (**1 passed**). |
| FW-13-03A | DL-13 | NOT STARTED | Verify malformed operation payloads fail before effects. |
| FW-13-03B | DL-13 | PASS | Enabling `KOGWISTAR_RUNTIME_NATIVE_INLINE` with the in-memory graph-only service fails with `InvalidInput` before listener binding; disabled in-memory and enabled SQLite/PostgreSQL configurations retain their declared behavior. Evidence: `server_backend_tests::inline_native_runtime_rejects_backends_without_a_runtime_executor` (**1 passed**). |
| FW-13-04 | DL-13 | NOT STARTED | Verify native executor durable completion and final digest after SQLite service restart. |
| FW-13-04A | DL-13 | PASS | After the native PostgreSQL counter workflow reaches `succeeded`, reconstruct the run service against the same schema and verify the terminal result, persisted `counter=2` state, and exact unchanged `state_digest`. Evidence: `tests::postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available` (**1 passed**, live disposable pgvector test on Windows). This verifies service reconstruction/reopen, not an OS process crash. |
| FW-13-04B | DL-13 | NOT STARTED | Verify native executor error routing and recovery for one injected backend failure. |
| FW-13-02 | DL-13 | PARENT | Parent scope; children: FW-13-02-SQLITE,FW-13-02A,FW-13-02B |
| FW-13-02-SQLITE | DL-13 | PARENT | Parent scope; children: FW-13-02-SQLITE-NOOP,FW-13-02-SQLITE-COUNTER,FW-13-02-SQLITE-RESULT,FW-13-02-SQLITE-REQUEST,FW-13-02-SQLITE-RECEIVE,FW-13-02-SQLITE-ACK |
| FW-13-02-SQLITE-NOOP | DL-13 | PASS | The SQLite API route submits and claims a `rust-native/noop` step, executes it through the native endpoint, and durably records its unchanged app-owned state (`seed=1`) plus completion trace. Evidence: `tests::sqlite_native_noop_result_is_durable_through_runtime_route` (**1 passed**); exact operation-effect parity is covered by `tests::native_noop_emits_the_shared_worker_effect_and_static_successor` against `runtime-native-noop-v1.json` (**1 passed**). |
| FW-13-02-SQLITE-COUNTER | DL-13 | PASS | Runtime `native_executor_fails_closed_for_wrong_target_and_unregistered_operations` checks the shared `increment-counter` effect contract; API `sqlite_native_runtime_route_uses_shared_result_reducer` verifies its durable SQLite transition from counter 1 to 2 (**1 passed**). |
| FW-13-02-SQLITE-RESULT | DL-13 | PASS | API `sqlite_native_runtime_route_uses_shared_result_reducer` verifies the native `emit-result` transition is durably recorded and the SQLite run completes after its registered successor (**1 passed**). |
| FW-13-02-SQLITE-REQUEST | DL-13 | PASS | SQLite's native provider-event executor streams an OpenAI-compatible response, keeps text deltas transient, and durably stores semantic inbox events plus run traces under current owner authorization. Evidence: `tests::sqlite_native_provider_event_request_persists_semantic_events` (**1 passed**, focused Rust API test); the event envelope maps to Python `DurableProviderEvent` through the shared `agent-durable-provider-event-v1.json` fixture. This directly exercises the SQLite service executor, not HTTP ingress or the full workflow loop. |
| FW-13-02-SQLITE-RECEIVE | DL-13 | PASS | SQLite inline native workflow streams a provider response, receives an ACL-visible semantic event, and durably records its receive state. Evidence: `tests::sqlite_native_provider_event_request_persists_semantic_events` (**1 passed**); covers run admission through request and receive transitions. Acknowledgement and full loop parity remain open. |
| FW-13-02-SQLITE-ACK | DL-13 | PASS | Native SQLite inline run receives and acknowledges semantic provider events, then completes; lane statuses and handled traces match. Evidence: `tests::sqlite_native_provider_event_request_persists_semantic_events` (**1 passed**) plus `tests::sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` (**1 passed**, wrong-claimant denial and idempotent replay); checklist status mapping test (**1 passed**). |
| FW-13-02A | DL-13 | PARENT | Parent scope; children: FW-13-02A-NOOP,FW-13-02A-COUNTER,FW-13-02A-RESULT,FW-13-02A-REQUEST,FW-13-02A-RECEIVE,FW-13-02A-ACK |
| FW-13-02A-NOOP | DL-13 | PASS | Live `tests::postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available` durably records native `noop` with counter and seed unchanged, then completes the counter/result route and survives service reconstruction (**1 passed**, disposable pgvector container). |
| FW-13-02A-COUNTER | DL-13 | PASS | The PostgreSQL-backed native runtime route executes `increment-counter` from `counter=1`, persists the reducer transition to `counter=2`, and durably routes the run to its terminal successor. Evidence: `tests::postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available` (**1 passed**, ignored live-Postgres test explicitly enabled against the disposable pgvector test container on Windows). The matching SQLite counter transition is covered by `sqlite_native_runtime_route_uses_shared_result_reducer`. |
| FW-13-02A-RESULT | DL-13 | PASS | The PostgreSQL-backed native runtime route produces a durably inspectable terminal run with status `succeeded` and result payload `{operation: emit-result, schema_version: 1}` after the counter step. Evidence: `tests::postgres_native_runtime_route_uses_shared_result_reducer_when_dsn_available` (**1 passed**, ignored live-Postgres test explicitly enabled against the disposable pgvector test container on Windows). These two backend assertions close only their respective tasklets, not the parent DL-13 runtime scope. |
| FW-13-02A-REQUEST | DL-13 | PASS | Live `tests::postgres_provider_event_ack_persists_python_causal_links_atomically` streams a provider response, persists semantic inbox events, and retries without adding run events (**1 passed**, disposable pgvector container). The claim/receive context is exercised separately; this row asserts provider request publication and idempotency. |
| FW-13-02A-RECEIVE | DL-13 | PASS | Live `tests::postgres_provider_event_ack_persists_python_causal_links_atomically` invokes the native receive operation, verifies predicate routing to `ack-step`, and persists its returned state/result before acknowledgement (**1 passed**, disposable pgvector container). |
| FW-13-02A-ACK | DL-13 | PASS | PostgreSQL durably applies the native provider-event acknowledgement effect and completes the lane. Evidence: live `tests::postgres_provider_event_ack_persists_python_causal_links_atomically` (**1 passed**, disposable pgvector container); injected handled-event failure rolls back before valid retry. |
| FW-13-02B | DL-13 | PARENT | Parent scope; children: FW-13-02B-NOOP,FW-13-02B-COUNTER,FW-13-02B-RESULT,FW-13-02B-REQUEST,FW-13-02B-RECEIVE,FW-13-02B-ACK |
| FW-13-02B-NOOP | DL-13 | NOT STARTED | Verify native `noop` result and final state in-memory against the shared fixture. |
| FW-13-02B-COUNTER | DL-13 | NOT STARTED | Verify native `increment-counter` result and final state in-memory against the shared fixture. |
| FW-13-02B-RESULT | DL-13 | NOT STARTED | Verify native `emit-result` output and completion in-memory against its contract. |
| FW-13-02B-REQUEST | DL-13 | NOT STARTED | Verify native provider-event request behavior and result in-memory. |
| FW-13-02B-RECEIVE | DL-13 | NOT STARTED | Verify native provider-event receive routing and result in-memory. |
| FW-13-02B-ACK | DL-13 | NOT STARTED | Verify native provider-event acknowledgement effect and result in-memory. |
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
| FW-14C-03 | DL-14C | PASS | Defined strict `WorkerIdentityAttestationV1` with environment ID, artifact SHA-256, and dependency-lock SHA-256; payload is explicitly self-reported data, not trusted proof. Evidence: shared golden fixture plus Python contract test and Rust contract parser test pass. |
| FW-14C-03A | DL-14C | PASS | Python sync and async worker adapters reject missing, stale-artifact, lockfile-mismatched, and environment-mismatched required identities before callback execution; a matching registered identity executes and advertises its identity-specific claim recipient. Evidence: `tests/runtime/test_rust_runtime_python_worker.py::test_python_worker_identity_admission_rejects_before_callback`, `test_python_worker_identity_admission_accepts_registered_identity`, async rejection test, Rust step-payload forwarding test; focused worker suite passed. Attestation remains self-reported, not cryptographic proof. |
| FW-14C-03B | DL-14C | PASS | Rust rejects malformed authoritative graph identity pins before run creation or lane scheduling. Evidence: `sqlite_invalid_worker_identity_pin_is_rejected_before_scheduling` returns `409 KOGWISTAR_INVALID_WORKER_IDENTITY` and verifies the runtime queue remains empty; graph-plan and explicit-map validation tests also pass. |
| FW-14C-03C | DL-14C | PASS | Python and Rust accept the same valid attestation and reject all shared invalid cases (wrong version, noncanonical environment, malformed digest, unknown field). Evidence: `test_worker_identity_attestation_matches_shared_fixture`, Rust `worker_identity_attestation_matches_shared_python_golden_fixture`, and parity annotation verifier pass. |
| FW-14C-03D | DL-14C | PASS | A valid worker identity survives SQLite service reopen and still controls start-work routing and payload admission. Evidence: `sqlite_worker_identity_pin_survives_service_reopen_and_controls_claim` verifies the persisted identity map, identity-specific claim recipient, and `required_worker_identity` field after reopen. |
| FW-14C-03E | DL-14C | PASS | Identity-pinned claim recipients require exactly one matching `kogwistar_worker_identity` assertion in the cryptographically verified principal; missing, malformed, or mismatched registrations are rejected before queue claim handling in SQLite and PostgreSQL services. The trusted host JWT issuer must issue this claim only for a registered artifact/dependency-lock tuple. Evidence: SQLite signed-token registration/reopen regression; PostgreSQL rejects an unregistered principal before touching storage; Rust API suite **289 passed, 33 ignored**. This trusts issuer registration policy; it does not attest the live process image. |
| FW-15-01 | DL-15 | NOT STARTED | Reconcile selector names and precedence rules with the supported-mode manifest. |
| FW-15-01A | DL-15 | NOT STARTED | Reconcile legacy/default selector behavior with the supported-mode manifest. |
| FW-15-01B | DL-15 | NOT STARTED | Generate one fallback inventory and assert each entry has a compatibility/error class and owner evidence. |
| FW-15-03 | DL-15 | NOT STARTED | Verify selector behavior for one advertised backend and operation combination against the manifest. |
| FW-15-03A | DL-15 | PARENT | Parent scope; children: FW-15-03A-DETERMINISTIC-CONTRACTS,FW-15-03A-SQLITE-META,FW-15-03A-POSTGRES-EVENT-LOG,FW-15-03A-PROJECTIONS,FW-15-03A-QUEUES,FW-15-03A-GRAPH-PGVECTOR,FW-15-03A-CHROMA,FW-15-03A-WORKFLOW-RUNTIME,FW-15-03A-SERVER |
| FW-15-03A-DETERMINISTIC-CONTRACTS | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for deterministic contracts. |
| FW-15-03A-SQLITE-META | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for SQLite metadata. |
| FW-15-03A-POSTGRES-EVENT-LOG | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for PostgreSQL event storage. |
| FW-15-03A-PROJECTIONS | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for projections and snapshots. |
| FW-15-03A-QUEUES | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for queues, leases, and lanes. |
| FW-15-03A-GRAPH-PGVECTOR | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for graph and pgvector operations. |
| FW-15-03A-CHROMA | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for the Chroma adapter. |
| FW-15-03A-WORKFLOW-RUNTIME | DL-15 | NOT STARTED | Verify selector choices and explicit unsupported cells for workflow runtime operations. |
| FW-15-03A-SERVER | DL-15 | PARENT | Parent scope; children: FW-15-03A-SERVER-REST,FW-15-03A-SERVER-SSE,FW-15-03A-SERVER-MCP,FW-15-03A-SERVER-CLI |
| FW-15-03A-SERVER-REST | DL-15 | NOT STARTED | Verify Rust selector behavior and unsupported cells for REST. |
| FW-15-03A-SERVER-SSE | DL-15 | NOT STARTED | Verify Rust selector behavior and unsupported cells for SSE. |
| FW-15-03A-SERVER-MCP | DL-15 | NOT STARTED | Verify Rust selector behavior and unsupported cells for MCP. |
| FW-15-03A-SERVER-CLI | DL-15 | NOT STARTED | Verify Rust selector behavior and unsupported cells for CLI. |
| FW-15-03B | DL-15 | NOT STARTED | Add drift assertion that blocks a supported-mode manifest entry without a matching runtime selector. |
| FW-15-02 | DL-15 | NOT STARTED | Verify missing Rust extension fails before effects for one Rust-selected REST operation. |
| FW-15-02A | DL-15 | NOT STARTED | Verify unsupported backend fails before effects for one Rust-selected graph operation. |
| FW-15-02B | DL-15 | PARENT | Parent scope; children: FW-15-02B-DETERMINISTIC-CONTRACTS,FW-15-02B-SQLITE-META,FW-15-02B-POSTGRES-EVENT-LOG,FW-15-02B-QUEUES,FW-15-02B-GRAPH-PGVECTOR,FW-15-02B-WORKFLOW-RUNTIME,FW-15-02B-SERVER |
| FW-15-02B-DETERMINISTIC-CONTRACTS | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for deterministic contracts. |
| FW-15-02B-SQLITE-META | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for SQLite metadata. |
| FW-15-02B-POSTGRES-EVENT-LOG | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for PostgreSQL event storage. |
| FW-15-02B-QUEUES | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for queues, leases, and lanes. |
| FW-15-02B-GRAPH-PGVECTOR | DL-15 | PARENT | Parent scope; children: FW-15-02B-GRAPH,FW-15-02B-PGVECTOR |
| FW-15-02B-GRAPH | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for graph operations. |
| FW-15-02B-PGVECTOR | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for pgvector operations. |
| FW-15-02B-WORKFLOW-RUNTIME | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for workflow runtime operations. |
| FW-15-02B-SERVER | DL-15 | PARENT | Parent scope; children: FW-15-02B-SERVER-REST,FW-15-02B-SERVER-SSE,FW-15-02B-SERVER-MCP,FW-15-02B-SERVER-CLI |
| FW-15-02B-SERVER-REST | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for REST. |
| FW-15-02B-SERVER-SSE | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for SSE. |
| FW-15-02B-SERVER-MCP | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for MCP. |
| FW-15-02B-SERVER-CLI | DL-15 | NOT STARTED | Verify selected-Rust missing-extension and unsupported-backend failures for CLI. |
| FW-15C-01 | DL-15C | NOT STARTED | Verify transaction-owner guard on one workflow continuation write path. |
| FW-15C-01A | DL-15C | NOT STARTED | Verify transaction-owner guard on one callback state-changing path and reject cross-authority writes. |
| FW-15C-01B | DL-15C | NOT STARTED | Generate one guarded-write call-site inventory and assert every entry maps to a negative ownership test. |
| FW-15C-03 | DL-15C | NOT STARTED | Verify async adapter atomicity with transactional UoW. |
| FW-15C-03A | DL-15C | NOT STARTED | Verify async adapter without transactional UoW fails closed before partial effects. |
| FW-15C-04 | DL-15C | NOT STARTED | Verify nested framework invocation preserves selected transaction ownership. |
| FW-15C-04A | DL-15C | NOT STARTED | Verify framework-managed task/thread context propagation cannot bypass transaction owner guards. |
| FW-15C-04B | DL-15C | NOT STARTED | Verify retained framework handles cannot write after owner context ends. |
| FW-15C-02 | DL-15C | NOT STARTED | Verify async transactional backend commits or rolls back its declared atomic unit. |
| FW-15C-02A | DL-15C | PARENT | Parent scope; children: FW-15C-02A-REPORT,FW-15C-02A-REJECT |
| FW-15C-02A-REPORT | DL-15C | NOT STARTED | Verify an async non-transactional backend reports its weaker transaction semantics. |
| FW-15C-02A-REJECT | DL-15C | NOT STARTED | Verify an async non-transactional backend rejects one unsupported atomic operation. |
| FW-16-01 | DL-16 | PARENT | Parent scope; children: FW-16-01-NODE,FW-16-01-ACL-NODE,FW-16-01-CONVERSATION-NODE,FW-16-01-WORKFLOW-NODE,FW-16-01-RUN-NODE,FW-16-01-STEP-NODE,FW-16-01-CHECKPOINT-NODE,FW-16-01-COMPLETED-NODE,FW-16-01-CANCELLED-NODE,FW-16-01-FAILED-NODE,FW-16-01-LLM-NODE,FW-16-01-LLM-EXTRACTION-NODE,FW-16-01-FLATTENED-NODE,FW-16-01-ASSOC-FLATTENED-NODE |
| FW-16-01-NODE | DL-16 | NOT STARTED | Round-trip the base Node type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-ACL-NODE | DL-16 | NOT STARTED | Round-trip ACLNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-CONVERSATION-NODE | DL-16 | NOT STARTED | Round-trip ConversationNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-WORKFLOW-NODE | DL-16 | NOT STARTED | Round-trip WorkflowNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-RUN-NODE | DL-16 | NOT STARTED | Round-trip WorkflowRunNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-STEP-NODE | DL-16 | NOT STARTED | Round-trip WorkflowStepExecNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-CHECKPOINT-NODE | DL-16 | NOT STARTED | Round-trip WorkflowCheckpointNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-COMPLETED-NODE | DL-16 | NOT STARTED | Round-trip WorkflowCompletedNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-CANCELLED-NODE | DL-16 | NOT STARTED | Round-trip WorkflowCancelledNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-FAILED-NODE | DL-16 | NOT STARTED | Round-trip WorkflowFailedNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-LLM-NODE | DL-16 | NOT STARTED | Round-trip LLMNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-LLM-EXTRACTION-NODE | DL-16 | NOT STARTED | Round-trip LLMNodeExtraction type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-FLATTENED-NODE | DL-16 | NOT STARTED | Round-trip FlattenedLLMNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01-ASSOC-FLATTENED-NODE | DL-16 | NOT STARTED | Round-trip AssocFlattenedLLMNode type, wire tag, and extension fields across Python and Rust. |
| FW-16-01A | DL-16 | PARENT | Parent scope; children: FW-16-01A-EDGE,FW-16-01A-ACL-EDGE,FW-16-01A-CONVERSATION-EDGE,FW-16-01A-WORKFLOW-EDGE,FW-16-01A-WORKFLOW-RUNTIME-EDGE,FW-16-01A-LLM-EDGE,FW-16-01A-LLM-EXTRACTION-EDGE,FW-16-01A-FLATTENED-EDGE,FW-16-01A-ASSOC-FLATTENED-EDGE |
| FW-16-01A-EDGE | DL-16 | NOT STARTED | Round-trip base Edge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-ACL-EDGE | DL-16 | NOT STARTED | Round-trip ACLEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-CONVERSATION-EDGE | DL-16 | NOT STARTED | Round-trip ConversationEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-WORKFLOW-EDGE | DL-16 | NOT STARTED | Round-trip WorkflowEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-WORKFLOW-RUNTIME-EDGE | DL-16 | NOT STARTED | Round-trip WorkflowRuntimeEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-LLM-EDGE | DL-16 | NOT STARTED | Round-trip LLMEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-LLM-EXTRACTION-EDGE | DL-16 | NOT STARTED | Round-trip LLMEdgeExtraction type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-FLATTENED-EDGE | DL-16 | NOT STARTED | Round-trip FlattenedLLMEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01A-ASSOC-FLATTENED-EDGE | DL-16 | NOT STARTED | Round-trip AssocFlattenedLLMEdge type, endpoints, lifecycle, and extension fields across Python and Rust. |
| FW-16-01B | DL-16 | PARENT | Parent scope; children: FW-16-01B-NODE,FW-16-01B-EDGE |
| FW-16-01B-NODE | DL-16 | NOT STARTED | Reject malformed derived Node payloads before graph mutation in both implementations. |
| FW-16-01B-EDGE | DL-16 | NOT STARTED | Reject malformed derived Edge payloads before graph mutation in both implementations. |
| FW-16-03 | DL-16 | NOT STARTED | Generate one production Node-family inventory and assert each maps to a Python/Rust identity and field contract. |
| FW-16-03A | DL-16 | NOT STARTED | Generate one production Edge-family inventory and assert each maps to identity, endpoint, and lifecycle contracts. |
| FW-16-03B | DL-16 | NOT STARTED | Add one generated wire-contract assertion for explicit inclusion or exclusion of derived-family extension fields. |
| FW-16-04 | DL-16 | NOT STARTED | Verify shared canonical payload and digest parity for one derived family on the native path. |
| FW-16-04A | DL-16 | NOT STARTED | Verify recovery parity for one derived family on the native path. |
| FW-16-04B | DL-16 | NOT STARTED | Verify shared payload, digest, and invalid-input rejection for one derived family on the mixed path. |
| FW-16-02 | DL-16 | PARENT | Parent scope; children: FW-16-02-NODE,FW-16-02-ACL-NODE,FW-16-02-CONVERSATION-NODE,FW-16-02-WORKFLOW-NODE,FW-16-02-RUN-NODE,FW-16-02-STEP-NODE,FW-16-02-CHECKPOINT-NODE,FW-16-02-COMPLETED-NODE,FW-16-02-CANCELLED-NODE,FW-16-02-FAILED-NODE,FW-16-02-LLM-NODE,FW-16-02-LLM-EXTRACTION-NODE,FW-16-02-FLATTENED-NODE,FW-16-02-ASSOC-FLATTENED-NODE |
| FW-16-02-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for the base Node family on advertised paths. |
| FW-16-02-ACL-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for ACLNode on advertised paths. |
| FW-16-02-CONVERSATION-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for ConversationNode on advertised paths. |
| FW-16-02-WORKFLOW-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowNode on advertised paths. |
| FW-16-02-RUN-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowRunNode on advertised paths. |
| FW-16-02-STEP-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowStepExecNode on advertised paths. |
| FW-16-02-CHECKPOINT-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowCheckpointNode on advertised paths. |
| FW-16-02-COMPLETED-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowCompletedNode on advertised paths. |
| FW-16-02-CANCELLED-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowCancelledNode on advertised paths. |
| FW-16-02-FAILED-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowFailedNode on advertised paths. |
| FW-16-02-LLM-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for LLMNode on advertised paths. |
| FW-16-02-LLM-EXTRACTION-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for LLMNodeExtraction on advertised paths. |
| FW-16-02-FLATTENED-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for FlattenedLLMNode on advertised paths. |
| FW-16-02-ASSOC-FLATTENED-NODE | DL-16 | NOT STARTED | Verify shared payload and digest parity for AssocFlattenedLLMNode on advertised paths. |
| FW-16-02A | DL-16 | PARENT | Parent scope; children: FW-16-02A-EDGE,FW-16-02A-ACL-EDGE,FW-16-02A-CONVERSATION-EDGE,FW-16-02A-WORKFLOW-EDGE,FW-16-02A-WORKFLOW-RUNTIME-EDGE,FW-16-02A-LLM-EDGE,FW-16-02A-LLM-EXTRACTION-EDGE,FW-16-02A-FLATTENED-EDGE,FW-16-02A-ASSOC-FLATTENED-EDGE |
| FW-16-02A-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for the base Edge family on advertised paths. |
| FW-16-02A-ACL-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for ACLEdge on advertised paths. |
| FW-16-02A-CONVERSATION-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for ConversationEdge on advertised paths. |
| FW-16-02A-WORKFLOW-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowEdge on advertised paths. |
| FW-16-02A-WORKFLOW-RUNTIME-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for WorkflowRuntimeEdge on advertised paths. |
| FW-16-02A-LLM-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for LLMEdge on advertised paths. |
| FW-16-02A-LLM-EXTRACTION-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for LLMEdgeExtraction on advertised paths. |
| FW-16-02A-FLATTENED-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for FlattenedLLMEdge on advertised paths. |
| FW-16-02A-ASSOC-FLATTENED-EDGE | DL-16 | NOT STARTED | Verify shared payload and digest parity for AssocFlattenedLLMEdge on advertised paths. |
| FW-16-02B | DL-16 | PARENT | Parent scope; children: FW-16-02B-NODE-RECOVERY,FW-16-02B-EDGE-RECOVERY,FW-16-02B-NODE-INVALID,FW-16-02B-EDGE-INVALID |
| FW-16-02B-NODE-RECOVERY | DL-16 | NOT STARTED | Verify Node-family restart and recovery preserves the canonical payload and digest. |
| FW-16-02B-EDGE-RECOVERY | DL-16 | NOT STARTED | Verify Edge-family restart and recovery preserves endpoints, payload, and digest. |
| FW-16-02B-NODE-INVALID | DL-16 | NOT STARTED | Verify malformed derived Node payloads are rejected before graph mutation. |
| FW-16-02B-EDGE-INVALID | DL-16 | NOT STARTED | Verify malformed derived Edge payloads are rejected before graph mutation. |
| FW-16AC-01 | DL-16AC | PARENT | Parent scope; children: FW-16AC-01-PUBLIC,FW-16AC-01-OWNER,FW-16AC-01-PRIVATE,FW-16AC-01-SCOPE,FW-16AC-01-SHARED |
| FW-16AC-01-PUBLIC | DL-16AC | NOT STARTED | Verify node public-visibility ACL outcome against the Python fixture. |
| FW-16AC-01-OWNER | DL-16AC | NOT STARTED | Verify node owner-visibility ACL outcome against the Python fixture. |
| FW-16AC-01-PRIVATE | DL-16AC | NOT STARTED | Verify node private-visibility ACL outcome against the Python fixture. |
| FW-16AC-01-SCOPE | DL-16AC | NOT STARTED | Verify node namespace/scope ACL outcome against the Python fixture. |
| FW-16AC-01-SHARED | DL-16AC | NOT STARTED | Verify node shared/group-visibility ACL outcome against the Python fixture. |
| FW-16AC-01A | DL-16AC | PARENT | Parent scope; children: FW-16AC-01A-PUBLIC,FW-16AC-01A-OWNER,FW-16AC-01A-PRIVATE,FW-16AC-01A-SCOPE,FW-16AC-01A-SHARED |
| FW-16AC-01A-PUBLIC | DL-16AC | NOT STARTED | Verify edge public-visibility ACL outcome against the Python fixture. |
| FW-16AC-01A-OWNER | DL-16AC | NOT STARTED | Verify edge owner-visibility ACL outcome against the Python fixture. |
| FW-16AC-01A-PRIVATE | DL-16AC | NOT STARTED | Verify edge private-visibility ACL outcome against the Python fixture. |
| FW-16AC-01A-SCOPE | DL-16AC | NOT STARTED | Verify edge namespace/scope ACL outcome against the Python fixture. |
| FW-16AC-01A-SHARED | DL-16AC | NOT STARTED | Verify edge shared/group-visibility ACL outcome against the Python fixture. |
| FW-16AC-01B | DL-16AC | PARENT | Parent scope; children: FW-16AC-01B-NODE-LATEST,FW-16AC-01B-EDGE-LATEST,FW-16AC-01B-NODE-MISSING,FW-16AC-01B-EDGE-MISSING |
| FW-16AC-01B-NODE-LATEST | DL-16AC | NOT STARTED | Verify latest-version selection for node reads against Python. |
| FW-16AC-01B-EDGE-LATEST | DL-16AC | NOT STARTED | Verify latest-version selection for edge reads against Python. |
| FW-16AC-01B-NODE-MISSING | DL-16AC | NOT STARTED | Verify no-record denial for node reads against Python. |
| FW-16AC-01B-EDGE-MISSING | DL-16AC | NOT STARTED | Verify no-record denial for edge reads against Python. |
| FW-16AC-01C | DL-16AC | PARENT | Parent scope; children: FW-16AC-01C-NODE-TOMBSTONE,FW-16AC-01C-NODE-REDIRECT,FW-16AC-01C-EDGE-TOMBSTONE,FW-16AC-01C-EDGE-REDIRECT |
| FW-16AC-01C-NODE-TOMBSTONE | DL-16AC | NOT STARTED | Verify tombstoned nodes are filtered from reads against Python. |
| FW-16AC-01C-NODE-REDIRECT | DL-16AC | NOT STARTED | Verify node redirects resolve consistently on reads against Python. |
| FW-16AC-01C-EDGE-TOMBSTONE | DL-16AC | NOT STARTED | Verify tombstoned edges are filtered from reads against Python. |
| FW-16AC-01C-EDGE-REDIRECT | DL-16AC | NOT STARTED | Verify edge redirects resolve consistently on reads against Python. |
| FW-16AC-03 | DL-16AC | PARENT | Parent scope; children: FW-16AC-03-KNOWLEDGE,FW-16AC-03-CONVERSATION,FW-16AC-03-WORKFLOW,FW-16AC-03-WISDOM |
| FW-16AC-03-KNOWLEDGE | DL-16AC | PARENT | Parent scope; children: FW-16AC-03-KNOWLEDGE-NODE-ACL,FW-16AC-03-KNOWLEDGE-EDGE-ID |
| FW-16AC-03-KNOWLEDGE-NODE-ACL | DL-16AC | NOT STARTED | Verify node ACL decisions in the knowledge graph. |
| FW-16AC-03-KNOWLEDGE-EDGE-ID | DL-16AC | NOT STARTED | Verify same-string node and edge IDs remain isolated in the knowledge graph. |
| FW-16AC-03-CONVERSATION | DL-16AC | PARENT | Parent scope; children: FW-16AC-03-CONVERSATION-NODE-ACL,FW-16AC-03-CONVERSATION-EDGE-ID |
| FW-16AC-03-CONVERSATION-NODE-ACL | DL-16AC | NOT STARTED | Verify node ACL decisions in the conversation graph. |
| FW-16AC-03-CONVERSATION-EDGE-ID | DL-16AC | NOT STARTED | Verify same-string node and edge IDs remain isolated in the conversation graph. |
| FW-16AC-03-WORKFLOW | DL-16AC | PARENT | Parent scope; children: FW-16AC-03-WORKFLOW-NODE-ACL,FW-16AC-03-WORKFLOW-EDGE-ID |
| FW-16AC-03-WORKFLOW-NODE-ACL | DL-16AC | NOT STARTED | Verify node ACL decisions in the workflow graph. |
| FW-16AC-03-WORKFLOW-EDGE-ID | DL-16AC | NOT STARTED | Verify same-string node and edge IDs remain isolated in the workflow graph. |
| FW-16AC-03-WISDOM | DL-16AC | PARENT | Parent scope; children: FW-16AC-03-WISDOM-NODE-ACL,FW-16AC-03-WISDOM-EDGE-ID |
| FW-16AC-03-WISDOM-NODE-ACL | DL-16AC | NOT STARTED | Verify node ACL decisions in the wisdom graph. |
| FW-16AC-03-WISDOM-EDGE-ID | DL-16AC | NOT STARTED | Verify same-string node and edge IDs remain isolated in the wisdom graph. |
| FW-16AC-03A | DL-16AC | PARENT | Parent scope; children: FW-16AC-03A-KNOWLEDGE,FW-16AC-03A-CONVERSATION,FW-16AC-03A-WORKFLOW,FW-16AC-03A-WISDOM |
| FW-16AC-03A-KNOWLEDGE | DL-16AC | PARENT | Parent scope; children: FW-16AC-03A-KNOWLEDGE-EDGE-ACL,FW-16AC-03A-KNOWLEDGE-NODE-ID |
| FW-16AC-03A-KNOWLEDGE-EDGE-ACL | DL-16AC | NOT STARTED | Verify edge ACL decisions in the knowledge graph. |
| FW-16AC-03A-KNOWLEDGE-NODE-ID | DL-16AC | NOT STARTED | Verify same-string edge and node IDs remain isolated in the knowledge graph. |
| FW-16AC-03A-CONVERSATION | DL-16AC | PARENT | Parent scope; children: FW-16AC-03A-CONVERSATION-EDGE-ACL,FW-16AC-03A-CONVERSATION-NODE-ID |
| FW-16AC-03A-CONVERSATION-EDGE-ACL | DL-16AC | NOT STARTED | Verify edge ACL decisions in the conversation graph. |
| FW-16AC-03A-CONVERSATION-NODE-ID | DL-16AC | NOT STARTED | Verify same-string edge and node IDs remain isolated in the conversation graph. |
| FW-16AC-03A-WORKFLOW | DL-16AC | PARENT | Parent scope; children: FW-16AC-03A-WORKFLOW-EDGE-ACL,FW-16AC-03A-WORKFLOW-NODE-ID |
| FW-16AC-03A-WORKFLOW-EDGE-ACL | DL-16AC | NOT STARTED | Verify edge ACL decisions in the workflow graph. |
| FW-16AC-03A-WORKFLOW-NODE-ID | DL-16AC | NOT STARTED | Verify same-string edge and node IDs remain isolated in the workflow graph. |
| FW-16AC-03A-WISDOM | DL-16AC | PARENT | Parent scope; children: FW-16AC-03A-WISDOM-EDGE-ACL,FW-16AC-03A-WISDOM-NODE-ID |
| FW-16AC-03A-WISDOM-EDGE-ACL | DL-16AC | NOT STARTED | Verify edge ACL decisions in the wisdom graph. |
| FW-16AC-03A-WISDOM-NODE-ID | DL-16AC | NOT STARTED | Verify same-string edge and node IDs remain isolated in the wisdom graph. |
| FW-16AC-03B | DL-16AC | PARENT | Parent scope; children: FW-16AC-03B-SHARED,FW-16AC-03B-LATEST |
| FW-16AC-03B-SHARED | DL-16AC | NOT STARTED | Verify shared/group ACL decisions remain scoped to the selected graph role. |
| FW-16AC-03B-LATEST | DL-16AC | NOT STARTED | Verify latest-version ACL decisions remain scoped to the selected graph role. |
| FW-16AC-04 | DL-16AC | NOT STARTED | Verify source-evidence authorization on graph reads. Lifecycle filtering and redirect behavior are tracked by FW-16AC-01C. |
| FW-16AC-04A | DL-16AC | NOT STARTED | Verify edge endpoint dependency filtering for authorized and denied endpoint cases. |
| FW-16AC-04B | DL-16AC | NOT STARTED | Verify stale Chroma projection is reconciled after backend restart and graph mutation. |
| FW-16AC-02 | DL-16AC | NOT STARTED | Verify mutation and graph-read ACL parity for same-string node/edge identifiers. |
| FW-16AC-02A | DL-16AC | NOT STARTED | Verify stale Chroma projection recovery after service restart using a fake endpoint. |
| FW-16AC-02B | DL-16AC | NOT STARTED | Verify stale Chroma projection recovery after service restart against live Chroma and label evidence type. |
| FW-16AM-01 | DL-16AM | NOT STARTED | Inventory public MCP operation names, schemas, and discovery metadata against Python. |
| FW-16AM-01A | DL-16AM | NOT STARTED | Verify paired request/response and stable error semantics for one advertised MCP operation family. |
| FW-16AM-01B | DL-16AM | PARENT | Parent scope; children: FW-16AM-01B-ROOT,FW-16AM-01B-CONVERSATION,FW-16AM-01B-WORKFLOW |
| FW-16AM-01B-ROOT | DL-16AM | NOT STARTED | Verify root MCP authorization rejects a denied operation before effects. |
| FW-16AM-01B-CONVERSATION | DL-16AM | NOT STARTED | Verify conversation MCP authorization rejects a denied operation before effects. |
| FW-16AM-01B-WORKFLOW | DL-16AM | NOT STARTED | Verify workflow MCP authorization rejects a denied operation before effects. |
| FW-16AM-03 | DL-16AM | NOT STARTED | Verify supported server-initiated event type and sequence ordering against Python. |
| FW-16AM-03A | DL-16AM | NOT STARTED | Verify event replay/resume behavior and explicit unsupported status for unimplemented event types. |
| FW-16AM-03B | DL-16AM | NOT STARTED | Verify transport negotiation for the declared MCP event transport. |
| FW-16AM-04 | DL-16AM | NOT STARTED | Verify public ingestion operation request/response and capability advertisement. |
| FW-16AM-04A | DL-16AM | NOT STARTED | Verify public parsing operation behavior and capability advertisement. |
| FW-16AM-04B | DL-16AM | NOT STARTED | Verify public indexing/backend operation behavior and capability advertisement. |
| FW-16AM-02 | DL-16AM | NOT STARTED | Verify one remaining server-initiated event family and its production transport behavior. |
| FW-16AM-02A | DL-16AM | PARENT | Parent scope; children: FW-16AM-02A-INGEST,FW-16AM-02A-PARSE,FW-16AM-02A-INDEX |
| FW-16AM-02A-INGEST | DL-16AM | NOT STARTED | Verify the ingestion MCP boundary matches the declared support matrix. |
| FW-16AM-02A-PARSE | DL-16AM | NOT STARTED | Verify the parsing MCP boundary matches the declared support matrix. |
| FW-16AM-02A-INDEX | DL-16AM | NOT STARTED | Verify the indexing MCP boundary matches the declared support matrix. |
| FW-16AM-02B | DL-16AM | NOT STARTED | Mark unexercised backend/transport combinations explicitly unsupported in discovery. |
| FW-16BQ-01 | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-01-SYSCALL,FW-16BQ-01-REST,FW-16BQ-01-MCP |
| FW-16BQ-01-SYSCALL | DL-16BQ | NOT STARTED | Inventory advertised syscall operations against Python contracts and backend support. |
| FW-16BQ-01-REST | DL-16BQ | NOT STARTED | Inventory advertised REST operations against Python contracts and backend support. |
| FW-16BQ-01-MCP | DL-16BQ | NOT STARTED | Inventory advertised MCP operations against Python contracts and backend support. |
| FW-16BQ-01A | DL-16BQ | NOT STARTED | Verify syscall request/response parity for SQLite+Chroma on one operation family. |
| FW-16BQ-01B | DL-16BQ | NOT STARTED | Verify syscall request/response parity for PostgreSQL on one operation family. |
| FW-16BQ-01C | DL-16BQ | NOT STARTED | Verify syscall request/response parity for in-memory on one operation family. |
| FW-16BQ-03 | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-03-AUTH,FW-16BQ-03-NAMESPACE,FW-16BQ-03-SCOPE |
| FW-16BQ-03-AUTH | DL-16BQ | NOT STARTED | Verify syscall authorization outcome against Python for SQLite+Chroma. |
| FW-16BQ-03-NAMESPACE | DL-16BQ | NOT STARTED | Verify syscall namespace isolation against Python for SQLite+Chroma. |
| FW-16BQ-03-SCOPE | DL-16BQ | NOT STARTED | Verify syscall scope enforcement against Python for SQLite+Chroma. |
| FW-16BQ-03A | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-03A-AUTH,FW-16BQ-03A-NAMESPACE,FW-16BQ-03A-SCOPE |
| FW-16BQ-03A-AUTH | DL-16BQ | NOT STARTED | Verify syscall authorization outcome against Python for PostgreSQL. |
| FW-16BQ-03A-NAMESPACE | DL-16BQ | NOT STARTED | Verify syscall namespace isolation against Python for PostgreSQL. |
| FW-16BQ-03A-SCOPE | DL-16BQ | NOT STARTED | Verify syscall scope enforcement against Python for PostgreSQL. |
| FW-16BQ-03B | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-03B-AUTH,FW-16BQ-03B-NAMESPACE,FW-16BQ-03B-SCOPE |
| FW-16BQ-03B-AUTH | DL-16BQ | NOT STARTED | Verify syscall authorization outcome against Python for in-memory. |
| FW-16BQ-03B-NAMESPACE | DL-16BQ | NOT STARTED | Verify syscall namespace isolation against Python for in-memory. |
| FW-16BQ-03B-SCOPE | DL-16BQ | NOT STARTED | Verify syscall scope enforcement against Python for in-memory. |
| FW-16BQ-03C | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-03C-PG,FW-16BQ-03C-SQLITE-CHROMA,FW-16BQ-03C-MEMORY |
| FW-16BQ-03C-PG | DL-16BQ | NOT STARTED | Compare one shared syscall error envelope on PostgreSQL against the Python contract. |
| FW-16BQ-03C-SQLITE-CHROMA | DL-16BQ | NOT STARTED | Compare one shared syscall error envelope on SQLite+Chroma against the Python contract. |
| FW-16BQ-03C-MEMORY | DL-16BQ | NOT STARTED | Compare one shared syscall error envelope in-memory against the Python contract. |
| FW-16BQ-04 | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-04A,FW-16BQ-04B |
| FW-16BQ-04A | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-04A-HANDLE,FW-16BQ-04A-ACK,FW-16BQ-04A-TRACE,FW-16BQ-04A-REPLAY |
| FW-16BQ-04A-HANDLE | DL-16BQ | PASS | `tests::sqlite_native_provider_event_request_persists_semantic_events` runs the SQLite+Chroma provider-event workflow through receive, acknowledgement, and success; all execution lanes are completed without `python-worker` (**1 passed**). |
| FW-16BQ-04A-ACK | DL-16BQ | PASS | `sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` rejects the wrong claimant without consuming the message and acknowledges only the matching active claim (**1 passed**). |
| FW-16BQ-04A-TRACE | DL-16BQ | PASS | `sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` verifies the handled event is linked to its workflow step in the conversation graph and records the handled trace (**1 passed**). |
| FW-16BQ-04A-REPLAY | DL-16BQ | PASS | `sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` retries the exact committed effect and asserts unchanged graph-event and handled-trace counts (**1 passed**). |
| FW-16BQ-04B | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-04B-PUBLISH,FW-16BQ-04B-HANDLE,FW-16BQ-04B-ACK,FW-16BQ-04B-REPLAY |
| FW-16BQ-04B-PUBLISH | DL-16BQ | PASS | Live `postgres_agentic_answer_workflow_executes_durably_without_python_worker` verifies the streamed provider event, matching conversation-graph records/ACL, and PostgreSQL schema placement (**1 passed**). |
| FW-16BQ-04B-HANDLE | DL-16BQ | PASS | Live `postgres_agentic_answer_workflow_executes_durably_without_python_worker` completes a structured-answer turn through the Rust-native executor and verifies no lane message is assigned to `python-worker` (**1 passed**). |
| FW-16BQ-04B-ACK | DL-16BQ | PASS | Live `postgres_provider_event_ack_persists_python_causal_links_atomically` rejects a mismatched claimant without consuming the message or advancing runtime state, then accepts the active claim (**1 passed**, disposable pgvector container). |
| FW-16BQ-04B-REPLAY | DL-16BQ | PASS | Live `postgres_provider_event_ack_persists_python_causal_links_atomically` retries the exact committed acknowledgement and verifies the durable run-event list is unchanged (**1 passed**, disposable pgvector container). |
| FW-16BQ-05 | DL-16BQ | NOT STARTED | Verify SQLite+Chroma restart/retry and canonical digest for one advertised syscall path. |
| FW-16BQ-05A | DL-16BQ | NOT STARTED | Verify PostgreSQL restart/retry and canonical digest for one advertised syscall path. |
| FW-16BQ-05B | DL-16BQ | NOT STARTED | Verify supported interpreter/runtime cells and list untested/unsupported combinations. |
| FW-16BQ-02 | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-02-PG,FW-16BQ-02-SQLITE-CHROMA,FW-16BQ-02-MEMORY |
| FW-16BQ-02-PG | DL-16BQ | NOT STARTED | Verify PostgreSQL ACL parity for the frozen shared operation fixture. |
| FW-16BQ-02-SQLITE-CHROMA | DL-16BQ | NOT STARTED | Verify SQLite+Chroma ACL parity for the frozen shared operation fixture. |
| FW-16BQ-02-MEMORY | DL-16BQ | NOT STARTED | Verify in-memory ACL parity for the frozen shared operation fixture. |
| FW-16BQ-02A | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-02A-PG,FW-16BQ-02A-SQLITE-CHROMA,FW-16BQ-02A-MEMORY |
| FW-16BQ-02A-PG | DL-16BQ | NOT STARTED | Verify PostgreSQL event/worker state parity for the frozen shared operation fixture. |
| FW-16BQ-02A-SQLITE-CHROMA | DL-16BQ | NOT STARTED | Verify SQLite+Chroma event/worker state parity for the frozen shared operation fixture. |
| FW-16BQ-02A-MEMORY | DL-16BQ | NOT STARTED | Verify in-memory event/worker state parity for the frozen shared operation fixture. |
| FW-16BQ-02B | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-02B-PG,FW-16BQ-02B-SQLITE-CHROMA,FW-16BQ-02B-MEMORY |
| FW-16BQ-02B-PG | DL-16BQ | NOT STARTED | Verify PostgreSQL final graph digest for the frozen operation trace. |
| FW-16BQ-02B-SQLITE-CHROMA | DL-16BQ | NOT STARTED | Verify SQLite+Chroma final graph digest for the frozen operation trace. |
| FW-16BQ-02B-MEMORY | DL-16BQ | NOT STARTED | Verify in-memory final graph digest for the frozen operation trace. |
| FW-17-01 | DL-17 | NOT STARTED | Record the required CPython 3.12 Python-authority CI result on Ubuntu. |
| FW-17-01A | DL-17 | PARENT | Parent scope; children: FW-17-01A-CP313,FW-17-01A-CP314 |
| FW-17-01A-CP313 | DL-17 | NOT STARTED | Run required CPython 3.13 Python-authority CI cells and record outcomes. |
| FW-17-01A-CP314 | DL-17 | NOT STARTED | Run required CPython 3.14 Python-authority CI cells and record outcomes. |
| FW-17-01B | DL-17 | NOT STARTED | Run and record required PyPy 3.11 cells; retain best-effort status where a cell is not required. |
| FW-17-01C | DL-17 | PARENT | Parent scope; children: FW-17-01C-LINUX,FW-17-01C-WINDOWS,FW-17-01C-MACOS |
| FW-17-01C-LINUX | DL-17 | NOT STARTED | Verify required Rust toolchain and CI result on Linux; label optional jobs. |
| FW-17-01C-WINDOWS | DL-17 | NOT STARTED | Verify required Rust toolchain and CI result on Windows; label optional jobs. |
| FW-17-01C-MACOS | DL-17 | NOT STARTED | Verify required Rust toolchain and CI result on macOS; label optional jobs. |
| FW-17-03 | DL-17 | PARENT | Parent scope; children: FW-17-03-CP312,FW-17-03-CP313,FW-17-03-CP314 |
| FW-17-03-CP312 | DL-17 | NOT STARTED | Verify published wheel metadata and dependency installation on CPython 3.12. |
| FW-17-03-CP313 | DL-17 | NOT STARTED | Verify published wheel metadata and dependency installation on CPython 3.13. |
| FW-17-03-CP314 | DL-17 | NOT STARTED | Verify published wheel metadata and dependency installation on CPython 3.14. |
| FW-17-03A | DL-17 | PARENT | Parent scope; children: FW-17-03A-312-LINUX,FW-17-03A-312-WINDOWS,FW-17-03A-312-MACOS,FW-17-03A-313-LINUX,FW-17-03A-313-WINDOWS,FW-17-03A-313-MACOS,FW-17-03A-314-LINUX,FW-17-03A-314-WINDOWS,FW-17-03A-314-MACOS |
| FW-17-03A-312-LINUX | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.12 Linux. |
| FW-17-03A-312-WINDOWS | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.12 Windows. |
| FW-17-03A-312-MACOS | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.12 macOS. |
| FW-17-03A-313-LINUX | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.13 Linux. |
| FW-17-03A-313-WINDOWS | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.13 Windows. |
| FW-17-03A-313-MACOS | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.13 macOS. |
| FW-17-03A-314-LINUX | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.14 Linux. |
| FW-17-03A-314-WINDOWS | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.14 Windows. |
| FW-17-03A-314-MACOS | DL-17 | NOT STARTED | Verify native extension import and smoke behavior on CPython 3.14 macOS. |
| FW-17-03B | DL-17 | NOT STARTED | Verify package installation/import behavior on the required PyPy 3.11 cell. |
| FW-17-04 | DL-17 | NOT STARTED | Run downstream pinned-consumer feature tests for the Python implementation path. |
| FW-17-04A | DL-17 | PARENT | Parent scope; children: FW-17-04A-MIXED,FW-17-04A-RUST |
| FW-17-04A-MIXED | DL-17 | NOT STARTED | Run pinned-consumer regressions for the mixed Python/Rust implementation path. |
| FW-17-04A-RUST | DL-17 | NOT STARTED | Run pinned-consumer regressions for the Rust-native implementation path. |
| FW-17-04B | DL-17 | PARENT | Parent scope; children: FW-17-04B-CI,FW-17-04B-PACKAGE,FW-17-04B-DOCS |
| FW-17-04B-CI | DL-17 | NOT STARTED | Reconcile downstream CI results with the declared supported implementation paths. |
| FW-17-04B-PACKAGE | DL-17 | NOT STARTED | Reconcile published package metadata with the verified downstream dependency versions. |
| FW-17-04B-DOCS | DL-17 | NOT STARTED | Reconcile support documentation with verified downstream compatibility results. |
| FW-17-02 | DL-17 | NOT STARTED | Run pinned-consumer feature tests against the declared package version. |
| FW-17-02A | DL-17 | PARENT | Parent scope; children: FW-17-02A-PYTHON,FW-17-02A-MIXED,FW-17-02A-RUST |
| FW-17-02A-PYTHON | DL-17 | NOT STARTED | Run pinned-consumer regression for the Python implementation path. |
| FW-17-02A-MIXED | DL-17 | NOT STARTED | Run pinned-consumer regression for the mixed Python/Rust implementation path. |
| FW-17-02A-RUST | DL-17 | NOT STARTED | Run pinned-consumer regression for the Rust-native implementation path. |
| FW-17-02B | DL-17 | PARENT | Parent scope; children: FW-17-02B-METADATA,FW-17-02B-DOCS |
| FW-17-02B-METADATA | DL-17 | NOT STARTED | Verify published package metadata matches the tested consumer cells. |
| FW-17-02B-DOCS | DL-17 | NOT STARTED | Verify documented support matrix matches the tested consumer cells. |
| FW-18-01 | DL-18 | NOT STARTED | Freeze one deterministic balanced mixed read/write trace and expected read visibility. |
| FW-18-01A | DL-18 | NOT STARTED | Freeze one deterministic read-heavy trace and expected read visibility. |
| FW-18-01B | DL-18 | PARENT | Parent scope; children: FW-18-01B-PYTHON,FW-18-01B-MIXED,FW-18-01B-RUST |
| FW-18-01B-PYTHON | DL-18 | NOT STARTED | Verify the Python implementation's final digest for the balanced trace against its frozen expected digest. |
| FW-18-01B-MIXED | DL-18 | NOT STARTED | Verify the mixed Python/Rust implementation's final digest for the balanced trace against its frozen expected digest. |
| FW-18-01B-RUST | DL-18 | NOT STARTED | Verify the Rust-native implementation's final digest for the balanced trace against its frozen expected digest. |
| FW-18-03 | DL-18 | NOT STARTED | Verify seeded workload and expected per-read visibility at the small fixture size. |
| FW-18-03A | DL-18 | NOT STARTED | Verify seeded workload and expected per-read visibility at the larger fixture size. |
| FW-18-03B | DL-18 | PARENT | Parent scope; children: FW-18-03B-PY-SMALL,FW-18-03B-RUST-SMALL,FW-18-03B-PY-LARGE,FW-18-03B-RUST-LARGE |
| FW-18-03B-PY-SMALL | DL-18 | NOT STARTED | Verify the Python trace consumer matches the expected final digest for the small fixture. |
| FW-18-03B-RUST-SMALL | DL-18 | NOT STARTED | Verify the Rust trace consumer matches the expected final digest for the small fixture. |
| FW-18-03B-PY-LARGE | DL-18 | NOT STARTED | Verify the Python trace consumer matches the expected final digest for the large fixture. |
| FW-18-03B-RUST-LARGE | DL-18 | NOT STARTED | Verify the Rust trace consumer matches the expected final digest for the large fixture. |
| FW-18-04 | DL-18 | PARENT | Parent scope; children: FW-18-04-SQLITE-COMMIT,FW-18-04-SQLITE-DUPLICATE |
| FW-18-04-SQLITE-COMMIT | DL-18 | NOT STARTED | Verify one individual write commit boundary on SQLite. |
| FW-18-04-SQLITE-DUPLICATE | DL-18 | NOT STARTED | Verify duplicate-write semantics on SQLite. |
| FW-18-04A | DL-18 | PARENT | Parent scope; children: FW-18-04A-PG-COMMIT,FW-18-04A-PG-DUPLICATE,FW-18-04A-PG-REPLAY |
| FW-18-04A-PG-COMMIT | DL-18 | NOT STARTED | Verify one write commit boundary on PostgreSQL. |
| FW-18-04A-PG-DUPLICATE | DL-18 | NOT STARTED | Verify duplicate-write semantics on PostgreSQL. |
| FW-18-04A-PG-REPLAY | DL-18 | NOT STARTED | Verify replay semantics on PostgreSQL. |
| FW-18-04B | DL-18 | NOT STARTED | Verify supported Chroma projection semantics without claiming vector-store transactionality. |
| FW-18-02 | DL-18 | NOT STARTED | Verify batch atomicity on a backend that declares transaction support. |
| FW-18-02A | DL-18 | NOT STARTED | Verify weaker/non-transactional backend fails or reports partial outcomes according to its contract. |
| FW-19-01 | DL-19 | NOT STARTED | Run fixed-resource PostgreSQL graph operation benchmark at the larger fixture size. |
| FW-19-01A | DL-19 | NOT STARTED | Run fixed-resource indexing benchmark at the larger fixture size. |
| FW-19-01B | DL-19 | PARENT | Parent scope; children: FW-19-01B-ACL,FW-19-01B-AGENT |
| FW-19-01B-ACL | DL-19 | NOT STARTED | Run the fixed-resource ACL profile at the larger fixture size. |
| FW-19-01B-AGENT | DL-19 | NOT STARTED | Run the fixed-resource agent-request profile at the larger fixture size. |
| FW-19-03 | DL-19 | NOT STARTED | Run PostgreSQL profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-03A | DL-19 | NOT STARTED | Run indexing profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-03B | DL-19 | NOT STARTED | Run ACL profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-03C | DL-19 | NOT STARTED | Run agent-request profile with fixed CPU/RAM, warm-up, and fresh data per sample. |
| FW-19-04 | DL-19 | PARENT | Parent scope; children: FW-19-04-CP312,FW-19-04-CP313,FW-19-04-CP314,FW-19-04-RUST |
| FW-19-04-CP312 | DL-19 | NOT STARTED | Verify canonical digest parity for the supported CPython 3.12 runtime cell. |
| FW-19-04-CP313 | DL-19 | NOT STARTED | Verify canonical digest parity for the supported CPython 3.13 runtime cell. |
| FW-19-04-CP314 | DL-19 | NOT STARTED | Verify canonical digest parity for the supported CPython 3.14 runtime cell. |
| FW-19-04-RUST | DL-19 | NOT STARTED | Verify canonical digest parity for the Rust-native runtime cell. |
| FW-19-04A | DL-19 | NOT STARTED | Add replay/recovery benchmark runs and record state digest and resource limits. |
| FW-19-04B | DL-19 | NOT STARTED | Report medians, tail latency, and sample variation for the added runtime/recovery cells. |
| FW-19-02 | DL-19 | NOT STARTED | Add one supported-runtime replay/recovery cell and verify digest parity. |
| FW-19-02A | DL-19 | NOT STARTED | Report median and tail latency for replay/recovery samples. |
| FW-20-01 | DL-20 | PARENT | Parent scope; children: FW-20-01-TRANSPORT,FW-20-01-AUTHORIZATION |
| FW-20-01-TRANSPORT | DL-20 | NOT STARTED | Attribute representative endpoint latency to request transport and response serialization. |
| FW-20-01-AUTHORIZATION | DL-20 | NOT STARTED | Attribute representative endpoint latency to authentication and authorization stages. |
| FW-20-01A | DL-20 | PARENT | Parent scope; children: FW-20-01A-GRAPH-STORAGE,FW-20-01A-INDEXING |
| FW-20-01A-GRAPH-STORAGE | DL-20 | NOT STARTED | Attribute representative endpoint latency to canonical graph storage. |
| FW-20-01A-INDEXING | DL-20 | NOT STARTED | Attribute representative endpoint latency to semantic indexing. |
| FW-20-01B | DL-20 | PARENT | Parent scope; children: FW-20-01B-SQLITE-SMALL,FW-20-01B-SQLITE-LARGE,FW-20-01B-CHROMA-SMALL,FW-20-01B-CHROMA-LARGE,FW-20-01B-POSTGRES-SMALL,FW-20-01B-POSTGRES-LARGE |
| FW-20-01B-SQLITE-SMALL | DL-20 | NOT STARTED | Attribute worker-transfer latency for the small fixture on SQLite persistence. |
| FW-20-01B-SQLITE-LARGE | DL-20 | NOT STARTED | Attribute worker-transfer latency for the large fixture on SQLite persistence. |
| FW-20-01B-CHROMA-SMALL | DL-20 | NOT STARTED | Attribute worker-transfer latency for the small fixture with Chroma projection. |
| FW-20-01B-CHROMA-LARGE | DL-20 | NOT STARTED | Attribute worker-transfer latency for the large fixture with Chroma projection. |
| FW-20-01B-POSTGRES-SMALL | DL-20 | NOT STARTED | Attribute worker-transfer latency for the small fixture on PostgreSQL/pgvector. |
| FW-20-01B-POSTGRES-LARGE | DL-20 | NOT STARTED | Attribute worker-transfer latency for the large fixture on PostgreSQL/pgvector. |
| FW-20-03 | DL-20 | PARENT | Parent scope; children: FW-20-03-TIMING,FW-20-03-COUNTER |
| FW-20-03-TIMING | DL-20 | NOT STARTED | Add ingress/transport stage timing without changing request semantics. |
| FW-20-03-COUNTER | DL-20 | NOT STARTED | Add an ingress/transport operation counter without changing request semantics. |
| FW-20-03A | DL-20 | PARENT | Parent scope; children: FW-20-03A-AUTHENTICATION,FW-20-03A-AUTHORIZATION,FW-20-03A-GRAPH-STORAGE |
| FW-20-03A-AUTHENTICATION | DL-20 | NOT STARTED | Add an authentication-stage timing counter with a focused attribution test. |
| FW-20-03A-AUTHORIZATION | DL-20 | NOT STARTED | Add an authorization-stage timing counter with a focused attribution test. |
| FW-20-03A-GRAPH-STORAGE | DL-20 | NOT STARTED | Add a canonical graph-storage timing counter with a focused attribution test. |
| FW-20-03B | DL-20 | PARENT | Parent scope; children: FW-20-03B-INDEXING,FW-20-03B-WORKER-TRANSFER |
| FW-20-03B-INDEXING | DL-20 | NOT STARTED | Add an indexing-stage timing counter and verify attribution. |
| FW-20-03B-WORKER-TRANSFER | DL-20 | NOT STARTED | Add a worker-transfer timing counter and verify attribution. |
| FW-20-04 | DL-20 | NOT STARTED | Repeat resource-matched Python versus mixed endpoint comparison and verify identical digest. |
| FW-20-04A | DL-20 | NOT STARTED | Repeat resource-matched Rust-native endpoint comparison and verify identical digest. |
| FW-20-04B | DL-20 | PARENT | Parent scope; children: FW-20-04B-PYTHON,FW-20-04B-MIXED,FW-20-04B-RUST |
| FW-20-04B-PYTHON | DL-20 | NOT STARTED | Report Python-mode p95/p99 latency and sample variation from the matched-resource endpoint run. |
| FW-20-04B-MIXED | DL-20 | NOT STARTED | Report mixed-mode p95/p99 latency and sample variation from the matched-resource endpoint run. |
| FW-20-04B-RUST | DL-20 | NOT STARTED | Report Rust-native p95/p99 latency and sample variation from the matched-resource endpoint run. |
| FW-20-02 | DL-20 | PARENT | Parent scope; children: FW-20-02-PYTHON,FW-20-02-MIXED |
| FW-20-02-PYTHON | DL-20 | NOT STARTED | Repeat the endpoint comparison for Python mode with correctness digest and enough samples. |
| FW-20-02-MIXED | DL-20 | NOT STARTED | Repeat the endpoint comparison for mixed mode with correctness digest and enough samples. |
| FW-20-02A | DL-20 | NOT STARTED | Repeat endpoint comparison with correctness digest and enough samples for Rust-native mode. |
| FW-20-02B | DL-20 | NOT STARTED | State performance conclusions only where confidence/variation evidence supports them. |
| FW-20A-01 | DL-20A | NOT STARTED | Join FFI and serialized-byte counters for one representative graph trace. |
| FW-20A-01A | DL-20A | NOT STARTED | Join worker-dispatch and transaction counters for one representative workflow/storage trace. |
| FW-20A-01B | DL-20A | PARENT | Parent scope; children: FW-20A-01B-INDEXING,FW-20A-01B-MCP |
| FW-20A-01B-INDEXING | DL-20A | NOT STARTED | Add joined performance counters to one representative indexing trace. |
| FW-20A-01B-MCP | DL-20A | NOT STARTED | Add joined performance counters to one representative MCP trace. |
| FW-20A-03 | DL-20A | NOT STARTED | Verify counters join to one trace/run identity and count no duplicate state effects. |
| FW-20A-03A | DL-20A | NOT STARTED | Verify ingress, worker, FFI, serialization, and transaction ownership fields correlate for one mixed trace. |
| FW-20A-04 | DL-20A | PARENT | Parent scope; children: FW-20A-04-SMALL,FW-20A-04-LARGE |
| FW-20A-04-SMALL | DL-20A | NOT STARTED | Compare batch and per-item small fixtures and assert identical final-state digests. |
| FW-20A-04-LARGE | DL-20A | NOT STARTED | Compare batch and per-item large fixtures and assert identical final-state digests. |
| FW-20A-04A | DL-20A | NOT STARTED | Define regression thresholds for crossings, serialized bytes, and latency using measured baseline. |
| FW-20A-02 | DL-20A | NOT STARTED | Add batch-versus-per-item graph fixture and verify no duplicated state effects. |
| FW-20A-02A | DL-20A | NOT STARTED | Add batch-versus-per-item workflow fixture and verify crossing/serialization counts. |
| FW-20A-02B | DL-20A | NOT STARTED | Add regression assertion for threshold violations without comparing different operation semantics. |
