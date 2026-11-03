# ADR-022 Remaining Worklist

**Updated:** 2026-11-03
**Purpose:** Track bounded work beneath the coarse gates in [ADR-022-dual-language-capability-checklist.md](ADR-022-dual-language-capability-checklist.md).

## Completion Snapshot

- **Coarse checklist:** **156/177 PASS (88.1%)**.
- **Fine worklist:** **19/536 PASS (3.5%)**; **0 IN PROGRESS**, **517 NOT STARTED**, **0 BLOCKED**.
- **Tasklet normalization:** The frozen DL-03 inventory is expanded into 83 HTTP-route parity leaves, 62 MCP contract leaves, and 62 MCP authorization leaves. Inventory/reconciliation is represented by non-counted parent groups, not duplicate leaves. DL-16BQ backend checks and DL-18/DL-20 implementation-mode checks are split into independently passing leaves. The PostgreSQL Rust-native answer smoke verifies provider-event publication and handling; its transport test now negotiates Streamable HTTP and inspects the structured MCP result. Other flagged multi-cell leaves remain provisional and must be audited/split before implementation. The percentage is raw leaf-status arithmetic, not a claim that remaining work is uniform or nearly complete.
- **Equal-size tasklet unit:** one leaf is one independently decidable acceptance assertion for one contract family and one execution cell (operation/backend/runtime only where that dimension can fail independently). Target one focused implementation-and-verification session of about 45 minutes; split before starting if the estimate exceeds 60 minutes. The estimate is a guardrail, not evidence of completion. A tasklet must name one observable behavior, one expected result, and one focused test/check or artifact. A parameterized test is one tasklet only when all cases prove that same invariant under the same contract; list the cases in the evidence, and split cases that have separate implementation or ownership paths. Separate independently passing endpoints, DTO families, error envelopes, lifecycle transitions, resources, backends, runtime modes, and operation families. Never compress them into a matrix leaf. Do not split merely by file, field, or test-case count when one invariant is shared. Word count is not a sizing measure.
- The fine denominator counts leaf rows only; `PARENT` rows are retained ID/mapping records and are not progress items. When a row is split, preserve its ID as a parent and list all child IDs in its description. The parent remains uncounted and cannot pass; every child starts `NOT STARTED` unless prior evidence independently proves that exact child. Passed leaf IDs/evidence remain unchanged. Report coarse and fine percentages separately; neither percentage estimates schedule or nearness to completion.

## How To Track

- This is the detailed execution list; the linked checklist remains the sole coarse progress and denominator source. Only counted leaf tasklets contribute to the fine percentage.
- The baseline is every coarse ID currently marked `IN PROGRESS` in the linked checklist. Every such coarse ID must map to fine work below. Fine IDs do not add to the 177-item coarse inventory.
- A `PARENT` row groups tasklets or nested parent groups and is not countable or completable. CI validates that every parent resolves to at least two descendant leaf IDs under the same coarse gate and that the hierarchy has no missing references or cycles. No task may be worked or reported as complete through a parent row.
- A fine task reaches `PASS` only when its stated acceptance evidence is recorded here or linked from its referenced test/report. Tests must exercise the named behavior, not merely compile the surrounding code.
- Close a coarse ID only when its original exit criteria and all mapped required fine tasks pass. Do not reopen a passed coarse ID without a reproduced regression or new required contract.
- Equal-size acceptance test: in one sentence, a reviewer can identify the contract, execution cell, behavior, expected result, and focused evidence, then decide pass/fail without interpreting a multi-part report. A leaf with multiple independently decidable outcomes is oversized even if its description is short or its test is parameterized.
- Words such as `all`, `each`, `every`, `remaining`, `inventory`, `matrix`, `reconcile`, and `cross-backend` are audit flags, not automatically invalid. Keep such a row only when the deliverable is one generated artifact or one assertion of a genuinely shared invariant. Otherwise convert the existing ID to an uncounted `PARENT`, enumerate child leaves at the independent contract-cell boundary, and count only those leaves. Do not add new product scope during decomposition.
- Do not create rows per file, field, or platform when they are merely implementation details of one invariant. Do create separate rows when a capability family, operation, backend, or runtime cell can independently pass while another fails. Keep the original ID as an uncounted parent and enumerate the children; never count both parent and children. Do not split already-passed evidence unless it fails the same acceptance test or a regression is reproduced.

**Status vocabulary:** `NOT STARTED`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

| Fine ID | Coarse ID | Status | One-session tasklet and acceptance evidence |
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
| FW-03-03A-HTTP-GET-3f7c2705 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/auth/callback using paired route tests. |
| FW-03-03A-HTTP-GET-a8089377 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/auth/login using paired route tests. |
| FW-03-03A-HTTP-POST-dc297acc | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/auth/logout using paired route tests. |
| FW-03-03A-HTTP-GET-d41dac18 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/auth/me using paired route tests. |
| FW-03-03A-HTTP-GET-e0dc6609 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP GET /api/conversations using paired route tests. |
| FW-03-03A-HTTP-POST-06ebeb79 | DL-03 | NOT STARTED | Compare the Python and Rust success status/body contract for HTTP POST /api/conversations using paired route tests. |
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
| FW-05-03C | DL-05 | NOT STARTED | Verify lifecycle and tombstone DTOs preserve redirect and endpoint fields through Python/Rust round trips. |
| FW-05-04 | DL-05 | PARENT | Parent scope; children: FW-05-04-REQUEST,FW-05-04-EVENT |
| FW-05-04-REQUEST | DL-05 | NOT STARTED | Compare provider request DTO canonical serialization and stable validation errors. |
| FW-05-04-EVENT | DL-05 | NOT STARTED | Compare provider event DTO canonical serialization and stable validation errors. |
| FW-05-04A | DL-05 | PARENT | Parent scope; children: FW-05-04A-PERSISTENCE,FW-05-04A-TRANSACTION |
| FW-05-04A-PERSISTENCE | DL-05 | NOT STARTED | Compare persistence error code and detail fields for one shared error fixture. |
| FW-05-04A-TRANSACTION | DL-05 | NOT STARTED | Compare transaction error code and detail fields for one shared error fixture. |
| FW-05-04B | DL-05 | PARENT | Parent scope; children: FW-05-04B-REST,FW-05-04B-MCP |
| FW-05-04B-REST | DL-05 | NOT STARTED | Compare REST error envelope fields for one shared invalid-request fixture. |
| FW-05-04B-MCP | DL-05 | NOT STARTED | Compare MCP error envelope fields for one shared invalid-request fixture. |
| FW-05-02 | DL-05 | NOT STARTED | Verify identifier and source-span boundary cases with shared valid/invalid fixtures and matching offsets. |
| FW-05-02A | DL-05 | NOT STARTED | Verify additive schema-version evolution remains accepted in both languages for the declared compatible DTOs. |
| FW-05-02B | DL-05 | NOT STARTED | Verify unsupported schema versions and unknown fields fail consistently before effects. |
| FW-06-01 | DL-06 | NOT STARTED | Inventory canonical durable resources and their Python transaction owner, one resource family per evidence row. |
| FW-06-01A | DL-06 | NOT STARTED | Inventory Rust and mixed-mode write entry points and map each to its declared transaction owner. |
| FW-06-01B | DL-06 | NOT STARTED | Mark resources with explicitly weaker backend guarantees and verify those guarantees are stated in the backend contract. |
| FW-06-03 | DL-06 | NOT STARTED | Verify SQLite single-writer ownership for event journal and graph projection with a concurrent-writer test. |
| FW-06-03A | DL-06 | PARENT | Parent scope; children: FW-06-03A-GRAPH,FW-06-03A-RUN,FW-06-03A-WORKER,FW-06-03A-ACL,FW-06-03A-INDEX |
| FW-06-03A-GRAPH | DL-06 | NOT STARTED | Verify PostgreSQL graph mutation has one transaction owner under an overlapping-writer test. |
| FW-06-03A-RUN | DL-06 | NOT STARTED | Verify PostgreSQL workflow/run mutation has one transaction owner under an overlapping-writer test. |
| FW-06-03A-WORKER | DL-06 | NOT STARTED | Verify PostgreSQL worker-journal mutation has one transaction owner under an overlapping-writer test. |
| FW-06-03A-ACL | DL-06 | NOT STARTED | Verify PostgreSQL ACL mutation has one transaction owner under an overlapping-writer test. |
| FW-06-03A-INDEX | DL-06 | NOT STARTED | Verify PostgreSQL index-job intent has one transaction owner under an overlapping-writer test. |
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
| FW-14C-03 | DL-14C | PASS | Defined strict `WorkerIdentityAttestationV1` with environment ID, artifact SHA-256, and dependency-lock SHA-256; payload is explicitly self-reported data, not trusted proof. Evidence: shared golden fixture plus Python contract test and Rust contract parser test pass. |
| FW-14C-03A | DL-14C | PASS | Python sync and async worker adapters reject missing, stale-artifact, lockfile-mismatched, and environment-mismatched required identities before callback execution; a matching registered identity executes and advertises its identity-specific claim recipient. Evidence: `tests/runtime/test_rust_runtime_python_worker.py::test_python_worker_identity_admission_rejects_before_callback`, `test_python_worker_identity_admission_accepts_registered_identity`, async rejection test, Rust step-payload forwarding test; focused worker suite passed. Attestation remains self-reported, not cryptographic proof. |
| FW-14C-03B | DL-14C | PASS | Rust rejects malformed authoritative graph identity pins before run creation or lane scheduling. Evidence: `sqlite_invalid_worker_identity_pin_is_rejected_before_scheduling` returns `409 KOGWISTAR_INVALID_WORKER_IDENTITY` and verifies the runtime queue remains empty; graph-plan and explicit-map validation tests also pass. |
| FW-14C-03C | DL-14C | PASS | Python and Rust accept the same valid attestation and reject all shared invalid cases (wrong version, noncanonical environment, malformed digest, unknown field). Evidence: `test_worker_identity_attestation_matches_shared_fixture`, Rust `worker_identity_attestation_matches_shared_python_golden_fixture`, and parity annotation verifier pass. |
| FW-14C-03D | DL-14C | PASS | A valid worker identity survives SQLite service reopen and still controls start-work routing and payload admission. Evidence: `sqlite_worker_identity_pin_survives_service_reopen_and_controls_claim` verifies the persisted identity map, identity-specific claim recipient, and `required_worker_identity` field after reopen. |
| FW-14C-03E | DL-14C | PASS | Identity-pinned claim recipients require exactly one matching `kogwistar_worker_identity` assertion in the cryptographically verified principal; missing, malformed, or mismatched registrations are rejected before queue claim handling in SQLite and PostgreSQL services. The trusted host JWT issuer must issue this claim only for a registered artifact/dependency-lock tuple. Evidence: SQLite signed-token registration/reopen regression; PostgreSQL rejects an unregistered principal before touching storage; Rust API suite **289 passed, 33 ignored**. This trusts issuer registration policy; it does not attest the live process image. |
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
| FW-16BQ-03 | DL-16BQ | NOT STARTED | Verify syscall authorization and namespace/scope semantics against Python for SQLite+Chroma. |
| FW-16BQ-03A | DL-16BQ | NOT STARTED | Verify syscall authorization and namespace/scope semantics against Python for PostgreSQL. |
| FW-16BQ-03B | DL-16BQ | NOT STARTED | Verify syscall authorization and namespace/scope semantics against Python for in-memory. |
| FW-16BQ-03C | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-03C-PG,FW-16BQ-03C-SQLITE-CHROMA,FW-16BQ-03C-MEMORY |
| FW-16BQ-03C-PG | DL-16BQ | NOT STARTED | Compare one shared syscall error envelope on PostgreSQL against the Python contract. |
| FW-16BQ-03C-SQLITE-CHROMA | DL-16BQ | NOT STARTED | Compare one shared syscall error envelope on SQLite+Chroma against the Python contract. |
| FW-16BQ-03C-MEMORY | DL-16BQ | NOT STARTED | Compare one shared syscall error envelope in-memory against the Python contract. |
| FW-16BQ-04 | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-04A,FW-16BQ-04B |
| FW-16BQ-04A | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-04A-HANDLE,FW-16BQ-04A-ACK,FW-16BQ-04A-TRACE,FW-16BQ-04A-REPLAY |
| FW-16BQ-04A-HANDLE | DL-16BQ | NOT STARTED | Verify one native provider event is handled on SQLite+Chroma without Python fallback. |
| FW-16BQ-04A-ACK | DL-16BQ | PASS | `sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` rejects the wrong claimant without consuming the message and acknowledges only the matching active claim (**1 passed**). |
| FW-16BQ-04A-TRACE | DL-16BQ | PASS | `sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` verifies the handled event is linked to its workflow step in the conversation graph and records the handled trace (**1 passed**). |
| FW-16BQ-04A-REPLAY | DL-16BQ | PASS | `sqlite_provider_event_ack_commits_handler_transition_and_causal_graph_atomically` retries the exact committed effect and asserts unchanged graph-event and handled-trace counts (**1 passed**). |
| FW-16BQ-04B | DL-16BQ | PARENT | Parent scope; children: FW-16BQ-04B-PUBLISH,FW-16BQ-04B-HANDLE,FW-16BQ-04B-ACK,FW-16BQ-04B-REPLAY |
| FW-16BQ-04B-PUBLISH | DL-16BQ | PASS | Live `postgres_agentic_answer_workflow_executes_durably_without_python_worker` verifies the streamed provider event, matching conversation-graph records/ACL, and PostgreSQL schema placement (**1 passed**). |
| FW-16BQ-04B-HANDLE | DL-16BQ | PASS | Live `postgres_agentic_answer_workflow_executes_durably_without_python_worker` completes a structured-answer turn through the Rust-native executor and verifies no lane message is assigned to `python-worker` (**1 passed**). |
| FW-16BQ-04B-ACK | DL-16BQ | NOT STARTED | Verify PostgreSQL acknowledges only the active provider-event claim. |
| FW-16BQ-04B-REPLAY | DL-16BQ | NOT STARTED | Replay one acknowledged PostgreSQL provider event; assert no duplicate graph effect. |
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
| FW-18-01B | DL-18 | PARENT | Parent scope; children: FW-18-01B-PYTHON,FW-18-01B-MIXED,FW-18-01B-RUST |
| FW-18-01B-PYTHON | DL-18 | NOT STARTED | Verify the Python implementation's final digest for the balanced trace against its frozen expected digest. |
| FW-18-01B-MIXED | DL-18 | NOT STARTED | Verify the mixed Python/Rust implementation's final digest for the balanced trace against its frozen expected digest. |
| FW-18-01B-RUST | DL-18 | NOT STARTED | Verify the Rust-native implementation's final digest for the balanced trace against its frozen expected digest. |
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
| FW-20-04B | DL-20 | PARENT | Parent scope; children: FW-20-04B-PYTHON,FW-20-04B-MIXED,FW-20-04B-RUST |
| FW-20-04B-PYTHON | DL-20 | NOT STARTED | Report Python-mode p95/p99 latency and sample variation from the matched-resource endpoint run. |
| FW-20-04B-MIXED | DL-20 | NOT STARTED | Report mixed-mode p95/p99 latency and sample variation from the matched-resource endpoint run. |
| FW-20-04B-RUST | DL-20 | NOT STARTED | Report Rust-native p95/p99 latency and sample variation from the matched-resource endpoint run. |
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
