# Chroma Rust Interop Worklist

**Updated:** 2026-11-23  
**Scope:** Verification-only slice for Rust Chroma support and Python↔Rust data handoff. Existing architecture already provides `ChromaGraphStore` and SQLite+Chroma composition; expected outcome is test evidence, not product-code changes.  
**Exit:** All counted fine items PASS and all coarse gates derive PASS. Combined with accepted CI #859/#860 evidence for the six named non-Chroma capabilities, this closes the documented cutover/interoperability gates in scope. It does **not** prove every runtime/platform combination (the `sqlite-meta` profile still records partial mixed-runtime coverage), Chroma as Rust's canonical event authority, transactions spanning SQLite and Chroma, or universal Kogwistar behavior parity.

## Progress

- **Coarse:** 4/4 PASS (100%); 0 IN PROGRESS; 0 NOT STARTED.
- **Fine:** 11/11 PASS (100%); 0 IN PROGRESS; 0 NOT STARTED.
- Coarse status is mechanical: every mapped fine item must PASS. Parent rows do not count toward fine totals. Do not add scope absent a demonstrated missing contract.

## Coarse Gates

| ID | Status | Required fine items | Closure rule |
|---|---|---|---|
| CH-C1 Rust adapter contract | PASS | CH-F1–CH-F5 | Five adapter behavior cells pass against Rust Chroma `GraphReadStore`/`GraphWriteStore` contract. |
| CH-C2 Bidirectional Python↔Rust handoff | PASS | CH-F6–CH-F9 | Python-produced state is consumed by Rust and Rust-produced state by Python, including worker handoff. |
| CH-C3 Durable authority and eventual projection | PASS | CH-F10 | SQLite remains canonical; Chroma failure/retry/restart converges without claiming cross-store transactions. |
| CH-C4 Rust service path and selection | PASS | CH-F11 | Live Rust server binary uses configured Rust Chroma adapter for authenticated request paths. |

## Fine Tasklets

| ID | Coarse | Status | Invariant, cell, oracle, check/evidence |
|---|---|---|---|
| CH-F1 | CH-C1 | PASS | Invariant: Rust adapter implements scoped Chroma v2 graph CRUD and vector query. Cell: Rust `ChromaGraphStore` HTTP contract. Oracle: create/read/update/delete and query return canonical records, expected order and errors. Evidence: `cargo test --manifest-path rust/Cargo.toml -p kogwistar-store-chroma --test http_contract` (**8 passed**, includes this contract and metric normalization). |
| CH-F2 | CH-C1 | PASS | Invariant: node and edge sharing a canonical ID remain distinct. Cell: separate Rust Chroma entity collections. Oracle: kind-aware reads return the correct entity from each collection. Evidence: same Rust HTTP contract target (**8 passed**). |
| CH-F3 | CH-C1 | PASS | Invariant: vector collection selection and namespace scope prevent cross-scope reads. Cell: named Rust Chroma collections. Oracle: requested collection and namespace are both honored. Evidence: same Rust HTTP contract target (**8 passed**). |
| CH-F4 | CH-C1 | PASS | Invariant: paged Chroma reads return every row without skipping a page boundary. Cell: Rust snapshot reader. Oracle: ordered pages reconstruct expected full result. Evidence: same Rust HTTP contract target (**8 passed**). |
| CH-F5 | CH-C1 | PASS | Invariant: bounded Chroma snapshots fail closed rather than return truncated graph state. Cell: Rust snapshot reader. Oracle: overflow returns the documented error and no partial snapshot. Evidence: same Rust HTTP contract target (**8 passed**). |
| CH-F6 | CH-C2 | PASS | Invariant: Python-created Chroma projection is readable and reconcilable by Rust. Cell: live Chroma + Rust server. Oracle: Rust observes/reconciles Python projection without duplicate or missing graph records. Evidence: `.venv\Scripts\python.exe -m pytest tests/server/test_rust_api_chroma_interop_live.py -q --run-manual -p no:cacheprovider` (**4 passed**, 93.36s; includes `test_python_chroma_projection_is_read_and_reconciled_by_rust_server`). |
| CH-F7 | CH-C2 | PASS | Invariant: Rust-authenticated graph mutation is visible to Python after engine reopen and retry is idempotent. Cell: live Chroma with SQLite canonical events. Oracle: Python reads Rust-projected graph after reopen; retry adds no duplicate effects; canonical events remain readable. Evidence: same live suite (**4 passed**), including `test_rust_graph_mutation_is_readable_by_python_over_chroma`. |
| CH-F8 | CH-C2 | PASS | Invariant: Python Stage-1 metadata is promoted by Rust worker. Cell: live Chroma + Rust-native projection worker. Oracle: worker completes promotion and Python-visible projection matches canonical metadata. Evidence: same live suite (**4 passed**), including `test_python_stage1_is_promoted_by_rust_native_worker`. |
| CH-F9 | CH-C2 | PASS | Invariant: Rust Stage-1 metadata is promoted by Python worker. Cell: live Chroma + Python worker. Oracle: Python worker consumes Rust-produced metadata and completes matching projection. Evidence: same live suite (**4 passed**), including `test_rust_metadata_stage1_is_promoted_by_python_worker`. |
| CH-F10 | CH-C3 | PASS | Invariant: Chroma is an eventual projection, not part of SQLite's transaction; interrupted projection remains retryable and converges after restart. Cell: Rust embedding worker with injected first-write failure and restarted SQLite state. Oracle: canonical event survives, retry reuses accepted action, projection converges without duplicates. Evidence: `cargo test --manifest-path rust/Cargo.toml -p kogwistar-api sqlite_chroma_embedding_worker_projects_revision_fenced_stage1_document` (**1 passed**). |
| CH-F11 | CH-C4 | PASS | Invariant: configured Rust service routes graph operations through the Rust Chroma adapter. Cell: Rust server binary + SQLite metadata + live Chroma. Oracle: authenticated Rust request reads/reconciles Python projection and writes projection later readable by Python. Evidence: same live suite (**4 passed**); harness launches current Rust server binary with `KOGWISTAR_CHROMA_ENDPOINT` and SQLite metadata configuration. This does not infer process-wide zero crossings from response annotations. |

## Rules

- Run existing named tests on this checkout first. Do not change implementation pre-emptively.
- If a test fails, diagnose test/environment versus real behavior gap; make product-code changes only for a reproduced contract defect, then update only affected evidence.
- Do not promote `chroma-adapter` to Rust canonical-writer or transactional readiness: the accepted design keeps SQLite events/index jobs canonical and Chroma eventual/non-transactional.
- “Full Rust interop” here means the bounded Chroma adapter/service and Python↔Rust handoff scope above. It is not a claim of universal feature parity or a new 9th Rust cutover capability.
