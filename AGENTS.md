# Kogwistar Core Agent Guidance

## Scope

- Put reusable graph, runtime, workflow, storage, and substrate primitives in
  this repository. Keep product orchestration in the consuming application.
- Preserve the public Python API and existing wire defaults unless a reviewed
  compatibility change is explicitly intended.
- Prefer one end-to-end capability slice over broad file-by-file translation or
  repository reorganization.

## Dual-Language Work

- Follow `kogwistar/docs/ADR-022-dual-language-capability-paths.md` and its
  existing checklists. Select an existing open item; keep IDs and denominators
  fixed. New scope requires an approved, separately linked requirement.
- For each selected item, **before implementation**, enumerate and freeze its
  complete acceptance set by exact test selector, backend/contract cell, and
  baseline status. For a multi-item batch, freeze every item's complete set
  first. Add the whole planned test batch before implementation begins.
- Then follow this order: implement against that frozen set; run focused unit
  tests and fix implementation failures; run the smallest relevant integration
  batch and fix failures. Never add, replace, or broaden tests mid-batch merely
  to delay closure. Later tests are allowed only for a reproduced regression or
  approved contract change; record the reason without changing the original
  goalpost.
- **Mandatory closure:** when every frozen applicable test passes, mark the
  existing item `PASS` in that same slice and update its already-mapped coarse
  status. Do not wait for extra polish. Skipped/unrun tests are not passes; an
  incomplete frozen acceptance set cannot close the item.
- At every implementation turn, state the selected item(s), frozen test count,
  passed/failed/unrun counts, compatibility evidence, blockers, and coarse/fine
  completion percentages. Never call planned tests implemented or infer full
  parity from a narrow passing slice.
- Keep Python and Rust independently usable; switch only at declared safe seams,
  with no silent cross-language fallback. Preserve auth, scope, provenance,
  durability, wire behavior, and single-owner atomic transactions. Rust should
  project Python behavior where applicable, with source/test annotations.
- Preserve unrelated worktree changes and reuse the existing Cargo target
  unless isolation is necessary. Before claiming native end-to-end support,
  verify ingress through durable result across the applicable auth, retry,
  trace, backend, and interpreter paths.

## SQLite Authority

- Each execution context binds to exactly one SQLite implementation, and that
  binding does not change during the context.
- Python and Rust must not hold overlapping connections to the same SQLite
  database in one process. Cross-process ownership must use the existing
  database authority lock.
- Independent contexts are allowed for separate databases; they do not permit
  two implementations to write the same database concurrently.
- Keep these guarantees enforced at supported entrypoints and covered by the
  SQLite authority tests. Do not describe SQLite metadata support as canonical
  graph-write authority unless the graph transaction path is implemented and
  tested.

## Guidance Boundaries

- These are repository engineering rules, not a response-style mode. Adapt
  answer length and tone to the user's request; any terse or stylized mode is
  opt-in and must not override clarity, safety, or repository work.
- Do not treat generated reports, local caches, benchmark outputs, or unrelated
  dirty files as disposable without checking ownership first.
