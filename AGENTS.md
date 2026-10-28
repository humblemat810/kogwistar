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
  checklist. Select an existing open checklist item before proposing new scope.
- Keep Python and Rust implementations supported. Selecting one implementation
  for a capability does not deprecate the other.
- Freeze implementation selection at a declared capability/execution boundary.
  Do not silently fall back between languages.
- One implementation owns all effects in an atomic unit of work. Never split a
  transaction across Python and Rust connections; use a declared, versioned
  handoff before or after that transaction instead.
- Keep checklist IDs and denominator stable. Close a `PASS` item only with
  concrete code-path and test evidence; distinguish skipped tests from passes.
- Before claiming native end-to-end support, verify ingress through durable
  result, including authorization, persistence, retries, traces, and the
  applicable backend/interpreter matrix. A narrow benchmark is not whole-stack
  proof.
- Run focused Rust tests and the corresponding Python contract/parity tests for
  each change. Preserve unrelated worktree changes and use the existing Cargo
  target directory unless isolation is necessary.

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
