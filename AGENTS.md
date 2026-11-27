# Kogwistar Core Agent Guidance

## Scope

- Put reusable graph, runtime, workflow, storage, and substrate primitives in
  this repository. Keep product orchestration in the consuming application.
- Preserve the public Python API and existing wire defaults unless a reviewed
  compatibility change is explicitly intended.
- Prefer one end-to-end capability slice over broad file-by-file translation or
  repository reorganization.

## Dual-Language Work

- **Locked acceptance workflow:** For each existing goal item, first enumerate
  its complete required test set and baseline; add that full set before
  implementation, then freeze exact selectors. Next implement, run focused unit
  tests, then the smallest relevant integration batch. No tests or checklist
  requirements may be added mid-implementation to postpone closure. Add later
  coverage only for a reproduced defect or approved contract change, with the
  reason recorded separately. When every frozen required test passes, mark the
  existing item `PASS` in that same slice; do not expand its denominator.
- North star: keep Python and Rust stacks independently usable and
  interchangeable only at explicit, safe seams; pursue a Rust-native
  end-to-end path as an additional zero-cross-language-overhead option. Preserve
  auth, scope, provenance, durability, transaction ownership and wire behavior;
  never infer full parity from a narrow passing slice.
- Locked TDD sequence for every existing goal item: (1) enumerate the complete
  required test set and baseline; (2) add all tests for the selected batch and
  freeze selectors; (3) implement against that fixed set, run unit tests and
  fix failures; (4) run integration tests and fix failures. Do not add or replace
  tests mid-implementation to delay closure. When every frozen required test
  passes, close the existing item in the same slice; do not grow its denominator.

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
- Fixed-batch rule: before editing an open DLFC item, enumerate every required
  case by exact test name/selector, backend and contract cell; record baseline
  pass/fail/unrun counts. Create the complete planned Python/Rust test batch
  first, then freeze it before implementation starts. Implement against this
  fixed acceptance set; run focused unit tests, then integration tests. Once
  implementation starts, do not add, replace, or broaden tests to defer
  closure. Only a reproduced regression or approved contract change may justify
  an additional case, with its reason recorded separately. When every frozen
  required case passes, mark that existing item `PASS` immediately and update
  its mapped coarse status. Keep IDs and denominators unchanged; skipped or
  unrun tests are not passes. Passing starter tests alone does not close an item
  whose frozen acceptance set is incomplete.
- At each implementation turn, name the selected existing DLFC item and report
  its frozen case count plus passed/failed/unrun counts. Do not describe planned
  tests as implemented; do not resume implementation until the batch is frozen.
- Multi-item Rust batch: when one implementation slice targets several
  existing DLFC items, freeze each item's complete test set and baseline counts
  before changing code. Add all planned tests for all selected items first;
  only then implement the batch. Do not expand the frozen set mid-implementation
  unless a reproduced regression or approved contract change requires it.
  Once implementation begins, this is a locked acceptance batch: a failing
  case calls for implementation fixes, not new requirements or replacement
  tests. Passing every frozen required case closes its mapped existing item.
- Fixed-test-set execution rule: before any Rust implementation, enumerate the
  existing item's complete acceptance cases by exact selector and backend, then
  add the full planned test batch and freeze it. After implementation starts,
  do not add tests opportunistically, split one requirement into new checklist
  rows, or rerun already-passing unrelated suites as a substitute for progress.
  Fix code against the frozen failures. Run targeted unit tests first, then the
  smallest relevant integration batch. If all frozen required cases pass, mark
  the item `PASS` and update its existing coarse mapping immediately in that
  same slice. Report frozen total and pass/fail/unrun counts every turn; if no
  test is missing, say so plainly. Add a case later only for a reproduced
  defect or approved contract change, documenting why it does not move the
  original goalpost.
- Closure is mandatory, not discretionary: as soon as every frozen required
  case passes, mark that existing DLFC `PASS` in the same slice and update its
  coarse mapping. Do not leave it open for extra polish, add tests after seeing
  results, or change the denominator. Preserve prior evidence and record a
  separate regression or approved contract change only when one genuinely
  invalidates the gate.

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
