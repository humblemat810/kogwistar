# ADR-015 Case Study: Nondeterministic Compatibility Failures

**Date:** 2026-07-16  
**Scope:** ADR-015 Rust-port compatibility rehearsal on Windows  
**Status:** Living operational note

## Why this exists

ADR-015 requires semantic parity evidence. A rerun that happens to pass is not by
itself evidence that a failure was harmless. During the release rehearsal, three
different environmental defects appeared as product or Rust parity failures. This
note records how they were classified, reproduced, and contained without skipping
release gates.

## Classification rule

For each failure:

1. Preserve the exact candidate identity and failing command.
2. Read the complete traceback; do not classify from the final assertion alone.
3. Rerun the exact test node or file in a fresh process at least three times.
4. Treat a repeatable semantic mismatch as a defect until fixed.
5. Treat mixed pass/fail results as unresolved nondeterminism, not success.
6. Change source only after the cause is bounded. A source change creates a new
   candidate identity and requires fresh compatibility evidence.
7. Resume only when source and harness identity are unchanged. Reuse only commands
   already recorded as successful for that identity.

Tests are not removed, weakened, or marked out merely because they are flaky.

## Case 1: Embedded Chroma HNSW segment not visible

### Symptom

Several unrelated tests failed in `get` or `query` with:

```text
chromadb.errors.InternalError: Error executing plan: Internal error:
Error creating hnsw segment reader: Nothing found on disk
```

Observed in:

- `tests/primitives/test_document_rollback.py`;
- `tests/core/test_tutorial_ladder_smoke.py` subprocess tutorials;
- `tests/kg_conversation/test_conversation_flow.py` Chroma parameter;
- an earlier accounting test that opened multiple embedded Chroma roots.

The same files passed in fresh processes. One rollback file passed three consecutive
isolated runs; the tutorial file passed all 9 selected tests when isolated.

### Cause and trigger

Embedded Chroma uses process-global native state. Opening independent persistent
roots, rapidly creating collections, or reopening/querying immediately after writes
can expose a native HNSW segment before its on-disk reader is available. The error
is below Kogwistar's Rust facade and occurs on Python-authoritative Chroma adapter
paths.

### Containment

- Release core evidence uses one test file per process.
- Tests that need multiple logical graphs should share one isolated Chroma root and
  use unique collection names, or run independent roots in separate processes.
- A real Chroma-backed system under test must not be replaced with a fake merely to
  make the gate green.
- Exact failing files are rerun in isolation before a report is resumed.

### Follow-up

Prefer explicit Chroma client lifecycle/close support and add a stress regression
that repeatedly writes then queries distinct logical collections under one root.
Do not add blind sleeps or catch-and-retry every `InternalError`; those approaches
can hide real storage corruption.

## Case 2: Parser suite exhausts Windows file handles

### Symptom

The parser compatibility layer passed early tests, then many unrelated Chroma,
Ollama, TLS, OCR, and provider tests failed together. The earliest common traceback
was:

```text
OSError: [Errno 24] Too many open files
```

It occurred while `ssl.create_default_context()` loaded the CA file, not while
validating parser behavior.

### Cause and trigger

The entire parser directory originally ran in one pytest process. Repeated creation
of embedded Chroma clients, HTTP clients, provider adapters, and workflow engines
accumulated Windows handles. Later tests failed at whichever resource allocation
happened next, producing misleading provider-specific failures.

Representative parser files passed when executed in fresh processes, including
live Ollama embedding calls.

### Containment

Release parser coverage now uses one test file per process, just like release core.
The normal fast suite remains a single process. Pytest return code 5 is accepted
only for file-isolated release layers where the file contains no selected
`ci`/`ci_full` items; the raw code remains in the report.

This preserves full file discovery while releasing native and HTTP handles between
files.

### Follow-up

Audit engine, Chroma, HTTP, and provider objects for deterministic `close()` or
context-manager ownership. Process isolation is a valid release evidence boundary,
but it does not replace fixing resource ownership in long-lived applications.

## Case 3: Test classification and import defects looked environmental

### Repository-global joblib cache pollution

`tests/kg_conversation/test_conversation_flow.py` originally used `.joblib` at the
repository root for a parameterized filtering callback. Results from another
provider/backend invocation could be replayed into the current scenario, changing
the number of projected knowledge-reference nodes. The cache now lives under the
test's `tmp_path`; cache behavior remains exercised without cross-scenario state.
The same test parameterizes actual Gemini and Ollama task providers, so those cases
are classified as `llm_real` and `requires_ollama` rather than deterministic release
CI. Storage/backend parity remains covered by deterministic conversation suites.

### Real-LLM marker leak

`tests/primitives/test_adjudication_with_llm_cache.py` was marked `ci_full` but
explicitly invoked Azure OpenAI. Because credentials existed while DNS access did
not, release core attempted a real request and raised `openai.APIConnectionError`.

The test and analogous real-provider cases were marked `llm_real` (or
`requires_ollama`) so the deterministic release selector applies its documented
policy. This was a marker correction, not a skipped parity assertion.

### File-isolated fixture discovery

`tests/runtime/test_workflow_suspend_resume.py` requested `real_chroma_server`
dynamically but did not import its fixture into the module. Directory-wide runs
could discover it accidentally through other modules; file-isolated runs correctly
reported `fixture 'real_chroma_server' not found`. Importing the fixture explicitly
fixed all selected tests, including async Chroma and PostgreSQL paths.

### Consumer import typo

The parser's PDF-indexed test imported `pdf2png` as a top-level package, while the
owned module is `kg_doc_parser.pdf2png`. Correcting the import made the exact test
pass without installing an unrelated package or excluding the test.

## Case 4: Outer command timeout masquerades as parser failure

### Symptom

A four-layer feature run recorded parser return code `120` after 895.8 seconds.
No parser assertion or traceback was present. Core had already consumed 325
seconds, and the aggregate command ended at the orchestration tool's 20-minute
limit.

### Cause and proof

The compatibility runner has no 900-second parser timeout. Its parent command was
terminated by the outer execution deadline while pytest still used CPU. Running
the identical parser layer alone under the same candidate and interpreter passed:

```text
58 passed, 3 skipped, 143 deselected in 867.89s
```

Historical feature parser duration ranged from roughly 401 to 894 seconds, so
aggregate wall time can cross an outer deadline without any semantic regression.

### Containment

- Run long compatibility evidence one layer at a time.
- Keep one report and use `--resume`; successful commands are reused only when
  candidate identity is unchanged.
- Diagnose process CPU and report progress before declaring a hang.
- Treat an infrastructure return code without a pytest failure as inconclusive,
  never as a green or semantic red result.

### Follow-up

Expose an explicit outer orchestration deadline separate from pytest and record
termination provenance in reports. Parser performance variance remains worth
profiling; increasing a timeout alone is not a performance fix.

### Additional runtime-suite occurrence

On 2026-07-16, the focused runtime feature selector was first killed by a
120-second outer command deadline without an assertion or traceback. The exact
command, unchanged, passed under a 300-second execution window:

```text
210 passed, 146 deselected, 72 warnings in 189.75s
```

This confirms the same classification rule applies below the four-layer runner:
an outer timeout shorter than the observed suite duration is inconclusive, not a
runtime regression. The selector and test inputs were unchanged; only the command
execution window changed.

## Case 5: Worker journal leaked SQLite handles on Windows

### Symptom

A complete PostgreSQL true-socket runtime scenario passed all semantic assertions,
then temporary-directory cleanup failed with:

```text
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process
```

The locked file was a Python worker callback-result journal such as
`pg-fanout-branches.sqlite`.

### Cause and proof

`sqlite3.Connection` as a context manager commits or rolls back but does not close
the connection. `WorkerResultJournal` opened short-lived connections under `with`
and therefore retained native file handles until garbage collection. The entire
runtime flow, including deterministic out-of-order fanout retry and terminal state,
had already passed; only cleanup exposed the leak.

Replacing each journal scope with explicit `contextlib.closing` plus the transaction
context releases the handle deterministically. A Windows regression now creates a
journal row and immediately unlinks the database file.

### Containment and follow-up

- Explicitly close every short-lived SQLite journal connection.
- Keep temp-directory deletion as part of live test teardown; it is useful leak
  detection, not incidental cleanup.
- Audit other SQLite/Chroma/HTTP test helpers for the same mistaken assumption that
  a transaction context closes its resource.
- Do not add cleanup retries or sleeps; those hide ownership bugs and contribute to
  the file-handle exhaustion described in Case 2.

## Evidence and reporting policy

## Case 7: Pre-existing Python trace matrix mistaken for Rust transition scan

### Symptom

`tests/runtime/test_trace_sink_parallel_nested_minimal.py` took 610.8 seconds
for 15 cases (an earlier run took about 981.6 seconds). This was suspected to be
caused by the Rust recorded-runtime path replaying all run events.

### Cause and proof

The test constructs and calls Python `WorkflowRuntime` directly. It does not use
the Rust recorded-transition store path. Isolated minimum-case timings were:

```text
fake    7.73s
chroma 27.94s
pg     22.48s
```

A `cProfile` run of the fake case attributed roughly 6.9--8.1 seconds to
`pydantic_extension` validation stack inspection (`inspect.stack`,
`inspect.getmodule`, and related source-file lookup). Graph fixture construction
repeatedly triggers this pre-existing Python-side cost. Chroma and PostgreSQL add
backend work on top. The test was slow before Rust use; the Rust O(n) scan was a
real but independent serving-path defect.

### Containment

- Keep the minimum fake smoke cases in normal compatibility profiles.
- Mark backend and stress cross-products `slow`; milestone/nightly still run them.
- Do not claim the Rust projection/index fix improves this Python-owned test.
- Profile or fix `pydantic_extension` stack inference separately; do not weaken
  runtime semantics or replace live backend coverage.

## Case 6: Feature marker cross-product includes slow live embeddings

### Symptom

The parser feature layer appeared stalled at:

```text
test_fake_workflow_run_text_success_and_knowledge_persist[chroma-worker]
```

Pytest kept using CPU without an assertion or traceback. Two runs were stopped
after about eight and two minutes while diagnosing the apparent hang.

### Cause and proof

The test is a Cartesian product. `chroma` carries `ci_full`, while `worker` carries
`ci`; pytest propagates both parameter marks to the combined case. The feature
selector `(ci or regression) and not slow` therefore selects `chroma-worker`, even
though Chroma construction uses a live Ollama embedding function. Historical
evidence for the same parser layer passed in 794.31 seconds; another recorded run
passed 58 tests with 3 skips in 867.89 seconds. High CPU plus the lack of a pytest
failure was slow provider work, not a Rust parity deadlock.

### Containment and follow-up

- Let the parser feature layer run for at least its recorded 15-minute envelope.
- Inspect parameter-mark combinations, not only function-level marks.
- Mark live embedding cross-products `requires_ollama` (or split deterministic fake
  embedding coverage) in the consumer repository; do not merely increase every CI
  timeout.
- Continue to treat outer return code 120 without traceback as inconclusive.

Every compatibility report records:

- candidate and verification-harness hashes;
- resolved core import provenance and native extension version;
- capability ownership modes and active writers;
- per-layer interpreter, marker expression, command, duration, and return code;
- successful command reuse only under an identical candidate identity.

A report containing a failed command is not a green release report, even if the
same test later passes manually. The failing report may be resumed only under the
same identity; otherwise a fresh four-layer report is required.

## Anti-patterns

Do not:

- add blanket retries around semantic assertions;
- use arbitrary sleeps to wait for native storage;
- accept a single green rerun as proof;
- replace Chroma/PostgreSQL coverage with fake storage;
- broaden `manual`, `llm_real`, or provider markers to hide deterministic tests;
- reuse a report after source, tests, runner, config, or consumer sources change;
- claim ADR exit gates from focused tests alone.
