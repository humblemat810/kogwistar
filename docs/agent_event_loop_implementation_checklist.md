# Graph-Native Streaming Agent Implementation Checklist

Status meanings: `NOT STARTED`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.
Every `PASS` must cite a test, check, or smoke record. Update this ledger as
implementation and verification progress; do not infer completion from code
being present.

| ID | Requirement | Status | Evidence / notes |
| --- | --- | --- | --- |
| EVT-01 | Typed versioned provider and normalized event contracts | PASS | `tests/agent/test_agent_streaming_contracts.py`: typed OpenAI contract events; Codex fixture coverage in root `tests/unit/test_codex_streaming_provider.py`. |
| EVT-02 | Async streaming provider protocol preserves synchronous provider API | PASS | `tests/agent/test_agent_streaming_contracts.py::test_streaming_contract_preserves_existing_sync_model_provider_api` passes on CPython 3.13; verifies the legacy `complete()` contract and that a stream-only provider is not mistaken for a synchronous provider. `SequenceFakeModel` now also satisfies the existing provider identity contract. |
| EVT-03 | Bounded incremental JSON parser; completed fields are advisory only | PASS | `tests/agent/test_agent_streaming_contracts.py`: nested/split Unicode, every two-chunk boundary, duplicate keys, malformed/incomplete input, trailing bytes, and size cap; `tests/agent/test_agent_event_runtime.py` verifies completed fields are journaled, token deltas are not. |
| EVT-04 | OpenAI Responses streaming adapter and deterministic protocol tests | PASS | `tests/agent/test_agent_streaming_contracts.py`: mocked SSE normalization, credential absence, and tool-call completion marker. Live API smoke remains optional and was not run. |
| EVT-05 | Codex App Server streaming adapter and deterministic protocol tests | PASS | `tests/unit/test_codex_streaming_provider.py` (3 passed); real smoke on 2026-10-23: provider `openai.codex_app_server`, model `gpt-5.6-luna`, structured output valid, event kinds `field_completed,response_completed,response_started,text_delta`, 4764.66 ms, trace `1b02c44913b842238bd479868e644d80`, response `01a16824-ebb1-76a1-9f35-618353966ec6`. No prompt, answer, or credential persisted. |
| EVT-06 | Workflow design uses ordinary Kogwistar nodes and predicate edges | PASS | `tests/agent/test_agent_streaming_contracts.py::test_event_workflow_is_ordinary_predicate_routed_graph`; all event workflow nodes explicitly set `wf_cacheable=false` to prevent replaying side effects. |
| EVT-07 | Workflow consumes one durable semantic event per iteration | PASS | `tests/agent/test_agent_event_graph_run.py::test_fake_provider_runs_graph_workflow_and_keeps_events_out_of_transcript` passes on CPython 3.13. It runs the actual workflow against three durable semantic messages and verifies exactly three `agent.event.receive` execution steps, one per event. |
| EVT-08 | Semantic events persist in conversation graph and remain out of transcript | PASS | The same graph integration test verifies three durable lane messages complete, run/step causal edges persist, and the real transcript projection returns only the user and assistant turns. Token deltas remain excluded by journal/runtime tests. |
| EVT-09 | Run-to-event and execution-step-to-event causal links persist | PASS | `tests/core/test_lane_messaging.py::test_lane_message_can_link_to_its_run_scoped_workflow_step` verifies graph-backed `about_run` and `handled_by_step` edges, plus rejection when event/step are supplied under a different run. The event trace also stores the workflow design node ID. |
| EVT-10 | Event acknowledgement follows durable handler/checkpoint completion | PASS | `tests/agent/test_agent_event_journal.py::test_event_journal_persists_semantic_event_then_acknowledges_after_handler` verifies publish -> claim -> handler -> ack ordering; failure test verifies requeue instead of ack. |
| EVT-11 | TraceContext/EventEmitter correlation covers provider, event, hook, and tool spans | PASS | `tests/agent/test_agent_event_runtime.py::test_event_handler_trace_is_child_of_provider_request_span` and `::test_event_handling_telemetry_keeps_provider_parentage_without_content` verify child-span parentage and content-free telemetry; runtime emits after advisory, tool, final, and failure handling. |
| EVT-12 | Durable event identity, provider sequence, and SSE sequence remain distinct | PASS | `tests/agent/test_agent_event_runtime.py` checks provider and client sequence metadata; `tests/server/test_agent_event_api.py` verifies durable integer SSE IDs and separately namespaced `delta-*` IDs/client cursors while retaining provider sequence/event IDs as payload fields. |
| EVT-13 | Typed bounded hook points cannot grant authority | PASS | `tests/agent/test_agent_event_runtime.py::test_persisted_hook_and_tool_annotations_are_bounded_json` covers bounded annotations; `tests/agent/test_agent_event_tool_dispatch.py::test_event_dispatch_ignores_capabilities_forged_in_workflow_state` proves dispatch reads only runtime `authority_context`, not hook/workflow state. |
| EVT-14 | Tool calls use host-side catalog, ACL, scope, and effective-capability checks | PASS | `agent/event_tool_dispatch.py::CatalogAuthorizedToolDispatcher` uses a host tool-name allowlist, ACL-enabled scoped `CatalogStore.get`, host-owned capability requirements, trusted `StepContext.authority_context`, and active `ProviderRegistry` dispatch. Tests cover ACL/tenant denial, manifest/state capability spoofing, unmapped tools, and inactive providers. Production application binding remains under EVT-16. |
| EVT-15 | Retry/idempotency policy prevents unsafe repeated side effects | PASS | `NamedProjectionToolCallIdempotency` stores an `executing` marker and completed JSON result through the core named-projection CAS primitive. SQLite reopen and ambiguous-effect tests pass in `tests/agent/test_agent_event_tool_dispatch.py`; `test_event_effect_is_not_repeated_when_process_fails_before_ack` covers a durable effect followed by lane-message requeue and replay. Final-turn and failure callbacks receive stable effect-scoped keys. This is at-most-once invocation, not atomic exactly-once effects; host callbacks must honor their keys, and downstream idempotency/reconciliation remains necessary after ambiguous outcomes. |
| EVT-16 | Separate opt-in start/control/inspect/SSE server surface | PASS | `create_agent_event_router` is an opt-in FastAPI router factory, intentionally not mounted by default. `tests/server/test_agent_event_api.py` verifies start/control/inspect/SSE routes, role/namespace checks, control receipt, durable and transient SSE delivery with distinct IDs, canonical run ID projection, and termination when a terminal service returns only stale events. Application hosts must inject their run service and explicitly include the router; no default app is changed. |
| EVT-17 | User steering, safe-boundary queueing, and provider capability fallback | PASS | `tests/server/test_agent_event_service.py` verifies the concrete `ScheduledEventAgentRunService` binds run ownership, the injected scheduler, run registry, control journal, executor and API contract; control receipt state remains pending until the graph control-point checkpoint/ack test in `tests/agent/test_agent_event_graph_run.py`. Provider steering/cancellation are intersected with host support, so this v1 host queues controls only. Scheduler is in-process and does not recover queued executions after process loss; see ADR-021. |
| EVT-18 | Fake-provider graph, replay, crash, authorization, and transcript tests | PASS | `test_fake_provider_runs_graph_workflow_and_keeps_events_out_of_transcript` covers actual workflow checkpoints, queued control delivery, one-event-per-iteration, causal graph edges, and transcript isolation. `test_event_effect_is_not_repeated_when_process_fails_before_ack` covers duplicate provider delivery, a persisted lane event, durable tool effect, failure before ack, requeue, and restart-style replay with one external effect. Related tests cover stable event identity, ACL/scope denials, trace parentage, and transcript projection. Runtime recovery re-enters the journal handler; the core async workflow runtime itself still does not resume arbitrary interrupted runs from a checkpoint. |
| EVT-19 | Manual real Codex App Server streaming/structured-output smoke | PASS | 2026-10-23 run: `gpt-5.6-luna`, 4764.66 ms, valid structured output, event kinds `field_completed,response_completed,response_started,text_delta`, trace `1b02c44913b842238bd479868e644d80`, response `01a16824-ebb1-76a1-9f35-618353966ec6`. |
| EVT-20 | OpenAI live smoke remains opt-in; no credentials required in CI | PASS | `scripts/smoke_openai_responses_provider.py` requires an explicit API key; missing-credential test confirms the provider fails before network access. No live OpenAI credential was used. |
| EVT-21 | Typing, lint, and supported-runtime checks: PyPy 3.11 source profile; CPython 3.12-3.14 package/runtime matrix | PASS | The 90-test event-agent selection passed on CPython 3.12.13 (60.52s), CPython 3.13.3 (92.43s), CPython 3.14.3 (54.83s), and Linux PyPy 3.11.15 (122.42s); all had 1 deselected and no failures. Pyright on the changed production modules reports 0 errors/warnings/informations for Python 3.11, 3.12, 3.13, and 3.14. Ruff passes on changed production and test files. `tests/core/test_pypy_ci_contract.py`: 10 passed, including the new assertion that the PyPy profile installs `jsonschema`. PyPy 3.11 is source/Python-authority CI only; `rust/Cargo.toml` uses `abi3-py312`, so the published wheel's `requires-python >=3.12` remains intentional. These are local matrix results; hosted GitHub Actions has not yet run on a push of this commit. |
| EVT-22 | Documentation explains lifecycle, traces, recovery, limits, and provider boundaries | PASS | `docs/ADR-021-graph-native-streaming-agent.md` contains architecture diagram, event/recovery contract, authority/provider boundaries, opt-in API caveats, and checklist link. |
| EVT-23 | OpenAI Chat Completions streaming adapter maps SSE chunks to the normalized provider event union | PASS | `kogwistar/agent/openai_chat_completions.py`; `tests/agent/test_agent_streaming_contracts.py::test_chat_completions_builds_strict_schema_request_and_normalizes_split_sse`, plus fail-closed cases. Confirms response identity, normalized sequence, text, finish, usage, API errors, timeout, `[DONE]`, malformed/incomplete stream, and bounded SSE input. |
| EVT-24 | Chat Completions JSON-Schema output is incrementally parsed into completed-field events | PASS | `tests/agent/test_agent_streaming_contracts.py` covers split Unicode/escaped/nested JSON, field completion and final output, invalid/incomplete JSON, duplicate keys, and size limits. The completed object is additionally checked against host-supplied schema under EVT-31 before any field event is emitted. |
| EVT-25 | Chat Completions streamed tool-call fragments assemble safely before normalized dispatch events | PASS | `tests/agent/test_agent_streaming_contracts.py::test_chat_completions_does_not_emit_fragmented_tool_calls_before_done` verifies interleaved fragments are assembled in index order and emitted only after terminal `[DONE]`; malformed/duplicate JSON arguments and incomplete/oversized streams fail closed. Calls are requests only; host authorization remains EVT-14 and host argument-schema validation is verified under EVT-32. |
| EVT-26 | Provider adapter contract fixtures prove parity and edge behavior without credentials | PASS | Fake HTTP/`httpx.MockTransport` tests in `tests/agent/test_agent_streaming_contracts.py` cover arbitrary transport chunk boundaries, Unicode and escapes, interleaved tools, role-only/usage chunks, finish states, API errors, timeouts, missing `[DONE]`, response identity changes, and bounded input. No live credential or network is needed. Codex adapter regression passes separately. |
| EVT-27 | Codex field-event timing is tested against authoritative completed output | PASS | `tests/unit/test_codex_streaming_provider.py::test_codex_adapter_streams_and_parses_complete_structured_output` injects conflicting provisional delta and completed item values; only the authoritative completed item produces `ProviderFieldCompleted` and final structured output. |
| EVT-28 | Advisory hooks run only for explicitly eligible completed fields or provider summaries | PASS | `tests/agent/test_agent_event_runtime.py::test_advisory_hook_routing_requires_opt_in_and_respects_run_budget` covers default/disallowed fields, explicit field opt-in, summaries disabled by default, explicit summary opt-in, and a `reasoning` field not implicitly opting in. |
| EVT-29 | Provider endpoint support is explicit and distinguishes Responses, Chat Completions, and legacy Completions | PASS | `docs/ADR-021-graph-native-streaming-agent.md` documents `/responses` and `/chat/completions`, explicitly excludes legacy `/completions`; `tests/agent/test_agent_streaming_contracts.py` exercises both supported adapters. |
| EVT-30 | Provider-managed tools and unknown semantic events cannot bypass host policy or disappear silently | PASS | `tests/agent/test_agent_streaming_contracts.py::test_responses_rejects_unknown_events_and_provider_managed_tools`, `::test_chat_completions_rejects_unknown_delta_fields`, and `::test_chat_completions_rejects_provider_managed_tool_types` verify failures are surfaced without a successful completion or executable tool event. |
| EVT-31 | Completed assistant structured output is validated against the requested JSON Schema by trusted host code | PASS | `tests/agent/test_agent_streaming_contracts.py::test_openai_adapters_reject_schema_invalid_output_before_field_events` covers required properties, types, enums, and additional properties for both HTTP adapters; `::test_openai_adapters_reject_invalid_host_schema` covers invalid schema rejection. No field event or success event escapes validation failure. |
| EVT-32 | Completed tool arguments are validated against host-owned tool schemas before dispatch | PASS | `tests/agent/test_agent_event_tool_dispatch.py::test_event_dispatch_validates_arguments_against_host_owned_schema`, `::test_event_dispatch_requires_schema_for_every_host_tool`, and `::test_event_dispatch_rejects_missing_extra_or_wrong_type_arguments` verify host schema enforcement and zero invocation on invalid arguments. Schemas are deep-copied at dispatcher construction. |
| EVT-33 | Advisory-hook and continuation cycles are explicitly bounded and cannot self-trigger indefinitely | PASS | `tests/agent/test_agent_event_runtime.py::test_advisory_followup_model_call_is_opt_in_and_bounded`, `::test_advisory_continuation_limit_has_a_hard_ceiling`, `::test_advisory_hook_routing_requires_opt_in_and_respects_run_budget`, and `::test_advisory_hook_clamps_negative_recovered_counters_before_increment`; `tests/agent/test_agent_event_graph_run.py` verifies the stable hook idempotency key. Continuations default to zero, require host configuration, drain the prior response first, and have a hard cap of four. |

## Progress Notes

## Discussion-History Audit (2026-10-23)

- The active implementation checklist already captured Chat Completions parity
  and the Codex authoritative-completion regression. Review of the earlier
  design discussion found three additional semantics that were not explicit:
  constrained advisory-hook eligibility, exact endpoint support (including the
  distinction from legacy Completions), and explicit handling of provider-owned
  tools/unknown event types. These are tracked as EVT-28 through EVT-30 above.
- This audit changes documentation/checklist scope only. It does not claim these
  three requirements are implemented or verified.

- A follow-up implementation adds trusted JSON Schema validation for completed
  model output and host-bound tool argument schemas. EVT-31/32 are now verified
  by parity and no-invocation regressions; provider-side `strict` remains a
  request hint rather than an enforcement boundary.
- The earlier post-think/Whisper discussion also needs a hard bound on any
  future hook-to-follow-up-model cycle. Added EVT-33 to require explicit hook
  eligibility, per-turn limits, and idempotent continuation semantics.

## Follow-Up Implementation (2026-10-24)

- Advisory hooks now require host allowlisting for completed field names;
  provider reasoning summaries additionally require an explicit host flag.
  A field called `reasoning` receives no special treatment. Hook count is
  bounded per run, and negative recovered counters are clamped safely.
- The host can opt into a follow-up model request after an eligible hook, but
  only after the current provider response is drained and validated. The
  default continuation budget is zero and the hard maximum is four. This is
  not live steering; current OpenAI adapters do not advertise steering.
- Both OpenAI adapters validate final structured output with `jsonschema`
  before emitting field-completed events or response success. The authorized
  tool dispatcher requires host-owned argument schemas and validates them
  before invoking the tool provider.
- Responses unknown event/output-item types and Chat Completions unknown delta
  or provider-managed tool types now fail closed. Legacy `/completions` remains
  unsupported.
- Verification on CPython 3.13: focused provider/runtime/dispatcher/graph suite
  **70 passed, 4 warnings**; Ruff passed; Pyright reported **0 errors, warnings,
  or informations** for changed production files. The run used
  `-p no:cacheprovider` and was performed outside the restricted Windows
  sandbox because HTTPX's async stream fixtures require Windows loopback
  initialization. Hosted CI has not been run by this change.

## Verification Run (2026-10-23)

- Current expanded streaming-agent regression suite: **66 passed, 10 warnings**
  on CPython 3.13 in 143.38s. It covered streaming contracts, event
  journal/runtime, durable tool dispatch, full graph run, goal-mode regression,
  REST route contract, and transcript projection. The run used
  `-p no:cacheprovider`; the Windows sandbox can block asyncio loopback
  socket-pair creation for network-backed tests.

- Kogwistar focused suite: 20 passed on CPython 3.13 with
  `python -m pytest -p no:cacheprovider tests/agent/test_agent_streaming_contracts.py tests/agent/test_agent_event_journal.py tests/agent/test_agent_event_runtime.py tests/server/test_agent_event_api.py -q`.
- LLM-Wiki Codex adapter suite: 3 passed on CPython 3.13.
- Kogwistar run-scoped event-to-step graph-link test: 1 passed.
- Bounded hook/tool-result projection test passed; persisted tool output is now
  constrained to a validated JSON object within the same 6,000-character cap.
- Existing async resolver regression tests for synchronous and awaited handlers:
  2 passed; `AsyncMappingStepResolver.register` now has typed sync/async handler
  overload semantics so async event steps are not hidden behind a sync-only
  registration contract.
- Ruff checks passed for all changed LLM-Wiki files and all changed Kogwistar
  implementation/test files under their respective repository configurations.
- `compileall` passed for the Codex adapter and both smoke scripts.
- Real signed-in Codex App Server smoke passed; see EVT-05 and EVT-19.
- Root LLM-Wiki Codex adapter tests reached **3 passed** in 0.34s and its
  changed-file Ruff check passed. The pytest process did not exit promptly
  after reporting success and was interrupted; therefore this is test-case
  evidence, not a clean process-exit verification. Earlier adapter smoke and
  runtime evidence remain recorded under EVT-05/EVT-19.
- Durable lane-message crash-window replay regression passed on CPython 3.13:
  duplicate provider delivery reuses one message ID; a simulated process loss
  after a durable tool effect but before acknowledgment requeues the message;
  replay returns the stored effect result and invokes the side effect once.
- Requeue diagnostics now persist only an error code and exception type, not
  arbitrary exception text that could contain prompts or credentials.
- Terminal SSE streaming now stops if a service returns only malformed or
  stale sequence entries, and does not accept booleans as sequence numbers.
  Both cases are covered by the route tests.
- Final assistant-turn and failure-record callbacks now receive stable,
  effect-scoped idempotency keys so hosts can prevent duplicates if the event
  is replayed after the effect but before acknowledgment.
- The existing `ChatRunService` runs the separate conversation/RAG and ordinary
  workflow patterns; it is intentionally not treated as an event-agent host
  without an explicit product-level composition decision, to avoid merging
  their conversation/run semantics.

## Control Negotiation Follow-up (2026-10-23)

- Added per-run provider capability negotiation at the REST control boundary.
  A request for unsupported live steering or cancellation is downgraded to
  queueing; the run service must return the same effective mode or the route
  rejects the response. This prevents the API from representing unsupported
  provider behavior as successful steering.
- The control-mode resolver unit test passes. Route-level TestClient checks
  could not be completed in this Windows session: the test process stalls while
  AnyIO/Starlette creates its Proactor event loop socketpair, before entering
  the ASGI route. The control endpoint itself is now invoked directly in a
  route-contract test, including a mismatched receipt rejection. The
  network-backed ASGI/SSE transport test remains unverified here. This is an
  environment limitation, not a passing transport test.
- Added a focused transcript projection regression proving the current
  conversation transcript includes user/assistant turns but filters a durable
  provider-event node. This is a projection unit test, not an end-to-end graph
  persistence test.
- Tool idempotency keys now include provider and response identity as well as
  run and call ID, preventing call-ID reuse in a later provider response from
  being mistaken for a retry. The unit test verifies stable replay and
  cross-response/provider isolation. Host-side deduplication remains required.
- Idempotency remains an explicit host-dispatch contract, not a core guarantee:
  durable external effects cannot be made exactly-once by a workflow checkpoint
  alone. The host must use the supplied stable key with its own idempotency
  support or fail closed on unsafe replay.
- Added `CatalogAuthorizedToolDispatcher` for trusted in-process tools. Host
  configuration maps model-visible names to catalog IDs and required
  capabilities; dispatch performs ACL/scope, runtime-authority, provider
  lifecycle, JSON-output, and required-idempotency checks. Adversarial tests
  cover denial and replay; external MCP execution remains a separate boundary.
- Event-handler telemetry creates a child span from the persisted provider
  request span, recording event identity/kind and workflow coordinates only.
  Tests confirm parentage and that reasoning text is not included.
- Durable event IDs now derive from provider/response/event coordinates rather
  than workflow step and attempt, so a provider event replayed by a later
  workflow attempt reuses the same message idempotency key. Providers omitting
  both event ID and sequence only receive attempt-scoped deduplication; equal
  payloads are not collapsed because they may be distinct legitimate events.
- Sync-provider compatibility regression passes; fixture `SequenceFakeModel`
  now declares the provider identity required by the pre-existing protocol.
- Added a named-projection-backed durable tool-call executor. It persists an
  `executing` claim before invoking the side effect and stores the JSON result
  afterward. Reopen tests use SQLite metadata; an unfinished claim after a
  simulated post-effect failure is rejected on retry rather than invoked a
  second time. The implementation does not claim transactional exactly-once
  behavior across a separate tool's effect and Kogwistar metadata commit.
- The full fake-provider event workflow integration test passed on CPython
  3.13 outside the sandbox: three provider semantic events each produced one
  receive execution step, three durable lane messages completed with causal
  run/step links, and transcript projection returned only user/assistant turns.
- Control lifecycle receipt regression passes in
  `tests/agent/test_agent_event_graph_run.py` and
  `tests/server/test_agent_event_api.py`: queued control is pending before the
  graph handles it, committed after checkpoint/ack, and its receipt can be
  queried through the authenticated opt-in API route. Control idempotency is
  scoped by run, sender, and client control ID, so two users reusing the same
  control ID do not collide. Unknown receipt IDs return 404. Focused run:
  **7 passed** on CPython 3.13 in 44.49s.
- A concrete opt-in `ScheduledEventAgentRunService` now binds the API to the
  run registry, existing scheduler protocol, control journal, host authorization,
  user-turn creation, provider capability lookup and event-workflow executor.
  Focused service/API/graph run: **13 passed** on CPython 3.13 in 39.76s.
  The service does not alter `ChatRunService`; the host still owns conversation
  authorization, graph execution and idempotent user-turn creation. The existing
  scheduler is in-process, so run/event records are durable but queued execution
  recovery after process loss is not implemented.
- Durable semantic publication and transient token-delta delivery are now
  separate typed callbacks (`EventAgentPublishCallbacks`). A regression proves
  token text goes to a bounded process-local buffer/optional host transient sink
  and is absent from the durable run-event log. The opt-in REST `/events` route
  multiplexes durable events and transient deltas with separate SSE IDs and
  cursors; a gap event reports delta-buffer eviction. Tests cover the bounded
  buffer and gap projection. Deltas may be lost on restart or run-buffer eviction
  by design.
- Latest broad event-agent regression run: **45 passed, 3 warnings** on CPython
  3.13 in 47.18s, covering service, API/SSE, graph execution, runtime, journal,
  streaming contracts and tool dispatch. Final service/API rerun: **12 passed,
  1 warning** in 1.27s. Both ran outside the sandbox because of Windows
  asyncio/ASGI socket restrictions. Ruff passed and Pyright reported 0 errors,
  0 warnings and 0 informational diagnostics for the changed production files.
- All seven event-agent focused test modules now carry the repository `ci`
  marker. `pytest --collect-only -m ci` confirmed selection; the final exact
  CI-marked run passed **49 tests, 3 warnings** in 48.41s on CPython 3.13.
  It includes bounded-buffer, SSE injection, scheduler-exception, async-sink,
  API, graph, runtime, journal, streaming, and tool-dispatch regressions.
  Ruff passed and Pyright reports 0 errors/warnings for changed production files.
- After tightening the run executor to an async contract and awaiting optional
  async transient sinks, the service/API subset passed **16 tests, 1 warning**
  in 1.39s on CPython 3.13.

## Runtime Matrix Completion (2026-10-23)

- The seven event-agent test modules, selected with the repository `ci` marker
  and exclusions used by the provider-free core CI lane, passed on CPython
  3.12.13 (**49 passed, 4 warnings, 51.77s**), CPython 3.13.3 (**49 passed,
  3 warnings, 48.41s**), and CPython 3.14.3 (**49 passed, 6 warnings,
  48.73s**).
- The same test selection passed in a Linux PyPy 3.11 container
  (**49 passed, 4 warnings, 125.23s**; PyPy 7.3.23 / Python 3.11.15). The
  container used `KOGWISTAR_IMPL_MODE=python` and the PyPy experimental
  requirements, matching the provider-free PyPy job's implementation mode.
- The LLM-Wiki Codex adapter fixture suite completed cleanly on CPython 3.13.3
  outside the restricted Windows sandbox: **3 passed in 12.69s**. The sandbox
  itself blocks the Windows Proactor loop's socket-pair initialization before
  the test body; this is why that focused check was run outside it.
- Ruff passed on the changed Kogwistar event-agent production and test files.
  Pyright reported **0 errors, 0 warnings, 0 informations** with target Python
  versions 3.11, 3.12, 3.13, and 3.14.
- These are local checks, not hosted GitHub Actions results. No push was made,
  so repository CI has not been triggered by this verification.
- Scope correction as of 2026-10-23: the goal had been marked complete against
  the narrower written plan, but that plan omitted the earlier-discussed Chat
  Completions adapter. EVT-23 through EVT-27 were implemented then; EVT-28
  through EVT-33 were follow-up requirements and were not complete at that
  point. They are now implemented and verified in the 2026-10-24 section below.
  The current
  Kogwistar package metadata requires Python >=3.12; earlier PyPy 3.11 run
  evidence is historical and should not be interpreted as current package
  support without reconciling that metadata with the repository compatibility
  policy.

## Provider-Parity Verification (2026-10-23)

- The focused Kogwistar streaming/event workflow set passed **31 tests** on
  CPython 3.12.13 (45.48s), CPython 3.13.3 (69.16s), and CPython 3.14.3
  (40.84s). This set covers provider contracts, event runtime, graph-run
  integration, replay, and transcript isolation.
- LLM-Wiki Codex provider regression passed **3 tests** on CPython 3.12, 3.13,
  and 3.14. The conflicting provisional/completed output test proves fields
  come from the completed item only. The tests used fake local sessions and no
  credentials. Windows loopback restrictions required running these focused
  tests outside the restricted sandbox.
- Ruff passed for the adapter and its Kogwistar test module under the core
  project configuration, and separately for the root Codex provider regression.
  Pyright passed with 0 diagnostics for the new Chat Completions adapter and
  separately for the root Codex streaming provider. A combined invocation over
  test files was not a valid typecheck because its interpreter lacked the test
  and root package import paths.
- These are local checks, not hosted CI results. No push was made and no GitHub
  Actions result is claimed.

## Final Runtime Matrix (2026-10-24)

- Added `jsonschema>=4.21,<5` to the PyPy 3.11 source-profile requirements,
  because importing the event-agent provider path loads host-side JSON Schema
  validation. `tests/core/test_pypy_ci_contract.py` now prevents the CI profile
  from omitting this runtime dependency.
- The 90-test event-agent selection passed on CPython 3.12.13 (**90 passed,
  1 deselected, 60.52s**), CPython 3.13.3 (**90 passed, 1 deselected,
  92.43s**), CPython 3.14.3 (**90 passed, 1 deselected, 54.83s**), and Linux
  PyPy 3.11.15 (**90 passed, 1 deselected, 122.42s**). The PyPy test ran with
  `KOGWISTAR_IMPL_MODE=python`, matching the hosted Python-authority profile.
- The PyPy CI contract module passed **10 tests**. Ruff passed on all changed
  event-agent production and test files. Pyright reported **0 errors, 0
  warnings, 0 informations** for the changed production modules at each target
  Python version 3.11, 3.12, 3.13, and 3.14.
- PyPy 3.11 remains a source-level CI target, not a published native-wheel
  target. The Rust extension is built with `abi3-py312`; therefore the package
  metadata's `requires-python >=3.12` is deliberately retained.
- These are local runtime-matrix results. Hosted GitHub Actions still needs to
  run after the commit is pushed; no remote CI result is implied here.
