# ARD: ContextSnapshot and PromptContext

## Status
Accepted (implemented; coverage still expanding).

## Last Updated
2026-03-08

## Context
We currently have multiple overlapping notions of "context":

- **ContextSources / ConversationContextView**: a runtime-generated view of items and messages gathered from the conversation graph (useful for debugging and prompt construction).
- **Evidence pack**: a materialized JSON payload derived from selected KG nodes, used for claim-level citations and answer generation.
- **ContextSnapshot**: intended to be a persisted snapshot of what actually went into the LLM prompt window for a specific step.

The previous implementation of ContextSnapshot only hashed `view.messages` and stored minimal metadata. It did not persist the *actual LLM inputs* (system prompt, question, evidence-pack parameters), and did not provide a rehydratable description of the evidence pack.

## Current Implementation Note (2026-03-08)

The current repo implements the main elements in this ARD:

- `kogwistar/conversation/conversation_context.py`
  defines `PromptContext` and keeps `ConversationContextView` as a
  backwards-compatible alias.
- `kogwistar/conversation/service.py` persists context
  snapshots with deterministic IDs, prompt messages,
  `llm_input_payload`, `evidence_pack_digest`, and `depends_on` edges.
- `kogwistar/conversation/resolvers.py` exposes a
  `context_snapshot` workflow op.
- `kogwistar/conversation/agentic_answering.py` persists
  snapshots around multiple agentic answering stages and supports
  evidence-pack rehydration.

What is still expanding is coverage consistency across all legacy answer
paths; snapshotting is implemented, but not every older call path is
guaranteed to snapshot automatically. Old code path serve as an introductory usage and snapshot complicates the flow to first time users.

## Decision
1. Rename **ConversationContextView → PromptContext** and document it explicitly as an LLM-facing debug/telemetry artifact.
   - Keep `ConversationContextView` as an alias for backwards compatibility.

2. Upgrade **ContextSnapshot** persistence so it captures the *actual LLM inputs*:
   - Persist `prompt_messages` (normalized role/content list)
   - Persist `llm_input_payload` (e.g. system prompt, question, answer text)
   - Persist `evidence_pack_digest` (parameters sufficient to rebuild the evidence pack)

   These are stored in `ConversationNode.properties` (nested JSON allowed) while keeping `metadata` flat/Chroma-friendly.

3. Introduce **EvidencePackDigest** (Pydantic model) as the canonical rehydratable description of an evidence pack.
   - Store materialization parameters and `evidence_pack_hash`.
   - Rehydration is best-effort; hash comparison can detect drift.

4. Implement **rehydration**:
   - `AgenticAnsweringAgent.rehydrate_evidence_pack_from_digest()` rebuilds an evidence pack from a persisted digest.

5. Provide a lightweight workflow op:
   - `default_resolver` now includes a `context_snapshot` op that persists a snapshot for the current conversation. It is best-effort unless callers provide stable `run_id/run_step_seq`.

## Consequences
- ContextSnapshot nodes are now self-describing and usable for audit/replay/debug.
- Evidence-pack rehydration is possible and drift-detectable.
- PromptContext is clarified as the thing that can be snapshotted (not the entire conversation graph).

## Non-goals
- Full determinism against a mutable KG is not guaranteed.
- This ARD does not redesign how evidence selection/materialization works; it only makes the inputs observable and replayable.
