# 05 Context Snapshot and Replay

Audience: Builder / integrator
Time: 20-25 minutes
Companion: [scripts/tutorial_sections/05_context_snapshot_and_replay.py](../../scripts/tutorial_sections/05_context_snapshot_and_replay.py)

## What You Will Build

You will assemble a prompt view from conversation state, persist it as a `context_snapshot`, and inspect the snapshot payload and dependency edges.

## Why This Matters

This is where the architecture becomes serious. Instead of saying "the model saw some context," the repo can persist a replayable artifact that records which items were packed and what the final prompt payload looked like.

## Run or Inspect

- Run the companion file to build a view with `ConversationService.get_conversation_view(...)` and persist it with `persist_context_snapshot(...)`.
- Read `graph_knowledge_engine/docs/ARD-context-snapshot-and-prompt-context.md` for the broader design.
- Inspect the snapshot-related tests in `tests/kg_conversation/test_phase2b_context_snapshot.py`.

## Inspect The Result

- Confirm a `context_snapshot` node exists for the conversation.
- Inspect the snapshot metadata for run id, stage, used node ids, and token cost.
- Inspect the `depends_on` edges from the snapshot to the items it captured.

## Invariant Demonstrated

Prompt construction is replayable. The system can persist what the model actually saw instead of only preserving the final answer.

## Next Tutorial

Continue to [06 First Workflow](./06_first_workflow.md).
