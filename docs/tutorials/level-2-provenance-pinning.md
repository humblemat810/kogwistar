# RAG Level 2: Reinforced Provenance and Pinning

Goal: materialize inspectable evidence references in the conversation graph.

## What You Will Build

You will create a user turn, select evidence, and project that evidence into the conversation graph as pointer nodes and reference edges.

## Why This Matters

This is where the repo's provenance story becomes visible. Instead of only saying "the answer used these facts," the system writes graph artifacts you can inspect later.

## Run or Inspect

## Quick Run

```powershell
python scripts/rag_tutorial_ladder.py level2 `
  --data-dir .gke-data/tutorial-ladder `
  --question "Show evidence and provenance for retrieval decisions." `
  --max-retrieval-level 2
```

Equivalent workflow-driven v2 path:

```powershell
python scripts/rag_tutorial_ladder.py level2b `
  --data-dir .gke-data/tutorial-ladder `
  --question "Show the equivalent provenance flow through add_turn_workflow_v2." `
  --max-retrieval-level 2
```

Live LLM answer with Gemini:

```powershell
python scripts/rag_tutorial_ladder.py level2b `
  --data-dir .gke-data/tutorial-ladder `
  --question "Answer from the collected evidence pack." `
  --max-retrieval-level 2 `
  --llm-provider gemini `
  --llm-model gemini-2.5-flash
```

Live LLM answer with OpenAI:

```powershell
python scripts/rag_tutorial_ladder.py level2b `
  --data-dir .gke-data/tutorial-ladder `
  --question "Answer from the collected evidence pack." `
  --max-retrieval-level 2 `
  --llm-provider openai `
  --llm-model gpt-4.1-mini
```

Live LLM answer with Ollama:

```powershell
python scripts/rag_tutorial_ladder.py level2b `
  --data-dir .gke-data/tutorial-ladder `
  --question "Answer from the collected evidence pack." `
  --max-retrieval-level 2 `
  --llm-provider ollama `
  --llm-model qwen3:4b
```

On the first Ollama run, the model may need to be pulled locally. The tutorial
script prints a note before invocation, and Ollama's own download progress is
shown in the terminal when a pull is required.

If you want the smallest local option, swap the model to `phi4-mini`:

```powershell
python scripts/rag_tutorial_ladder.py level2b `
  --data-dir .gke-data/tutorial-ladder `
  --question "Answer from the collected evidence pack." `
  --max-retrieval-level 2 `
  --llm-provider ollama `
  --llm-model phi4-mini
```

Expected output fields:

- `"pinned_kg_pointer_node_ids"`: non-empty
- `"pinned_kg_edge_ids"`: non-empty
- `"checkpoint_pass": true`
- `"llm_provider"`: `deterministic`, `gemini`, `openai`, or `ollama`
- `"assistant_text"`: the visible assistant reply from the v2 path
- `"assistant_turn_node_id"`: non-empty in `level2b`

## Inspect The Result

- Inspect the conversation graph for the new `reference_pointer` nodes.
- Confirm the turn is linked to those pointers by `references` edges.
- Compare this with [04 Conversation Graph Basics](./04_conversation_graph_basics.md), [11 Build a Mini GraphRAG App](./11_build_a_mini_graphrag_app.md), and [15 Historical Search With Tombstone and Redirect](./15_historical_search_tombstone_redirect.md).

## Inside The Engine

- Creates a user turn via `ConversationService.add_conversation_turn(..., add_turn_only=True)`.
- Retrieves memory and KG candidates with deterministic filtering.
- Calls:
  - `MemoryRetriever.pin_selected(...)`
  - `KnowledgeRetriever.pin_selected(...)`
- Writes `reference_pointer` nodes and `references` edges for replay and audit.

The `level2b` variant demonstrates the equivalent behavior through
`ConversationService.add_turn_workflow_v2(...)`. That path is wrapped in the
workflow/runtime operation registry primitive, so the provenance pinning is
performed as part of the workflow-driven v2 conversation execution rather than
as a direct local pinning sequence.

By default `level2b` stays deterministic so the tutorial smoke tests remain
stable. When you pass `--llm-provider`, the same v2 workflow path still runs,
but the tutorial answer step switches to a live model over the assembled
conversation and pinned-evidence context. This gives you a local development
path that moves from inspectable provenance to an actual evidence-grounded
assistant reply without switching to REST.

The deterministic `answer_only` harness used in the tutorial is intentional.
It is the explicit local "use the current conversation view plus assembled KG
evidence to answer now" path, not an accidental fallback. In other words, it
lets you exercise the knowledge-aware answer step without explicitly depending
on the full workflow-driven conversation-graph response materialization flow.

## Checkpoint

Pass when:

- Pointer nodes are created for selected evidence.
- Reference edges link the turn to those pointers.
- In `level2b`, the assistant turn is also materialized through the v2 workflow path.

## Invariant Demonstrated

Used evidence becomes part of the conversation graph. Provenance is explicit, not reconstructed later from guesswork.

## Troubleshooting

- If pinning is empty, verify Level 1 passes first.
- If conversation artifacts look inconsistent, rerun `reset` + `seed`.
- Keep `--data-dir` identical across levels.
- `--llm-provider gemini` needs `GOOGLE_API_KEY` plus the Gemini dependency.
- `--llm-provider openai` needs `OPENAI_API_KEY` plus `langchain-openai`.
- `--llm-provider ollama` needs a running Ollama server plus `langchain-ollama`.
- First-time Ollama runs may spend a while downloading the selected model.

## Next Tutorial

Continue to [RAG Level 3 - Event-Sourced Loop Control](./level-3-event-loop-control.md) or return to [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md).
