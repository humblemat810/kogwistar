# One Demo, Three Proofs

This demo is the fastest way to understand the repo's value:

it shows that workflow execution, memory, conversation, and provenance can live in one connected graph instead of being split across separate systems.

Run it with:

`python -m kogwistar.demo.graph_native_artifact_demo --summary-only`

Use the full report only if you want supporting IDs:

`python -m kogwistar.demo.graph_native_artifact_demo`

## Why You Should Care

If you are building an AI system, this matters because you usually end up stitching together:
- a workflow runner
- a memory store
- a conversation store
- logs or traces for debugging

This demo shows a simpler model:
- execution writes graph data
- that graph data becomes memory
- conversation links into the same graph
- provenance can explain results later

That reduces glue code and makes later retrieval and explanation much easier.

## What The Demo Does

The workflow is intentionally small:

`ingest -> validate -> normalize -> link -> commit`

It processes three mock notes such as:
- `Vendor invoice`
- `Budget check-in`
- `Team meeting`

The important part is not the note task itself. The important part is what gets stored while the task runs:
- the source note documents
- the workflow run
- each workflow step
- the conversation turn
- the committed note artifacts
- the grounding spans on those artifacts
- the relationships between them

The demo also produces assistant-style answers that look like LLM output:
- a short final answer
- a grounded evidence trail
- confidence and citation-like fields

Those responses are deterministic in this demo, but they are shaped so a real LLM call can replace the current logic later.

## Proof 1: Execution Becomes Memory

The first run processes:

```json
["note-1", "note-2", "note-3"]
```

The second run then skips known work:

```json
second_run_skipped_as_known: ["note-2", "note-3"]
second_run_processed: ["note-4"]
```

What happened:
- the first run committed graph artifacts for the notes
- the second run queried those artifacts as memory
- already-known note IDs were skipped

What this means:

Past execution is not just a trace you inspect manually. It becomes structured memory that changes future behavior.

## Proof 2: Conversation And Workflow Are Unified

The demo stores a simple request:

```json
question: "Organize my notes"
```

And shows that one query can see:

```json
conversation_nodes: 2
workflow_run_nodes: 1
workflow_step_nodes: 6
artifact_nodes: 3
```

What happened:
- the user and assistant turns were stored as conversation nodes
- the workflow run was linked to that conversation
- the workflow steps and committed artifacts were stored in the same graph

What this means:

Conversation is not separate application state wrapped around a workflow. It is part of the same connected system.

## Proof 3: Provenance Can Answer "Why"

The demo asks:

```json
question: "Why did we move note-1 to finance?"
```

And answers from stored history:

```json
answer: "Note note-1 moved to finance/ because its committed artifact node is grounded to 'Vendor invoice' ..."
evidence_steps: ["normalize", "link", "commit"]
grounding_excerpt: "Vendor invoice"
```

What happened:
- the system first built a model-shaped provenance response
- the response included answer text, citations, and confidence
- the system found the committed artifact for `note-1`
- it followed the artifact's stored grounding span back to the source note document
- it traced that artifact back to the run that created it
- it read the stored step history for that run
- it built the explanation from both the grounding span and the stored execution evidence

What this means:

The system can explain results from provenance already in the graph, in an assistant-like format. It does not need to guess, recompute, or rely on an external log search.

## What Provenance Means In This Demo

This demo now shows provenance at two levels at once.

This demo **does** trace:
- artifact node -> grounding -> span -> source document
- artifact node -> workflow run
- workflow run -> workflow step executions
- stored step outputs -> explanation

In other words, the answer to "why did note-1 move to finance?" is built from:
- the committed artifact node for `note-1`
- the grounding span on that node:
  - `doc_id = doc|graph_native_artifact_demo|note-1`
  - `excerpt = "Vendor invoice"`
  - `start_char = 7`
  - `end_char = 21`
- the `source_run` stored on that artifact
- the stored `normalize`, `link`, and `commit` step records for that run
- the model-shaped response fields that present the result like a grounded assistant answer

That distinction matters:
- execution provenance answers: "which run and which steps produced this result?"
- span-grounded provenance answers: "which exact source text span grounded this node or edge?"
- assistant-shaped output answers: "how would an LLM present the grounded conclusion?"

This demo proves both together for one artifact.

## What The Repo Can Support Beyond This Demo

The broader repo has the richer provenance model that this demo is now touching directly:
- mentions / groundings on nodes and edges
- span objects with offsets and excerpts
- grounding-to-span associations

This demo still keeps the setup intentionally small:
- one source document node per note
- one grounded artifact node per normalized note
- one workflow run with inspectable step executions

So the honest reading is:
- the repo supports provenance as a first-class model
- this demo now shows span-grounded traversal in a minimal way
- deeper variants are still possible, but not needed to make the point

## Why This Is Different From Typical Agent Frameworks

Typical systems often look like this:

`execution -> logs`

This demo shows:

`execution -> graph -> memory -> grounded reasoning`

That is the value proposition.

You are not only getting a workflow runner. You are getting a substrate where:
- execution data is reusable
- conversation is linkable
- provenance is queryable

## How To Read The Output

The output has two parts:
- `summary`: the one-screen proof
- `details`: the supporting IDs and evidence

Use `summary` first. It is enough to answer:
- did past execution affect a later run?
- are conversation and workflow linked?
- can the system explain why a result happened?

Use `details` only when you want to inspect the supporting graph objects.

## Final Takeaway

This demo does not try to show a smarter agent loop.

It shows a more useful system model:

execution, memory, conversation, and provenance are all part of the same graph, so the system can reuse what happened instead of only recording it.
