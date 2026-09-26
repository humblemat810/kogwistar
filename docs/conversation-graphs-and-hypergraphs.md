# Conversation Graphs and Hypergraphs

A conversation is not just a list of messages.

It is a graph of turns, summaries, snapshots, tool calls, and references. Once you model it that way, a lot of the system becomes easier to explain: ordering is explicit, provenance is explicit, and context is not hidden in a string buffer.

The useful mental model is simple:

- nodes are conversation artifacts
- edges are relationships between artifacts
- metadata carries role, turn order, and causal intent

In this repository, that means a user turn, an assistant turn, a summary node, and a context snapshot are all first-class graph objects. The graph is not only a storage format. It is the structure the runtime uses to explain what happened.

## What Belongs In The Conversation Graph

A conversation graph usually contains a few different kinds of nodes.

Turn nodes represent user, assistant, system, or tool turns. Summary nodes represent compressed history. Context snapshot nodes represent the exact prompt context that was assembled for an LLM call. Pointer or pin nodes represent references to knowledge that was pulled into the conversation.

The relationships between those nodes are just as important:

- `next_turn` connects the main turn chain
- `summarizes` connects a summary to the turns it compresses
- `depends_on` connects a snapshot to the items that shaped it
- `references` or pin edges connect a response to supporting evidence

That is already enough structure to explain most of the conversation behavior in the system.

## A Minimal Conversation Graph

The repository uses typed conversation nodes and edges on top of the shared graph substrate. A simple turn chain can look like this:

```python
from kogwistar.conversation.models import ConversationNode, ConversationEdge

user_turn = ConversationNode(
    id="conv|42|turn|user-1",
    type="entity",
    label="User turn",
    summary="What did we decide about graph modeling?",
    role="user",
    turn_index=1,
    conversation_id="conv|42",
    user_id="user|7",
    level_from_root=0,
    metadata={
        "entity_type": "conversation_node",
        "role": "user",
        "turn_index": 1,
        "conversation_id": "conv|42",
        "user_id": "user|7",
        "in_conversation_chain": True,
        "in_ui_chain": True,
    },
)

assistant_turn = ConversationNode(
    id="conv|42|turn|assistant-1",
    type="entity",
    label="Assistant turn",
    summary="We decided to model the conversation as a graph.",
    role="assistant",
    turn_index=2,
    conversation_id="conv|42",
    user_id="user|7",
    level_from_root=0,
    metadata={
        "entity_type": "conversation_node",
        "role": "assistant",
        "turn_index": 2,
        "conversation_id": "conv|42",
        "user_id": "user|7",
        "in_conversation_chain": True,
        "in_ui_chain": True,
    },
)

turn_edge = ConversationEdge(
    id="conv|42|edge|next_turn|1",
    type="relationship",
    label="next_turn",
    relation="next_turn",
    source_ids=[user_turn.id],
    target_ids=[assistant_turn.id],
    summary="Main conversation chain",
    metadata={
        "entity_type": "conversation_edge",
        "causal_type": "chain",
        "conversation_id": "conv|42",
    },
)
```

The point of this shape is not just that it is tidy. It makes the chain inspectable. You can ask for the previous turn, the next turn, or the summary node that compresses a stretch of history, and the answer is carried by edges rather than inferred from array position.

## Modeling Context As Graph

The most useful part of the conversation model is not the turn chain itself. It is the way the system models context around a turn.

When the runtime prepares an LLM call, it can persist a context snapshot node that records what the model saw. That snapshot can then point back to the exact conversation items and evidence items that were included.

```python
from kogwistar.conversation.models import ContextSnapshotMetadata

snapshot_meta = ContextSnapshotMetadata(
    run_id="run_123",
    run_step_seq=18,
    attempt_seq=0,
    stage="answer",
    model_name="gpt-5",
    budget_tokens=4096,
    tail_turn_index=24,
    used_node_ids=[
        "conv|42|turn|user-1",
        "conv|42|summary|head",
        "kg|entity|graph-substrate",
    ],
    rendered_context_hash="sha256:abc123",
)
```

That is a better model than "the prompt was assembled somewhere." It tells you which items participated in the context, which stage used them, and how to reconstruct the evidence later.

The same idea applies to memory pinning. A pin is not just a cached string. It is a graph artifact that says "this piece of knowledge was made relevant to this conversation at this point in time."

## Why Hypergraph Thinking Helps

Conversation rarely stays pairwise.

A single answer can depend on:

- the current user turn
- a previous summary
- a retrieved knowledge item
- a tool result
- a policy decision

A simple directed graph is enough to show the chain of turns, but it becomes awkward when one artifact needs to point to several inputs at once. Hypergraph thinking makes that dependency structure explicit. Instead of flattening the relationship into a single arrow, you keep the full context of the relationship intact.

That matters for conversation because the important unit is often not a message by itself. It is the combination of items that shaped the response.

## Why This Is Not The Same As Workflow Design

This distinction matters, and it is easy to get wrong.

Workflow design and conversation graphs are both graph-shaped, but they solve different problems.

Workflow design is about control flow:

- which step runs next
- which branches can fan out
- where joins happen
- what happens on suspend or resume

Conversation graphs are about domain history:

- which turn happened when
- what context was shown
- what evidence was pinned
- what summary compresses what span of conversation

If you collapse those two models, the meaning gets blurry fast. A workflow edge is not the same thing as a `next_turn` edge. A join node is not the same thing as a summary node. A suspended workflow token is not the same thing as a conversation artifact.

The repository keeps them separate because they need different invariants. The workflow design graph needs routing and execution semantics. The conversation graph needs provenance, ordering, and replayable history. One describes how the system should run. The other records what the system said and saw.

## Practical Pattern

If you want to model a conversation well, start with these questions:

1. What are the stable nodes?
2. What relationships must be explicit?
3. What context should be replayable later?
4. Which artifacts belong to the conversation graph, and which belong to the workflow that produced it?

A good conversation graph usually includes:

- the main turn chain
- summaries over turn ranges
- context snapshots before model calls
- evidence pins and pointer nodes
- tool call and tool result artifacts

That is enough structure to support replay, auditing, and better retrieval without turning the conversation model into a workflow engine.

## Closing

The useful shift is not "messages become graph objects."

It is "conversation becomes inspectable structure."

Once that happens, you can reason about memory, context, citations, and conversation history with the same tools you use for the rest of the graph substrate.
