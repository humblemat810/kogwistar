The actual implementation of the default conversation logic in `service.py`, `conversation_orchestrator.py`, and `agentic_answering.py`. 

While the underlying engine supports CRUD-like projections (via the `LifecycleSubsystem`), the **conversation implementation itself is strictly Append-only (Create-Read only)**.

### 1. No Deletions or Soft-Deletes
In the high-level conversation modules, there are **no calls** to `tombstone_node` or `tombstone_edge`. Even when a workflow is cancelled, the system doesn't delete the previous steps; instead, it appends a `WorkflowCancelledNode` to the graph to record the event.

### 2. No In-place Updates
There is **no logic** in the conversation orchestrator to update the content of an existing turn. If the agent needs to refine an answer or summarize history, it creates **new nodes** (e.g., `ConversationNode` for summaries or `ContextSnapshotNode`) and links them to the existing ones.

### 3. Identity and Idempotency (CR Semantics)
The implementation follows a **Create-if-missing** pattern:
*   It uses deterministic `pointer_id` generation based on the conversation context.
*   It checks for node existence (`eng.backend.node_get`) before adding.
*   If a node exists, it is reused. If not, it is created.
*   Once created, conversation nodes are treated as immutable records of a specific point in the interaction.

### 4. Comparison by Domain
The repo effectively uses different mutation semantics for different parts of the unified graph:

| Domain | Mutation Semantic | Logic |
| :--- | :--- | :--- |
| **Conversation** | **Append-only (CR)** | Every turn, tool-call, and trace step is a new immutable node. The history is a growing chain. |
| **Workflow Run** | **Append-only (CR)** | Every execution step is a new node. Checkpoints are added sequentially. |
| **Knowledge Graph** | **CRUD-capable** | Uses the `LifecycleSubsystem` to allow for merging (Redirects) and invalidating facts (Tombstones). |

### Summary
The conversation implementation is designed as a **permanent ledger** of the interaction. It prioritizes auditability and provenance over mutability. Even though the engine *could* perform CRUD operations, the conversation "business logic" intentionally restricts itself to an **Append-only (CR)** model to ensure the integrity of the agent's memory and execution trace.

I have completed the analysis of the conversation graph's mutation semantics. Please toggle to **ACT MODE** if you would like me to proceed with any other tasks.

# Comparison with Langchain/ Langgraph

When compared to LangChain and LangGraph, the conversation aspects of Kogwistar represent a shift from "Message Lists" to "Graph-Native Memory."

### 1. Memory Structure: Linear List vs. Connected Graph
*   **LangChain/LangGraph:** Memory is conceptually a **linear sequence** of `BaseMessage` objects. Even when using vector retrieval, the goal is to select a subset of these messages to form a new linear list.
*   **Kogwistar:** Memory is a **Conversation Graph**. Every turn, tool call, and summary is a node. 
    *   Edges represent relationships like `next_turn`, `summarizes`, and `depends_on`.
    *   This allows the orchestrator to pull in context not just by "time" (recency) or "similarity," but by **Structural Lineage** (e.g., "show me the knowledge pointers that were active when we discussed this summary 5 turns ago").

### 2. Context Assembly: Tiered Packing vs. List Pruning
*   **LangChain:** Usually prunes messages based on a window size (`K` last messages) or vector similarity scores.
*   **Kogwistar:** Uses a **Tiered Context Assembly** strategy (`ConversationContextView`). It gathers items from distinct functional tiers:
    1.  **Head Summary:** The "distilled" long-term history.
    2.  **Memory Context:** Dynamically retrieved relevant nodes from *other* past conversations.
    3.  **Pinned KG Refs:** Domain knowledge currently "in scope" for the agent.
    4.  **Tail Turns:** The most recent "live" interaction.
    It then applies a pluggable **Ordering Strategy** (e.g., `grouped_policy`) and a **Packing Logic** that can compress or drop items to fit a strict token budget.

### 3. Auditing: Logs vs. Graph Snapshots
*   **LangGraph:** Observability (tracing the prompt) is typically a side-effect, often requiring an external platform like LangSmith to see what was sent to the LLM.
*   **Kogwistar:** Implements **Graph-Native Snapshots**. 
    *   Before every LLM call, the system persists a `ContextSnapshotNode`.
    *   This node records the exact prompt string and hash.
    *   Crucially, the snapshot node has `depends_on` edges to every other node in the graph that contributed to that prompt. This creates a **forensic trail** within the graph itself, allowing you to prove exactly why an agent gave a specific answer.

### 4. Citations: Text Labels vs. Structural Pointers
*   **LangChain:** Citations are typically handled by the LLM generating text like `[1]` or `[Source: X]`.
*   **Kogwistar:** Uses **Evidence Pinning**. When an agent retrieves knowledge, it creates "Pointer Nodes" in the conversation graph. The final AI response node is then linked to these pointers via graph edges.
    *   A citation is not just a string; it is a **traversable relationship** from the Answer -> Pointer -> KG Entity -> Original Document Span.

### 5. Summarization: Windowing vs. Hierarchical Nodes
*   **LangChain:** Summarization is often a "flat" process that replaces the message list with a single summary string.
*   **Kogwistar:** Supports **Hierarchical Summarization**. A `conversation_summary` node is a first-class entity that explicitly points to the batch of turns it represents. You can have multiple summaries at different "levels" of the graph, allowing the agent to look at different granularities of its own history.

### Summary Comparison Table

| Conversation Aspect | LangChain / LangGraph | Kogwistar |
| :--- | :--- | :--- |
| **Data Model** | Linear Message List | Directed Graph of Nodes/Edges |
| **History Pruning** | Windowing / Vector search | Tiered Gathering + Budgeted Packing |
| **Tool Traces** | Part of the message list | Explicit Tool Nodes in the Graph |
| **Citations** | Textual references | Structural Pointer Nodes |
| **Observability** | External Tracing (LangSmith) | Internal Context Snapshots |
| **Summarization** | List Compression | Hierarchical Summary Nodes |

In essence, **LangChain** focuses on the **Interface** (messages), while **Kogwistar** focuses on the **Substrate** (the graph of relationships that produced those messages).


![Screnshot of graph trace](screenshots\conversation-graph-with-trace-and-events-on-cdc-panel.png)