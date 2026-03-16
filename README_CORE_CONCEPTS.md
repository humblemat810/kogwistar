# Core Concepts & Architecture

This repository implements a graph-based knowledge engine where **provenance** and **observability** are first-class primitives.

## Base Abstractions

### Graph Engine & Knowledge Graph
The `GraphKnowledgeEngine` (in `graph_knowledge_engine/engine.py`) serves as the base abstraction for the system. It manages the underlying "knowledge graph" or "graph database".
- **Primitives**: The fundamental units are `Node` and `Edge` (in `graph_knowledge_engine/models.py`).
- **Provenance-Heavy**: Unlike typical graph databases, every primitive in this system is designed to be **provenance-heavy**. This means nodes and edges carry rich metadata about their origin, including source documents, spans, verification status, and insertion methods.
- **Leakage Prevention**: All core models use `pydantic-extension`'s `ModeSlicingMixin`. This allows designating specific fields as internal-only (using `ExcludeMode("llm")`), preventing sensitive metadata or backend IDs from leaking into LLM prompts via automated schema generation.
- **Local-First Performance**: The abstractions map cleanly to embedded databases (SQLite/Chroma) and lightweight local containers, enabling fast experimentation on regular machines while remaining scalable to full deployments.

## Extensions

The system extends the base knowledge graph abstraction for specific domains:

### 1. Conversation
- **Domain**: Modeling conversation history, turns, and memory.
- **Implementation**: `ConversationNode` and `ConversationEdge` are subclasses extending the base `Node` and `Edge`.
- **Orchestrator**: The `ConversationOrchestrator` (in `graph_knowledge_engine/conversation_orchestrator.py`) acts as the **workflow entry point** for conversation interactions. It manages the flow of user turns, retrieval, answer generation, and summarization using `add_conversation_turn_workflow_v2`.

### 2. Workflow Design
- **Domain**: defining and executing agentic workflows (DAGs/Cyclic graphs).
- **Implementation**: `WorkflowNode` and `WorkflowEdge` are subclasses extending the base `Node` and `Edge`.
- **Design as Graph**: The workflow definition itself is stored *as a graph* in the knowledge engine. Steps are nodes; transitions are edges.
- **Runtime**: The workflow is executed by the runtime (e.g., `WorkflowRuntime`), traversing the graph structure defined by `WorkflowNode` and `WorkflowEdge`.

## Key Files
- `graph_knowledge_engine/engine.py`: Defines `GraphKnowledgeEngine`, the base graph abstraction.
- `graph_knowledge_engine/models.py`: Defines the provenance-heavy `Node`/`Edge` primitives and their `Conversation`/`Workflow` extensions.
- `graph_knowledge_engine/conversation_orchestrator.py`: The entry point for conversation workflows, coordinating the engine and agents.
