# System Architecture

## Component Tree

```text
System Architecture
├── Entry & Security Layer
│   ├── MCP Server (API Interface)
│   ├── Security & RBAC Middleware
│   │   ├── JWT Authentication (Bearer Tokens)
│   │   ├── Role-Based Access Control (RO/RW Roles)
│   │   └── Namespace Isolation (Docs vs. Wisdom)
│   ├── Conversation Orchestrator (Command Dispatcher)
│   └── Visualization / D3 (Read Model View)
│
├── Workflow & Execution Layer (Command Side - Produces Events)
│   ├── Workflow Runtime (State Machine Execution)
│   ├── Agentic Answering (Logic & Reasoning)
│   ├── Resolvers (Step Implementation)
│   └── Strategies (Decision Logic)
│
├── Event Sourcing Backbone (The Source of Truth)
│   ├── OpLog (Append-Only Event Store)
│   │   └── Sequential Change Events
│   ├── CDC / Change Bus (Event Propagation)
│   └── Change Bridge (Sync & Replay Mechanism)
│
├── Knowledge Graph Layer (The "Brain")
│   ├── Unified Graph Model (Hypergraph & Schema)
│   │   ├── Nodes & Edges (Pydantic Models)
│   │   ├── Hyper-edges (Edge-to-Edge Relations)
│   │   └── Rich Metadata Schemas
│   │
│   ├── Domain Graphs (Projections)
│   │   ├── Conversation Graph ("Canvas", User Context)
│   │   ├── Workflow Design Graph (Intent/Template)
│   │   ├── Orchestration Trace Graph (Execution History)
│   │   └── Wisdom Graph (Meta-patterns)
│
├── Provenance & Grounding Layer (The "Trust")
│   ├── Spans & Groundings (Source Locators)
│   ├── Document Ingestion (OCR, Splitter)
│   └── Verification (Traceability back to Source)
│
└── Physical Storage (Persistence)
    ├── PostgreSQL / SQLite (Relational/Graph Store)
    └── Vector Store (Chroma / PgVector - Embedding Index)

Key Abstractions:
- Event Sourcing (State derived from OpLog replay)
- Hypergraph Schema (Rich, typed, edge-to-edge relations)
- Provenance First (Every node/edge linked to source spans)
- RBAC & Namespaces (Granular Security Model)
- Context Snapshot (Observable LLM Inputs)
- Token Nesting (Parallel Execution State)

In sample conversation pipelines primitives
- Summary is treated as a evidence set level compression for embedding index.
- From kg graph pin a knowledge and become referencible in conversation thread(graph), pinning a memory is a summary of past conversation that is summarised to be relevant to the current turn. Context snapshot is a curated selection of evidence with summaries.
- Conversation and workflow design are knowledge graphs. Allow RF RL graph navigation to infer wisdom.
- Wisdom is a future layer (only implented as namespaced abstraction) that can learn about user, conclusion knowledge from a conversation, and how to get knowledge via orchestration pattern workflows.
- Potentially a wisdom is just another layer that have curated projections nodes pinned from multiple conversation graphs.
```

## Component Description

### 1. Entry & Security Layer
- **MCP Server**: Exposes system capabilities via the Model Context Protocol.
- **Security & RBAC**: Implements a comprehensive security model:
    - **JWT Auth**: Verifies identity via Bearer tokens (HS256/RS256).
    - **RBAC Middleware**: Enforces `RO` (Read-Only) and `RW` (Read-Write) roles at the tool level using `FastMCP` middleware hooks.
    - **Namespace Isolation**: Segregates access between `DOCS` and `WISDOM` namespaces.
- **Conversation Orchestrator**: Acts as the command dispatcher, initiating workflows that generate events.

### 2. Workflow & Execution Layer (Command Side)
- **Workflow Runtime**: Executes the business logic (commands). Instead of mutating state directly, it produces atomic changes.
- **Agentic Answering**: Complex reasoning logic that determines *what* changes need to happen.

### 3. Event Sourcing Backbone
- **OpLog**: The append-only log of all `ChangeEvents`. This is the ultimate source of truth for the system's history.
- **Change Bridge / CDC**: Responsible for propagating events from the OpLog to various projections.

### 4. Knowledge Graph Layer
- **Unified Graph Model**: Implements a **Hypergraph** structure where edges can point to other edges, defined by strict **Pydantic Schemas**.
- **Domain Graphs**: The system treats different concerns (Conversation, Workflow, Trace, Wisdom) as distinct but interoperable sub-graphs within the same engine.

### 5. Provenance & Grounding Layer
- **Provenance First**: Every piece of knowledge is "grounded" in source documents via **Spans** and **Verifications**.
- **Ingestion**: The pipeline that brings external data into the trusted provenance boundary.

### 6. Physical Storage
- **Databases & Vector Stores**: The physical storage media where the materialized views (projections) reside for efficient querying.

## Technical Implementation Details

### Hybrid Search Implementation
- **SQLite FTS5 + Vector**: The system implements a custom ranking algorithm that blends **BM25** scores (from SQLite FTS5) with **Cosine Similarity** (from Vector Store) to handle both exact keyword matches and semantic queries effectively.

### Advanced Optimization & Caching
- **Structured Output Caching**: Uses a specialized decorator (`cache_pydantic_structured`) that hashes both function arguments and **Pydantic JSON Schemas**. This ensures that if the data model evolves, cached LLM responses are automatically invalidated.
- **Joblib Memory**: Utilizes persistent file-based caching for expensive document extraction and graph processing tasks.

### Language Patterns
- **Mixin Composition**: Leveraging Pydantic Mixins (`IdPolicyMixin`, `TombstoneMixin`, `LevelAwareMixin`) to compose complex behaviors (like identity generation, soft deletion, and hierarchy) across different node types without deep inheritance.
- **Async Middleware**: Security policies are implemented as asynchronous middleware, intercepting MCP calls at the protocol layer to enforce Role and Namespace constraints consistently.
