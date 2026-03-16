
<p align="center">
  <img src="assets/kogwistar_256.png" width="280"/>
</p>

<h1 align="center">Kogwistar</h1>

<p align="center">
Graph / Hypergraph-Native Agent Intelligence Platform
</p>

<p align="center">
Build knowledge graphs, workflow runtimes, conversation memory, and provenance systems
as a single substrate for AI agents.
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/YOURORG/kogwistar?style=flat-square"/>
  <img src="https://img.shields.io/github/license/YOURORG/kogwistar?style=flat-square"/>
  <img src="https://img.shields.io/badge/python-3.13+-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/runtime-graph--native-purple?style=flat-square"/>
</p>

`kogwistar` is a graph/hypergraph-native agent platform.

It treats knowledge, conversation, workflow/runtime, provenance, and future wisdom as one connected substrate.

It is better understood as a graph-native memory and execution substrate than as only another GraphRAG repository.

Today the repo implements graph memory and query, workflow design/runtime, provenance and replay-oriented surfaces, CDC/event-oriented patterns, and self-hostable development paths.

## Quickstart

- Standalone setup (no frontend integration): [QUICKSTART.md](QUICKSTART.md)
- Roadmap and research direction: [docs/roadmap.md](docs/roadmap.md)
- Detailed comparison with adjacent products/frameworks: [docs/llm-generated-comparison.md](docs/llm-generated-comparison.md)
- Author notes, build context, and design history: [docs/author-notes.md](docs/author-notes.md)
- Runtime rationale: [graph_knowledge_engine/docs/ARD-custom-runtime-rationale.md](graph_knowledge_engine/docs/ARD-custom-runtime-rationale.md)
- Local conversation walkthrough: [docs/tutorials/conversation-pipeline-basics.md](docs/tutorials/conversation-pipeline-basics.md)

- Alternatively, let your AI agent read throught and set up the credential/ keys and environemnt variables for you, and run the appropriate commands.

## Core Features

- Graph/hypergraph-oriented memory and query surfaces.
- Workflow design stored as graph structure, with runtime, replay, and event-oriented execution seams.
- Support conversation execution events stored as hypergraph/graph
- CDC-oriented graph updates and replay workflows.
- Provenance-heavy first class primitives with lifecycle-aware and temporal retrieval support.
- Multiple storage backends, including Chroma and PostgreSQL/pgvector paths. With dual-store eventual consistency or transactional atomicity.
- MCP tooling surface for graph query, extraction, and admin operations.
- Visualization helpers for D3 payloads.

## How This Differs

- Unlike typical agent products, this repo centers a unified graph/hypergraph substrate rather than only chat, skills, or tool orchestration.
- Unlike workflow-first frameworks, it treats provenance, replay, and event history (event source) as part of the core data model rather than secondary runtime features.
- Compared with local/self-hosted agent products, it emphasizes graph-native memory and workflow design seams more than channel breadth or app-registry breadth.
- It spans retrieval, memory, runtime, and provenance concerns together, so it maps less cleanly to a single existing OSS category.

## Application Best Fit

- A strong base platform for building audit-heavy systems, ranging from local personal agents to scalable AI backends.
- Best suited for use cases where provenance, replay, lifecycle-aware retrieval, and workflow history matter.
- Designed to run efficiently on normal local machines (via SQLite/Chroma or lightweight Docker with PostgreSQL/pgvector).
- A foundation repo, not yet a finished enterprise product.

## Security Support

- Role- and namespace-based access control across API and MCP surfaces.
- OIDC/PKCE support for authenticated deployments, with simpler dev-mode auth for local work.
- Sandboxable runtime paths, including container-based execution with networking disabled by default.
- Security-aware boundaries are part of the design, but this repo should still be treated as a foundation platform rather than a fully hardened security product.


## Research Direction

- Move toward privacy-first personal agents with stronger local-first memory and execution.
- Distill conversation and workflow traces into a future wisdom layer.
- Explore agents that can propose and revise their own workflow graphs under human-auditable constraints.
- These are research directions and design seams, not completed product claims.
  Runtime design rationale: [graph_knowledge_engine/docs/ARD-custom-runtime-rationale.md](graph_knowledge_engine/docs/ARD-custom-runtime-rationale.md)

## Why This Repo Exists

- Raise the engineering bar for graph-native agent systems.
- Lower the barrier to entry for building high-quality agent memory and execution layers on normal local hardware.
- Provide a reusable foundation where retrieval, memory, workflow, and provenance are already structurally integrated.
  Additional motivation, build-cost context, and design history: [docs/author-notes.md](docs/author-notes.md)

## Important Tool Sets

- Graph substrate and provenance toolkit for storing nodes, edges, grounding, lifecycle state, and replayable graph changes as first-class primitives.
- Conversation and retrieval orchestration toolkit for building local conversation flows, memory retrieval, KG retrieval, evidence pinning, and workflow-driven v2 turn execution.
- Workflow runtime and replay toolkit for graph-defined execution, checkpoints, joins, suspend/resume flows, and inspectable run traces.
- LLM task abstraction and strategy toolkit for provider-neutral extraction, filtering, summarization, answering, citation repair, verification, and merge/adjudication behavior.
- Server and MCP integration toolkit for exposing conversations, workflow runs, admin operations, and graph tooling through app and tool boundaries.
- Tutorial, CDC, and debugging toolkit for local development ladders, event-stream inspection, CDC viewers, and operational debugging workflows.

## Run (Development)

1. Create and activate a Python 3.13 environment.
2. Install dependencies for local work.
3. Pick a development mode:
   - Server-style MCP app:
     - Start the app with `knowledge-mcp` (defaults to port `28110`).
   - CLI-style workflow/runtime loop:
     - Use the standalone tutorial/runtime script in [QUICKSTART.md](QUICKSTART.md) under `scripts/claw_runtime_loop.py`.
     - This mode is useful for local development when you want to iterate on workflow execution, CDC flow, and event-loop behavior without running the full server surface.

## Development and Test Install

- Minimal local development:
  - `pip install -e ".[chroma,test]"`
- Development with LangGraph-related tests:
  - `pip install -e ".[chroma,test,langgraph]"`
- PostgreSQL parity/integration work:
  - `pip install -e ".[pgvector,test]"`
- Full local contributor setup:
  - `pip install -e ".[full,test]"`

## Run Tests

- Run the default test suite:
  - `pytest`
- Run a specific test file:
  - `pytest tests/workflow/test_save_load_progress.py -q`
- Real Keycloak OIDC end-to-end test:
  - `pytest tests/server/test_oidc_keycloak_e2e.py -q`
- Browser-visible manual Keycloak OIDC test:
  - `pytest tests/server/test_oidc_keycloak_manual.py::test_oidc_keycloak_browser_manual --run-manual -q -s`
- Some integration tests may require Docker/testcontainers or optional extras such as `langgraph`.

## Install Options

- Base/core only:
  - `pip install -e .`
- Chroma backend:
  - `pip install -e ".[chroma]"`
- PostgreSQL + pgvector backend:
  - `pip install -e ".[pgvector]"`
- LLM provider (OpenAI):
  - `pip install -e ".[openai]"`
- LLM provider (Gemini):
  - `pip install -e ".[gemini]"`
- Ingestion with OpenAI:
  - `pip install -e ".[ingestion-openai]"`
- Ingestion with Gemini:
  - `pip install -e ".[ingestion-gemini]"`
- LangGraph converter:
  - `pip install -e ".[langgraph]"`
- Everything:
  - `pip install -e ".[full]"`

## GitHub Install

- Use the package name, not the repo name, when installing with extras from a repository.
- Examples:
  - `pip install "kogwistar[chroma] @ git+ssh://git@github.com/<org>/<repo>.git@main"`
  - `pip install "kogwistar[pgvector,openai] @ git+ssh://git@github.com/<org>/<repo>.git@<commit>"`
- For HTTPS-based installs, use the same direct-reference form with a token-authenticated `git+https://...` URL.

## Runtime Configuration

- `GKE_BACKEND=chroma|pg`
- Shared local persistence root:
  - `GKE_PERSIST_DIRECTORY=/path/to/data`
- Chroma-specific overrides:
  - `GKE_KNOWLEDGE_PERSIST_DIRECTORY`
  - `GKE_CONVERSATION_PERSIST_DIRECTORY`
  - `GKE_WORKFLOW_PERSIST_DIRECTORY`
  - `GKE_WISDOM_PERSIST_DIRECTORY`
  - Legacy `MCP_CHROMA_DIR*` envs still work.
- Postgres-specific settings:
  - `GKE_PG_URL`
  - `GKE_PG_SCHEMA`
  - `GKE_EMBEDDING_DIM`

## Local Docker / Compose

- Build and run with embedded Chroma persistence:
  - `docker compose --profile chroma up -d`
- Build and run with PostgreSQL + pgvector:
  - `docker compose --profile pg up -d`
- Start Keycloak only (OIDC dev realm):
  - `docker compose --profile auth up -d keycloak`
- Stop the stack:
  - `docker compose down`
- Stop and delete named volumes:
  - `docker compose down -v`
- The `chroma` profile uses an embedded Chroma-backed app container with named volumes.
- The `pg` profile starts both the app container and a `pgvector` Postgres container.

## Auth Modes

The server supports two auth modes controlled by `AUTH_MODE`:

- `AUTH_MODE=oidc` (default)
  - Uses OIDC Authorization Code + PKCE.
  - Compose defaults point to the local Keycloak realm imported from `keycloak/realm-kge.json`.
  - Default client: `kge-local` (public client).
  - Default user: `dev` / `dev`.
  - Required envs when running locally without compose:
    - `OIDC_DISCOVERY_URL`
    - `OIDC_CLIENT_ID`
    - `OIDC_REDIRECT_URI`
    - `UI_URL`

- `AUTH_MODE=dev`
  - Skips OIDC and issues an app JWT directly on `GET /api/auth/login`.
  - Defaults are configurable with:
    - `DEV_AUTH_EMAIL`, `DEV_AUTH_SUBJECT`, `DEV_AUTH_NAME`
    - `DEV_AUTH_ROLE`, `DEV_AUTH_NS`

### Windows note (Keycloak realm volume)

`compose.yml` uses a relative bind mount for `keycloak/realm-kge.json`. On Windows with Docker Desktop, this is the most reliable option.

Best practices:

- Keep the realm file inside the repo (relative path).
- Avoid drive-letter paths in `compose.yml` unless you must.
- Ensure the file exists before running `docker compose up`, otherwise Keycloak starts without importing.

## License

MIT. See [LICENSE](LICENSE).


