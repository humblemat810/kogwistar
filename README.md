# kogwistar

Knowledge engine plus MCP server for graph and document querying.

## Highlights

- CDC-oriented graph updates and replay workflows.
- Multiple storage backends (including Chroma and PostgreSQL/pgvector paths).
- MCP tooling surface for graph query, extraction, and admin operations.
- Visualization helpers for D3 and Cytoscape payloads.

## Run (Development)

1. Create and activate a Python environment (3.10+).
2. Install dependencies for local work.
3. Start the app with `uvicorn graph_knowledge_engine.server_mcp_with_admin:app --port 8765`.

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

## Private GitHub Install

- Use the package name, not the repo name, when installing with extras from a private repository.
- Examples:
  - `pip install "kogwistar[chroma] @ git+ssh://git@github.com/<org>/<repo>.git@main"`
  - `pip install "kogwistar[pgvector,openai] @ git+ssh://git@github.com/<org>/<repo>.git@<commit>"`
- For HTTPS-based private installs, use the same direct-reference form with a token-authenticated `git+https://...` URL.

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
- Stop the stack:
  - `docker compose down`
- Stop and delete named volumes:
  - `docker compose down -v`
- The `chroma` profile uses an embedded Chroma-backed app container with named volumes.
- The `pg` profile starts both the app container and a `pgvector` Postgres container.

## License

MIT. See [LICENSE](LICENSE).


