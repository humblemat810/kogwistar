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

## License

MIT. See [LICENSE](LICENSE).


