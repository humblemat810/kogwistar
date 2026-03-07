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

## License

MIT. See [LICENSE](LICENSE).


