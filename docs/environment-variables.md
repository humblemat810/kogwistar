# Environment Variables

This file describes the environment variables currently read by this repo.
It is grouped by subsystem so you can tell what is required for local dev,
tests, and optional integrations.

## Core Storage And Runtime

| Variable | Used By | Purpose |
| --- | --- | --- |
| `GKE_BACKEND` | `kogwistar.workers.run_index_job_worker` | Selects the storage backend, usually `chroma` or `pg`. |
| `GKE_PERSIST_DIRECTORY` | `kogwistar.workers.run_index_job_worker` | Base directory for local persistence when no explicit path is passed. |
| `GKE_NAMESPACE` | `kogwistar.workers.run_index_job_worker` | Logical namespace for worker/index isolation. |
| `GKE_PG_URL` | `kogwistar.workers.run_index_job_worker` | PostgreSQL connection string for `pg` backend runs. |
| `GKE_PG_SCHEMA` | `kogwistar.workers.run_index_job_worker` | PostgreSQL schema name for pgvector collections. |
| `GKE_EMBEDDING_DIM` | `kogwistar.workers.run_index_job_worker` | Embedding dimension used when constructing the backend. |
| `GKE_PERSISTENCE_MODE` | `kogwistar.workers.run_index_job_worker` | Persistence mode shared with the producer: `single_stage` (default) or `two_stage`. |
| `GKE_WORKER_REPAIR_SCAN_INTERVAL_S` | `kogwistar.workers.run_index_job_worker` | Interval for bounded two-stage lost-enqueue scans; defaults to `0` (disabled). Enable it on at most one designated worker per metadata store and namespace. Scanner ownership is operator-managed, not enforced by a lease. |
| `GKE_JOBLIB_CACHE_DIR` | `scripts/tutorial_ladder.py`, `kogwistar.utils.cache_paths` | Cache directory for tutorial and runtime joblib caches. |
| `GKE_TEST_PG_IMAGE` | `tests/conftest.py` | Docker image used for PostgreSQL testcontainers. |
| `FRONTEND_SRC_PATH` | `scripts/export_schemas.py`, `kogwistar/scripts/schema_export.py` | Path to the frontend source tree for schema export tooling. |
| `CDC_PUBLISH_ENDPOINT` | `scripts/claw_runtime_loop.py` | CDC publish endpoint used by the claw runtime demo. |

## Authentication And Server

| Variable | Used By | Purpose |
| --- | --- | --- |
| `AUTH_MODE` | `kogwistar.server_mcp_with_admin`, `kogwistar.server.auth_middleware` | Auth mode selector, commonly `dev` or `oidc`. |
| `AUTH_DB_URL` | `kogwistar.server.resources`, `kogwistar.server.auth.seed_cli` | Database URL for auth storage, default `sqlite:///auth.sqlite`. |
| `OIDC_PROVIDERS_JSON` | `kogwistar.server.auth.provider_config` | JSON configuration for OIDC providers. |
| `JWT_SECRET` | `kogwistar.server.auth_middleware` | Shared secret for HS256 development auth. |
| `JWT_ALG` | `kogwistar.server.auth_middleware` | JWT algorithm, usually `HS256` or `RS256`. |
| `JWT_ISS` | `kogwistar.server.auth_middleware` | Optional JWT issuer check. |
| `JWT_AUD` | `kogwistar.server.auth_middleware` | Optional JWT audience check. |

## Native Rust API Server

| Variable | Used By | Purpose |
| --- | --- | --- |
| `KOGWISTAR_BACKEND` | Rust `kogwistar-server` | Selects the native service backend. The inline embedding worker supports PostgreSQL or SQLite metadata with an explicitly configured Chroma graph projection. |
| `KOGWISTAR_PG_DSN` | Rust `kogwistar-server` | PostgreSQL DSN for the native Rust service and its durable index-job queue. |
| `KOGWISTAR_PG_SCHEMA_BASE` | Rust `kogwistar-server` | PostgreSQL schema base; defaults to `kogwistar`. |
| `KOGWISTAR_META_SQLITE_PATH` | Rust `kogwistar-server` | Selects SQLite for metadata/runtime/queue storage when the PostgreSQL backend is not selected. It does not make SQLite authoritative for graph data. |
| `KOGWISTAR_CHROMA_ENDPOINT` | Rust `kogwistar-server` | Optional Chroma HTTP endpoint. When paired with `KOGWISTAR_META_SQLITE_PATH`, enables the authenticated graph-neighbor read route against Chroma while SQLite remains the metadata/runtime/queue store. Startup fails if Chroma cannot be configured; no graph-write or cross-store transaction path is implied. |
| `KOGWISTAR_CHROMA_NODE_COLLECTION` | Rust `kogwistar-server` | Node collection for the optional SQLite-metadata/Chroma-graph composition; defaults to `nodes`. |
| `KOGWISTAR_CHROMA_EDGE_COLLECTION` | Rust `kogwistar-server` | Edge collection for the optional SQLite-metadata/Chroma-graph composition; defaults to `edges`. |
| `KOGWISTAR_CHROMA_METRIC` | Rust `kogwistar-server` | Distance metric for the optional Chroma collections: `cosine` (default), `l2`/`squared_l2`, or `inner_product`/`ip`. Invalid values fail startup. |
| `KOGWISTAR_EMBEDDING_PROVIDER` | Rust `kogwistar-server` | Enables Rust-side embedding provider construction; currently accepts `ollama`. |
| `KOGWISTAR_EMBEDDING_MODEL` | Rust `kogwistar-server` | Ollama embedding model name; required when the provider is configured. |
| `KOGWISTAR_EMBEDDING_DIMENSION` | Rust `kogwistar-server` | Positive vector dimension expected from the configured provider. |
| `KOGWISTAR_EMBEDDING_ENDPOINT` | Rust `kogwistar-server` | Ollama endpoint; defaults to `http://127.0.0.1:11434`. |
| `KOGWISTAR_EMBEDDING_WORKER_INLINE` | Rust `kogwistar-server` | Opts into an in-process Rust worker that polls supported embedding-job queues (batches up to 32, 60-second leases). Defaults off. Enabling it without a supported graph backend and configured provider fails startup. |
| `KOGWISTAR_PROJECTION_RECONCILE_INTERVAL_S` | Rust `kogwistar-server` | Optional low-frequency full-namespace Chroma projection reconciliation interval in seconds. Defaults to `0` (off); enabling requires SQLite metadata plus `KOGWISTAR_CHROMA_ENDPOINT`. It can run without enabling embedding polling. |
| `KOGWISTAR_PROJECTION_RECONCILE_NAMESPACE` | Rust `kogwistar-server` | Namespace reconciled by the optional Rust Chroma scanner. Required when `KOGWISTAR_PROJECTION_RECONCILE_INTERVAL_S` is positive. Configure only one designated scanner per namespace. |
| `KOGWISTAR_RUNTIME_NATIVE_INLINE` | Rust `kogwistar-server` | Opts into the separate inline recorded-workflow runtime worker. Defaults off. |
| `JWT_PROTECTED_PATHS` | `kogwistar.server.auth_middleware` | Comma-separated list of protected paths. |
| `DEV_STREAM_GUARD` | `kogwistar.server.auth_middleware` | Enables or disables the dev stream guard. |
| `DEV_AUTH_EMAIL` | `kogwistar.server.auth.router` | Default developer auth email. |
| `DEV_AUTH_SUBJECT` | `kogwistar.server.auth.router` | Default developer auth subject. |
| `DEV_AUTH_NAME` | `kogwistar.server.auth.router` | Default developer display name. |
| `DEV_AUTH_ROLE` | `kogwistar.server.auth.router` | Default developer role. |
| `DEV_AUTH_NS` | `kogwistar.server.auth.router`, `kogwistar.server_mcp_with_admin` | Dev auth namespaces. Accepts a single namespace or a comma-separated list such as `docs,conversation,workflow,wisdom`. |
| `DEV_AUTH_SEED_JSON` | `kogwistar.server.auth.seeding` | Inline JSON seed for auth bootstrap. |
| `DEV_AUTH_SEED_PATH` | `kogwistar.server.auth.seeding` | File path for auth bootstrap seed data. |
| `UI_URL` | `kogwistar.server.auth.router` | UI redirect target after auth flows. |
| `MCP_URL` | `kogwistar.server.bootstrap`, `.env.example` | MCP server URL used by local bootstrap and examples. |

## LLM Providers And Embeddings

| Variable | Used By | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `kogwistar.engine_core.embedding_factory`, `tests` | Generic OpenAI key for non-Azure OpenAI code paths. |
| `AZURE_OPENAI_API_KEY` | `kogwistar.engine_core.embedding_factory` | Azure OpenAI API key for embedding/model calls. |
| `AZURE_OPENAI_ENDPOINT` | `kogwistar.engine_core.embedding_factory` | Azure OpenAI endpoint for embeddings. |
| `OPENAI_API_KEY_GPT4_1` | `kogwistar.integrations.openai_embeddings`, `kogwistar.llm_tasks.default_provider`, `scripts/claw_runtime_loop.py`, `tests` | Azure OpenAI key used by the repo's GPT-4.1/GPT-5 tutorial and test paths. |
| `OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1` | same as above | Azure OpenAI deployment endpoint for the GPT-4.1 tutorial path. |
| `OPENAI_DEPLOYMENT_NAME_GPT4_1` | same as above | Azure deployment name for the GPT-4.1 tutorial path. |
| `OPENAI_MODEL_NAME_GPT4_1` | same as above | Model name associated with the GPT-4.1 deployment. |
| `OPENAI_DEPLOYMENT_VERSION_GPT4_1` | same as above | Azure deployment API version for the GPT-4.1 path. |
| `OPENAI_EMBED_DEPLOYMENT` | `kogwistar.integrations.openai_embeddings` | Azure OpenAI embedding deployment name. |
| `OPENAI_EMBED_ENDPOINT` | `kogwistar.integrations.openai_embeddings` | Azure OpenAI embedding endpoint. |
| `OPENAI_EMBED_API_VERSION` | `kogwistar.integrations.openai_embeddings` | Azure OpenAI embedding API version, default `2024-08-01-preview`. |
| `GOOGLE_API_KEY` | `kogwistar.engine_core.embedding_factory`, `scripts/claw_runtime_loop.py`, `tests` | Google/Gemini API key for real model paths. |
| `GEMINI_API_KEY` | `scripts/tutorial_sections/17_custom_llm_provider.py`, `tests` | Alternate Gemini key alias accepted by some scripts/tests. |
| `GEMINI_MODEL_NAME` | `scripts/claw_runtime_loop.py` | Optional Gemini model name for claw/tutorial demo code. |
| `GKE_TUTORIAL_GEMINI_MODEL` | `scripts/tutorial_ladder.py` | Overrides the default Gemini model used by tutorial provider mode. |
| `GKE_TUTORIAL_OPENAI_MODEL` | `scripts/tutorial_ladder.py` | Overrides the default OpenAI model used by tutorial provider mode. |
| `GKE_TUTORIAL_OLLAMA_MODEL` | `scripts/tutorial_ladder.py` | Overrides the default Ollama model used by tutorial provider mode. |
| `EMBEDDING_PROVIDER` | `kogwistar.engine_core.embedding_factory` | Chooses the embedding provider, default `ollama`. |
| `EMBEDDING_MODEL` | `kogwistar.engine_core.embedding_factory` | Optional embedding model override. |
| `TEST_LLM_MODEL` | `tests/kg_conversation/test_agentic_answering.py`, `tests` | Real-model test override for LLM model name. |
| `TEST_LLM_TEMPERATURE` | `tests/kg_conversation/test_agentic_answering.py` | Real-model test override for sampling temperature. |

## Test And CI Controls

| Variable | Used By | Purpose |
| --- | --- | --- |
| `ANONYMIZED_TELEMETRY` | `tests/conftest.py` | Forced to `FALSE` in tests to disable telemetry noise. |
| `TESTING` | `kogwistar.server.auth.db` | Marks the process as test mode for auth DB helpers. |
| `ENV` | `kogwistar.server.auth.db` | Alternate environment flag; `test` or `ci` enables test behavior. |
| `PYTEST_CURRENT_TEST` | `kogwistar.server.auth.db` | Pytest sentinel used to detect test execution. |
| `TESTCONTAINERS_RYUK_DISABLED` | `tests/conftest.py`, `tests/test_pg_container_fixture.py` | Disables Ryuk for Windows/testcontainer compatibility. |
| `INGESTER_TEST_CACHE` | `tests/primitives/test_ingester_long_doc.py` | Cache directory for ingestion tests. |
| `GEMINI_ASSOC_FILL_CACHE_DIR` | `tests/primitives/test_gemini_assoc_flattened_model_fill.py` | Cache directory for Gemini schema fill tests. |
| `GEMINI_SCHEMA_DEPTH_CACHE_DIR` | `tests/primitives/test_gemini_schema_nesting_limit.py` | Cache directory for schema depth probes. |
| `GEMINI_SCHEMA_DEPTH_HARD_CAP` | `tests/primitives/test_gemini_schema_nesting_limit.py` | Safety cap for schema nesting tests. |
| `GEMINI_SCHEMA_DEPTH_PROBE_ATTEMPTS` | `tests/primitives/test_gemini_schema_nesting_limit.py` | Retry count for schema depth probes. |
| `TEST_GEMINI_MODEL` | `tests/primitives/test_gemini_schema_nesting_limit.py` | Model name override for the schema nesting test. |
| `TEST_GEMINI_MODELS` | `tests/primitives/test_gemini_assoc_flattened_model_fill.py` | Comma-separated model list for Gemini fill tests. |
| `TEST_GEMINI_SCHEMA_MODES` | `tests/primitives/test_gemini_assoc_flattened_model_fill.py` | Comma-separated schema mode list for Gemini fill tests. |

## Notes

- Keep secrets out of version control. Use `.env` only for local development.
- `.env.example` should remain safe to commit and should contain placeholders only.
- If you add a new environment variable, update this file and the relevant `.env.example` entry together.
- In `AUTH_MODE=dev`, leaving `DEV_AUTH_NS` unset in the login flow defaults to all app namespaces, but explicit local debug configs may still override it to a narrower value.
