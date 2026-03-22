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
| `JWT_PROTECTED_PATHS` | `kogwistar.server.auth_middleware` | Comma-separated list of protected paths. |
| `DEV_STREAM_GUARD` | `kogwistar.server.auth_middleware` | Enables or disables the dev stream guard. |
| `DEV_AUTH_EMAIL` | `kogwistar.server.auth.router` | Default developer auth email. |
| `DEV_AUTH_SUBJECT` | `kogwistar.server.auth.router` | Default developer auth subject. |
| `DEV_AUTH_NAME` | `kogwistar.server.auth.router` | Default developer display name. |
| `DEV_AUTH_ROLE` | `kogwistar.server.auth.router` | Default developer role. |
| `DEV_AUTH_NS` | `kogwistar.server.auth.router` | Optional namespace override for dev auth. |
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
