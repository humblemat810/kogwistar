FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+container \
    HOST=0.0.0.0 \
    PORT=8765 \
    GKE_BACKEND=chroma \
    GKE_PERSIST_DIRECTORY=/app/data

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY graph_knowledge_engine ./graph_knowledge_engine

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[full]"

EXPOSE 8765

CMD ["knowledge-mcp"]
