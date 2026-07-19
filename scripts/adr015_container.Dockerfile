# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.13.14-slim-bookworm
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

ARG WHEEL_NAME
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=60 \
    UV_HTTP_RETRIES=2

RUN apt-get update -qq \
    && apt-get install -y -qq git >/dev/null \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/core && python -m venv /opt/consumer

COPY ${WHEEL_NAME} /wheel/${WHEEL_NAME}
COPY kg-doc-parser /build/kg-doc-parser
COPY kogwistar-obsidian-sink /build/kogwistar-obsidian-sink

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/core/bin/python "/wheel/${WHEEL_NAME}[full,test]" "pytest-xdist>=3.8,<4" \
    && uv pip install --python /opt/consumer/bin/python "/wheel/${WHEEL_NAME}[test,chroma]" \
        "pytest-xdist>=3.8,<4" fastapi langchain-openai \
        /build/kg-doc-parser /build/kogwistar-obsidian-sink \
    && rm -rf /build
