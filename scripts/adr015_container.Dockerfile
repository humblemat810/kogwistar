ARG BASE_IMAGE=python:3.13.14-slim-bookworm
FROM ${BASE_IMAGE}

ARG WHEEL_NAME
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq \
    && apt-get install -y -qq git >/dev/null \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/core && python -m venv /opt/consumer

COPY ${WHEEL_NAME} /wheel/${WHEEL_NAME}
COPY kg-doc-parser /build/kg-doc-parser
COPY kogwistar-obsidian-sink /build/kogwistar-obsidian-sink

RUN /opt/core/bin/pip install --quiet "/wheel/${WHEEL_NAME}[full,test]" "pytest-xdist>=3.8,<4" \
    && /opt/consumer/bin/pip install --quiet "/wheel/${WHEEL_NAME}[test,chroma]" \
        "pytest-xdist>=3.8,<4" fastapi langchain-openai \
        /build/kg-doc-parser /build/kogwistar-obsidian-sink \
    && rm -rf /root/.cache/pip /build
