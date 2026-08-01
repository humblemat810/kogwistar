FROM rust:1.91.1-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

ENV CARGO_HOME=/cargo \
    CARGO_TARGET_DIR=/target

RUN apt-get update -qq \
    && apt-get install -y -qq python3 python3-venv patchelf >/dev/null \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/build \
    && uv pip install --python /opt/build/bin/python "maturin>=1.8,<2"

WORKDIR /source
ENTRYPOINT ["/bin/sh", "/source/scripts/build_wheel.sh"]
