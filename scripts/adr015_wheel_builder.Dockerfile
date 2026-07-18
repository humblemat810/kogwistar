FROM rust:1.91.1-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARGO_HOME=/cargo \
    CARGO_TARGET_DIR=/target

RUN apt-get update -qq \
    && apt-get install -y -qq python3 python3-venv python3-pip patchelf >/dev/null \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/build \
    && /opt/build/bin/pip install --quiet "maturin>=1.8,<2"

WORKDIR /source
ENTRYPOINT ["/bin/sh", "/source/scripts/adr015_build_wheel.sh"]
