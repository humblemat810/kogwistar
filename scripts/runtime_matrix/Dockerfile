# syntax=docker/dockerfile:1.7

FROM debian:trixie-slim AS cpython315-source
ARG PYTHON_VERSION=3.15.1
ARG PYTHON_TARBALL_SHA256=133d85e118dc32621cda1d9d1d7d9694fe0c5e82bc339a7d528c109fe8b36387
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
      "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
      --output /tmp/python.tgz \
    && echo "${PYTHON_TARBALL_SHA256}  /tmp/python.tgz" | sha256sum --check --status \
    && mkdir /tmp/python-src \
    && tar -xzf /tmp/python.tgz --strip-components=1 -C /tmp/python-src \
    && rm /tmp/python.tgz

FROM debian:trixie-slim AS cpython314-source
ARG PYTHON_VERSION=3.14.9
ARG PYTHON_TARBALL_SHA256=8b47e85ee67e93945e93620a051a8f44bfc1e9735ed67075d166f4a74b7a060b
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
      "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
      --output /tmp/python.tgz \
    && echo "${PYTHON_TARBALL_SHA256}  /tmp/python.tgz" | sha256sum --check --status \
    && mkdir /tmp/python-src \
    && tar -xzf /tmp/python.tgz --strip-components=1 -C /tmp/python-src \
    && rm /tmp/python.tgz

FROM debian:trixie-slim AS cpython314-gil-build
COPY --from=cpython314-source /tmp/python-src /tmp/python-src
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential ca-certificates curl libbz2-dev libffi-dev libgdbm-dev \
    liblzma-dev libncurses-dev libreadline-dev libsqlite3-dev libssl-dev \
    tk-dev uuid-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
RUN cd /tmp/python-src \
    && ./configure --prefix=/usr/local --with-ensurepip=install \
    && make -j2 \
    && make install \
    && rm -rf /tmp/python-src

FROM debian:trixie-slim AS cpython315-gil-build
COPY --from=cpython315-source /tmp/python-src /tmp/python-src
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential ca-certificates curl libbz2-dev libffi-dev libgdbm-dev \
    liblzma-dev libncurses-dev libreadline-dev libsqlite3-dev libssl-dev \
    tk-dev uuid-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
RUN cd /tmp/python-src \
    && ./configure --prefix=/usr/local --with-ensurepip=install \
    && make -j2 \
    && make install \
    && rm -rf /tmp/python-src

FROM debian:trixie-slim AS cpython315-nogil-build
COPY --from=cpython315-source /tmp/python-src /tmp/python-src
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential ca-certificates libbz2-dev libffi-dev libgdbm-dev \
    liblzma-dev libncurses-dev libreadline-dev libsqlite3-dev libssl-dev \
    tk-dev uuid-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
RUN cd /tmp/python-src \
    && ./configure --prefix=/usr/local --disable-gil --with-ensurepip=install \
    && make -j2 \
    && make install \
    && rm -rf /tmp/python-src

FROM debian:trixie-slim AS cpython315-gil
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates libbz2-1.0 libffi8 libgdbm6t64 liblzma5 libncursesw6 \
    libreadline8t64 libsqlite3-0 libssl3t64 libuuid1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cpython315-gil-build /usr/local /usr/local
COPY requirements.txt /tmp/requirements.txt
RUN python3.15 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY kogwistar /src/kogwistar
COPY scripts /src/scripts
WORKDIR /workspace
ENV PYTHONPATH=/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3.15", "/src/scripts/mixed_runtime_container_benchmark.py"]

FROM debian:trixie-slim AS cpython315-nogil
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates libbz2-1.0 libffi8 libgdbm6t64 liblzma5 libncursesw6 \
    libreadline8t64 libsqlite3-0 libssl3t64 libuuid1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cpython315-nogil-build /usr/local /usr/local
COPY requirements.txt /tmp/requirements.txt
RUN python3.15t -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY kogwistar /src/kogwistar
COPY scripts /src/scripts
WORKDIR /workspace
ENV PYTHONPATH=/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3.15t", "/src/scripts/mixed_runtime_container_benchmark.py"]

FROM cpython315-nogil AS cpython315-nogil-rust-build
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential ca-certificates curl libffi-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
        https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.97.1 --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"
COPY pyproject.toml README.md LICENSE /build/
COPY rust /build/rust
COPY contracts /build/contracts
COPY kogwistar /build/kogwistar
RUN python3.15t -m pip install --no-cache-dir maturin==1.15.0 \
    && python3.15t -m pip install --no-cache-dir --no-build-isolation --no-deps /build \
    && env -u PYTHONPATH python3.15t -c \
        'import importlib.util, shutil; from pathlib import Path; p = importlib.util.find_spec("kogwistar._rust"); assert p and p.origin; out = Path("/tmp/native-extension"); out.mkdir(); shutil.copy2(p.origin, out / Path(p.origin).name)'
RUN cargo build --release --manifest-path /build/rust/Cargo.toml \
        -p kogwistar-api --example mixed_mcp_authority_benchmark \
    && mkdir -p /tmp/native-endpoint \
    && cp /build/rust/target/release/examples/mixed_mcp_authority_benchmark \
        /tmp/native-endpoint/mixed_mcp_authority_benchmark

FROM cpython315-nogil AS cpython315-nogil-rust
COPY --from=cpython315-nogil-rust-build /tmp/native-extension/_rust* /src/kogwistar/
ENTRYPOINT ["python3.15t", "/src/scripts/mixed_runtime_container_benchmark.py"]

FROM cpython315-nogil-rust AS cpython315-nogil-rust-endpoint
RUN apt-get update && apt-get install --no-install-recommends -y build-essential \
    && python3.15t -m pip install --no-cache-dir "uvicorn>=0.30" "psutil>=6" \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cpython315-nogil-rust-build /tmp/native-endpoint/mixed_mcp_authority_benchmark \
    /usr/local/bin/mixed_mcp_authority_benchmark
ENV KOGWISTAR_MIXED_NATIVE_BINARY=/usr/local/bin/mixed_mcp_authority_benchmark
ENTRYPOINT ["python3.15t", "/src/scripts/mixed_endpoint_container_benchmark.py"]

FROM debian:trixie-slim AS cpython314-gil
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates libbz2-1.0 libffi8 libgdbm6t64 liblzma5 libncursesw6 \
    libreadline8t64 libsqlite3-0 libssl3t64 libuuid1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cpython314-gil-build /usr/local /usr/local
COPY requirements.txt /tmp/requirements.txt
RUN python3.14 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY kogwistar /src/kogwistar
COPY scripts /src/scripts
WORKDIR /workspace
ENV PYTHONPATH=/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3.14", "/src/scripts/mixed_runtime_container_benchmark.py"]

FROM pypy:3.11-slim-trixie AS pypy311
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential ca-certificates curl libffi-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
        https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.97.1 --profile minimal \
    && pypy3 -m pip install --no-cache-dir --upgrade pip \
    && PATH="/root/.cargo/bin:$PATH" pypy3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY kogwistar /src/kogwistar
COPY scripts /src/scripts
WORKDIR /workspace
ENV PYTHONPATH=/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["pypy3", "/src/scripts/mixed_runtime_container_benchmark.py"]
