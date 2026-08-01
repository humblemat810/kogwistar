#!/bin/sh
set -eu

test -f /source/pyproject.toml
test -f /source/rust/Cargo.lock
mkdir -p /wheelhouse

exec /opt/build/bin/python -m maturin build \
    --manifest-path /source/rust/crates/kogwistar-python/Cargo.toml \
    --release \
    --locked \
    --out /wheelhouse
