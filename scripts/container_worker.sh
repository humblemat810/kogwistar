#!/bin/sh
set -eu

mkdir -p /workspace/core /workspace/application/logs
cp -a /source/core/. /workspace/core/
cp -a /source/application/. /workspace/application/

native="$({
    cd /tmp
    /opt/core/bin/python -P /source/core/scripts/native_extension_path.py
})"
cp "$native" /workspace/core/kogwistar/_rust.abi3.so

cd /workspace/core
exec /opt/core/bin/python scripts/rust_port_compat.py \
    --application-root /workspace/application \
    --python /opt/core/bin/python \
    --consumer-python /opt/consumer/bin/python \
    "$@"
