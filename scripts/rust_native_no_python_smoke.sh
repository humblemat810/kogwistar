#!/usr/bin/env bash
set -euo pipefail

ROOT=/source
MANIFEST="$ROOT/rust/Cargo.toml"
SERVER="${CARGO_TARGET_DIR:-/target}/release/kogwistar-server"
PORT=18088
LOG=/tmp/kogwistar-native-server.log

cargo build --release --locked --manifest-path "$MANIFEST" \
  -p kogwistar-api --bin kogwistar-server
test -x "$SERVER"
cargo tree --locked --manifest-path "$MANIFEST" \
  -p kogwistar-api --edges normal > /tmp/kogwistar-api-dependencies.txt
if grep -Eiq '(^|[[:space:]])(kogwistar-python|pyo3)([[:space:]v(]|$)' \
  /tmp/kogwistar-api-dependencies.txt; then
  echo "native API dependency graph includes a Python seam" >&2
  exit 1
fi

if ldd "$SERVER" | grep -Eiq 'libpython|python[0-9]'; then
  echo "native server links a Python runtime" >&2
  exit 1
fi

KOGWISTAR_BACKEND=sqlite \
KOGWISTAR_IMPL_SERVER=rust \
KOGWISTAR_SERVER_HOST=127.0.0.1 \
KOGWISTAR_SERVER_PORT="$PORT" \
KOGWISTAR_PERSIST_DIRECTORY=/tmp/kogwistar-native-data \
KOGWISTAR_CONVERSATION_PERSIST_DIRECTORY=/tmp/kogwistar-native-conversations \
KOGWISTAR_WORKFLOW_PERSIST_DIRECTORY=/tmp/kogwistar-native-workflows \
KOGWISTAR_WISDOM_PERSIST_DIRECTORY=/tmp/kogwistar-native-wisdom \
KOGWISTAR_META_SQLITE_PATH=/tmp/kogwistar-native-server.sqlite \
AUTH_MODE=dev \
JWT_SECRET=adr015-native-smoke-secret \
AUTH_DB_URL=sqlite:////tmp/kogwistar-native-auth.sqlite \
  "$SERVER" >"$LOG" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

expected_exe=$(readlink -f "$SERVER")
actual_exe=$(readlink -f "/proc/$server_pid/exe")
test "$actual_exe" = "$expected_exe"
if ldd "/proc/$server_pid/exe" | grep -Eiq 'libpython|python[0-9]'; then
  echo "running server loaded a Python runtime" >&2
  exit 1
fi

health=
for _ in $(seq 1 100); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$LOG" >&2
    exit 1
  fi
  if health=$(timeout 2 bash -c \
    'exec 3<>/dev/tcp/127.0.0.1/'"$PORT"'; printf "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n" >&3; cat <&3' \
    2>/dev/null) && grep -q 'HTTP/1.1 200' <<<"$health"; then
    break
  fi
  health=
  sleep 0.1
done
test -n "$health" || { cat "$LOG" >&2; exit 1; }
grep -q '"mode":"rust"' <<<"$health"
grep -q '"ok":true' <<<"$health"

children=$(cat "/proc/$server_pid/task/$server_pid/children")
test -z "$children"
printf 'native_executable=%s\npython_seam_dependencies=0\npython_runtime_linked=false\npython_child_processes=0\nhealth=200\n' \
  "$actual_exe"
