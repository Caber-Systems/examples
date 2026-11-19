#!/usr/bin/env bash
# MCP server entrypoint, with optional embedded Milvus Lite.
#
# With MCP_EMBEDDED_MILVUS=true this starts Milvus Lite's server binary as a
# second process in the container, waits for it to accept TCP connections, then
# runs the MCP server. Any other value runs the command unchanged, so images
# built for demos that keep a separate Milvus (or none) behave as before.
#
# Milvus Lite listens on TCP, never on its default Unix socket: PyCaber maps a
# gRPC target to its ApiMap entry by port, and a socket target has none, which
# would drop the Milvus hop to the ApiMap "default" (log-only) entry.
#
# The binary is executed directly rather than through milvus_lite.server: the
# package's __init__ imports pkg_resources, which setuptools 81+ no longer has.
#
# If either process exits, the other is stopped and the container exits, so
# Docker's restart policy and healthcheck see a real failure instead of an MCP
# server answering /healthz with no vector store behind it.
#
# Written for bash 3.2 (no `wait -n`) so the unit tests run on a macOS host.
#
# Settings (all optional):
#   MILVUS_LITE_BIN            /opt/milvus-lite/milvus
#   MCP_MILVUS_DATA_DIR        /var/lib/milvus-lite   (a named volume in compose)
#   MILVUS_PORT                19530
#   MILVUS_LITE_START_TIMEOUT  60 (seconds)
#   MILVUS_LITE_LOG_LEVEL      ERROR

set -u

if [ "${MCP_EMBEDDED_MILVUS:-false}" != "true" ]; then
    exec "$@"
fi

log() { echo "mcp-entrypoint: $*" >&2; }

milvus_bin="${MILVUS_LITE_BIN:-/opt/milvus-lite/milvus}"
data_dir="${MCP_MILVUS_DATA_DIR:-/var/lib/milvus-lite}"
port="${MILVUS_PORT:-19530}"
start_timeout="${MILVUS_LITE_START_TIMEOUT:-60}"
log_level="${MILVUS_LITE_LOG_LEVEL:-ERROR}"

if [ ! -x "$milvus_bin" ]; then
    log "ERROR: Milvus Lite binary not found or not executable: $milvus_bin"
    exit 1
fi
if [ ! -d "$data_dir" ] || [ ! -w "$data_dir" ]; then
    log "ERROR: Milvus Lite data dir missing or not writable: $data_dir"
    exit 1
fi

milvus_pid=""
app_pid=""
tick_pid=""

stop_children() {
    if [ -n "$tick_pid" ]; then kill "$tick_pid" 2>/dev/null; fi
    if [ -n "$app_pid" ]; then kill -TERM "$app_pid" 2>/dev/null; fi
    if [ -n "$milvus_pid" ]; then kill -TERM "$milvus_pid" 2>/dev/null; fi
}

on_signal() {
    log "signal received, stopping MCP server and Milvus Lite"
    stop_children
    app_rc=0
    if [ -n "$app_pid" ]; then wait "$app_pid"; app_rc=$?; fi
    if [ -n "$milvus_pid" ]; then wait "$milvus_pid"; fi
    exit "$app_rc"
}
trap on_signal TERM INT

# The bundled shared libraries live beside the binary. Scope LD_LIBRARY_PATH to
# this one command so the Python app keeps the system's libraries.
lib_dir="$(dirname "$milvus_bin")"
LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$milvus_bin" "$data_dir/corpus.db" "0.0.0.0:$port" "$log_level" &
milvus_pid=$!
log "Milvus Lite started (pid $milvus_pid) on 0.0.0.0:$port, data $data_dir/corpus.db"

deadline=$((SECONDS + start_timeout))
until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    if ! kill -0 "$milvus_pid" 2>/dev/null; then
        wait "$milvus_pid"
        log "ERROR: Milvus Lite exited during startup (rc $?)"
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        log "ERROR: Milvus Lite did not open port $port within ${start_timeout}s"
        kill -TERM "$milvus_pid" 2>/dev/null
        wait "$milvus_pid"
        exit 1
    fi
    sleep 0.25
done
log "Milvus Lite is accepting connections on port $port"

"$@" &
app_pid=$!

# bash reaps exited background children on SIGCHLD, so `kill -0` fails as soon
# as a child is gone, and `wait` still returns its recorded exit status.
while :; do
    if ! kill -0 "$app_pid" 2>/dev/null; then
        wait "$app_pid"
        app_rc=$?
        log "MCP server exited (rc $app_rc), stopping Milvus Lite"
        app_pid=""
        stop_children
        wait "$milvus_pid"
        exit "$app_rc"
    fi
    if ! kill -0 "$milvus_pid" 2>/dev/null; then
        wait "$milvus_pid"
        log "ERROR: Milvus Lite exited (rc $?), stopping MCP server"
        milvus_pid=""
        stop_children
        wait "$app_pid"
        exit 1
    fi
    sleep 1 &
    tick_pid=$!
    wait "$tick_pid"
    tick_pid=""
done
