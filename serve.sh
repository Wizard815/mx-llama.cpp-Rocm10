#!/bin/sh
# serve — launch llama-server so its output reaches BOTH:
#   * /app/logs/llama-server.log        (persistent, survives the container)
#   * the docker console                 (via /proc/1/fd/1, PID 1's stdout pipe)
#
# Why this exists: a llama-server started by hand inside tmux writes to the pty
# in the container, and `docker logs` only reads PID 1's stdout/stderr — so the
# output is invisible from outside. Redirecting to /proc/1/fd/1 is what puts it
# on `docker logs -f mx-llamacpp-ssh`.
#
# Usage inside the container:
#   serve -m /app/models/HFCache/hub/.../model.gguf --port 8000
# Everything after `serve` is passed through to llama-server unchanged.
set -e

LOG=/app/logs/llama-server.log
mkdir -p /app/logs

echo "=== $(date -Iseconds) serve: llama-server $* ===" >>"$LOG"

# stdout+stderr -> tee -> logfile and -> PID 1's stdout (docker log stream).
# exec keeps this as the foreground process so Ctrl-C / ssh disconnect
# tears down llama-server with it.
exec llama-server "$@" 2>&1 | tee -a "$LOG" >/proc/1/fd/1
