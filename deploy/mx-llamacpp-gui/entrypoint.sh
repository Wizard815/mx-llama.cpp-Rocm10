#!/bin/sh
# llama-swap is PID 1 here.
#
# Why: llama-swap spawns llama-server as a child process and pipes the child's
# stdout/stderr into its own log pipeline. So its own stdout IS the container
# console, and every llama-server log line ends up in `docker logs` *and* in the
# GUI's log viewer. That is the whole fix - no `tee`, no /proc/1/fd/1, no tmux.
set -e

# --- sshd, in the background, logging to the container console ---
# -e sends sshd's log to stderr (there is no syslogd in this image, so without
# -e sshd's messages are silently dropped). -D keeps it in the foreground of
# this subshell so the log stream stays attached to the container.
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -f /run/secrets/authorized_key ]; then
    cp /run/secrets/authorized_key /root/.ssh/authorized_keys
    chown root:root /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd -D -e &
else
    echo "WARN: no /run/secrets/authorized_key mounted, sshd disabled"
fi

mkdir -p /app/logs

# --- llama-swap as PID 1 ---
exec /app/llama-swap \
    -config /app/config.yaml \
    -config-dir /app/config.d \
    -listen 0.0.0.0:8080 \
    -watch-config
