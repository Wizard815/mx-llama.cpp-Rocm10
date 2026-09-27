#!/bin/sh
# mx-llamacpp-ssh entrypoint — same as before, plus sshd logging to the
# container console.
#
# `sshd -e` writes its log to stderr instead of syslog. There is no syslogd in
# this image, so without -e sshd's connection/auth messages are discarded and
# `docker logs mx-llamacpp-ssh` stays empty. With -e they land on PID 1's
# stderr, i.e. the docker log stream.
set -e

mkdir -p /root/.ssh
chmod 700 /root/.ssh
cp /run/secrets/authorized_key /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

mkdir -p /app/logs

exec /usr/sbin/sshd -D -e
