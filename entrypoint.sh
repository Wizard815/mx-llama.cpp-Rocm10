#!/bin/sh
set -e
mkdir -p /root/.ssh
chmod 700 /root/.ssh
cp /run/secrets/authorized_key /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
exec /usr/sbin/sshd -D
