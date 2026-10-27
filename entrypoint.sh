#!/bin/sh
set -e
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Authorized keys: the primary key plus every *.pub in authorized_keys.d
# (e.g. the Argos/llama-swap launcher's key), so adding a client is a file
# drop, not an image change.
: > /root/.ssh/authorized_keys
[ -f /run/secrets/authorized_key ] && cat /run/secrets/authorized_key >> /root/.ssh/authorized_keys
for k in /run/secrets/authorized_keys.d/*.pub; do
    [ -f "$k" ] && { cat "$k"; echo; } >> /root/.ssh/authorized_keys
done
chown root:root /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Host keys: openssh-server generates them at image build time, so every
# rebuild changed the server's identity and clients with strict host-key
# checking refused to connect. When /etc/ssh/host-keys is a mounted volume,
# the first start saves the keys there and later starts reuse them.
if [ -d /etc/ssh/host-keys ]; then
    if ls /etc/ssh/host-keys/ssh_host_*_key >/dev/null 2>&1; then
        cp /etc/ssh/host-keys/ssh_host_* /etc/ssh/
    else
        cp /etc/ssh/ssh_host_* /etc/ssh/host-keys/
    fi
    chmod 600 /etc/ssh/ssh_host_*_key
fi

exec /usr/sbin/sshd -D
