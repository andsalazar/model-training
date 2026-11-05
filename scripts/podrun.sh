#!/usr/bin/env bash
# Run a command on the pod's SSH proxy.
#
# The proxy allocates a PTY only, so `ssh host "cmd"` and non-PTY stdin are both
# rejected with "Your SSH client doesn't support PTY". `-tt` forces a PTY, which
# means the command has to be supplied on stdin with an explicit `exit`, and the
# output carries terminal echo. Callers strip it.
#
# Usage: scripts/podrun.sh user@host < commands.sh
set -eu

ssh -tt \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR \
  "$1"
