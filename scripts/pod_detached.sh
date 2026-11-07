#!/bin/bash
# Detached capture runner.
#
# `nohup cmd &` inside an ssh command still dies when the ssh session closes on these
# pods: the shell exits, and nohup only blocks SIGHUP, not the process-group teardown.
# The evidence was an empty output file and no python process within 60s. setsid puts
# the job in its own session so it survives.
set -uo pipefail

SCRIPT="${1:?usage: detached.sh <script> [log]}"
LOG="${2:-/workspace/detached.out}"

rm -f "$LOG"
setsid bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "started pid $PID, logging to $LOG"
sleep 5
echo "--- alive? ---"
if kill -0 "$PID" 2>/dev/null; then echo "YES pid $PID running"; else echo "NO pid $PID already gone"; fi
echo "--- first output ---"
head -20 "$LOG" 2>/dev/null
