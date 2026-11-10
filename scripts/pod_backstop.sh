#!/bin/bash
# Pod-side backstop: the pod removes itself at a deadline, whatever else is happening.
#
# The local watchdog (pod_watchdog.py) copies results and terminates the pod, but it only
# runs while the local machine is awake. This runs on the pod, so the cost stays bounded
# even if the local machine is off. Set it later than the watchdog's deadline: removing a
# pod destroys its disk, and the watchdog should get the chance to copy results first.
#
# Usage: bash pod_backstop.sh <pod-id> <YYYY-MM-DD> <HH:MM:SS>     (date and time in UTC)
#
# Run it from a file. Passing this logic inline through ssh from PowerShell loses the
# quotes, and a mis-parsed date once made the removal fire immediately.
set -euo pipefail
POD=$1
WHEN="$2 $3"
N=$(( $(date -u -d "$WHEN" +%s) - $(date -u +%s) ))
if [ "$N" -lt 600 ]; then
  echo "refusing: the deadline must be more than 10 minutes away (got $N s)"
  exit 1
fi
# An ssh session does not inherit the container's environment; the pod-scoped key is in
# the environment of PID 1. It is never printed.
KEY=${RUNPOD_API_KEY:-$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')}
if [ -z "$KEY" ]; then
  echo "refusing: no pod-scoped RUNPOD_API_KEY in this shell or in PID 1's environment"
  exit 1
fi
runpodctl config --apiKey "$KEY" > /dev/null 2>&1 || true
# The pod-scoped key is narrow. On the pod this was written for it could not read the pod
# ("Unauthorized"), so whether it may remove it is unknown until the backstop fires. The
# backstop is therefore a second line, not a guarantee; the watchdog is the first.
if runpodctl get pod "$POD" 2>&1 | grep -q "$POD"; then
  echo "verified: the pod's key can read pod $POD"
else
  echo "UNVERIFIED: the pod's key cannot read pod $POD; the removal may be refused too"
fi
# '&&': a sleep that fails must never fall through to the removal.
nohup bash -c "sleep $N && echo fired >> /root/backstop.log && runpodctl remove pod $POD >> /root/backstop.log 2>&1" \
  > /dev/null 2>&1 < /dev/null &
echo "backstop set: pod $POD removes itself in $N s, at $WHEN UTC"
